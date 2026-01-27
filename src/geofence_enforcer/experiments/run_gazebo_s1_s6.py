#!/usr/bin/env python3
"""
Gazebo-based S1-S6 Experiment Runner
=====================================

Runs S1-S6 scenarios with Gazebo simulation instead of mathematical simulation.

Features:
1. Process management - cleanup between trials, CPU/memory monitoring
2. Gazebo/Nav2/Geofence lifecycle management
3. Sequential trial execution with proper cleanup
4. Checkpoint/resume capability
5. Real-time violation monitoring

Usage:
    python run_gazebo_s1_s6.py                     # Run all experiments
    python run_gazebo_s1_s6.py --resume            # Resume from checkpoint
    python run_gazebo_s1_s6.py --method geofence   # Run specific method
    python run_gazebo_s1_s6.py --scenario S4       # Run specific scenario
    python run_gazebo_s1_s6.py --quick             # Quick test (S1 only)
"""

import os
import sys
import json
import time
import signal
import subprocess
import argparse
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
from enum import Enum
import traceback


class SafeJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles sets and other non-serializable types"""
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, float) and (obj == float('inf') or obj == float('-inf')):
            return None  # JSON doesn't support infinity
        return super().default(obj)


# =============================================================================
# Configuration
# =============================================================================

WORKSPACE_DIR = Path("/home/jim/ros2_motion_planning_tutorials")
EXPERIMENT_DIR = WORKSPACE_DIR / "experiment_results" / "gazebo_s1_s6"
CHECKPOINT_FILE = EXPERIMENT_DIR / "checkpoint.json"
RESULTS_FILE = EXPERIMENT_DIR / "results.jsonl"
SUMMARY_FILE = EXPERIMENT_DIR / "summary.json"
LOG_FILE = EXPERIMENT_DIR / "experiment.log"

# Process cleanup patterns
# Note: Patterns must be specific to avoid matching the script's own path
# (e.g., "ros2" matches "/home/jim/ros2_motion_planning_tutorials")
CLEANUP_PATTERNS = [
    "gzserver", "gzclient", "gz sim", "ruby.*gz",
    "rviz2", "/opt/ros/.*/bin/ros2", "nav2_",  # More specific patterns
    "controller_server", "planner_server",
    "behavior_server", "bt_navigator", "lifecycle_manager",
    "goal_gate_node", "cmd_vel_guard", "path_watchdog",
    "attack_velocity", "attack_odom", "attack_pose", "attack_direct",
    "relay /odom_real",  # odom relay for normal operation
    "violation_monitor", "parameter_bridge", "ros_gz"
]

# Max CPU load before waiting
MAX_CPU_LOAD = 6.0  # Increased: 4.0 → 6.0 (more tolerant of system load)
MAX_MEMORY_PCT = 80.0

# Timeouts
GAZEBO_STARTUP_TIMEOUT = 30  # seconds
NAV2_STARTUP_TIMEOUT = 30  # Increased: 20 → 30 (more time for Nav2 nodes to spawn)
GEOFENCE_STARTUP_TIMEOUT = 5
GOAL_TIMEOUT = 90  # Increased: 60 → 90 (more time for long-distance navigation)
CLEANUP_TIMEOUT = 5
LIFECYCLE_CMD_TIMEOUT = 15  # Increased timeout for lifecycle commands
COSTMAP_CHECK_TIMEOUT = 8  # Timeout for costmap hz check

# Zone definitions (matching geofence.yaml for home.sdf/warehouse world)
# Robot spawns at origin (0,0)
# Single forbidden zone on y=0 line for clean straight-line tests
# Zone center at (5, 0), size 2m x 2m
ZONES = {
    'forbidden_zone': {'x_min': 4.0, 'x_max': 6.0, 'y_min': -1.0, 'y_max': 1.0, 'name': 'forbidden_zone'},
}

# Methods to test
# selp_proper: SELP without margin (only checks if goal is inside zone)
# geofence_hw: Hardware-level geofence guard (cannot be bypassed)
METHODS = ["no_guard", "selp", "selp_proper", "cbf", "ssm", "geofence", "geofence_hw"]


# =============================================================================
# Data Structures
# =============================================================================

class Method(Enum):
    NO_GUARD = "no_guard"
    SELP = "selp"
    CBF = "cbf"
    SSM = "ssm"
    GEOFENCE = "geofence"
    GEOFENCE_HW = "geofence_hw"


@dataclass
class TrialConfig:
    """Configuration for a single trial"""
    trial_id: str
    method: str
    scenario: str
    intensity: str
    seed: int
    goal_x: float
    goal_y: float
    velocity: float = 0.5
    sigma_loc: float = 0.15
    has_physical_barrier: bool = True
    latency_ms: float = 0.0
    boundary_distance: Optional[float] = None
    description: str = ""
    enable_runtime_monitoring: bool = False  # S7: velocity-dependent runtime monitoring
    # S4: Real attack parameters
    attack_type: Optional[str] = None  # "velocity_scaling", "odom_spoofing", or "direct_control"
    attack_scale_factor: float = 1.0  # Scale factor for attack (2.0 = double speed/half position)
    attack_target_x: Optional[float] = None  # For direct_control: target x in forbidden zone
    attack_target_y: Optional[float] = None  # For direct_control: target y in forbidden zone


@dataclass
class TrialResult:
    """Result of a single trial"""
    trial_id: str
    method: str
    scenario: str
    intensity: str
    seed: int
    goal_x: float
    goal_y: float
    decision: str = "unknown"  # "allow", "reject", "runtime_reject", "nav_fail", "timeout"
    violated: bool = False
    task_completed: bool = False
    runtime_rejected: bool = False  # True if rejected during navigation (not at submission)
    nav_failed: bool = False  # True if geofence allowed but Nav2 failed
    min_distance: float = float('inf')
    execution_time_s: float = 0.0
    reason: str = ""
    timestamp: str = ""
    error: str = ""
    # Runtime position monitoring results
    violation_count: int = 0  # Number of position samples inside forbidden zones
    violation_duration_s: float = 0.0  # Total time spent inside forbidden zones
    violated_zones: List[str] = field(default_factory=list)  # Names of violated zones
    path_min_distance: float = float('inf')  # Minimum distance to any zone during navigation


@dataclass
class Checkpoint:
    """Checkpoint for resumable experiments"""
    experiment_id: str
    started_at: str
    last_updated: str
    total_trials: int
    completed_trials: int
    current_trial_idx: int
    completed_trial_ids: List[str] = field(default_factory=list)
    results_summary: Dict[str, Any] = field(default_factory=dict)

    def save(self, path: Path):
        self.last_updated = datetime.now().isoformat()
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2, cls=SafeJSONEncoder)
        print(f"[CHECKPOINT] Saved at trial {self.completed_trials}/{self.total_trials}")

    @classmethod
    def load(cls, path: Path) -> Optional['Checkpoint']:
        if not path.exists():
            return None
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            return cls(**data)
        except Exception as e:
            print(f"[WARNING] Failed to load checkpoint: {e}")
            return None


# =============================================================================
# Safe Process Kill Helper
# =============================================================================

def safe_pkill(pattern: str) -> int:
    """Safely kill processes matching pattern, excluding current process.

    Returns number of processes killed.
    """
    my_pid = os.getpid()
    my_ppid = os.getppid()
    killed = 0

    try:
        result = subprocess.run(
            f"pgrep -f '{pattern}'",
            shell=True, capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            pids = [int(p) for p in result.stdout.strip().split('\n') if p.strip()]
            # Filter out our own process tree
            pids = [p for p in pids if p != my_pid and p != my_ppid]
            for pid in pids:
                try:
                    os.kill(pid, signal.SIGKILL)
                    killed += 1
                except (ProcessLookupError, PermissionError):
                    pass
    except Exception:
        pass

    return killed


# =============================================================================
# Process Manager
# =============================================================================

class ProcessManager:
    """Manages cleanup of ROS2/Gazebo processes to prevent CPU overload"""

    @staticmethod
    def cleanup_all(patterns: List[str] = None, force: bool = False, reset_daemon: bool = False):
        """Kill all related processes"""
        patterns = patterns or CLEANUP_PATTERNS
        print("[CLEANUP] Starting process cleanup...")
        killed = 0
        my_pid = os.getpid()

        for pattern in patterns:
            try:
                # Use pgrep to find PIDs, then filter out our own process
                pgrep_result = subprocess.run(
                    f"pgrep -f '{pattern}'",
                    shell=True, capture_output=True, text=True, timeout=5
                )
                if pgrep_result.returncode == 0 and pgrep_result.stdout.strip():
                    pids = [int(p) for p in pgrep_result.stdout.strip().split('\n') if p.strip()]
                    # Filter out our own process and its parent
                    pids = [p for p in pids if p != my_pid and p != os.getppid()]
                    if pids:
                        for pid in pids:
                            try:
                                os.kill(pid, signal.SIGKILL)
                                killed += 1
                            except (ProcessLookupError, PermissionError):
                                pass
            except subprocess.TimeoutExpired:
                pass
            except Exception:
                pass

        # Wait for processes to die
        time.sleep(2)

        # Force cleanup shared memory if requested
        if force:
            try:
                subprocess.run("rm -rf /dev/shm/fastrtps_*", shell=True, timeout=5)
                subprocess.run("rm -rf /tmp/ros2*", shell=True, timeout=5)
            except Exception:
                pass

        # Reset ROS2 daemon if requested (helps with stuck discovery)
        if reset_daemon:
            try:
                print("[CLEANUP] Resetting ROS2 daemon...")
                subprocess.run(
                    "source /opt/ros/jazzy/setup.bash && ros2 daemon stop",
                    shell=True, executable='/bin/bash', capture_output=True, timeout=10
                )
                time.sleep(1)
                subprocess.run(
                    "source /opt/ros/jazzy/setup.bash && ros2 daemon start",
                    shell=True, executable='/bin/bash', capture_output=True, timeout=10
                )
                time.sleep(2)
            except Exception:
                pass

        print(f"[CLEANUP] Complete (killed {killed} process groups)")
        return killed

    @staticmethod
    def check_system_load() -> Tuple[float, float, float]:
        """Check CPU and memory load"""
        try:
            with open('/proc/loadavg', 'r') as f:
                load1, load5, load15 = f.read().split()[:3]

            with open('/proc/meminfo', 'r') as f:
                meminfo = {}
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        meminfo[parts[0].rstrip(':')] = int(parts[1])

            total = meminfo.get('MemTotal', 1)
            available = meminfo.get('MemAvailable', total)
            mem_used_pct = (total - available) / total * 100

            return float(load1), float(load5), mem_used_pct
        except Exception:
            return 0.0, 0.0, 0.0

    @staticmethod
    def wait_for_system_ready(max_load: float = MAX_CPU_LOAD,
                               max_mem: float = MAX_MEMORY_PCT,
                               max_wait: float = 60.0):
        """Wait until system load is acceptable"""
        start = time.time()
        while True:
            load1, _, mem_pct = ProcessManager.check_system_load()

            if load1 < max_load and mem_pct < max_mem:
                return True

            if time.time() - start > max_wait:
                print(f"[WAIT] Timeout waiting for system (load={load1:.1f}, mem={mem_pct:.1f}%)")
                return False

            print(f"[WAIT] System busy (load={load1:.1f}, mem={mem_pct:.1f}%), waiting...")
            time.sleep(5)

    @staticmethod
    def is_gazebo_running() -> bool:
        """Check if Gazebo is running"""
        try:
            result = subprocess.run(
                "pgrep -f 'gz sim'",
                shell=True, capture_output=True, timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False


# =============================================================================
# Position Monitor (Runtime Zone Violation Detection)
# =============================================================================

class PositionMonitor:
    """
    Monitors robot position during navigation to detect zone violations.
    Runs as a background subprocess that logs robot positions.
    Uses Gazebo model pose (ground truth) for accurate monitoring.
    """

    def __init__(self, zones: Dict = None, check_rate_hz: float = 10.0):
        self.zones = zones or ZONES
        self.check_rate_hz = check_rate_hz
        self.monitor_proc = None
        self.log_file = Path("/tmp/position_monitor.log")
        self.is_running = False

    def start(self):
        """Start position monitoring in background"""
        # Clear previous log
        if self.log_file.exists():
            self.log_file.unlink()

        # Create monitoring script that uses gz topic for ground truth position
        # This reads the actual Gazebo model pose, which is NOT affected by odometry drift
        monitor_script = f'''
import subprocess
import time
import json
import re

ZONES = {json.dumps(self.zones)}
LOG_FILE = "{self.log_file}"

def get_gazebo_pose():
    """Get mobile_manip pose from Gazebo using gz topic"""
    try:
        result = subprocess.run(
            ["gz", "topic", "-e", "-n", "1", "-t", "/world/empty/pose/info"],
            capture_output=True, text=True, timeout=2
        )
        output = result.stdout

        # Parse the protobuf-like text output to find mobile_manip pose
        # Look for: name: "mobile_manip" followed by position {{x: ..., y: ...}}
        match = re.search(
            r\'name: "mobile_manip".*?position \\{{\\s*x: ([\\d.e+-]+)\\s*y: ([\\d.e+-]+)\',
            output, re.DOTALL
        )
        if match:
            return float(match.group(1)), float(match.group(2))
    except Exception as e:
        pass
    return None, None

def main():
    log_file = open(LOG_FILE, "w")
    print("Position monitor started (using Gazebo ground truth pose)")

    try:
        while True:
            x, y = get_gazebo_pose()
            if x is not None:
                t = time.time()

                # Check which zone (if any) the robot is in
                in_zone = None
                min_dist = float("inf")

                for zone_name, zone in ZONES.items():
                    # Check if inside zone
                    if (zone["x_min"] <= x <= zone["x_max"] and
                        zone["y_min"] <= y <= zone["y_max"]):
                        in_zone = zone_name
                        min_dist = 0.0
                        break

                    # Calculate distance to zone
                    closest_x = max(zone["x_min"], min(x, zone["x_max"]))
                    closest_y = max(zone["y_min"], min(y, zone["y_max"]))
                    dist = ((x - closest_x)**2 + (y - closest_y)**2)**0.5
                    min_dist = min(min_dist, dist)

                # Log: timestamp, x, y, in_zone, min_distance
                entry = {{"t": t, "x": x, "y": y, "zone": in_zone, "min_dist": min_dist}}
                log_file.write(json.dumps(entry) + "\\n")
                log_file.flush()

            time.sleep(0.1)  # ~10Hz
    except KeyboardInterrupt:
        pass
    finally:
        log_file.close()

if __name__ == "__main__":
    main()
'''

        # Write script to temp file
        script_file = Path("/tmp/position_monitor_node.py")
        script_file.write_text(monitor_script)

        # Start monitor process (no ROS2 needed - uses gz topic directly)
        cmd = f"python3 {script_file}"
        self.monitor_proc = subprocess.Popen(
            cmd,
            shell=True,
            executable='/bin/bash',
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid
        )
        self.is_running = True
        time.sleep(0.5)  # Give it time to start

    def stop(self) -> Dict:
        """Stop monitoring and return violation statistics"""
        if self.monitor_proc:
            try:
                os.killpg(os.getpgid(self.monitor_proc.pid), signal.SIGTERM)
            except:
                pass
            self.monitor_proc = None
        self.is_running = False

        # Parse log file and compute statistics
        return self._analyze_log()

    def _analyze_log(self) -> Dict:
        """Analyze position log to compute violation statistics"""
        result = {
            'violation_count': 0,
            'violation_duration_s': 0.0,
            'violated_zones': [],  # Use list for JSON serialization
            'path_min_distance': float('inf'),
            'total_samples': 0
        }

        if not self.log_file.exists():
            return result

        try:
            entries = []
            with open(self.log_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

            if not entries:
                return result

            result['total_samples'] = len(entries)

            # Analyze violations
            violated_zones_set = set()
            prev_time = None
            for entry in entries:
                t = entry.get('t', 0)
                zone = entry.get('zone')
                min_dist = entry.get('min_dist', float('inf'))

                # Track minimum distance
                if min_dist < result['path_min_distance']:
                    result['path_min_distance'] = min_dist

                # Track violations
                if zone is not None:
                    result['violation_count'] += 1
                    violated_zones_set.add(zone)

                    # Estimate duration (time since last sample)
                    if prev_time is not None:
                        dt = t - prev_time
                        if dt < 1.0:  # Sanity check - max 1 second gap
                            result['violation_duration_s'] += dt

                prev_time = t

            result['violated_zones'] = list(violated_zones_set)

        except Exception as e:
            print(f"[MONITOR] Error analyzing log: {e}")

        return result


# =============================================================================
# Simulation Manager
# =============================================================================

class SimulationManager:
    """Manages Gazebo/Nav2/Geofence lifecycle"""

    def __init__(self):
        self.gazebo_proc = None
        self.nav2_proc = None
        self.geofence_proc = None
        self.attack_proc = None  # S4: Attack node process
        self.odom_relay_proc = None  # S4: Odom relay for normal operation
        self.current_method = None
        self.current_attack = None  # S4: Current attack type
        self.use_amcl = True  # If False, disable AMCL for dead reckoning experiments

    def start_gazebo(self, headless: bool = True, use_hw_guard: bool = False) -> bool:
        """Start Gazebo simulation

        Args:
            headless: Run without GUI
            use_hw_guard: Use hardware guard bridge config (cmd_vel_safe instead of cmd_vel)
        """
        print("[SIM] Starting Gazebo...")

        # Kill any existing instances first
        ProcessManager.cleanup_all(["gz sim", "gzserver", "gzclient", "ruby.*gz"], force=True)
        time.sleep(2)

        headless_arg = "headless:=true" if headless else ""

        # Use custom bridge config for hardware guard mode
        if use_hw_guard:
            bridge_config = f"{WORKSPACE_DIR}/src/mobile_manipulator_tutorial/src/mobile_manip_moveit_config/config/gz_bridge_with_hw_guard.yaml"
            bridge_arg = f"gz_bridge_config:={bridge_config}"
            print("[SIM] Using HARDWARE GUARD bridge config (/cmd_vel_safe → Gazebo)")
        else:
            bridge_arg = ""

        launch_cmd = f"""
            source /opt/ros/jazzy/setup.bash && \
            source {WORKSPACE_DIR}/install/setup.bash && \
            ros2 launch mobile_manip_moveit_config mobile_manipulator.launch.py \
                use_sim_time:=true {headless_arg} {bridge_arg}
        """

        self.gazebo_proc = subprocess.Popen(
            launch_cmd,
            shell=True,
            executable='/bin/bash',
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid
        )

        print(f"[SIM] Waiting for Gazebo ({GAZEBO_STARTUP_TIMEOUT}s)...")
        time.sleep(GAZEBO_STARTUP_TIMEOUT)

        if self.gazebo_proc.poll() is None:
            print("[SIM] Gazebo started successfully")
            # Start odom relay for normal operation (odom_real → odom)
            self.start_odom_relay()
            return True
        else:
            print("[ERROR] Gazebo failed to start")
            return False

    def start_odom_relay(self) -> bool:
        """Start odom relay node: /odom_real → /odom for normal operation"""
        print("[SIM] Starting odom relay (odom_real → odom)...")

        # Stop any existing relay
        self.stop_odom_relay()

        relay_cmd = f"""
            source /opt/ros/jazzy/setup.bash && \
            source {WORKSPACE_DIR}/install/setup.bash && \
            ros2 run topic_tools relay /odom_real /odom
        """

        self.odom_relay_proc = subprocess.Popen(
            relay_cmd,
            shell=True,
            executable='/bin/bash',
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid
        )

        time.sleep(2)

        if self.odom_relay_proc.poll() is None:
            print("[SIM] Odom relay started")
            return True
        else:
            print("[WARN] Odom relay may have failed to start")
            return False

    def stop_odom_relay(self):
        """Stop odom relay node"""
        if self.odom_relay_proc and self.odom_relay_proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.odom_relay_proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass

        safe_pkill('relay /odom_real')
        self.odom_relay_proc = None

    def check_nav2_lifecycle(self, quiet: bool = False) -> bool:
        """Check if Nav2 is ready by checking for published topics and actions.

        Uses combination of topic list and action list for reliability.
        """
        try:
            # Check for costmap topic (indicates controller_server active)
            result = subprocess.run(
                f"source /opt/ros/jazzy/setup.bash && source {WORKSPACE_DIR}/install/setup.bash && ros2 topic list 2>/dev/null",
                shell=True, executable='/bin/bash',
                capture_output=True, text=True, timeout=LIFECYCLE_CMD_TIMEOUT
            )
            topic_list = result.stdout

            if '/local_costmap/costmap' not in topic_list:
                if not quiet:
                    print(f"[NAV2] local_costmap: NOT found")
                return False
            else:
                if not quiet:
                    print(f"[NAV2] local_costmap: found")

            # Check for navigate_to_pose action (indicates bt_navigator active)
            result = subprocess.run(
                f"source /opt/ros/jazzy/setup.bash && source {WORKSPACE_DIR}/install/setup.bash && ros2 action list 2>/dev/null",
                shell=True, executable='/bin/bash',
                capture_output=True, text=True, timeout=LIFECYCLE_CMD_TIMEOUT
            )
            action_list = result.stdout

            if '/navigate_to_pose' not in action_list:
                if not quiet:
                    print(f"[NAV2] navigate_to_pose action: NOT found")
                return False
            else:
                if not quiet:
                    print(f"[NAV2] navigate_to_pose action: found")

            return True

        except Exception as e:
            if not quiet:
                print(f"[NAV2] Failed to check Nav2 status: {e}")
            return False

    def activate_nav2_lifecycle(self, max_retries: int = 2) -> bool:
        """Activate Nav2 lifecycle nodes if not already active"""
        # Activation order matters: controller/planner first, then bt_navigator
        critical_nodes = ['controller_server', 'planner_server', 'bt_navigator', 'behavior_server', 'smoother_server']

        for retry in range(max_retries + 1):
            all_activated = True

            for node in critical_nodes:
                try:
                    # Check current state
                    result = subprocess.run(
                        f"source /opt/ros/jazzy/setup.bash && source {WORKSPACE_DIR}/install/setup.bash && ros2 lifecycle get /{node}",
                        shell=True, executable='/bin/bash',
                        capture_output=True, text=True, timeout=LIFECYCLE_CMD_TIMEOUT
                    )

                    current_state = result.stdout.strip()

                    if 'active [3]' in current_state:
                        continue  # Already active

                    # Node needs activation - check if it's in configured state first
                    if 'inactive [2]' in current_state:
                        # Can directly activate
                        print(f"[NAV2] Activating {node}...")
                        activate_result = subprocess.run(
                            f"source /opt/ros/jazzy/setup.bash && source {WORKSPACE_DIR}/install/setup.bash && ros2 lifecycle set /{node} activate",
                            shell=True, executable='/bin/bash',
                            capture_output=True, text=True, timeout=LIFECYCLE_CMD_TIMEOUT
                        )
                        if 'Transitioning successful' in activate_result.stdout:
                            print(f"[NAV2] {node}: activated successfully")
                        else:
                            print(f"[NAV2] {node}: activation result - {activate_result.stdout.strip()}")
                            all_activated = False
                    elif 'unconfigured [1]' in current_state:
                        # Need to configure first, then activate
                        print(f"[NAV2] {node} unconfigured, configuring first...")
                        subprocess.run(
                            f"source /opt/ros/jazzy/setup.bash && source {WORKSPACE_DIR}/install/setup.bash && ros2 lifecycle set /{node} configure",
                            shell=True, executable='/bin/bash',
                            capture_output=True, text=True, timeout=LIFECYCLE_CMD_TIMEOUT
                        )
                        time.sleep(1)
                        subprocess.run(
                            f"source /opt/ros/jazzy/setup.bash && source {WORKSPACE_DIR}/install/setup.bash && ros2 lifecycle set /{node} activate",
                            shell=True, executable='/bin/bash',
                            capture_output=True, text=True, timeout=LIFECYCLE_CMD_TIMEOUT
                        )
                        all_activated = False  # Check again in next iteration
                    else:
                        print(f"[NAV2] {node}: unknown state ({current_state})")
                        all_activated = False

                except subprocess.TimeoutExpired:
                    print(f"[NAV2] {node}: command timeout")
                    all_activated = False
                except Exception as e:
                    print(f"[NAV2] Failed to activate {node}: {e}")
                    all_activated = False

            if all_activated:
                return True

            if retry < max_retries:
                print(f"[NAV2] Activation incomplete, retrying ({retry + 1}/{max_retries})...")
                time.sleep(3)

        return False

    def check_costmap_publishing(self, timeout: float = None, retries: int = 2) -> bool:
        """Check if costmaps are being published"""
        if timeout is None:
            timeout = COSTMAP_CHECK_TIMEOUT

        for attempt in range(retries + 1):
            try:
                result = subprocess.run(
                    f"source /opt/ros/jazzy/setup.bash && timeout {timeout} ros2 topic hz /global_costmap/costmap --window 3",
                    shell=True, executable='/bin/bash',
                    capture_output=True, text=True, timeout=timeout + 3
                )
                # If we get average rate info, costmap is publishing
                if 'average rate' in result.stdout:
                    print("[NAV2] Costmap publishing: OK")
                    return True
                else:
                    if attempt < retries:
                        print(f"[NAV2] Costmap not yet publishing, waiting... ({attempt + 1}/{retries})")
                        time.sleep(3)
                    else:
                        print("[NAV2] Costmap NOT publishing after retries")
                        return False
            except subprocess.TimeoutExpired:
                if attempt < retries:
                    print(f"[NAV2] Costmap check timeout, retrying... ({attempt + 1}/{retries})")
                    time.sleep(2)
                else:
                    print("[NAV2] Costmap check timeout after retries")
                    return False
            except Exception as e:
                print(f"[NAV2] Costmap check failed: {e}")
                return False
        return False

    def wait_for_nav2_ready(self, max_wait: float = 75.0, check_interval: float = 5.0) -> bool:
        """Wait for Nav2 to be fully ready (nodes present and action available)"""
        start = time.time()
        quiet = False

        while time.time() - start < max_wait:
            # Check if critical nodes exist and action is available
            if self.check_nav2_lifecycle(quiet=quiet):
                # Nodes and action available - verify costmap
                if self.check_costmap_publishing(retries=1):
                    print("[NAV2] All systems ready!")
                    return True
                else:
                    # Nodes OK but costmap not publishing - wait a bit more
                    print(f"[NAV2] Nodes OK, waiting for costmap... ({int(time.time() - start)}s)")
            else:
                elapsed = int(time.time() - start)
                if elapsed > 30:
                    # Only print waiting message after initial 30s
                    print(f"[NAV2] Waiting for Nav2 to be ready... ({elapsed}s)")

            quiet = True  # Reduce spam after first check
            time.sleep(check_interval)

        print("[NAV2] Timeout waiting for Nav2 to be ready")
        return False

    def start_nav2(self, verify: bool = True, retry_count: int = 0, use_amcl: bool = None) -> bool:
        """Start Nav2 navigation stack with lifecycle verification

        Args:
            verify: Whether to verify Nav2 lifecycle is ready
            retry_count: Current retry attempt number
            use_amcl: If False, disable AMCL for dead reckoning only (odom spoofing experiments)
                      If None, uses self.use_amcl
        """
        if use_amcl is None:
            use_amcl = self.use_amcl
        max_retries = 2
        print(f"[SIM] Starting Nav2...{' (retry ' + str(retry_count) + ')' if retry_count > 0 else ''}")
        if not use_amcl:
            print("[SIM] AMCL disabled - using dead reckoning only")

        # Clean up any leftover Nav2 processes before starting
        if retry_count > 0:
            self.stop_nav2()
            time.sleep(3)

        amcl_arg = "use_amcl:=true" if use_amcl else "use_amcl:=false"
        launch_cmd = f"""
            source /opt/ros/jazzy/setup.bash && \
            source {WORKSPACE_DIR}/install/setup.bash && \
            ros2 launch mobile_manip_moveit_config navigation.launch.py \
                use_sim_time:=true {amcl_arg} rviz:=false
        """

        self.nav2_proc = subprocess.Popen(
            launch_cmd,
            shell=True,
            executable='/bin/bash',
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid
        )

        print(f"[SIM] Waiting for Nav2 initial startup ({NAV2_STARTUP_TIMEOUT}s)...")
        time.sleep(NAV2_STARTUP_TIMEOUT)

        if self.nav2_proc.poll() is not None:
            print("[ERROR] Nav2 process died")
            if retry_count < max_retries:
                return self.start_nav2(verify=verify, retry_count=retry_count + 1)
            return False

        if verify:
            # Wait for lifecycle to be ready (with longer timeout)
            if not self.wait_for_nav2_ready(max_wait=75.0):
                print("[ERROR] Nav2 lifecycle not ready")
                if retry_count < max_retries:
                    print(f"[SIM] Attempting Nav2 restart ({retry_count + 1}/{max_retries})...")
                    return self.start_nav2(verify=True, retry_count=retry_count + 1)
                else:
                    print("[ERROR] Nav2 failed to stabilize after all retries")
                    return False

        print("[SIM] Nav2 started successfully")
        return True

    def stop_nav2(self):
        """Stop Nav2 processes"""
        if self.nav2_proc and self.nav2_proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.nav2_proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass

        # Kill all Nav2 related processes
        nav2_patterns = [
            "controller_server", "planner_server", "bt_navigator",
            "behavior_server", "smoother_server", "waypoint_follower",
            "velocity_smoother", "collision_monitor", "lifecycle_manager",
            "map_server", "amcl"
        ]
        for pattern in nav2_patterns:
            safe_pkill(pattern)

        self.nav2_proc = None
        time.sleep(2)

    def start_geofence(self, method: str, params: Dict = None) -> bool:
        """Start geofence goal_gate node with specified method"""
        print(f"[SIM] Starting geofence with method: {method}")

        # Stop existing geofence first
        self.stop_geofence()
        time.sleep(2)

        self.current_method = method

        # Special handling for geofence_hw: use hardware-level guard
        if method == 'geofence_hw':
            # Start hardware guard node (gz_bridge already configured at Gazebo start)
            if not self.start_hardware_guard_node():
                print("[ERROR] Failed to start hardware guard node")
                return False

            # Use regular geofence for goal-level checking, but no cmd_vel_guard
            # (hardware guard handles runtime protection)
            actual_method = 'geofence'
            enable_cmd_vel_guard = False
            print("[SIM] Using HARDWARE-LEVEL protection (Gazebo ground truth)")
        else:
            actual_method = method
            # Determine whether to enable cmd_vel_guard (runtime velocity monitoring)
            # Only enable for methods that should block runtime attacks:
            # - geofence, cbf, ssm: have runtime monitoring, should block attacks
            # - no_guard, selp, selp_proper: only goal-level checking, attacks should succeed
            runtime_monitoring_methods = ['geofence', 'cbf', 'ssm']
            enable_cmd_vel_guard = method in runtime_monitoring_methods

        # Build launch arguments
        launch_args = [f"safety_method:={actual_method}"]
        launch_args.append(f"enable_cmd_vel_guard:={'true' if enable_cmd_vel_guard else 'false'}")

        # For geofence method, use full interception mode:
        # cmd_vel_guard subscribes to /cmd_vel and publishes to /cmd_vel_safe
        # This requires gz_bridge to be configured with hw_guard config
        if method == 'geofence':
            launch_args.append("cmd_vel_input_topic:=/cmd_vel")
            launch_args.append("cmd_vel_output_topic:=/cmd_vel_safe")
            print("[SIM] Using FULL INTERCEPTION mode (/cmd_vel → guard → /cmd_vel_safe)")

        if params:
            valid_params = ['k_sigma', 'localization_sigma', 'tracking_error',
                          'v_max', 'latency', 'enable_estimation_term',
                          'enable_tracking_term', 'enable_latency_term',
                          'enable_runtime_monitoring', 'runtime_monitoring_rate']
            for key, value in params.items():
                if key in valid_params:
                    if isinstance(value, bool):
                        launch_args.append(f"{key}:={'true' if value else 'false'}")
                    else:
                        launch_args.append(f"{key}:={value}")

        launch_cmd = f"""
            source /opt/ros/jazzy/setup.bash && \
            source {WORKSPACE_DIR}/install/setup.bash && \
            ros2 launch geofence_policy_enforcer demo.launch.py \
                {' '.join(launch_args)}
        """

        self.geofence_proc = subprocess.Popen(
            launch_cmd,
            shell=True,
            executable='/bin/bash',
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid
        )

        print(f"[SIM] Waiting for geofence ({GEOFENCE_STARTUP_TIMEOUT}s)...")
        time.sleep(GEOFENCE_STARTUP_TIMEOUT)

        if self.geofence_proc.poll() is None:
            print(f"[SIM] Geofence started with method: {method} (cmd_vel_guard: {enable_cmd_vel_guard})")
            return True
        else:
            print("[ERROR] Geofence failed to start")
            return False

    def stop_geofence(self):
        """Stop only geofence nodes"""
        if self.geofence_proc and self.geofence_proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.geofence_proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass

        safe_pkill('goal_gate_node')
        safe_pkill('cmd_vel_guard')
        safe_pkill('hardware_geofence_guard')
        self.geofence_proc = None
        self.current_method = None

    def start_hardware_guard_node(self) -> bool:
        """Start hardware-level geofence guard node (cannot be bypassed).

        This guard intercepts ALL /cmd_vel commands and uses Gazebo ground truth
        position to ensure the robot cannot enter forbidden zones.

        NOTE: gz_bridge must already be configured with hardware guard config
        at Gazebo startup time (use_hw_guard=True in start_gazebo).

        Architecture:
          Any source → /cmd_vel → [HW Guard] → /cmd_vel_safe → gz_bridge → Gazebo
        """
        print("[SIM] Starting hardware geofence guard node...")

        # Stop any existing guard
        self.stop_hardware_guard()

        # Start the hardware guard node
        guard_cmd = f"""
            source /opt/ros/jazzy/setup.bash && \
            source {WORKSPACE_DIR}/install/setup.bash && \
            ros2 run geofence_policy_enforcer hardware_geofence_guard \
                --ros-args \
                -p input_topic:=/cmd_vel \
                -p output_topic:=/cmd_vel_safe \
                -p safety_margin:=0.3 \
                -p simulation_horizon:=1.0 \
                -p gazebo_world:=empty \
                -p model_name:=mobile_manip
        """

        self.hw_guard_proc = subprocess.Popen(
            guard_cmd,
            shell=True,
            executable='/bin/bash',
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid
        )

        time.sleep(3)

        if self.hw_guard_proc.poll() is None:
            print("[SIM] Hardware geofence guard started (CANNOT BE BYPASSED)")
            return True
        else:
            print("[ERROR] Hardware geofence guard failed to start")
            return False

    def stop_hardware_guard(self):
        """Stop hardware geofence guard"""
        if hasattr(self, 'hw_guard_proc') and self.hw_guard_proc and self.hw_guard_proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.hw_guard_proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass

        safe_pkill('hardware_geofence_guard')
        self.hw_guard_proc = None

    def restart_gz_bridge_for_hw_guard(self) -> bool:
        """Restart gz_bridge to use /cmd_vel_safe instead of /cmd_vel"""
        print("[SIM] Restarting gz_bridge for hardware guard mode...")

        # Kill existing bridge
        safe_pkill('parameter_bridge')
        time.sleep(2)

        # Start new bridge with hardware guard config
        hw_bridge_config = f"{WORKSPACE_DIR}/src/mobile_manipulator_tutorial/src/mobile_manip_moveit_config/config/gz_bridge_with_hw_guard.yaml"

        bridge_cmd = f"""
            source /opt/ros/jazzy/setup.bash && \
            source {WORKSPACE_DIR}/install/setup.bash && \
            ros2 run ros_gz_bridge parameter_bridge \
                --ros-args -p config_file:={hw_bridge_config}
        """

        self.hw_bridge_proc = subprocess.Popen(
            bridge_cmd,
            shell=True,
            executable='/bin/bash',
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid
        )

        time.sleep(3)

        if self.hw_bridge_proc.poll() is None:
            print("[SIM] gz_bridge restarted with hardware guard config")
            return True
        else:
            print("[ERROR] gz_bridge restart failed")
            return False

    def start_attack(self, attack_type: str, scale_factor: float = 2.0,
                     target_x: float = None, target_y: float = None) -> bool:
        """Start S4 attack node (velocity_scaling, odom_spoofing, or direct_control)

        Args:
            attack_type: "velocity_scaling", "odom_spoofing", or "direct_control"
            scale_factor: For velocity_scaling, 2.0 = double speed
                         For odom_spoofing, 0.5 = robot appears to move half distance
            target_x, target_y: For direct_control, the target position to drive to

        Note:
            velocity_scaling works with current setup (cmd_vel_nav → attack → cmd_vel)
            odom_spoofing: gz_bridge publishes to /odom_real, attack node spoofs to /odom
            direct_control: Bypasses Nav2 entirely and drives directly to target
        """
        if attack_type == "direct_control":
            print(f"[ATTACK] Starting {attack_type} attack with target=({target_x}, {target_y})")
        else:
            print(f"[ATTACK] Starting {attack_type} attack with scale_factor={scale_factor}")

        # Stop any existing attack first
        self.stop_attack()
        time.sleep(1)

        self.current_attack = attack_type

        # For odom_spoofing, stop the normal odom relay first
        if attack_type == "odom_spoofing":
            print("[ATTACK] Stopping odom relay for odom spoofing attack...")
            self.stop_odom_relay()
            time.sleep(1)

        if attack_type == "velocity_scaling":
            # Intercept cmd_vel_nav → cmd_vel
            attack_cmd = f"""
                source /opt/ros/jazzy/setup.bash && \
                source {WORKSPACE_DIR}/install/setup.bash && \
                ros2 run geofence_policy_enforcer attack_velocity_scaling \
                    --ros-args \
                    -p scale_factor:={scale_factor} \
                    -p input_topic:=/cmd_vel_nav \
                    -p output_topic:=/cmd_vel \
                    -p attack_enabled:=true
            """
        elif attack_type == "odom_spoofing":
            # For odom spoofing, scale_factor < 1.0 makes robot appear to move less
            # e.g., scale_factor=0.5 means robot at 6m appears at 3m
            attack_cmd = f"""
                source /opt/ros/jazzy/setup.bash && \
                source {WORKSPACE_DIR}/install/setup.bash && \
                ros2 run geofence_policy_enforcer attack_odom_spoofing \
                    --ros-args \
                    -p scale_factor:={scale_factor} \
                    -p input_topic:=/odom_real \
                    -p output_topic:=/odom \
                    -p attack_enabled:=true
            """
        elif attack_type == "direct_control":
            # Direct control: bypass Nav2 and drive directly into forbidden zone
            # Robot spawns facing -y (yaw=-1.5707), so we need to:
            # 1. Rotate 90 degrees counter-clockwise to face +x
            # 2. Then drive forward into the forbidden zone
            print("[ATTACK] Using rotate-then-drive velocity injection")

            # Create a Python script that rotates then drives at high speed
            # Tested values: rotate 4s at 0.8 rad/s, then drive at 1.5 m/s
            attack_script = '''
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time

class DirectControlAttack(Node):
    def __init__(self):
        super().__init__("direct_control_attack")
        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.timer = self.create_timer(0.1, self.send_cmd)
        self.start_time = time.time()
        self.phase = "rotate"
        self.rotate_duration = 4.0  # 4 seconds rotation
        self.rotate_speed = 0.8     # Faster rotation
        self.drive_speed = 1.5      # Fast forward (1.5 m/s)
        self.get_logger().info("Attack started: rotating to face +x direction")

    def send_cmd(self):
        elapsed = time.time() - self.start_time
        msg = Twist()

        if self.phase == "rotate" and elapsed < self.rotate_duration:
            # Rotate counter-clockwise to face +x direction
            msg.angular.z = self.rotate_speed
        else:
            if self.phase == "rotate":
                self.phase = "drive"
                self.get_logger().info("Driving towards forbidden zone at 1.5 m/s!")
            # Drive forward into forbidden zone at high speed
            msg.linear.x = self.drive_speed

        self.pub.publish(msg)

def main():
    rclpy.init()
    node = DirectControlAttack()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
'''
            # Write script to temp file and run
            script_file = "/tmp/direct_control_attack.py"
            with open(script_file, 'w') as f:
                f.write(attack_script)

            attack_cmd = f"""
                source /opt/ros/jazzy/setup.bash && \
                source {WORKSPACE_DIR}/install/setup.bash && \
                python3 {script_file}
            """
        elif attack_type == "param_injection":
            # Parameter injection: modify Nav2 controller velocity/acceleration limits
            # This is more realistic - attacker changes parameters, Nav2 generates faster velocities
            # The robot will overshoot because it's moving faster than expected
            print(f"[ATTACK] Using parameter injection (scale_factor={scale_factor})")

            # Create script that continuously sets parameters (to overcome any resets)
            attack_script = f'''
import subprocess
import time

# Target parameters and their boosted values
# Original: max_vel_x=0.22, max_speed_xy=0.22
# Boosted: multiply by scale_factor
scale = {scale_factor}
original_vel = 0.22
boosted_vel = min(original_vel * scale, 3.0)  # Cap at 3.0 m/s

print(f"[PARAM_ATTACK] Boosting velocity limits by {{scale}}x: {{original_vel}} -> {{boosted_vel}} m/s")

# Parameters to modify
params = [
    # Controller server (DWB planner)
    ("/controller_server", "FollowPath.max_vel_x", str(boosted_vel)),
    ("/controller_server", "FollowPath.max_speed_xy", str(boosted_vel)),
    # Also boost acceleration for faster response
    ("/controller_server", "FollowPath.acc_lim_x", str(5.0)),
    # Velocity smoother limits
    ("/velocity_smoother", "max_velocity", f"[{{boosted_vel}}, 0.0, 2.0]"),
    ("/velocity_smoother", "max_accel", "[5.0, 0.0, 3.2]"),
]

def set_params():
    for node, param, value in params:
        cmd = f"ros2 param set {{node}} {{param}} '{{value}}'"
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=3)
            if "Set parameter successful" in result.stdout:
                print(f"[PARAM_ATTACK] Set {{node}}/{{param}} = {{value}}")
            else:
                print(f"[PARAM_ATTACK] Failed: {{node}}/{{param}}: {{result.stderr.strip()}}")
        except Exception as e:
            print(f"[PARAM_ATTACK] Error setting {{param}}: {{e}}")

# Keep setting params periodically (in case of resets)
print("[PARAM_ATTACK] Starting continuous parameter injection...")
while True:
    set_params()
    time.sleep(2.0)
'''
            script_file = "/tmp/param_injection_attack.py"
            with open(script_file, 'w') as f:
                f.write(attack_script)

            attack_cmd = f"""
                source /opt/ros/jazzy/setup.bash && \
                source {WORKSPACE_DIR}/install/setup.bash && \
                python3 {script_file}
            """
        else:
            print(f"[ERROR] Unknown attack type: {attack_type}")
            return False

        self.attack_proc = subprocess.Popen(
            attack_cmd,
            shell=True,
            executable='/bin/bash',
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid
        )

        # Wait for attack node to start
        time.sleep(2)

        if self.attack_proc.poll() is None:
            print(f"[ATTACK] {attack_type} attack started successfully")
            return True
        else:
            print(f"[ERROR] {attack_type} attack failed to start")
            return False

    def stop_attack(self):
        """Stop attack nodes and restart odom relay if needed"""
        was_odom_spoofing = (self.current_attack == "odom_spoofing")
        was_param_injection = (self.current_attack == "param_injection")

        if self.attack_proc and self.attack_proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.attack_proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass

        safe_pkill('attack_velocity_scaling')
        safe_pkill('attack_odom_spoofing')
        safe_pkill('attack_direct_control')
        safe_pkill('param_injection_attack')
        self.attack_proc = None
        self.current_attack = None

        # Restore original parameters after param_injection attack
        if was_param_injection:
            print("[ATTACK] Restoring original Nav2 parameters...")
            restore_cmds = [
                "ros2 param set /controller_server FollowPath.max_vel_x 0.22",
                "ros2 param set /controller_server FollowPath.max_speed_xy 0.22",
                "ros2 param set /controller_server FollowPath.acc_lim_x 2.5",
                "ros2 param set /velocity_smoother max_velocity '[0.5, 0.0, 2.0]'",
                "ros2 param set /velocity_smoother max_accel '[2.5, 0.0, 3.2]'",
            ]
            for cmd in restore_cmds:
                try:
                    subprocess.run(
                        f"source /opt/ros/jazzy/setup.bash && {cmd}",
                        shell=True, executable='/bin/bash',
                        capture_output=True, timeout=3
                    )
                except:
                    pass

        # Restart odom relay after stopping odom_spoofing
        if was_odom_spoofing:
            print("[ATTACK] Restarting odom relay after odom spoofing attack...")
            time.sleep(1)
            self.start_odom_relay()

    def stop_all(self, reset_daemon: bool = False):
        """Stop all simulation processes"""
        print("[SIM] Stopping all processes...")

        self.stop_attack()  # Stop any running attack nodes
        self.stop_odom_relay()  # Stop odom relay
        self.stop_geofence()
        self.stop_nav2()

        if self.gazebo_proc and self.gazebo_proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.gazebo_proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass

        ProcessManager.cleanup_all(force=True, reset_daemon=reset_daemon)
        time.sleep(CLEANUP_TIMEOUT)

        self.gazebo_proc = None
        self.nav2_proc = None
        self.geofence_proc = None
        self.attack_proc = None
        self.odom_relay_proc = None
        self.current_method = None
        self.current_attack = None
        print("[SIM] All processes stopped")

    def health_check(self) -> bool:
        """Quick health check of simulation stack"""
        # Check processes are running
        if not self.gazebo_proc or self.gazebo_proc.poll() is not None:
            print("[HEALTH] Gazebo process died!")
            return False

        if not self.nav2_proc or self.nav2_proc.poll() is not None:
            print("[HEALTH] Nav2 process died!")
            return False

        if not self.geofence_proc or self.geofence_proc.poll() is not None:
            print("[HEALTH] Geofence process died!")
            return False

        return True

    def recover_nav2(self, reset_daemon: bool = False) -> bool:
        """Attempt to recover Nav2 by restarting it"""
        print("[RECOVER] Attempting Nav2 recovery...")

        self.stop_nav2()
        time.sleep(3)

        # Reset daemon only if requested (fallback for stuck discovery)
        if reset_daemon:
            ProcessManager.cleanup_all(patterns=[], force=False, reset_daemon=True)

        if not self.start_nav2(verify=True):
            print("[RECOVER] Nav2 restart failed")
            # Try once more with daemon reset as fallback
            if not reset_daemon:
                print("[RECOVER] Trying with daemon reset...")
                return self.recover_nav2(reset_daemon=True)
            return False

        # Restart geofence too since it depends on Nav2
        if self.current_method:
            method = self.current_method
            self.stop_geofence()
            time.sleep(2)
            if not self.start_geofence(method):
                print("[RECOVER] Geofence restart failed")
                return False

        print("[RECOVER] Nav2 recovery successful")
        return True

    def reset_robot_pose(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0) -> bool:
        """Reset robot to specified pose in both Gazebo and Nav2"""
        try:
            import math
            qz = math.sin(theta / 2)
            qw = math.cos(theta / 2)

            # 1. Teleport robot in Gazebo using gz service
            # World name from SDF file: empty (defined in warehouse_walk.sdf)
            # Robot model name from URDF: mobile_manip
            print(f"[RESET] Teleporting robot to ({x}, {y}, θ={theta})")

            gz_teleport_cmd = f"""gz service -s /world/empty/set_pose \
                --reqtype gz.msgs.Pose --reptype gz.msgs.Boolean --timeout 2000 \
                --req 'name: "mobile_manip", position: {{x: {x}, y: {y}, z: 0.1}}, orientation: {{x: 0.0, y: 0.0, z: {qz}, w: {qw}}}'"""

            result = subprocess.run(
                gz_teleport_cmd,
                shell=True, executable='/bin/bash',
                capture_output=True, timeout=5, text=True
            )
            if result.returncode != 0:
                print(f"[WARNING] Gazebo teleport may have failed: {result.stderr[:100] if result.stderr else 'no error'}")

            time.sleep(0.5)  # Wait for physics to settle

            # 2. Publish to /initialpose for Nav2 AMCL
            initialpose_cmd = f"""ros2 topic pub --once /initialpose geometry_msgs/msg/PoseWithCovarianceStamped '{{
                header: {{frame_id: "map"}},
                pose: {{
                    pose: {{
                        position: {{x: {x}, y: {y}, z: 0.0}},
                        orientation: {{x: 0.0, y: 0.0, z: {qz}, w: {qw}}}
                    }},
                    covariance: [0.25, 0.0, 0.0, 0.0, 0.0, 0.0,
                                 0.0, 0.25, 0.0, 0.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.06853891945200942]
                }}
            }}'"""

            subprocess.run(
                f"source /opt/ros/jazzy/setup.bash && {initialpose_cmd}",
                shell=True, executable='/bin/bash',
                capture_output=True, timeout=5
            )

            time.sleep(1.0)
            print("[RESET] Robot pose reset complete")
            return True
        except Exception as e:
            print(f"[WARNING] Robot reset failed: {e}")
            return False

    def is_simulation_ready(self) -> bool:
        """Check if simulation is ready"""
        gazebo_ok = self.gazebo_proc and self.gazebo_proc.poll() is None
        nav2_ok = self.nav2_proc and self.nav2_proc.poll() is None
        geofence_ok = self.geofence_proc and self.geofence_proc.poll() is None
        return gazebo_ok and nav2_ok and geofence_ok

    def restart_with_method(self, method: str, params: Dict = None) -> bool:
        """Restart entire simulation with new method for reliability"""
        if self.current_method == method and self.is_simulation_ready():
            # Already running with correct method
            return True

        # Always do full restart when changing methods to avoid Nav2 issues
        print(f"[SIM] Full restart for method: {method}")
        self.stop_all()
        time.sleep(3)

        # Use hardware guard bridge config for geofence and geofence_hw methods
        # This makes the bridge use /cmd_vel_safe instead of /cmd_vel
        use_hw_guard = method in ['geofence', 'geofence_hw']
        if use_hw_guard:
            print(f"[SIM] Starting Gazebo with HARDWARE GUARD bridge config (method: {method})")

        if not self.start_gazebo(use_hw_guard=use_hw_guard):
            return False
        if not self.start_nav2():
            return False
        if not self.start_geofence(method, params):
            return False
        return True


# =============================================================================
# Goal Sender
# =============================================================================

class GoalSender:
    """Sends navigation goals to the robot"""

    @staticmethod
    def send_goal(x: float, y: float, timeout: float = GOAL_TIMEOUT,
                  safety_method: str = None) -> Tuple[str, str]:
        """
        Send navigation goal and wait for result.

        Args:
            x, y: Goal coordinates
            timeout: Timeout in seconds
            safety_method: Current safety method ('geofence', 'cbf', 'ssm', 'selp', 'no_guard')
                          Used to interpret ABORTED results correctly

        Returns:
            Tuple of (decision, reason)
            decision: 'allow', 'reject', 'project', 'timeout', 'error', 'nav_fail', 'runtime_reject'
        """
        # Helper: Check if goal is inside any forbidden zone
        def is_inside_zone(gx: float, gy: float) -> bool:
            for zone in ZONES.values():
                if (zone['x_min'] <= gx <= zone['x_max'] and
                    zone['y_min'] <= gy <= zone['y_max']):
                    return True
            return False

        # Helper: Calculate minimum distance to any zone boundary
        def min_distance_to_zones(gx: float, gy: float) -> float:
            """
            Calculate the minimum distance from a point to any zone boundary.
            Uses signed distance: negative if inside zone, positive if outside.
            """
            min_dist = float('inf')
            for zone in ZONES.values():
                # Check if inside zone
                if (zone['x_min'] <= gx <= zone['x_max'] and
                    zone['y_min'] <= gy <= zone['y_max']):
                    return 0.0  # Inside zone

                # Find closest point on rectangle boundary
                closest_x = max(zone['x_min'], min(gx, zone['x_max']))
                closest_y = max(zone['y_min'], min(gy, zone['y_max']))

                # Distance to closest point
                dist = ((gx - closest_x)**2 + (gy - closest_y)**2)**0.5
                min_dist = min(min_dist, dist)

            return min_dist

        # Helper: Check if goal should be rejected based on safety method and margins
        def should_be_rejected(gx: float, gy: float, method: str) -> bool:
            # Inside zone - always rejected
            if is_inside_zone(gx, gy):
                return True

            # Check margin-based rejection
            dist = min_distance_to_zones(gx, gy)
            # Use small epsilon for floating point comparison
            eps = 1e-6

            if method == 'cbf':
                # CBF margin = 0.3m, h(x) >= 0 means safe (dist >= margin)
                return dist < (0.3 - eps)
            elif method == 'ssm':
                # SSM minimum margin at v=0: intrusion(0.1) + base(0.2) = 0.3m
                return dist < (0.3 - eps)
            elif method == 'selp':
                # SELP uses uncertainty margin = 0.55m for goal checking
                return dist < (0.55 - eps)
            elif method == 'geofence':
                # Geofence uses uncertainty margin = 0.55m
                return dist < (0.55 - eps)
            else:
                # no_guard - only inside zone
                return False

        goal_cmd = f"""
            source /opt/ros/jazzy/setup.bash && \
            source {WORKSPACE_DIR}/install/setup.bash && \
            ros2 action send_goal /navigate_to_pose_safe nav2_msgs/action/NavigateToPose \
                "{{pose: {{header: {{frame_id: 'map'}}, pose: {{position: {{x: {x}, y: {y}, z: 0.0}}, orientation: {{w: 1.0}}}}}}}}" \
                --feedback 2>&1
        """

        try:
            result = subprocess.run(
                goal_cmd,
                shell=True,
                executable='/bin/bash',
                capture_output=True,
                text=True,
                timeout=timeout
            )

            output = result.stdout + result.stderr

            # Parse result - distinguish geofence rejection vs Nav2 failure
            # Geofence logs: "REJECTED goal (x, y): reason" or "ALLOWED goal (x, y)"

            # Check for geofence-specific rejection messages first
            if "REJECTED goal" in output or "REJECTED (projection failed)" in output:
                # Geofence explicitly rejected the goal
                return "reject", "Goal rejected by geofence policy"

            if "Goal accepted" in output:
                if "SUCCEEDED" in output or "succeeded" in output:
                    return "allow", "Goal reached successfully"
                elif "ABORTED" in output or "aborted" in output:
                    # Goal was accepted but aborted - check why
                    if "REJECTED goal" in output:
                        # Should have been caught above, but just in case
                        return "reject", "Goal rejected by geofence policy"
                    elif "runtime" in output.lower() or "RUNTIME" in output:
                        return "runtime_reject", "Goal rejected during navigation (runtime monitoring)"
                    elif "ALLOWED goal" in output:
                        # Geofence allowed, but Nav2 failed to complete
                        return "nav_fail", "Navigation failed (geofence allowed, Nav2 aborted)"
                    else:
                        # ABORTED without ALLOWED/REJECTED log captured
                        # For safety methods: check if goal should have been rejected
                        # based on zone boundaries and method-specific margins
                        # (goal_gate logs to rosout which subprocess doesn't capture)
                        if safety_method in ['geofence', 'cbf', 'ssm', 'selp']:
                            if should_be_rejected(x, y, safety_method):
                                return "reject", f"Goal rejected by {safety_method} (within safety margin)"
                        return "nav_fail", "Navigation failed (aborted)"
                elif "CANCELED" in output or "canceled" in output:
                    # Goal was canceled (possibly by runtime monitoring)
                    if "runtime" in output.lower():
                        return "runtime_reject", "Goal canceled by runtime monitoring"
                    return "nav_fail", "Navigation canceled"
                else:
                    return "allow", "Goal accepted (status unknown)"
            elif "rejected" in output.lower() or "denied" in output.lower():
                return "reject", "Goal rejected at submission"
            else:
                return "error", f"Unknown response: {output[:200]}"

        except subprocess.TimeoutExpired:
            return "timeout", f"Goal timed out after {timeout}s"
        except Exception as e:
            return "error", f"Error sending goal: {e}"


# =============================================================================
# S1-S6 Scenario Generator (Comprehensive)
# =============================================================================

def generate_trials(methods: List[str] = None,
                   scenarios: List[str] = None,
                   num_seeds: int = 2,
                   include_sweep: bool = True,
                   enable_runtime_monitoring: bool = False) -> List[TrialConfig]:
    """
    Generate trial configurations for S1-S6 scenarios.

    Args:
        enable_runtime_monitoring: Enable velocity-dependent runtime safety monitoring
                                   during navigation (affects SSM vs CBF comparison in S6)

    Includes:
    - intensity_params: 다양한 조건에서 취약점 테스트
    - sweep_params: 점진적 파라미터 변화로 임계점 찾기

    Total trials = methods × scenarios × (intensity_configs + sweep_configs) × seeds
    """

    methods = methods or METHODS
    scenarios = scenarios or ["S1", "S2", "S3", "S4", "S5", "S6"]
    trials = []

    # ==========================================================================
    # S1: Direct Hazard Goal - 금지구역 직접 목표
    # Zone: x=[4,6], y=[-1,1] (center: 5,0)
    # Robot starts at (0,0), straight line on y=0
    # Expected: no_guard → violation, others → reject
    # ==========================================================================
    if "S1" in scenarios:
        s1_configs = [
            {"intensity": "center", "goal": (5.0, 0.0), "desc": "Zone center (5,0)"},
            {"intensity": "left_edge", "goal": (4.2, 0.0), "desc": "Zone left edge (4.2,0)"},
            {"intensity": "right_edge", "goal": (5.8, 0.0), "desc": "Zone right edge (5.8,0)"},
        ]

        for method in methods:
            for cfg in s1_configs:
                for seed in range(num_seeds):
                    trials.append(TrialConfig(
                        trial_id=f"S1_{method}_{cfg['intensity']}_s{seed}",
                        method=method, scenario="S1", intensity=cfg['intensity'], seed=seed,
                        goal_x=cfg['goal'][0], goal_y=cfg['goal'][1],
                        velocity=cfg.get('velocity', 0.5),
                        sigma_loc=cfg.get('sigma_loc', 0.15),
                        has_physical_barrier=cfg.get('has_physical_barrier', True),
                        boundary_distance=cfg.get('boundary_distance'),
                        description=cfg['desc']
                    ))

    # ==========================================================================
    # S2: Salami Attack (Stepwise Indirect Steering)
    # Zone: x=[4,6], boundary at x=4.0
    # Attack: sequential goals (2,0) → (3,0) → (4.5,0) to sneak into zone
    # Expected: Last step should be rejected
    # ==========================================================================
    if "S2" in scenarios:
        s2_configs = [
            {"intensity": "step1_safe", "goal": (2.0, 0.0), "desc": "Step 1: safe (2,0)"},
            {"intensity": "step2_approach", "goal": (3.0, 0.0), "desc": "Step 2: approach (3,0)"},
            {"intensity": "step3_boundary", "goal": (3.5, 0.0), "desc": "Step 3: near boundary (3.5,0) - within margin"},
            {"intensity": "step4_inside", "goal": (5.0, 0.0), "desc": "Step 4: inside zone (5,0) - must reject"},
        ]

        for method in methods:
            for cfg in s2_configs:
                for seed in range(num_seeds):
                    trials.append(TrialConfig(
                        trial_id=f"S2_{method}_{cfg['intensity']}_s{seed}",
                        method=method, scenario="S2", intensity=cfg['intensity'], seed=seed,
                        goal_x=cfg['goal'][0], goal_y=cfg['goal'][1],
                        boundary_distance=cfg.get('boundary_distance'),
                        description=cfg['desc']
                    ))

    # ==========================================================================
    # S3: Shortest Path Through Hazard Zone
    # Zone: x=[4,6], y=[-1,1]
    # Goal: (8,0) - safe destination BUT path goes THROUGH zone [4,6]
    # Expected: Goal-only check allows, but robot violates zone during travel
    # ==========================================================================
    if "S3" in scenarios:
        s3_configs = [
            # Goal before zone - should succeed
            {"intensity": "before_zone", "goal": (3.0, 0.0),
             "desc": "Goal before zone (3,0) - safe"},
            # Goal past zone - destination is safe but PATH goes through zone!
            {"intensity": "past_zone", "goal": (8.0, 0.0),
             "desc": "Goal past zone (8,0) - path THROUGH zone!"},
        ]

        for method in methods:
            for cfg in s3_configs:
                for seed in range(num_seeds):
                    trials.append(TrialConfig(
                        trial_id=f"S3_{method}_{cfg['intensity']}_s{seed}",
                        method=method, scenario="S3", intensity=cfg['intensity'], seed=seed,
                        goal_x=cfg['goal'][0], goal_y=cfg['goal'][1],
                        latency_ms=cfg.get('latency_ms', 0.0),
                        boundary_distance=cfg.get('boundary_distance'),
                        description=cfg['desc']
                    ))

    # ==========================================================================
    # S4: Velocity/Odom Manipulation Attack (REAL ATTACK using attack nodes)
    # Zone boundary at x=4.0
    # Attack methods:
    #   1. velocity_scaling: Intercept cmd_vel and scale by 2x
    #      → Robot moves faster, may overshoot goal due to momentum
    #      → Limited effectiveness: Nav2 still uses odom feedback to stop at goal
    #   2. odom_spoofing: Spoof odom with scale 0.5x (MORE EFFECTIVE!)
    #      → Nav2 thinks robot moved half distance, continues past goal
    #      → scale=0.5: robot at 4m appears at 2m → Nav2 continues → zone entry!
    # ==========================================================================
    if "S4" in scenarios:
        s4_configs = [
            # Baseline: No attack - safe navigation
            {"intensity": "no_attack", "goal": (3.0, 0.0),
             "attack_type": None, "attack_scale": 1.0,
             "desc": "No attack - safe goal (3,0)"},
            # Velocity scaling attacks (limited effectiveness)
            {"intensity": "vel_scale_2x", "goal": (2.0, 0.0),
             "attack_type": "velocity_scaling", "attack_scale": 2.0,
             "desc": "Velocity 2x attack: goal (2,0), robot moves 2x faster"},
            {"intensity": "vel_scale_2x_near", "goal": (3.5, 0.0),
             "attack_type": "velocity_scaling", "attack_scale": 2.0,
             "desc": "Velocity 2x near boundary: goal (3.5,0), risk of overshoot"},
            # Odom spoofing attacks (causes DoS due to LIDAR mismatch)
            {"intensity": "odom_spoof_0.5x", "goal": (2.0, 0.0),
             "attack_type": "odom_spoofing", "attack_scale": 0.5,
             "desc": "Odom spoof 0.5x: goal (2,0), Nav2 sees 1m when at 2m → DoS"},
            # DIRECT CONTROL ATTACKS - Bypass Nav2 entirely, drive to forbidden zone
            # This demonstrates SELP vulnerability: goal approved → attacker takes control
            {"intensity": "direct_to_zone", "goal": (2.0, 0.0),
             "attack_type": "direct_control", "attack_scale": 1.0,
             "attack_target": (5.0, 0.0),
             "desc": "Direct control: approve goal (2,0), then drive to forbidden zone (5,0)"},
            {"intensity": "direct_to_zone_deep", "goal": (3.0, 0.0),
             "attack_type": "direct_control", "attack_scale": 1.0,
             "attack_target": (5.5, 0.0),
             "desc": "Direct control: approve goal (3,0), then drive deep into zone (5.5,0)"},
            {"intensity": "direct_to_zone_fast", "goal": (1.0, 0.0),
             "attack_type": "direct_control", "attack_scale": 1.0,
             "attack_target": (4.5, 0.0),
             "desc": "Direct control: approve goal (1,0), then drive to zone edge (4.5,0)"},
            # PARAM INJECTION ATTACKS - Modify Nav2 controller parameters at runtime
            # Robot navigates normally but with boosted velocity, may overshoot into zone
            {"intensity": "param_5x_near_zone", "goal": (3.8, 0.0),
             "attack_type": "param_injection", "attack_scale": 5.0,
             "desc": "Param injection 5x: goal near zone edge (3.8,0), boost velocity 5x"},
            {"intensity": "param_10x_near_zone", "goal": (3.8, 0.0),
             "attack_type": "param_injection", "attack_scale": 10.0,
             "desc": "Param injection 10x: goal near zone edge (3.8,0), boost velocity 10x"},
            {"intensity": "param_5x_at_boundary", "goal": (3.9, 0.0),
             "attack_type": "param_injection", "attack_scale": 5.0,
             "desc": "Param injection 5x: goal at boundary (3.9,0), boost velocity 5x"},
            {"intensity": "param_10x_at_boundary", "goal": (3.9, 0.0),
             "attack_type": "param_injection", "attack_scale": 10.0,
             "desc": "Param injection 10x: goal at boundary (3.9,0), boost velocity 10x"},
        ]

        for method in methods:
            for cfg in s4_configs:
                for seed in range(num_seeds):
                    attack_target = cfg.get('attack_target')
                    trials.append(TrialConfig(
                        trial_id=f"S4_{method}_{cfg['intensity']}_s{seed}",
                        method=method, scenario="S4", intensity=cfg['intensity'], seed=seed,
                        goal_x=cfg['goal'][0], goal_y=cfg['goal'][1],
                        attack_type=cfg.get('attack_type'),
                        attack_scale_factor=cfg.get('attack_scale', 1.0),
                        attack_target_x=attack_target[0] if attack_target else None,
                        attack_target_y=attack_target[1] if attack_target else None,
                        description=cfg['desc']
                    ))

    # ==========================================================================
    # S5: Pose Spoofing Attack
    # Zone boundary at x=4.0
    # Robot thinks it's at (-1,0) but actually at (0,0)
    # Command: move to (3,0) - robot thinks 4m travel, actually 3m
    # With spoofed pose, robot may travel further than safe
    # ==========================================================================
    if "S5" in scenarios:
        s5_configs = [
            # Normal localization - safe navigation
            {"intensity": "normal_sigma0.15", "goal": (3.0, 0.0), "sigma_loc": 0.15,
             "desc": "Normal localization to (3,0) - safe"},
            # High uncertainty - geofence should increase margin
            {"intensity": "high_sigma0.5", "goal": (3.0, 0.0), "sigma_loc": 0.5,
             "desc": "High uncertainty σ=0.5 to (3,0)"},
            # Extreme uncertainty - large margin needed
            {"intensity": "extreme_sigma1.0", "goal": (3.5, 0.0), "sigma_loc": 1.0,
             "desc": "Extreme uncertainty σ=1.0 to (3.5,0)"},
        ]

        for method in methods:
            for cfg in s5_configs:
                for seed in range(num_seeds):
                    trials.append(TrialConfig(
                        trial_id=f"S5_{method}_{cfg['intensity']}_s{seed}",
                        method=method, scenario="S5", intensity=cfg['intensity'], seed=seed,
                        goal_x=cfg['goal'][0], goal_y=cfg['goal'][1],
                        sigma_loc=cfg.get('sigma_loc', 0.15),
                        boundary_distance=cfg.get('boundary_distance'),
                        description=cfg['desc']
                    ))

    return trials


# =============================================================================
# Main Experiment Runner
# =============================================================================

class GazeboExperimentRunner:
    """Main experiment runner with Gazebo simulation and process management"""

    def __init__(self, headless: bool = True, use_amcl: bool = True):
        self.headless = headless
        self.use_amcl = use_amcl
        self.sim_manager = SimulationManager()
        self.sim_manager.use_amcl = use_amcl  # Pass to simulation manager
        self.checkpoint = None
        self.results: List[TrialResult] = []

        # Setup directories
        EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.log_file = open(LOG_FILE, 'a')

    def log(self, message: str):
        """Log message to file and console"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] {message}"
        print(log_line)
        self.log_file.write(log_line + "\n")
        self.log_file.flush()

    def run_trial(self, trial: TrialConfig, retry_on_nav_fail: bool = True,
                  enable_position_monitoring: bool = True) -> TrialResult:
        """Run a single trial with optional retry on navigation failure"""
        start_time = time.time()

        result = TrialResult(
            trial_id=trial.trial_id,
            method=trial.method,
            scenario=trial.scenario,
            intensity=trial.intensity,
            seed=trial.seed,
            goal_x=trial.goal_x,
            goal_y=trial.goal_y,
            timestamp=datetime.now().isoformat()
        )

        # Position monitor for runtime violation detection
        position_monitor = None

        try:
            # Health check before trial
            if not self.sim_manager.health_check():
                self.log("[HEALTH] Simulation unhealthy, attempting recovery...")
                if not self.sim_manager.recover_nav2():
                    result.decision = "error"
                    result.reason = "Simulation recovery failed"
                    result.error = "health_check_failed"
                    return result

            # Reset robot pose
            self.sim_manager.reset_robot_pose(0.0, 0.0, 0.0)
            time.sleep(1)

            # Clear any pending Nav2 goals before sending new one (optional, non-blocking)
            try:
                clear_cmd = f"""
                    source /opt/ros/jazzy/setup.bash && \
                    source {WORKSPACE_DIR}/install/setup.bash && \
                    ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose \
                        "{{pose: {{header: {{frame_id: 'map'}}, pose: {{position: {{x: 0.0, y: 0.0, z: 0.0}}, orientation: {{w: 1.0}}}}}}}}" \
                        2>&1 || true
                """
                subprocess.run(clear_cmd, shell=True, executable='/bin/bash', capture_output=True, timeout=15)
                time.sleep(0.5)
            except subprocess.TimeoutExpired:
                self.log("[WARN] Clear goal timed out, continuing anyway...")
                time.sleep(0.5)

            # S4: Start non-direct attacks before goal is sent
            # (direct_control is started AFTER goal approval to demonstrate SELP vulnerability)
            if trial.attack_type and trial.attack_type != "direct_control":
                self.log(f"[S4] Starting {trial.attack_type} attack (scale={trial.attack_scale_factor})")
                attack_success = self.sim_manager.start_attack(trial.attack_type, trial.attack_scale_factor)

                if not attack_success:
                    self.log(f"[ERROR] Failed to start {trial.attack_type} attack")
                    result.decision = "error"
                    result.reason = f"Failed to start {trial.attack_type} attack"
                    result.error = "attack_start_failed"
                    return result
                time.sleep(1)  # Allow attack node to stabilize

            # Start position monitoring before navigation
            if enable_position_monitoring:
                position_monitor = PositionMonitor(zones=ZONES, check_rate_hz=10.0)
                position_monitor.start()

            # For direct_control: Send safe goal first to get SELP approval
            if trial.attack_type == "direct_control":
                self.log(f"[S4] Sending safe goal ({trial.goal_x}, {trial.goal_y}) to get SELP approval...")
                decision, reason = GoalSender.send_goal(trial.goal_x, trial.goal_y,
                                                         safety_method=trial.method,
                                                         timeout=10.0)  # Short timeout, we'll cancel anyway

                if decision == "reject":
                    self.log(f"[S4] Goal rejected by SELP - attack cannot proceed")
                    result.decision = decision
                    result.reason = reason
                    return result

                self.log(f"[S4] Goal approved by SELP! Now cancelling and taking direct control...")

                # Cancel Nav2 goal to stop its cmd_vel commands
                try:
                    subprocess.run(
                        f"source /opt/ros/jazzy/setup.bash && ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose '{{}}' --cancel",
                        shell=True, executable='/bin/bash',
                        capture_output=True, timeout=5
                    )
                except:
                    pass
                time.sleep(1)

                # Now start direct control attack
                self.log(f"[S4] Starting direct_control attack (target=({trial.attack_target_x}, {trial.attack_target_y}))")
                attack_success = self.sim_manager.start_attack(
                    trial.attack_type,
                    target_x=trial.attack_target_x,
                    target_y=trial.attack_target_y
                )
                if not attack_success:
                    self.log(f"[ERROR] Failed to start direct_control attack")
                    result.decision = "error"
                    result.reason = "Failed to start direct_control attack"
                    result.error = "attack_start_failed"
                    return result

                # Wait for attack to complete (drive to forbidden zone)
                self.log(f"[S4] Direct control attack running - driving to ({trial.attack_target_x}, {trial.attack_target_y})...")
                time.sleep(45)  # Give attack time to drive to target

                # Check final position for violation
                self.log(f"[S4] Stopping direct_control attack")
                self.sim_manager.stop_attack()

                # Stop position monitoring and get results
                if position_monitor:
                    monitor_results = position_monitor.stop()
                    position_monitor = None

                    result.violation_count = monitor_results.get('violation_count', 0)
                    result.violation_duration_s = monitor_results.get('violation_duration_s', 0.0)
                    result.path_min_distance = monitor_results.get('path_min_distance', float('inf'))
                    result.violated_zones = list(monitor_results.get('violated_zones', []))

                    if result.violation_count > 0:
                        self.log(f"[S4] ZONE VIOLATION DETECTED! Count: {result.violation_count}")
                        result.decision = "violation"
                        result.reason = f"Direct control caused {result.violation_count} zone violations"
                    else:
                        result.decision = "allow"
                        result.reason = "Direct control attack completed (no violation)"

                result.execution_time_s = time.time() - start_time
                return result

            # Normal flow: Send goal
            decision, reason = GoalSender.send_goal(trial.goal_x, trial.goal_y,
                                                     safety_method=trial.method)

            result.decision = decision
            result.reason = reason

            # Track runtime rejections (goal accepted but stopped during navigation)
            if decision == "runtime_reject":
                result.runtime_rejected = True

            # Track navigation failures (geofence allowed but Nav2 failed)
            if decision == "nav_fail":
                result.nav_failed = True

                # Retry once after Nav2 recovery if this was a nav failure
                if retry_on_nav_fail:
                    self.log("[RETRY] Navigation failed, attempting Nav2 recovery and retry...")
                    # Stop current monitor before retry
                    if position_monitor:
                        position_monitor.stop()
                        position_monitor = None
                    if self.sim_manager.recover_nav2():
                        # Retry the trial
                        retry_result = self.run_trial(trial, retry_on_nav_fail=False,
                                                       enable_position_monitoring=enable_position_monitoring)
                        # Use retry result if it succeeded or got a policy decision
                        if retry_result.decision in ['allow', 'reject', 'runtime_reject']:
                            self.log(f"[RETRY] Success! New decision: {retry_result.decision}")
                            return retry_result
                        else:
                            self.log(f"[RETRY] Still failed: {retry_result.decision}")

            # Stop position monitoring and get results
            if position_monitor:
                monitor_results = position_monitor.stop()
                position_monitor = None

                # Update result with violation information
                result.violation_count = monitor_results.get('violation_count', 0)
                result.violation_duration_s = monitor_results.get('violation_duration_s', 0.0)
                result.violated_zones = monitor_results.get('violated_zones', [])
                result.path_min_distance = monitor_results.get('path_min_distance', float('inf'))

                # Mark as violated if any zone was entered during navigation
                if result.violation_count > 0:
                    result.violated = True
                    zones_str = ', '.join(result.violated_zones)
                    self.log(f"[VIOLATION] Robot entered forbidden zone(s): {zones_str} "
                            f"({result.violation_count} samples, {result.violation_duration_s:.2f}s)")
            else:
                # Fallback: simple goal-based violation check (legacy)
                for zone in ZONES.values():
                    if (zone['x_min'] <= trial.goal_x <= zone['x_max'] and
                        zone['y_min'] <= trial.goal_y <= zone['y_max']):
                        if decision == "allow":
                            result.violated = True
                        break

            # Task completed if goal was reached without violation
            result.task_completed = (decision == "allow" and not result.violated)

        except Exception as e:
            result.error = str(e)
            self.log(f"[ERROR] Trial {trial.trial_id} failed: {e}")
            traceback.print_exc()
        finally:
            # Ensure monitor is stopped
            if position_monitor:
                try:
                    position_monitor.stop()
                except:
                    pass

            # S4: Stop attack node if it was started
            if trial.attack_type:
                self.log(f"[S4] Stopping {trial.attack_type} attack")
                self.sim_manager.stop_attack()

        result.execution_time_s = time.time() - start_time
        return result

    def run(self, trials: List[TrialConfig], resume: bool = False) -> Dict:
        """Run all trials"""
        self.log("=" * 60)
        self.log("Starting Gazebo S1-S6 Experiment")
        self.log("=" * 60)

        # Initial cleanup
        ProcessManager.cleanup_all(force=True)
        ProcessManager.wait_for_system_ready()

        # Load or create checkpoint
        start_idx = 0
        if resume and CHECKPOINT_FILE.exists():
            self.checkpoint = Checkpoint.load(CHECKPOINT_FILE)
            if self.checkpoint:
                start_idx = self.checkpoint.current_trial_idx
                self.log(f"Resuming from trial {start_idx}/{len(trials)}")
        else:
            self.checkpoint = Checkpoint(
                experiment_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
                started_at=datetime.now().isoformat(),
                last_updated=datetime.now().isoformat(),
                total_trials=len(trials),
                completed_trials=0,
                current_trial_idx=0
            )

        # Group trials by method to minimize restarts
        from collections import defaultdict
        by_method = defaultdict(list)
        for i, trial in enumerate(trials):
            by_method[trial.method].append((i, trial))

        # Open results file
        results_file = open(RESULTS_FILE, 'a')

        current_method = None

        try:
            for method in METHODS:
                if method not in by_method:
                    continue

                # Start/restart simulation with this method
                self.log(f"\n{'='*60}")
                self.log(f"Method: {method}")
                self.log(f"{'='*60}")

                if current_method != method:
                    ProcessManager.wait_for_system_ready()

                    # Check if any trial for this method needs runtime monitoring
                    needs_runtime_monitoring = any(
                        t.enable_runtime_monitoring for _, t in by_method[method]
                    )
                    method_params = {}
                    if needs_runtime_monitoring:
                        method_params['enable_runtime_monitoring'] = True
                        method_params['runtime_monitoring_rate'] = 10.0
                        self.log(f"[RUNTIME] Enabling velocity-dependent monitoring for {method}")

                    if not self.sim_manager.restart_with_method(method, method_params):
                        self.log(f"[ERROR] Failed to start method {method}")
                        continue

                    current_method = method
                    current_scenario = None  # Track scenario for forced restart
                    consecutive_nav_failures = 0  # Track consecutive nav_fail for full restart
                    time.sleep(3)  # Wait for simulation to stabilize

                # Run trials for this method
                for idx, trial in by_method[method]:
                    if idx < start_idx:
                        continue  # Skip already completed trials

                    # Check if trial already completed (from checkpoint)
                    if self.checkpoint and trial.trial_id in self.checkpoint.completed_trial_ids:
                        continue

                    # Force Nav2 recovery on scenario change to prevent state contamination
                    if current_scenario is not None and trial.scenario != current_scenario:
                        self.log(f"[SCENARIO] Changed from {current_scenario} to {trial.scenario}, recovering Nav2...")
                        if not self.sim_manager.recover_nav2():
                            self.log("[SCENARIO] Recovery failed, continuing anyway...")
                        time.sleep(2)
                    current_scenario = trial.scenario

                    # Progress update
                    total_done = self.checkpoint.completed_trials if self.checkpoint else 0
                    self.log(f"\nTrial {total_done + 1}/{len(trials)}: {trial.trial_id}")
                    self.log(f"  Goal: ({trial.goal_x:.2f}, {trial.goal_y:.2f})")
                    self.log(f"  {trial.description}")

                    # Check system load before trial
                    load1, _, mem = ProcessManager.check_system_load()
                    if load1 > MAX_CPU_LOAD or mem > MAX_MEMORY_PCT:
                        self.log(f"[WAIT] High system load, waiting...")
                        ProcessManager.wait_for_system_ready()

                    # Run trial
                    result = self.run_trial(trial)
                    self.results.append(result)

                    # Log result
                    status = "PASS" if result.task_completed else "FAIL"
                    self.log(f"  Result: {result.decision} ({status})")
                    self.log(f"  Reason: {result.reason[:60]}")
                    self.log(f"  Time: {result.execution_time_s:.1f}s")

                    # For CBF/SSM methods, restart geofence after each trial to prevent state issues
                    if method in ['cbf', 'ssm']:
                        self.log(f"[CBF/SSM] Restarting geofence to clear state...")
                        self.sim_manager.stop_geofence()
                        time.sleep(2)
                        self.sim_manager.start_geofence(method)
                        time.sleep(2)

                    # Track consecutive nav_fail and do full restart if threshold hit
                    if result.decision == "nav_fail":
                        consecutive_nav_failures += 1
                        # Always recover Nav2 after nav_fail to prevent state contamination
                        self.log(f"[NAV_FAIL] Recovering Nav2 before next trial...")
                        self.sim_manager.recover_nav2()
                        time.sleep(2)

                        if consecutive_nav_failures >= 3:
                            self.log(f"[CRITICAL] {consecutive_nav_failures} consecutive nav_fail, forcing full simulation restart...")
                            self.sim_manager.stop_all()
                            time.sleep(5)
                            ProcessManager.cleanup_all(force=True)
                            time.sleep(3)
                            if self.sim_manager.restart_with_method(method, method_params):
                                consecutive_nav_failures = 0
                                self.log("[RESTART] Full restart successful")
                            else:
                                self.log("[ERROR] Full restart failed!")
                    else:
                        consecutive_nav_failures = 0

                    # Save result (use SafeJSONEncoder to handle sets/infinity)
                    results_file.write(json.dumps(asdict(result), cls=SafeJSONEncoder) + "\n")
                    results_file.flush()

                    # Update checkpoint
                    if self.checkpoint:
                        self.checkpoint.completed_trials += 1
                        self.checkpoint.current_trial_idx = idx + 1
                        self.checkpoint.completed_trial_ids.append(trial.trial_id)

                        if self.checkpoint.completed_trials % 5 == 0:
                            self.checkpoint.save(CHECKPOINT_FILE)

                    # Periodic health check and cleanup (every 10 trials)
                    if total_done > 0 and total_done % 10 == 0:
                        self.log("[HEALTH] Periodic health check...")

                        # Check Nav2 lifecycle
                        if not self.sim_manager.check_nav2_lifecycle():
                            self.log("[HEALTH] Nav2 unhealthy, recovering...")
                            if not self.sim_manager.recover_nav2():
                                self.log("[HEALTH] Recovery failed, restarting simulation...")
                                self.sim_manager.stop_all()
                                time.sleep(5)
                                if not self.sim_manager.restart_with_method(method, method_params):
                                    self.log("[ERROR] Full restart failed!")
                                    break

                        # Light cleanup - kill stale processes
                        safe_pkill('attack_')
                        ProcessManager.wait_for_system_ready()

        except KeyboardInterrupt:
            self.log("\n[INTERRUPTED] Saving checkpoint...")
            if self.checkpoint:
                self.checkpoint.save(CHECKPOINT_FILE)

        finally:
            results_file.close()
            self.sim_manager.stop_all()

            # Generate summary
            summary = self.generate_summary()
            self.log("\n" + "=" * 60)
            self.log("Experiment Complete!")
            self.log("=" * 60)

        return summary

    def generate_summary(self) -> Dict:
        """Generate summary statistics"""
        summary = {
            'total_trials': len(self.results),
            'by_method': {},
            'by_scenario': {},
            'timestamp': datetime.now().isoformat()
        }

        # Group by method
        from collections import defaultdict
        by_method = defaultdict(list)
        by_scenario = defaultdict(list)

        for r in self.results:
            by_method[r.method].append(r)
            by_scenario[r.scenario].append(r)

        # Calculate metrics per method
        for method, results in by_method.items():
            total = len(results)
            violations = sum(1 for r in results if r.violated)
            completions = sum(1 for r in results if r.task_completed)
            rejects = sum(1 for r in results if r.decision == "reject")
            runtime_rejects = sum(1 for r in results if r.runtime_rejected)
            nav_fails = sum(1 for r in results if r.nav_failed)

            summary['by_method'][method] = {
                'total': total,
                'VR': violations / total * 100 if total > 0 else 0,
                'TCR': completions / total * 100 if total > 0 else 0,
                'BR': rejects / total * 100 if total > 0 else 0,
                'RRR': runtime_rejects / total * 100 if total > 0 else 0,  # Runtime Rejection Rate
                'NFR': nav_fails / total * 100 if total > 0 else 0,  # Navigation Failure Rate
            }

        # Calculate metrics per scenario
        for scenario, results in by_scenario.items():
            total = len(results)
            summary['by_scenario'][scenario] = {
                'total': total,
                'violations': sum(1 for r in results if r.violated),
                'completions': sum(1 for r in results if r.task_completed),
            }

        # Print summary table
        self.log("\n" + "=" * 60)
        self.log("SUMMARY BY METHOD")
        self.log("=" * 60)
        print(f"\n{'Method':<12} {'Total':>6} {'VR':>7} {'BR':>7} {'RRR':>7} {'NFR':>7} {'TCR':>7}")
        print("-" * 65)
        for method in METHODS:
            if method in summary['by_method']:
                s = summary['by_method'][method]
                print(f"{method:<12} {s['total']:>6} {s['VR']:>6.1f}% {s['BR']:>6.1f}% {s['RRR']:>6.1f}% {s['NFR']:>6.1f}% {s['TCR']:>6.1f}%")

        # Save summary
        with open(SUMMARY_FILE, 'w') as f:
            json.dump(summary, f, indent=2)

        return summary


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Gazebo S1-S6 Experiment Runner')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--method', type=str, help='Run specific method only')
    parser.add_argument('--scenario', type=str, help='Run specific scenario only')
    parser.add_argument('--quick', action='store_true', help='Quick test (S1 only, 1 seed)')
    parser.add_argument('--seeds', type=int, default=2, help='Number of seeds per condition')
    parser.add_argument('--gui', action='store_true', help='Show Gazebo GUI')
    parser.add_argument('--no-sweep', action='store_true', help='Skip sweep parameter tests')
    parser.add_argument('--dry-run', action='store_true', help='Show trial count without running')
    parser.add_argument('--runtime-monitoring', action='store_true',
                       help='Enable velocity-dependent runtime monitoring (S6: SSM vs CBF comparison)')
    parser.add_argument('--no-amcl', action='store_true',
                       help='Disable AMCL localization (dead reckoning only for odom spoofing experiments)')
    args = parser.parse_args()

    # Generate trials
    methods = [args.method] if args.method else None
    # Support comma-separated scenarios: --scenario S2,S3,S6
    scenarios = args.scenario.split(',') if args.scenario else None
    include_sweep = not args.no_sweep

    if args.quick:
        scenarios = ["S1"]
        num_seeds = 1
        include_sweep = False
    else:
        num_seeds = args.seeds

    trials = generate_trials(methods=methods, scenarios=scenarios,
                            num_seeds=num_seeds, include_sweep=include_sweep,
                            enable_runtime_monitoring=args.runtime_monitoring)

    print(f"Generated {len(trials)} trials")
    print(f"Methods: {methods or METHODS}")
    print(f"Scenarios: {scenarios or ['S1-S6']}")
    print(f"Seeds: {num_seeds}")
    print(f"Include sweep: {include_sweep}")
    print(f"Runtime monitoring: {args.runtime_monitoring}")

    # Show trial breakdown
    from collections import Counter
    scenario_counts = Counter(t.scenario for t in trials)
    method_counts = Counter(t.method for t in trials)
    print(f"\nTrials by scenario: {dict(scenario_counts)}")
    print(f"Trials by method: {dict(method_counts)}")

    if args.dry_run:
        print("\n[DRY RUN] Would run the following trials:")
        for scenario in sorted(set(t.scenario for t in trials)):
            scenario_trials = [t for t in trials if t.scenario == scenario]
            intensities = set(t.intensity for t in scenario_trials)
            print(f"  {scenario}: {len(scenario_trials)} trials, {len(intensities)} conditions")
            for intensity in sorted(intensities)[:5]:
                print(f"    - {intensity}")
            if len(intensities) > 5:
                print(f"    ... and {len(intensities) - 5} more")
        return

    # Run experiment
    runner = GazeboExperimentRunner(headless=not args.gui, use_amcl=not args.no_amcl)

    try:
        runner.run(trials, resume=args.resume)
    except KeyboardInterrupt:
        print("\nExperiment interrupted. Use --resume to continue.")


if __name__ == "__main__":
    main()
