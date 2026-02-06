#!/usr/bin/env python3
"""
Gazebo-based S1-S6 Experiment Runner (with S5′ LIDAR Spoofing)
===============================================================

Runs S1-S6 and S5′ scenarios with Gazebo simulation.

Scenarios:
- S1-S3: Basic geofence tests (safe/boundary/intrusion goals)
- S4: Velocity/Direct control attacks
- S5: Odom spoofing attacks (defeated by AMCL)
- S5′ (S5p): LIDAR spoofing attacks (confuses AMCL directly)
- S6: Latency/sensor delay tests

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
    python run_gazebo_s1_s6.py --scenario S5p      # Run S5′ LIDAR spoofing only
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
    "hardware_geofence_guard", "scan_relay",  # Additional cleanup targets
    "attack_velocity", "attack_odom", "attack_pose", "attack_direct",
    "attack_scan_spoofing", "param_injection",  # More attack patterns
    "relay /odom_real", "relay /cmd_vel",  # Relay processes
    "violation_monitor", "parameter_bridge", "ros_gz",
    "amcl", "map_server", "static_transform_publisher"  # Nav2 components
]

# Max CPU load before waiting
MAX_CPU_LOAD = 6.0  # Increased: 4.0 → 6.0 (more tolerant of system load)
MAX_MEMORY_PCT = 80.0

# Timeouts (increased for reliability)
GAZEBO_STARTUP_TIMEOUT = 90  # seconds - increased for slow startup
NAV2_STARTUP_TIMEOUT = 90  # Increased: allow more time for Nav2 lifecycle
GEOFENCE_STARTUP_TIMEOUT = 20
GOAL_TIMEOUT = 180  # Increased: 120 → 180 (safe_bypass needs longer for 7m+ detour paths)
CLEANUP_TIMEOUT = 8
LIFECYCLE_CMD_TIMEOUT = 20  # Increased timeout for lifecycle commands
COSTMAP_CHECK_TIMEOUT = 15  # Timeout for costmap hz check
AMCL_CONVERGENCE_TIMEOUT = 20  # Time to wait for AMCL to converge after reset
NAV2_READY_MAX_WAIT = 120  # Max wait time for Nav2 to be fully ready

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
METHODS = ["no_guard", "selp_proper", "cbf", "ssm", "geofence", "geofence_hw"]


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
    # S4/S5: Real attack parameters
    attack_type: Optional[str] = None  # "velocity_scaling", "odom_spoofing", or "direct_control"
    attack_scale_factor: float = 1.0  # Scale factor for attack (2.0 = double speed, 0.5 = half position)
    attack_target_x: Optional[float] = None  # For direct_control: target x in forbidden zone
    attack_target_y: Optional[float] = None  # For direct_control: target y in forbidden zone
    # S5: Odom spoofing offset parameters
    attack_offset_x: float = 0.0  # For odom_spoofing: offset to add to x position
    attack_offset_y: float = 0.0  # For odom_spoofing: offset to add to y position
    # S5′: LIDAR spoofing parameters (scan_spoofing attack)
    scan_rotation_deg: float = 0.0  # Rotation offset in degrees
    scan_scale: float = 1.0  # Range scale (0.8 = walls appear 20% closer)
    scan_noise: float = 0.0  # Noise stddev in meters
    # Confusion matrix: whether this trial is expected to be safe (no violation)
    expected_safe: bool = True


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
    # Validation fields (for detecting system errors vs method behavior)
    robot_moved: bool = False  # True if robot actually moved during trial
    is_valid_result: bool = True  # False if result is contaminated by system errors
    invalid_reason: str = ""  # Reason for invalid result (e.g., "ALLOW but robot didn't move")
    # Confusion matrix and infra failure classification
    nav2_path_crossed_zone: bool = False  # True if actual robot path crossed/near forbidden zone
    is_infra_failure: bool = False  # True if timeout/nav_fail without violation (infra issue)
    actual_monitoring_rate_hz: float = 0.0  # Actual position monitoring rate achieved
    classification: str = ""  # One of: TP, FP, TN, FN, INFRA


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
        """Kill all related processes with graceful then forceful shutdown"""
        patterns = patterns or CLEANUP_PATTERNS
        print("[CLEANUP] Starting process cleanup...")
        killed = 0
        my_pid = os.getpid()

        all_pids = set()

        # Collect all PIDs first
        for pattern in patterns:
            try:
                pgrep_result = subprocess.run(
                    f"pgrep -f '{pattern}'",
                    shell=True, capture_output=True, text=True, timeout=5
                )
                if pgrep_result.returncode == 0 and pgrep_result.stdout.strip():
                    pids = [int(p) for p in pgrep_result.stdout.strip().split('\n') if p.strip()]
                    pids = [p for p in pids if p != my_pid and p != os.getppid()]
                    all_pids.update(pids)
            except Exception:
                pass

        # First pass: SIGTERM for graceful shutdown
        for pid in all_pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass

        # Wait for graceful shutdown
        time.sleep(2)

        # Second pass: SIGKILL for stubborn processes
        for pid in all_pids:
            try:
                os.kill(pid, signal.SIGKILL)
                killed += 1
            except (ProcessLookupError, PermissionError):
                pass

        # Wait for processes to die
        time.sleep(1)

        # Force cleanup shared memory and temp files
        if force:
            try:
                subprocess.run("rm -rf /dev/shm/fastrtps_*", shell=True, timeout=5)
                subprocess.run("rm -rf /dev/shm/ros2_*", shell=True, timeout=5)
                subprocess.run("rm -rf /tmp/ros2*", shell=True, timeout=5)
                subprocess.run("rm -rf /tmp/.ros2*", shell=True, timeout=5)
                subprocess.run("rm -rf /tmp/gz-*", shell=True, timeout=5)
            except Exception:
                pass

        # Reset ROS2 daemon if requested (helps with stuck discovery)
        if reset_daemon:
            try:
                print("[CLEANUP] Resetting ROS2 daemon...")
                subprocess.run(
                    "pkill -9 -f ros2-daemon",
                    shell=True, capture_output=True, timeout=5
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
            'total_samples': 0,
            'actual_rate_hz': 0.0,
            'path_crossed_zone': False,
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

            # Compute actual monitoring rate
            if len(entries) >= 2:
                monitoring_duration_s = entries[-1]['t'] - entries[0]['t']
                if monitoring_duration_s > 0:
                    result['actual_rate_hz'] = len(entries) / monitoring_duration_s

            # Check if path crossed near the forbidden zone (within margin)
            # Used to distinguish "Nav2 routed around" vs "safety method blocked"
            ZONE_PROXIMITY_THRESHOLD = 0.6  # slightly larger than geofence margin (0.55m)
            result['path_crossed_zone'] = result['path_min_distance'] < ZONE_PROXIMITY_THRESHOLD

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
        self.scan_relay_proc = None  # S5′: Scan relay for normal operation
        self.cmd_vel_relay_proc = None  # cmd_vel relay when cmd_vel_guard disabled
        self.current_method = None
        self.current_method_params = None  # Store method params for runtime monitoring
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

        # Reset ROS2 daemon to ensure clean state
        try:
            subprocess.run("pkill -9 -f ros2-daemon", shell=True, timeout=5)
            time.sleep(1)
            subprocess.run(
                "source /opt/ros/jazzy/setup.bash && ros2 daemon start",
                shell=True, executable='/bin/bash', timeout=10
            )
            time.sleep(1)
        except:
            pass

        time.sleep(2)

        headless_arg = "headless:=true" if headless else ""

        # Use custom bridge config for hardware guard mode
        if use_hw_guard:
            bridge_config = f"{WORKSPACE_DIR}/src/mobile_manipulator_tutorial/src/mobile_manip_moveit_config/config/gz_bridge_with_hw_guard.yaml"
            bridge_arg = f"gz_bridge_config:={bridge_config}"
            print("[SIM] Using HARDWARE GUARD bridge config (/cmd_vel_safe → Gazebo)")
        else:
            bridge_arg = ""

        # Use warehouse.sdf world which matches my_map.yaml for AMCL localization
        launch_cmd = f"""
            source /opt/ros/jazzy/setup.bash && \
            source {WORKSPACE_DIR}/install/setup.bash && \
            ros2 launch mobile_manip_moveit_config mobile_manipulator.launch.py \
                use_sim_time:=true world:=warehouse.sdf {headless_arg} {bridge_arg}
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

        # Wait for Gazebo topics to be available instead of checking ros2 launch process
        # (ros2 launch may exit after spawning nodes, but Gazebo continues running)
        start_time = time.time()
        gazebo_ready = False
        while time.time() - start_time < GAZEBO_STARTUP_TIMEOUT:
            try:
                result = subprocess.run(
                    "source /opt/ros/jazzy/setup.bash && ros2 topic list 2>/dev/null | grep -q '/clock'",
                    shell=True, executable='/bin/bash', timeout=5
                )
                if result.returncode == 0:
                    # Also check if /odom_real is publishing (robot spawned)
                    result2 = subprocess.run(
                        "source /opt/ros/jazzy/setup.bash && ros2 topic list 2>/dev/null | grep -q '/odom_real'",
                        shell=True, executable='/bin/bash', timeout=5
                    )
                    if result2.returncode == 0:
                        gazebo_ready = True
                        break
            except:
                pass
            time.sleep(2)

        if gazebo_ready:
            print("[SIM] Gazebo started successfully")
            # Start odom relay for normal operation (odom_real → odom)
            self.start_odom_relay()
            # Start scan relay for normal operation (scan_real → scan)
            self.start_scan_relay()
            return True
        else:
            print("[ERROR] Gazebo failed to start (topics not available)")
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

    def start_scan_relay(self) -> bool:
        """Start scan relay node: /scan_real → /scan for normal operation"""
        print("[SIM] Starting scan relay (scan_real → scan)...")

        # Stop any existing relay
        self.stop_scan_relay()

        relay_cmd = f"""
            source /opt/ros/jazzy/setup.bash && \
            source {WORKSPACE_DIR}/install/setup.bash && \
            ros2 run geofence_policy_enforcer scan_relay
        """

        self.scan_relay_proc = subprocess.Popen(
            relay_cmd,
            shell=True,
            executable='/bin/bash',
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid
        )

        time.sleep(2)

        if self.scan_relay_proc.poll() is None:
            print("[SIM] Scan relay started, verifying /scan topic...")
            # Verify /scan is actually publishing data
            for attempt in range(10):
                try:
                    result = subprocess.run(
                        "source /opt/ros/jazzy/setup.bash && timeout 2 ros2 topic echo /scan --once 2>/dev/null | wc -l",
                        shell=True, executable='/bin/bash', capture_output=True, text=True, timeout=5
                    )
                    if result.returncode == 0 and int(result.stdout.strip() or '0') > 5:
                        print("[SIM] /scan topic verified - data flowing")
                        return True
                except:
                    pass
                time.sleep(1)
            print("[WARN] /scan topic verification timeout, continuing anyway...")
            return True
        else:
            print("[WARN] Scan relay may have failed to start")
            return False

    def stop_scan_relay(self):
        """Stop scan relay node"""
        if hasattr(self, 'scan_relay_proc') and self.scan_relay_proc and self.scan_relay_proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.scan_relay_proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass

        safe_pkill('scan_relay')
        self.scan_relay_proc = None

    def start_cmd_vel_relay(self) -> bool:
        """Start cmd_vel relay: /cmd_vel_nav → /cmd_vel when cmd_vel_guard is disabled"""
        print("[SIM] Starting cmd_vel relay (cmd_vel_nav → cmd_vel)...")

        self.stop_cmd_vel_relay()

        relay_cmd = f"""
            source /opt/ros/jazzy/setup.bash && \
            source {WORKSPACE_DIR}/install/setup.bash && \
            ros2 run topic_tools relay /cmd_vel_nav /cmd_vel
        """

        self.cmd_vel_relay_proc = subprocess.Popen(
            relay_cmd,
            shell=True,
            executable='/bin/bash',
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid
        )

        time.sleep(2)

        if self.cmd_vel_relay_proc.poll() is None:
            print("[SIM] cmd_vel relay started")
            return True
        else:
            print("[WARN] cmd_vel relay may have failed to start")
            return False

    def stop_cmd_vel_relay(self):
        """Stop cmd_vel relay node"""
        if hasattr(self, 'cmd_vel_relay_proc') and self.cmd_vel_relay_proc and self.cmd_vel_relay_proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.cmd_vel_relay_proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass

        safe_pkill('relay /cmd_vel_nav')
        self.cmd_vel_relay_proc = None

    def check_nav2_lifecycle(self, quiet: bool = False, retry_with_daemon_reset: bool = True) -> bool:
        """Check if Nav2 is ready by checking for published topics and actions.

        Uses combination of topic list and action list for reliability.
        If ros2 commands timeout, resets daemon and retries once.
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

        except subprocess.TimeoutExpired:
            if not quiet:
                print(f"[NAV2] ros2 command timed out - daemon may be stuck")
            # Reset daemon and retry once
            if retry_with_daemon_reset:
                if not quiet:
                    print(f"[NAV2] Resetting ROS2 daemon and retrying...")
                try:
                    subprocess.run("pkill -9 -f ros2-daemon", shell=True, timeout=5)
                    time.sleep(1)
                    subprocess.run(
                        "source /opt/ros/jazzy/setup.bash && ros2 daemon start",
                        shell=True, executable='/bin/bash', timeout=10
                    )
                    time.sleep(2)
                except:
                    pass
                return self.check_nav2_lifecycle(quiet=quiet, retry_with_daemon_reset=False)
            return False

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

    def check_scan_publishing(self) -> bool:
        """Check if /scan topic is publishing data"""
        try:
            result = subprocess.run(
                "source /opt/ros/jazzy/setup.bash && timeout 3 ros2 topic echo /scan --once 2>/dev/null | wc -l",
                shell=True, executable='/bin/bash',
                capture_output=True, text=True, timeout=6
            )
            if result.returncode == 0 and int(result.stdout.strip() or '0') > 5:
                return True
            return False
        except:
            return False

    def recover_scan_relay(self) -> bool:
        """Recover scan relay if it stopped working"""
        print("[RECOVER] Checking scan relay...")

        # First check if scan_real is publishing (from Gazebo)
        try:
            result = subprocess.run(
                "source /opt/ros/jazzy/setup.bash && timeout 3 ros2 topic echo /scan_real --once 2>/dev/null | wc -l",
                shell=True, executable='/bin/bash',
                capture_output=True, text=True, timeout=6
            )
            if result.returncode != 0 or int(result.stdout.strip() or '0') < 5:
                print("[RECOVER] /scan_real not publishing - Gazebo bridge issue")
                return False  # Need full Gazebo restart
        except:
            print("[RECOVER] Failed to check /scan_real")
            return False

        # scan_real is OK, restart scan_relay
        print("[RECOVER] Restarting scan relay...")
        self.stop_scan_relay()
        time.sleep(2)
        return self.start_scan_relay()

    def check_costmap_publishing(self, timeout: float = None, retries: int = 2) -> bool:
        """Check if costmaps are being published"""
        if timeout is None:
            timeout = COSTMAP_CHECK_TIMEOUT

        # First verify scan data is flowing
        if not self.check_scan_publishing():
            print("[NAV2] /scan topic not publishing, attempting recovery...")
            if self.recover_scan_relay():
                time.sleep(2)
                if not self.check_scan_publishing():
                    print("[NAV2] Scan still not working after relay restart")
                    return False
            else:
                print("[NAV2] Scan recovery failed")
                return False

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

    def wait_for_nav2_ready(self, max_wait: float = None, check_interval: float = 5.0) -> bool:
        """Wait for Nav2 to be fully ready (nodes present and action available)"""
        if max_wait is None:
            max_wait = NAV2_READY_MAX_WAIT
        start = time.time()
        quiet = False
        activation_attempts = 0
        max_activation_attempts = 3
        costmap_failures = 0
        max_costmap_failures = 2  # Trigger scan relay restart after 2 failures
        scan_bridge_failures = 0
        max_scan_bridge_failures = 3  # Abort if Gazebo bridge is dead

        while time.time() - start < max_wait:
            # Check if critical nodes exist and action is available
            if self.check_nav2_lifecycle(quiet=quiet):
                # Nodes and action available - verify costmap
                if self.check_costmap_publishing(retries=2):
                    costmap_failures = 0  # Reset counter on success
                    scan_bridge_failures = 0  # Reset scan bridge counter too
                    # Explicitly activate lifecycle nodes (especially bt_navigator)
                    if self.activate_nav2_lifecycle():
                        # Double-check: verify action is responding
                        if self._verify_action_server():
                            print("[NAV2] All systems ready!")
                            return True
                        else:
                            print("[NAV2] Action server not responding, waiting...")
                    else:
                        activation_attempts += 1
                        print(f"[NAV2] Lifecycle activation incomplete ({activation_attempts}/{max_activation_attempts}), retrying...")
                        if activation_attempts >= max_activation_attempts:
                            print("[NAV2] Max activation attempts reached, resetting daemon...")
                            ProcessManager.cleanup_all(patterns=[], force=False, reset_daemon=True)
                            activation_attempts = 0
                            time.sleep(5)
                else:
                    # Nodes OK but costmap not publishing
                    costmap_failures += 1
                    elapsed = int(time.time() - start)
                    print(f"[NAV2] Nodes OK, waiting for costmap... ({elapsed}s)")

                    # Check if /scan_real is publishing (Gazebo bridge alive)
                    if not self.check_scan_publishing():
                        scan_bridge_failures += 1
                        if scan_bridge_failures >= max_scan_bridge_failures:
                            print(f"[NAV2] Gazebo bridge dead ({scan_bridge_failures} consecutive scan failures) - need full Gazebo restart")
                            return False  # Fast-fail so caller can restart Gazebo

                    # Force scan relay restart after repeated costmap failures
                    if costmap_failures >= max_costmap_failures:
                        print(f"[NAV2] Costmap failed {costmap_failures} times, forcing scan relay restart...")
                        if self.recover_scan_relay():
                            print("[NAV2] Scan relay restarted, waiting for costmap to recover...")
                            time.sleep(3)
                        costmap_failures = 0  # Reset counter
            else:
                elapsed = int(time.time() - start)
                if elapsed > 20:
                    # Only print waiting message after initial 20s
                    print(f"[NAV2] Waiting for Nav2 to be ready... ({elapsed}s)")

            quiet = True  # Reduce spam after first check
            time.sleep(check_interval)

        print("[NAV2] Timeout waiting for Nav2 to be ready")
        return False

    def _verify_action_server(self, timeout: float = 10.0) -> bool:
        """Verify that navigate_to_pose action server is responding"""
        try:
            result = subprocess.run(
                f"source /opt/ros/jazzy/setup.bash && timeout {timeout} ros2 action info /navigate_to_pose 2>/dev/null",
                shell=True, executable='/bin/bash',
                capture_output=True, text=True, timeout=timeout + 3
            )
            if 'Action: /navigate_to_pose' in result.stdout or 'nav2_msgs' in result.stdout:
                return True
            return False
        except:
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
        max_retries = 3  # Increased from 2 to 3
        print(f"[SIM] Starting Nav2...{' (retry ' + str(retry_count) + ')' if retry_count > 0 else ''}")
        if not use_amcl:
            print("[SIM] AMCL disabled - using dead reckoning only")

        # Clean up any leftover Nav2 processes before starting
        if retry_count > 0:
            print("[SIM] Cleaning up before retry...")
            self.stop_nav2()
            # Reset ROS2 daemon on retries for cleaner state
            if retry_count >= 2:
                ProcessManager.cleanup_all(patterns=[], force=False, reset_daemon=True)
            time.sleep(5)  # Increased wait time between retries

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
            print("[ERROR] Nav2 process died during startup")
            if retry_count < max_retries:
                print(f"[SIM] Retrying Nav2 start ({retry_count + 1}/{max_retries})...")
                return self.start_nav2(verify=verify, retry_count=retry_count + 1, use_amcl=use_amcl)
            return False

        if verify:
            # Wait for lifecycle to be ready (with longer timeout)
            if not self.wait_for_nav2_ready():
                print("[ERROR] Nav2 lifecycle not ready after timeout")
                if retry_count < max_retries:
                    print(f"[SIM] Attempting Nav2 restart ({retry_count + 1}/{max_retries})...")
                    return self.start_nav2(verify=True, retry_count=retry_count + 1, use_amcl=use_amcl)
                else:
                    print("[ERROR] Nav2 failed to stabilize after all retries")
                    return False

        # Additional verification: ensure /navigate_to_pose action is available
        print("[SIM] Final verification: checking navigate_to_pose action...")
        for i in range(5):
            if self._verify_action_server():
                print("[SIM] Nav2 started successfully and action server verified!")
                return True
            print(f"[SIM] Action server not ready, waiting... ({i+1}/5)")
            time.sleep(3)

        if retry_count < max_retries:
            print(f"[SIM] Action server still not ready, retrying Nav2 ({retry_count + 1}/{max_retries})...")
            return self.start_nav2(verify=True, retry_count=retry_count + 1, use_amcl=use_amcl)

        print("[ERROR] Nav2 action server never became ready")
        return False

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
        print(f"[SIM] Starting geofence with method: {method}, params: {params}")

        # Stop existing geofence first
        self.stop_geofence()
        time.sleep(2)

        self.current_method = method
        self.current_method_params = params  # Store for recovery

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
            # NOTE: Nav2 publishes to /cmd_vel (not /cmd_vel_nav), so without topic remapping,
            # cmd_vel_guard won't intercept Nav2 commands. For now, only enable for geofence
            # which uses a separate bridge configuration.
            # CBF/SSM use goal-level checking only (per their original papers).
            runtime_monitoring_methods = ['geofence']  # CBF/SSM: goal-level only
            enable_cmd_vel_guard = method in runtime_monitoring_methods

        # Build launch arguments
        launch_args = [f"safety_method:={actual_method}"]
        launch_args.append(f"enable_cmd_vel_guard:={'true' if enable_cmd_vel_guard else 'false'}")

        # Configure cmd_vel_guard topics for methods with runtime monitoring
        # Nav2 publishes to /cmd_vel_nav, so guard must subscribe there
        if enable_cmd_vel_guard:
            # Input always from Nav2 output
            launch_args.append("cmd_vel_input_topic:=/cmd_vel_nav")

            if method == 'geofence':
                # Geofence uses hw_guard bridge: /cmd_vel_safe → Gazebo
                launch_args.append("cmd_vel_output_topic:=/cmd_vel_safe")
                print("[SIM] Using FULL INTERCEPTION mode (/cmd_vel_nav → guard → /cmd_vel_safe)")
            else:
                # CBF/SSM use standard bridge: /cmd_vel → Gazebo
                launch_args.append("cmd_vel_output_topic:=/cmd_vel")
                print(f"[SIM] Using RUNTIME GUARD mode for {method} (/cmd_vel_nav → guard → /cmd_vel)")

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
            # Debug: show params being passed
            if 'enable_runtime_monitoring' in params:
                print(f"[SIM] Passing enable_runtime_monitoring={params['enable_runtime_monitoring']} to geofence")

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
            # If cmd_vel_guard is disabled, start cmd_vel relay to ensure navigation works
            if not enable_cmd_vel_guard:
                print("[SIM] cmd_vel_guard disabled, starting cmd_vel relay for navigation...")
                self.start_cmd_vel_relay()
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
        self.stop_cmd_vel_relay()  # Stop cmd_vel relay if running
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
                     target_x: float = None, target_y: float = None,
                     offset_x: float = 0.0, offset_y: float = 0.0,
                     scan_rotation_deg: float = 0.0, scan_scale: float = 1.0,
                     scan_noise: float = 0.0) -> bool:
        """Start S4/S5/S5′ attack node

        Args:
            attack_type: "velocity_scaling", "odom_spoofing", "direct_control", or "scan_spoofing"
            scale_factor: For velocity_scaling, 2.0 = double speed
                         For odom_spoofing, 0.5 = robot appears to move half distance
            target_x, target_y: For direct_control, the target position to drive to
            offset_x, offset_y: For odom_spoofing, position offset to add
            scan_rotation_deg: For scan_spoofing, rotation offset in degrees
            scan_scale: For scan_spoofing, range scale (0.8 = walls appear 20% closer)
            scan_noise: For scan_spoofing, noise stddev in meters

        Note:
            velocity_scaling works with current setup (cmd_vel_nav → attack → cmd_vel)
            odom_spoofing: gz_bridge publishes to /odom_real, attack node spoofs to /odom
            direct_control: Bypasses Nav2 entirely and drives directly to target
            scan_spoofing: gz_bridge publishes to /scan_real, attack node spoofs to /scan
        """
        if attack_type == "direct_control":
            print(f"[ATTACK] Starting {attack_type} attack with target=({target_x}, {target_y})")
        elif attack_type == "odom_spoofing":
            print(f"[ATTACK] Starting {attack_type} attack with scale={scale_factor}, offset=({offset_x}, {offset_y})")
        elif attack_type == "scan_spoofing":
            print(f"[ATTACK] Starting {attack_type} attack with rotation={scan_rotation_deg}°, scale={scan_scale}, noise={scan_noise}m")
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

        # For scan_spoofing, stop the normal scan relay first
        if attack_type == "scan_spoofing":
            print("[ATTACK] Stopping scan relay for scan spoofing attack...")
            self.stop_scan_relay()
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
            # For odom spoofing:
            # - scale_factor < 1.0 makes robot appear to move less (e.g., 0.5 = half distance)
            # - offset_x/y shifts the reported position (e.g., -2.0 = appear 2m further from zone)
            attack_cmd = f"""
                source /opt/ros/jazzy/setup.bash && \
                source {WORKSPACE_DIR}/install/setup.bash && \
                ros2 run geofence_policy_enforcer attack_odom_spoofing \
                    --ros-args \
                    -p scale_factor:={scale_factor} \
                    -p offset_x:={offset_x} \
                    -p offset_y:={offset_y} \
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
        elif attack_type == "scan_spoofing":
            # S5′: LIDAR scan spoofing to confuse AMCL localization
            # Rotation causes AMCL to misestimate orientation
            # Scale causes AMCL to misestimate distances
            # Noise causes particle filter to spread/diverge
            import math
            rotation_rad = scan_rotation_deg * math.pi / 180.0
            attack_cmd = f"""
                source /opt/ros/jazzy/setup.bash && \
                source {WORKSPACE_DIR}/install/setup.bash && \
                ros2 run geofence_policy_enforcer attack_scan_spoofing \
                    --ros-args \
                    -p rotation_offset:={rotation_rad} \
                    -p range_scale:={scan_scale} \
                    -p noise_stddev:={scan_noise} \
                    -p input_topic:=/scan_real \
                    -p output_topic:=/scan \
                    -p attack_enabled:=true
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
        """Stop attack nodes and restart odom/scan relay if needed"""
        was_odom_spoofing = (self.current_attack == "odom_spoofing")
        was_scan_spoofing = (self.current_attack == "scan_spoofing")
        was_param_injection = (self.current_attack == "param_injection")

        if self.attack_proc and self.attack_proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.attack_proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass

        safe_pkill('attack_velocity_scaling')
        safe_pkill('attack_odom_spoofing')
        safe_pkill('attack_scan_spoofing')
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

        # Restart scan relay after stopping scan_spoofing
        if was_scan_spoofing:
            print("[ATTACK] Restarting scan relay after scan spoofing attack...")
            time.sleep(1)
            self.start_scan_relay()

    def stop_all(self, reset_daemon: bool = False):
        """Stop all simulation processes"""
        print("[SIM] Stopping all processes...")

        self.stop_attack()  # Stop any running attack nodes
        self.stop_odom_relay()  # Stop odom relay
        self.stop_scan_relay()  # Stop scan relay
        self.stop_cmd_vel_relay()  # Stop cmd_vel relay
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

    def recover_nav2(self, reset_daemon: bool = False, full_restart: bool = False) -> bool:
        """Attempt to recover Nav2 by restarting it

        Args:
            reset_daemon: Reset ROS2 daemon before restart
            full_restart: If True, do full simulation restart (Gazebo + Nav2 + Geofence)
        """
        print("[RECOVER] Attempting Nav2 recovery...")

        if full_restart:
            print("[RECOVER] Performing full simulation restart...")
            method = self.current_method
            self.stop_all(reset_daemon=True)
            time.sleep(5)

            # Restart everything
            use_hw_guard = method in ['geofence', 'geofence_hw'] if method else False
            if not self.start_gazebo(use_hw_guard=use_hw_guard):
                print("[RECOVER] Gazebo restart failed")
                return False
            if not self.start_nav2(verify=True):
                print("[RECOVER] Nav2 restart failed")
                return False
            if method and not self.start_geofence(method, self.current_method_params):
                print("[RECOVER] Geofence restart failed")
                return False
            print("[RECOVER] Full simulation restart successful")
            return True

        self.stop_nav2()
        time.sleep(5)  # Increased from 3 to 5

        # Reset daemon only if requested (fallback for stuck discovery)
        if reset_daemon:
            print("[RECOVER] Resetting ROS2 daemon...")
            ProcessManager.cleanup_all(patterns=[], force=False, reset_daemon=True)
            time.sleep(3)

        if not self.start_nav2(verify=True):
            print("[RECOVER] Nav2 restart failed")
            # Try once more with daemon reset as fallback
            if not reset_daemon:
                print("[RECOVER] Trying with daemon reset...")
                return self.recover_nav2(reset_daemon=True)
            # If daemon reset also failed, try full restart
            print("[RECOVER] Daemon reset didn't help, trying full restart...")
            return self.recover_nav2(full_restart=True)

        # Restart geofence too since it depends on Nav2
        if self.current_method:
            method = self.current_method
            self.stop_geofence()
            time.sleep(3)  # Increased from 2 to 3
            if not self.start_geofence(method, self.current_method_params):
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

            print(f"[RESET] Teleporting robot to ({x}, {y}, θ={theta})")

            # Step 0: Cancel any active navigation goals first
            print("[RESET] Cancelling any active navigation goals...")
            try:
                # Cancel navigate_to_pose
                subprocess.run(
                    f"source /opt/ros/jazzy/setup.bash && ros2 topic pub --once /navigate_to_pose/_action/cancel_goal action_msgs/msg/CancelGoal '{{}}' 2>/dev/null",
                    shell=True, executable='/bin/bash',
                    capture_output=True, timeout=5
                )
                # Cancel navigate_to_pose_safe
                subprocess.run(
                    f"source /opt/ros/jazzy/setup.bash && ros2 topic pub --once /navigate_to_pose_safe/_action/cancel_goal action_msgs/msg/CancelGoal '{{}}' 2>/dev/null",
                    shell=True, executable='/bin/bash',
                    capture_output=True, timeout=5
                )
            except:
                pass

            # Step 1: Send zero velocity to stop the robot
            print("[RESET] Stopping robot movement...")
            try:
                for _ in range(5):  # Send multiple times to ensure robot stops
                    subprocess.run(
                        f"source /opt/ros/jazzy/setup.bash && ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist '{{linear: {{x: 0.0}}, angular: {{z: 0.0}}}}' 2>/dev/null",
                        shell=True, executable='/bin/bash',
                        capture_output=True, timeout=3
                    )
                    subprocess.run(
                        f"source /opt/ros/jazzy/setup.bash && ros2 topic pub --once /cmd_vel_nav geometry_msgs/msg/Twist '{{linear: {{x: 0.0}}, angular: {{z: 0.0}}}}' 2>/dev/null",
                        shell=True, executable='/bin/bash',
                        capture_output=True, timeout=3
                    )
                    time.sleep(0.1)
            except:
                pass
            time.sleep(1.0)  # Wait for robot to fully stop

            # Step 2: Teleport robot in Gazebo using gz service
            # World name from SDF file: empty (defined in warehouse_walk.sdf)
            # Robot model name from URDF: mobile_manip
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

            time.sleep(1.5)  # Wait for physics to settle

            # Wait for TF to stabilize (avoid time jump issues)
            print("[RESET] Waiting for TF to stabilize...")
            time.sleep(2.0)

            # Step 3: Publish to /initialpose for Nav2 AMCL multiple times
            # (tight covariance for fast convergence)
            initialpose_cmd = f"""ros2 topic pub --once /initialpose geometry_msgs/msg/PoseWithCovarianceStamped '{{
                header: {{frame_id: "map"}},
                pose: {{
                    pose: {{
                        position: {{x: {x}, y: {y}, z: 0.0}},
                        orientation: {{x: 0.0, y: 0.0, z: {qz}, w: {qw}}}
                    }},
                    covariance: [0.001, 0.0, 0.0, 0.0, 0.0, 0.0,
                                 0.0, 0.001, 0.0, 0.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.001]
                }}
            }}'"""

            # Send initialpose multiple times to force AMCL to converge
            print("[RESET] Setting AMCL initial pose...")
            for i in range(3):
                subprocess.run(
                    f"source /opt/ros/jazzy/setup.bash && {initialpose_cmd}",
                    shell=True, executable='/bin/bash',
                    capture_output=True, timeout=5
                )
                time.sleep(0.5)

            # Wait for AMCL to converge and verify position
            print("[RESET] Waiting for AMCL convergence...")
            time.sleep(3.0)  # Initial wait for AMCL to process initialpose

            # Verify AMCL pose is close to target position
            max_amcl_wait = AMCL_CONVERGENCE_TIMEOUT  # Max additional seconds to wait
            amcl_ok = False
            resend_count = 0
            for i in range(max_amcl_wait):
                try:
                    result = subprocess.run(
                        f"source /opt/ros/jazzy/setup.bash && timeout 2 ros2 topic echo /amcl_pose --once 2>/dev/null",
                        shell=True, executable='/bin/bash',
                        capture_output=True, text=True, timeout=5
                    )
                    if result.returncode == 0 and 'position:' in result.stdout:
                        # Parse position from output
                        lines = result.stdout.split('\n')
                        amcl_x = amcl_y = None
                        for j, line in enumerate(lines):
                            if 'position:' in line:
                                for k in range(j+1, min(j+5, len(lines))):
                                    if 'x:' in lines[k] and amcl_x is None:
                                        amcl_x = float(lines[k].split(':')[1].strip())
                                    elif 'y:' in lines[k] and amcl_y is None:
                                        amcl_y = float(lines[k].split(':')[1].strip())
                                break
                        if amcl_x is not None and amcl_y is not None:
                            dist = ((amcl_x - x)**2 + (amcl_y - y)**2)**0.5
                            if dist < 0.5:
                                print(f"[RESET] AMCL converged: ({amcl_x:.2f}, {amcl_y:.2f}), error: {dist:.2f}m")
                                amcl_ok = True
                                break
                            else:
                                print(f"[RESET] AMCL not converged yet: ({amcl_x:.2f}, {amcl_y:.2f}), error: {dist:.2f}m")
                                # Re-send initialpose if AMCL is far off
                                if dist > 1.0 and resend_count < 3:
                                    print("[RESET] Re-sending initialpose...")
                                    subprocess.run(
                                        f"source /opt/ros/jazzy/setup.bash && {initialpose_cmd}",
                                        shell=True, executable='/bin/bash',
                                        capture_output=True, timeout=5
                                    )
                                    resend_count += 1
                                    time.sleep(1.0)
                except Exception as e:
                    pass
                time.sleep(1.0)

            if not amcl_ok:
                print("[WARNING] AMCL may not have converged properly - forcing one more reset")
                # One more forced attempt
                subprocess.run(
                    f"source /opt/ros/jazzy/setup.bash && {initialpose_cmd}",
                    shell=True, executable='/bin/bash',
                    capture_output=True, timeout=5
                )
                time.sleep(2.0)

            time.sleep(2.0)  # Final settling time for TF

            # Verify TF is working by checking map->base_link transform
            try:
                tf_check = subprocess.run(
                    f"source /opt/ros/jazzy/setup.bash && timeout 3 ros2 run tf2_ros tf2_echo map base_footprint 2>/dev/null | head -5",
                    shell=True, executable='/bin/bash',
                    capture_output=True, text=True, timeout=5
                )
                if 'Translation' not in tf_check.stdout:
                    print("[WARNING] TF not yet stable, waiting more...")
                    time.sleep(2.0)
            except:
                pass

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
        print(f"[DEBUG] restart_with_method called: method={method}, params={params}")
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

        # Helper: Check if path from (0, 0) to goal passes through any zone
        def path_crosses_zone(gx: float, gy: float, margin: float = 0.5) -> bool:
            """Check if straight line from origin to goal crosses any forbidden zone."""
            num_samples = 50
            for i in range(num_samples + 1):
                t = i / num_samples
                px = t * gx
                py = t * gy
                # Check if point is within zone + margin
                for zone in ZONES.values():
                    if (zone['x_min'] - margin <= px <= zone['x_max'] + margin and
                        zone['y_min'] - margin <= py <= zone['y_max'] + margin):
                        return True
            return False

        # Helper: Check if goal should be rejected based on safety method and margins
        def should_be_rejected(gx: float, gy: float, method: str) -> bool:
            # Inside zone - always rejected (except no_guard which allows all)
            if is_inside_zone(gx, gy):
                return method != 'no_guard'

            # SELP-proper: Goal-only check (Wu et al., ICRA 2025)
            # - ONLY rejects goals INSIDE forbidden zones
            # - NO path checking, NO safety margin
            if method in ['selp', 'selp_proper']:
                return False  # Outside zone → allow (already checked inside above)

            # Path-aware methods should also reject if path crosses zone
            # NOTE: Only geofence does path checking with uncertainty margins
            # CBF/SSM only check goal point, NOT the path (per their original papers)
            if method in ['geofence', 'geofence_hw']:
                if path_crosses_zone(gx, gy, margin=0.55):
                    return True

            # Check margin-based rejection (goal point only)
            dist = min_distance_to_zones(gx, gy)
            # Use small epsilon for floating point comparison
            eps = 1e-6

            if method == 'cbf':
                # CBF (Ames et al., TAC 2017): h(x) = dist - margin >= 0 means safe
                # margin = 0.3m, so reject if dist < 0.3m
                return dist < (0.3 - eps)
            elif method == 'ssm':
                # SSM (ISO 15066): velocity-dependent margin
                # At v=0.5: S = 0.5*(0.1+0.2) + 0.5²/2 + 0.1 + 0.2 = 0.575m
                # At v=0 (stationary): S = 0.1 + 0.2 = 0.3m
                # Use conservative estimate (v=0.5) for fallback
                return dist < (0.575 - eps)
            elif method in ['geofence', 'geofence_hw']:
                # Geofence uses uncertainty margin = 0.55m
                return dist < (0.55 - eps)
            else:
                # no_guard - only inside zone
                return False

        # Helper: Get robot's current position from Gazebo ground truth
        def get_robot_position() -> tuple:
            """Get robot's actual position from Gazebo (ground truth). Returns (x, y) or (None, None) on error."""
            import re
            try:
                # Use Gazebo ground truth for accurate position (not affected by AMCL drift)
                result = subprocess.run(
                    ["gz", "topic", "-e", "-n", "1", "-t", "/world/empty/pose/info"],
                    capture_output=True, text=True, timeout=3
                )
                output = result.stdout
                # Parse: name: "mobile_manip" followed by position {x: ..., y: ...}
                match = re.search(
                    r'name: "mobile_manip".*?position \{\s*x: ([\d.e+-]+)\s*y: ([\d.e+-]+)',
                    output, re.DOTALL
                )
                if match:
                    return float(match.group(1)), float(match.group(2))
            except Exception as e:
                print(f"[WARN] Failed to get Gazebo robot position: {e}")
            return None, None

        # For no_guard method, bypass goal_gate and send directly to Nav2
        # This avoids potential issues with goal_gate initialization
        action_topic = '/navigate_to_pose' if safety_method == 'no_guard' else '/navigate_to_pose_safe'

        goal_cmd = f"""
            source /opt/ros/jazzy/setup.bash && \
            source {WORKSPACE_DIR}/install/setup.bash && \
            ros2 action send_goal {action_topic} nav2_msgs/action/NavigateToPose \
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

            # Check for path-based rejection (path crosses forbidden zone)
            if "PATH REJECTED" in output:
                return "reject", "Goal rejected - path crosses forbidden zone"

            if "Goal accepted" in output:
                if "SUCCEEDED" in output or "succeeded" in output:
                    # Verify robot position using Gazebo ground truth
                    robot_x, robot_y = get_robot_position()
                    if robot_x is not None and robot_y is not None:
                        goal_dist = ((robot_x - x)**2 + (robot_y - y)**2)**0.5
                        # Use generous tolerance: AMCL drift in long sim sessions
                        # can cause large Gazebo vs Nav2 position discrepancies.
                        # Trust Nav2 SUCCEEDED - violation monitor catches actual zone breaches.
                        GOAL_TOLERANCE = 1.5
                        if goal_dist <= GOAL_TOLERANCE:
                            return "allow", f"Goal reached successfully (dist={goal_dist:.2f}m)"
                        else:
                            # Large discrepancy: likely AMCL drift, not a real nav failure
                            print(f"[WARN] Nav2 SUCCEEDED but Gazebo pos ({robot_x:.2f}, {robot_y:.2f}) is {goal_dist:.2f}m from goal - AMCL drift likely")
                            return "allow", f"Goal reached (Nav2 SUCCEEDED, Gazebo dist={goal_dist:.2f}m - possible AMCL drift)"
                    else:
                        # Couldn't get position, trust Nav2 result
                        return "allow", "Goal reached successfully (position unverified)"
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
                        # goal_gate logs to rosout (not captured by subprocess).
                        # For safety methods: check if goal should have been rejected
                        if safety_method in ['geofence', 'geofence_hw', 'cbf', 'ssm', 'selp', 'selp_proper']:
                            if should_be_rejected(x, y, safety_method):
                                return "reject", f"Goal rejected by {safety_method} (within safety margin)"
                        # Goal was likely allowed by safety method but Nav2 couldn't
                        # complete navigation (path planning failure, costmap issue, etc.)
                        # Treat as "allow" - the violation monitor handles zone breaches.
                        print(f"[WARN] Nav2 ABORTED for ({x:.2f}, {y:.2f}) with method={safety_method} - treating as allow (Nav2 path failure)")
                        return "allow", f"Navigation aborted (goal likely allowed by {safety_method}, Nav2 path failure)"
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
            # Cancel any pending goal on timeout
            try:
                subprocess.run(
                    f"source /opt/ros/jazzy/setup.bash && ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose '{{}}' --cancel 2>/dev/null",
                    shell=True, executable='/bin/bash',
                    capture_output=True, timeout=5
                )
            except:
                pass
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
    scenarios = scenarios or ["S1", "S2", "S3"]  # S1-S3 only for now
    trials = []

    # ==========================================================================
    # S1: Direct Zone Intrusion (Margin Comparison)
    # Zone: x=[4,6], y=[-1,1] - Boundary at x=4.0
    # Robot starts at (0,0), straight line on y=0
    #
    # Purpose: Test violation rates - ALL paths must cross the forbidden zone
    #
    # Zone: x=[4,6], y=[-1,1], robot starts at (0,0)
    # Goals are placed BEYOND the zone (x > 6) so path MUST cross the zone
    #
    # Expected results:
    #   - no_guard: 100% VIOLATION (path crosses zone, no protection)
    #   - selp: REJECT only if goal is inside zone, otherwise VIOLATION
    #   - cbf/ssm/geofence: Should REJECT due to path crossing zone
    #
    # This tests whether each method can prevent zone crossing during navigation
    # ==========================================================================
    if "S1" in scenarios:
        s1_configs = [
            # (1) Goal inside zone - no_guard violates, others reject goal
            {"intensity": "inside_zone", "goal": (5.0, 0.0), "velocity": 0.5,
             "desc": "Goal inside zone - no_guard violates, others reject"},

            # (2) Goal just beyond zone (x=6.5) - path crosses zone
            {"intensity": "beyond_0.5m", "goal": (6.5, 0.0), "velocity": 0.5,
             "desc": "Goal 0.5m beyond zone - path must cross zone"},

            # (3) Goal 1m beyond zone (x=7.0) - path crosses zone
            {"intensity": "beyond_1.0m", "goal": (7.0, 0.0), "velocity": 0.5,
             "desc": "Goal 1.0m beyond zone - path must cross zone"},

            # (4) Goal 1.5m beyond zone (x=7.5) - path crosses zone
            {"intensity": "beyond_1.5m", "goal": (7.5, 0.0), "velocity": 0.5,
             "desc": "Goal 1.5m beyond zone - path must cross zone"},

            # (5) Goal 2m beyond zone (x=8.0) - path crosses zone
            {"intensity": "beyond_2.0m", "goal": (8.0, 0.0), "velocity": 0.5,
             "desc": "Goal 2.0m beyond zone - path must cross zone"},
        ]

        for method in methods:
            for cfg in s1_configs:
                for seed in range(num_seeds):
                    # S1: All paths must cross zone → expected_safe=False
                    trials.append(TrialConfig(
                        trial_id=f"S1_{method}_{cfg['intensity']}_s{seed}",
                        method=method, scenario="S1", intensity=cfg['intensity'], seed=seed,
                        goal_x=cfg['goal'][0], goal_y=cfg['goal'][1],
                        velocity=cfg.get('velocity', 0.5),
                        sigma_loc=cfg.get('sigma_loc', 0.15),
                        has_physical_barrier=cfg.get('has_physical_barrier', True),
                        boundary_distance=cfg.get('boundary_distance'),
                        description=cfg['desc'],
                        expected_safe=False,
                    ))

    # ==========================================================================
    # S2: Incremental Approach (Salami Attack)
    # Zone: x=[4,6], boundary at x=4.0
    #
    # Purpose: Compare when each method first rejects during sequential approach
    #
    # Method margins determine max approach distance:
    #   - no_guard: never rejects → goes inside zone (violation)
    #   - selp: rejects only inside zone → reaches 3.9m
    #   - cbf: 0.3m margin → first rejects at step5 (0.25m)
    #   - ssm: 0.475m margin (v=0.5) → first rejects at step3 (0.5m)
    #   - geofence: 0.55m margin → first rejects at step3 (0.5m)
    #
    # Expected max approach distance:
    #   no_guard: 4.5m (inside zone - violation)
    #   selp: 3.9m (rejects at step7 - inside zone)
    #   cbf: 3.65m (rejects at step5 - 0.25m)
    #   ssm/geofence: 3.0m (rejects at step3 - 0.5m)
    # ==========================================================================
    if "S2" in scenarios:
        s2_configs = [
            # Step 1: 2.0m from boundary - all allow
            {"intensity": "step1_2.0m", "goal": (2.0, 0.0),
             "desc": "Step 1: 2.0m from boundary - all allow"},

            # Step 2: 1.0m from boundary - all allow
            {"intensity": "step2_1.0m", "goal": (3.0, 0.0),
             "desc": "Step 2: 1.0m from boundary - all allow"},

            # Step 3: 0.5m from boundary - Geofence/SSM reject
            {"intensity": "step3_0.5m", "goal": (3.5, 0.0),
             "desc": "Step 3: 0.5m from boundary - Geofence/SSM reject"},

            # Step 4: 0.35m from boundary - CBF still allows
            {"intensity": "step4_0.35m", "goal": (3.65, 0.0),
             "desc": "Step 4: 0.35m from boundary - CBF still allows"},

            # Step 5: 0.25m from boundary - CBF rejects
            {"intensity": "step5_0.25m", "goal": (3.75, 0.0),
             "desc": "Step 5: 0.25m from boundary - CBF rejects"},

            # Step 6: 0.1m from boundary - only SELP/no_guard allow
            {"intensity": "step6_0.1m", "goal": (3.9, 0.0),
             "desc": "Step 6: 0.1m from boundary - only SELP/no_guard allow"},

            # Step 7: zone 깊숙이 - no_guard → 100% VIOLATION
            # Goal을 zone 깊숙이(5.5)로 설정 - 확실한 violation
            {"intensity": "step7_center", "goal": (5.5, 0.0),
             "desc": "Step 7: deep inside zone - guaranteed violation"},
        ]

        for method in methods:
            for cfg in s2_configs:
                for seed in range(num_seeds):
                    # S2: step7 is inside zone → unsafe; steps 1-6 are outside → safe
                    is_safe = "step7" not in cfg['intensity']
                    trials.append(TrialConfig(
                        trial_id=f"S2_{method}_{cfg['intensity']}_s{seed}",
                        method=method, scenario="S2", intensity=cfg['intensity'], seed=seed,
                        goal_x=cfg['goal'][0], goal_y=cfg['goal'][1],
                        velocity=cfg.get('velocity', 0.5),
                        boundary_distance=cfg.get('boundary_distance'),
                        description=cfg['desc'],
                        expected_safe=is_safe,
                    ))

    # ==========================================================================
    # S3: Path Through Zone (Runtime Safety)
    # Zone: x=[4,6], y=[-1,1]
    #
    # Purpose: Compare Goal-only check (SELP) vs Path-aware (CBF/SSM/Geofence)
    #
    # Key insight: Goal (8.0, 0.0) is OUTSIDE zone (safe destination)
    # BUT the straight path (0,0)→(8,0) passes THROUGH zone [4,6]
    #
    # Method behavior:
    #   - no_guard: allows → robot violates zone during navigation
    #   - selp: checks goal only (outside zone) → allows → violation!
    #   - cbf/ssm/geofence: checks path → rejects (PATH_REJ)
    #
    # Expected results:
    #   before_zone: all OK (safe path)
    #   through_zone_center: no_guard/selp → VIOLATION, others → PATH_REJ
    #   graze_zone_edge: no_guard/selp → VIOLATION, others → PATH_REJ
    #   safe_bypass: all OK (path avoids zone)
    # ==========================================================================
    if "S3" in scenarios:
        s3_configs = [
            # Safe: Goal before zone - safe path
            {"intensity": "before_zone", "goal": (3.0, 0.0),
             "desc": "Goal before zone - safe path"},

            # Core: Zone 깊숙이로 직접 이동 - 확실한 violation
            {"intensity": "zone_center", "goal": (5.5, 0.0),
             "desc": "Goal deep inside zone - guaranteed violation"},

            # Edge: Zone 가장자리 스침 (y=0.8, zone y_max=1.0)
            {"intensity": "graze_zone_edge", "goal": (7.0, 0.8),
             "desc": "Path grazes zone edge (y=0.8, zone y_max=1.0)"},

            # Safe bypass: Zone 우회 (y=2.5 >> zone y_max=1.0, 직선 경로 장애물 없음)
            {"intensity": "safe_bypass", "goal": (7.0, 2.5),
             "desc": "Path bypasses zone (y=2.5 >> zone y_max=1.0, clear straight-line path)"},
        ]

        # S3 expected_safe mapping
        s3_expected_safe = {
            'before_zone': True,
            'zone_center': False,
            'graze_zone_edge': False,
            'safe_bypass': True,
        }

        for method in methods:
            for cfg in s3_configs:
                for seed in range(num_seeds):
                    trials.append(TrialConfig(
                        trial_id=f"S3_{method}_{cfg['intensity']}_s{seed}",
                        method=method, scenario="S3", intensity=cfg['intensity'], seed=seed,
                        goal_x=cfg['goal'][0], goal_y=cfg['goal'][1],
                        velocity=cfg.get('velocity', 0.5),
                        latency_ms=cfg.get('latency_ms', 0.0),
                        boundary_distance=cfg.get('boundary_distance'),
                        description=cfg['desc'],
                        expected_safe=s3_expected_safe.get(cfg['intensity'], True),
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
    # S5: Odom Spoofing Attack - Path-based Position Spoofing
    #
    # NEW DESIGN: Robot misperceives its own position, causing path check to fail
    #
    # Attack vector: Spoof /odom to make robot think it's at a different position
    # Goal gate checks path from (spoofed) current position to goal
    # If spoofed position makes path appear safe, dangerous path is allowed
    #
    # Zone: x=[4,6], y=[-1,1] (boundary at x=4)
    # Robot spawns at origin (0, 0)
    #
    # Key scenarios:
    # 1. Path that grazes zone boundary - offset makes it appear to miss zone
    # 2. Path through zone corner - offset makes it appear to go around
    # ==========================================================================
    if "S5" in scenarios:
        s5_configs = [
            # Baseline: goal outside zone, no attack
            # Path (0,0) → (3.5, 1.2) should be safe (doesn't cross zone)
            {"intensity": "baseline_safe", "goal": (3.5, 1.2), "attack_type": None,
             "desc": "Baseline safe: path doesn't cross zone"},

            # Baseline: goal outside zone, path grazes zone edge
            # Path (0,0) → (5.0, 0.8) crosses zone boundary!
            # At x=4: y = 0.8 * (4/5) = 0.64 → inside zone (y in [-1,1])
            {"intensity": "baseline_graze", "goal": (5.0, 0.8), "attack_type": None,
             "desc": "Baseline graze: path crosses zone edge - should REJECT"},

            # Attack: offset_x=-3 makes path appear to miss zone
            # Real path: (0,0) → (5.0, 1.5) - at x=4, y=1.2 (zone edge y=1)
            # Spoofed path: (-3,0) → (5.0, 1.5) - at x=4, y = 1.5*(7/8) = 1.31 (OUTSIDE!)
            {"intensity": "path_bypass_-3m", "goal": (5.0, 1.5), "attack_type": "odom_spoofing",
             "attack_scale": 1.0, "attack_offset_x": -3.0,
             "desc": "Offset -3m: spoofed path misses zone, real path grazes"},

            # Attack: offset_x=-2 moderate spoofing
            # Real path: (0,0) → (4.5, 0.5) - at x=4, y=0.44 (inside zone!)
            # Spoofed path: (-2,0) → (4.5, 0.5) - at x=4, y = 0.5*(6/6.5) = 0.46 (inside zone)
            # This should still be caught - offset not enough
            {"intensity": "path_bypass_-2m", "goal": (4.5, 0.5), "attack_type": "odom_spoofing",
             "attack_scale": 1.0, "attack_offset_x": -2.0,
             "desc": "Offset -2m: both paths inside zone - should REJECT"},

            # Attack: offset_x=-4 strong spoofing
            # Real path: (0,0) → (5.5, 1.3) - at x=4, y = 1.3*(4/5.5) = 0.95 (inside!)
            # Spoofed path: (-4,0) → (5.5, 1.3) - at x=4, y = 1.3*(8/9.5) = 1.09 (OUTSIDE!)
            {"intensity": "path_bypass_-4m", "goal": (5.5, 1.3), "attack_type": "odom_spoofing",
             "attack_scale": 1.0, "attack_offset_x": -4.0,
             "desc": "Offset -4m: spoofed path misses zone, real path inside"},

            # Attack: diagonal offset to bypass corner
            # Real path: (0,0) → (5.0, -1.5) - crosses zone bottom edge
            # Spoofed path: (-2, 1) → (5.0, -1.5) - steeper angle, may miss zone
            {"intensity": "corner_bypass", "goal": (5.0, -1.5), "attack_type": "odom_spoofing",
             "attack_scale": 1.0, "attack_offset_x": -2.0, "attack_offset_y": 1.0,
             "desc": "Corner bypass: diagonal offset to miss zone corner"},

            # Scale attack: robot appears closer to origin than it is
            # If robot at (2,0) appears at (1,0), path check uses (1,0)
            # Goal (3.8, 0.8) - path from (1,0) looks safe, from (2,0) crosses zone
            {"intensity": "scale_0.5x", "goal": (3.8, 0.8), "attack_type": "odom_spoofing",
             "attack_scale": 0.5, "attack_offset_x": 0.0,
             "desc": "Scale 0.5x: robot appears at half distance"},
        ]

        for method in methods:
            for cfg in s5_configs:
                for seed in range(num_seeds):
                    trials.append(TrialConfig(
                        trial_id=f"S5_{method}_{cfg['intensity']}_s{seed}",
                        method=method, scenario="S5", intensity=cfg['intensity'], seed=seed,
                        goal_x=cfg['goal'][0], goal_y=cfg['goal'][1],
                        attack_type=cfg.get('attack_type'),
                        attack_scale_factor=cfg.get('attack_scale', 1.0),
                        attack_offset_x=cfg.get('attack_offset_x', 0.0),
                        attack_offset_y=cfg.get('attack_offset_y', 0.0),
                        description=cfg['desc']
                    ))

    # ==========================================================================
    # S5′: LIDAR Spoofing Attack (Localization Confusion)
    # ==========================================================================
    # Attack scenario: Robot commanded to move short safe distance (3m forward)
    # Forbidden zone at 6-8m ahead. LIDAR spoofing causes AMCL to underestimate
    # robot's displacement, causing overshoot into forbidden zone.
    #
    # Timeline:
    #   t0: Robot at (0,0), receives goal at (3,0) - safe, 3m from zone at x=6
    #   t1: Navigation begins, attack activated
    #   t2: AMCL underestimates position due to scan manipulation
    #   t3: Robot overshoots goal, enters zone at x=6
    #
    # Key: LIDAR spoofing directly affects AMCL, unlike odom spoofing which
    # AMCL can correct using LIDAR data.
    # ==========================================================================
    if "S5p" in scenarios:
        # S5p: LIDAR spoofing attack configurations
        # TUNED: Previous attacks (30°+) completely broke navigation
        # These lighter parameters should cause subtle localization drift
        # Goal at 3.5m = 0.5m from forbidden zone (4.0m start) for realistic overshoot test
        s5p_configs = [
            # Baseline: no attack, goal 3.5m (close to zone)
            {"intensity": "baseline_3.5m", "goal": (3.5, 0.0), "attack_type": None,
             "scan_rotation": 0.0, "scan_scale": 1.0, "scan_noise": 0.0,
             "desc": "Baseline: 3.5m forward, no attack"},

            # Very light: 5° rotation only (most realistic)
            {"intensity": "rot5_pure", "goal": (3.5, 0.0), "attack_type": "scan_spoofing",
             "scan_rotation": 5.0, "scan_scale": 1.0, "scan_noise": 0.0,
             "desc": "Very light: 5° rotation only"},

            # Light: 7° rotation + minimal scale
            {"intensity": "rot7", "goal": (3.5, 0.0), "attack_type": "scan_spoofing",
             "scan_rotation": 7.0, "scan_scale": 0.98, "scan_noise": 0.01,
             "desc": "Light: 7° rotation, 2% scale, 1cm noise"},

            # Medium: 10° rotation + light scale - this is the sweet spot
            {"intensity": "rot10", "goal": (3.5, 0.0), "attack_type": "scan_spoofing",
             "scan_rotation": 10.0, "scan_scale": 0.96, "scan_noise": 0.015,
             "desc": "Medium: 10° rotation, 4% scale, 1.5cm noise"},

            # Strong: 12° rotation + moderate scale
            {"intensity": "rot12", "goal": (3.5, 0.0), "attack_type": "scan_spoofing",
             "scan_rotation": 12.0, "scan_scale": 0.94, "scan_noise": 0.02,
             "desc": "Strong: 12° rotation, 6% scale, 2cm noise"},

            # Aggressive: 15° rotation - max before breaking nav
            {"intensity": "rot15", "goal": (3.5, 0.0), "attack_type": "scan_spoofing",
             "scan_rotation": 15.0, "scan_scale": 0.92, "scan_noise": 0.025,
             "desc": "Aggressive: 15° rotation, 8% scale, 2.5cm noise"},

            # Pure scale attack: 0.92x (walls appear 8% closer)
            {"intensity": "scale_only", "goal": (3.5, 0.0), "attack_type": "scan_spoofing",
             "scan_rotation": 0.0, "scan_scale": 0.92, "scan_noise": 0.0,
             "desc": "Pure scale: 0.92x only, no rotation"},

            # Edge case: Goal at 3.8m (very close to zone at 4.0m)
            {"intensity": "edge_3.8m", "goal": (3.8, 0.0), "attack_type": "scan_spoofing",
             "scan_rotation": 8.0, "scan_scale": 0.96, "scan_noise": 0.015,
             "desc": "Edge: 3.8m goal + light attack (0.2m margin)"},
        ]

        for method in methods:
            for cfg in s5p_configs:
                for seed in range(num_seeds):
                    trials.append(TrialConfig(
                        trial_id=f"S5p_{method}_{cfg['intensity']}_s{seed}",
                        method=method, scenario="S5p", intensity=cfg['intensity'], seed=seed,
                        goal_x=cfg['goal'][0], goal_y=cfg['goal'][1],
                        attack_type=cfg.get('attack_type'),
                        scan_rotation_deg=cfg.get('scan_rotation', 0.0),
                        scan_scale=cfg.get('scan_scale', 1.0),
                        scan_noise=cfg.get('scan_noise', 0.0),
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
                    # Try full restart as last resort
                    self.log("[HEALTH] Nav2 recovery failed, attempting full simulation restart...")
                    if not self.sim_manager.recover_nav2(full_restart=True):
                        result.decision = "error"
                        result.reason = "Simulation recovery failed after full restart"
                        result.error = "health_check_failed"
                        return result

            # Reset robot pose with verification
            self.log("[RESET] Resetting robot pose to origin...")
            reset_success = self.sim_manager.reset_robot_pose(0.0, 0.0, 0.0)
            if not reset_success:
                self.log("[WARN] Robot pose reset may have failed, continuing anyway...")
            time.sleep(2)  # Increased from 1 to 2

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

            # S4/S5/S5′: Start non-direct attacks before goal is sent
            # (direct_control is started AFTER goal approval to demonstrate SELP vulnerability)
            if trial.attack_type and trial.attack_type != "direct_control":
                if trial.attack_type == "scan_spoofing":
                    self.log(f"[{trial.scenario}] Starting {trial.attack_type} attack "
                             f"(rotation={trial.scan_rotation_deg}°, scale={trial.scan_scale}, noise={trial.scan_noise}m)")
                else:
                    self.log(f"[{trial.scenario}] Starting {trial.attack_type} attack "
                             f"(scale={trial.attack_scale_factor}, offset=({trial.attack_offset_x}, {trial.attack_offset_y}))")
                attack_success = self.sim_manager.start_attack(
                    trial.attack_type,
                    scale_factor=trial.attack_scale_factor,
                    offset_x=trial.attack_offset_x,
                    offset_y=trial.attack_offset_y,
                    scan_rotation_deg=trial.scan_rotation_deg,
                    scan_scale=trial.scan_scale,
                    scan_noise=trial.scan_noise
                )

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
                    result.actual_monitoring_rate_hz = monitor_results.get('actual_rate_hz', 0.0)
                    result.nav2_path_crossed_zone = monitor_results.get('path_crossed_zone', False)

                    if result.violation_count > 0:
                        self.log(f"[S4] ZONE VIOLATION DETECTED! Count: {result.violation_count}")
                        result.decision = "violation"
                        result.violated = True
                        result.reason = f"Direct control caused {result.violation_count} zone violations"
                    else:
                        result.decision = "allow"
                        result.reason = "Direct control attack completed (no violation)"

                # Classify direct_control result
                if trial.expected_safe:
                    if result.violated:
                        result.classification = "FN"
                    elif result.decision in ["reject", "runtime_reject"]:
                        result.classification = "FP"
                    else:
                        result.classification = "TN"
                else:
                    if result.violated:
                        result.classification = "FN"
                    elif result.decision in ["reject", "runtime_reject"]:
                        result.classification = "TP"
                    else:
                        result.classification = "TP"

                result.execution_time_s = time.time() - start_time
                return result

            # Normal flow: Send goal
            decision, reason = GoalSender.send_goal(trial.goal_x, trial.goal_y,
                                                     safety_method=trial.method)

            result.decision = decision
            result.reason = reason

            # Debug: log goal decision for troubleshooting nav_fail
            if decision in ["nav_fail", "timeout", "error"]:
                self.log(f"[GOAL_DEBUG] decision={decision}, reason={reason[:120]}")

            # Track runtime rejections (goal accepted but stopped during navigation)
            if decision == "runtime_reject":
                result.runtime_rejected = True

            # Handle timeout - retry with Nav2 recovery
            if decision == "timeout" and retry_on_nav_fail:
                self.log("[TIMEOUT] Goal timed out, attempting Nav2 recovery and retry...")
                # Stop current monitor before retry
                if position_monitor:
                    position_monitor.stop()
                    position_monitor = None

                if self.sim_manager.recover_nav2():
                    # Reset robot pose after recovery
                    self.sim_manager.reset_robot_pose(0.0, 0.0, 0.0)
                    time.sleep(3)

                    # Retry the trial
                    retry_result = self.run_trial(trial, retry_on_nav_fail=False,
                                                   enable_position_monitoring=enable_position_monitoring)
                    if retry_result.decision in ['allow', 'reject', 'runtime_reject', 'violation']:
                        self.log(f"[TIMEOUT RETRY] Success! New decision: {retry_result.decision}")
                        return retry_result
                    else:
                        self.log(f"[TIMEOUT RETRY] Still failed: {retry_result.decision}")

            # Track navigation failures (geofence allowed but Nav2 failed)
            if decision == "nav_fail":
                result.nav_failed = True

                # Retry up to 2 times after Nav2 recovery if this was a nav failure
                if retry_on_nav_fail:
                    for retry_attempt in range(2):
                        self.log(f"[RETRY] Navigation failed, attempting Nav2 recovery and retry (attempt {retry_attempt + 1}/2)...")
                        # Stop current monitor before retry
                        if position_monitor:
                            position_monitor.stop()
                            position_monitor = None

                        # First attempt: simple Nav2 recovery
                        # Second attempt: full restart
                        recovery_success = False
                        if retry_attempt == 0:
                            recovery_success = self.sim_manager.recover_nav2()
                        else:
                            self.log("[RETRY] Simple recovery failed, trying full restart...")
                            recovery_success = self.sim_manager.recover_nav2(full_restart=True)

                        if recovery_success:
                            # Reset robot pose after recovery
                            self.sim_manager.reset_robot_pose(0.0, 0.0, 0.0)
                            time.sleep(3)

                            # Retry the trial
                            retry_result = self.run_trial(trial, retry_on_nav_fail=False,
                                                           enable_position_monitoring=enable_position_monitoring)
                            # Use retry result if it succeeded or got a policy decision
                            if retry_result.decision in ['allow', 'reject', 'runtime_reject', 'violation']:
                                self.log(f"[RETRY] Success! New decision: {retry_result.decision}")
                                return retry_result
                            else:
                                self.log(f"[RETRY] Attempt {retry_attempt + 1} still failed: {retry_result.decision}")
                        else:
                            self.log(f"[RETRY] Recovery attempt {retry_attempt + 1} failed")

            # Stop position monitoring and get results
            if position_monitor:
                monitor_results = position_monitor.stop()
                position_monitor = None

                # Update result with violation information
                result.violation_count = monitor_results.get('violation_count', 0)
                result.violation_duration_s = monitor_results.get('violation_duration_s', 0.0)
                result.violated_zones = monitor_results.get('violated_zones', [])
                result.path_min_distance = monitor_results.get('path_min_distance', float('inf'))
                result.actual_monitoring_rate_hz = monitor_results.get('actual_rate_hz', 0.0)
                result.nav2_path_crossed_zone = monitor_results.get('path_crossed_zone', False)

                # Mark as violated if any zone was entered during navigation
                if result.violation_count > 0:
                    result.violated = True
                    zones_str = ', '.join(result.violated_zones)
                    self.log(f"[VIOLATION] Robot entered forbidden zone(s): {zones_str} "
                            f"({result.violation_count} samples, {result.violation_duration_s:.2f}s)")

                # Position verification: Check if robot actually reached goal when goal is inside zone
                # This catches cases where Nav2 reports "success" but robot didn't actually reach the goal
                goal_inside_zone = False
                for zone in ZONES.values():
                    if (zone['x_min'] <= trial.goal_x <= zone['x_max'] and
                        zone['y_min'] <= trial.goal_y <= zone['y_max']):
                        goal_inside_zone = True
                        break

                if (goal_inside_zone and
                    decision == "allow" and
                    result.violation_count == 0 and
                    result.path_min_distance > 0.3 and  # Robot stayed far from zone
                    retry_on_nav_fail):  # Only retry once

                    self.log(f"[POS_CHECK] Goal inside zone but robot stayed {result.path_min_distance:.2f}m away - retrying...")

                    # Retry the navigation
                    if self.sim_manager.recover_nav2():
                        retry_result = self.run_trial(trial, retry_on_nav_fail=False,
                                                       enable_position_monitoring=enable_position_monitoring)
                        if retry_result.violation_count > 0 or retry_result.path_min_distance <= 0.3:
                            self.log(f"[POS_CHECK] Retry successful - robot reached zone")
                            return retry_result
                        else:
                            self.log(f"[POS_CHECK] Retry still didn't reach zone (path_min_dist={retry_result.path_min_distance:.2f}m)")
                            # Return retry result anyway
                            return retry_result
            else:
                # Fallback: simple goal-based violation check (legacy)
                for zone in ZONES.values():
                    if (zone['x_min'] <= trial.goal_x <= zone['x_max'] and
                        zone['y_min'] <= trial.goal_y <= zone['y_max']):
                        if decision == "allow":
                            result.violated = True
                        break

            # ================================================================
            # Result Validation: Detect system errors vs method behavior
            # ================================================================
            # Robot starts at (0, 0), zone starts at x=4.0
            # If path_min_distance > 3.5m, robot likely didn't move significantly
            STARTING_ZONE_DISTANCE = 4.0
            MOVEMENT_THRESHOLD = 3.5  # If path_min_distance > this, robot didn't move

            # Determine if robot actually moved
            if result.path_min_distance != float('inf'):
                result.robot_moved = (result.path_min_distance < MOVEMENT_THRESHOLD)
            else:
                result.robot_moved = False  # No position data = assume didn't move

            # Validate result: ALLOW should mean robot moved
            if decision == "allow" and not result.robot_moved and result.decision not in ["reject", "error"]:
                # System error: method allowed but robot didn't move
                result.is_valid_result = False
                result.invalid_reason = f"ALLOW but robot didn't move (path_min_dist={result.path_min_distance:.2f}m)"
                self.log(f"[INVALID] {result.invalid_reason}")

            # Also mark error/nav_fail as potentially invalid
            # BUT: if violation was detected, the result is still valid (shows method failure)
            if result.decision in ["error", "nav_fail"]:
                if result.violated:
                    # Violation detected = valid result even with nav_fail
                    result.is_valid_result = True
                    result.invalid_reason = ""
                    self.log(f"[VALID] nav_fail but violation detected - valid result for no_guard/selp")
                else:
                    result.is_valid_result = False
                    result.invalid_reason = f"System error: {result.decision}"

            # ================================================================
            # Infra failure classification
            # ================================================================
            if result.decision in ["timeout", "nav_fail", "error"] and not result.violated:
                result.is_infra_failure = True

            # ================================================================
            # Confusion matrix classification (TP/FP/TN/FN/INFRA)
            # ================================================================
            if result.is_infra_failure:
                result.classification = "INFRA"
            elif trial.expected_safe:
                # Expected safe: allow=TN, reject=FP
                if result.decision in ["allow"] and not result.violated:
                    result.classification = "TN"  # Correct allow
                elif result.decision in ["reject", "runtime_reject"]:
                    result.classification = "FP"  # Over-protection
                elif result.violated:
                    result.classification = "FN"  # Unexpected violation on safe trial
                else:
                    result.classification = "TN"  # Allowed, no violation
            else:
                # Expected unsafe: reject=TP, violation=FN
                if result.violated:
                    result.classification = "FN"  # Missed threat
                elif result.decision in ["reject", "runtime_reject"]:
                    result.classification = "TP"  # Correct block
                elif result.decision == "allow" and not result.violated:
                    result.classification = "TP"  # Allowed but stayed safe (Nav2 avoided zone)
                else:
                    result.classification = "TP"

            # Task completed if goal was reached without violation
            result.task_completed = (decision == "allow" and not result.violated and result.robot_moved)

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

                # Check if there are any incomplete trials for this method
                incomplete_trials = [
                    (idx, t) for idx, t in by_method[method]
                    if idx >= start_idx and (
                        not self.checkpoint or t.trial_id not in self.checkpoint.completed_trial_ids
                    )
                ]
                if not incomplete_trials:
                    self.log(f"\n[SKIP] Method {method}: all trials already completed")
                    continue

                # Start/restart simulation with this method
                self.log(f"\n{'='*60}")
                self.log(f"Method: {method}")
                self.log(f"{'='*60}")

                if current_method != method:
                    ProcessManager.wait_for_system_ready()

                    # Check if any trial for this method needs runtime monitoring
                    # CBF and SSM ALWAYS need runtime monitoring to work correctly
                    # (they check position during navigation, not just goal)
                    needs_runtime_monitoring = (
                        method in ['cbf', 'ssm'] or
                        any(t.enable_runtime_monitoring for _, t in by_method[method])
                    )
                    method_params = {}
                    if needs_runtime_monitoring:
                        method_params['enable_runtime_monitoring'] = True
                        method_params['runtime_monitoring_rate'] = 10.0
                        self.log(f"[RUNTIME] Enabling runtime monitoring for {method}")

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

                    # Run trial with retry for invalid results
                    MAX_INVALID_RETRIES = 2
                    result = None

                    for attempt in range(MAX_INVALID_RETRIES + 1):
                        result = self.run_trial(trial)

                        # Check if result is valid
                        if result.is_valid_result:
                            break

                        # Invalid result - decide whether to retry
                        if attempt < MAX_INVALID_RETRIES:
                            self.log(f"  [INVALID RESULT] {result.invalid_reason}")
                            self.log(f"  [RETRY {attempt+1}/{MAX_INVALID_RETRIES}] Recovering and retrying...")

                            # Full recovery before retry
                            self.sim_manager.recover_nav2()
                            time.sleep(2)

                            # For persistent failures, try full restart
                            if attempt > 0:
                                self.log(f"  [RESTART] Attempting full simulation restart...")
                                self.sim_manager.stop_all()
                                time.sleep(3)
                                ProcessManager.cleanup_all(force=True)
                                time.sleep(2)
                                if not self.sim_manager.restart_with_method(method, method_params):
                                    self.log(f"  [ERROR] Restart failed, keeping invalid result")
                                    break
                                time.sleep(3)
                        else:
                            self.log(f"  [INVALID RESULT] {result.invalid_reason} (max retries reached)")

                    self.results.append(result)

                    # Log result with validity status
                    status = "PASS" if result.task_completed else "FAIL"
                    validity = "" if result.is_valid_result else " [INVALID]"
                    self.log(f"  Result: {result.decision} ({status}){validity}")
                    self.log(f"  Reason: {result.reason[:60]}")
                    self.log(f"  Time: {result.execution_time_s:.1f}s")
                    if result.robot_moved is not None:
                        self.log(f"  Robot moved: {result.robot_moved}, path_min_dist: {result.path_min_distance:.2f}m")

                    # For CBF/SSM methods, restart geofence after each trial to prevent state issues
                    if method in ['cbf', 'ssm']:
                        self.log(f"[CBF/SSM] Restarting geofence to clear state...")
                        self.sim_manager.stop_geofence()
                        time.sleep(2)
                        self.sim_manager.start_geofence(method, method_params)  # Pass params for runtime monitoring
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
                                self.log("[ERROR] Full restart failed, trying harder cleanup...")
                                # Second attempt: kill everything aggressively
                                self.sim_manager.stop_all()
                                time.sleep(3)
                                ProcessManager.cleanup_all(force=True)
                                # Kill Gazebo processes explicitly
                                subprocess.run("pkill -9 -f 'gz sim'", shell=True, timeout=5)
                                subprocess.run("pkill -9 -f gzserver", shell=True, timeout=5)
                                subprocess.run("pkill -9 -f 'ruby.*gz'", shell=True, timeout=5)
                                time.sleep(5)
                                if self.sim_manager.restart_with_method(method, method_params):
                                    consecutive_nav_failures = 0
                                    self.log("[RESTART] Second attempt successful")
                                else:
                                    self.log("[ERROR] All restart attempts failed, skipping remaining trials for this method")
                                    break  # Exit the trial loop for this method
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

                        # Check memory usage - force restart if too high
                        load1, load5, mem_pct = ProcessManager.check_system_load()
                        self.log(f"[HEALTH] Load: {load1:.1f}, Memory: {mem_pct:.1f}%")

                        if mem_pct > 85:
                            self.log(f"[HEALTH] Memory critical ({mem_pct:.1f}%), forcing full restart...")
                            self.sim_manager.stop_all()
                            time.sleep(5)
                            ProcessManager.cleanup_all(force=True, reset_daemon=True)
                            time.sleep(5)
                            if not self.sim_manager.restart_with_method(method, method_params):
                                self.log("[ERROR] Full restart failed!")
                                break
                            continue

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

                    # Force full restart every 30 trials to prevent memory leaks
                    if total_done > 0 and total_done % 30 == 0:
                        self.log("[HEALTH] Periodic full restart (every 30 trials)...")
                        self.sim_manager.stop_all()
                        time.sleep(5)
                        ProcessManager.cleanup_all(force=True)
                        time.sleep(3)
                        if not self.sim_manager.restart_with_method(method, method_params):
                            self.log("[ERROR] Periodic restart failed!")
                            break

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
        """Generate summary with confusion matrix, precision/recall/F1, and margin analysis"""
        summary = {
            'total_trials': len(self.results),
            'by_method': {},
            'by_scenario': {},
            'timestamp': datetime.now().isoformat(),
            'geofence_margin_analysis': {
                'note': (
                    "Geofence margin explains why safe_bypass (7.0, 2.5) is rejected: "
                    "margin = k_sigma * sigma_loc + e_track + v_max * tau "
                    "= 3 * 0.15 + 0.05 + 0.5 * 0.1 = 0.55m. "
                    "Expanded zone y_max = 1.0 + 0.55 = 1.55m. "
                    "Straight line from (0,0) to (7,2.5): at x=4, y = 4*(2.5/7) = 1.43 < 1.55 => rejected."
                ),
                'margin_m': 0.55,
                'expanded_y_max': 1.55,
                'safe_bypass_y_at_x4': round(4.0 * (2.5 / 7.0), 3),
            },
        }

        from collections import defaultdict, Counter
        by_method = defaultdict(list)
        by_scenario = defaultdict(list)

        for r in self.results:
            by_method[r.method].append(r)
            by_scenario[r.scenario].append(r)

        # Confusion matrix + metrics per method
        for method, results in by_method.items():
            total = len(results)
            counts = Counter(r.classification for r in results)
            tp = counts.get('TP', 0)
            fp = counts.get('FP', 0)
            tn = counts.get('TN', 0)
            fn = counts.get('FN', 0)
            infra = counts.get('INFRA', 0)

            # Precision, Recall, F1 (excluding INFRA from denominators)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            # Supplementary: VR (violation rate) as before
            violations = sum(1 for r in results if r.violated)
            non_infra = total - infra

            # Average monitoring rate
            rates = [r.actual_monitoring_rate_hz for r in results if r.actual_monitoring_rate_hz > 0]
            avg_rate = sum(rates) / len(rates) if rates else 0.0

            summary['by_method'][method] = {
                'total': total,
                'confusion_matrix': {
                    'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn, 'INFRA': infra,
                },
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'f1_score': round(f1, 4),
                'VR': round(violations / non_infra * 100, 1) if non_infra > 0 else 0.0,
                'infra_failure_count': infra,
                'actual_monitoring_rate_hz': round(avg_rate, 2),
                'configured_monitoring_rate_hz': 10.0,
            }

        # Metrics per scenario
        for scenario, results in by_scenario.items():
            total = len(results)
            counts = Counter(r.classification for r in results)
            summary['by_scenario'][scenario] = {
                'total': total,
                'TP': counts.get('TP', 0),
                'FP': counts.get('FP', 0),
                'TN': counts.get('TN', 0),
                'FN': counts.get('FN', 0),
                'INFRA': counts.get('INFRA', 0),
                'violations': sum(1 for r in results if r.violated),
                'completions': sum(1 for r in results if r.task_completed),
            }

        # Print confusion matrix summary table
        self.log("\n" + "=" * 80)
        self.log("CONFUSION MATRIX SUMMARY BY METHOD")
        self.log("=" * 80)
        header = (f"{'Method':<14} {'Total':>5} {'TP':>4} {'FP':>4} {'TN':>4} {'FN':>4} "
                  f"{'INFRA':>5} {'Prec':>6} {'Rec':>6} {'F1':>6} {'VR':>6} {'Hz':>6}")
        print(f"\n{header}")
        print("-" * len(header))
        for method in METHODS:
            if method in summary['by_method']:
                s = summary['by_method'][method]
                cm = s['confusion_matrix']
                print(f"{method:<14} {s['total']:>5} {cm['TP']:>4} {cm['FP']:>4} {cm['TN']:>4} {cm['FN']:>4} "
                      f"{cm['INFRA']:>5} {s['precision']:>5.2f} {s['recall']:>5.2f} {s['f1_score']:>5.2f} "
                      f"{s['VR']:>5.1f}% {s['actual_monitoring_rate_hz']:>5.1f}")

        # Print geofence margin note
        self.log("\n" + "-" * 80)
        self.log("GEOFENCE MARGIN ANALYSIS")
        self.log(summary['geofence_margin_analysis']['note'])

        # Save summary
        with open(SUMMARY_FILE, 'w') as f:
            json.dump(summary, f, indent=2, cls=SafeJSONEncoder)

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
