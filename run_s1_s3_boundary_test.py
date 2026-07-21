#!/usr/bin/env python3
"""
S1-S3 Boundary Test Experiment (Improved Version)
==================================================

Tests defense method differences at margin boundaries.

Improvements over original:
1. Simulation reuse - Gazebo/Nav2 stay running, only defense methods change
2. Increased wait times for Nav2 activation
3. State verification (map frame check) before navigation

Forbidden Zone: x=[4.0, 6.0], y=[-1.0, 1.0]
Zone boundary at x=4.0

Defense Method Margins:
- no_guard: 0m (no safety)
- SELP: 0m (goal gate only, no margin)
- CBF: 0.3m fixed
- SSM: ~0.475m at v=0.5
- Geofence: 0.55m adaptive
"""

import subprocess
import time
import os
import sys
import json
import signal
from datetime import datetime
from typing import List, Dict, Tuple, Optional

WORKSPACE = "/home/jim/ros2_motion_planning_tutorials"
RESULTS_DIR = f"{WORKSPACE}/experiment_results/s1_s3_boundary"

# Forbidden Zone: x=[4.0, 6.0], y=[-1.0, 1.0]
ZONE_X_MIN = 4.0
ZONE_X_MAX = 6.0
ZONE_Y_MIN = -1.0
ZONE_Y_MAX = 1.0

# Defense methods and their margins
METHODS = ['no_guard', 'selp', 'cbf', 'ssm', 'geofence']
MARGINS = {
    'no_guard': 0.0,
    'selp': 0.0,
    'cbf': 0.3,
    'ssm': 0.475,
    'geofence': 0.55
}

NUM_SEEDS = 2  # Trials per configuration (reduced for reliability)


def run_cmd(cmd: str, timeout: int = 10) -> str:
    try:
        result = subprocess.run(
            cmd, shell=True, executable='/bin/bash',
            capture_output=True, text=True, timeout=timeout
        )
        return result.stdout + result.stderr
    except:
        return ""


def run_bg(cmd: str, log_file: str = None):
    if log_file:
        f = open(log_file, 'w')
        return subprocess.Popen(
            cmd, shell=True, executable='/bin/bash',
            stdout=f, stderr=subprocess.STDOUT, preexec_fn=os.setsid
        )
    return subprocess.Popen(
        cmd, shell=True, executable='/bin/bash',
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid
    )


def kill_proc(proc):
    """Safely kill a process and its process group"""
    if proc is None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=5)
    except:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except:
            pass


def source_ros():
    return f"source /opt/ros/jazzy/setup.bash && source {WORKSPACE}/install/setup.bash"


def cleanup_all():
    """Full cleanup - kill everything"""
    patterns = [
        "gz sim", "gzserver", "gzclient", "ruby.*gz",
        "nav2_", "controller_server", "planner_server",
        "behavior_server", "bt_navigator", "lifecycle_manager",
        "goal_gate", "cmd_vel_guard", "hardware_geofence",
        "attack_", "relay", "ros_gz", "parameter_bridge", "amcl"
    ]
    for p in patterns:
        subprocess.run(f"pkill -9 -f '{p}'", shell=True, capture_output=True, timeout=5)
    time.sleep(3)


def cleanup_defense_only():
    """Clean only defense-related processes (keep Gazebo/Nav2)"""
    patterns = [
        "goal_gate", "cmd_vel_guard", "hardware_geofence",
        "relay.*cmd_vel"
    ]
    for p in patterns:
        subprocess.run(f"pkill -9 -f '{p}'", shell=True, capture_output=True, timeout=3)
    time.sleep(1)


def activate_nav2_nodes():
    """Activate all Nav2 lifecycle nodes with extended timeouts"""
    # Nodes that show up in ros2 node list
    regular_nodes = [
        'map_server', 'amcl',
        'controller_server', 'smoother_server',
        'behavior_server', 'bt_navigator', 'waypoint_follower',
        'velocity_smoother', 'collision_monitor'
    ]

    # planner_server doesn't always show in ros2 node list (DDS issue)
    # so we handle it separately via service calls
    service_nodes = ['planner_server']

    # Configure regular nodes
    for node in regular_nodes:
        run_cmd(f"{source_ros()} && ros2 lifecycle set {node} configure 2>&1", timeout=10)
        time.sleep(0.5)

    # Configure planner_server via service call
    for node in service_nodes:
        run_cmd(f'{source_ros()} && ros2 service call /{node}/change_state lifecycle_msgs/srv/ChangeState "{{transition: {{id: 1}}}}" 2>&1', timeout=15)
        time.sleep(1)

    time.sleep(3)  # Wait after configure

    # Activate regular nodes
    for node in regular_nodes:
        run_cmd(f"{source_ros()} && ros2 lifecycle set {node} activate 2>&1", timeout=10)
        time.sleep(0.5)

    # Activate planner_server via service call
    for node in service_nodes:
        run_cmd(f'{source_ros()} && ros2 service call /{node}/change_state lifecycle_msgs/srv/ChangeState "{{transition: {{id: 3}}}}" 2>&1', timeout=15)
        time.sleep(1)

    time.sleep(5)  # Extended wait after activation


def check_map_frame_available() -> bool:
    """FIX #3: Verify map frame is available before navigation"""
    # Check if AMCL is publishing particle cloud (indicates localization is working)
    output = run_cmd(f"{source_ros()} && ros2 topic echo /amcl_pose --once 2>&1", timeout=8)
    if "pose:" in output and "position:" in output:
        return True

    # Fallback: check tf2 (with longer timeout)
    output = run_cmd(f"{source_ros()} && timeout 5 ros2 run tf2_ros tf2_echo map odom 2>&1", timeout=10)
    return "Translation:" in output


def wait_for_map_frame(timeout: int = 60) -> bool:
    """Wait until map frame is available"""
    start = time.time()
    while time.time() - start < timeout:
        if check_map_frame_available():
            return True
        time.sleep(2)
    return False


def set_initial_pose(x=0.0, y=0.0, yaw=0.0):
    import math
    qz = math.sin(yaw / 2.0)
    qw = math.cos(yaw / 2.0)
    cmd = f'''{source_ros()} && ros2 topic pub --once /initialpose geometry_msgs/msg/PoseWithCovarianceStamped "{{
      header: {{frame_id: 'map'}},
      pose: {{pose: {{position: {{x: {x}, y: {y}, z: 0.0}}, orientation: {{z: {qz}, w: {qw}}}}},
        covariance: [0.25,0,0,0,0,0, 0,0.25,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0.07]}}
    }}"'''
    return run_cmd(cmd, timeout=10)


def reset_robot_position():
    """Reset robot to start position in Gazebo"""
    # Use Gazebo service to teleport robot (model name is 'mobile_manip')
    cmd = f'''{source_ros()} && timeout 5 ros2 service call /world/empty/set_pose ros_gz_interfaces/srv/SetEntityPose "{{
        entity: {{name: 'mobile_manip', type: 2}},
        pose: {{position: {{x: 0.0, y: 0.0, z: 0.1}}, orientation: {{w: 1.0}}}}
    }}" 2>&1'''
    result = run_cmd(cmd, timeout=10)
    time.sleep(2)

    # Send initial pose to AMCL to reset localization (multiple times for reliability)
    for _ in range(3):
        set_initial_pose(0.0, 0.0, 0.0)
        time.sleep(1)
    time.sleep(2)

    # Verify position
    x, y = get_robot_position()
    if x is not None and abs(x) < 1.0 and abs(y) < 1.0:
        return True
    return False


def get_robot_position() -> Tuple[Optional[float], Optional[float]]:
    """Get robot position from AMCL pose (map frame, not odom)"""
    # Use AMCL pose for actual map-relative position
    output = run_cmd(f"{source_ros()} && ros2 topic echo /amcl_pose --once 2>&1", timeout=8)
    if 'position:' in output:
        lines = output.split('\n')
        for i, line in enumerate(lines):
            if 'position:' in line and i+2 < len(lines):
                try:
                    x = float(lines[i+1].split(':')[1].strip())
                    y = float(lines[i+2].split(':')[1].strip())
                    return x, y
                except:
                    pass

    # Fallback to TF lookup if AMCL pose not available
    output = run_cmd(f"{source_ros()} && timeout 3 ros2 run tf2_ros tf2_echo map base_footprint 2>&1", timeout=5)
    if 'Translation:' in output:
        lines = output.split('\n')
        for line in lines:
            if 'Translation:' in line:
                try:
                    # Format: - Translation: [x, y, z]
                    coords = line.split('[')[1].split(']')[0]
                    parts = coords.split(',')
                    x = float(parts[0].strip())
                    y = float(parts[1].strip())
                    return x, y
                except:
                    pass
    return None, None


def is_in_zone(x: float, y: float) -> bool:
    """Check if position is inside forbidden zone"""
    return (ZONE_X_MIN <= x <= ZONE_X_MAX and
            ZONE_Y_MIN <= y <= ZONE_Y_MAX)


def distance_to_zone(x: float, y: float) -> float:
    """Distance from position to zone boundary (negative if inside)"""
    if is_in_zone(x, y):
        return 0.0
    dx = max(ZONE_X_MIN - x, 0, x - ZONE_X_MAX)
    dy = max(ZONE_Y_MIN - y, 0, y - ZONE_Y_MAX)
    return (dx**2 + dy**2)**0.5


def send_nav_goal(x: float, y: float, use_gate: bool = False):
    """Send navigation goal. If use_gate=True, send via goal gate."""
    topic = '/navigate_to_pose_safe' if use_gate else '/navigate_to_pose'
    cmd = f'''{source_ros()} && ros2 action send_goal {topic} nav2_msgs/action/NavigateToPose "{{
      pose: {{header: {{frame_id: 'map'}}, pose: {{position: {{x: {x}, y: {y}, z: 0.0}}, orientation: {{w: 1.0}}}}}}
    }}"'''
    return run_bg(cmd, '/tmp/nav_goal_s1s3.log')


def cancel_nav_goal():
    """Cancel any active navigation goal"""
    run_cmd(f"{source_ros()} && ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose --cancel", timeout=5)
    time.sleep(1)


def generate_trials() -> List[Dict]:
    """Generate trial configurations"""
    trials = []

    # ==========================================================
    # S1: Direct Zone Goal with Boundary Tests
    # Zone boundary at x=4.0
    # ==========================================================
    s1_configs = [
        # Inside zone - all should block/violate
        {"intensity": "inside", "goal_x": 5.0, "goal_y": 0.0,
         "boundary_dist": -1.0, "desc": "Inside zone (x=5.0)"},

        # Just outside zone - tests SELP (no margin)
        {"intensity": "d0.02m", "goal_x": 3.98, "goal_y": 0.0,
         "boundary_dist": 0.02, "desc": "0.02m from boundary"},

        # Outside CBF margin (0.3m) but within SSM/Geo
        {"intensity": "d0.32m", "goal_x": 3.68, "goal_y": 0.0,
         "boundary_dist": 0.32, "desc": "0.32m - CBF allows"},

        # Outside SSM margin (~0.475m) but within Geo (0.55m)
        {"intensity": "d0.50m", "goal_x": 3.50, "goal_y": 0.0,
         "boundary_dist": 0.50, "desc": "0.50m - SSM allows"},

        # Outside all margins
        {"intensity": "d0.60m", "goal_x": 3.40, "goal_y": 0.0,
         "boundary_dist": 0.60, "desc": "0.60m - all allow"},
    ]

    for method in METHODS:
        for cfg in s1_configs:
            for seed in range(NUM_SEEDS):
                trials.append({
                    "trial_id": f"S1_{method}_{cfg['intensity']}_s{seed}",
                    "scenario": "S1",
                    "method": method,
                    "intensity": cfg['intensity'],
                    "goal_x": cfg['goal_x'],
                    "goal_y": cfg['goal_y'],
                    "boundary_dist": cfg['boundary_dist'],
                    "desc": cfg['desc'],
                    "seed": seed
                })

    # ==========================================================
    # S2: Incremental Approach (Salami Attack)
    # ==========================================================
    s2_configs = [
        {"intensity": "step1_2.0m", "goal_x": 2.0, "goal_y": 0.0,
         "boundary_dist": 2.0, "desc": "Step 1: 2.0m from boundary"},
        {"intensity": "step2_1.0m", "goal_x": 3.0, "goal_y": 0.0,
         "boundary_dist": 1.0, "desc": "Step 2: 1.0m from boundary"},
        {"intensity": "step3_0.5m", "goal_x": 3.5, "goal_y": 0.0,
         "boundary_dist": 0.5, "desc": "Step 3: 0.5m - within Geo margin"},
        {"intensity": "step4_0.25m", "goal_x": 3.75, "goal_y": 0.0,
         "boundary_dist": 0.25, "desc": "Step 4: 0.25m - within CBF margin"},
    ]

    for method in METHODS:
        for cfg in s2_configs:
            for seed in range(NUM_SEEDS):
                trials.append({
                    "trial_id": f"S2_{method}_{cfg['intensity']}_s{seed}",
                    "scenario": "S2",
                    "method": method,
                    "intensity": cfg['intensity'],
                    "goal_x": cfg['goal_x'],
                    "goal_y": cfg['goal_y'],
                    "boundary_dist": cfg['boundary_dist'],
                    "desc": cfg['desc'],
                    "seed": seed
                })

    # ==========================================================
    # S3: Path Through Zone
    # Goal is safe but path might cross zone
    # ==========================================================
    s3_configs = [
        # Path crosses zone center (robot starts at 0,0, goes to 8,0)
        {"intensity": "through_center", "goal_x": 8.0, "goal_y": 0.0,
         "boundary_dist": 0.0, "desc": "Path crosses zone center"},

        # Path grazes zone boundary (y=-1.3 is 0.3m below zone y_min=-1)
        {"intensity": "graze_0.3m", "goal_x": 8.0, "goal_y": -1.3,
         "boundary_dist": 0.3, "desc": "Path grazes 0.3m below zone"},

        # Safe path (y=-2.0 is 1m below zone)
        {"intensity": "safe_path", "goal_x": 8.0, "goal_y": -2.0,
         "boundary_dist": 1.0, "desc": "Safe path 1m below zone"},
    ]

    for method in METHODS:
        for cfg in s3_configs:
            for seed in range(NUM_SEEDS):
                trials.append({
                    "trial_id": f"S3_{method}_{cfg['intensity']}_s{seed}",
                    "scenario": "S3",
                    "method": method,
                    "intensity": cfg['intensity'],
                    "goal_x": cfg['goal_x'],
                    "goal_y": cfg['goal_y'],
                    "boundary_dist": cfg['boundary_dist'],
                    "desc": cfg['desc'],
                    "seed": seed
                })

    return trials


class SimulationStack:
    """FIX #1: Manages simulation stack - keeps Gazebo/Nav2 running"""

    def __init__(self):
        self.gz_proc = None
        self.nav_proc = None
        self.odom_relay = None
        self.scan_relay = None
        self.defense_procs = []
        self.is_running = False

    def start(self):
        """Start Gazebo, relays, and Nav2 (one time setup)"""
        print("\n" + "=" * 60)
        print("STARTING SIMULATION STACK (one-time setup)")
        print("=" * 60)

        # 1. Start Gazebo
        print("[1/5] Starting Gazebo...")
        gz_cmd = f"{source_ros()} && ros2 launch mobile_manip_moveit_config mobile_manipulator.launch.py use_sim_time:=true headless:=true"
        self.gz_proc = run_bg(gz_cmd, '/tmp/gz_s1s3_stack.log')
        time.sleep(50)  # FIX #2: Extended wait time

        # 2. Start odom relay
        print("[2/5] Starting odom relay...")
        odom_cmd = f"{source_ros()} && ros2 run topic_tools relay /odom_real /odom"
        self.odom_relay = run_bg(odom_cmd)
        time.sleep(2)

        # 3. Start scan relay
        print("[3/5] Starting scan relay...")
        scan_cmd = f"{source_ros()} && ros2 run topic_tools relay /scan_real /scan"
        self.scan_relay = run_bg(scan_cmd)
        time.sleep(3)

        # 4. Start Navigation
        print("[4/5] Starting Navigation...")
        nav_cmd = f"{source_ros()} && ros2 launch mobile_manip_moveit_config navigation.launch.py use_sim_time:=true rviz:=false"
        self.nav_proc = run_bg(nav_cmd, '/tmp/nav_s1s3_stack.log')
        time.sleep(50)  # FIX #2: Extended wait time

        # 5. Activate Nav2 nodes
        print("[5/5] Activating Nav2 nodes...")
        activate_nav2_nodes()

        # FIX #3: Wait for map frame to be available
        print("    Waiting for map frame...")
        if wait_for_map_frame(60):
            print("    Map frame available!")
        else:
            print("    WARNING: Map frame not available after 60s")

        # Set initial pose
        print("    Setting initial pose...")
        set_initial_pose(0.0, 0.0, 0.0)
        time.sleep(8)  # FIX #2: Extended wait

        self.is_running = True
        print("\nSimulation stack ready!\n")

    def stop(self):
        """Stop everything"""
        print("\nStopping simulation stack...")
        self.stop_defense()
        kill_proc(self.nav_proc)
        kill_proc(self.odom_relay)
        kill_proc(self.scan_relay)
        kill_proc(self.gz_proc)
        cleanup_all()
        self.is_running = False

    def stop_defense(self):
        """Stop only defense-related processes"""
        for proc in self.defense_procs:
            kill_proc(proc)
        self.defense_procs = []
        cleanup_defense_only()

    def start_defense(self, method: str) -> bool:
        """Start defense method, returns use_gate flag"""
        use_gate = False

        if method == 'no_guard':
            print(f"    Defense: no_guard (direct relay)")
            relay_cmd = f"{source_ros()} && ros2 run topic_tools relay /cmd_vel_nav /cmd_vel"
            self.defense_procs.append(run_bg(relay_cmd))

        elif method == 'selp':
            print(f"    Defense: SELP (Goal Gate only)")
            gate_cmd = f"{source_ros()} && ros2 run geofence_policy_enforcer goal_gate_node --ros-args -p safety_method:=selp"
            self.defense_procs.append(run_bg(gate_cmd, f'/tmp/gate_s1s3_{method}.log'))
            time.sleep(3)
            relay_cmd = f"{source_ros()} && ros2 run topic_tools relay /cmd_vel_nav /cmd_vel"
            self.defense_procs.append(run_bg(relay_cmd))
            use_gate = True

        elif method in ['cbf', 'ssm']:
            print(f"    Defense: {method.upper()} (Cmd Vel Guard)")
            guard_cmd = f"{source_ros()} && ros2 run geofence_policy_enforcer cmd_vel_guard_node --ros-args -p safety_method:={method} -p input_topic:=/cmd_vel_nav -p output_topic:=/cmd_vel"
            self.defense_procs.append(run_bg(guard_cmd, f'/tmp/guard_s1s3_{method}.log'))

        elif method == 'geofence':
            print(f"    Defense: Geofence (Full stack)")
            gate_cmd = f"{source_ros()} && ros2 run geofence_policy_enforcer goal_gate_node --ros-args -p safety_method:=geofence"
            self.defense_procs.append(run_bg(gate_cmd, f'/tmp/gate_s1s3_{method}.log'))
            time.sleep(3)
            guard_cmd = f"{source_ros()} && ros2 run geofence_policy_enforcer cmd_vel_guard_node --ros-args -p safety_method:=geofence -p input_topic:=/cmd_vel_nav -p output_topic:=/cmd_vel"
            self.defense_procs.append(run_bg(guard_cmd, f'/tmp/guard_s1s3_{method}.log'))
            use_gate = True

        time.sleep(3)
        return use_gate

    def reset_for_trial(self) -> bool:
        """Reset robot position for next trial"""
        # Cancel any active navigation
        cancel_nav_goal()

        # Reset robot position
        if not reset_robot_position():
            # Fallback: just set initial pose multiple times
            for _ in range(3):
                set_initial_pose(0.0, 0.0, 0.0)
                time.sleep(2)

        # FIX #3: Verify map frame still valid
        if not check_map_frame_available():
            print("    WARNING: Map frame lost, re-setting initial pose...")
            for _ in range(5):
                set_initial_pose(0.0, 0.0, 0.0)
                time.sleep(2)
                if check_map_frame_available():
                    break

        time.sleep(3)
        return True


def run_trial_with_stack(trial: Dict, stack: SimulationStack, current_method: str) -> Dict:
    """Run a single trial using the persistent simulation stack"""
    method = trial['method']
    goal_x = trial['goal_x']
    goal_y = trial['goal_y']

    result = {
        **trial,
        'max_x': 0.0,
        'violated': False,
        'decision': '',
        'timestamp': datetime.now().isoformat()
    }

    # Switch defense method if needed
    if method != current_method:
        stack.stop_defense()
        time.sleep(2)

    # Start defense method
    use_gate = stack.start_defense(method)

    # FIX #3: Verify map frame before sending goal
    print("    Verifying map frame...")
    if not check_map_frame_available():
        print("    WARNING: Map frame not available, waiting...")
        if not wait_for_map_frame(30):
            print("    ERROR: Map frame never became available")
            result['error'] = "map_frame_unavailable"
            return result

    # Send goal
    gate_str = " via gate" if use_gate else ""
    print(f"    Sending goal ({goal_x}, {goal_y}){gate_str}...")
    goal_proc = send_nav_goal(goal_x, goal_y, use_gate=use_gate)

    # Monitor position
    print("    Monitoring (60s)...")
    start_time = time.time()
    max_x = 0.0
    violated = False
    movement_detected = False

    while time.time() - start_time < 60:
        x, y = get_robot_position()
        if x is not None:
            if abs(x) > 0.3:  # Robot moved from start
                movement_detected = True
            max_x = max(max_x, x)
            in_zone = is_in_zone(x, y)

            elapsed = time.time() - start_time
            if in_zone:
                print(f"      t={elapsed:.0f}s: x={x:.2f}, y={y:.2f} VIOLATION!")
                violated = True
                break
            elif elapsed % 10 < 5:  # Print every ~10s
                print(f"      t={elapsed:.0f}s: x={x:.2f}, y={y:.2f}")

        time.sleep(3)

    # Kill goal process
    kill_proc(goal_proc)

    result['max_x'] = max_x
    result['violated'] = violated or is_in_zone(max_x, 0.0)
    result['movement_detected'] = movement_detected

    return result


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 70)
    print("S1-S3 BOUNDARY TEST EXPERIMENT (IMPROVED)")
    print("=" * 70)
    print(f"Zone: x=[{ZONE_X_MIN}, {ZONE_X_MAX}], y=[{ZONE_Y_MIN}, {ZONE_Y_MAX}]")
    print(f"Methods: {METHODS}")
    print(f"Seeds per config: {NUM_SEEDS}")
    print("\nImprovements:")
    print("  1. Simulation reuse (Gazebo/Nav2 stay running)")
    print("  2. Extended wait times")
    print("  3. Map frame verification before navigation")
    print("=" * 70)

    trials = generate_trials()
    print(f"\nGenerated {len(trials)} trials")

    # Group by method to minimize defense switches
    trials_by_method = {}
    for t in trials:
        if t['method'] not in trials_by_method:
            trials_by_method[t['method']] = []
        trials_by_method[t['method']].append(t)

    all_results = []
    stack = None

    try:
        cleanup_all()
        trial_num = 0

        for method in METHODS:
            print(f"\n{'='*60}")
            print(f"METHOD: {method.upper()} (margin: {MARGINS[method]}m)")
            print(f"{'='*60}")

            # RESTART SIMULATION FOR EACH METHOD (for reliability)
            print("    Restarting simulation stack for new method...")
            cleanup_all()
            time.sleep(5)
            stack = SimulationStack()
            stack.start()

            method_trials = trials_by_method.get(method, [])

            for i, trial in enumerate(method_trials):
                trial_num += 1
                print(f"\n[{trial_num}/{len(trials)}] {trial['trial_id']}")
                print(f"    Goal: ({trial['goal_x']}, {trial['goal_y']})")
                print(f"    Desc: {trial['desc']}")

                # Reset robot position before each trial
                print("    Resetting robot position...")
                stack.reset_for_trial()

                try:
                    result = run_trial_with_stack(trial, stack, method)
                    all_results.append(result)

                    if result.get('error'):
                        print(f"    [ERROR] {result['error']}")
                    elif result['violated']:
                        print(f"    [RESULT] VIOLATED! (max_x={result['max_x']:.2f})")
                    else:
                        moved = "moved" if result.get('movement_detected') else "NO MOVEMENT"
                        print(f"    [RESULT] max_x={result['max_x']:.2f}m ({moved})")

                except Exception as e:
                    print(f"    [ERROR] {e}")
                    all_results.append({
                        **trial,
                        'max_x': 0.0,
                        'violated': False,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    })

                # Save checkpoint every 10 trials
                if trial_num % 10 == 0:
                    save_results(all_results, "checkpoint")

            # Cleanup after each method
            stack.stop()
            time.sleep(3)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user!")
    finally:
        # Final save and cleanup
        save_results(all_results, "final")
        cleanup_all()

    # Print summary
    print_summary(all_results)


def save_results(results: List[Dict], suffix: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Raw results
    raw_path = f"{RESULTS_DIR}/results_{timestamp}_{suffix}.jsonl"
    with open(raw_path, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nSaved {len(results)} results to {raw_path}")


def print_summary(results: List[Dict]):
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for scenario in ["S1", "S2", "S3"]:
        print(f"\n{scenario}:")
        print("-" * 60)
        print(f"{'Method':<12} {'Intensity':<15} {'Trials':>8} {'Violations':>12} {'Rate':>8}")
        print("-" * 60)

        for method in METHODS:
            method_results = [r for r in results
                           if r['scenario'] == scenario and r['method'] == method]

            if not method_results:
                continue

            # Group by intensity
            intensities = {}
            for r in method_results:
                if r['intensity'] not in intensities:
                    intensities[r['intensity']] = []
                intensities[r['intensity']].append(r)

            first = True
            for intensity, int_results in intensities.items():
                violations = sum(1 for r in int_results if r['violated'])
                rate = violations / len(int_results) * 100 if int_results else 0
                method_str = method if first else ""
                first = False
                print(f"{method_str:<12} {intensity:<15} {len(int_results):>8} {violations:>12} {rate:>7.1f}%")

    # Overall by method
    print("\n" + "=" * 70)
    print("OVERALL BY METHOD")
    print("=" * 70)
    print(f"{'Method':<12} {'Total':>8} {'Violations':>12} {'Rate':>10}")
    print("-" * 50)

    for method in METHODS:
        method_results = [r for r in results if r['method'] == method]
        violations = sum(1 for r in method_results if r['violated'])
        rate = violations / len(method_results) * 100 if method_results else 0
        print(f"{method:<12} {len(method_results):>8} {violations:>12} {rate:>9.1f}%")


if __name__ == "__main__":
    main()
