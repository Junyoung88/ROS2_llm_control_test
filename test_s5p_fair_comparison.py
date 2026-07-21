#!/usr/bin/env python3
"""
S5' Fair Method Comparison Test

Each method uses its ORIGINAL intended architecture:

- none: No defense (baseline)
- selp: Goal Gate ONLY (planning-time check, no runtime intervention)
- cbf: Cmd Vel Guard with CBF (runtime velocity check)
- ssm: Cmd Vel Guard with SSM (runtime velocity check)
- geofence: Goal Gate + Cmd Vel Guard (full stack, our method)

This is a FAIR comparison where each method operates as originally designed.
"""

import subprocess
import time
import os
import sys

WORKSPACE = "/home/jim/ros2_motion_planning_tutorials"
FORBIDDEN_ZONE_X = 4.0
GOAL_X = 3.8

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

def source_ros():
    return f"source /opt/ros/jazzy/setup.bash && source {WORKSPACE}/install/setup.bash"

def cleanup():
    patterns = [
        "gz sim", "gzserver", "gzclient", "ruby.*gz",
        "nav2_", "controller_server", "planner_server",
        "behavior_server", "bt_navigator", "lifecycle_manager",
        "goal_gate", "cmd_vel_guard", "hardware_geofence",
        "attack_", "relay", "ros_gz", "parameter_bridge", "amcl"
    ]
    for p in patterns:
        subprocess.run(f"pkill -9 -f '{p}'", shell=True, capture_output=True, timeout=5)
    time.sleep(2)

def activate_nav2_nodes():
    nodes = [
        'map_server', 'amcl',
        'controller_server', 'smoother_server', 'planner_server',
        'behavior_server', 'bt_navigator', 'waypoint_follower',
        'velocity_smoother', 'collision_monitor'
    ]
    for node in nodes:
        run_cmd(f"{source_ros()} && ros2 lifecycle set {node} configure 2>&1", timeout=5)
    time.sleep(2)
    for node in nodes:
        run_cmd(f"{source_ros()} && ros2 lifecycle set {node} activate 2>&1", timeout=5)
    time.sleep(3)

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

def get_robot_position():
    output = run_cmd(f"{source_ros()} && ros2 topic echo /odom --once", timeout=5)
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
    return None, None

def send_nav_goal_direct(x: float, y: float):
    """Send goal directly to Nav2 (bypassing any goal gate)"""
    cmd = f'''{source_ros()} && ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose "{{
      pose: {{header: {{frame_id: 'map'}}, pose: {{position: {{x: {x}, y: {y}, z: 0.0}}, orientation: {{w: 1.0}}}}}}
    }}"'''
    return run_bg(cmd, '/tmp/nav_goal_fair.log')

def send_nav_goal_via_gate(x: float, y: float):
    """Send goal through goal gate (for methods that use planning-time check)"""
    cmd = f'''{source_ros()} && ros2 action send_goal /navigate_to_pose_safe nav2_msgs/action/NavigateToPose "{{
      pose: {{header: {{frame_id: 'map'}}, pose: {{position: {{x: {x}, y: {y}, z: 0.0}}, orientation: {{w: 1.0}}}}}}
    }}"'''
    return run_bg(cmd, '/tmp/nav_goal_fair.log')

def run_test(method: str, procs: list) -> dict:
    """
    Run test with method's ORIGINAL intended architecture:

    - none: Nav2 only, no safety
    - selp: Goal Gate (SELP) → Nav2, no runtime check
    - cbf: Nav2 → Cmd Vel Guard (CBF), runtime check only
    - ssm: Nav2 → Cmd Vel Guard (SSM), runtime check only
    - geofence: Goal Gate → Nav2 → Cmd Vel Guard, full stack
    """
    print(f"\n{'='*50}")
    print(f"Testing: {method}")
    print(f"{'='*50}")

    # === COMMON SETUP ===

    # 1. Gazebo
    print("[1] Starting Gazebo...")
    gz_cmd = f"{source_ros()} && ros2 launch mobile_manip_moveit_config mobile_manipulator.launch.py use_sim_time:=true headless:=true"
    gz_proc = run_bg(gz_cmd, f'/tmp/gz_{method}.log')
    procs.append(gz_proc)
    time.sleep(35)

    # 2. Odom relay
    print("[2] Starting odom relay...")
    odom_cmd = f"{source_ros()} && ros2 run topic_tools relay /odom_real /odom"
    odom_proc = run_bg(odom_cmd)
    procs.append(odom_proc)
    time.sleep(2)

    # 3. LIDAR spoofing attack (always on for this test)
    print("[3] Starting LIDAR spoofing attack...")
    scan_cmd = f"{source_ros()} && ros2 run geofence_policy_enforcer attack_scan_spoofing --ros-args -p rotation_offset:=0.05 -p range_scale:=1.08 -p noise_stddev:=0.01"
    scan_proc = run_bg(scan_cmd, f'/tmp/attack_{method}.log')
    procs.append(scan_proc)
    time.sleep(3)

    # 4. Navigation
    print("[4] Starting Navigation...")
    nav_cmd = f"{source_ros()} && ros2 launch mobile_manip_moveit_config navigation.launch.py use_sim_time:=true rviz:=false"
    nav_proc = run_bg(nav_cmd, f'/tmp/nav_{method}.log')
    procs.append(nav_proc)
    time.sleep(35)

    # 5. Activate Nav2
    print("[5] Activating Nav2 nodes...")
    activate_nav2_nodes()

    # === METHOD-SPECIFIC SETUP ===

    use_goal_gate = False

    if method == 'none':
        # No safety - direct relay
        print("[6] No safety (direct relay)...")
        guard_cmd = f"{source_ros()} && ros2 run topic_tools relay /cmd_vel_nav /cmd_vel"
        guard_proc = run_bg(guard_cmd)
        procs.append(guard_proc)

    elif method == 'selp':
        # SELP: Goal Gate ONLY (planning-time), NO runtime cmd_vel check
        print("[6] SELP: Goal Gate only (no runtime check)...")
        # Start goal gate with SELP method
        gate_cmd = f"{source_ros()} && ros2 run geofence_policy_enforcer goal_gate_node --ros-args -p safety_method:=selp"
        gate_proc = run_bg(gate_cmd, f'/tmp/gate_{method}.log')
        procs.append(gate_proc)
        time.sleep(3)
        # Direct relay for cmd_vel (NO runtime safety check)
        guard_cmd = f"{source_ros()} && ros2 run topic_tools relay /cmd_vel_nav /cmd_vel"
        guard_proc = run_bg(guard_cmd)
        procs.append(guard_proc)
        use_goal_gate = True

    elif method == 'cbf':
        # CBF: Runtime cmd_vel check ONLY (no goal gate)
        print("[6] CBF: Cmd Vel Guard only (runtime check)...")
        guard_cmd = f"{source_ros()} && ros2 run geofence_policy_enforcer cmd_vel_guard_node --ros-args -p safety_method:=cbf -p input_topic:=/cmd_vel_nav -p output_topic:=/cmd_vel"
        guard_proc = run_bg(guard_cmd, f'/tmp/guard_{method}.log')
        procs.append(guard_proc)

    elif method == 'ssm':
        # SSM: Runtime cmd_vel check ONLY (no goal gate)
        print("[6] SSM: Cmd Vel Guard only (runtime check)...")
        guard_cmd = f"{source_ros()} && ros2 run geofence_policy_enforcer cmd_vel_guard_node --ros-args -p safety_method:=ssm -p input_topic:=/cmd_vel_nav -p output_topic:=/cmd_vel"
        guard_proc = run_bg(guard_cmd, f'/tmp/guard_{method}.log')
        procs.append(guard_proc)

    elif method == 'geofence':
        # Geofence: FULL STACK (Goal Gate + Cmd Vel Guard)
        print("[6] Geofence: Full stack (Goal Gate + Cmd Vel Guard)...")
        # Start goal gate
        gate_cmd = f"{source_ros()} && ros2 run geofence_policy_enforcer goal_gate_node --ros-args -p safety_method:=geofence"
        gate_proc = run_bg(gate_cmd, f'/tmp/gate_{method}.log')
        procs.append(gate_proc)
        time.sleep(3)
        # Start cmd_vel guard
        guard_cmd = f"{source_ros()} && ros2 run geofence_policy_enforcer cmd_vel_guard_node --ros-args -p safety_method:=geofence -p input_topic:=/cmd_vel_nav -p output_topic:=/cmd_vel"
        guard_proc = run_bg(guard_cmd, f'/tmp/guard_{method}.log')
        procs.append(guard_proc)
        use_goal_gate = True

    time.sleep(3)

    # 7. Initial pose
    print("[7] Setting initial pose...")
    set_initial_pose(0.0, 0.0, 0.0)
    time.sleep(3)

    # 8. Send goal (via gate or direct)
    print(f"[8] Sending goal ({GOAL_X}, 0.0)...")
    if use_goal_gate:
        goal_proc = send_nav_goal_via_gate(GOAL_X, 0.0)
    else:
        goal_proc = send_nav_goal_direct(GOAL_X, 0.0)
    procs.append(goal_proc)

    # 9. Monitor
    print("[9] Monitoring (45s)...")
    start_time = time.time()
    max_x = 0.0
    violated = False

    while time.time() - start_time < 45:
        x, y = get_robot_position()
        if x is not None:
            max_x = max(max_x, x)
            status = "VIOLATION!" if x >= FORBIDDEN_ZONE_X else ""
            print(f"  t={time.time()-start_time:.0f}s: x={x:.2f}m {status}")

            if x >= FORBIDDEN_ZONE_X:
                violated = True
                break
            if x >= GOAL_X - 0.1:
                print(f"  Goal reached!")
                break
        time.sleep(5)

    return {
        'method': method,
        'max_x': max_x,
        'violated': violated or max_x >= FORBIDDEN_ZONE_X
    }

def main():
    print("=" * 60)
    print("S5' FAIR METHOD COMPARISON")
    print("=" * 60)
    print("Each method uses its ORIGINAL intended architecture:")
    print("- none:     No defense")
    print("- selp:     Goal Gate ONLY (planning-time)")
    print("- cbf:      Cmd Vel Guard ONLY (runtime)")
    print("- ssm:      Cmd Vel Guard ONLY (runtime)")
    print("- geofence: Goal Gate + Cmd Vel Guard (full stack)")
    print(f"\nGoal: {GOAL_X}m | Forbidden: x>={FORBIDDEN_ZONE_X}m")
    print("=" * 60)

    methods = ['none', 'selp', 'cbf', 'ssm', 'geofence']
    results = []

    for method in methods:
        cleanup()
        procs = []
        try:
            result = run_test(method, procs)
            results.append(result)
            print(f"\n[RESULT] {method}: Max X={result['max_x']:.2f}m, Violated={result['violated']}")
        except Exception as e:
            print(f"[ERROR] {method}: {e}")
            results.append({'method': method, 'max_x': 0.0, 'violated': False, 'error': str(e)})
        finally:
            cleanup()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY - S5' Attack (LIDAR Spoofing)")
    print("=" * 60)
    print(f"{'Method':<12} {'Architecture':<30} {'Max X':>8} {'Status':>10}")
    print("-" * 60)

    arch_desc = {
        'none': 'No defense',
        'selp': 'Goal Gate only (planning)',
        'cbf': 'Cmd Vel Guard only (runtime)',
        'ssm': 'Cmd Vel Guard only (runtime)',
        'geofence': 'Goal Gate + Cmd Vel Guard'
    }

    for r in results:
        status = "FAILED" if r['violated'] else "PROTECTED"
        arch = arch_desc.get(r['method'], '')
        print(f"{r['method']:<12} {arch:<30} {r['max_x']:>7.2f}m {status:>10}")

    print("=" * 60)

    protected = sum(1 for r in results if not r['violated'])
    failed = sum(1 for r in results if r['violated'])
    print(f"\nProtected: {protected}/{len(results)} | Failed: {failed}/{len(results)}")

    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted!")
        cleanup()
        sys.exit(1)
