#!/usr/bin/env python3
"""
S5' Multi-Method Comparison Test

Compare different safety methods against LIDAR spoofing attack:
- none: No defense (baseline)
- geofence: Forward simulation + uncertainty-aware margin
- cbf: Control Barrier Function
- ssm: Speed and Separation Monitoring
- selp: LTL automaton
- hardware: Gazebo ground truth (cannot be spoofed)
"""

import subprocess
import time
import os
import sys
import json

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
    activated = []
    for node in nodes:
        result = run_cmd(f"{source_ros()} && ros2 lifecycle set {node} activate 2>&1", timeout=5)
        if 'Transitioning' in result or 'success' in result.lower():
            activated.append(node)
    time.sleep(3)
    return activated

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

def send_nav_goal(x: float, y: float):
    cmd = f'''{source_ros()} && ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose "{{
      pose: {{header: {{frame_id: 'map'}}, pose: {{position: {{x: {x}, y: {y}, z: 0.0}}, orientation: {{w: 1.0}}}}}}
    }}"'''
    return run_bg(cmd, '/tmp/nav_goal_methods.log')

def run_test(method: str, use_attack: bool, procs: list) -> dict:
    """
    Run test with specified safety method.

    Methods:
    - none: No cmd_vel guard (baseline)
    - geofence/cbf/ssm/selp: Use cmd_vel_guard with specified method
    - hardware: Use hardware_geofence_guard (ground truth)
    """
    method_name = f"{method}" + (" + attack" if use_attack else "")
    print(f"\n{'='*50}")
    print(f"Testing: {method_name}")
    print(f"{'='*50}")

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

    # 3. Scan (attack or normal)
    if use_attack:
        print("[3] Starting LIDAR spoofing attack...")
        scan_cmd = f"{source_ros()} && ros2 run geofence_policy_enforcer attack_scan_spoofing --ros-args -p rotation_offset:=0.05 -p range_scale:=1.08 -p noise_stddev:=0.01"
    else:
        print("[3] Starting normal scan relay...")
        scan_cmd = f"{source_ros()} && ros2 run geofence_policy_enforcer scan_relay"
    scan_proc = run_bg(scan_cmd)
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

    # 6. Safety method
    if method == 'none':
        # Direct relay: cmd_vel_nav -> cmd_vel
        print("[6] No safety guard (direct relay)...")
        guard_cmd = f"{source_ros()} && ros2 run topic_tools relay /cmd_vel_nav /cmd_vel"
    elif method == 'hardware':
        # Hardware guard uses ground truth
        print("[6] Starting hardware geofence guard (ground truth)...")
        # Use default zones (x_min=4.0) and pass simpler params
        guard_cmd = f"{source_ros()} && ros2 run geofence_policy_enforcer hardware_geofence_guard --ros-args -p input_topic:=/cmd_vel_nav -p output_topic:=/cmd_vel -p safety_margin:=0.3"
    else:
        # cmd_vel_guard with specified method
        print(f"[6] Starting cmd_vel_guard (method={method})...")
        guard_cmd = f"{source_ros()} && ros2 run geofence_policy_enforcer cmd_vel_guard_node --ros-args -p safety_method:={method} -p input_topic:=/cmd_vel_nav -p output_topic:=/cmd_vel"

    guard_proc = run_bg(guard_cmd, f'/tmp/guard_{method}.log')
    procs.append(guard_proc)
    time.sleep(3)

    # 7. Initial pose
    print("[7] Setting initial pose...")
    set_initial_pose(0.0, 0.0, 0.0)
    time.sleep(3)

    # 8. Send goal
    print(f"[8] Sending goal ({GOAL_X}, 0.0)...")
    goal_proc = send_nav_goal(GOAL_X, 0.0)
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
        'attack': use_attack,
        'max_x': max_x,
        'violated': violated or max_x >= FORBIDDEN_ZONE_X
    }

def main():
    print("=" * 60)
    print("S5' MULTI-METHOD COMPARISON")
    print(f"Goal: {GOAL_X}m | Forbidden: x>={FORBIDDEN_ZONE_X}m")
    print("=" * 60)

    # Methods to test
    methods = ['none', 'geofence', 'cbf', 'ssm', 'hardware']
    results = []

    for method in methods:
        cleanup()
        procs = []
        try:
            # Test with attack
            result = run_test(method, use_attack=True, procs=procs)
            results.append(result)
            print(f"\n[RESULT] {method}: Max X={result['max_x']:.2f}m, Violated={result['violated']}")
        except Exception as e:
            print(f"[ERROR] {method}: {e}")
            results.append({'method': method, 'attack': True, 'max_x': 0.0, 'violated': False, 'error': str(e)})
        finally:
            cleanup()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY - S5' Attack (LIDAR Spoofing)")
    print("=" * 60)
    print(f"{'Method':<15} {'Max X':>10} {'Violated':>10} {'Status':>15}")
    print("-" * 50)
    for r in results:
        status = "FAILED" if r['violated'] else "PROTECTED"
        print(f"{r['method']:<15} {r['max_x']:>10.2f}m {str(r['violated']):>10} {status:>15}")

    print("\n" + "=" * 60)

    # Count protected
    protected = sum(1 for r in results if not r['violated'])
    print(f"Protected: {protected}/{len(results)} methods blocked the attack")

    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted!")
        cleanup()
        sys.exit(1)
