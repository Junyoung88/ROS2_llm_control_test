#!/usr/bin/env python3
"""
Test S5' edge case: 3.8m goal with LIDAR spoofing attack.
Goal is only 0.2m from forbidden zone (4.0m).

Expected: Attack causes overshoot into forbidden zone.
"""

import subprocess
import time
import os
import signal
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
        "goal_gate_node", "cmd_vel_guard", "attack_",
        "relay", "ros_gz", "parameter_bridge", "amcl"
    ]
    for p in patterns:
        subprocess.run(f"pkill -9 -f '{p}'", shell=True, capture_output=True, timeout=5)
    time.sleep(2)

def activate_nav2_nodes():
    """Manually activate Nav2 nodes"""
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
        result = run_cmd(f"{source_ros()} && ros2 lifecycle set {node} activate 2>&1", timeout=5)
        if 'Transitioning' in result or 'success' in result.lower():
            print(f"      {node}: activated")
    time.sleep(3)

def set_initial_pose(x=0.0, y=0.0, yaw=0.0):
    import math
    qz = math.sin(yaw / 2.0)
    qw = math.cos(yaw / 2.0)
    cmd = f'''{source_ros()} && ros2 topic pub --once /initialpose geometry_msgs/msg/PoseWithCovarianceStamped "{{
      header: {{frame_id: 'map'}},
      pose: {{
        pose: {{position: {{x: {x}, y: {y}, z: 0.0}}, orientation: {{z: {qz}, w: {qw}}}}},
        covariance: [0.25, 0, 0, 0, 0, 0, 0, 0.25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.07]
      }}
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

def get_amcl_position():
    output = run_cmd(f"{source_ros()} && ros2 topic echo /amcl_pose --once", timeout=5)
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
    return run_bg(cmd, '/tmp/nav_goal_edge.log')

def run_test(use_attack: bool, procs: list):
    """Run single test (baseline or attack)"""

    # 1. Start Gazebo
    print("\n[1] Starting Gazebo...")
    gz_cmd = f"{source_ros()} && ros2 launch mobile_manip_moveit_config mobile_manipulator.launch.py use_sim_time:=true headless:=true"
    gz_proc = run_bg(gz_cmd, '/tmp/gz_edge.log')
    procs.append(gz_proc)
    time.sleep(35)

    # 2. Odom relay
    print("[2] Starting odom relay...")
    odom_cmd = f"{source_ros()} && ros2 run topic_tools relay /odom_real /odom"
    odom_proc = run_bg(odom_cmd)
    procs.append(odom_proc)
    time.sleep(2)

    # 3. Scan relay or attack
    if use_attack:
        print("[3] Starting LIDAR spoofing (scale=1.08 -> walls appear farther)...")
        scan_cmd = f"{source_ros()} && ros2 run geofence_policy_enforcer attack_scan_spoofing --ros-args -p rotation_offset:=0.05 -p range_scale:=1.08 -p noise_stddev:=0.01"
    else:
        print("[3] Starting normal scan relay...")
        scan_cmd = f"{source_ros()} && ros2 run geofence_policy_enforcer scan_relay"
    scan_proc = run_bg(scan_cmd)
    procs.append(scan_proc)
    time.sleep(3)

    # 4. Nav2
    print("[4] Starting Navigation...")
    nav_cmd = f"{source_ros()} && ros2 launch mobile_manip_moveit_config navigation.launch.py use_sim_time:=true rviz:=false"
    nav_proc = run_bg(nav_cmd, '/tmp/nav_edge.log')
    procs.append(nav_proc)
    time.sleep(35)

    # 5. Activate Nav2 nodes
    print("[5] Activating Nav2 nodes...")
    activate_nav2_nodes()

    # 6. cmd_vel relay
    print("[6] Starting cmd_vel relay...")
    cmdvel_cmd = f"{source_ros()} && ros2 run topic_tools relay /cmd_vel_nav /cmd_vel"
    cmdvel_proc = run_bg(cmdvel_cmd)
    procs.append(cmdvel_proc)
    time.sleep(2)

    # 7. Initial pose
    print("[7] Setting initial pose...")
    set_initial_pose(0.0, 0.0, 0.0)
    time.sleep(3)

    # 8. Send goal
    print(f"[8] Sending goal ({GOAL_X}, 0.0)...")
    goal_proc = send_nav_goal(GOAL_X, 0.0)
    procs.append(goal_proc)

    # 9. Monitor
    print("[9] Monitoring (60s timeout)...")
    start_time = time.time()
    max_x = 0.0
    violated = False

    while time.time() - start_time < 60:
        x, y = get_robot_position()
        if x is not None:
            max_x = max(max_x, x)
            amcl_x, amcl_y = get_amcl_position()
            if amcl_x is not None:
                error = abs(x - amcl_x)
                status = "VIOLATION!" if x >= FORBIDDEN_ZONE_X else ""
                print(f"  t={time.time()-start_time:.0f}s: actual=({x:.2f}, {y:.2f}), AMCL=({amcl_x:.2f}), error={error:.2f}m {status}")

            if x >= FORBIDDEN_ZONE_X:
                violated = True
                print(f"  >>> VIOLATION at x={x:.2f}m <<<")
                break

            if x >= GOAL_X - 0.1:
                print(f"  Goal reached at x={x:.2f}m")
                break
        time.sleep(3)

    final_x, final_y = get_robot_position()
    return {
        'final_x': final_x or 0.0,
        'max_x': max_x,
        'violated': violated or max_x >= FORBIDDEN_ZONE_X
    }

def main():
    print("=" * 60)
    print("S5' EDGE CASE TEST")
    print(f"Goal: {GOAL_X}m | Forbidden Zone: x >= {FORBIDDEN_ZONE_X}m | Margin: {FORBIDDEN_ZONE_X - GOAL_X}m")
    print("=" * 60)

    results = {}

    # BASELINE
    print("\n" + "=" * 60)
    print("PHASE 1: BASELINE (no attack)")
    print("=" * 60)
    cleanup()
    procs = []
    try:
        results['baseline'] = run_test(use_attack=False, procs=procs)
        print(f"\n[BASELINE] Max X: {results['baseline']['max_x']:.2f}m, Violated: {results['baseline']['violated']}")
    finally:
        cleanup()

    # ATTACK
    print("\n" + "=" * 60)
    print("PHASE 2: WITH LIDAR SPOOFING (range_scale=1.08)")
    print("=" * 60)
    procs = []
    try:
        results['attack'] = run_test(use_attack=True, procs=procs)
        print(f"\n[ATTACK] Max X: {results['attack']['max_x']:.2f}m, Violated: {results['attack']['violated']}")
    finally:
        cleanup()

    # SUMMARY
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Baseline - Max X: {results['baseline']['max_x']:.2f}m, Violation: {results['baseline']['violated']}")
    print(f"Attack   - Max X: {results['attack']['max_x']:.2f}m, Violation: {results['attack']['violated']}")

    diff = results['attack']['max_x'] - results['baseline']['max_x']
    if results['attack']['violated'] and not results['baseline']['violated']:
        print(f"\n[SUCCESS] Attack caused violation! Baseline stopped safely.")
    elif diff > 0.1:
        print(f"\n[PARTIAL] Attack caused {diff:.2f}m overshoot")
    else:
        print(f"\n[INCONCLUSIVE] Difference: {diff:.2f}m")

    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted!")
        cleanup()
        sys.exit(1)
