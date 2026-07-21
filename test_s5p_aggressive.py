#!/usr/bin/env python3
"""
S5' Aggressive Attack Test

Try multiple attack variations to find the most effective one:
1. Aggressive goal (3.95m, 5cm from forbidden zone)
2. Stronger attack parameters
3. Runtime attack (start mid-navigation)
4. Combined attack (LIDAR + odometry spoofing)
"""

import subprocess
import time
import os
import sys
import json

WORKSPACE = "/home/jim/ros2_motion_planning_tutorials"
FORBIDDEN_ZONE_X = 4.0

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

def send_nav_goal(x: float, y: float):
    cmd = f'''{source_ros()} && ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose "{{
      pose: {{header: {{frame_id: 'map'}}, pose: {{position: {{x: {x}, y: {y}, z: 0.0}}, orientation: {{w: 1.0}}}}}}
    }}"'''
    return run_bg(cmd, '/tmp/nav_goal_aggressive.log')

def start_lidar_attack(rotation: float, scale: float, noise: float):
    """Start LIDAR spoofing attack"""
    cmd = f"{source_ros()} && ros2 run geofence_policy_enforcer attack_scan_spoofing --ros-args -p rotation_offset:={rotation} -p range_scale:={scale} -p noise_stddev:={noise}"
    return run_bg(cmd, '/tmp/attack_lidar.log')

def start_odom_attack(scale_factor: float):
    """Start odometry spoofing attack - scale_factor < 1.0 makes robot appear closer than actual"""
    cmd = f"{source_ros()} && ros2 run geofence_policy_enforcer attack_odom_spoofing --ros-args -p scale_factor:={scale_factor} -p input_topic:=/odom_real -p output_topic:=/odom"
    return run_bg(cmd, '/tmp/attack_odom.log')

def start_normal_scan_relay():
    """Normal scan relay (no attack)"""
    cmd = f"{source_ros()} && ros2 run geofence_policy_enforcer scan_relay"
    return run_bg(cmd)

def run_variation_test(variation: dict, procs: list) -> dict:
    """
    Run test with specified attack variation.

    Variation dict:
    - name: variation name
    - goal_x: goal X position
    - attack_type: 'none', 'lidar', 'odom', 'combined', 'runtime'
    - lidar_params: {rotation, scale, noise}
    - odom_drift: drift rate for odom attack
    - defense: 'none' or 'geofence'
    """
    name = variation['name']
    goal_x = variation.get('goal_x', 3.8)
    attack_type = variation.get('attack_type', 'lidar')
    defense = variation.get('defense', 'none')

    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"Goal: {goal_x}m | Attack: {attack_type} | Defense: {defense}")
    print(f"{'='*60}")

    # 1. Gazebo
    print("[1] Starting Gazebo...")
    gz_cmd = f"{source_ros()} && ros2 launch mobile_manip_moveit_config mobile_manipulator.launch.py use_sim_time:=true headless:=true"
    gz_proc = run_bg(gz_cmd, f'/tmp/gz_{name}.log')
    procs.append(gz_proc)
    time.sleep(35)

    # 2. Odom relay (or odom attack)
    if attack_type in ['odom', 'combined']:
        print("[2] Starting odom attack...")
        odom_scale = variation.get('odom_scale', 0.7)  # 0.7 = robot appears 30% closer than actual
        odom_proc = start_odom_attack(odom_scale)
    else:
        print("[2] Starting odom relay...")
        odom_cmd = f"{source_ros()} && ros2 run topic_tools relay /odom_real /odom"
        odom_proc = run_bg(odom_cmd)
    procs.append(odom_proc)
    time.sleep(2)

    # 3. Scan (attack or normal) - for non-runtime attacks
    scan_proc = None
    if attack_type == 'runtime':
        # Start with normal relay, will switch later
        print("[3] Starting normal scan relay (attack delayed)...")
        scan_proc = start_normal_scan_relay()
    elif attack_type in ['lidar', 'combined']:
        print("[3] Starting LIDAR spoofing attack...")
        params = variation.get('lidar_params', {'rotation': 0.05, 'scale': 1.08, 'noise': 0.01})
        scan_proc = start_lidar_attack(params['rotation'], params['scale'], params['noise'])
    else:
        print("[3] Starting normal scan relay...")
        scan_proc = start_normal_scan_relay()
    procs.append(scan_proc)
    time.sleep(3)

    # 4. Navigation
    print("[4] Starting Navigation...")
    nav_cmd = f"{source_ros()} && ros2 launch mobile_manip_moveit_config navigation.launch.py use_sim_time:=true rviz:=false"
    nav_proc = run_bg(nav_cmd, f'/tmp/nav_{name}.log')
    procs.append(nav_proc)
    time.sleep(35)

    # 5. Activate Nav2
    print("[5] Activating Nav2 nodes...")
    activate_nav2_nodes()

    # 6. Safety method
    if defense == 'none':
        print("[6] No defense (direct relay)...")
        guard_cmd = f"{source_ros()} && ros2 run topic_tools relay /cmd_vel_nav /cmd_vel"
    else:
        print(f"[6] Starting cmd_vel_guard (method={defense})...")
        guard_cmd = f"{source_ros()} && ros2 run geofence_policy_enforcer cmd_vel_guard_node --ros-args -p safety_method:={defense} -p input_topic:=/cmd_vel_nav -p output_topic:=/cmd_vel"
    guard_proc = run_bg(guard_cmd, f'/tmp/guard_{name}.log')
    procs.append(guard_proc)
    time.sleep(3)

    # 7. Initial pose
    print("[7] Setting initial pose...")
    set_initial_pose(0.0, 0.0, 0.0)
    time.sleep(3)

    # 8. Send goal
    print(f"[8] Sending goal ({goal_x}, 0.0)...")
    goal_proc = send_nav_goal(goal_x, 0.0)
    procs.append(goal_proc)

    # 9. Monitor
    print("[9] Monitoring (60s)...")
    start_time = time.time()
    max_x = 0.0
    violated = False
    runtime_attack_started = False

    while time.time() - start_time < 60:
        x, y = get_robot_position()
        if x is not None:
            max_x = max(max_x, x)
            status = "VIOLATION!" if x >= FORBIDDEN_ZONE_X else ""
            elapsed = time.time() - start_time
            print(f"  t={elapsed:.0f}s: x={x:.2f}m {status}")

            # Runtime attack: start when robot reaches 2.5m
            if attack_type == 'runtime' and not runtime_attack_started and x >= 2.5:
                print(f"  >>> Starting runtime LIDAR attack at x={x:.2f}m!")
                # Kill normal relay
                subprocess.run("pkill -9 -f 'scan_relay'", shell=True, capture_output=True)
                time.sleep(0.5)
                # Start attack
                params = variation.get('lidar_params', {'rotation': 0.1, 'scale': 1.15, 'noise': 0.02})
                attack_proc = start_lidar_attack(params['rotation'], params['scale'], params['noise'])
                procs.append(attack_proc)
                runtime_attack_started = True

            if x >= FORBIDDEN_ZONE_X:
                violated = True
                break
            if x >= goal_x - 0.05:  # Tighter goal check
                print(f"  Goal reached at x={x:.2f}m")
                time.sleep(3)  # Wait to see if it overshoots
                x2, _ = get_robot_position()
                if x2:
                    max_x = max(max_x, x2)
                    if x2 >= FORBIDDEN_ZONE_X:
                        violated = True
                break
        time.sleep(3)

    return {
        'name': name,
        'goal_x': goal_x,
        'attack_type': attack_type,
        'defense': defense,
        'max_x': max_x,
        'violated': violated or max_x >= FORBIDDEN_ZONE_X
    }

def main():
    print("=" * 70)
    print("S5' AGGRESSIVE ATTACK VARIATIONS TEST")
    print("=" * 70)
    print("Testing multiple attack variations to find most effective one")
    print(f"Forbidden zone: x >= {FORBIDDEN_ZONE_X}m")
    print("=" * 70)

    # Define variations to test
    variations = [
        # Variation 1: Baseline - original parameters
        {
            'name': 'v1_baseline',
            'goal_x': 3.8,
            'attack_type': 'lidar',
            'lidar_params': {'rotation': 0.05, 'scale': 1.08, 'noise': 0.01},
            'defense': 'none'
        },
        # Variation 2: Aggressive goal (3.95m)
        {
            'name': 'v2_aggressive_goal',
            'goal_x': 3.95,
            'attack_type': 'lidar',
            'lidar_params': {'rotation': 0.05, 'scale': 1.08, 'noise': 0.01},
            'defense': 'none'
        },
        # Variation 3: Strong attack parameters
        {
            'name': 'v3_strong_attack',
            'goal_x': 3.8,
            'attack_type': 'lidar',
            'lidar_params': {'rotation': 0.12, 'scale': 1.20, 'noise': 0.03},
            'defense': 'none'
        },
        # Variation 4: Aggressive goal + strong attack
        {
            'name': 'v4_aggressive_strong',
            'goal_x': 3.95,
            'attack_type': 'lidar',
            'lidar_params': {'rotation': 0.12, 'scale': 1.20, 'noise': 0.03},
            'defense': 'none'
        },
        # Variation 5: Runtime attack (start at 2.5m)
        {
            'name': 'v5_runtime_attack',
            'goal_x': 3.95,
            'attack_type': 'runtime',
            'lidar_params': {'rotation': 0.15, 'scale': 1.25, 'noise': 0.03},
            'defense': 'none'
        },
        # Variation 6: Combined LIDAR + odom attack
        {
            'name': 'v6_combined_attack',
            'goal_x': 3.95,
            'attack_type': 'combined',
            'lidar_params': {'rotation': 0.10, 'scale': 1.15, 'noise': 0.02},
            'odom_scale': 0.7,  # Robot appears 30% closer than actual
            'defense': 'none'
        },
    ]

    results = []

    for var in variations:
        cleanup()
        procs = []
        try:
            result = run_variation_test(var, procs)
            results.append(result)
            status = "VIOLATED!" if result['violated'] else "safe"
            print(f"\n[RESULT] {result['name']}: Max X={result['max_x']:.2f}m, {status}")
        except Exception as e:
            print(f"[ERROR] {var['name']}: {e}")
            results.append({
                'name': var['name'],
                'goal_x': var.get('goal_x', 3.8),
                'attack_type': var.get('attack_type', 'unknown'),
                'defense': var.get('defense', 'unknown'),
                'max_x': 0.0,
                'violated': False,
                'error': str(e)
            })
        finally:
            cleanup()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY - Attack Variation Results")
    print("=" * 70)
    print(f"{'Variation':<25} {'Goal':>6} {'Attack':<12} {'Max X':>8} {'Status':>10}")
    print("-" * 70)

    for r in results:
        status = "VIOLATED" if r['violated'] else "safe"
        print(f"{r['name']:<25} {r['goal_x']:>5.2f}m {r['attack_type']:<12} {r['max_x']:>7.2f}m {status:>10}")

    print("=" * 70)

    # Find best attack (highest max_x with violation)
    violated_results = [r for r in results if r['violated']]
    if violated_results:
        best = max(violated_results, key=lambda x: x['max_x'])
        print(f"\nBest attack variation: {best['name']}")
        print(f"  Max X: {best['max_x']:.2f}m (violated {FORBIDDEN_ZONE_X}m boundary)")
    else:
        # No violation, find highest max_x
        if results:
            best = max(results, key=lambda x: x['max_x'])
            print(f"\nNo violations achieved. Closest approach: {best['name']}")
            print(f"  Max X: {best['max_x']:.2f}m (boundary at {FORBIDDEN_ZONE_X}m)")

    # Save results
    with open('/tmp/s5p_aggressive_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to /tmp/s5p_aggressive_results.json")

    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted!")
        cleanup()
        sys.exit(1)
