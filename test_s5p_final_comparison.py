#!/usr/bin/env python3
"""
S5' Final Comparison Test

Using the best attack parameters found (v3_strong_attack):
- Goal: 3.8m
- rotation_offset: 0.12
- range_scale: 1.20
- noise_stddev: 0.03

Compare: none vs geofence defense
"""

import subprocess
import time
import os
import sys
import json

WORKSPACE = "/home/jim/ros2_motion_planning_tutorials"
FORBIDDEN_ZONE_X = 4.0
GOAL_X = 3.8

# Best attack parameters from aggressive test
ATTACK_PARAMS = {
    'rotation': 0.12,
    'scale': 1.20,
    'noise': 0.03
}

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
    return run_bg(cmd, '/tmp/nav_goal_final.log')

def run_test(defense: str, procs: list) -> dict:
    """Run test with specified defense method."""
    print(f"\n{'='*60}")
    print(f"Testing: {defense} defense")
    print(f"Attack: rotation={ATTACK_PARAMS['rotation']}, scale={ATTACK_PARAMS['scale']}, noise={ATTACK_PARAMS['noise']}")
    print(f"{'='*60}")

    # 1. Gazebo
    print("[1] Starting Gazebo...")
    gz_cmd = f"{source_ros()} && ros2 launch mobile_manip_moveit_config mobile_manipulator.launch.py use_sim_time:=true headless:=true"
    gz_proc = run_bg(gz_cmd, f'/tmp/gz_final_{defense}.log')
    procs.append(gz_proc)
    time.sleep(35)

    # 2. Odom relay
    print("[2] Starting odom relay...")
    odom_cmd = f"{source_ros()} && ros2 run topic_tools relay /odom_real /odom"
    odom_proc = run_bg(odom_cmd)
    procs.append(odom_proc)
    time.sleep(2)

    # 3. LIDAR spoofing attack (strong)
    print("[3] Starting strong LIDAR spoofing attack...")
    scan_cmd = f"{source_ros()} && ros2 run geofence_policy_enforcer attack_scan_spoofing --ros-args -p rotation_offset:={ATTACK_PARAMS['rotation']} -p range_scale:={ATTACK_PARAMS['scale']} -p noise_stddev:={ATTACK_PARAMS['noise']}"
    scan_proc = run_bg(scan_cmd, f'/tmp/attack_final_{defense}.log')
    procs.append(scan_proc)
    time.sleep(3)

    # 4. Navigation
    print("[4] Starting Navigation...")
    nav_cmd = f"{source_ros()} && ros2 launch mobile_manip_moveit_config navigation.launch.py use_sim_time:=true rviz:=false"
    nav_proc = run_bg(nav_cmd, f'/tmp/nav_final_{defense}.log')
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
    guard_proc = run_bg(guard_cmd, f'/tmp/guard_final_{defense}.log')
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

    # 9. Monitor for longer (90s)
    print("[9] Monitoring (90s)...")
    start_time = time.time()
    max_x = 0.0
    violated = False
    positions = []

    while time.time() - start_time < 90:
        x, y = get_robot_position()
        if x is not None:
            max_x = max(max_x, x)
            positions.append((time.time() - start_time, x, y))
            status = "VIOLATION!" if x >= FORBIDDEN_ZONE_X else ""
            print(f"  t={time.time()-start_time:.0f}s: x={x:.2f}m {status}")

            if x >= FORBIDDEN_ZONE_X:
                violated = True
                break
        time.sleep(5)

    return {
        'defense': defense,
        'max_x': max_x,
        'violated': violated or max_x >= FORBIDDEN_ZONE_X,
        'positions': positions
    }

def main():
    print("=" * 70)
    print("S5' FINAL COMPARISON TEST")
    print("=" * 70)
    print(f"Attack: Strong LIDAR spoofing")
    print(f"  rotation_offset: {ATTACK_PARAMS['rotation']} rad (~{ATTACK_PARAMS['rotation']*180/3.14159:.1f}°)")
    print(f"  range_scale: {ATTACK_PARAMS['scale']} ({(ATTACK_PARAMS['scale']-1)*100:.0f}% inflation)")
    print(f"  noise_stddev: {ATTACK_PARAMS['noise']}m")
    print(f"Goal: {GOAL_X}m | Forbidden: x>={FORBIDDEN_ZONE_X}m")
    print("=" * 70)

    defenses = ['none', 'geofence']
    results = []

    for defense in defenses:
        cleanup()
        procs = []
        try:
            result = run_test(defense, procs)
            results.append(result)
            status = "VIOLATED!" if result['violated'] else "PROTECTED"
            print(f"\n[RESULT] {defense}: Max X={result['max_x']:.2f}m, {status}")
        except Exception as e:
            print(f"[ERROR] {defense}: {e}")
            results.append({
                'defense': defense,
                'max_x': 0.0,
                'violated': False,
                'error': str(e)
            })
        finally:
            cleanup()

    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY - S5' Attack (Strong LIDAR Spoofing)")
    print("=" * 70)
    print(f"{'Defense':<15} {'Max X':>10} {'Status':>15}")
    print("-" * 40)

    for r in results:
        status = "VIOLATED" if r['violated'] else "PROTECTED"
        print(f"{r['defense']:<15} {r['max_x']:>9.2f}m {status:>15}")

    print("=" * 70)

    # Analysis
    none_result = next((r for r in results if r['defense'] == 'none'), None)
    geofence_result = next((r for r in results if r['defense'] == 'geofence'), None)

    if none_result and geofence_result:
        if none_result['violated'] and not geofence_result['violated']:
            print("\n✓ SUCCESS: Attack breaks 'none' defense, Geofence protects!")
            print(f"  - none: {none_result['max_x']:.2f}m (violated)")
            print(f"  - geofence: {geofence_result['max_x']:.2f}m (protected)")
        elif not none_result['violated']:
            print("\n⚠ Attack failed to break 'none' defense")
            print("  Consider stronger attack parameters")
        elif geofence_result['violated']:
            print("\n⚠ Geofence also violated!")
            print("  Need to improve geofence parameters")

    # Save results
    with open('/tmp/s5p_final_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to /tmp/s5p_final_results.json")

    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted!")
        cleanup()
        sys.exit(1)
