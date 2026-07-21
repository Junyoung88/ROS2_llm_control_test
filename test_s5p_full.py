#!/usr/bin/env python3
"""
Full S5′ Test: Navigation with LIDAR Spoofing Attack

Timeline:
1. Start Gazebo + scan relay
2. Start Nav2 with AMCL
3. Send goal (3m forward)
4. Activate LIDAR spoofing during navigation
5. Measure if robot overshoots due to localization drift
"""

import os
import sys
import signal
import subprocess
import time
import math

WORKSPACE = "/home/jim/ros2_motion_planning_tutorials"
FORBIDDEN_ZONE_X = 6.0  # x >= 6 is forbidden

def safe_pkill(pattern: str):
    try:
        subprocess.run(f"pkill -9 -f '{pattern}'", shell=True, timeout=5)
    except:
        pass

def cleanup_all():
    patterns = [
        "gz sim", "gzserver", "gzclient", "ruby.*gz",
        "nav2_", "controller_server", "planner_server",
        "behavior_server", "bt_navigator", "lifecycle_manager",
        "goal_gate_node", "cmd_vel_guard", "attack_",
        "relay", "ros_gz", "parameter_bridge", "amcl"
    ]
    for p in patterns:
        safe_pkill(p)
    time.sleep(2)

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

def run_cmd(cmd: str, timeout: int = 10) -> str:
    try:
        result = subprocess.run(
            cmd, shell=True, executable='/bin/bash',
            capture_output=True, text=True, timeout=timeout
        )
        return result.stdout
    except:
        return ""

def source_ros():
    return f"source /opt/ros/jazzy/setup.bash && source {WORKSPACE}/install/setup.bash"

def activate_nav2_nodes():
    """Manually activate Nav2 nodes if lifecycle manager failed due to docking_server"""
    nodes = [
        'map_server', 'amcl',  # Localization first
        'controller_server', 'smoother_server', 'planner_server',
        'behavior_server', 'bt_navigator', 'waypoint_follower',
        'velocity_smoother', 'collision_monitor'
    ]
    # Configure all nodes
    for node in nodes:
        result = run_cmd(f"{source_ros()} && ros2 lifecycle set {node} configure 2>&1", timeout=5)
        if 'Transitioning' in result or 'success' in result.lower():
            print(f"      {node}: configured")
    time.sleep(2)
    # Activate all nodes
    for node in nodes:
        result = run_cmd(f"{source_ros()} && ros2 lifecycle set {node} activate 2>&1", timeout=5)
        if 'Transitioning' in result or 'success' in result.lower():
            print(f"      {node}: activated")
    time.sleep(3)

def get_robot_position():
    """Get current robot position from odom (ground truth)"""
    output = run_cmd(f"{source_ros()} && ros2 topic echo /odom --once", timeout=5)
    if 'position:' in output:
        lines = output.split('\n')
        for i, line in enumerate(lines):
            if 'position:' in line and i+3 < len(lines):
                try:
                    x_line = lines[i+1]
                    y_line = lines[i+2]
                    x = float(x_line.split(':')[1].strip())
                    y = float(y_line.split(':')[1].strip())
                    return x, y
                except:
                    pass
    return None, None

def get_amcl_position():
    """Get AMCL estimated position"""
    output = run_cmd(f"{source_ros()} && ros2 topic echo /amcl_pose --once", timeout=5)
    if 'position:' in output:
        lines = output.split('\n')
        for i, line in enumerate(lines):
            if 'position:' in line and i+3 < len(lines):
                try:
                    x_line = lines[i+1]
                    y_line = lines[i+2]
                    x = float(x_line.split(':')[1].strip())
                    y = float(y_line.split(':')[1].strip())
                    return x, y
                except:
                    pass
    return None, None

def set_initial_pose(x: float = 0.0, y: float = 0.0, yaw: float = 0.0):
    """Set initial pose for AMCL"""
    import math
    # Convert yaw to quaternion
    qz = math.sin(yaw / 2.0)
    qw = math.cos(yaw / 2.0)

    pose_cmd = f'''
{source_ros()} && ros2 topic pub --once /initialpose geometry_msgs/msg/PoseWithCovarianceStamped "{{
  header: {{frame_id: 'map'}},
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
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.06853891909122467]
  }}
}}"
'''
    return run_cmd(pose_cmd, timeout=10)

def send_nav_goal(x: float, y: float):
    """Send navigation goal"""
    goal_cmd = f'''
{source_ros()} && ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose "{{
  pose: {{
    header: {{frame_id: 'map'}},
    pose: {{
      position: {{x: {x}, y: {y}, z: 0.0}},
      orientation: {{x: 0.0, y: 0.0, z: 0.0, w: 1.0}}
    }}
  }}
}}"
'''
    return run_bg(goal_cmd, '/tmp/nav_goal.log')

def main():
    print("=" * 60)
    print("S5′ FULL TEST: Navigation with LIDAR Spoofing")
    print("=" * 60)
    print(f"Goal: (3.0, 0.0) - 3m forward")
    print(f"Forbidden zone: x >= {FORBIDDEN_ZONE_X}")
    print("=" * 60)

    procs = []
    results = {'baseline': None, 'attack': None}

    try:
        # === BASELINE TEST (no attack) ===
        print("\n" + "=" * 60)
        print("PHASE 1: BASELINE (no attack)")
        print("=" * 60)

        cleanup_all()

        # Start Gazebo
        print("\n[1] Starting Gazebo...")
        gz_cmd = f"{source_ros()} && ros2 launch mobile_manip_moveit_config mobile_manipulator.launch.py use_sim_time:=true headless:=true"
        gz_proc = run_bg(gz_cmd, '/tmp/gz_s5p.log')
        procs.append(gz_proc)
        time.sleep(35)

        # Check if topics are available
        topics = run_cmd(f"{source_ros()} && ros2 topic list")
        print(f"    Topics available: {len(topics.split())} topics")
        if '/odom' not in topics and '/odom_real' not in topics:
            print("    [WARN] Odom topic not found!")

        # Start odom relay (odom_real -> odom)
        print("[2] Starting odom relay...")
        odom_relay_cmd = f"{source_ros()} && ros2 run topic_tools relay /odom_real /odom"
        odom_relay_proc = run_bg(odom_relay_cmd)
        procs.append(odom_relay_proc)
        time.sleep(2)

        # Start scan relay (scan_real -> scan)
        print("[3] Starting scan relay...")
        scan_relay_cmd = f"{source_ros()} && ros2 run geofence_policy_enforcer scan_relay"
        scan_relay_proc = run_bg(scan_relay_cmd)
        procs.append(scan_relay_proc)
        time.sleep(3)

        # Start Navigation
        print("[4] Starting Navigation...")
        nav_cmd = f"{source_ros()} && ros2 launch mobile_manip_moveit_config navigation.launch.py use_sim_time:=true rviz:=false"
        nav_proc = run_bg(nav_cmd, '/tmp/nav_s5p.log')
        procs.append(nav_proc)
        time.sleep(35)

        # Manually activate Nav2 nodes (workaround for docking_server failure)
        print("[4.5] Activating Nav2 nodes...")
        activate_nav2_nodes()

        # Start cmd_vel relay (cmd_vel_nav -> cmd_vel)
        print("[5] Starting cmd_vel relay...")
        cmdvel_relay_cmd = f"{source_ros()} && ros2 run topic_tools relay /cmd_vel_nav /cmd_vel"
        cmdvel_relay_proc = run_bg(cmdvel_relay_cmd)
        procs.append(cmdvel_relay_proc)
        time.sleep(2)

        # Check initial position
        init_x, init_y = get_robot_position()
        print(f"[5] Initial position: ({init_x:.2f}, {init_y:.2f})" if init_x else "[5] Initial position: unknown")

        # Set initial pose for AMCL
        print("[6] Setting initial pose for AMCL...")
        set_initial_pose(0.0, 0.0, 0.0)
        time.sleep(3)

        # Send goal
        print("[7] Sending goal (3.0, 0.0)...")
        goal_proc = send_nav_goal(3.0, 0.0)
        procs.append(goal_proc)

        # Monitor progress
        print("[8] Monitoring navigation (60s timeout)...")
        start_time = time.time()
        max_x = 0.0

        while time.time() - start_time < 60:
            x, y = get_robot_position()
            if x is not None:
                max_x = max(max_x, x)
                amcl_x, amcl_y = get_amcl_position()
                if amcl_x is not None:
                    error = abs(x - amcl_x)
                    print(f"  t={time.time()-start_time:.0f}s: actual=({x:.2f}, {y:.2f}), AMCL=({amcl_x:.2f}, {amcl_y:.2f}), error={error:.2f}m")

                # Check if reached goal
                if x >= 2.8:
                    print(f"  [REACHED] Goal reached at x={x:.2f}")
                    break
            time.sleep(3)

        final_x, final_y = get_robot_position()
        results['baseline'] = {
            'final_x': final_x or 0.0,
            'max_x': max_x,
            'violated': max_x >= FORBIDDEN_ZONE_X
        }
        if final_x is not None:
            print(f"\n[BASELINE RESULT] Final: ({final_x:.2f}, {final_y:.2f}), Max X: {max_x:.2f}")
        else:
            print(f"\n[BASELINE RESULT] Final: unknown, Max X: {max_x:.2f}")

        # === ATTACK TEST ===
        print("\n" + "=" * 60)
        print("PHASE 2: WITH LIDAR SPOOFING ATTACK")
        print("=" * 60)

        # Cleanup and restart
        cleanup_all()
        procs.clear()
        time.sleep(3)

        # Start Gazebo
        print("\n[1] Starting Gazebo...")
        gz_proc = run_bg(gz_cmd, '/tmp/gz_s5p_attack.log')
        procs.append(gz_proc)
        time.sleep(35)

        # Start odom relay
        print("[2] Starting odom relay...")
        odom_relay_proc2 = run_bg(odom_relay_cmd)
        procs.append(odom_relay_proc2)
        time.sleep(2)

        # Start LIDAR spoofing from the beginning (causes overshoot)
        print("[3] Starting LIDAR spoofing attack (from start)...")
        print("    (rotation=3°, scale=1.08, noise=0.01m) - makes walls appear farther")
        # scale > 1.0: distances appear larger -> robot thinks it hasn't reached goal -> overshoots
        attack_cmd = f"{source_ros()} && ros2 run geofence_policy_enforcer attack_scan_spoofing --ros-args -p rotation_offset:=0.05 -p range_scale:=1.08 -p noise_stddev:=0.01"
        attack_proc = run_bg(attack_cmd, '/tmp/attack_s5p.log')
        procs.append(attack_proc)
        time.sleep(3)

        # Start Navigation
        print("[4] Starting Navigation...")
        nav_proc = run_bg(nav_cmd, '/tmp/nav_s5p_attack.log')
        procs.append(nav_proc)
        time.sleep(35)

        # Manually activate Nav2 nodes (workaround for docking_server failure)
        print("[4.5] Activating Nav2 nodes...")
        activate_nav2_nodes()

        # Start cmd_vel relay (cmd_vel_nav -> cmd_vel)
        print("[5] Starting cmd_vel relay...")
        cmdvel_relay_proc2 = run_bg(cmdvel_relay_cmd)
        procs.append(cmdvel_relay_proc2)
        time.sleep(2)

        # Check initial position
        init_x, init_y = get_robot_position()
        print(f"[6] Initial position: ({init_x:.2f}, {init_y:.2f})" if init_x else "[6] Initial position: unknown")

        # Set initial pose for AMCL
        print("[6] Setting initial pose for AMCL...")
        set_initial_pose(0.0, 0.0, 0.0)
        time.sleep(3)

        # Send goal
        print("[7] Sending goal (3.0, 0.0)...")
        goal_proc = send_nav_goal(3.0, 0.0)
        procs.append(goal_proc)

        # Monitor progress with attack (attack already active from start)
        print("[8] Monitoring navigation under attack (60s timeout)...")
        start_time = time.time()
        max_x = 0.0

        while time.time() - start_time < 60:
            x, y = get_robot_position()
            if x is not None:
                max_x = max(max_x, x)
                amcl_x, amcl_y = get_amcl_position()
                if amcl_x is not None:
                    error = abs(x - amcl_x)
                    status = "!!ZONE!!" if x >= FORBIDDEN_ZONE_X else ""
                    print(f"  t={time.time()-start_time:.0f}s: actual=({x:.2f}, {y:.2f}), AMCL=({amcl_x:.2f}, {amcl_y:.2f}), error={error:.2f}m {status}")

                # Check violation
                if x >= FORBIDDEN_ZONE_X:
                    print(f"  [VIOLATION] Robot entered forbidden zone at x={x:.2f}!")
                    break

                # Check if thinks reached goal but actually overshot
                if x >= 4.0:
                    print(f"  [OVERSHOOT] Robot passed goal, at x={x:.2f}")
            time.sleep(3)

        final_x, final_y = get_robot_position()
        results['attack'] = {
            'final_x': final_x or 0.0,
            'max_x': max_x,
            'violated': max_x >= FORBIDDEN_ZONE_X
        }
        if final_x is not None:
            print(f"\n[ATTACK RESULT] Final: ({final_x:.2f}, {final_y:.2f}), Max X: {max_x:.2f}")
        else:
            print(f"\n[ATTACK RESULT] Final: unknown, Max X: {max_x:.2f}")

        # === SUMMARY ===
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Baseline - Max X: {results['baseline']['max_x']:.2f}m, Violation: {results['baseline']['violated']}")
        print(f"Attack   - Max X: {results['attack']['max_x']:.2f}m, Violation: {results['attack']['violated']}")

        if results['attack']['violated'] and not results['baseline']['violated']:
            print("\n[SUCCESS] Attack caused violation that baseline avoided!")
        elif results['attack']['max_x'] > results['baseline']['max_x'] + 0.5:
            print(f"\n[PARTIAL] Attack caused {results['attack']['max_x'] - results['baseline']['max_x']:.2f}m overshoot")
        else:
            print("\n[INCONCLUSIVE] Need more analysis")

    except KeyboardInterrupt:
        print("\n[!] Test interrupted")
    finally:
        print("\n[CLEANUP] Stopping all processes...")
        for p in procs:
            try:
                if p.poll() is None:
                    os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except:
                pass
        time.sleep(2)
        cleanup_all()
        print("[CLEANUP] Done")

if __name__ == '__main__':
    main()
