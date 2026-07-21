#!/usr/bin/env python3
"""
Quick validation test - just 4 trials to verify system works
"""

import subprocess
import time
import os
import sys

WORKSPACE = "/home/jim/ros2_motion_planning_tutorials/src/mobile_manipulator_tutorial"

def kill_all_ros():
    patterns = ["gz sim", "gzserver", "rviz", "goal_gate", "path_watchdog",
                "cmd_vel_guard", "nav2", "controller_server", "planner_server",
                "bt_navigator", "ros2.*launch", "image_bridge", "opennav"]
    for p in patterns:
        os.system(f"pkill -9 -f '{p}' 2>/dev/null")
    os.system("rm -rf /dev/shm/fastrtps* 2>/dev/null")
    time.sleep(3)

def run_trial(method, goal_x, goal_y):
    log_file = "/tmp/geofence_trial.log"
    processes = []
    decision = None

    try:
        kill_all_ros()

        # Gazebo
        p = subprocess.Popen(
            f"source /opt/ros/jazzy/setup.bash && source {WORKSPACE}/install/setup.bash && "
            f"ros2 launch mobile_manip_moveit_config mobile_manipulator.launch.py headless:=true 2>&1",
            shell=True, executable="/bin/bash", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        processes.append(p)
        time.sleep(25)

        # Nav2
        p = subprocess.Popen(
            f"source /opt/ros/jazzy/setup.bash && source {WORKSPACE}/install/setup.bash && "
            f"ros2 launch mobile_manip_moveit_config navigation.launch.py use_sim_time:=true rviz:=false 2>&1",
            shell=True, executable="/bin/bash", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        processes.append(p)
        time.sleep(15)

        # Geofence
        p = subprocess.Popen(
            f"source /opt/ros/jazzy/setup.bash && source {WORKSPACE}/install/setup.bash && "
            f"ros2 launch geofence_policy_enforcer demo.launch.py safety_method:={method} use_sim_time:=true 2>&1 | tee {log_file}",
            shell=True, executable="/bin/bash", stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        processes.append(p)
        time.sleep(10)

        # Wait for action server
        for _ in range(30):
            if subprocess.run("source /opt/ros/jazzy/setup.bash && ros2 action list 2>/dev/null | grep -q navigate_to_pose_safe",
                            shell=True, executable="/bin/bash").returncode == 0:
                break
            time.sleep(1)

        # Send goal
        subprocess.run(
            f'''source /opt/ros/jazzy/setup.bash && source {WORKSPACE}/install/setup.bash && \
            timeout 45 ros2 action send_goal /navigate_to_pose_safe nav2_msgs/action/NavigateToPose \
            "{{pose: {{header: {{frame_id: 'map'}}, pose: {{position: {{x: {goal_x}, y: {goal_y}, z: 0.0}}, orientation: {{w: 1.0}}}}}}}}" 2>&1''',
            shell=True, executable="/bin/bash", capture_output=True, timeout=60)

        time.sleep(3)

        # Read log
        with open(log_file, 'r') as f:
            log = f.read()

        if "REJECTED" in log:
            decision = "REJECT"
        elif "PROJECTED" in log:
            decision = "PROJECT"
        elif "ALLOWED" in log:
            decision = "ALLOW"

    except Exception as e:
        print(f"Error: {e}")
    finally:
        for p in processes:
            try:
                p.terminate()
                p.wait(timeout=3)
            except:
                p.kill()
        kill_all_ros()

    return decision

def main():
    tests = [
        ("geofence", 2.0, 1.0, "REJECT"),   # Inside Zone A
        ("geofence", 0.0, 0.0, "ALLOW"),    # Safe
        ("no_guard", 2.0, 1.0, "ALLOW"),    # Inside Zone A, no protection
        ("selp_proper", 2.0, 1.0, "REJECT"), # Inside Zone A
    ]

    print("=" * 60)
    print("QUICK VALIDATION TEST (4 trials, ~6 minutes)")
    print("=" * 60)

    results = []
    for i, (method, x, y, expected) in enumerate(tests):
        print(f"\n[{i+1}/4] {method} at ({x}, {y}) [expected: {expected}]...")
        decision = run_trial(method, x, y)
        match = "✓" if decision == expected else "✗"
        print(f"  Result: {decision or 'N/A'} {match}")
        results.append((method, x, y, expected, decision, match))

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"{'Method':<15} {'Goal':<15} {'Expected':<10} {'Actual':<10} {'Match'}")
    print("-" * 60)
    for method, x, y, expected, decision, match in results:
        print(f"{method:<15} ({x}, {y}){'':5} {expected:<10} {decision or 'N/A':<10} {match}")

    success = sum(1 for r in results if r[5] == "✓")
    print(f"\n{success}/4 tests passed")

    kill_all_ros()
    subprocess.run("uptime", shell=True)

if __name__ == "__main__":
    main()
