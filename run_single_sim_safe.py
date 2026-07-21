#!/usr/bin/env python3
"""
Safe Single Simulation Trial - with aggressive cleanup
Run ONE trial at a time to avoid system overload
"""

import subprocess
import time
import signal
import sys
import os
import re

WORKSPACE = "/home/jim/ros2_motion_planning_tutorials/src/mobile_manipulator_tutorial"

def kill_all_ros():
    """Aggressively kill all ROS2/Gazebo processes"""
    patterns = [
        "gz sim", "gzserver", "gzclient",
        "rviz", "goal_gate", "path_watchdog", "cmd_vel_guard",
        "metrics_logger", "nav2", "controller_server", "planner_server",
        "bt_navigator", "amcl", "lifecycle", "robot_state",
        "ros2.*launch", "ros2.*daemon", "image_bridge",
        "opennav", "slam_toolbox"
    ]

    for pattern in patterns:
        os.system(f"pkill -9 -f '{pattern}' 2>/dev/null")

    # Also kill by process name
    os.system("killall -9 ruby 2>/dev/null")  # Gazebo uses ruby

    # Clean shared memory
    os.system("rm -rf /dev/shm/fastrtps* 2>/dev/null")

    time.sleep(3)

def check_clean():
    """Verify no ROS processes running"""
    result = subprocess.run(
        "ps aux | grep -E 'ros2|gz|nav2|goal_gate' | grep -v grep | wc -l",
        shell=True, capture_output=True, text=True
    )
    count = int(result.stdout.strip())
    return count == 0

def run_single_trial(method: str, goal_x: float, goal_y: float, timeout: int = 120):
    """Run a single simulation trial with proper cleanup"""

    print(f"\n{'='*60}")
    print(f"TRIAL: method={method}, goal=({goal_x}, {goal_y})")
    print(f"{'='*60}")

    result = {
        'method': method,
        'goal_x': goal_x,
        'goal_y': goal_y,
        'decision': None,
        'success': False,
        'error': None
    }

    processes = []

    try:
        # 1. Ensure clean start
        print("[0/5] Cleaning up previous processes...")
        kill_all_ros()
        if not check_clean():
            print("  Warning: Some processes still running, forcing cleanup...")
            time.sleep(5)
            kill_all_ros()
        print("  Clean!")

        # 2. Start Gazebo (headless)
        print("[1/5] Starting Gazebo (headless)...")
        gazebo = subprocess.Popen(
            f"source /opt/ros/jazzy/setup.bash && source {WORKSPACE}/install/setup.bash && "
            f"ros2 launch mobile_manip_moveit_config mobile_manipulator.launch.py headless:=true 2>&1",
            shell=True, executable="/bin/bash",
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        processes.append(gazebo)
        time.sleep(25)
        print("  Gazebo started")

        # 3. Start Nav2
        print("[2/5] Starting Nav2...")
        nav2 = subprocess.Popen(
            f"source /opt/ros/jazzy/setup.bash && source {WORKSPACE}/install/setup.bash && "
            f"ros2 launch mobile_manip_moveit_config navigation.launch.py use_sim_time:=true rviz:=false 2>&1",
            shell=True, executable="/bin/bash",
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        processes.append(nav2)
        time.sleep(15)
        print("  Nav2 started")

        # 4. Start Geofence with log file
        log_file = "/tmp/geofence_trial.log"
        print(f"[3/5] Starting Geofence (method={method})...")
        geofence = subprocess.Popen(
            f"source /opt/ros/jazzy/setup.bash && source {WORKSPACE}/install/setup.bash && "
            f"ros2 launch geofence_policy_enforcer demo.launch.py safety_method:={method} use_sim_time:=true 2>&1 | tee {log_file}",
            shell=True, executable="/bin/bash",
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        processes.append(geofence)
        time.sleep(10)
        print("  Geofence started")

        # 5. Wait for action server
        print("[4/5] Waiting for action server...")
        for i in range(30):
            check = subprocess.run(
                f"source /opt/ros/jazzy/setup.bash && ros2 action list 2>/dev/null | grep -q navigate_to_pose_safe",
                shell=True, executable="/bin/bash"
            )
            if check.returncode == 0:
                print("  Action server ready!")
                break
            time.sleep(1)
        else:
            result['error'] = "Action server timeout"
            return result

        # 6. Send goal and capture output
        print(f"[5/5] Sending goal to ({goal_x}, {goal_y})...")

        goal_cmd = subprocess.run(
            f'''source /opt/ros/jazzy/setup.bash && source {WORKSPACE}/install/setup.bash && \
            timeout 45 ros2 action send_goal /navigate_to_pose_safe nav2_msgs/action/NavigateToPose \
            "{{pose: {{header: {{frame_id: 'map'}}, pose: {{position: {{x: {goal_x}, y: {goal_y}, z: 0.0}}, orientation: {{w: 1.0}}}}}}}}" 2>&1
            ''',
            shell=True, executable="/bin/bash",
            capture_output=True, text=True, timeout=60
        )

        # Read geofence logs from file
        time.sleep(3)

        # Read log file
        try:
            with open(log_file, 'r') as f:
                log_output = f.read()
        except:
            log_output = ""

        # Parse decision from logs - look for goal_gate decision
        if "REJECTED goal" in log_output or "Policy decision: REJECT" in log_output:
            result['decision'] = "REJECT"
        elif "PROJECTED goal" in log_output or "Policy decision: PROJECT" in log_output:
            result['decision'] = "PROJECT"
        elif "ALLOWED goal" in log_output or "Policy decision: ALLOW" in log_output:
            result['decision'] = "ALLOW"

        # Also check goal command output for clues
        if result['decision'] is None:
            if "Goal rejected" in goal_cmd.stdout or "aborted" in goal_cmd.stdout.lower():
                result['decision'] = "REJECT"
            elif "Goal accepted" in goal_cmd.stdout and "succeeded" in goal_cmd.stdout.lower():
                result['decision'] = "ALLOW"

        # Debug: show relevant log lines
        print(f"  Log search: ", end="")
        if "REJECTED" in log_output:
            print("found REJECTED")
        elif "PROJECTED" in log_output:
            print("found PROJECTED")
        elif "ALLOWED" in log_output:
            print("found ALLOWED")
        else:
            print("no decision keyword found")

        result['success'] = result['decision'] is not None
        print(f"  Decision: {result['decision'] or 'N/A'}")

    except subprocess.TimeoutExpired:
        result['error'] = "Timeout"
        print("  ERROR: Timeout!")
    except Exception as e:
        result['error'] = str(e)
        print(f"  ERROR: {e}")
    finally:
        # Aggressive cleanup
        print("\n[Cleanup] Terminating all processes...")
        for p in processes:
            try:
                p.terminate()
                p.wait(timeout=3)
            except:
                try:
                    p.kill()
                except:
                    pass

        kill_all_ros()

        # Verify cleanup
        if check_clean():
            print("[Cleanup] Done - system clean")
        else:
            print("[Cleanup] Warning: some processes may remain")
            kill_all_ros()

    return result


def main():
    # Example: run single trial
    if len(sys.argv) >= 4:
        method = sys.argv[1]
        goal_x = float(sys.argv[2])
        goal_y = float(sys.argv[3])
    else:
        # Default test
        method = "geofence"
        goal_x = 2.0
        goal_y = 1.0

    result = run_single_trial(method, goal_x, goal_y)

    print(f"\n{'='*60}")
    print("RESULT:")
    print(f"{'='*60}")
    print(f"  Method: {result['method']}")
    print(f"  Goal: ({result['goal_x']}, {result['goal_y']})")
    print(f"  Decision: {result['decision'] or 'N/A'}")
    print(f"  Success: {result['success']}")
    if result['error']:
        print(f"  Error: {result['error']}")
    print(f"{'='*60}")

    # Final cleanup check
    time.sleep(3)
    subprocess.run("uptime", shell=True)

    return 0 if result['success'] else 1


if __name__ == "__main__":
    sys.exit(main())
