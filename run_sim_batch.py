#!/usr/bin/env python3
"""
Safe Batch Simulation - Run multiple trials with proper cleanup
"""

import subprocess
import time
import os
import sys
import json
import csv
from datetime import datetime

WORKSPACE = "/home/jim/ros2_motion_planning_tutorials/src/mobile_manipulator_tutorial"

# Test configuration
# Zone coordinates from geofence.yaml:
#   Zone A: x=[1.5, 2.5], y=[0.5, 1.5] -> center (2.0, 1.0)
#   Zone B: x=[1.5, 2.5], y=[2.5, 3.5] -> center (2.0, 3.0)
#   Zone C: x=[-4.5, -3.5], y=[0.5, 1.5] -> center (-4.0, 1.0)
#   Zone D: x=[3.0, 4.0], y=[-0.5, 0.5] -> center (3.5, 0.0)
TEST_CASES = [
    # Attack scenarios (inside forbidden zones)
    {"name": "A_center", "x": 2.0, "y": 1.0, "type": "attack", "expected": "REJECT"},
    {"name": "B_center", "x": 2.0, "y": 3.0, "type": "attack", "expected": "REJECT"},

    # Boundary scenarios
    {"name": "A_boundary_0.1m", "x": 1.4, "y": 1.0, "type": "boundary", "expected": "REJECT"},

    # Benign scenarios
    {"name": "safe_origin", "x": 0.0, "y": 0.0, "type": "benign", "expected": "ALLOW"},
]

METHODS = ["no_guard", "selp_proper", "cbf", "ssm", "geofence"]
TRIALS_PER_COMBO = 3  # Start with 3, can increase

def kill_all_ros():
    """Aggressively kill all ROS2/Gazebo processes"""
    patterns = [
        "gz sim", "gzserver", "gzclient", "rviz",
        "goal_gate", "path_watchdog", "cmd_vel_guard", "metrics_logger",
        "nav2", "controller_server", "planner_server", "bt_navigator",
        "amcl", "lifecycle", "robot_state", "ros2.*launch", "ros2.*daemon",
        "image_bridge", "opennav", "slam_toolbox"
    ]
    for pattern in patterns:
        os.system(f"pkill -9 -f '{pattern}' 2>/dev/null")
    os.system("killall -9 ruby 2>/dev/null")
    os.system("rm -rf /dev/shm/fastrtps* 2>/dev/null")
    time.sleep(3)

def check_clean():
    result = subprocess.run(
        "ps aux | grep -E 'ros2|gz|nav2|goal_gate' | grep -v grep | wc -l",
        shell=True, capture_output=True, text=True
    )
    return int(result.stdout.strip()) == 0

def run_single_trial(method: str, goal_x: float, goal_y: float):
    """Run a single simulation trial"""

    result = {
        'method': method,
        'goal_x': goal_x,
        'goal_y': goal_y,
        'decision': None,
        'success': False,
    }

    processes = []
    log_file = "/tmp/geofence_trial.log"

    try:
        # Clean start
        kill_all_ros()
        if not check_clean():
            time.sleep(5)
            kill_all_ros()

        # 1. Gazebo
        gazebo = subprocess.Popen(
            f"source /opt/ros/jazzy/setup.bash && source {WORKSPACE}/install/setup.bash && "
            f"ros2 launch mobile_manip_moveit_config mobile_manipulator.launch.py headless:=true 2>&1",
            shell=True, executable="/bin/bash",
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        processes.append(gazebo)
        time.sleep(25)

        # 2. Nav2
        nav2 = subprocess.Popen(
            f"source /opt/ros/jazzy/setup.bash && source {WORKSPACE}/install/setup.bash && "
            f"ros2 launch mobile_manip_moveit_config navigation.launch.py use_sim_time:=true rviz:=false 2>&1",
            shell=True, executable="/bin/bash",
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        processes.append(nav2)
        time.sleep(15)

        # 3. Geofence
        geofence = subprocess.Popen(
            f"source /opt/ros/jazzy/setup.bash && source {WORKSPACE}/install/setup.bash && "
            f"ros2 launch geofence_policy_enforcer demo.launch.py safety_method:={method} use_sim_time:=true 2>&1 | tee {log_file}",
            shell=True, executable="/bin/bash",
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        processes.append(geofence)
        time.sleep(10)

        # 4. Wait for action server
        for i in range(30):
            check = subprocess.run(
                f"source /opt/ros/jazzy/setup.bash && ros2 action list 2>/dev/null | grep -q navigate_to_pose_safe",
                shell=True, executable="/bin/bash"
            )
            if check.returncode == 0:
                break
            time.sleep(1)
        else:
            return result

        # 5. Send goal
        subprocess.run(
            f'''source /opt/ros/jazzy/setup.bash && source {WORKSPACE}/install/setup.bash && \
            timeout 45 ros2 action send_goal /navigate_to_pose_safe nav2_msgs/action/NavigateToPose \
            "{{pose: {{header: {{frame_id: 'map'}}, pose: {{position: {{x: {goal_x}, y: {goal_y}, z: 0.0}}, orientation: {{w: 1.0}}}}}}}}" 2>&1
            ''',
            shell=True, executable="/bin/bash",
            capture_output=True, text=True, timeout=60
        )

        time.sleep(3)

        # Read log
        try:
            with open(log_file, 'r') as f:
                log_output = f.read()
        except:
            log_output = ""

        # Parse decision
        if "REJECTED goal" in log_output or "Policy decision: REJECT" in log_output:
            result['decision'] = "REJECT"
        elif "PROJECTED goal" in log_output or "Policy decision: PROJECT" in log_output:
            result['decision'] = "PROJECT"
        elif "ALLOWED goal" in log_output or "Policy decision: ALLOW" in log_output:
            result['decision'] = "ALLOW"

        result['success'] = result['decision'] is not None

    except Exception as e:
        pass
    finally:
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

    return result


def main():
    total_trials = len(METHODS) * len(TEST_CASES) * TRIALS_PER_COMBO
    print("=" * 70)
    print("BATCH SIMULATION EXPERIMENT")
    print("=" * 70)
    print(f"Methods: {METHODS}")
    print(f"Test cases: {len(TEST_CASES)}")
    print(f"Trials per combo: {TRIALS_PER_COMBO}")
    print(f"Total trials: {total_trials}")
    print(f"Estimated time: ~{total_trials * 1.5:.0f} minutes")
    print("=" * 70)

    all_results = []
    trial_num = 0

    for method in METHODS:
        print(f"\n[{method}]")

        for tc in TEST_CASES:
            decisions = []

            for trial in range(TRIALS_PER_COMBO):
                trial_num += 1
                print(f"  [{trial_num}/{total_trials}] {tc['name']} trial {trial+1}... ", end="", flush=True)

                result = run_single_trial(method, tc['x'], tc['y'])
                decision = result['decision'] or 'N/A'
                decisions.append(decision)

                # Check vs expected
                match = "✓" if decision == tc['expected'] or (tc['expected'] == 'REJECT' and decision == 'PROJECT') else "✗"
                print(f"{decision} {match}")

                # Record
                all_results.append({
                    'method': method,
                    'test_case': tc['name'],
                    'test_type': tc['type'],
                    'x': tc['x'],
                    'y': tc['y'],
                    'trial': trial + 1,
                    'decision': decision,
                    'expected': tc['expected'],
                    'match': match == "✓"
                })

            # Summary for this test case
            success_rate = sum(1 for d in decisions if d != 'N/A') / len(decisions) * 100
            print(f"    -> {success_rate:.0f}% captured")

    # Save results
    csv_path = "/tmp/sim_batch_results.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nResults saved to: {csv_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for method in METHODS:
        method_results = [r for r in all_results if r['method'] == method]
        success = sum(1 for r in method_results if r['match'])
        total = len(method_results)
        print(f"  {method:<15} {success}/{total} correct ({success/total*100:.0f}%)")

    # Final cleanup
    kill_all_ros()
    print("\n" + "=" * 70)
    subprocess.run("uptime", shell=True)


if __name__ == "__main__":
    main()
