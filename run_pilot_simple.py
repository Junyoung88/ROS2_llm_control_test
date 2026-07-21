#!/usr/bin/env python3
"""
Simple Pilot Test - File-based logging for reliability
"""

import subprocess
import time
import os
import json
from datetime import datetime
from pathlib import Path

WORKSPACE = "/home/jim/ros2_motion_planning_tutorials/src/mobile_manipulator_tutorial"
LOG_FILE = "/tmp/geofence_trial.log"

METHODS = ["no_guard", "selp_proper", "cbf", "ssm", "geofence"]

TEST_CASES = [
    ("A1_inside", 2.0, 1.0),
    ("A2_boundary", 1.4, 1.0),
    ("B1_safe", 0.0, 0.0),
]

EXPECTED = {
    "no_guard": {"A1_inside": "allow", "A2_boundary": "allow", "B1_safe": "allow"},
    "selp_proper": {"A1_inside": "reject", "A2_boundary": "allow", "B1_safe": "allow"},
    "cbf": {"A1_inside": "reject", "A2_boundary": "reject", "B1_safe": "allow"},
    "ssm": {"A1_inside": "reject", "A2_boundary": "reject", "B1_safe": "allow"},
    "geofence": {"A1_inside": "reject", "A2_boundary": "project", "B1_safe": "allow"},
}

def cleanup():
    os.system("pkill -9 -f 'gz sim' 2>/dev/null")
    os.system("pkill -9 -f 'rviz' 2>/dev/null")
    os.system("pkill -9 -f 'goal_gate' 2>/dev/null")
    os.system("pkill -9 -f 'nav2' 2>/dev/null")
    os.system("pkill -9 -f 'robot_state' 2>/dev/null")
    time.sleep(3)

def run_trial(method, scenario, goal_x, goal_y):
    cleanup()

    # Remove old log
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)

    result = {
        'method': method,
        'scenario': scenario,
        'goal': (goal_x, goal_y),
        'expected': EXPECTED[method][scenario],
        'decision': None,
        'correct': False
    }

    try:
        # Launch all in one script, redirect geofence output to file
        cmd = f'''
        source /opt/ros/jazzy/setup.bash && source {WORKSPACE}/install/setup.bash

        # Start Gazebo
        ros2 launch mobile_manip_moveit_config mobile_manipulator.launch.py headless:=true &
        sleep 20

        # Start Nav2
        ros2 launch mobile_manip_moveit_config navigation.launch.py use_sim_time:=true rviz:=false &
        sleep 12

        # Start Geofence with logging to file
        ros2 launch geofence_policy_enforcer demo.launch.py safety_method:={method} use_sim_time:=true 2>&1 | tee {LOG_FILE} &
        sleep 6

        # Send goal
        ros2 action send_goal /navigate_to_pose_safe nav2_msgs/action/NavigateToPose \
            "{{pose: {{header: {{frame_id: 'map'}}, pose: {{position: {{x: {goal_x}, y: {goal_y}, z: 0.0}}, orientation: {{w: 1.0}}}}}}}}"

        sleep 5
        '''

        subprocess.run(cmd, shell=True, executable="/bin/bash", timeout=70)

        # Read log file
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE) as f:
                log = f.read()

            if "REJECTED" in log:
                result['decision'] = "reject"
            elif "PROJECTED" in log:
                result['decision'] = "project"
            elif "ALLOWED" in log:
                result['decision'] = "allow"

        # Check correctness
        exp = result['expected']
        dec = result['decision']
        if exp == "reject":
            result['correct'] = dec in ["reject", "project"]
        elif exp == "project":
            result['correct'] = dec in ["reject", "project"]
        else:
            result['correct'] = dec == "allow"

    except subprocess.TimeoutExpired:
        result['decision'] = "TIMEOUT"
    except Exception as e:
        result['error'] = str(e)[:30]
    finally:
        cleanup()

    return result

def main():
    print("="*60)
    print("SIMPLE PILOT TEST")
    print("="*60)

    results = []
    total = len(METHODS) * len(TEST_CASES)
    trial_num = 0

    for method in METHODS:
        print(f"\n[{method}]")
        for scenario, gx, gy in TEST_CASES:
            trial_num += 1
            exp = EXPECTED[method][scenario]
            print(f"  [{trial_num}/{total}] {scenario} (exp:{exp})...", end=" ", flush=True)

            r = run_trial(method, scenario, gx, gy)
            results.append(r)

            dec = r['decision'] or 'N/A'
            status = "OK" if r['correct'] else "FAIL"
            print(f"{dec} [{status}]")

    # Summary
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)

    for method in METHODS:
        mr = [r for r in results if r['method'] == method]
        correct = sum(1 for r in mr if r['correct'])
        print(f"  {method:<15} {correct}/{len(mr)}")

    total_correct = sum(1 for r in results if r['correct'])
    print(f"\nTotal: {total_correct}/{total}")

    if total_correct == total:
        print("\n[PASS] Ready for full experiment!")
    else:
        print(f"\n[WARN] {total - total_correct} failures")

if __name__ == "__main__":
    main()
