#!/usr/bin/env python3
"""
Quick comparison - runs each method and captures key logs
"""

import subprocess
import time
import os
import sys
import re

WORKSPACE = "/home/jim/ros2_motion_planning_tutorials/src/mobile_manipulator_tutorial"
METHODS = ["no_guard", "selp_proper", "cbf", "ssm", "geofence"]
GOAL_X = 2.0
GOAL_Y = 1.0

def cleanup():
    os.system("pkill -9 -f 'gz sim' 2>/dev/null")
    os.system("pkill -9 -f 'rviz' 2>/dev/null")
    os.system("pkill -9 -f 'goal_gate' 2>/dev/null")
    os.system("pkill -9 -f 'nav2' 2>/dev/null")
    os.system("pkill -9 -f 'robot_state' 2>/dev/null")
    time.sleep(3)

def run_method(method):
    """Run single method test"""
    print(f"\n{'='*60}")
    print(f"Testing: {method}")
    print(f"{'='*60}")

    cleanup()

    # Combined launch command with log output
    cmd = f'''
    source /opt/ros/jazzy/setup.bash && source {WORKSPACE}/install/setup.bash && \
    ros2 launch mobile_manip_moveit_config mobile_manipulator.launch.py &
    sleep 25 && \
    ros2 launch mobile_manip_moveit_config navigation.launch.py use_sim_time:=true &
    sleep 15 && \
    ros2 launch geofence_policy_enforcer demo.launch.py safety_method:={method} use_sim_time:=true 2>&1 &
    GEOFENCE_PID=$!
    sleep 8 && \
    ros2 action send_goal /navigate_to_pose_safe nav2_msgs/action/NavigateToPose \
        "{{pose: {{header: {{frame_id: 'map'}}, pose: {{position: {{x: {GOAL_X}, y: {GOAL_Y}, z: 0.0}}, orientation: {{w: 1.0}}}}}}}}" 2>&1
    sleep 5
    '''

    try:
        result = subprocess.run(
            cmd, shell=True, executable="/bin/bash",
            capture_output=True, text=True, timeout=120
        )
        output = result.stdout + result.stderr

        # Parse output
        decision = "UNKNOWN"
        reason = ""
        violations = 0

        if "REJECTED" in output:
            decision = "REJECT"
            match = re.search(r"REJECTED goal.*?:(.*?)(?:\n|$)", output)
            if match:
                reason = match.group(1).strip()[:40]
        elif "PROJECTED" in output:
            decision = "PROJECT"
            match = re.search(r"PROJECTED.*?to \(([\d.]+), ([\d.]+)\)", output)
            if match:
                reason = f"to ({match.group(1)}, {match.group(2)})"
        elif "ALLOWED" in output:
            decision = "ALLOW"

        violations = output.count("VIOLATION: Robot inside forbidden zone")

        # Nav status
        nav_status = "UNKNOWN"
        if "SUCCEEDED" in output:
            nav_status = "SUCCEEDED"
        elif "ABORTED" in output:
            nav_status = "ABORTED"

        return {
            'decision': decision,
            'reason': reason,
            'violations': violations,
            'nav_status': nav_status
        }

    except subprocess.TimeoutExpired:
        return {'decision': 'TIMEOUT', 'reason': '', 'violations': 0, 'nav_status': 'TIMEOUT'}
    except Exception as e:
        return {'decision': 'ERROR', 'reason': str(e)[:30], 'violations': 0, 'nav_status': 'ERROR'}
    finally:
        cleanup()

def main():
    print("="*70)
    print("SAFETY METHOD COMPARISON")
    print(f"Goal: ({GOAL_X}, {GOAL_Y}) - Inside Zone A (forbidden)")
    print("="*70)

    results = {}

    for method in METHODS:
        results[method] = run_method(method)
        r = results[method]
        print(f"Result: {r['decision']}, Violations: {r['violations']}, Nav: {r['nav_status']}")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Method':<15} {'Decision':<10} {'Reason':<30} {'Violations':<10} {'Safe?'}")
    print("-"*80)

    for method in METHODS:
        r = results[method]
        is_safe = "YES" if (r['decision'] == 'REJECT' or (r['decision'] == 'PROJECT')) else "NO"
        if r['decision'] == 'ALLOW' and r['violations'] == 0:
            is_safe = "?"
        print(f"{method:<15} {r['decision']:<10} {r['reason']:<30} {r['violations']:<10} {is_safe}")

    print("="*80)

    print("\nEXPECTED BEHAVIOR:")
    print("  - geofence:    REJECT (point inside forbidden zone)")
    print("  - selp_proper: REJECT or ALLOW (no margin, boundary cases vary)")
    print("  - cbf:         REJECT (0.3m margin)")
    print("  - ssm:         REJECT (0.2m base margin)")
    print("  - no_guard:    ALLOW (no safety check)")

if __name__ == "__main__":
    main()
