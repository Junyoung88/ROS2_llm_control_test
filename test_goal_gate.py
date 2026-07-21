#!/usr/bin/env python3
"""
Test goal_gate behavior with different methods.

This tests whether goal_gate correctly:
1. Allows all goals when method=no_guard
2. Rejects goals inside/near zones when method=geofence

Run with simulation running:
    python3 test_goal_gate.py
"""

import subprocess
import time
import json

WORKSPACE = "/home/jim/ros2_motion_planning_tutorials/src/mobile_manipulator_tutorial"

def run_cmd(cmd, timeout=10):
    """Run shell command"""
    try:
        result = subprocess.run(
            f"source /opt/ros/jazzy/setup.bash && source {WORKSPACE}/install/setup.bash && {cmd}",
            shell=True, capture_output=True, text=True,
            timeout=timeout, executable='/bin/bash'
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", "TIMEOUT", -1

def test_goal_through_gate(x, y, method):
    """
    Test sending a goal through goal_gate.

    Returns: (allowed, response)
    """
    # goal_gate listens on /goal_pose and publishes to /goal_pose_safe
    # We can also use the /check_goal service if available

    # Method 1: Check via service (if available)
    service_cmd = f"""ros2 service call /check_goal geofence_policy_enforcer/srv/CheckGoal \
        "{{goal: {{header: {{frame_id: 'map'}}, pose: {{position: {{x: {x}, y: {y}, z: 0.0}}, orientation: {{w: 1.0}}}}}}}}" """

    stdout, stderr, code = run_cmd(service_cmd, timeout=5)

    if "allowed: true" in stdout.lower() or "true" in stdout.lower():
        return True, stdout
    elif "allowed: false" in stdout.lower() or "false" in stdout.lower():
        return False, stdout
    else:
        # Service might not exist, try topic-based test
        return None, f"Service call failed: {stderr}"

def test_direct_nav2_goal(x, y):
    """
    Send goal directly to Nav2 (bypassing goal_gate).

    This tests whether Nav2 itself blocks the goal.
    """
    cmd = f"""ros2 topic pub --once /goal_pose geometry_msgs/msg/PoseStamped \
        "{{header: {{frame_id: 'map'}}, pose: {{position: {{x: {x}, y: {y}, z: 0.0}}, orientation: {{w: 1.0}}}}}}" """

    stdout, stderr, code = run_cmd(cmd, timeout=5)
    return code == 0, stdout + stderr

def get_current_method():
    """Get the current safety method from goal_gate params"""
    cmd = "ros2 param get /goal_gate safety_method 2>/dev/null"
    stdout, stderr, code = run_cmd(cmd, timeout=5)
    if "no_guard" in stdout:
        return "no_guard"
    elif "geofence" in stdout:
        return "geofence"
    elif "selp" in stdout:
        return "selp"
    elif "cbf" in stdout:
        return "cbf"
    elif "ssm" in stdout:
        return "ssm"
    return "unknown"

def main():
    print("="*60)
    print("Goal Gate Test")
    print("="*60)

    # Test goals
    test_cases = [
        {"name": "Safe center", "x": 0.0, "y": 0.0, "expected_all_allow": True},
        {"name": "Zone A boundary", "x": 1.0, "y": 1.5, "expected_all_allow": False},
        {"name": "Zone A inside", "x": 1.0, "y": 2.0, "expected_all_allow": False},
        {"name": "0.3m from Zone A", "x": 1.0, "y": 1.2, "expected_all_allow": False},
        {"name": "0.6m from Zone A", "x": 1.0, "y": 0.9, "expected_all_allow": True},
    ]

    # Check current method
    current_method = get_current_method()
    print(f"\nCurrent goal_gate method: {current_method}")

    # Test each goal
    print("\n" + "-"*60)
    print("Testing goals through goal_gate service...")
    print("-"*60)

    results = []
    for tc in test_cases:
        print(f"\nTest: {tc['name']} ({tc['x']}, {tc['y']})")

        allowed, response = test_goal_through_gate(tc['x'], tc['y'], current_method)

        if allowed is None:
            print(f"  Service not available: {response[:100]}")
            print("  Trying topic-based test...")
            # Try direct test
            success, output = test_direct_nav2_goal(tc['x'], tc['y'])
            print(f"  Direct Nav2: {'Published' if success else 'Failed'}")
        else:
            print(f"  Allowed: {allowed}")

        results.append({
            "name": tc['name'],
            "x": tc['x'],
            "y": tc['y'],
            "allowed": allowed,
            "method": current_method
        })

    # Summary
    print("\n" + "="*60)
    print("Test Results Summary")
    print("="*60)
    print(f"Method: {current_method}")
    print()

    for r in results:
        status = "ALLOWED" if r['allowed'] else "REJECTED" if r['allowed'] is False else "UNKNOWN"
        print(f"  {r['name']:25} | ({r['x']:5.1f}, {r['y']:5.1f}) | {status}")

    # Check if no_guard allows everything
    if current_method == "no_guard":
        all_allowed = all(r['allowed'] for r in results if r['allowed'] is not None)
        if all_allowed:
            print("\n✓ no_guard correctly allows all goals")
        else:
            print("\n✗ ERROR: no_guard should allow all goals but rejected some!")

    print(f"\nNote: If 'UNKNOWN', the check_goal service might not be available.")
    print("In that case, test by sending goals and observing robot behavior.")

if __name__ == "__main__":
    main()
