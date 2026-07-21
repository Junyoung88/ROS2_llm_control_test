#!/usr/bin/env python3
"""
Test straight-line navigation without obstacles.
Verifies basic Nav2 functionality before running experiments.
"""

import subprocess
import time
import sys

WORKSPACE = "/home/jim/ros2_motion_planning_tutorials"
TIMEOUT = 60  # seconds per goal

def run_cmd(cmd, timeout=30):
    """Run command and return output"""
    try:
        result = subprocess.run(
            cmd, shell=True, executable='/bin/bash',
            capture_output=True, text=True, timeout=timeout
        )
        return result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    except Exception as e:
        return f"ERROR: {e}"

def send_goal(x, y, use_safe=True, timeout=TIMEOUT):
    """Send navigation goal and wait for result"""
    action = "/navigate_to_pose_safe" if use_safe else "/navigate_to_pose"

    cmd = f"""
        source /opt/ros/jazzy/setup.bash && \
        source {WORKSPACE}/install/setup.bash && \
        ros2 action send_goal {action} nav2_msgs/action/NavigateToPose \
            "{{pose: {{header: {{frame_id: 'map'}}, pose: {{position: {{x: {x}, y: {y}, z: 0.0}}, orientation: {{w: 1.0}}}}}}}}" \
            --feedback 2>&1
    """

    print(f"  Sending goal to ({x}, {y})...")
    start = time.time()
    output = run_cmd(cmd, timeout=timeout)
    elapsed = time.time() - start

    # Parse result
    if "TIMEOUT" in output:
        return "timeout", elapsed
    elif "REJECTED goal" in output or "rejected" in output.lower():
        return "reject", elapsed
    elif "SUCCEEDED" in output or "succeeded" in output:
        return "allow", elapsed
    elif "ABORTED" in output:
        if "REJECTED" in output:
            return "reject", elapsed
        return "nav_fail", elapsed
    else:
        return "unknown", elapsed

def reset_robot():
    """Reset robot to origin"""
    cmd = f"""
        source /opt/ros/jazzy/setup.bash && \
        source {WORKSPACE}/install/setup.bash && \
        ros2 service call /set_entity_state ros_gz_interfaces/srv/SetEntityState \
            "{{state: {{name: 'mobile_manipulator', pose: {{position: {{x: 0.0, y: 0.0, z: 0.1}}, orientation: {{w: 1.0}}}}}}}}" 2>&1
    """
    run_cmd(cmd, timeout=10)
    time.sleep(2)

def main():
    print("=" * 60)
    print("Straight-Line Navigation Test")
    print("=" * 60)
    print()

    # Test cases: (x, y, use_safe, expected_result, description)
    tests = [
        # Test 1: Simple navigation to safe location
        (3.0, 0.0, False, "allow", "Direct nav to (3,0) - should succeed"),

        # Test 2: Navigation to zone interior (should be rejected by geofence)
        (5.0, 0.0, True, "reject", "Nav to zone center (5,0) - should reject"),

        # Test 3: Navigation past zone (should succeed)
        (8.0, 0.0, True, "allow", "Nav past zone to (8,0) - should succeed"),

        # Test 4: Navigation to margin area
        (3.6, 0.0, True, "reject", "Nav to margin (3.6,0) - should reject (within 0.55m margin)"),
    ]

    results = []

    for i, (x, y, use_safe, expected, desc) in enumerate(tests, 1):
        print(f"\nTest {i}: {desc}")
        print("-" * 50)

        # Reset robot
        print("  Resetting robot...")
        reset_robot()

        # Send goal
        result, elapsed = send_goal(x, y, use_safe)

        # Check result
        status = "PASS" if result == expected else "FAIL"
        print(f"  Result: {result} (expected: {expected}) - {status}")
        print(f"  Time: {elapsed:.1f}s")

        results.append((desc, result, expected, status))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(1 for _, _, _, s in results if s == "PASS")
    total = len(results)

    for desc, result, expected, status in results:
        print(f"  [{status}] {desc}")
        print(f"        Got: {result}, Expected: {expected}")

    print(f"\nPassed: {passed}/{total}")
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
