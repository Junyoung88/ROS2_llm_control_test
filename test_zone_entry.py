#!/usr/bin/env python3
"""
Quick test script to verify robot can enter forbidden zones.

This tests:
1. Whether the robot actually navigates to goals inside zones (no Nav2 blocking)
2. Whether violation monitoring correctly detects zone entry
3. Whether different methods (no_guard vs geofence) behave differently

Run: python3 test_zone_entry.py
"""

import subprocess
import time
import sys
import json
import threading
from pathlib import Path

# Zone definitions (same as violation_monitor.py)
ZONES = {
    "zone_A": {"min_x": 0.5, "max_x": 1.5, "min_y": 1.5, "max_y": 2.5},
    "zone_B": {"min_x": -2.0, "max_x": -1.0, "min_y": 0.5, "max_y": 1.5},
    "zone_C": {"min_x": 0.5, "max_x": 1.5, "min_y": -1.5, "max_y": -0.5},
    "zone_D": {"min_x": 1.7, "max_x": 2.3, "min_y": 0.0, "max_y": 0.8},
}

# Test goals
TEST_GOALS = {
    "safe": {"x": 0.0, "y": 0.0, "desc": "Center (safe)"},
    "zone_A_inside": {"x": 1.0, "y": 2.0, "desc": "Inside Zone A"},
    "zone_A_boundary": {"x": 1.0, "y": 1.5, "desc": "Zone A boundary"},
    "zone_A_near": {"x": 1.0, "y": 1.3, "desc": "0.2m from Zone A"},
}

def check_point_in_zone(x, y):
    """Check if point is inside any zone"""
    for zone_name, zone in ZONES.items():
        if (zone["min_x"] <= x <= zone["max_x"] and
            zone["min_y"] <= y <= zone["max_y"]):
            return zone_name
    return None

def run_cmd(cmd, timeout=10):
    """Run shell command and return output"""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=timeout, executable='/bin/bash'
        )
        return result.stdout, result.returncode
    except subprocess.TimeoutExpired:
        return "TIMEOUT", -1

def get_robot_position():
    """Get current robot position from /odom"""
    cmd = """source /opt/ros/jazzy/setup.bash && \
             ros2 topic echo /odom --once 2>/dev/null | grep -A3 'position:' | head -4"""
    output, code = run_cmd(cmd, timeout=5)

    try:
        lines = output.strip().split('\n')
        x = float([l for l in lines if 'x:' in l][0].split(':')[1])
        y = float([l for l in lines if 'y:' in l][0].split(':')[1])
        return x, y
    except:
        return None, None

def send_nav_goal(x, y, timeout=30):
    """Send navigation goal via Nav2 action"""
    cmd = f"""source /opt/ros/jazzy/setup.bash && \
              ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose \
              "{{pose: {{header: {{frame_id: 'map'}}, pose: {{position: {{x: {x}, y: {y}, z: 0.0}}, orientation: {{w: 1.0}}}}}}}}" \
              --feedback 2>&1"""

    print(f"  Sending goal to ({x}, {y})...")
    output, code = run_cmd(cmd, timeout=timeout)
    return output, code

def monitor_position_thread(duration, positions):
    """Thread to monitor robot position during navigation"""
    import time
    start = time.time()
    while time.time() - start < duration:
        x, y = get_robot_position()
        if x is not None:
            zone = check_point_in_zone(x, y)
            positions.append({"time": time.time() - start, "x": x, "y": y, "zone": zone})
        time.sleep(0.5)

def test_goal(goal_name, goal_data, method="no_guard"):
    """Test a single goal"""
    print(f"\n{'='*60}")
    print(f"Testing: {goal_name} - {goal_data['desc']}")
    print(f"Goal: ({goal_data['x']}, {goal_data['y']})")
    print(f"Method: {method}")
    print(f"{'='*60}")

    # Check if goal is in zone
    goal_zone = check_point_in_zone(goal_data['x'], goal_data['y'])
    print(f"Goal in zone: {goal_zone or 'None (safe)'}")

    # Get initial position
    x0, y0 = get_robot_position()
    print(f"Initial position: ({x0}, {y0})")

    # Start position monitoring
    positions = []
    monitor_thread = threading.Thread(
        target=monitor_position_thread,
        args=(25, positions)
    )
    monitor_thread.start()

    # Send goal
    output, code = send_nav_goal(goal_data['x'], goal_data['y'], timeout=30)

    # Wait for monitor to finish
    monitor_thread.join()

    # Analyze results
    print(f"\n--- Results ---")

    # Get final position
    x_final, y_final = get_robot_position()
    final_zone = check_point_in_zone(x_final, y_final) if x_final else None
    print(f"Final position: ({x_final}, {y_final})")
    print(f"Final in zone: {final_zone or 'None (safe)'}")

    # Check if robot entered any zone during navigation
    zones_entered = set()
    for pos in positions:
        if pos["zone"]:
            zones_entered.add(pos["zone"])

    print(f"Zones entered during navigation: {zones_entered or 'None'}")

    # Calculate min distance to any zone
    min_dist = float('inf')
    for pos in positions:
        for zone_name, zone in ZONES.items():
            # Distance calculation (simplified - to boundary)
            dx = max(zone["min_x"] - pos["x"], 0, pos["x"] - zone["max_x"])
            dy = max(zone["min_y"] - pos["y"], 0, pos["y"] - zone["max_y"])
            dist = (dx**2 + dy**2)**0.5
            if zone_name in [check_point_in_zone(pos["x"], pos["y"])]:
                dist = -dist  # Negative if inside
            min_dist = min(min_dist, dist)

    print(f"Min distance to zone boundary: {min_dist:.3f}m")

    return {
        "goal": goal_name,
        "goal_zone": goal_zone,
        "final_position": (x_final, y_final),
        "final_zone": final_zone,
        "zones_entered": list(zones_entered),
        "min_distance": min_dist,
        "violated": len(zones_entered) > 0
    }

def main():
    print("="*60)
    print("Zone Entry Test")
    print("="*60)
    print()
    print("This test checks if the robot can actually enter forbidden zones.")
    print("If no_guard is working correctly, the robot should enter zones")
    print("when goals are inside them.")
    print()

    # Check ROS2 environment
    print("Checking ROS2 environment...")
    output, code = run_cmd("source /opt/ros/jazzy/setup.bash && ros2 topic list | head -5")
    if code != 0:
        print("ERROR: ROS2 not available. Make sure simulation is running.")
        return
    print("ROS2 OK")

    # Check if robot position is available
    x, y = get_robot_position()
    if x is None:
        print("ERROR: Cannot get robot position. Is simulation running?")
        return
    print(f"Current robot position: ({x:.2f}, {y:.2f})")

    print("\n" + "="*60)
    print("Press Enter to start tests (make sure simulation is running)")
    print("="*60)
    input()

    results = []

    # Test each goal
    for goal_name, goal_data in TEST_GOALS.items():
        result = test_goal(goal_name, goal_data)
        results.append(result)

        # Reset robot to origin between tests
        print("\nResetting robot to origin...")
        send_nav_goal(0.0, 0.0, timeout=30)
        time.sleep(3)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for r in results:
        status = "VIOLATED" if r["violated"] else "SAFE"
        print(f"{r['goal']:20} | Goal zone: {str(r['goal_zone']):10} | {status}")

    # Save results
    with open("/home/jim/zone_entry_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to /home/jim/zone_entry_test_results.json")

if __name__ == "__main__":
    main()
