#!/usr/bin/env python3
"""
Debug script to test if Nav2 reaches zone_center (5.5, 0) without geofence.
This tests the hypothesis that Nav2 itself is avoiding the zone.
"""

import subprocess
import time
import re
import sys

ZONE = {'x_min': 4.0, 'x_max': 6.0, 'y_min': -1.0, 'y_max': 1.0}
GOAL = (5.5, 0.0)  # Zone center

def get_robot_pose():
    """Get robot pose from Gazebo"""
    try:
        result = subprocess.run(
            ["gz", "topic", "-e", "-n", "1", "-t", "/world/empty/pose/info"],
            capture_output=True, text=True, timeout=3
        )
        match = re.search(
            r'name: "mobile_manip".*?position \{\s*x: ([\d.e+-]+)\s*y: ([\d.e+-]+)',
            result.stdout, re.DOTALL
        )
        if match:
            return float(match.group(1)), float(match.group(2))
    except Exception as e:
        print(f"Error getting pose: {e}")
    return None, None

def is_in_zone(x, y):
    """Check if position is inside forbidden zone"""
    return (ZONE['x_min'] <= x <= ZONE['x_max'] and
            ZONE['y_min'] <= y <= ZONE['y_max'])

def distance_to_zone(x, y):
    """Calculate distance to zone boundary"""
    if is_in_zone(x, y):
        return 0.0
    closest_x = max(ZONE['x_min'], min(x, ZONE['x_max']))
    closest_y = max(ZONE['y_min'], min(y, ZONE['y_max']))
    return ((x - closest_x)**2 + (y - closest_y)**2)**0.5

def main():
    print("=" * 60)
    print("DEBUG: Testing Nav2 navigation to zone_center WITHOUT geofence")
    print("=" * 60)
    print(f"Goal: {GOAL}")
    print(f"Zone: x=[{ZONE['x_min']}, {ZONE['x_max']}], y=[{ZONE['y_min']}, {ZONE['y_max']}]")
    print()

    # Step 1: Start Gazebo
    print("[1] Starting Gazebo...")
    gazebo_cmd = """
    source /opt/ros/jazzy/setup.bash && \
    source install/setup.bash && \
    ros2 launch mobile_manip_moveit_config world.launch.py use_sim_time:=true headless:=true
    """
    gazebo_proc = subprocess.Popen(
        gazebo_cmd, shell=True, executable='/bin/bash',
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    time.sleep(15)

    # Verify Gazebo started
    x, y = get_robot_pose()
    if x is None:
        print("[ERROR] Gazebo failed to start")
        return
    print(f"[OK] Gazebo started. Robot at ({x:.2f}, {y:.2f})")

    # Step 2: Start Nav2 (WITHOUT geofence)
    print("\n[2] Starting Nav2 (no geofence)...")
    nav2_cmd = """
    source /opt/ros/jazzy/setup.bash && \
    source install/setup.bash && \
    ros2 launch mobile_manip_moveit_config navigation.launch.py use_sim_time:=true
    """
    nav2_proc = subprocess.Popen(
        nav2_cmd, shell=True, executable='/bin/bash',
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    time.sleep(20)
    print("[OK] Nav2 started (waiting for initialization...)")
    time.sleep(10)

    # Step 3: Send goal directly to Nav2 (no geofence/goal_gate)
    print(f"\n[3] Sending goal to ({GOAL[0]}, {GOAL[1]}) via /navigate_to_pose...")
    goal_cmd = f"""
    source /opt/ros/jazzy/setup.bash && \
    ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose \
    "{{pose: {{header: {{frame_id: 'map'}}, pose: {{position: {{x: {GOAL[0]}, y: {GOAL[1]}, z: 0.0}}, orientation: {{w: 1.0}}}}}}}}" \
    --feedback &
    """
    subprocess.Popen(goal_cmd, shell=True, executable='/bin/bash',
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Step 4: Monitor robot position
    print("\n[4] Monitoring robot position...")
    print("-" * 60)

    start_time = time.time()
    max_time = 60
    min_dist = float('inf')
    entered_zone = False
    final_x, final_y = 0, 0

    while time.time() - start_time < max_time:
        x, y = get_robot_pose()
        if x is not None:
            dist = distance_to_zone(x, y)
            min_dist = min(min_dist, dist)
            final_x, final_y = x, y

            status = "IN ZONE" if is_in_zone(x, y) else f"dist={dist:.2f}m"
            print(f"t={time.time()-start_time:5.1f}s: x={x:6.2f}, y={y:6.2f}  [{status}]")

            if is_in_zone(x, y):
                entered_zone = True

        time.sleep(1)

    # Step 5: Results
    print("\n" + "=" * 60)
    print("RESULTS:")
    print("=" * 60)
    print(f"Final position: ({final_x:.2f}, {final_y:.2f})")
    print(f"Minimum distance to zone: {min_dist:.2f}m")
    print(f"Entered zone: {entered_zone}")

    if entered_zone:
        print("\n[SUCCESS] Robot entered the zone!")
        print("-> Nav2 does NOT avoid the zone by itself")
    else:
        print(f"\n[PROBLEM] Robot did NOT enter zone (stopped {min_dist:.2f}m away)")
        print("-> This indicates Nav2 or costmap is avoiding the zone area")

    # Cleanup
    print("\n[5] Cleaning up...")
    subprocess.run("pkill -f 'gz sim'", shell=True)
    subprocess.run("pkill -f nav2", shell=True)
    print("[DONE]")

if __name__ == "__main__":
    main()
