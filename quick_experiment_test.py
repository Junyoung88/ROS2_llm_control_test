#!/usr/bin/env python3
"""
Quick experiment test - minimal test to verify zone entry behavior.

This script:
1. Starts simulation with no_guard method
2. Sends a goal inside a zone
3. Monitors if robot actually enters the zone
4. Reports results

Usage:
    python3 quick_experiment_test.py [--method METHOD] [--goal-inside] [--goal-boundary]

Example:
    python3 quick_experiment_test.py --method no_guard --goal-inside
"""

import argparse
import subprocess
import time
import os
import signal
import sys
import json
from pathlib import Path

WORKSPACE = "/home/jim/ros2_motion_planning_tutorials/src/mobile_manipulator_tutorial"

# Zone A: x=[0.5, 1.5], y=[1.5, 2.5]
ZONE_A = {"min_x": 0.5, "max_x": 1.5, "min_y": 1.5, "max_y": 2.5}

TEST_GOALS = {
    "safe": {"x": 0.0, "y": 0.0, "desc": "Safe center"},
    "boundary": {"x": 1.0, "y": 1.5, "desc": "Zone A boundary (y=1.5)"},
    "inside_shallow": {"x": 1.0, "y": 1.6, "desc": "Zone A inside 0.1m"},
    "inside_deep": {"x": 1.0, "y": 2.0, "desc": "Zone A inside 0.5m"},
}

def run_cmd(cmd, timeout=10, capture=True):
    """Run shell command"""
    full_cmd = f"source /opt/ros/jazzy/setup.bash && source {WORKSPACE}/install/setup.bash && {cmd}"
    try:
        if capture:
            result = subprocess.run(
                full_cmd, shell=True, capture_output=True, text=True,
                timeout=timeout, executable='/bin/bash'
            )
            return result.stdout, result.stderr, result.returncode
        else:
            return subprocess.Popen(
                full_cmd, shell=True, executable='/bin/bash',
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                preexec_fn=os.setsid
            )
    except subprocess.TimeoutExpired:
        return "", "TIMEOUT", -1

def check_point_in_zone(x, y):
    """Check if point is inside Zone A"""
    if (ZONE_A["min_x"] <= x <= ZONE_A["max_x"] and
        ZONE_A["min_y"] <= y <= ZONE_A["max_y"]):
        return True
    return False

def distance_to_zone_boundary(x, y):
    """Calculate distance to Zone A boundary (negative if inside)"""
    if check_point_in_zone(x, y):
        dx = min(x - ZONE_A["min_x"], ZONE_A["max_x"] - x)
        dy = min(y - ZONE_A["min_y"], ZONE_A["max_y"] - y)
        return -min(dx, dy)  # Negative inside
    else:
        dx = max(ZONE_A["min_x"] - x, 0, x - ZONE_A["max_x"])
        dy = max(ZONE_A["min_y"] - y, 0, y - ZONE_A["max_y"])
        return (dx**2 + dy**2)**0.5

def get_robot_position():
    """Get robot position from /odom"""
    cmd = "ros2 topic echo /odom --once 2>/dev/null"
    stdout, stderr, code = run_cmd(cmd, timeout=5)

    try:
        lines = stdout.split('\n')
        x = y = None
        for i, line in enumerate(lines):
            if 'position:' in line:
                for j in range(i, min(i+5, len(lines))):
                    if 'x:' in lines[j] and x is None:
                        x = float(lines[j].split(':')[1].strip())
                    elif 'y:' in lines[j] and y is None:
                        y = float(lines[j].split(':')[1].strip())
                break
        return x, y
    except:
        return None, None

def monitor_robot(duration_sec, interval=0.5):
    """Monitor robot position for duration seconds"""
    positions = []
    start = time.time()

    while time.time() - start < duration_sec:
        x, y = get_robot_position()
        if x is not None:
            in_zone = check_point_in_zone(x, y)
            dist = distance_to_zone_boundary(x, y)
            positions.append({
                "time": time.time() - start,
                "x": x, "y": y,
                "in_zone": in_zone,
                "dist_to_boundary": dist
            })
        time.sleep(interval)

    return positions

def send_goal(x, y, use_safe_topic=True):
    """Send navigation goal"""
    topic = "navigate_to_pose_safe" if use_safe_topic else "navigate_to_pose"

    cmd = f"""ros2 action send_goal /{topic} nav2_msgs/action/NavigateToPose \
        "{{pose: {{header: {{frame_id: 'map'}}, pose: {{position: {{x: {x}, y: {y}, z: 0.0}}, orientation: {{w: 1.0}}}}}}}}" \
        --feedback 2>&1"""

    print(f"Sending goal to ({x}, {y}) via /{topic}...")
    return run_cmd(cmd, timeout=60, capture=False)

def test_single_goal(goal_name, goal_data, method, use_safe_topic=True):
    """Test a single goal"""
    print(f"\n{'='*60}")
    print(f"TEST: {goal_name}")
    print(f"Goal: ({goal_data['x']}, {goal_data['y']}) - {goal_data['desc']}")
    print(f"Method: {method}")
    print(f"Use goal_gate: {use_safe_topic}")
    print(f"{'='*60}")

    # Get initial position
    x0, y0 = get_robot_position()
    print(f"Initial position: ({x0:.2f}, {y0:.2f})")

    # Send goal (async)
    goal_proc = send_goal(goal_data['x'], goal_data['y'], use_safe_topic)

    # Monitor robot for 30 seconds
    print("Monitoring robot position for 30 seconds...")
    positions = monitor_robot(30)

    # Analyze results
    entered_zone = any(p["in_zone"] for p in positions)
    min_dist = min(p["dist_to_boundary"] for p in positions) if positions else float('inf')
    max_penetration = -min_dist if min_dist < 0 else 0

    # Get final position
    x_final, y_final = get_robot_position()
    final_in_zone = check_point_in_zone(x_final, y_final) if x_final else False

    print(f"\n--- RESULTS ---")
    print(f"Positions recorded: {len(positions)}")
    print(f"Final position: ({x_final:.2f}, {y_final:.2f})")
    print(f"Entered zone: {entered_zone}")
    print(f"Min distance to boundary: {min_dist:.3f}m")
    print(f"Max penetration: {max_penetration:.3f}m")
    print(f"Final in zone: {final_in_zone}")

    # Kill goal process
    if isinstance(goal_proc, subprocess.Popen):
        try:
            os.killpg(os.getpgid(goal_proc.pid), signal.SIGTERM)
        except:
            pass

    return {
        "goal_name": goal_name,
        "goal_x": goal_data['x'],
        "goal_y": goal_data['y'],
        "method": method,
        "use_goal_gate": use_safe_topic,
        "entered_zone": entered_zone,
        "min_distance": min_dist,
        "max_penetration": max_penetration,
        "final_in_zone": final_in_zone,
        "positions": positions[:10]  # First 10 positions for brevity
    }

def check_simulation_running():
    """Check if simulation is running"""
    stdout, stderr, code = run_cmd("ros2 topic list 2>/dev/null | grep -c /odom", timeout=5)
    return code == 0 and "1" in stdout

def main():
    parser = argparse.ArgumentParser(description='Quick experiment test')
    parser.add_argument('--method', default='no_guard',
                        choices=['no_guard', 'geofence', 'selp', 'cbf', 'ssm'],
                        help='Safety method')
    parser.add_argument('--goal', default='inside_deep',
                        choices=list(TEST_GOALS.keys()),
                        help='Goal to test')
    parser.add_argument('--bypass-gate', action='store_true',
                        help='Send goal directly to Nav2 (bypass goal_gate)')
    parser.add_argument('--all-goals', action='store_true',
                        help='Test all goals')
    args = parser.parse_args()

    print("="*60)
    print("Quick Experiment Test")
    print("="*60)
    print(f"Method: {args.method}")
    print(f"Bypass goal_gate: {args.bypass_gate}")

    # Check simulation
    if not check_simulation_running():
        print("\nERROR: Simulation not running!")
        print("Start simulation first with:")
        print(f"  cd {WORKSPACE}")
        print("  ros2 launch mobile_manip_moveit_config mobile_manipulator.launch.py")
        print("  ros2 launch mobile_manip_moveit_config navigation.launch.py")
        print("  ros2 launch geofence_policy_enforcer demo.launch.py safety_method:=" + args.method)
        return 1

    print("\nSimulation detected. Starting test...")

    # Get current method from goal_gate
    stdout, stderr, code = run_cmd("ros2 param get /goal_gate safety_method 2>/dev/null", timeout=5)
    if code == 0:
        current_method = stdout.strip().split()[-1] if stdout else "unknown"
        print(f"Current goal_gate method: {current_method}")
        if current_method != args.method:
            print(f"WARNING: goal_gate method ({current_method}) != requested ({args.method})")

    results = []

    if args.all_goals:
        for goal_name, goal_data in TEST_GOALS.items():
            result = test_single_goal(goal_name, goal_data, args.method, not args.bypass_gate)
            results.append(result)

            # Reset to origin
            print("\nResetting to origin...")
            send_goal(0.0, 0.0, not args.bypass_gate)
            time.sleep(10)
    else:
        goal_data = TEST_GOALS[args.goal]
        result = test_single_goal(args.goal, goal_data, args.method, not args.bypass_gate)
        results.append(result)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for r in results:
        status = "VIOLATED" if r['entered_zone'] else "SAFE"
        print(f"{r['goal_name']:20} | min_dist: {r['min_distance']:6.3f}m | {status}")

    # Save results
    output_file = "/home/jim/quick_test_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_file}")

    # Verdict
    print("\n" + "="*60)
    print("VERDICT")
    print("="*60)

    inside_goal_results = [r for r in results if "inside" in r['goal_name']]
    if inside_goal_results:
        violated_inside = any(r['entered_zone'] for r in inside_goal_results)
        if args.method == "no_guard":
            if violated_inside:
                print("✓ no_guard correctly allows robot to enter zones")
            else:
                print("✗ ERROR: no_guard should allow zone entry but robot didn't enter!")
                print("  Possible causes:")
                print("  1. Nav2 costmap has zones as obstacles")
                print("  2. Physical obstacles in Gazebo at zone locations")
                print("  3. Goal tolerance preventing full navigation")
        else:
            if not violated_inside:
                print(f"✓ {args.method} correctly prevents zone entry")
            else:
                print(f"✗ WARNING: {args.method} allowed zone entry!")

    return 0

if __name__ == "__main__":
    sys.exit(main())
