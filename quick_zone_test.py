#!/usr/bin/env python3
"""
Quick zone violation test - verifies robot enters forbidden zone when no guard is active.
"""
import subprocess
import time
import sys
import signal
import yaml
from pathlib import Path

def run_cmd(cmd, timeout=30):
    """Run command and return output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    except Exception as e:
        return f"ERROR: {e}"

def get_robot_pose():
    """Get current robot pose from AMCL"""
    output = run_cmd("ros2 topic echo /amcl_pose --once", timeout=5)
    try:
        # Parse x, y from output
        lines = output.split('\n')
        x, y = None, None
        for i, line in enumerate(lines):
            if 'x:' in line and x is None:
                x = float(line.split(':')[1].strip())
            elif 'y:' in line and y is None:
                y = float(line.split(':')[1].strip())
            if x is not None and y is not None:
                break
        return x, y
    except:
        return None, None

def is_in_zone_c(x, y):
    """Check if position is in zone C (x: [-4.5, -3.5], y: [-3, 3])"""
    if x is None or y is None:
        return False
    return -4.5 <= x <= -3.5 and -3.0 <= y <= 3.0

def send_nav_goal(x, y):
    """Send navigation goal"""
    cmd = f"""ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose "{{
        pose: {{
            header: {{frame_id: 'map'}},
            pose: {{
                position: {{x: {x}, y: {y}, z: 0.0}},
                orientation: {{w: 1.0}}
            }}
        }}
    }}" --feedback"""
    return subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def main():
    print("=" * 60)
    print("Quick Zone Violation Test")
    print("=" * 60)
    print("\nZone C: x=[-4.5, -3.5], y=[-3.0, 3.0]")
    print("Goal: (-5.0, 0.0) - requires passing through zone C")
    print()

    # Check if simulation is running
    print("[1] Checking simulation status...")
    pose = get_robot_pose()
    if pose[0] is None:
        print("    ERROR: Cannot get robot pose. Is simulation running?")
        print("    Run: ros2 launch mobile_manip_moveit_config mobile_manipulator.launch.py")
        return 1
    print(f"    Robot at: ({pose[0]:.2f}, {pose[1]:.2f})")

    # Reset robot to origin
    print("\n[2] Resetting robot to origin...")
    run_cmd("ros2 service call /set_entity_state gazebo_msgs/srv/SetEntityState \"{state: {name: 'tb3', pose: {position: {x: 0, y: 0, z: 0.1}}}}\"", timeout=5)
    time.sleep(2)

    pose = get_robot_pose()
    print(f"    Robot at: ({pose[0]:.2f}, {pose[1]:.2f})")

    # Send navigation goal through zone C
    print("\n[3] Sending navigation goal to (-5.0, 0.0)...")
    nav_proc = send_nav_goal(-5.0, 0.0)

    # Monitor robot position for zone entry
    print("\n[4] Monitoring robot position (max 60s)...")
    start_time = time.time()
    violation_detected = False
    violation_samples = 0
    max_time = 60

    while time.time() - start_time < max_time:
        x, y = get_robot_pose()
        if x is not None:
            in_zone = is_in_zone_c(x, y)
            status = "IN ZONE C!" if in_zone else ""
            print(f"    [{time.time()-start_time:5.1f}s] x={x:6.2f}, y={y:6.2f} {status}")

            if in_zone:
                violation_detected = True
                violation_samples += 1

        # Check if navigation completed
        if nav_proc.poll() is not None:
            print("    Navigation completed")
            break

        time.sleep(1)

    # Cleanup
    nav_proc.terminate()

    # Report results
    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    if violation_detected:
        print(f"  VIOLATION DETECTED! ({violation_samples} samples in zone)")
        print("  -> Robot entered forbidden zone C")
        return 0
    else:
        print("  NO VIOLATION")
        print("  -> Robot did NOT enter zone C (unexpected for no_guard)")
        return 1

if __name__ == "__main__":
    sys.exit(main())
