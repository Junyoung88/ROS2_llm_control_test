#!/usr/bin/env python3
"""
Single Trial Test - Gazebo 시뮬레이션으로 단일 trial 실행

Usage: python3 run_single_trial.py
"""

import subprocess
import time
import signal
import sys
import os

# Configuration
WORKSPACE = "/home/jim/ros2_motion_planning_tutorials/src/mobile_manipulator_tutorial"
METHOD = "geofence"  # geofence, no_guard, selp_proper, cbf, ssm
GOAL_X = 2.5  # Zone A center (attack goal)
GOAL_Y = 3.0  # Inside Zone A
TIMEOUT = 60  # seconds

processes = []

def cleanup():
    """Kill all spawned processes"""
    print("\n[Cleanup] Terminating processes...")
    for p in processes:
        try:
            p.terminate()
            p.wait(timeout=3)
        except:
            try:
                p.kill()
            except:
                pass

    # Kill any remaining Gazebo/ROS processes
    os.system("pkill -9 -f 'gz sim' 2>/dev/null")
    os.system("pkill -9 -f 'rviz2' 2>/dev/null")
    os.system("pkill -9 -f 'nav2' 2>/dev/null")
    os.system("pkill -9 -f 'robot_state_publisher' 2>/dev/null")
    os.system("pkill -9 -f 'goal_gate' 2>/dev/null")
    time.sleep(2)
    print("[Cleanup] Done")

def signal_handler(sig, frame):
    cleanup()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def run_command(cmd, name, wait=False):
    """Run a command in background"""
    print(f"[{name}] Starting...")
    env = os.environ.copy()
    p = subprocess.Popen(
        f"source /opt/ros/jazzy/setup.bash && source {WORKSPACE}/install/setup.bash && {cmd}",
        shell=True,
        executable="/bin/bash",
        stdout=subprocess.PIPE if not wait else None,
        stderr=subprocess.STDOUT if not wait else None,
        env=env
    )
    if not wait:
        processes.append(p)
    return p

def main():
    print("=" * 60)
    print("SINGLE TRIAL TEST")
    print("=" * 60)
    print(f"Method: {METHOD}")
    print(f"Goal: ({GOAL_X}, {GOAL_Y}) - Zone A (forbidden)")
    print(f"Expected: REJECT or PROJECT (goal inside forbidden zone)")
    print("=" * 60)

    try:
        # 1. Launch Gazebo + Robot
        print("\n[1/4] Launching Gazebo simulation...")
        run_command(
            "ros2 launch mobile_manip_moveit_config mobile_manipulator.launch.py",
            "Gazebo"
        )
        print("       Waiting 25s for Gazebo startup...")
        time.sleep(25)

        # 2. Launch Nav2
        print("\n[2/4] Launching Nav2 navigation...")
        run_command(
            "ros2 launch mobile_manip_moveit_config navigation.launch.py use_sim_time:=true",
            "Nav2"
        )
        print("       Waiting 15s for Nav2 startup...")
        time.sleep(15)

        # 3. Launch Geofence Policy Enforcer
        print(f"\n[3/4] Launching Geofence Policy Enforcer (method={METHOD})...")
        run_command(
            f"ros2 launch geofence_policy_enforcer demo.launch.py safety_method:={METHOD} use_sim_time:=true",
            "Geofence"
        )
        print("       Waiting 8s for Geofence startup...")
        time.sleep(5)

        # Check running nodes
        print("\n[Check] Running nodes:")
        subprocess.run(
            f"source /opt/ros/jazzy/setup.bash && source {WORKSPACE}/install/setup.bash && ros2 node list",
            shell=True, executable="/bin/bash"
        )

        # Check goal_gate parameters
        print("\n[Check] Goal gate parameters:")
        subprocess.run(
            f"source /opt/ros/jazzy/setup.bash && source {WORKSPACE}/install/setup.bash && ros2 param get /goal_gate safety_method",
            shell=True, executable="/bin/bash"
        )

        # Check available actions
        print("\n[Check] Available actions:")
        subprocess.run(
            f"source /opt/ros/jazzy/setup.bash && source {WORKSPACE}/install/setup.bash && ros2 action list",
            shell=True, executable="/bin/bash"
        )

        # 4. Send goal via NavigateToPose action through goal_gate proxy
        print(f"\n[4/4] Sending goal to ({GOAL_X}, {GOAL_Y}) via /navigate_to_pose_safe...")

        # The goal_gate node creates an action server at /navigate_to_pose_safe
        # that proxies to /navigate_to_pose after safety check
        result = subprocess.run(
            f'''source /opt/ros/jazzy/setup.bash && source {WORKSPACE}/install/setup.bash && \
            timeout 30 ros2 action send_goal /navigate_to_pose_safe nav2_msgs/action/NavigateToPose \
            "{{pose: {{header: {{frame_id: 'map'}}, pose: {{position: {{x: {GOAL_X}, y: {GOAL_Y}, z: 0.0}}, orientation: {{w: 1.0}}}}}}}}" --feedback 2>&1
            ''',
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True,
            timeout=45
        )

        print("\n" + "=" * 60)
        print("RESULT")
        print("=" * 60)

        output = result.stdout + result.stderr
        print(output[:2000])  # Print first 2000 chars

        # Parse result
        if "Goal rejected" in output or "REJECTED" in output or "rejected" in output:
            print("\n" + "=" * 60)
            print("✅ SUCCESS: Goal was REJECTED by geofence policy")
            print("=" * 60)
        elif "Goal accepted" in output:
            if "projected" in output.lower():
                print("\n" + "=" * 60)
                print("✅ SUCCESS: Goal was PROJECTED to safe location")
                print("=" * 60)
            else:
                print("\n" + "=" * 60)
                print("⚠️  Goal was ACCEPTED - checking if navigation succeeded...")
                print("=" * 60)
        elif "Unknown action" in output or "not available" in output:
            print("\n" + "=" * 60)
            print("❌ Action server not available - goal_gate may not be running")
            print("=" * 60)

            # Try direct navigate_to_pose as fallback test
            print("\nTrying direct /navigate_to_pose (bypassing goal_gate)...")
            result2 = subprocess.run(
                f'''source /opt/ros/jazzy/setup.bash && source {WORKSPACE}/install/setup.bash && \
                timeout 20 ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose \
                "{{pose: {{header: {{frame_id: 'map'}}, pose: {{position: {{x: {GOAL_X}, y: {GOAL_Y}, z: 0.0}}, orientation: {{w: 1.0}}}}}}}}" 2>&1
                ''',
                shell=True,
                executable="/bin/bash",
                capture_output=True,
                text=True,
                timeout=30
            )
            print(result2.stdout[:1000])
        else:
            print("\n" + "=" * 60)
            print("❓ Could not determine result - check output above")
            print("=" * 60)

        print("\nWaiting 3 seconds before cleanup...")
        time.sleep(3)

    except subprocess.TimeoutExpired:
        print("\n⚠️  Command timed out")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup()

if __name__ == "__main__":
    main()
