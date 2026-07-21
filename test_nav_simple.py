#!/usr/bin/env python3
"""Simple navigation test to verify simulation is working."""

import subprocess
import time
import sys

WORKSPACE = "/home/jim/ros2_motion_planning_tutorials"

def run_cmd(cmd, timeout=30):
    """Run command and return (success, output)"""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout,
            executable='/bin/bash'
        )
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, "Timeout"

def main():
    print("=" * 60)
    print("Simple Navigation Test")
    print("=" * 60)

    # Start simulation
    print("\n[1/4] Starting Gazebo simulation...")
    sim_cmd = f"""
        source /opt/ros/jazzy/setup.bash &&
        source {WORKSPACE}/install/setup.bash &&
        ros2 launch mobile_manip_moveit_config mobile_manipulator.launch.py \
            use_sim_time:=true headless:=true
    """
    sim_proc = subprocess.Popen(sim_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, executable='/bin/bash')
    time.sleep(20)  # Wait for Gazebo

    # Check if simulation is running
    check = subprocess.run("pgrep -f 'gz sim'", shell=True)
    if check.returncode != 0:
        print("ERROR: Gazebo failed to start!")
        return 1
    print("  Gazebo running.")

    # Start odom relay
    print("\n[2/4] Starting odom relay...")
    odom_cmd = f"""
        source /opt/ros/jazzy/setup.bash &&
        source {WORKSPACE}/install/setup.bash &&
        ros2 run topic_tools relay /odom_real /odom
    """
    odom_proc = subprocess.Popen(odom_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, executable='/bin/bash')
    time.sleep(2)
    print("  Odom relay started.")

    # Start navigation
    print("\n[3/4] Starting Nav2...")
    nav_cmd = f"""
        source /opt/ros/jazzy/setup.bash &&
        source {WORKSPACE}/install/setup.bash &&
        ros2 launch mobile_manip_moveit_config navigation.launch.py \
            use_sim_time:=true use_amcl:=true rviz:=false
    """
    nav_proc = subprocess.Popen(nav_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, executable='/bin/bash')
    time.sleep(25)  # Wait for Nav2 nodes to be ready

    # Check if nav2 is running
    check = subprocess.run("pgrep -f 'bt_navigator'", shell=True)
    if check.returncode != 0:
        print("ERROR: Nav2 failed to start!")
        return 1
    print("  Nav2 running.")

    # Send a simple navigation goal (2m forward - safe distance)
    print("\n[4/4] Sending navigation goal: (2.0, 0.0)...")
    goal_cmd = f"""
        source /opt/ros/jazzy/setup.bash &&
        source {WORKSPACE}/install/setup.bash &&
        ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose \
            "{{pose: {{header: {{frame_id: 'map'}}, pose: {{position: {{x: 2.0, y: 0.0, z: 0.0}}, orientation: {{w: 1.0}}}}}}}}" \
            --feedback
    """

    start_time = time.time()
    goal_proc = subprocess.Popen(goal_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, executable='/bin/bash')

    # Wait for goal to complete (max 60s)
    try:
        stdout, stderr = goal_proc.communicate(timeout=60)
        elapsed = time.time() - start_time
        output = stdout.decode() + stderr.decode()

        if "Goal accepted" in output and ("succeeded" in output.lower() or "result:" in output.lower()):
            print(f"\n  SUCCESS! Navigation completed in {elapsed:.1f}s")
            result = 0
        elif "Goal accepted" in output:
            print(f"\n  PARTIAL: Goal accepted but status unclear ({elapsed:.1f}s)")
            result = 0
        elif "rejected" in output.lower():
            print(f"\n  FAILED: Goal rejected")
            result = 1
        else:
            print(f"\n  UNKNOWN: {output[:200]}")
            result = 1

    except subprocess.TimeoutExpired:
        print(f"\n  TIMEOUT: Navigation took >60s")
        goal_proc.kill()
        result = 1

    # Cleanup
    print("\nCleaning up...")
    subprocess.run("pkill -9 -f 'gz sim'", shell=True)
    subprocess.run("pkill -9 -f 'navigation.launch'", shell=True)
    subprocess.run("pkill -9 -f 'mobile_manipulator'", shell=True)
    subprocess.run("pkill -9 -f 'nav2_'", shell=True)
    subprocess.run("pkill -9 -f 'topic_tools'", shell=True)

    print("\n" + "=" * 60)
    if result == 0:
        print("TEST PASSED - Navigation is working!")
    else:
        print("TEST FAILED - Navigation has issues")
    print("=" * 60)

    return result

if __name__ == "__main__":
    sys.exit(main())
