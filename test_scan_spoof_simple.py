#!/usr/bin/env python3
"""
Simple LIDAR Scan Spoofing Test

1. Start Gazebo simulation with proper launch
2. Start scan relay (normal operation)
3. Start Navigation with AMCL
4. Verify localization is working
5. Switch to scan spoofing attack
6. Observe AMCL confusion
"""

import os
import sys
import signal
import subprocess
import time

WORKSPACE = "/home/jim/ros2_motion_planning_tutorials"


def safe_pkill(pattern: str):
    """Kill processes matching pattern"""
    try:
        subprocess.run(f"pkill -9 -f '{pattern}'", shell=True, timeout=5)
    except:
        pass


def cleanup_all():
    """Kill all related processes"""
    patterns = [
        "gz sim", "gzserver", "gzclient", "ruby.*gz",
        "rviz2", "nav2_", "controller_server", "planner_server",
        "behavior_server", "bt_navigator", "lifecycle_manager",
        "goal_gate_node", "cmd_vel_guard", "path_watchdog",
        "attack_", "relay", "ros_gz", "parameter_bridge"
    ]
    for p in patterns:
        safe_pkill(p)
    time.sleep(2)


def run_bg(cmd: str, log_file: str = None):
    """Run command in background, return Popen"""
    if log_file:
        f = open(log_file, 'w')
        return subprocess.Popen(
            cmd, shell=True, executable='/bin/bash',
            stdout=f, stderr=subprocess.STDOUT, preexec_fn=os.setsid
        )
    return subprocess.Popen(
        cmd, shell=True, executable='/bin/bash',
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid
    )


def run_cmd(cmd: str, timeout: int = 10) -> str:
    """Run command and return output"""
    try:
        result = subprocess.run(
            cmd, shell=True, executable='/bin/bash',
            capture_output=True, text=True, timeout=timeout
        )
        return result.stdout
    except:
        return ""


def source_ros():
    return f"source /opt/ros/jazzy/setup.bash && source {WORKSPACE}/install/setup.bash"


def main():
    print("=" * 60)
    print("LIDAR SCAN SPOOFING TEST")
    print("=" * 60)

    procs = []

    try:
        # Step 0: Cleanup
        print("\n[0] Cleaning up existing processes...")
        cleanup_all()

        # Step 1: Start Gazebo simulation
        print("\n[1] Starting Gazebo simulation (headless)...")
        gz_cmd = f"""
            {source_ros()} && \
            ros2 launch mobile_manip_moveit_config mobile_manipulator.launch.py \
                use_sim_time:=true headless:=true
        """
        gz_proc = run_bg(gz_cmd, '/tmp/gz_spoof_test.log')
        procs.append(gz_proc)
        print("    Waiting 30s for Gazebo...")
        time.sleep(30)

        if gz_proc.poll() is not None:
            print("    [ERROR] Gazebo failed to start!")
            with open('/tmp/gz_spoof_test.log') as f:
                print(f.read()[-2000:])
            return

        # Step 2: Check scan topics
        print("\n[2] Checking scan topics...")
        topics = run_cmd(f"{source_ros()} && ros2 topic list")
        print(f"    Topics containing 'scan': {[t for t in topics.split() if 'scan' in t]}")

        if '/scan_real' not in topics:
            print("    [WARN] /scan_real not found - checking bridge config...")
            print(f"    All topics: {topics}")

        # Step 3: Start scan relay (normal operation)
        print("\n[3] Starting scan relay (scan_real -> scan)...")
        relay_cmd = f"""
            {source_ros()} && \
            ros2 run geofence_policy_enforcer scan_relay
        """
        relay_proc = run_bg(relay_cmd, '/tmp/scan_relay.log')
        procs.append(relay_proc)
        time.sleep(3)

        # Verify /scan is now available
        topics = run_cmd(f"{source_ros()} && ros2 topic list")
        print(f"    Topics with 'scan': {[t for t in topics.split() if 'scan' in t]}")

        # Step 4: Start Navigation with AMCL
        print("\n[4] Starting Navigation with AMCL...")
        nav_cmd = f"""
            {source_ros()} && \
            ros2 launch mobile_manip_moveit_config navigation.launch.py \
                use_sim_time:=true use_rviz:=false
        """
        nav_proc = run_bg(nav_cmd, '/tmp/nav_spoof_test.log')
        procs.append(nav_proc)
        print("    Waiting 30s for Nav2...")
        time.sleep(30)

        # Step 5: Check AMCL pose
        print("\n[5] Checking AMCL localization...")
        for i in range(3):
            amcl_output = run_cmd(
                f"{source_ros()} && ros2 topic echo /amcl_pose --once",
                timeout=15
            )
            if 'position:' in amcl_output:
                # Extract position
                lines = amcl_output.split('\n')
                for j, line in enumerate(lines):
                    if 'position:' in line:
                        print(f"    AMCL pose (measurement {i+1}):")
                        print('\n'.join(['      ' + l for l in lines[j:j+4]]))
                        break
                break
            else:
                print(f"    Waiting for AMCL... ({i+1}/3)")
                time.sleep(5)

        # Step 6: Now switch to spoofing mode
        print("\n[6] SWITCHING TO ATTACK MODE")
        print("    Stopping scan relay, starting scan spoofer...")

        # Stop relay
        if relay_proc.poll() is None:
            os.killpg(os.getpgid(relay_proc.pid), signal.SIGTERM)
            time.sleep(1)

        # Start spoofer with rotation
        print("    Starting scan spoofer (rotation=30deg, scale=0.8)...")
        spoof_cmd = f"""
            {source_ros()} && \
            ros2 run geofence_policy_enforcer attack_scan_spoofing --ros-args \
                -p rotation_offset:=0.523 \
                -p range_scale:=0.8 \
                -p noise_stddev:=0.05
        """
        spoof_proc = run_bg(spoof_cmd, '/tmp/scan_spoof.log')
        procs.append(spoof_proc)
        time.sleep(5)

        # Step 7: Monitor AMCL pose changes under attack
        print("\n[7] MONITORING AMCL UNDER ATTACK...")
        print("    (Expecting localization drift due to scan manipulation)")

        for i in range(5):
            time.sleep(5)
            amcl_output = run_cmd(
                f"{source_ros()} && ros2 topic echo /amcl_pose --once",
                timeout=10
            )
            if 'position:' in amcl_output:
                lines = amcl_output.split('\n')
                for j, line in enumerate(lines):
                    if 'position:' in line:
                        print(f"\n    Measurement {i+1}/5 (t={5*(i+1)}s):")
                        print('\n'.join(['      ' + l for l in lines[j:j+4]]))
                        break
            else:
                print(f"    [WARN] No AMCL pose at measurement {i+1}")

        # Check attack log
        print("\n[8] Attack node status:")
        with open('/tmp/scan_spoof.log') as f:
            content = f.read()
            if content:
                print('    ' + '\n    '.join(content.strip().split('\n')[-5:]))
            else:
                print("    (no output yet)")

        print("\n" + "=" * 60)
        print("TEST COMPLETE")
        print("=" * 60)
        print("Check if AMCL position drifted significantly under attack.")

    except KeyboardInterrupt:
        print("\n[!] Test interrupted")
    finally:
        print("\n[CLEANUP] Stopping all processes...")
        for p in procs:
            try:
                if p.poll() is None:
                    os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except:
                pass
        time.sleep(2)
        cleanup_all()
        print("[CLEANUP] Done")


if __name__ == '__main__':
    main()
