#!/usr/bin/env python3
"""
LIDAR Scan Spoofing Attack Test
================================

Tests if LIDAR spoofing can confuse AMCL localization.

Usage:
    python3 test_scan_spoofing.py [--rotation DEG] [--scale FACTOR] [--noise M]
"""

import subprocess
import time
import argparse
import sys
import os
import signal

# ROS2 setup
os.environ['ROS_DOMAIN_ID'] = '0'

def run_cmd(cmd, bg=False, log_file=None):
    """Run command, optionally in background"""
    if bg:
        if log_file:
            f = open(log_file, 'w')
            return subprocess.Popen(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT,
                                    preexec_fn=os.setsid)
        return subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, preexec_fn=os.setsid)
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


def main():
    parser = argparse.ArgumentParser(description='Test LIDAR scan spoofing attack')
    parser.add_argument('--rotation', type=float, default=30.0,
                        help='Rotation offset in degrees (default: 30)')
    parser.add_argument('--scale', type=float, default=0.8,
                        help='Range scale factor (default: 0.8 = 20%% closer)')
    parser.add_argument('--noise', type=float, default=0.05,
                        help='Noise stddev in meters (default: 0.05)')
    parser.add_argument('--headless', action='store_true',
                        help='Run Gazebo headless')
    args = parser.parse_args()

    print("=" * 60)
    print("LIDAR SCAN SPOOFING TEST")
    print("=" * 60)
    print(f"  Rotation: {args.rotation} degrees")
    print(f"  Scale: {args.scale}x")
    print(f"  Noise: {args.noise}m")
    print("=" * 60)

    procs = []

    try:
        # 1. Start Gazebo simulation
        print("\n[1/4] Starting Gazebo simulation...")
        headless_arg = '-r' if args.headless else ''
        gz_cmd = f"""
source /opt/ros/jazzy/setup.bash && \
source ~/ros2_motion_planning_tutorials/install/setup.bash && \
ros2 launch mobile_manip_moveit_config world.launch.py {headless_arg}
"""
        procs.append(run_cmd(gz_cmd, bg=True, log_file='/tmp/gz_scan_test.log'))
        time.sleep(10)

        # 2. Start Navigation (with AMCL)
        print("[2/4] Starting Navigation with AMCL...")
        nav_cmd = """
source /opt/ros/jazzy/setup.bash && \
source ~/ros2_motion_planning_tutorials/install/setup.bash && \
ros2 launch mobile_manip_moveit_config navigation.launch.py use_sim_time:=true
"""
        procs.append(run_cmd(nav_cmd, bg=True, log_file='/tmp/nav_scan_test.log'))
        time.sleep(15)

        # 3. Get initial AMCL pose estimate
        print("[3/4] Getting initial AMCL pose...")
        result = subprocess.run(
            "source /opt/ros/jazzy/setup.bash && ros2 topic echo /amcl_pose --once",
            shell=True, capture_output=True, text=True, timeout=10,
            executable='/bin/bash'
        )
        print("Initial AMCL pose:")
        if result.stdout:
            # Parse position from output
            lines = result.stdout.split('\n')
            for i, line in enumerate(lines):
                if 'position:' in line:
                    print('\n'.join(lines[i:i+5]))
                    break

        # 4. Start scan spoofing attack
        print("\n[4/4] Starting LIDAR spoofing attack...")
        rotation_rad = args.rotation * 3.14159 / 180.0
        attack_cmd = f"""
source /opt/ros/jazzy/setup.bash && \
source ~/ros2_motion_planning_tutorials/install/setup.bash && \
ros2 run geofence_policy_enforcer attack_scan_spoofing --ros-args \
    -p rotation_offset:={rotation_rad} \
    -p range_scale:={args.scale} \
    -p noise_stddev:={args.noise}
"""
        procs.append(run_cmd(attack_cmd, bg=True, log_file='/tmp/attack_scan_test.log'))
        time.sleep(5)

        # Monitor AMCL pose changes
        print("\n" + "=" * 60)
        print("MONITORING AMCL POSE (attack active)...")
        print("=" * 60)

        for i in range(5):
            print(f"\n--- Measurement {i+1}/5 (at {5*(i+1)}s) ---")
            result = subprocess.run(
                "source /opt/ros/jazzy/setup.bash && ros2 topic echo /amcl_pose --once",
                shell=True, capture_output=True, text=True, timeout=10,
                executable='/bin/bash'
            )
            if result.stdout:
                lines = result.stdout.split('\n')
                for j, line in enumerate(lines):
                    if 'position:' in line:
                        print('\n'.join(lines[j:j+5]))
                        break
            time.sleep(5)

        print("\n" + "=" * 60)
        print("Test complete. Check if AMCL position drifted from attack.")
        print("=" * 60)

        # Check attack log
        print("\nAttack node log:")
        subprocess.run("tail -5 /tmp/attack_scan_test.log", shell=True)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        print("\nCleaning up...")
        for p in procs:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except:
                pass
        time.sleep(2)
        subprocess.run("pkill -f 'gz sim' 2>/dev/null", shell=True)
        subprocess.run("pkill -f 'nav2' 2>/dev/null", shell=True)
        subprocess.run("pkill -f 'attack_scan' 2>/dev/null", shell=True)


if __name__ == '__main__':
    main()
