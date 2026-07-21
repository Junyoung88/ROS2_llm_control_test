#!/usr/bin/env python3
"""
Single Trial Test v2 - With real-time log monitoring
"""

import subprocess
import time
import signal
import sys
import os
import threading

WORKSPACE = "/home/jim/ros2_motion_planning_tutorials/src/mobile_manipulator_tutorial"
METHOD = "geofence"
# Zone A center: (2.0, 1.0) - confirmed navigable area
GOAL_X = 2.0
GOAL_Y = 1.0

# GUI options - set to False for headless/faster experiments
GAZEBO_GUI = False  # Gazebo GUI (large window)
RVIZ_GUI = False    # RViz visualization

processes = []

def cleanup():
    print("\n[Cleanup] Terminating...")
    for p in processes:
        try:
            p.terminate()
            p.wait(timeout=2)
        except:
            try:
                p.kill()
            except:
                pass
    os.system("pkill -9 -f 'gz sim' 2>/dev/null")
    os.system("pkill -9 -f 'rviz' 2>/dev/null")
    os.system("pkill -9 -f 'goal_gate' 2>/dev/null")
    time.sleep(2)
    print("[Cleanup] Done")

def signal_handler(sig, frame):
    cleanup()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def run_bg(cmd, name):
    print(f"[{name}] Starting...")
    p = subprocess.Popen(
        f"source /opt/ros/jazzy/setup.bash && source {WORKSPACE}/install/setup.bash && {cmd}",
        shell=True, executable="/bin/bash",
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    processes.append(p)
    return p

def stream_output(proc, name, stop_event):
    """Stream process output in real-time"""
    while not stop_event.is_set():
        line = proc.stdout.readline()
        if line:
            print(f"[{name}] {line.decode().strip()}")
        elif proc.poll() is not None:
            break

def main():
    print("=" * 60)
    print("SINGLE TRIAL TEST v2 (with logs)")
    print(f"Method: {METHOD}, Goal: ({GOAL_X}, {GOAL_Y})")
    print("=" * 60)

    stop_event = threading.Event()

    try:
        # 1. Gazebo
        headless_flag = "true" if not GAZEBO_GUI else "false"
        print(f"\n[1/4] Launching Gazebo (headless={headless_flag})...")
        gazebo = run_bg(
            f"ros2 launch mobile_manip_moveit_config mobile_manipulator.launch.py headless:={headless_flag}",
            "Gazebo"
        )
        time.sleep(20 if not GAZEBO_GUI else 25)

        # 2. Nav2
        rviz_flag = "true" if RVIZ_GUI else "false"
        print(f"\n[2/4] Launching Nav2 (rviz={rviz_flag})...")
        nav2 = run_bg(
            f"ros2 launch mobile_manip_moveit_config navigation.launch.py use_sim_time:=true rviz:={rviz_flag}",
            "Nav2"
        )
        time.sleep(15)

        # 3. Geofence - with output streaming
        print("\n[3/4] Launching Geofence (watching logs)...")
        geofence = subprocess.Popen(
            f"source /opt/ros/jazzy/setup.bash && source {WORKSPACE}/install/setup.bash && "
            f"ros2 launch geofence_policy_enforcer demo.launch.py safety_method:={METHOD} use_sim_time:=true 2>&1",
            shell=True, executable="/bin/bash",
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        processes.append(geofence)

        # Stream geofence output in background
        log_thread = threading.Thread(target=stream_output, args=(geofence, "Geofence", stop_event))
        log_thread.daemon = True
        log_thread.start()

        time.sleep(8)

        # 4. Send goal
        print(f"\n[4/4] Sending goal to ({GOAL_X}, {GOAL_Y})...")
        print("=" * 60)

        result = subprocess.run(
            f'''source /opt/ros/jazzy/setup.bash && source {WORKSPACE}/install/setup.bash && \
            ros2 action send_goal /navigate_to_pose_safe nav2_msgs/action/NavigateToPose \
            "{{pose: {{header: {{frame_id: 'map'}}, pose: {{position: {{x: {GOAL_X}, y: {GOAL_Y}, z: 0.0}}, orientation: {{w: 1.0}}}}}}}}" 2>&1
            ''',
            shell=True, executable="/bin/bash",
            capture_output=True, text=True, timeout=30
        )

        print("\n" + "=" * 60)
        print("ACTION RESULT:")
        print("=" * 60)
        print(result.stdout)

        # Wait a bit for logs to flush
        time.sleep(3)
        stop_event.set()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        stop_event.set()
        cleanup()

if __name__ == "__main__":
    main()
