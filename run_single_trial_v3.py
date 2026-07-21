#!/usr/bin/env python3
"""
Single Trial Test v3 - With action server wait and better timeout
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

# Timing (seconds)
GAZEBO_WAIT = 25    # Wait for Gazebo to initialize
NAV2_WAIT = 20      # Wait for Nav2 to initialize
GEOFENCE_WAIT = 10  # Wait for geofence to initialize
ACTION_TIMEOUT = 60 # Action command timeout

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
    os.system("pkill -9 -f 'nav2' 2>/dev/null")
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

def stream_output(proc, name, stop_event, decision_holder):
    """Stream process output in real-time and capture decision"""
    while not stop_event.is_set():
        line = proc.stdout.readline()
        if line:
            decoded = line.decode().strip()
            print(f"[{name}] {decoded}")
            # Capture decision
            if "ALLOWED" in decoded or "REJECTED" in decoded or "PROJECTED" in decoded:
                for word in ["ALLOWED", "REJECTED", "PROJECTED"]:
                    if word in decoded:
                        decision_holder.append(word)
                        break
        elif proc.poll() is not None:
            break

def wait_for_action_server(action_name, timeout=30):
    """Wait for action server to be available"""
    print(f"[Check] Waiting for action server {action_name}...")
    start = time.time()
    while time.time() - start < timeout:
        result = subprocess.run(
            f"source /opt/ros/jazzy/setup.bash && source {WORKSPACE}/install/setup.bash && "
            f"ros2 action list 2>/dev/null | grep -q {action_name}",
            shell=True, executable="/bin/bash"
        )
        if result.returncode == 0:
            print(f"[Check] Action server {action_name} is ready!")
            return True
        time.sleep(1)
    print(f"[Check] Timeout waiting for {action_name}")
    return False

def main():
    print("=" * 60)
    print("SINGLE TRIAL TEST v3 (with action server wait)")
    print(f"Method: {METHOD}, Goal: ({GOAL_X}, {GOAL_Y})")
    print("=" * 60)

    stop_event = threading.Event()
    decision_holder = []

    try:
        # 1. Gazebo
        headless_flag = "true" if not GAZEBO_GUI else "false"
        print(f"\n[1/4] Launching Gazebo (headless={headless_flag})...")
        gazebo = run_bg(
            f"ros2 launch mobile_manip_moveit_config mobile_manipulator.launch.py headless:={headless_flag}",
            "Gazebo"
        )
        time.sleep(GAZEBO_WAIT)

        # 2. Nav2
        rviz_flag = "true" if RVIZ_GUI else "false"
        print(f"\n[2/4] Launching Nav2 (rviz={rviz_flag})...")
        nav2 = run_bg(
            f"ros2 launch mobile_manip_moveit_config navigation.launch.py use_sim_time:=true rviz:={rviz_flag}",
            "Nav2"
        )
        time.sleep(NAV2_WAIT)

        # 3. Geofence - with output streaming
        print(f"\n[3/4] Launching Geofence (method={METHOD})...")
        geofence = subprocess.Popen(
            f"source /opt/ros/jazzy/setup.bash && source {WORKSPACE}/install/setup.bash && "
            f"ros2 launch geofence_policy_enforcer demo.launch.py safety_method:={METHOD} use_sim_time:=true 2>&1",
            shell=True, executable="/bin/bash",
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        processes.append(geofence)

        # Stream geofence output in background
        log_thread = threading.Thread(target=stream_output, args=(geofence, "Geofence", stop_event, decision_holder))
        log_thread.daemon = True
        log_thread.start()

        time.sleep(GEOFENCE_WAIT)

        # Wait for action server to be available
        if not wait_for_action_server("navigate_to_pose_safe", timeout=30):
            print("[ERROR] Action server not available!")
            return

        # 4. Send goal
        print(f"\n[4/4] Sending goal to ({GOAL_X}, {GOAL_Y})...")
        print("=" * 60)

        result = subprocess.run(
            f'''source /opt/ros/jazzy/setup.bash && source {WORKSPACE}/install/setup.bash && \
            ros2 action send_goal --feedback /navigate_to_pose_safe nav2_msgs/action/NavigateToPose \
            "{{pose: {{header: {{frame_id: 'map'}}, pose: {{position: {{x: {GOAL_X}, y: {GOAL_Y}, z: 0.0}}, orientation: {{w: 1.0}}}}}}}}" 2>&1
            ''',
            shell=True, executable="/bin/bash",
            capture_output=True, text=True, timeout=ACTION_TIMEOUT
        )

        print("\n" + "=" * 60)
        print("ACTION RESULT:")
        print("=" * 60)
        print(result.stdout[:2000] if result.stdout else "(no output)")
        if result.stderr:
            print("STDERR:", result.stderr[:500])

        # Wait a bit for logs to flush
        time.sleep(5)
        stop_event.set()

        # Report decision
        print("\n" + "=" * 60)
        print(f"DECISION CAPTURED: {decision_holder[0] if decision_holder else 'N/A'}")
        print("=" * 60)

    except subprocess.TimeoutExpired:
        print("[ERROR] Action command timed out!")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        stop_event.set()
        cleanup()

if __name__ == "__main__":
    main()
