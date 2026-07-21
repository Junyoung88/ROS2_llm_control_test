#!/usr/bin/env python3
"""
Boundary Case Test - Test point just outside forbidden zone
Shows difference in safety margins between methods
"""

import subprocess
import time
import signal
import sys
import os
import re
import threading
from collections import deque

WORKSPACE = "/home/jim/ros2_motion_planning_tutorials/src/mobile_manipulator_tutorial"
METHODS = ["no_guard", "selp_proper", "cbf", "ssm", "geofence"]

# Boundary test point: 0.1m outside Zone A west edge (1.5)
# Zone A: x=[1.5, 2.5], y=[0.5, 1.5]
GOAL_X = 1.4  # 0.1m outside zone
GOAL_Y = 1.0

processes = []
results = {}

def cleanup():
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
    os.system("pkill -9 -f 'robot_state' 2>/dev/null")
    time.sleep(3)

def signal_handler(sig, frame):
    cleanup()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

class LogCapture:
    def __init__(self, proc):
        self.proc = proc
        self.lines = deque(maxlen=500)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._capture)
        self.thread.daemon = True
        self.thread.start()

    def _capture(self):
        while not self.stop_event.is_set():
            if self.proc.poll() is not None:
                break
            try:
                line = self.proc.stdout.readline()
                if line:
                    self.lines.append(line.decode().strip())
            except:
                break

    def stop(self):
        self.stop_event.set()

    def get_log(self):
        return "\n".join(self.lines)

def run_bg(cmd):
    p = subprocess.Popen(
        f"source /opt/ros/jazzy/setup.bash && source {WORKSPACE}/install/setup.bash && {cmd}",
        shell=True, executable="/bin/bash",
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    processes.append(p)
    return p

def test_method(method, goal_x, goal_y):
    global processes
    processes = []

    print(f"\n[{method}] Testing...", end=" ", flush=True)

    result = {'decision': None, 'violations': 0, 'reason': ''}
    log_capture = None

    try:
        # 1. Gazebo
        run_bg("ros2 launch mobile_manip_moveit_config mobile_manipulator.launch.py headless:=true")
        time.sleep(18)

        # 2. Nav2
        run_bg("ros2 launch mobile_manip_moveit_config navigation.launch.py use_sim_time:=true rviz:=false")
        time.sleep(10)

        # 3. Geofence
        geofence = subprocess.Popen(
            f"source /opt/ros/jazzy/setup.bash && source {WORKSPACE}/install/setup.bash && "
            f"ros2 launch geofence_policy_enforcer demo.launch.py safety_method:={method} use_sim_time:=true 2>&1",
            shell=True, executable="/bin/bash",
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        processes.append(geofence)
        log_capture = LogCapture(geofence)
        time.sleep(5)

        # 4. Send goal
        goal_result = subprocess.run(
            f'''source /opt/ros/jazzy/setup.bash && source {WORKSPACE}/install/setup.bash && \
            ros2 action send_goal /navigate_to_pose_safe nav2_msgs/action/NavigateToPose \
            "{{pose: {{header: {{frame_id: 'map'}}, pose: {{position: {{x: {goal_x}, y: {goal_y}, z: 0.0}}, orientation: {{w: 1.0}}}}}}}}" 2>&1
            ''',
            shell=True, executable="/bin/bash",
            capture_output=True, text=True, timeout=30
        )

        time.sleep(5)
        log_capture.stop()
        log = log_capture.get_log()

        # Parse
        if "REJECTED" in log:
            result['decision'] = "REJECT"
            match = re.search(r"REJECTED goal.*?:(.*?)(?:\n|$)", log)
            if match:
                result['reason'] = match.group(1).strip()[:35]
        elif "PROJECTED" in log:
            result['decision'] = "PROJECT"
            match = re.search(r"PROJECTED.*?to \(([\d.-]+), ([\d.-]+)\)", log)
            if match:
                result['reason'] = f"to ({match.group(1)}, {match.group(2)})"
        elif "ALLOWED" in log:
            result['decision'] = "ALLOW"

        result['violations'] = log.count("VIOLATION: Robot inside forbidden zone")

        print(f"{result['decision'] or 'N/A'}", end="")
        if result['violations'] > 0:
            print(f" ({result['violations']} violations!)", end="")
        print()

    except Exception as e:
        result['decision'] = "ERROR"
        print(f"ERROR: {e}")
    finally:
        if log_capture:
            log_capture.stop()
        cleanup()

    return result

def main():
    print("="*70)
    print("BOUNDARY CASE TEST")
    print("="*70)
    print(f"Zone A: x=[1.5, 2.5], y=[0.5, 1.5]")
    print(f"Test point: ({GOAL_X}, {GOAL_Y}) - 0.1m OUTSIDE Zone A")
    print("="*70)
    print("\nExpected behavior:")
    print("  no_guard:     ALLOW (no check)")
    print("  selp_proper:  ALLOW (no margin - point outside zone)")
    print("  cbf:          REJECT (0.3m margin)")
    print("  ssm:          REJECT (~0.35m margin)")
    print("  geofence:     PROJECT (0.55m margin)")
    print("="*70)

    for method in METHODS:
        results[method] = test_method(method, GOAL_X, GOAL_Y)

    # Summary
    print("\n" + "="*70)
    print("BOUNDARY TEST RESULTS")
    print("="*70)
    print(f"{'Method':<15} {'Decision':<10} {'Violations':<12} {'Reason'}")
    print("-"*70)

    for method in METHODS:
        r = results[method]
        dec = r['decision'] or 'N/A'
        viol = r['violations']
        reason = r['reason'] or '-'
        print(f"{method:<15} {dec:<10} {viol:<12} {reason}")

    print("="*70)

    # Analysis
    print("\nANALYSIS:")
    print("-"*70)

    # Expected results for 0.1m outside zone
    expected = {
        'no_guard': 'ALLOW',
        'selp_proper': 'ALLOW',  # No margin!
        'cbf': 'REJECT',         # 0.3m margin
        'ssm': 'REJECT',         # ~0.35m margin
        'geofence': 'PROJECT',   # 0.55m margin, projects to safe
    }

    all_match = True
    for method in METHODS:
        r = results[method]
        exp = expected[method]
        actual = r['decision'] or 'N/A'

        # PROJECT is also acceptable for REJECT expected
        if exp == 'PROJECT' and actual == 'REJECT':
            match = True
        elif exp == 'REJECT' and actual == 'PROJECT':
            match = True
        else:
            match = (actual == exp)

        status = "OK" if match else "MISMATCH"
        if not match:
            all_match = False
        print(f"  {method:<15} Expected: {exp:<10} Actual: {actual:<10} [{status}]")

    print("\n" + "="*70)
    print("KEY INSIGHT:")
    print("="*70)
    print("""
At 0.1m outside forbidden zone boundary:

  no_guard + selp_proper: ALLOW the goal
    -> Robot may drift into zone during execution!
    -> This is why margin matters for safety

  cbf + ssm + geofence: REJECT/PROJECT the goal
    -> Proactive safety with margin
    -> geofence has largest margin (0.55m) for uncertainty
""")

if __name__ == "__main__":
    main()
