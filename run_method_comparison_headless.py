#!/usr/bin/env python3
"""
Compare all 5 safety methods in headless mode - with proper log capture
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
GOAL_X = 2.0
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
    """Capture logs from a process in background"""
    def __init__(self, proc, name):
        self.proc = proc
        self.name = name
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
                    decoded = line.decode().strip()
                    self.lines.append(decoded)
            except:
                break

    def stop(self):
        self.stop_event.set()

    def get_log(self):
        return "\n".join(self.lines)

def run_bg(cmd, name):
    p = subprocess.Popen(
        f"source /opt/ros/jazzy/setup.bash && source {WORKSPACE}/install/setup.bash && {cmd}",
        shell=True, executable="/bin/bash",
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    processes.append(p)
    return p

def test_method(method):
    global processes
    processes = []

    print(f"\n{'='*60}")
    print(f"Testing: {method}")
    print(f"{'='*60}")

    result = {
        'method': method,
        'decision': None,
        'violations': 0,
        'nav_status': None,
        'reason': ''
    }

    log_capture = None

    try:
        # 1. Gazebo
        print("[1/4] Gazebo (headless)...", end=" ", flush=True)
        run_bg("ros2 launch mobile_manip_moveit_config mobile_manipulator.launch.py headless:=true", "Gazebo")
        time.sleep(20)
        print("OK")

        # 2. Nav2
        print("[2/4] Nav2...", end=" ", flush=True)
        run_bg("ros2 launch mobile_manip_moveit_config navigation.launch.py use_sim_time:=true rviz:=false", "Nav2")
        time.sleep(12)
        print("OK")

        # 3. Geofence with log capture
        print(f"[3/4] Geofence ({method})...", end=" ", flush=True)
        geofence = subprocess.Popen(
            f"source /opt/ros/jazzy/setup.bash && source {WORKSPACE}/install/setup.bash && "
            f"ros2 launch geofence_policy_enforcer demo.launch.py safety_method:={method} use_sim_time:=true 2>&1",
            shell=True, executable="/bin/bash",
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        processes.append(geofence)
        log_capture = LogCapture(geofence, "Geofence")
        time.sleep(6)
        print("OK")

        # 4. Send goal
        print(f"[4/4] Goal ({GOAL_X}, {GOAL_Y})...", end=" ", flush=True)
        goal_result = subprocess.run(
            f'''source /opt/ros/jazzy/setup.bash && source {WORKSPACE}/install/setup.bash && \
            ros2 action send_goal /navigate_to_pose_safe nav2_msgs/action/NavigateToPose \
            "{{pose: {{header: {{frame_id: 'map'}}, pose: {{position: {{x: {GOAL_X}, y: {GOAL_Y}, z: 0.0}}, orientation: {{w: 1.0}}}}}}}}" 2>&1
            ''',
            shell=True, executable="/bin/bash",
            capture_output=True, text=True, timeout=30
        )
        print("OK")

        # Wait for violations
        time.sleep(5)

        # Stop log capture and get logs
        log_capture.stop()
        log = log_capture.get_log()
        goal_output = goal_result.stdout

        # Parse decision from log
        if "REJECTED" in log:
            result['decision'] = "REJECT"
            match = re.search(r"REJECTED goal.*?:(.*?)(?:\n|$)", log)
            if match:
                result['reason'] = match.group(1).strip()[:40]
        elif "PROJECTED" in log:
            result['decision'] = "PROJECT"
        elif "ALLOWED" in log:
            result['decision'] = "ALLOW"

        # Count violations
        result['violations'] = log.count("VIOLATION: Robot inside forbidden zone")

        # Nav status
        if "SUCCEEDED" in goal_output:
            result['nav_status'] = "SUCCEEDED"
        elif "ABORTED" in goal_output:
            result['nav_status'] = "ABORTED"

    except subprocess.TimeoutExpired:
        result['decision'] = "TIMEOUT"
    except Exception as e:
        result['decision'] = "ERROR"
        result['reason'] = str(e)[:30]
    finally:
        if log_capture:
            log_capture.stop()
        cleanup()

    return result

def main():
    print("="*60)
    print("SAFETY METHOD COMPARISON (Headless)")
    print(f"Goal: ({GOAL_X}, {GOAL_Y}) - Inside Zone A")
    print("="*60)

    for method in METHODS:
        results[method] = test_method(method)
        r = results[method]
        dec = r['decision'] or 'N/A'
        print(f"=> {dec}, Violations: {r['violations']}")

    # Summary Table
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    print(f"{'Method':<15} {'Decision':<10} {'Violations':<12} {'Status':<12} {'Safe?'}")
    print("-"*70)

    for method in METHODS:
        r = results[method]
        dec = r['decision'] or 'N/A'
        viol = r['violations']
        nav = r['nav_status'] or 'N/A'

        if dec in ["REJECT", "PROJECT"]:
            safe = "YES"
        elif dec == "ALLOW" and viol > 0:
            safe = "NO"
        elif dec == "ALLOW":
            safe = "?"
        else:
            safe = "?"

        print(f"{method:<15} {dec:<10} {viol:<12} {nav:<12} {safe}")

    print("="*70)

    # Expected vs Actual
    print("\nEXPECTED BEHAVIOR (Goal inside forbidden zone):")
    print("-"*70)
    expected = {
        'no_guard': ('ALLOW', 'Should allow, violations expected'),
        'selp_proper': ('REJECT', 'LTL automaton rejects zone entry'),
        'cbf': ('REJECT', 'CBF barrier h(x) < 0'),
        'ssm': ('REJECT', 'SSM protective distance'),
        'geofence': ('REJECT', 'Uncertainty-aware margin'),
    }

    for method in METHODS:
        r = results[method]
        exp_dec, exp_reason = expected[method]
        actual = r['decision'] or 'N/A'
        match = "OK" if actual == exp_dec else "MISMATCH"
        print(f"  {method:<15} Expected: {exp_dec:<10} Actual: {actual:<10} [{match}]")

    print("\n" + "="*70)

if __name__ == "__main__":
    main()
