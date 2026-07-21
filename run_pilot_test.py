#!/usr/bin/env python3
"""
Pilot Test - Quick validation of experiment framework
Tests each method with 1 attack and 1 benign scenario
"""

import subprocess
import time
import signal
import sys
import os
import json
import re
import threading
from datetime import datetime
from pathlib import Path
from collections import deque

WORKSPACE = "/home/jim/ros2_motion_planning_tutorials/src/mobile_manipulator_tutorial"
OUTPUT_DIR = Path("/home/jim/ros2_motion_planning_tutorials/experiment_results/pilot")

METHODS = ["no_guard", "selp_proper", "cbf", "ssm", "geofence"]

# Test cases: (scenario, goal_x, goal_y, description)
# Expected decision varies by method
TEST_CASES = [
    # Attack cases
    ("A1_inside", 2.0, 1.0, "Inside Zone A"),
    ("A2_boundary", 1.4, 1.0, "0.1m outside Zone A"),
    # Benign cases
    ("B1_safe", 0.0, 0.0, "Far from zones (origin)"),
]

# Expected decisions per method per scenario
EXPECTED = {
    "no_guard": {"A1_inside": "allow", "A2_boundary": "allow", "B1_safe": "allow"},
    "selp_proper": {"A1_inside": "reject", "A2_boundary": "allow", "B1_safe": "allow"},  # No margin!
    "cbf": {"A1_inside": "reject", "A2_boundary": "reject", "B1_safe": "allow"},
    "ssm": {"A1_inside": "reject", "A2_boundary": "reject", "B1_safe": "allow"},
    "geofence": {"A1_inside": "reject", "A2_boundary": "project", "B1_safe": "allow"},
}

processes = []
results = []

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
    time.sleep(3)

def signal_handler(sig, frame):
    cleanup()
    print("\n[Interrupted]")
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

def run_trial(method, scenario, goal_x, goal_y):
    global processes
    processes = []

    expected = EXPECTED[method][scenario]

    result = {
        'method': method,
        'scenario': scenario,
        'goal': (goal_x, goal_y),
        'expected': expected,
        'decision': None,
        'violations': 0,
        'correct': False
    }

    log_capture = None

    try:
        # 1. Gazebo (increased wait time)
        p = subprocess.Popen(
            f"source /opt/ros/jazzy/setup.bash && source {WORKSPACE}/install/setup.bash && "
            f"ros2 launch mobile_manip_moveit_config mobile_manipulator.launch.py headless:=true 2>&1",
            shell=True, executable="/bin/bash",
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        processes.append(p)
        time.sleep(20)

        # 2. Nav2
        p = subprocess.Popen(
            f"source /opt/ros/jazzy/setup.bash && source {WORKSPACE}/install/setup.bash && "
            f"ros2 launch mobile_manip_moveit_config navigation.launch.py use_sim_time:=true rviz:=false 2>&1",
            shell=True, executable="/bin/bash",
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        processes.append(p)
        time.sleep(12)

        # 3. Geofence
        geofence = subprocess.Popen(
            f"source /opt/ros/jazzy/setup.bash && source {WORKSPACE}/install/setup.bash && "
            f"ros2 launch geofence_policy_enforcer demo.launch.py safety_method:={method} use_sim_time:=true 2>&1",
            shell=True, executable="/bin/bash",
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        processes.append(geofence)
        log_capture = LogCapture(geofence)
        time.sleep(6)

        # 4. Send goal
        subprocess.run(
            f'''source /opt/ros/jazzy/setup.bash && source {WORKSPACE}/install/setup.bash && \
            ros2 action send_goal /navigate_to_pose_safe nav2_msgs/action/NavigateToPose \
            "{{pose: {{header: {{frame_id: 'map'}}, pose: {{position: {{x: {goal_x}, y: {goal_y}, z: 0.0}}, orientation: {{w: 1.0}}}}}}}}" 2>&1
            ''',
            shell=True, executable="/bin/bash",
            capture_output=True, text=True, timeout=30
        )

        # Longer wait for logs
        time.sleep(6)
        log_capture.stop()
        time.sleep(1)  # Extra time for buffer flush
        log = log_capture.get_log()

        # Parse decision
        if "REJECTED" in log:
            result['decision'] = "reject"
        elif "PROJECTED" in log:
            result['decision'] = "project"
        elif "ALLOWED" in log:
            result['decision'] = "allow"

        # Violations
        result['violations'] = log.count("VIOLATION: Robot inside forbidden zone")

        # Check correctness (project is acceptable for reject expected)
        if expected == "reject":
            result['correct'] = result['decision'] in ["reject", "project"]
        elif expected == "project":
            result['correct'] = result['decision'] in ["reject", "project"]
        else:  # allow
            result['correct'] = result['decision'] == "allow"

    except Exception as e:
        result['error'] = str(e)[:50]
    finally:
        if log_capture:
            log_capture.stop()
        cleanup()

    return result

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_trials = len(METHODS) * len(TEST_CASES)
    est_time = total_trials * 50 // 60  # ~50 sec per trial

    print("="*70)
    print("PILOT TEST (Improved)")
    print("="*70)
    print(f"Methods: {METHODS}")
    print(f"Test cases: {[t[0] for t in TEST_CASES]}")
    print(f"Total trials: {total_trials}")
    print(f"Estimated time: ~{est_time} minutes")
    print("="*70)
    print("\nExpected decisions per method:")
    for method in METHODS:
        exps = [f"{s}:{EXPECTED[method][s]}" for s, _, _, _ in TEST_CASES]
        print(f"  {method}: {', '.join(exps)}")
    print("="*70)

    trial_num = 0
    for method in METHODS:
        print(f"\n[{method}]")
        for scenario, gx, gy, desc in TEST_CASES:
            trial_num += 1
            expected = EXPECTED[method][scenario]
            print(f"  [{trial_num}/{total_trials}] {scenario} (exp:{expected})...", end=" ", flush=True)

            result = run_trial(method, scenario, gx, gy)
            results.append(result)

            dec = result['decision'] or 'N/A'
            status = "OK" if result['correct'] else "FAIL"
            print(f"{dec} [{status}]")

    # Summary
    print("\n" + "="*70)
    print("PILOT TEST RESULTS")
    print("="*70)

    # By method
    print(f"\n{'Method':<15} {'Correct':<10} {'Attack':<12} {'Benign':<12}")
    print("-"*50)

    for method in METHODS:
        method_results = [r for r in results if r['method'] == method]
        correct = sum(1 for r in method_results if r['correct'])
        total = len(method_results)

        attack_results = [r for r in method_results if r['scenario'].startswith('A')]
        benign_results = [r for r in method_results if r['scenario'].startswith('B')]

        attack_correct = sum(1 for r in attack_results if r['correct'])
        benign_correct = sum(1 for r in benign_results if r['correct'])

        print(f"{method:<15} {correct}/{total:<8} {attack_correct}/{len(attack_results):<10} {benign_correct}/{len(benign_results):<10}")

    print("="*70)

    # Detailed results
    print("\nDETAILED RESULTS:")
    print("-"*70)
    for r in results:
        dec = r['decision'] or 'N/A'
        exp = r['expected']
        status = "OK" if r['correct'] else "FAIL"
        viol = r['violations']
        print(f"  {r['method']:<12} {r['scenario']:<15} exp:{exp:<8} got:{dec:<8} [{status}] viol:{viol}")

    # Save results
    results_file = OUTPUT_DIR / f"pilot_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # Check if ready for full experiment
    total_correct = sum(1 for r in results if r['correct'])
    if total_correct == total_trials:
        print("\n[PASS] All tests passed - ready for full experiment!")
    else:
        print(f"\n[WARN] {total_trials - total_correct} tests failed - review before full experiment")

if __name__ == "__main__":
    main()
