#!/usr/bin/env python3
"""
Compare all safety methods with single trial each
"""

import subprocess
import time
import signal
import sys
import os

WORKSPACE = "/home/jim/ros2_motion_planning_tutorials/src/mobile_manipulator_tutorial"
METHODS = ["no_guard", "selp_proper", "cbf", "ssm", "geofence"]
GOAL_X = 2.0
GOAL_Y = 1.0  # Inside Zone A (forbidden)

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
    time.sleep(3)

def signal_handler(sig, frame):
    cleanup()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def run_single_trial(method):
    """Run a single trial and return the result"""
    global processes
    processes = []

    print(f"\n{'='*60}")
    print(f"Testing: {method}")
    print(f"{'='*60}")

    result = {
        'method': method,
        'decision': None,
        'reason': None,
        'violations': 0,
        'nav_status': None
    }

    try:
        # 1. Gazebo
        print("[1/4] Launching Gazebo...")
        p = subprocess.Popen(
            f"source /opt/ros/jazzy/setup.bash && source {WORKSPACE}/install/setup.bash && "
            f"ros2 launch mobile_manip_moveit_config mobile_manipulator.launch.py 2>&1",
            shell=True, executable="/bin/bash",
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        processes.append(p)
        time.sleep(25)

        # 2. Nav2
        print("[2/4] Launching Nav2...")
        p = subprocess.Popen(
            f"source /opt/ros/jazzy/setup.bash && source {WORKSPACE}/install/setup.bash && "
            f"ros2 launch mobile_manip_moveit_config navigation.launch.py use_sim_time:=true 2>&1",
            shell=True, executable="/bin/bash",
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        processes.append(p)
        time.sleep(15)

        # 3. Geofence with specific method
        print(f"[3/4] Launching Geofence (method={method})...")
        geofence = subprocess.Popen(
            f"source /opt/ros/jazzy/setup.bash && source {WORKSPACE}/install/setup.bash && "
            f"ros2 launch geofence_policy_enforcer demo.launch.py safety_method:={method} use_sim_time:=true 2>&1",
            shell=True, executable="/bin/bash",
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        processes.append(geofence)
        time.sleep(8)

        # 4. Send goal and capture output
        print(f"[4/4] Sending goal ({GOAL_X}, {GOAL_Y})...")
        goal_result = subprocess.run(
            f'''source /opt/ros/jazzy/setup.bash && source {WORKSPACE}/install/setup.bash && \
            ros2 action send_goal /navigate_to_pose_safe nav2_msgs/action/NavigateToPose \
            "{{pose: {{header: {{frame_id: 'map'}}, pose: {{position: {{x: {GOAL_X}, y: {GOAL_Y}, z: 0.0}}, orientation: {{w: 1.0}}}}}}}}" 2>&1
            ''',
            shell=True, executable="/bin/bash",
            capture_output=True, text=True, timeout=30
        )

        # Wait for logs
        time.sleep(3)

        # Read geofence output
        geofence.terminate()
        try:
            stdout, _ = geofence.communicate(timeout=2)
            log_output = stdout.decode() if stdout else ""
        except:
            log_output = ""

        # Parse results
        goal_output = goal_result.stdout

        # Decision
        if "REJECTED" in log_output:
            result['decision'] = "REJECT"
            # Extract reason
            for line in log_output.split('\n'):
                if "REJECTED" in line:
                    if ":" in line:
                        result['reason'] = line.split(":")[-1].strip()[:50]
                    break
        elif "ALLOWED" in log_output:
            result['decision'] = "ALLOW"
        elif "PROJECTED" in log_output:
            result['decision'] = "PROJECT"
            for line in log_output.split('\n'):
                if "PROJECTED" in line:
                    result['reason'] = "projected to safe location"
                    break

        # Violations
        result['violations'] = log_output.count("VIOLATION: Robot inside forbidden zone")

        # Nav status
        if "SUCCEEDED" in goal_output:
            result['nav_status'] = "SUCCEEDED"
        elif "ABORTED" in goal_output:
            result['nav_status'] = "ABORTED"
        elif "rejected" in goal_output.lower():
            result['nav_status'] = "REJECTED"
        else:
            result['nav_status'] = "UNKNOWN"

    except Exception as e:
        print(f"Error: {e}")
        result['decision'] = "ERROR"
        result['reason'] = str(e)[:30]
    finally:
        cleanup()

    return result

def main():
    print("="*60)
    print("SAFETY METHOD COMPARISON TEST")
    print(f"Goal: ({GOAL_X}, {GOAL_Y}) - Inside Zone A (forbidden)")
    print("="*60)

    for method in METHODS:
        results[method] = run_single_trial(method)
        print(f"\n[{method}] Decision: {results[method]['decision']}, "
              f"Violations: {results[method]['violations']}, "
              f"Nav: {results[method]['nav_status']}")

    # Summary table
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    print(f"{'Method':<15} {'Decision':<10} {'Reason':<35} {'Violations':<12} {'Nav Status'}")
    print("-"*80)

    for method in METHODS:
        r = results[method]
        reason = r['reason'] or '-'
        print(f"{method:<15} {r['decision'] or 'N/A':<10} {reason:<35} {r['violations']:<12} {r['nav_status'] or 'N/A'}")

    print("="*80)

    # Analysis
    print("\nANALYSIS:")
    for method in METHODS:
        r = results[method]
        if r['decision'] == 'REJECT':
            print(f"  {method}: Goal blocked at safety layer (proactive)")
        elif r['decision'] == 'ALLOW' and r['violations'] > 0:
            print(f"  {method}: Goal allowed, {r['violations']} violations occurred (UNSAFE)")
        elif r['decision'] == 'PROJECT':
            print(f"  {method}: Goal projected to safe location (adaptive)")
        elif r['decision'] == 'ALLOW' and r['violations'] == 0:
            print(f"  {method}: Goal allowed, no violations (may have secondary defense)")

if __name__ == "__main__":
    main()
