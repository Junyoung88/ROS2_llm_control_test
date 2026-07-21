#!/usr/bin/env python3
"""
Full Metrics Experiment Framework
Measures all paper-level metrics:
- VR_exec: Violation Rate (robot actually enters forbidden zone)
- ASR_phys: Attack Success Rate (attack goals result in zone entry)
- FN_viol: False Negative violations count
- TCR: Task Completion Rate
- FPR: False Positive Rate
- MD_exec: Minimum Distance during execution
- RT: Response Time
"""

import subprocess
import time
import os
import sys
import json
import csv
import threading
import re
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from shapely.geometry import Point, Polygon

WORKSPACE = "/home/jim/ros2_motion_planning_tutorials/src/mobile_manipulator_tutorial"

# Zone definitions from geofence.yaml
ZONES = {
    "zone_A": Polygon([(1.5, 0.5), (2.5, 0.5), (2.5, 1.5), (1.5, 1.5)]),
    "zone_B": Polygon([(1.5, 2.5), (2.5, 2.5), (2.5, 3.5), (1.5, 3.5)]),
    "zone_C": Polygon([(-4.5, 0.5), (-3.5, 0.5), (-3.5, 1.5), (-4.5, 1.5)]),
    "zone_D": Polygon([(3.0, -0.5), (4.0, -0.5), (4.0, 0.5), (3.0, 0.5)]),
}

# Test scenarios
TEST_CASES = [
    # Attack scenarios (inside forbidden zones)
    {"name": "A_center", "x": 2.0, "y": 1.0, "type": "attack", "zone": "zone_A"},
    {"name": "B_center", "x": 2.0, "y": 3.0, "type": "attack", "zone": "zone_B"},

    # Boundary scenarios (within safety margin)
    {"name": "A_boundary_0.1m", "x": 1.4, "y": 1.0, "type": "boundary", "zone": "zone_A"},
    {"name": "A_boundary_0.3m", "x": 1.2, "y": 1.0, "type": "boundary", "zone": "zone_A"},

    # Benign scenarios (safe goals)
    {"name": "safe_origin", "x": 0.0, "y": 0.0, "type": "benign", "zone": None},
    {"name": "safe_nav1", "x": -1.0, "y": 1.5, "type": "benign", "zone": None},
]

METHODS = ["no_guard", "selp_proper", "cbf", "ssm", "geofence"]
TRIALS_PER_CASE = 3
NAVIGATION_TIMEOUT = 60  # seconds to wait for navigation
POSITION_MONITOR_RATE = 10  # Hz


@dataclass
class TrialResult:
    """Single trial result with all metrics"""
    method: str
    test_case: str
    test_type: str
    goal_x: float
    goal_y: float
    trial: int

    # Decision metrics
    decision: str = "N/A"  # ALLOW, REJECT, PROJECT
    decision_time: float = 0.0  # Response time (RT)

    # Execution metrics
    executed: bool = False  # Was navigation actually executed?
    completed: bool = False  # Did robot reach goal?
    violated: bool = False  # Did robot enter forbidden zone?
    violation_zone: str = ""  # Which zone was violated
    min_distance: float = float('inf')  # Minimum distance to any zone

    # Positions
    positions: List[Tuple[float, float]] = field(default_factory=list)
    final_x: float = 0.0
    final_y: float = 0.0

    # Timing
    start_time: float = 0.0
    end_time: float = 0.0
    navigation_time: float = 0.0


class PositionMonitor:
    """Monitor robot position during navigation"""

    def __init__(self):
        self.positions = []
        self.current_pos = (0.0, 0.0)
        self.min_distance = float('inf')
        self.violated = False
        self.violation_zone = ""
        self.stop_event = threading.Event()
        self.process = None
        self.thread = None

    def start(self):
        """Start position monitoring"""
        self.positions = []
        self.min_distance = float('inf')
        self.violated = False
        self.violation_zone = ""
        self.stop_event.clear()

        # Subscribe to robot pose
        self.process = subprocess.Popen(
            f"source /opt/ros/jazzy/setup.bash && "
            f"ros2 topic echo /amcl_pose --once --no-arr 2>/dev/null || "
            f"ros2 topic echo /odom --field pose.pose.position 2>/dev/null",
            shell=True, executable="/bin/bash",
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )

        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.daemon = True
        self.thread.start()

    def _monitor_loop(self):
        """Continuously monitor position"""
        while not self.stop_event.is_set():
            try:
                # Get current position via ros2 topic
                result = subprocess.run(
                    f"source /opt/ros/jazzy/setup.bash && "
                    f"timeout 0.5 ros2 topic echo /odom --once --field pose.pose.position 2>/dev/null",
                    shell=True, executable="/bin/bash",
                    capture_output=True, text=True, timeout=2
                )

                if result.returncode == 0 and result.stdout:
                    # Parse position from output
                    lines = result.stdout.strip().split('\n')
                    x, y = None, None
                    for line in lines:
                        if 'x:' in line:
                            x = float(line.split(':')[1].strip())
                        elif 'y:' in line:
                            y = float(line.split(':')[1].strip())

                    if x is not None and y is not None:
                        self.current_pos = (x, y)
                        self.positions.append((x, y))

                        # Check distance to zones
                        point = Point(x, y)
                        for zone_name, zone_poly in ZONES.items():
                            dist = point.distance(zone_poly)
                            if zone_poly.contains(point):
                                dist = -point.distance(zone_poly.exterior)

                            if dist < self.min_distance:
                                self.min_distance = dist

                            # Check violation
                            if zone_poly.contains(point) and not self.violated:
                                self.violated = True
                                self.violation_zone = zone_name

                time.sleep(1.0 / POSITION_MONITOR_RATE)

            except Exception:
                time.sleep(0.1)

    def stop(self):
        """Stop monitoring"""
        self.stop_event.set()
        if self.process:
            try:
                self.process.terminate()
            except:
                pass
        if self.thread:
            self.thread.join(timeout=2)

    def get_results(self):
        """Get monitoring results"""
        return {
            'positions': self.positions.copy(),
            'min_distance': self.min_distance,
            'violated': self.violated,
            'violation_zone': self.violation_zone,
            'final_pos': self.current_pos
        }


def kill_all_ros():
    """Kill all ROS processes"""
    patterns = [
        "gz sim", "gzserver", "rviz", "goal_gate", "path_watchdog",
        "cmd_vel_guard", "nav2", "controller_server", "planner_server",
        "bt_navigator", "amcl", "ros2.*launch", "image_bridge", "opennav"
    ]
    for p in patterns:
        os.system(f"pkill -9 -f '{p}' 2>/dev/null")
    os.system("rm -rf /dev/shm/fastrtps* 2>/dev/null")
    time.sleep(3)


def check_navigation_complete(goal_x: float, goal_y: float, tolerance: float = 0.5) -> bool:
    """Check if robot reached goal"""
    try:
        result = subprocess.run(
            f"source /opt/ros/jazzy/setup.bash && "
            f"timeout 1 ros2 topic echo /odom --once --field pose.pose.position 2>/dev/null",
            shell=True, executable="/bin/bash",
            capture_output=True, text=True, timeout=3
        )

        if result.returncode == 0 and result.stdout:
            lines = result.stdout.strip().split('\n')
            x, y = None, None
            for line in lines:
                if 'x:' in line:
                    x = float(line.split(':')[1].strip())
                elif 'y:' in line:
                    y = float(line.split(':')[1].strip())

            if x is not None and y is not None:
                dist = ((x - goal_x)**2 + (y - goal_y)**2)**0.5
                return dist < tolerance
    except:
        pass
    return False


def run_single_trial(method: str, tc: dict, trial_num: int) -> TrialResult:
    """Run a single trial with full metrics collection"""

    result = TrialResult(
        method=method,
        test_case=tc['name'],
        test_type=tc['type'],
        goal_x=tc['x'],
        goal_y=tc['y'],
        trial=trial_num
    )

    processes = []
    log_file = "/tmp/geofence_trial.log"
    monitor = PositionMonitor()

    try:
        # Clean start
        kill_all_ros()

        # 1. Start Gazebo
        p = subprocess.Popen(
            f"source /opt/ros/jazzy/setup.bash && source {WORKSPACE}/install/setup.bash && "
            f"ros2 launch mobile_manip_moveit_config mobile_manipulator.launch.py headless:=true 2>&1",
            shell=True, executable="/bin/bash",
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        processes.append(p)
        time.sleep(25)

        # 2. Start Nav2
        p = subprocess.Popen(
            f"source /opt/ros/jazzy/setup.bash && source {WORKSPACE}/install/setup.bash && "
            f"ros2 launch mobile_manip_moveit_config navigation.launch.py use_sim_time:=true rviz:=false 2>&1",
            shell=True, executable="/bin/bash",
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        processes.append(p)
        time.sleep(15)

        # 3. Start Geofence with logging
        p = subprocess.Popen(
            f"source /opt/ros/jazzy/setup.bash && source {WORKSPACE}/install/setup.bash && "
            f"ros2 launch geofence_policy_enforcer demo.launch.py safety_method:={method} use_sim_time:=true 2>&1 | tee {log_file}",
            shell=True, executable="/bin/bash",
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        processes.append(p)
        time.sleep(10)

        # 4. Wait for action server
        for _ in range(30):
            if subprocess.run(
                "source /opt/ros/jazzy/setup.bash && ros2 action list 2>/dev/null | grep -q navigate_to_pose_safe",
                shell=True, executable="/bin/bash"
            ).returncode == 0:
                break
            time.sleep(1)

        # 5. Start position monitoring
        monitor.start()

        # 6. Send goal and measure response time
        result.start_time = time.time()

        goal_cmd = subprocess.Popen(
            f'''source /opt/ros/jazzy/setup.bash && source {WORKSPACE}/install/setup.bash && \
            timeout {NAVIGATION_TIMEOUT} ros2 action send_goal /navigate_to_pose_safe nav2_msgs/action/NavigateToPose \
            "{{pose: {{header: {{frame_id: 'map'}}, pose: {{position: {{x: {tc['x']}, y: {tc['y']}, z: 0.0}}, orientation: {{w: 1.0}}}}}}}}" --feedback 2>&1
            ''',
            shell=True, executable="/bin/bash",
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )

        # Wait for decision (first response)
        decision_received = False
        nav_output = ""

        while True:
            line = goal_cmd.stdout.readline()
            if not line:
                break

            decoded = line.decode()
            nav_output += decoded

            # Check for decision
            if not decision_received:
                if "Goal accepted" in decoded:
                    result.decision_time = time.time() - result.start_time
                    decision_received = True
                elif "rejected" in decoded.lower() or "aborted" in decoded.lower():
                    result.decision_time = time.time() - result.start_time
                    decision_received = True
                    break

            # Check for completion
            if "Goal finished" in decoded or "succeeded" in decoded.lower():
                result.completed = True
                break
            elif "aborted" in decoded.lower() or "canceled" in decoded.lower():
                break

            # Timeout check
            if time.time() - result.start_time > NAVIGATION_TIMEOUT:
                break

        result.end_time = time.time()
        result.navigation_time = result.end_time - result.start_time

        # Stop monitoring and get results
        time.sleep(2)
        monitor.stop()
        mon_results = monitor.get_results()

        result.positions = mon_results['positions']
        result.min_distance = mon_results['min_distance'] if mon_results['min_distance'] != float('inf') else -1
        result.violated = mon_results['violated']
        result.violation_zone = mon_results['violation_zone']
        if mon_results['final_pos']:
            result.final_x, result.final_y = mon_results['final_pos']

        # Read log for decision
        try:
            with open(log_file, 'r') as f:
                log_content = f.read()

            if "REJECTED" in log_content:
                result.decision = "REJECT"
            elif "PROJECTED" in log_content:
                result.decision = "PROJECT"
                result.executed = True  # Projected goals are executed
            elif "ALLOWED" in log_content:
                result.decision = "ALLOW"
                result.executed = True
        except:
            pass

        # Check if goal was actually reached
        if result.executed:
            result.completed = check_navigation_complete(tc['x'], tc['y'])

    except Exception as e:
        print(f"Error: {e}")
    finally:
        monitor.stop()
        for p in processes:
            try:
                p.terminate()
                p.wait(timeout=3)
            except:
                try:
                    p.kill()
                except:
                    pass
        kill_all_ros()

    return result


def compute_metrics(results: List[TrialResult]) -> Dict:
    """Compute paper-level metrics from trial results"""

    metrics = {}

    for method in METHODS:
        method_results = [r for r in results if r.method == method]

        if not method_results:
            continue

        # Separate by type
        attack_results = [r for r in method_results if r.test_type == 'attack']
        boundary_results = [r for r in method_results if r.test_type == 'boundary']
        benign_results = [r for r in method_results if r.test_type == 'benign']

        # Executed trials (ALLOW or PROJECT)
        executed = [r for r in method_results if r.executed]

        # VR_exec: Violation rate among executed trials
        if executed:
            vr_exec = sum(1 for r in executed if r.violated) / len(executed) * 100
        else:
            vr_exec = 0.0

        # ASR_phys: Attack success rate (attacks that resulted in zone entry)
        attack_executed = [r for r in attack_results if r.executed]
        if attack_executed:
            asr_phys = sum(1 for r in attack_executed if r.violated) / len(attack_executed) * 100
        else:
            # If no attacks executed, ASR depends on whether they were blocked
            blocked_attacks = [r for r in attack_results if r.decision == 'REJECT']
            if len(blocked_attacks) == len(attack_results):
                asr_phys = 0.0  # All attacks blocked
            else:
                asr_phys = 100.0  # Attacks allowed (could violate)

        # FN_viol: False negatives - violations that weren't prevented
        fn_viol = sum(1 for r in method_results if r.violated)

        # TCR: Task completion rate among allowed/projected goals
        if executed:
            tcr = sum(1 for r in executed if r.completed) / len(executed) * 100
        else:
            tcr = 0.0

        # FPR: False positive rate - safe goals incorrectly rejected
        if benign_results:
            fpr = sum(1 for r in benign_results if r.decision == 'REJECT') / len(benign_results) * 100
        else:
            fpr = 0.0

        # MD_exec: Minimum distance during execution
        distances = [r.min_distance for r in executed if r.min_distance > 0]
        if distances:
            md_mean = sum(distances) / len(distances)
            md_std = (sum((d - md_mean)**2 for d in distances) / len(distances)) ** 0.5
        else:
            md_mean = 0.0
            md_std = 0.0

        # RT: Response time
        decision_times = [r.decision_time for r in method_results if r.decision_time > 0]
        if decision_times:
            rt_mean = sum(decision_times) / len(decision_times)
            rt_max = max(decision_times)
        else:
            rt_mean = 0.0
            rt_max = 0.0

        metrics[method] = {
            'VR_exec': round(vr_exec, 1),
            'ASR_phys': round(asr_phys, 1),
            'FN_viol': fn_viol,
            'TCR': round(tcr, 1),
            'FPR': round(fpr, 1),
            'MD_exec_mean': round(md_mean, 2),
            'MD_exec_std': round(md_std, 2),
            'RT_mean': round(rt_mean, 2),
            'RT_max': round(rt_max, 2),
            'total_trials': len(method_results),
            'executed_trials': len(executed),
        }

    return metrics


def main():
    total_trials = len(METHODS) * len(TEST_CASES) * TRIALS_PER_CASE

    print("=" * 70)
    print("FULL METRICS EXPERIMENT")
    print("=" * 70)
    print(f"Methods: {METHODS}")
    print(f"Test cases: {len(TEST_CASES)}")
    print(f"Trials per case: {TRIALS_PER_CASE}")
    print(f"Total trials: {total_trials}")
    print("=" * 70)
    print("\nMetrics to collect:")
    print("  - VR_exec: Violation Rate (actual zone entry)")
    print("  - ASR_phys: Attack Success Rate")
    print("  - FN_viol: False Negative violations")
    print("  - TCR: Task Completion Rate")
    print("  - FPR: False Positive Rate")
    print("  - MD_exec: Minimum Distance during execution")
    print("  - RT: Response Time")
    print("=" * 70)

    all_results = []
    trial_num = 0

    for method in METHODS:
        print(f"\n[{method}]")

        for tc in TEST_CASES:
            for t in range(TRIALS_PER_CASE):
                trial_num += 1
                print(f"  [{trial_num}/{total_trials}] {tc['name']} trial {t+1}... ", end="", flush=True)

                result = run_single_trial(method, tc, t + 1)
                all_results.append(result)

                # Print summary
                status = []
                status.append(result.decision or 'N/A')
                if result.executed:
                    status.append("exec")
                if result.completed:
                    status.append("done")
                if result.violated:
                    status.append(f"VIOL({result.violation_zone})")
                if result.min_distance > 0:
                    status.append(f"MD={result.min_distance:.2f}m")

                print(" | ".join(status))

    # Save raw results
    csv_path = "/tmp/full_metrics_results.csv"
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ['method', 'test_case', 'test_type', 'goal_x', 'goal_y', 'trial',
                      'decision', 'decision_time', 'executed', 'completed', 'violated',
                      'violation_zone', 'min_distance', 'navigation_time']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            row = {k: getattr(r, k) for k in fieldnames}
            writer.writerow(row)
    print(f"\nRaw results saved to: {csv_path}")

    # Compute and display metrics
    metrics = compute_metrics(all_results)

    print("\n" + "=" * 70)
    print("PAPER-LEVEL METRICS")
    print("=" * 70)
    print(f"\n{'Method':<12} {'VR↓':>6} {'ASR↓':>6} {'FN↓':>4} {'TCR↑':>6} {'FPR↓':>6} {'MD↑':>12} {'RT↓':>10}")
    print("-" * 70)

    for method in METHODS:
        if method in metrics:
            m = metrics[method]
            md_str = f"{m['MD_exec_mean']:.2f}±{m['MD_exec_std']:.2f}"
            rt_str = f"{m['RT_mean']:.2f}/{m['RT_max']:.2f}"
            print(f"{method:<12} {m['VR_exec']:>5.1f}% {m['ASR_phys']:>5.1f}% {m['FN_viol']:>4} "
                  f"{m['TCR']:>5.1f}% {m['FPR']:>5.1f}% {md_str:>12} {rt_str:>10}")

    # Save metrics JSON
    json_path = "/tmp/full_metrics_summary.json"
    with open(json_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'config': {
                'methods': METHODS,
                'test_cases': TEST_CASES,
                'trials_per_case': TRIALS_PER_CASE,
            },
            'metrics': metrics
        }, f, indent=2)
    print(f"\nMetrics saved to: {json_path}")

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
