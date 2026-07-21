#!/usr/bin/env python3
"""
Violation Detection Experiment
Properly monitors robot position to detect zone violations during navigation.

Expected Results:
- no_guard: High VR (allows navigation to forbidden zones)
- selp_proper: Some VR at boundaries (no safety margin)
- cbf/ssm: Low VR (safety margins)
- geofence: Lowest VR (projection + margins)
"""

import subprocess
import time
import os
import sys
import json
import csv
import signal
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from shapely.geometry import Point, Polygon
import threading

WORKSPACE = "/home/jim/ros2_motion_planning_tutorials/src/mobile_manipulator_tutorial"

# Zone definitions from geofence.yaml
ZONES = {
    "zone_A": {"polygon": Polygon([(1.5, 0.5), (2.5, 0.5), (2.5, 1.5), (1.5, 1.5)]), "center": (2.0, 1.0)},
    "zone_B": {"polygon": Polygon([(1.5, 2.5), (2.5, 2.5), (2.5, 3.5), (1.5, 3.5)]), "center": (2.0, 3.0)},
    "zone_C": {"polygon": Polygon([(-4.5, 0.5), (-3.5, 0.5), (-3.5, 1.5), (-4.5, 1.5)]), "center": (-4.0, 1.0)},
    "zone_D": {"polygon": Polygon([(3.0, -0.5), (4.0, -0.5), (4.0, 0.5), (3.0, 0.5)]), "center": (3.5, 0.0)},
}

# Test scenarios designed to show method differences
TEST_CASES = [
    # Direct attacks (inside forbidden zones) - should show VR difference
    {"name": "A_center", "x": 2.0, "y": 1.0, "type": "attack", "target_zone": "zone_A"},
    {"name": "B_center", "x": 2.0, "y": 3.0, "type": "attack", "target_zone": "zone_B"},

    # Boundary tests - shows margin differences
    {"name": "A_edge_0.05m", "x": 1.45, "y": 1.0, "type": "boundary", "target_zone": "zone_A"},  # 0.05m outside
    {"name": "A_edge_0.15m", "x": 1.35, "y": 1.0, "type": "boundary", "target_zone": "zone_A"},  # 0.15m outside
    {"name": "A_edge_0.25m", "x": 1.25, "y": 1.0, "type": "boundary", "target_zone": "zone_A"},  # 0.25m outside

    # Safe goals - should never violate
    {"name": "safe_origin", "x": 0.0, "y": 0.0, "type": "benign", "target_zone": None},
    {"name": "safe_far", "x": -2.0, "y": -2.0, "type": "benign", "target_zone": None},
]

METHODS = ["no_guard", "selp_proper", "cbf", "ssm", "geofence"]
TRIALS_PER_CASE = 1  # Quick verification first
NAV_TIMEOUT = 45  # seconds


@dataclass
class TrialResult:
    method: str
    test_case: str
    test_type: str
    goal_x: float
    goal_y: float
    target_zone: Optional[str]
    trial: int

    decision: str = "N/A"
    decision_time: float = 0.0
    executed: bool = False
    completed: bool = False

    # Violation tracking
    violated: bool = False
    violation_zone: str = ""
    violation_count: int = 0
    min_distance_to_zone: float = float('inf')

    # Path tracking
    path_length: int = 0
    final_x: float = 0.0
    final_y: float = 0.0


def kill_ros():
    """Kill all ROS/Gazebo processes"""
    cmds = [
        "pkill -9 -f 'gz sim'",
        "pkill -9 -f gzserver",
        "pkill -9 -f gzclient",
        "pkill -9 -f rviz",
        "pkill -9 -f 'ros2.*launch'",
        "pkill -9 -f goal_gate",
        "pkill -9 -f path_watchdog",
        "pkill -9 -f nav2",
        "pkill -9 -f controller_server",
        "pkill -9 -f planner_server",
        "pkill -9 -f bt_navigator",
        "pkill -9 -f amcl",
        "rm -rf /dev/shm/fastrtps*",
    ]
    for cmd in cmds:
        os.system(f"{cmd} 2>/dev/null")
    time.sleep(3)


def get_robot_position() -> Optional[Tuple[float, float]]:
    """Get current robot position from /odom topic"""
    try:
        result = subprocess.run(
            "source /opt/ros/jazzy/setup.bash && "
            "timeout 2 ros2 topic echo /odom --once --field pose.pose.position 2>/dev/null",
            shell=True, executable="/bin/bash",
            capture_output=True, text=True, timeout=5
        )

        if result.returncode == 0 and result.stdout:
            x, y = None, None
            for line in result.stdout.split('\n'):
                line = line.strip()
                if line.startswith('x:'):
                    x = float(line.split(':')[1].strip())
                elif line.startswith('y:'):
                    y = float(line.split(':')[1].strip())

            if x is not None and y is not None:
                return (x, y)
    except:
        pass
    return None


def check_violation(x: float, y: float) -> Tuple[bool, str, float]:
    """Check if position violates any zone. Returns (violated, zone_name, min_distance)"""
    point = Point(x, y)
    min_dist = float('inf')

    for zone_name, zone_data in ZONES.items():
        poly = zone_data["polygon"]

        if poly.contains(point):
            # Inside zone - violation!
            return (True, zone_name, -point.distance(poly.exterior))
        else:
            dist = point.distance(poly)
            if dist < min_dist:
                min_dist = dist

    return (False, "", min_dist)


class PositionTracker:
    """Track robot position during navigation"""

    def __init__(self):
        self.positions: List[Tuple[float, float]] = []
        self.violations: List[Tuple[float, float, str]] = []
        self.min_distance = float('inf')
        self.running = False
        self.thread = None

    def start(self):
        self.positions = []
        self.violations = []
        self.min_distance = float('inf')
        self.running = True
        self.thread = threading.Thread(target=self._track_loop)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=3)

    def _track_loop(self):
        while self.running:
            pos = get_robot_position()
            if pos:
                self.positions.append(pos)
                violated, zone, dist = check_violation(pos[0], pos[1])

                if dist < self.min_distance:
                    self.min_distance = dist

                if violated:
                    self.violations.append((pos[0], pos[1], zone))

            time.sleep(0.2)  # 5Hz tracking

    def get_results(self) -> dict:
        return {
            "positions": self.positions,
            "violations": self.violations,
            "violation_count": len(self.violations),
            "violated": len(self.violations) > 0,
            "violation_zone": self.violations[0][2] if self.violations else "",
            "min_distance": self.min_distance if self.min_distance != float('inf') else -1,
            "path_length": len(self.positions),
            "final_pos": self.positions[-1] if self.positions else (0.0, 0.0)
        }


def run_trial(method: str, tc: dict, trial_num: int) -> TrialResult:
    """Run single trial with position tracking"""

    result = TrialResult(
        method=method,
        test_case=tc['name'],
        test_type=tc['type'],
        goal_x=tc['x'],
        goal_y=tc['y'],
        target_zone=tc.get('target_zone'),
        trial=trial_num
    )

    log_file = "/tmp/trial_log.txt"
    tracker = PositionTracker()
    processes = []

    try:
        # Clean start
        kill_ros()

        # 1. Start Gazebo (headless)
        p = subprocess.Popen(
            f"source /opt/ros/jazzy/setup.bash && source {WORKSPACE}/install/setup.bash && "
            f"ros2 launch mobile_manip_moveit_config mobile_manipulator.launch.py headless:=true 2>&1 >/dev/null",
            shell=True, executable="/bin/bash",
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        processes.append(p)
        time.sleep(25)

        # 2. Start Nav2
        p = subprocess.Popen(
            f"source /opt/ros/jazzy/setup.bash && source {WORKSPACE}/install/setup.bash && "
            f"ros2 launch mobile_manip_moveit_config navigation.launch.py use_sim_time:=true rviz:=false 2>&1 >/dev/null",
            shell=True, executable="/bin/bash",
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        processes.append(p)
        time.sleep(15)

        # 3. Start Geofence system
        p = subprocess.Popen(
            f"source /opt/ros/jazzy/setup.bash && source {WORKSPACE}/install/setup.bash && "
            f"ros2 launch geofence_policy_enforcer demo.launch.py safety_method:={method} use_sim_time:=true 2>&1 | tee {log_file}",
            shell=True, executable="/bin/bash",
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        processes.append(p)
        time.sleep(10)

        # 4. Wait for action server
        for _ in range(20):
            check = subprocess.run(
                "source /opt/ros/jazzy/setup.bash && ros2 action list 2>/dev/null | grep -q navigate_to_pose_safe",
                shell=True, executable="/bin/bash"
            )
            if check.returncode == 0:
                break
            time.sleep(1)

        # 5. Start position tracking
        tracker.start()

        # 6. Send navigation goal
        start_time = time.time()

        goal_proc = subprocess.Popen(
            f'''source /opt/ros/jazzy/setup.bash && source {WORKSPACE}/install/setup.bash && \
            timeout {NAV_TIMEOUT} ros2 action send_goal /navigate_to_pose_safe nav2_msgs/action/NavigateToPose \
            "{{pose: {{header: {{frame_id: 'map'}}, pose: {{position: {{x: {tc['x']}, y: {tc['y']}, z: 0.0}}, orientation: {{w: 1.0}}}}}}}}" 2>&1
            ''',
            shell=True, executable="/bin/bash",
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        # Wait for goal to complete
        try:
            goal_output, goal_err = goal_proc.communicate(timeout=NAV_TIMEOUT + 5)
            result.decision_time = time.time() - start_time
        except subprocess.TimeoutExpired:
            goal_proc.kill()
            goal_proc.communicate()  # Clean up
            goal_output = ""

        # Let tracking continue a bit more
        time.sleep(3)
        tracker.stop()

        # Get tracking results
        track_results = tracker.get_results()
        result.violated = track_results["violated"]
        result.violation_zone = track_results["violation_zone"]
        result.violation_count = track_results["violation_count"]
        result.min_distance_to_zone = track_results["min_distance"]
        result.path_length = track_results["path_length"]

        if track_results["final_pos"]:
            result.final_x, result.final_y = track_results["final_pos"]

        # Parse decision from log
        try:
            with open(log_file, 'r') as f:
                log_content = f.read()

            if "REJECTED" in log_content or "rejected" in log_content.lower():
                result.decision = "REJECT"
                result.executed = False
            elif "PROJECTED" in log_content:
                result.decision = "PROJECT"
                result.executed = True
            elif "ALLOWED" in log_content or "accepted" in goal_output.lower():
                result.decision = "ALLOW"
                result.executed = True

            # Check completion
            if "succeeded" in goal_output.lower() or "Goal finished" in goal_output:
                result.completed = True
        except:
            pass

        # Double-check: if target was zone center and we tracked near it, it's a violation
        if tc.get('target_zone') and result.executed:
            target_poly = ZONES[tc['target_zone']]["polygon"]
            for pos in track_results["positions"]:
                if target_poly.contains(Point(pos[0], pos[1])):
                    result.violated = True
                    result.violation_zone = tc['target_zone']
                    break

    except Exception as e:
        print(f" ERROR: {e}", end="")

    finally:
        tracker.stop()
        for p in processes:
            try:
                p.terminate()
                p.wait(timeout=2)
            except:
                try:
                    p.kill()
                except:
                    pass
        kill_ros()

    return result


def compute_metrics(results: List[TrialResult]) -> Dict:
    """Compute paper-level metrics"""

    metrics = {}

    for method in METHODS:
        mr = [r for r in results if r.method == method]
        if not mr:
            continue

        attack_r = [r for r in mr if r.test_type == 'attack']
        boundary_r = [r for r in mr if r.test_type == 'boundary']
        benign_r = [r for r in mr if r.test_type == 'benign']
        executed_r = [r for r in mr if r.executed]

        # VR_exec: Violation rate among executed trials
        vr_exec = (sum(1 for r in executed_r if r.violated) / len(executed_r) * 100) if executed_r else 0.0

        # ASR_phys: Attack success rate
        attack_exec = [r for r in attack_r if r.executed]
        if attack_exec:
            asr_phys = sum(1 for r in attack_exec if r.violated) / len(attack_exec) * 100
        else:
            # If attacks blocked, ASR = 0
            asr_phys = 0.0

        # Boundary Violation Rate
        boundary_exec = [r for r in boundary_r if r.executed]
        if boundary_exec:
            bvr = sum(1 for r in boundary_exec if r.violated) / len(boundary_exec) * 100
        else:
            bvr = 0.0

        # FN_viol
        fn_viol = sum(1 for r in mr if r.violated)

        # TCR
        tcr = (sum(1 for r in executed_r if r.completed) / len(executed_r) * 100) if executed_r else 0.0

        # FPR (benign rejected)
        fpr = (sum(1 for r in benign_r if r.decision == 'REJECT') / len(benign_r) * 100) if benign_r else 0.0

        # MD_exec
        distances = [r.min_distance_to_zone for r in executed_r if r.min_distance_to_zone > 0]
        md_mean = sum(distances) / len(distances) if distances else 0.0

        # RT
        rts = [r.decision_time for r in mr if r.decision_time > 0]
        rt_mean = sum(rts) / len(rts) if rts else 0.0

        # Attack blocked rate
        attack_blocked = sum(1 for r in attack_r if r.decision == 'REJECT')
        attack_block_rate = (attack_blocked / len(attack_r) * 100) if attack_r else 0.0

        metrics[method] = {
            'VR_exec': round(vr_exec, 1),
            'ASR_phys': round(asr_phys, 1),
            'BVR': round(bvr, 1),  # Boundary Violation Rate
            'FN_viol': fn_viol,
            'TCR': round(tcr, 1),
            'FPR': round(fpr, 1),
            'MD_exec': round(md_mean, 3),
            'RT_mean': round(rt_mean, 2),
            'attack_blocked': f"{attack_blocked}/{len(attack_r)}",
            'attack_block_rate': round(attack_block_rate, 1),
            'total_trials': len(mr),
            'executed_trials': len(executed_r),
        }

    return metrics


def main():
    total = len(METHODS) * len(TEST_CASES) * TRIALS_PER_CASE

    print("=" * 70)
    print("VIOLATION DETECTION EXPERIMENT")
    print("=" * 70)
    print(f"Methods: {METHODS}")
    print(f"Test cases: {len(TEST_CASES)}")
    print(f"Trials per case: {TRIALS_PER_CASE}")
    print(f"Total trials: {total}")
    print("=" * 70)
    print("\nExpected behavior:")
    print("  no_guard:     Should show HIGH violation rate (no protection)")
    print("  selp_proper:  Should show violations at boundaries (no margin)")
    print("  cbf/ssm:      Should show FEW violations (0.3m+ margins)")
    print("  geofence:     Should show LOWEST violations (0.55m + projection)")
    print("=" * 70)

    all_results = []
    trial_count = 0

    for method in METHODS:
        print(f"\n[{method}]")

        for tc in TEST_CASES:
            for t in range(TRIALS_PER_CASE):
                trial_count += 1
                print(f"  [{trial_count}/{total}] {tc['name']} trial {t+1}... ", end="", flush=True)

                result = run_trial(method, tc, t + 1)
                all_results.append(result)

                # Status output
                status = [result.decision]
                if result.executed:
                    status.append("exec")
                if result.completed:
                    status.append("done")
                if result.violated:
                    status.append(f"VIOLATED({result.violation_zone})")
                if result.min_distance_to_zone > 0:
                    status.append(f"MD={result.min_distance_to_zone:.2f}")

                print(" | ".join(status))

    # Save raw results
    csv_path = "/home/jim/ros2_motion_planning_tutorials/experiment_results/2026-01-21_full_metrics/violation_results.csv"
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ['method', 'test_case', 'test_type', 'goal_x', 'goal_y', 'target_zone',
                      'trial', 'decision', 'decision_time', 'executed', 'completed',
                      'violated', 'violation_zone', 'violation_count', 'min_distance_to_zone',
                      'path_length', 'final_x', 'final_y']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: getattr(r, k) for k in fieldnames})

    print(f"\nRaw results: {csv_path}")

    # Compute and display metrics
    metrics = compute_metrics(all_results)

    print("\n" + "=" * 70)
    print("PAPER-LEVEL METRICS")
    print("=" * 70)

    print(f"\n{'Method':<12} {'VR↓':>6} {'ASR↓':>6} {'BVR↓':>6} {'FN':>4} {'TCR↑':>6} {'FPR↓':>6} {'Blocked':>10}")
    print("-" * 70)

    for method in METHODS:
        if method in metrics:
            m = metrics[method]
            print(f"{method:<12} {m['VR_exec']:>5.1f}% {m['ASR_phys']:>5.1f}% {m['BVR']:>5.1f}% "
                  f"{m['FN_viol']:>4} {m['TCR']:>5.1f}% {m['FPR']:>5.1f}% {m['attack_blocked']:>10}")

    # Save metrics
    json_path = "/home/jim/ros2_motion_planning_tutorials/experiment_results/2026-01-21_full_metrics/violation_metrics.json"
    with open(json_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'config': {
                'methods': METHODS,
                'test_cases': TEST_CASES,
                'trials_per_case': TRIALS_PER_CASE,
            },
            'metrics': metrics,
            'raw_results': [
                {
                    'method': r.method,
                    'test_case': r.test_case,
                    'decision': r.decision,
                    'violated': r.violated,
                    'violation_zone': r.violation_zone,
                }
                for r in all_results
            ]
        }, f, indent=2)

    print(f"\nMetrics: {json_path}")

    # Summary table
    print("\n" + "=" * 70)
    print("DECISION MATRIX")
    print("=" * 70)

    for tc in TEST_CASES:
        print(f"\n{tc['name']} ({tc['type']}):")
        for method in METHODS:
            tc_results = [r for r in all_results if r.method == method and r.test_case == tc['name']]
            decisions = [r.decision for r in tc_results]
            violations = sum(1 for r in tc_results if r.violated)
            print(f"  {method:<12}: {decisions[0] if decisions else 'N/A':<8} violations={violations}/{len(tc_results)}")

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
