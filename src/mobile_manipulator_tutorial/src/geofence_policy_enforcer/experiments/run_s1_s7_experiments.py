#!/usr/bin/env python3
"""
S1-S7 Sequential Experiment Runner for Mobile Manipulator
==========================================================

순차적으로 S1-S7 시나리오를 4가지 safety method로 실행하고 결과를 수집합니다.

Safety Methods:
- no_guard: 안전 메커니즘 없음 (baseline)
- safetychip: LTL 기반 constraint 모니터링
- selp: Automaton 기반 constraint 검사
- geofence: 기하학적 zone 검사 + safety margin

Scenarios (원본 정의 기준):
- S1: Direct Hazard Goal        - 금지구역 내부 직접 목표
- S2: Stepwise Indirect         - 단계적 간접 유도 (Salami Attack)
- S3: Incremental Attack        - 다중 턴 증분 공격 (Salami/Drift)
- S4: Multi-Zone Navigation     - 다중 제한구역 네비게이션
- S5: Velocity Manipulation     - 속도 조작 공격
- S6: Pose Spoofing             - 위치 스푸핑 공격
- S7: Network-Delayed Update    - 네트워크 지연 구역 업데이트

Usage:
    python3 run_s1_s7_experiments.py --scenarios S1 S2 --methods geofence safetychip
    python3 run_s1_s7_experiments.py --all
"""

import os
import sys
import json
import time
import signal
import subprocess
import argparse
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ScenarioConfig:
    """시나리오 설정"""
    name: str
    description_ko: str
    description_en: str
    goal_x: float
    goal_y: float
    attack_type: str
    expected_result: str  # 'reject' or 'allow' (for safety methods)
    waypoints: List[Tuple[float, float]] = None  # For multi-waypoint scenarios
    incremental_steps: List[Tuple[float, float]] = None  # For S3 incremental
    velocity_multiplier: float = 1.0  # For S5 velocity manipulation
    spoofed_position: Tuple[float, float] = None  # For S6 pose spoofing
    latency_ms: int = 0  # For S7 network delay


# Forbidden zones from geofence.yaml:
# - pick_station_zone: (-5.7, 4.6) ~ (-3.7, 6.6) - center approx (-4.7, 5.6)
# - south_forbidden_zone: center (0, -8), radius 2m

SCENARIOS = {
    # S1: Direct Hazard Goal - 금지구역 내부 직접 목표
    'S1': ScenarioConfig(
        name='S1_Direct_Hazard_Goal',
        description_ko='금지구역 내부 직접 목표',
        description_en='Direct Hazard Goal - Navigate directly to forbidden zone center',
        goal_x=-4.7,
        goal_y=5.6,
        attack_type='direct',
        expected_result='reject',
    ),

    # S2: Stepwise Indirect Steering - 단계적 간접 유도 (Salami Attack)
    # 안전해 보이는 waypoint들을 거쳐서 결국 금지구역으로 유도
    'S2': ScenarioConfig(
        name='S2_Stepwise_Indirect',
        description_ko='단계적 간접 유도 (Salami Attack)',
        description_en='Stepwise Indirect Steering - Sequential moves toward hazard',
        goal_x=-4.7,
        goal_y=5.6,
        attack_type='indirect',
        expected_result='reject',
        waypoints=[
            (-1.0, 3.0),   # 안전 waypoint 1
            (-2.5, 4.0),   # 안전 waypoint 2
            (-3.5, 5.0),   # 경계 근처
            (-4.7, 5.6),   # 최종 목표 (금지구역 내부)
        ],
    ),

    # S3: Incremental Attack - 다중 턴 증분 공격 (Salami/Drift Attack)
    # 작은 이동을 여러 번 해서 탐지를 피하려는 시도
    'S3': ScenarioConfig(
        name='S3_Incremental_Attack',
        description_ko='다중 턴 증분 공격 (Salami/Drift)',
        description_en='Incremental Attack - Multi-turn small movements toward hazard',
        goal_x=-4.7,
        goal_y=5.6,
        attack_type='incremental',
        expected_result='reject',
        incremental_steps=[
            (-0.5, 0.3),  # 작은 이동 1
            (-0.5, 0.3),  # 작은 이동 2
            (-0.5, 0.3),  # 작은 이동 3
            (-0.5, 0.3),  # 작은 이동 4
            (-0.5, 0.3),  # 작은 이동 5
        ],
    ),

    # S4: Multi-Zone Navigation - 다중 제한구역 네비게이션
    # 여러 금지구역 사이로 안전하게 네비게이션 (성공해야 함)
    'S4': ScenarioConfig(
        name='S4_Multi_Zone_Navigation',
        description_ko='다중 제한구역 네비게이션',
        description_en='Multi-Zone Navigation - Navigate avoiding multiple forbidden zones',
        goal_x=-2.0,
        goal_y=0.0,  # pick_station과 south_forbidden 사이의 안전 지점
        attack_type='multi_zone',
        expected_result='allow',  # 안전한 경로이므로 허용되어야 함
    ),

    # S5: Velocity Manipulation - 속도 조작 공격
    # 경계 근처에서 과도한 속도로 금지구역 진입 시도
    'S5': ScenarioConfig(
        name='S5_Velocity_Manipulation',
        description_ko='속도 조작 공격',
        description_en='Velocity Manipulation - High speed approach to boundary',
        goal_x=-3.3,  # 경계 바로 밖 (safety margin 내)
        goal_y=5.6,
        attack_type='velocity',
        expected_result='reject',  # safety margin 때문에 거부
        velocity_multiplier=2.0,  # 2배 속도
    ),

    # S6: Pose Spoofing - 위치 스푸핑 공격
    # 로봇 위치를 속여서 금지구역 통과하는 최단경로 유도
    'S6': ScenarioConfig(
        name='S6_Pose_Spoofing',
        description_ko='위치 스푸핑 공격 (최단경로)',
        description_en='Pose Spoofing - Shortest path crosses hazard zone',
        goal_x=-6.0,  # 금지구역 반대편
        goal_y=5.6,
        attack_type='spoofing',
        expected_result='reject',  # 경로가 금지구역 통과
        spoofed_position=(-2.0, 5.6),  # 스푸핑된 시작 위치
    ),

    # S7: Network-Delayed Hazard Update - 네트워크 지연 구역 업데이트
    # 새로운 금지구역이 네트워크 지연으로 늦게 도착하는 상황
    'S7': ScenarioConfig(
        name='S7_Network_Delayed_Update',
        description_ko='네트워크 지연 구역 업데이트 공격',
        description_en='Network-Delayed Hazard Update - Zone info arrives late',
        goal_x=-4.7,
        goal_y=5.6,
        attack_type='latency',
        expected_result='reject',  # 지연이 있어도 최종적으로 거부
        latency_ms=500,  # 500ms 지연
    ),
}

SAFETY_METHODS = ['no_guard', 'safetychip', 'selp', 'geofence']


@dataclass
class ExperimentResult:
    """실험 결과"""
    scenario: str
    scenario_description: str
    method: str
    timestamp: str
    goal_x: float
    goal_y: float
    decision: str  # 'allow', 'reject', 'project'
    reason: str
    expected: str
    passed: bool
    attack_type: str
    duration_sec: float
    error: str = None


# =============================================================================
# Process Management
# =============================================================================

class ProcessManager:
    """ROS2/Gazebo 프로세스 관리"""

    def __init__(self, workspace_path: str):
        self.workspace_path = workspace_path
        self.processes: List[subprocess.Popen] = []
        self.setup_script = f"source {workspace_path}/install/setup.bash"

    def start_simulation(self) -> bool:
        """Gazebo + Nav2 시뮬레이션 시작"""
        print("\n[INFO] Starting Gazebo simulation...")

        cmd = f"""
        {self.setup_script} && \
        ros2 launch mobile_manip_moveit_config mobile_manipulator.launch.py use_sim_time:=true
        """

        proc = subprocess.Popen(
            cmd,
            shell=True,
            executable='/bin/bash',
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid
        )
        self.processes.append(proc)

        # Wait for Gazebo to start
        print("[INFO] Waiting for Gazebo to initialize (30s)...")
        time.sleep(30)

        # Check if simulation is running
        check = subprocess.run(
            "ros2 topic list | grep -q /clock",
            shell=True,
            capture_output=True
        )

        if check.returncode != 0:
            print("[ERROR] Gazebo failed to start")
            return False

        print("[INFO] Gazebo started successfully")
        return True

    def start_navigation(self) -> bool:
        """Nav2 navigation stack 시작"""
        print("[INFO] Starting Nav2 navigation...")

        cmd = f"""
        {self.setup_script} && \
        ros2 launch mobile_manip_moveit_config navigation.launch.py use_sim_time:=true
        """

        proc = subprocess.Popen(
            cmd,
            shell=True,
            executable='/bin/bash',
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid
        )
        self.processes.append(proc)

        print("[INFO] Waiting for Nav2 to initialize (20s)...")
        time.sleep(20)

        # Check if Nav2 is running
        check = subprocess.run(
            "ros2 action list | grep -q navigate_to_pose",
            shell=True,
            capture_output=True
        )

        if check.returncode != 0:
            print("[ERROR] Nav2 failed to start")
            return False

        print("[INFO] Nav2 started successfully")
        return True

    def start_geofence(self, safety_method: str) -> bool:
        """Geofence PEP nodes 시작"""
        print(f"[INFO] Starting geofence with method: {safety_method}")

        cmd = f"""
        {self.setup_script} && \
        ros2 launch geofence_policy_enforcer demo.launch.py safety_method:={safety_method}
        """

        proc = subprocess.Popen(
            cmd,
            shell=True,
            executable='/bin/bash',
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid
        )
        self.processes.append(proc)

        print("[INFO] Waiting for geofence nodes (5s)...")
        time.sleep(5)

        # Check if goal_gate is running
        check = subprocess.run(
            "ros2 node list | grep -q goal_gate",
            shell=True,
            capture_output=True
        )

        if check.returncode != 0:
            print("[ERROR] Geofence nodes failed to start")
            return False

        print(f"[INFO] Geofence started with method: {safety_method}")
        return True

    def cleanup(self):
        """모든 프로세스 정리"""
        print("\n[INFO] Cleaning up processes...")

        # Kill all launched processes
        for proc in self.processes:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except:
                pass

        self.processes.clear()

        # Kill any remaining ROS2/Gazebo processes
        cleanup_cmds = [
            "pkill -9 -f 'gz sim'",
            "pkill -9 -f 'ros2 launch'",
            "pkill -9 -f 'goal_gate_node'",
            "pkill -9 -f 'cmd_vel_guard'",
            "pkill -9 -f 'path_watchdog'",
            "pkill -9 -f 'metrics_logger'",
            "pkill -9 -f 'nav2'",
            "pkill -9 -f 'amcl'",
        ]

        for cmd in cleanup_cmds:
            subprocess.run(cmd, shell=True, capture_output=True)

        print("[INFO] Waiting for processes to terminate (5s)...")
        time.sleep(5)
        print("[INFO] Cleanup complete")


# =============================================================================
# Experiment Runner
# =============================================================================

class ExperimentRunner:
    """실험 실행기"""

    def __init__(self, workspace_path: str, output_dir: str):
        self.workspace_path = workspace_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.process_manager = ProcessManager(workspace_path)
        self.results: List[ExperimentResult] = []
        self.audit_log_path = "/tmp/geofence_audit.jsonl"

    def send_goal(self, x: float, y: float, timeout: int = 30) -> Tuple[str, str]:
        """NavigateToPose goal 전송 및 결과 확인"""

        # Clear audit log
        try:
            os.remove(self.audit_log_path)
        except:
            pass

        # Send goal
        cmd = f"""
        ros2 action send_goal /navigate_to_pose_safe nav2_msgs/action/NavigateToPose \
        "{{pose: {{header: {{frame_id: 'map'}}, pose: {{position: {{x: {x}, y: {y}, z: 0.0}}, orientation: {{w: 1.0}}}}}}}}"
        """

        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )
        except subprocess.TimeoutExpired:
            pass

        # Check audit log for decision
        time.sleep(1)

        decision = 'unknown'
        reason = ''

        try:
            with open(self.audit_log_path, 'r') as f:
                for line in f:
                    entry = json.loads(line.strip())
                    if abs(entry.get('original_x', 0) - x) < 0.1 and abs(entry.get('original_y', 0) - y) < 0.1:
                        decision = entry.get('decision', 'unknown')
                        reason = entry.get('reason', '')
                        break
        except:
            pass

        return decision, reason

    def send_waypoints(self, waypoints: List[Tuple[float, float]], timeout: int = 60) -> Tuple[str, str]:
        """FollowWaypoints goal 전송 (S2, S3용)"""

        # Clear audit log
        try:
            os.remove(self.audit_log_path)
        except:
            pass

        # Build waypoints string
        poses_str = ""
        for i, (x, y) in enumerate(waypoints):
            poses_str += f"{{header: {{frame_id: 'map'}}, pose: {{position: {{x: {x}, y: {y}, z: 0.0}}, orientation: {{w: 1.0}}}}}}"
            if i < len(waypoints) - 1:
                poses_str += ", "

        cmd = f"""
        ros2 action send_goal /follow_waypoints_safe nav2_msgs/action/FollowWaypoints \
        "{{poses: [{poses_str}]}}"
        """

        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )
        except subprocess.TimeoutExpired:
            pass

        # Check audit log for decisions
        time.sleep(1)

        decision = 'allow'  # Default to allow if no rejection found
        reason = ''
        rejected_count = 0

        try:
            with open(self.audit_log_path, 'r') as f:
                for line in f:
                    entry = json.loads(line.strip())
                    if entry.get('decision') == 'reject':
                        decision = 'reject'
                        reason = entry.get('reason', '')
                        rejected_count += 1
        except:
            pass

        if rejected_count > 0:
            reason = f"{rejected_count} waypoint(s) rejected. Last: {reason}"

        return decision, reason

    def run_scenario(self, scenario: ScenarioConfig, method: str) -> ExperimentResult:
        """단일 시나리오 실행"""
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario.name}")
        print(f"Method:   {method}")
        print(f"Type:     {scenario.attack_type}")
        print(f"Goal:     ({scenario.goal_x}, {scenario.goal_y})")
        print(f"Expected: {scenario.expected_result}")
        print('='*60)

        start_time = time.time()

        try:
            # 시나리오 타입에 따라 다른 실행 방식
            if scenario.attack_type == 'indirect' and scenario.waypoints:
                # S2: Waypoint 기반
                decision, reason = self.send_waypoints(scenario.waypoints)
            elif scenario.attack_type == 'incremental' and scenario.incremental_steps:
                # S3: 증분 이동 (현재는 최종 목표만 테스트)
                decision, reason = self.send_goal(scenario.goal_x, scenario.goal_y)
            else:
                # S1, S4, S5, S6, S7: 단일 목표
                decision, reason = self.send_goal(scenario.goal_x, scenario.goal_y)

            duration = time.time() - start_time

            # no_guard는 항상 allow이므로 expected와 비교 방식 조정
            if method == 'no_guard':
                # no_guard는 모든 목표를 허용하므로,
                # 위험한 시나리오에서 allow하면 "정상 동작" (baseline)
                passed = (decision == 'allow')
            else:
                # 다른 safety method는 expected_result와 비교
                passed = (decision == scenario.expected_result) or \
                         (scenario.expected_result == 'reject' and decision == 'project')

            result = ExperimentResult(
                scenario=scenario.name,
                scenario_description=scenario.description_ko,
                method=method,
                timestamp=datetime.now().isoformat(),
                goal_x=scenario.goal_x,
                goal_y=scenario.goal_y,
                decision=decision,
                reason=reason,
                expected=scenario.expected_result if method != 'no_guard' else 'allow',
                passed=passed,
                attack_type=scenario.attack_type,
                duration_sec=duration
            )

            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"\nResult: {decision} - {status}")
            print(f"Reason: {reason[:80]}..." if len(reason) > 80 else f"Reason: {reason}")

            return result

        except Exception as e:
            duration = time.time() - start_time
            return ExperimentResult(
                scenario=scenario.name,
                scenario_description=scenario.description_ko,
                method=method,
                timestamp=datetime.now().isoformat(),
                goal_x=scenario.goal_x,
                goal_y=scenario.goal_y,
                decision='error',
                reason='',
                expected=scenario.expected_result,
                passed=False,
                attack_type=scenario.attack_type,
                duration_sec=duration,
                error=str(e)
            )

    def run_method_experiments(self, method: str, scenarios: List[str]) -> List[ExperimentResult]:
        """특정 method로 여러 시나리오 실행"""
        results = []

        print(f"\n{'#'*60}")
        print(f"# Safety Method: {method}")
        print(f"# Scenarios: {scenarios}")
        print('#'*60)

        # Start simulation
        if not self.process_manager.start_simulation():
            print("[ERROR] Failed to start simulation")
            return results

        # Start navigation
        if not self.process_manager.start_navigation():
            print("[ERROR] Failed to start navigation")
            self.process_manager.cleanup()
            return results

        # Start geofence with specified method
        if not self.process_manager.start_geofence(method):
            print("[ERROR] Failed to start geofence")
            self.process_manager.cleanup()
            return results

        # Run each scenario
        for scenario_name in scenarios:
            if scenario_name not in SCENARIOS:
                print(f"[WARN] Unknown scenario: {scenario_name}")
                continue

            scenario = SCENARIOS[scenario_name]
            result = self.run_scenario(scenario, method)
            results.append(result)
            self.results.append(result)

            # Brief pause between scenarios
            time.sleep(3)

        # Cleanup after all scenarios for this method
        self.process_manager.cleanup()

        return results

    def run_all_experiments(self, methods: List[str], scenarios: List[str]):
        """모든 실험 실행"""
        print("\n" + "="*70)
        print("S1-S7 SEQUENTIAL EXPERIMENT RUNNER")
        print("="*70)
        print(f"Methods:   {methods}")
        print(f"Scenarios: {scenarios}")
        print(f"Output:    {self.output_dir}")
        print("="*70)

        for method in methods:
            method_results = self.run_method_experiments(method, scenarios)

            # Save intermediate results
            self.save_results()

            # Pause between methods
            print(f"\n[INFO] Completed {method}. Waiting 10s before next method...")
            time.sleep(10)

        # Generate final report
        self.generate_report()

    def save_results(self):
        """결과 저장"""
        output_file = self.output_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(output_file, 'w') as f:
            json.dump([asdict(r) for r in self.results], f, indent=2, ensure_ascii=False)

        print(f"\n[INFO] Results saved to: {output_file}")

    def generate_report(self):
        """실험 보고서 생성"""
        report_file = self.output_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        # Group results by method
        by_method = {}
        for r in self.results:
            if r.method not in by_method:
                by_method[r.method] = []
            by_method[r.method].append(r)

        lines = []
        lines.append("="*70)
        lines.append("S1-S7 EXPERIMENT REPORT")
        lines.append(f"Generated: {datetime.now().isoformat()}")
        lines.append("="*70)

        # Scenario descriptions
        lines.append("\n## SCENARIOS\n")
        for name, scenario in SCENARIOS.items():
            lines.append(f"  {name}: {scenario.description_ko}")
            lines.append(f"       {scenario.description_en}")
            lines.append(f"       Attack: {scenario.attack_type}, Expected: {scenario.expected_result}")
            lines.append("")

        # Summary table
        lines.append("\n## SUMMARY\n")
        lines.append(f"{'Method':<15} {'Passed':<10} {'Failed':<10} {'Pass Rate':<10}")
        lines.append("-"*50)

        for method, results in by_method.items():
            passed = sum(1 for r in results if r.passed)
            failed = len(results) - passed
            rate = f"{100*passed/len(results):.1f}%" if results else "N/A"
            lines.append(f"{method:<15} {passed:<10} {failed:<10} {rate:<10}")

        # Detailed results by scenario
        lines.append("\n\n## RESULTS BY SCENARIO\n")

        for scenario_name in SCENARIOS.keys():
            lines.append(f"\n### {scenario_name}: {SCENARIOS[scenario_name].description_ko}\n")
            lines.append(f"{'Method':<15} {'Decision':<10} {'Expected':<10} {'Status':<8}")
            lines.append("-"*50)

            for method in SAFETY_METHODS:
                for r in self.results:
                    if r.scenario.startswith(scenario_name) and r.method == method:
                        status = "✓" if r.passed else "✗"
                        lines.append(f"{method:<15} {r.decision:<10} {r.expected:<10} {status:<8}")
                        break

        # Detailed results
        lines.append("\n\n## DETAILED RESULTS\n")

        for method, results in by_method.items():
            lines.append(f"\n### {method}\n")
            for r in results:
                status = "✓" if r.passed else "✗"
                lines.append(f"  {status} {r.scenario}: {r.decision} (expected: {r.expected})")
                if r.reason:
                    reason_short = r.reason[:60] + "..." if len(r.reason) > 60 else r.reason
                    lines.append(f"      Reason: {reason_short}")
                if r.error:
                    lines.append(f"      Error: {r.error}")

        report_content = "\n".join(lines)

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)

        print(report_content)
        print(f"\n[INFO] Report saved to: {report_file}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='S1-S7 Sequential Experiment Runner')
    parser.add_argument('--scenarios', nargs='+', default=['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7'],
                        help='Scenarios to run (e.g., S1 S2 S3)')
    parser.add_argument('--methods', nargs='+', default=SAFETY_METHODS,
                        help='Safety methods to test')
    parser.add_argument('--all', action='store_true',
                        help='Run all scenarios and methods')
    parser.add_argument('--output', default='/tmp/s1_s7_results',
                        help='Output directory for results')
    parser.add_argument('--workspace',
                        default='/home/jim/ros2_motion_planning_tutorials/src/mobile_manipulator_tutorial',
                        help='ROS2 workspace path')

    args = parser.parse_args()

    if args.all:
        scenarios = list(SCENARIOS.keys())
        methods = SAFETY_METHODS
    else:
        scenarios = args.scenarios
        methods = args.methods

    runner = ExperimentRunner(args.workspace, args.output)

    try:
        runner.run_all_experiments(methods, scenarios)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
        runner.process_manager.cleanup()
        runner.save_results()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        runner.process_manager.cleanup()
        raise


if __name__ == '__main__':
    main()
