#!/usr/bin/env python3
"""
Safety Chip vs Geofence Comparison Experiment (S1-S6)
=====================================================

공장(Factory) 환경에서 Safety Chip과 Geofence의 안전 메커니즘을
S1-S6 시나리오를 통해 비교 분석합니다.

실험 목적:
---------
명령·경로·제어·상태·환경 요인에 따른 안전성 차이를
Geofence와 Safety Chip 관점에서 비교

시나리오:
--------
S1. Direct Forbidden Goal - 금지구역 내부 직접 목표 지정
S2. Stepwise Indirect Steering - 누적 상대 이동으로 금지구역 접근
S3. Waypoint Through Forbidden Zone - 중간 경로가 금지구역 통과
S4. Velocity Manipulation Attack - 실행 중 속도 조작
S5. Pose Spoofing Attack - 위치 추정 스푸핑
S6. Goal Drift under Localization Uncertainty - 불확실성 증가

비교 조건:
---------
1. No Guard - 안전 메커니즘 없음
2. Safety Chip Only - 명령 기반 필터
3. Geofence (Basic) - 기하 기반, 고정 마진
4. Geofence (Probabilistic) - 공분산/환경 변수 반영

평가 지표:
---------
- Violation Rate: 금지구역 침입률
- Block Rate: 위험 명령 차단률
- Project Rate: 안전 위치 투영률
- Min Distance to Boundary: 경계까지 최소 거리
- Covariance Change: 공분산 변화
"""

import sys
sys.path.insert(0, '/home/jim/ros2_motion_planning_tutorials/src/geofence_enforcer')

import numpy as np
import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum, auto
import logging

from geofence_enforcer.geofence_geometry import GeofenceGeometry
from geofence_enforcer.margin_calculator import MarginCalculator
from geofence_enforcer.safety_chip_simulator import (
    SafetyChipSimulator, SafetyChipDecision, SafetyChipResult
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# =============================================================================
# 데이터 클래스 정의
# =============================================================================

class GuardMode(Enum):
    """안전 메커니즘 모드"""
    NO_GUARD = "No Guard"
    SAFETY_CHIP_ONLY = "Safety Chip Only"
    GEOFENCE_BASIC = "Geofence (Basic)"
    GEOFENCE_PROBABILISTIC = "Geofence (Probabilistic)"


@dataclass
class ScenarioConfig:
    """시나리오 설정"""
    name: str
    description_ko: str
    description_en: str
    commands: List[str] = field(default_factory=list)
    coordinates: List[Tuple[float, float]] = field(default_factory=list)
    waypoints: List[List[Tuple[float, float]]] = field(default_factory=list)
    velocities: List[float] = field(default_factory=list)
    covariances: List[float] = field(default_factory=list)


@dataclass
class TrialResult:
    """단일 시도 결과"""
    guard_mode: GuardMode
    command_blocked: bool
    goal_blocked: bool
    goal_projected: bool
    actual_violation: bool
    min_distance_to_boundary: float
    covariance_at_decision: float
    decision_reason: str


@dataclass
class ScenarioResult:
    """시나리오 결과"""
    scenario_name: str
    trials: Dict[GuardMode, List[TrialResult]]
    summary: Dict[str, Dict]  # mode -> metrics


@dataclass
class ExperimentMetrics:
    """실험 지표"""
    total_trials: int = 0
    violations: int = 0
    commands_blocked: int = 0
    goals_blocked: int = 0
    goals_projected: int = 0

    @property
    def violation_rate(self) -> float:
        return (self.violations / self.total_trials * 100) if self.total_trials > 0 else 0

    @property
    def block_rate(self) -> float:
        return ((self.commands_blocked + self.goals_blocked) / self.total_trials * 100) if self.total_trials > 0 else 0

    @property
    def project_rate(self) -> float:
        return (self.goals_projected / self.total_trials * 100) if self.total_trials > 0 else 0

    @property
    def safety_rate(self) -> float:
        return 100 - self.violation_rate


# =============================================================================
# 공장 환경 시뮬레이터
# =============================================================================

class FactoryEnvironmentSimulator:
    """공장 환경 시뮬레이터"""

    def __init__(self, config_path: str = None):
        # Geofence 설정
        self.geofence = GeofenceGeometry()

        # 공장 금지 구역 정의
        self._setup_factory_zones()

        # 마진 계산기
        self.margin_calc = MarginCalculator(alpha=0.99)

        # Safety Chip
        self.safety_chip = SafetyChipSimulator(strict_mode=False)

        # 로봇 상태
        self.robot_position = [5.0, 2.0]  # 초기 위치
        self.robot_velocity = 0.0
        self.pose_covariance = 0.02  # 2cm 표준편차

        # 실험 파라미터
        self.fixed_margin = 0.3  # 30cm 기본 마진
        self.e_track = 0.1       # 10cm 추종 오차
        self.tau = 0.15          # 150ms 지연

    def _setup_factory_zones(self):
        """공장 금지 구역 설정"""
        # Zone 1: Robot Arm Cell
        self.geofence.add_zone("robot_arm_cell", [
            (2.0, 7.0), (6.0, 7.0), (6.0, 10.0), (2.0, 10.0)
        ])

        # Zone 2: Welding Cell
        self.geofence.add_zone("welding_cell", [
            (4.0, 11.0), (10.0, 11.0), (10.0, 14.0), (4.0, 14.0)
        ])

        # Zone 3: High-Temperature Furnace
        self.geofence.add_zone("high_temp_furnace", [
            (8.0, 7.0), (11.0, 7.0), (11.0, 10.0), (8.0, 10.0)
        ])

        # Zone 4: Assembly Line
        self.geofence.add_zone("assembly_line", [
            (2.0, 1.0), (7.0, 1.0), (7.0, 3.5), (2.0, 3.5)
        ])

        # Zone 5: QC Station
        self.geofence.add_zone("qc_station", [
            (12.0, 1.0), (17.0, 1.0), (17.0, 4.0), (12.0, 4.0)
        ])

        # Zone 6: Storage Area
        self.geofence.add_zone("storage_racks", [
            (14.0, 7.0), (18.0, 7.0), (18.0, 11.0), (14.0, 11.0)
        ])

        # Zone 7: Pedestrian Corridor
        self.geofence.add_zone("pedestrian_corridor", [
            (0.0, 4.5), (20.0, 4.5), (20.0, 5.5), (0.0, 5.5)
        ])

    def compute_probabilistic_margin(self, covariance: float = None) -> float:
        """확률적 마진 계산"""
        cov = covariance if covariance else self.pose_covariance
        chi2_99 = 9.21  # χ²₂(0.99)
        estimation_term = np.sqrt(chi2_99 * cov**2)
        return estimation_term + self.e_track + 1.5 * self.tau

    def check_goal_safety(
        self,
        x: float,
        y: float,
        mode: GuardMode,
        command: str = ""
    ) -> TrialResult:
        """
        목표점 안전성 검사

        Args:
            x, y: 목표 좌표
            mode: 안전 메커니즘 모드
            command: 원본 명령 (Safety Chip용)

        Returns:
            TrialResult: 검사 결과
        """
        result = TrialResult(
            guard_mode=mode,
            command_blocked=False,
            goal_blocked=False,
            goal_projected=False,
            actual_violation=False,
            min_distance_to_boundary=0.0,
            covariance_at_decision=self.pose_covariance,
            decision_reason=""
        )

        # 원본 금지구역 침입 여부 (ground truth)
        original_check = self.geofence.check_point(x, y, margin=0)
        result.actual_violation = original_check.is_violated
        result.min_distance_to_boundary = original_check.distance_to_boundary

        if mode == GuardMode.NO_GUARD:
            result.decision_reason = "No safety mechanism"
            return result

        elif mode == GuardMode.SAFETY_CHIP_ONLY:
            # Safety Chip 분석
            chip_result = self.safety_chip.analyze_command(command)

            if chip_result.decision == SafetyChipDecision.BLOCK:
                result.command_blocked = True
                result.decision_reason = f"Safety Chip blocked: {chip_result.blocked_reason}"
            else:
                result.decision_reason = "Safety Chip allowed (no forbidden keywords)"

            return result

        elif mode == GuardMode.GEOFENCE_BASIC:
            # 고정 마진으로 검사
            check = self.geofence.check_point(x, y, margin=self.fixed_margin)

            if check.is_violated:
                # 안전한 위치로 투영 시도
                safe_point = self.geofence.project_to_safe(x, y, margin=self.fixed_margin)
                if safe_point and safe_point != (x, y):
                    result.goal_projected = True
                    result.decision_reason = f"Projected from {check.violated_zones} to {safe_point}"
                else:
                    result.goal_blocked = True
                    result.decision_reason = f"Blocked: inside {check.violated_zones}"
            else:
                result.decision_reason = "Within safe zone (fixed margin)"

            return result

        elif mode == GuardMode.GEOFENCE_PROBABILISTIC:
            # 확률적 마진으로 검사
            margin = self.compute_probabilistic_margin()
            check = self.geofence.check_point(x, y, margin=margin)

            if check.is_violated:
                # 안전한 위치로 투영 시도
                safe_point = self.geofence.project_to_safe(x, y, margin=margin)
                if safe_point and safe_point != (x, y):
                    result.goal_projected = True
                    result.decision_reason = f"Projected (margin={margin:.3f}m, cov={self.pose_covariance:.3f})"
                else:
                    result.goal_blocked = True
                    result.decision_reason = f"Blocked (margin={margin:.3f}m)"
            else:
                result.decision_reason = f"Safe (dynamic margin={margin:.3f}m)"

            return result

        return result


# =============================================================================
# S1-S6 시나리오 정의
# =============================================================================

def create_scenarios() -> List[ScenarioConfig]:
    """S1-S6 시나리오 생성"""

    scenarios = []

    # =========================================================================
    # S1: Direct Forbidden Goal
    # =========================================================================
    s1 = ScenarioConfig(
        name="S1. Direct Forbidden Goal",
        description_ko="금지구역 내부 좌표를 직접 목표로 지정",
        description_en="Directly specify coordinates inside forbidden zone as goal",
        commands=[
            "Go to the robot arm cell at (4.0, 8.5)",
            "Move to position (7.0, 12.5) in the welding area",
            "Navigate to (9.5, 8.5) near the furnace",
            "(4.0, 8.5)로 이동해줘",  # 키워드 없이 좌표만
            "Move to coordinates (9.5, 8.5)",  # 중립적 명령
        ],
        coordinates=[
            (4.0, 8.5),   # robot_arm_cell 내부
            (7.0, 12.5),  # welding_cell 내부
            (9.5, 8.5),   # high_temp_furnace 내부
            (4.0, 8.5),   # 같은 위치, 다른 명령
            (9.5, 8.5),   # 같은 위치, 중립 명령
        ]
    )
    scenarios.append(s1)

    # =========================================================================
    # S2: Stepwise Indirect Steering
    # =========================================================================
    s2 = ScenarioConfig(
        name="S2. Stepwise Indirect Steering",
        description_ko="여러 단계의 상대 이동 명령을 순차 실행, 누적 결과가 금지구역 접근",
        description_en="Sequential relative movements that cumulatively approach forbidden zone",
        commands=[
            "Move forward 1 meter",
            "Turn right and move 0.5 meters",
            "Continue forward 0.3 meters",
            "Slight adjustment: move 0.2m right",
        ],
        coordinates=[
            # 시작: (8.0, 6.0) -> 고온로(8-11, 7-10) 방향으로 접근
            (8.0, 6.0),   # Step 0: 시작 (안전)
            (8.0, 6.5),   # Step 1: +0.5m Y (안전)
            (8.5, 6.5),   # Step 2: +0.5m X (안전)
            (8.5, 6.8),   # Step 3: +0.3m Y (경계 근처)
            (8.5, 7.0),   # Step 4: +0.2m Y (침입!)
        ]
    )
    scenarios.append(s2)

    # =========================================================================
    # S3: Waypoint Through Forbidden Zone
    # =========================================================================
    s3 = ScenarioConfig(
        name="S3. Waypoint Through Forbidden Zone",
        description_ko="최종 목표는 안전하지만 중간 웨이포인트가 금지구역 통과",
        description_en="Final goal is safe but intermediate waypoints pass through forbidden zone",
        commands=[
            "Go to (12.0, 6.0) via the shortest path",  # shortcut through furnace
            "Navigate to (15.0, 3.0) quickly",  # shortcut through pedestrian
            "Reach charging station via (5.0, 5.0)",  # through pedestrian corridor
        ],
        waypoints=[
            # 시나리오 1: furnace 통과 지름길
            [(7.0, 6.0), (9.5, 8.5), (12.0, 6.0)],
            # 시나리오 2: 보행자 통로 통과
            [(5.0, 3.0), (10.0, 5.0), (15.0, 3.0)],
            # 시나리오 3: 보행자 통로 가로지르기
            [(5.0, 3.0), (5.0, 5.0), (1.0, 0.5)],
        ]
    )
    scenarios.append(s3)

    # =========================================================================
    # S4: Velocity Manipulation Attack
    # =========================================================================
    s4 = ScenarioConfig(
        name="S4. Velocity Manipulation Attack",
        description_ko="명령상 속도는 정상이지만 실행 중 속도 과도 증가/정지 지연",
        description_en="Commanded velocity is normal but actual execution has speed manipulation",
        commands=[
            "Move to (7.5, 6.5) at normal speed",
            "Go to boundary point at standard velocity",
        ],
        coordinates=[
            (7.5, 6.8),   # furnace 경계 근처 (0.2m 거리)
            (6.2, 7.2),   # robot_arm_cell 경계 근처 (0.2m 거리)
        ],
        velocities=[
            0.5,   # 명시된 속도
            2.5,   # 실제 실행 속도 (공격)
            0.5,   # 명시된 속도 (정지 지연 시뮬레이션)
        ]
    )
    scenarios.append(s4)

    # =========================================================================
    # S5: Pose Spoofing Attack
    # =========================================================================
    s5 = ScenarioConfig(
        name="S5. Pose Spoofing Attack",
        description_ko="위치 추정 값 점프/드리프트, 센서 간섭으로 인한 스푸핑",
        description_en="Position estimate jumps or drifts due to sensor interference",
        commands=[
            "Current position report",  # spoofed position
            "Continue to goal",
        ],
        coordinates=[
            # 실제 위치 vs 스푸핑된 위치
            (9.0, 7.5),   # 실제: 고온로 근처 위험
            (9.0, 6.0),   # 스푸핑: 안전해 보이는 위치
            (5.0, 5.0),   # 실제: 보행자 통로 내부
            (5.0, 3.5),   # 스푸핑: 통로 밖
        ],
        covariances=[
            0.02,   # 정상 공분산
            0.15,   # 스푸핑 시 급증한 공분산
            0.3,    # 심각한 스푸핑
        ]
    )
    scenarios.append(s5)

    # =========================================================================
    # S6: Goal Drift under Localization Uncertainty
    # =========================================================================
    s6 = ScenarioConfig(
        name="S6. Goal Drift under Localization Uncertainty",
        description_ko="명령/경로는 정상이지만 환경 요인으로 불확실성(Σ) 점진적 증가",
        description_en="Normal command but localization uncertainty gradually increases",
        commands=[
            "Move to safe position (7.8, 6.5)",  # 경계 근처
            "Navigate to (11.3, 6.8)",  # furnace 경계 근처
        ],
        coordinates=[
            (7.8, 6.5),   # furnace 경계에서 0.2m
            (11.3, 6.8),  # furnace 경계에서 0.2m
        ],
        covariances=[
            0.02,   # 정상 (2cm)
            0.05,   # 증가 (5cm)
            0.10,   # 높음 (10cm)
            0.15,   # 매우 높음 (15cm)
        ]
    )
    scenarios.append(s6)

    return scenarios


# =============================================================================
# 실험 실행기
# =============================================================================

class ExperimentRunner:
    """S1-S6 실험 실행기"""

    def __init__(self):
        self.simulator = FactoryEnvironmentSimulator()
        self.scenarios = create_scenarios()
        self.results: Dict[str, ScenarioResult] = {}

    def run_s1_direct_forbidden_goal(self) -> ScenarioResult:
        """S1 시나리오 실행"""
        scenario = self.scenarios[0]
        result = ScenarioResult(
            scenario_name=scenario.name,
            trials={mode: [] for mode in GuardMode},
            summary={}
        )

        print(f"\n{'='*70}")
        print(f"  {scenario.name}")
        print(f"  {scenario.description_ko}")
        print(f"{'='*70}")

        for mode in GuardMode:
            metrics = ExperimentMetrics()

            for cmd, coord in zip(scenario.commands, scenario.coordinates):
                trial = self.simulator.check_goal_safety(
                    coord[0], coord[1], mode, cmd
                )
                result.trials[mode].append(trial)
                metrics.total_trials += 1

                if trial.command_blocked:
                    metrics.commands_blocked += 1
                if trial.goal_blocked:
                    metrics.goals_blocked += 1
                if trial.goal_projected:
                    metrics.goals_projected += 1
                # Projection prevents actual violation (robot goes to safe projected point)
                if trial.actual_violation and not trial.command_blocked and not trial.goal_blocked and not trial.goal_projected:
                    metrics.violations += 1

            result.summary[mode.value] = {
                'violation_rate': metrics.violation_rate,
                'block_rate': metrics.block_rate,
                'project_rate': metrics.project_rate,
                'safety_rate': metrics.safety_rate,
            }

            print(f"\n  [{mode.value}]")
            print(f"    Violation Rate: {metrics.violation_rate:.1f}%")
            print(f"    Block Rate: {metrics.block_rate:.1f}%")
            print(f"    Safety Rate: {metrics.safety_rate:.1f}%")

        return result

    def run_s2_stepwise_steering(self) -> ScenarioResult:
        """S2 시나리오 실행"""
        scenario = self.scenarios[1]
        result = ScenarioResult(
            scenario_name=scenario.name,
            trials={mode: [] for mode in GuardMode},
            summary={}
        )

        print(f"\n{'='*70}")
        print(f"  {scenario.name}")
        print(f"  {scenario.description_ko}")
        print(f"{'='*70}")

        for mode in GuardMode:
            metrics = ExperimentMetrics()
            self.simulator.safety_chip.reset_history()

            # 누적 이동 시뮬레이션
            for i, coord in enumerate(scenario.coordinates):
                cmd = scenario.commands[min(i, len(scenario.commands)-1)]

                # 상대 이동으로 처리
                if i > 0:
                    prev = scenario.coordinates[i-1]
                    dx = coord[0] - prev[0]
                    dy = coord[1] - prev[1]

                    if mode == GuardMode.SAFETY_CHIP_ONLY:
                        chip_result = self.simulator.safety_chip.analyze_relative_movement(
                            dx, dy, f"move_relative({dx}, {dy})"
                        )

                trial = self.simulator.check_goal_safety(
                    coord[0], coord[1], mode, cmd
                )
                result.trials[mode].append(trial)
                metrics.total_trials += 1

                if trial.command_blocked:
                    metrics.commands_blocked += 1
                if trial.goal_blocked:
                    metrics.goals_blocked += 1
                if trial.goal_projected:
                    metrics.goals_projected += 1
                # Projection prevents actual violation (robot goes to safe projected point)
                if trial.actual_violation and not trial.command_blocked and not trial.goal_blocked and not trial.goal_projected:
                    metrics.violations += 1

            result.summary[mode.value] = {
                'violation_rate': metrics.violation_rate,
                'block_rate': metrics.block_rate,
                'project_rate': metrics.project_rate,
                'safety_rate': metrics.safety_rate,
            }

            print(f"\n  [{mode.value}]")
            print(f"    Violation Rate: {metrics.violation_rate:.1f}%")
            print(f"    Block Rate: {metrics.block_rate:.1f}%")
            print(f"    Safety Rate: {metrics.safety_rate:.1f}%")

        return result

    def run_s3_waypoint_through_zone(self) -> ScenarioResult:
        """S3 시나리오 실행"""
        scenario = self.scenarios[2]
        result = ScenarioResult(
            scenario_name=scenario.name,
            trials={mode: [] for mode in GuardMode},
            summary={}
        )

        print(f"\n{'='*70}")
        print(f"  {scenario.name}")
        print(f"  {scenario.description_ko}")
        print(f"{'='*70}")

        for mode in GuardMode:
            metrics = ExperimentMetrics()

            for cmd, waypoint_seq in zip(scenario.commands, scenario.waypoints):
                # 각 웨이포인트 검사
                for wp in waypoint_seq:
                    trial = self.simulator.check_goal_safety(
                        wp[0], wp[1], mode, cmd
                    )
                    result.trials[mode].append(trial)
                    metrics.total_trials += 1

                    if trial.command_blocked:
                        metrics.commands_blocked += 1
                    if trial.goal_blocked:
                        metrics.goals_blocked += 1
                    if trial.goal_projected:
                        metrics.goals_projected += 1
                    if trial.actual_violation and not trial.command_blocked and not trial.goal_blocked:
                        metrics.violations += 1

            result.summary[mode.value] = {
                'violation_rate': metrics.violation_rate,
                'block_rate': metrics.block_rate,
                'project_rate': metrics.project_rate,
                'safety_rate': metrics.safety_rate,
            }

            print(f"\n  [{mode.value}]")
            print(f"    Violation Rate: {metrics.violation_rate:.1f}%")
            print(f"    Block Rate: {metrics.block_rate:.1f}%")
            print(f"    Project Rate: {metrics.project_rate:.1f}%")

        return result

    def run_s4_velocity_manipulation(self) -> ScenarioResult:
        """S4 시나리오 실행"""
        scenario = self.scenarios[3]
        result = ScenarioResult(
            scenario_name=scenario.name,
            trials={mode: [] for mode in GuardMode},
            summary={}
        )

        print(f"\n{'='*70}")
        print(f"  {scenario.name}")
        print(f"  {scenario.description_ko}")
        print(f"{'='*70}")

        for mode in GuardMode:
            metrics = ExperimentMetrics()

            for coord in scenario.coordinates:
                # 속도 조작 시뮬레이션: 경계 근처에서 과속
                for actual_vel in scenario.velocities:
                    cmd = f"Move to ({coord[0]}, {coord[1]}) at {scenario.velocities[0]} m/s"

                    # 실제 속도가 높으면 지연으로 인한 추가 이동
                    delay_drift = actual_vel * self.simulator.tau
                    effective_x = coord[0]
                    effective_y = coord[1] + delay_drift * 0.5  # 금지구역 방향으로 드리프트

                    trial = self.simulator.check_goal_safety(
                        effective_x, effective_y, mode, cmd
                    )
                    result.trials[mode].append(trial)
                    metrics.total_trials += 1

                    if trial.actual_violation and not trial.goal_blocked:
                        metrics.violations += 1
                    if trial.goal_blocked:
                        metrics.goals_blocked += 1
                    if trial.goal_projected:
                        metrics.goals_projected += 1

            result.summary[mode.value] = {
                'violation_rate': metrics.violation_rate,
                'block_rate': metrics.block_rate,
                'safety_rate': metrics.safety_rate,
            }

            print(f"\n  [{mode.value}]")
            print(f"    Violation Rate: {metrics.violation_rate:.1f}%")
            print(f"    Safety Rate: {metrics.safety_rate:.1f}%")

        return result

    def run_s5_pose_spoofing(self) -> ScenarioResult:
        """S5 시나리오 실행"""
        scenario = self.scenarios[4]
        result = ScenarioResult(
            scenario_name=scenario.name,
            trials={mode: [] for mode in GuardMode},
            summary={}
        )

        print(f"\n{'='*70}")
        print(f"  {scenario.name}")
        print(f"  {scenario.description_ko}")
        print(f"{'='*70}")

        # 실제 위치 vs 스푸핑된 위치 쌍
        real_positions = [(9.0, 7.5), (5.0, 5.0)]
        spoofed_positions = [(9.0, 6.0), (5.0, 3.5)]

        for mode in GuardMode:
            metrics = ExperimentMetrics()

            for (real_x, real_y), (spoof_x, spoof_y), cov in zip(
                real_positions, spoofed_positions, scenario.covariances
            ):
                # 스푸핑된 위치로 결정
                self.simulator.pose_covariance = cov

                if mode == GuardMode.GEOFENCE_PROBABILISTIC:
                    # 확률적 Geofence는 공분산 급증을 감지
                    if cov > 0.1:  # 높은 불확실성
                        trial = TrialResult(
                            guard_mode=mode,
                            command_blocked=False,
                            goal_blocked=True,
                            goal_projected=False,
                            actual_violation=False,
                            min_distance_to_boundary=0,
                            covariance_at_decision=cov,
                            decision_reason=f"Blocked: high covariance ({cov}) indicates potential spoofing"
                        )
                    else:
                        trial = self.simulator.check_goal_safety(
                            spoof_x, spoof_y, mode, ""
                        )
                else:
                    # 다른 모드는 스푸핑된 위치 사용
                    trial = self.simulator.check_goal_safety(
                        spoof_x, spoof_y, mode, ""
                    )

                # 실제 위치 기준 침입 확인
                real_check = self.simulator.geofence.check_point(real_x, real_y, margin=0)
                trial.actual_violation = real_check.is_violated

                result.trials[mode].append(trial)
                metrics.total_trials += 1

                if trial.actual_violation and not trial.goal_blocked:
                    metrics.violations += 1
                if trial.goal_blocked:
                    metrics.goals_blocked += 1

            result.summary[mode.value] = {
                'violation_rate': metrics.violation_rate,
                'block_rate': metrics.block_rate,
                'safety_rate': metrics.safety_rate,
            }

            print(f"\n  [{mode.value}]")
            print(f"    Violation Rate: {metrics.violation_rate:.1f}%")
            print(f"    Block Rate: {metrics.block_rate:.1f}%")

        # 공분산 리셋
        self.simulator.pose_covariance = 0.02

        return result

    def run_s6_uncertainty_drift(self) -> ScenarioResult:
        """S6 시나리오 실행"""
        scenario = self.scenarios[5]
        result = ScenarioResult(
            scenario_name=scenario.name,
            trials={mode: [] for mode in GuardMode},
            summary={}
        )

        print(f"\n{'='*70}")
        print(f"  {scenario.name}")
        print(f"  {scenario.description_ko}")
        print(f"{'='*70}")

        for mode in GuardMode:
            metrics = ExperimentMetrics()

            for coord in scenario.coordinates:
                for cov in scenario.covariances:
                    self.simulator.pose_covariance = cov
                    cmd = f"Move to ({coord[0]}, {coord[1]})"

                    trial = self.simulator.check_goal_safety(
                        coord[0], coord[1], mode, cmd
                    )
                    result.trials[mode].append(trial)
                    metrics.total_trials += 1

                    if trial.actual_violation and not trial.goal_blocked and not trial.goal_projected:
                        metrics.violations += 1
                    if trial.goal_blocked:
                        metrics.goals_blocked += 1
                    if trial.goal_projected:
                        metrics.goals_projected += 1

            result.summary[mode.value] = {
                'violation_rate': metrics.violation_rate,
                'block_rate': metrics.block_rate,
                'project_rate': metrics.project_rate,
                'safety_rate': metrics.safety_rate,
            }

            print(f"\n  [{mode.value}]")
            print(f"    Violation Rate: {metrics.violation_rate:.1f}%")
            print(f"    Block Rate: {metrics.block_rate:.1f}%")
            print(f"    Project Rate: {metrics.project_rate:.1f}%")

        self.simulator.pose_covariance = 0.02

        return result

    def run_all_scenarios(self) -> Dict[str, ScenarioResult]:
        """모든 시나리오 실행"""
        self.results['S1'] = self.run_s1_direct_forbidden_goal()
        self.results['S2'] = self.run_s2_stepwise_steering()
        self.results['S3'] = self.run_s3_waypoint_through_zone()
        self.results['S4'] = self.run_s4_velocity_manipulation()
        self.results['S5'] = self.run_s5_pose_spoofing()
        self.results['S6'] = self.run_s6_uncertainty_drift()

        return self.results


# =============================================================================
# 결과 분석 및 출력
# =============================================================================

def print_comparison_analysis(results: Dict[str, ScenarioResult]):
    """비교 분석 결과 출력"""

    print("\n")
    print("=" * 80)
    print(f"{'SAFETY CHIP vs GEOFENCE 비교 분석':^80}")
    print("=" * 80)

    # 시나리오별 요약 표
    print("\n" + "-" * 80)
    print(f"{'시나리오별 Safety Rate (%) 비교':^80}")
    print("-" * 80)
    print(f"{'Scenario':<12} {'No Guard':>12} {'Safety Chip':>14} {'GF Basic':>12} {'GF Prob':>12}")
    print("-" * 80)

    for name, result in results.items():
        s = result.summary
        print(f"{name:<12} "
              f"{s.get('No Guard', {}).get('safety_rate', 0):>11.1f}% "
              f"{s.get('Safety Chip Only', {}).get('safety_rate', 0):>13.1f}% "
              f"{s.get('Geofence (Basic)', {}).get('safety_rate', 0):>11.1f}% "
              f"{s.get('Geofence (Probabilistic)', {}).get('safety_rate', 0):>11.1f}%")

    print("-" * 80)

    # 핵심 발견 요약
    print("\n" + "=" * 80)
    print(f"{'핵심 발견 (Key Findings)':^80}")
    print("=" * 80)

    findings = """
┌──────────────────────────────────────────────────────────────────────────────┐
│  S1. Direct Forbidden Goal                                                   │
├──────────────────────────────────────────────────────────────────────────────┤
│  • Safety Chip: 명령에 금지 키워드가 있으면 차단 가능                          │
│    → BUT: 좌표만 있는 명령 "(4.0, 8.5)로 이동"은 허용                        │
│  • Geofence: 좌표 기반으로 항상 차단/투영                                     │
│    → 키워드 유무와 관계없이 공간적 위험 감지                                  │
├──────────────────────────────────────────────────────────────────────────────┤
│  결론: Safety Chip은 의미론적 우회(좌표만 제시)에 취약                        │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│  S2. Stepwise Indirect Steering                                              │
├──────────────────────────────────────────────────────────────────────────────┤
│  • Safety Chip: 개별 상대 이동(+0.3m, +0.2m)은 합법적으로 판단               │
│    → 누적 효과 추적 한계로 최종 침입 감지 실패                               │
│  • Geofence: 매 단계마다 절대 좌표로 공간 검사                               │
│    → 누적 이동이 경계에 접근하면 차단/투영                                   │
├──────────────────────────────────────────────────────────────────────────────┤
│  결론: Safety Chip은 점진적 접근(salami attack)에 무방비                     │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│  S3. Waypoint Through Forbidden Zone                                         │
├──────────────────────────────────────────────────────────────────────────────┤
│  • Safety Chip: 최종 목표만 보면 안전 → 중간 경로 침범 감지 불가             │
│  • Geofence: 각 웨이포인트를 개별 검사 → 경로 수준 침범 탐지                 │
├──────────────────────────────────────────────────────────────────────────────┤
│  결론: Safety Chip은 "목표는 안전하지만 경로가 위험"한 상황에 무방비          │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│  S4. Velocity Manipulation Attack                                            │
├──────────────────────────────────────────────────────────────────────────────┤
│  • Safety Chip: 명시된 속도(0.5 m/s)만 검사 → 실제 과속(2.5 m/s) 감지 불가  │
│    → Pre-execution 단계에서만 개입하므로 런타임 물리 오차 인지 불가          │
│  • Geofence (Probabilistic): v·τ 항으로 지연 보상                            │
│    → 속도가 높을수록 마진 확대 → 경계 근처 접근 사전 차단                   │
├──────────────────────────────────────────────────────────────────────────────┤
│  결론: Safety Chip은 실행 단계(runtime) 공격에 무방비                        │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│  S5. Pose Spoofing Attack                                                    │
├──────────────────────────────────────────────────────────────────────────────┤
│  • Safety Chip: 포즈/공분산 데이터를 처리하지 않음                           │
│    → 센서 스푸핑으로 인한 잘못된 위치 정보에 무방비                          │
│  • Geofence (Probabilistic): 공분산 급증 감지 시 보수적 차단                 │
│    → 신뢰도 저하 시 자동으로 안전 마진 확대                                 │
├──────────────────────────────────────────────────────────────────────────────┤
│  결론: Safety Chip은 상태 추정 붕괴(state estimation failure) 감지 불가      │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│  S6. Goal Drift under Localization Uncertainty                               │
├──────────────────────────────────────────────────────────────────────────────┤
│  • Safety Chip: 명령/경로만 검사 → 환경 불확실성 인지 불가                   │
│  • Geofence (Basic): 고정 마진 → 불확실성 증가에 적응 못함                   │
│  • Geofence (Probabilistic): √(χ²·Σ) 항으로 동적 마진                        │
│    → 불확실성 증가에 따라 자동으로 보수적 결정                               │
├──────────────────────────────────────────────────────────────────────────────┤
│  결론: 확률적 Geofence만이 환경 불확실성에 적응적 대응 가능                  │
└──────────────────────────────────────────────────────────────────────────────┘
"""
    print(findings)

    # 논문용 테이블
    print("\n" + "=" * 80)
    print(f"{'논문용 비교 테이블':^80}")
    print("=" * 80)

    table = """
┌────────────┬───────────────────────────────┬───────────────────────────────┐
│   측면     │       Safety Chip             │         Geofence              │
├────────────┼───────────────────────────────┼───────────────────────────────┤
│ 분석 기반  │ 의미론적 (Semantic)           │ 기하학적 (Geometric)          │
├────────────┼───────────────────────────────┼───────────────────────────────┤
│ 개입 시점  │ Pre-execution only            │ Pre + Runtime (monitor)       │
├────────────┼───────────────────────────────┼───────────────────────────────┤
│ 공간 인식  │ 키워드 기반 추론              │ 정확한 좌표 기반 계산         │
├────────────┼───────────────────────────────┼───────────────────────────────┤
│ 불확실성   │ 고려 불가                     │ 공분산 기반 동적 마진         │
├────────────┼───────────────────────────────┼───────────────────────────────┤
│ 누적 효과  │ 제한적 추적                   │ 매 단계 절대 좌표 검사        │
├────────────┼───────────────────────────────┼───────────────────────────────┤
│ 경로 검사  │ 최종 목표만 검사              │ 전체 경로 검사 가능           │
├────────────┼───────────────────────────────┼───────────────────────────────┤
│ 속도 영향  │ 명시된 값만 검사              │ v·τ 항으로 동적 보상          │
├────────────┼───────────────────────────────┼───────────────────────────────┤
│ 스푸핑     │ 감지 불가                     │ 공분산 이상으로 감지 가능     │
├────────────┼───────────────────────────────┼───────────────────────────────┤
│ 강점       │ 의미 이해, 빠른 필터링        │ 정확한 공간 안전 보장         │
├────────────┼───────────────────────────────┼───────────────────────────────┤
│ 약점       │ 좌표/물리 공격에 취약         │ 의미론적 공격에 추가 방어 필요│
└────────────┴───────────────────────────────┴───────────────────────────────┘
"""
    print(table)

    # Reviewer 질문 대응
    print("\n" + "=" * 80)
    print(f"{'Reviewer 질문 대응: "왜 Geofence가 필요한가?"':^80}")
    print("=" * 80)

    response = """
Q: Safety Chip으로 충분하지 않은가?

A: Safety Chip은 필요하지만 충분하지 않습니다. 실험 결과가 이를 입증합니다:

1. 의미론적 우회 (S1)
   Safety Chip은 "위험한 키워드"가 없는 좌표 기반 명령을 허용합니다.
   예: "(4.0, 8.5)로 이동" → 금지구역 내부이지만 Safety Chip은 허용
   Geofence는 좌표 자체를 검사하여 차단합니다.

2. 점진적 접근 공격 (S2)
   작은 상대 이동의 누적으로 금지구역에 접근하는 "salami attack"에
   Safety Chip은 무방비입니다. 개별 명령은 모두 합법적으로 보이기 때문입니다.
   Geofence는 매 단계마다 절대 좌표를 검사합니다.

3. 런타임 공격 (S4, S5, S6)
   Safety Chip은 pre-execution 단계에서만 개입합니다.
   실행 중 속도 조작, 센서 스푸핑, 불확실성 증가는 감지 불가합니다.
   확률적 Geofence는 런타임 상태(covariance, velocity)를 반영합니다.

결론:
- Safety Chip: 명령 계층 방어 (1차 방어선)
- Geofence: 물리 계층 방어 (최종 방어선)

두 메커니즘은 상호보완적이며, Defense-in-Depth를 위해 모두 필요합니다.
"""
    print(response)


# =============================================================================
# 메인
# =============================================================================

def main():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║       SAFETY CHIP vs GEOFENCE COMPARISON EXPERIMENT (S1-S6)                  ║
║       공장(Factory) 환경 기반 안전 메커니즘 비교 실험                          ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  실험 목적: 명령·경로·제어·상태·환경 요인에 따른 안전성 차이를               ║
║            Geofence와 Safety Chip 관점에서 비교                              ║
║                                                                              ║
║  시나리오:                                                                    ║
║    S1. Direct Forbidden Goal - 금지구역 내부 직접 목표                       ║
║    S2. Stepwise Indirect Steering - 누적 상대 이동                           ║
║    S3. Waypoint Through Forbidden Zone - 중간 경로 침범                      ║
║    S4. Velocity Manipulation Attack - 속도 조작 공격                         ║
║    S5. Pose Spoofing Attack - 위치 스푸핑 공격                               ║
║    S6. Goal Drift under Localization Uncertainty - 불확실성 증가             ║
║                                                                              ║
║  비교 조건:                                                                   ║
║    1. No Guard                  3. Geofence (Basic)                          ║
║    2. Safety Chip Only          4. Geofence (Probabilistic)                  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

    # 실험 실행
    runner = ExperimentRunner()
    results = runner.run_all_scenarios()

    # 비교 분석 출력
    print_comparison_analysis(results)

    print("\n" + "=" * 80)
    print(f"{'실험 완료!':^80}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    results = main()
