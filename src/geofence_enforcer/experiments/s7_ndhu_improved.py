#!/usr/bin/env python3
"""
S7: Network-Delayed Hazard Update (NDHU) Attack - Improved Version
===================================================================

리뷰어 피드백 반영 개선사항:
1. τ_total 측정 기반 (τ_net + τ_rx + τ_costmap + τ_plan + τ_ctrl)
2. Central Safety Manager 모델 명시
3. a_eff (유효 감속) 실측
4. 시작 위치 수정 (storage_racks와 분리)
5. Cascade 공격: 동적 구역 → storage_racks 침입 유도
6. Jitter + Packet Loss 추가
7. Seeds 30 (전이 구간 50)
8. 재현성을 위한 지연 구현 상세 문서화

Attack Model: Central Safety Manager
------------------------------------
- 작업자 감지/관제 서버가 temporary_worker_zone 생성
- 로봇에 /forbidden_zones 토픽으로 배포
- Ground truth는 환경 기준 즉시, 로봇 반영은 지연

Cascade Attack:
- 동적 구역은 "회피를 강제하는 트리거"
- 회피 경로가 storage_racks 쪽으로 유도됨
- 실제 피해: storage_racks (정적 high-risk zone) 침입

τ_total Breakdown:
- τ_net: 네트워크 전송 지연
- τ_rx: ROS2 subscription 처리 지연
- τ_costmap: costmap 반영 지연
- τ_plan: 플래너 재계획 주기
- τ_ctrl: 제어 명령 발행 지연

Effective Deceleration:
- a_eff = max(-Δv/Δt) from velocity logs
- 실제 정지 거리: v²/(2×a_eff)
- Nav2 가속 제한, jerk, 컨트롤러 특성 반영

Usage:
    python3 s7_ndhu_improved.py --sweep --output s7_ndhu_improved_results.json
    python3 s7_ndhu_improved.py --v 1.5 --L 4 --tau_net_ms 300 --jitter_ms 50 --loss_pct 1
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import math
import time
import random
import threading
import queue
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Callable
from enum import Enum
from datetime import datetime
from collections import deque

from shapely.geometry import Point as ShapelyPoint, Polygon, box, LineString


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class WorldConfig:
    """Factory map configuration - 리뷰어 피드백 #4 반영"""
    width: float = 20.0
    height: float = 15.0

    # Static forbidden zone: storage_racks (high-risk zone)
    # 로봇 시작 위치와 y 범위가 겹치지 않도록 설계
    static_zone_xmin: float = 14.0
    static_zone_xmax: float = 18.0
    static_zone_ymin: float = 9.0   # 수정: 7→9 (로봇 경로와 분리)
    static_zone_ymax: float = 13.0  # 수정: 11→13

    def get_static_zone(self) -> Polygon:
        return box(self.static_zone_xmin, self.static_zone_ymin,
                   self.static_zone_xmax, self.static_zone_ymax)


@dataclass
class RobotConfig:
    """Robot physical parameters"""
    max_velocity: float = 2.0
    max_acceleration: float = 2.0
    nominal_deceleration: float = 3.0  # 명목상 최대 감속
    control_frequency: float = 20.0
    pose_noise_std: float = 0.1
    velocity_noise_std: float = 0.05

    # Nav2 컨트롤러 특성 (실제 감속에 영향)
    jerk_limit: float = 5.0  # m/s³
    controller_delay: float = 0.05  # 50ms 제어 지연

    @property
    def dt(self) -> float:
        return 1.0 / self.control_frequency


@dataclass
class LatencyBreakdown:
    """τ_total 구성요소 - 리뷰어 피드백 #1 반영"""
    tau_net: float = 0.0       # 네트워크 전송 지연
    tau_rx: float = 0.02       # ROS2 subscription 처리 (20ms typical)
    tau_costmap: float = 0.05  # costmap 반영 (50ms @ 20Hz)
    tau_plan: float = 0.1      # 플래너 재계획 (100ms typical)
    tau_ctrl: float = 0.05     # 제어 명령 발행 (50ms)

    @property
    def tau_total(self) -> float:
        return self.tau_net + self.tau_rx + self.tau_costmap + self.tau_plan + self.tau_ctrl

    def to_dict(self) -> dict:
        return {
            "tau_net": self.tau_net,
            "tau_rx": self.tau_rx,
            "tau_costmap": self.tau_costmap,
            "tau_plan": self.tau_plan,
            "tau_ctrl": self.tau_ctrl,
            "tau_total": self.tau_total,
        }


@dataclass
class NetworkConfig:
    """네트워크 조건 - 리뷰어 피드백 #6 반영"""
    latency_ms: int = 0
    jitter_ms: int = 0         # ±jitter (50% of latency recommended)
    loss_percent: float = 0.0  # 패킷 손실률 (1%, 3%, 5%)
    burst_loss_prob: float = 0.0  # 연속 손실 확률
    burst_loss_duration: float = 0.0  # 연속 손실 지속 시간

    def get_actual_delay(self) -> float:
        """지터 포함 실제 지연 계산"""
        base = self.latency_ms / 1000.0
        if self.jitter_ms > 0:
            jitter = random.gauss(0, self.jitter_ms / 1000.0 / 2)  # ±2σ ≈ jitter
            return max(0, base + jitter)
        return base

    def is_packet_lost(self) -> bool:
        """패킷 손실 여부"""
        return random.random() < self.loss_percent / 100.0


@dataclass
class ExperimentConfig:
    """Experiment parameters - 리뷰어 피드백 #7 반영"""
    # Sweep parameters
    velocities: List[float] = field(default_factory=lambda: [1.0, 1.5, 2.0])
    distances: List[float] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 8, 10])
    latencies_ms: List[int] = field(default_factory=lambda: [0, 100, 300, 500, 700])
    jitter_ratios: List[float] = field(default_factory=lambda: [0.0, 0.5])  # jitter = ratio * latency
    loss_percents: List[float] = field(default_factory=lambda: [0.0, 1.0, 3.0])

    # Seeds: 기본 30, 전이 구간(L=2~4)은 50
    base_seeds: int = 30
    transition_seeds: int = 50
    transition_L_range: Tuple[float, float] = (2.0, 5.0)

    # Dynamic zone parameters (temporary_worker_zone)
    zone_radius: float = 1.5
    trigger_delay: float = 3.0

    # Safety margin
    safety_margin: float = 0.3

    def get_num_seeds(self, L: float) -> int:
        """전이 구간에서는 더 많은 시드 사용"""
        if self.transition_L_range[0] <= L <= self.transition_L_range[1]:
            return self.transition_seeds
        return self.base_seeds


# =============================================================================
# Guard Modes
# =============================================================================

class GuardMode(Enum):
    NO_GUARD = "no_guard"
    SAFETYCHIP = "safetychip"
    SELP = "selp"
    GEOFENCE_ADAPTIVE = "geofence_adaptive"


@dataclass
class GuardState:
    stop_issued: bool = False
    stop_time: Optional[float] = None
    stop_reason: str = ""
    min_distance_observed: float = float('inf')
    projection_distance: float = 0.0


class GuardBase:
    def __init__(self, robot_config: RobotConfig, safety_margin: float = 0.3):
        self.robot_config = robot_config
        self.safety_margin = safety_margin
        self.state = GuardState()

    def reset(self):
        self.state = GuardState()

    def check_safety(self, pose, heading, velocity, zones) -> Tuple[bool, str]:
        raise NotImplementedError

    def update_latency_estimate(self, measured_latency: float):
        pass


class NoGuard(GuardBase):
    """No Guard: 방어 메커니즘 없음"""
    def check_safety(self, pose, heading, velocity, zones) -> Tuple[bool, str]:
        robot_point = ShapelyPoint(pose)
        for zone in zones:
            dist = robot_point.distance(zone)
            self.state.min_distance_observed = min(self.state.min_distance_observed, dist)
        return True, ""


class SafetyChipGuard(GuardBase):
    """SafetyChip: 고정 마진 (latency 미고려)"""
    def __init__(self, robot_config: RobotConfig, safety_margin: float = 0.5):
        super().__init__(robot_config, safety_margin)
        self.fixed_margin = safety_margin

    def check_safety(self, pose, heading, velocity, zones) -> Tuple[bool, str]:
        robot_point = ShapelyPoint(pose)
        for zone in zones:
            dist = robot_point.distance(zone)
            self.state.min_distance_observed = min(self.state.min_distance_observed, dist)
            if dist < self.fixed_margin:
                if not self.state.stop_issued:
                    self.state.stop_issued = True
                    self.state.stop_time = time.time()
                    self.state.stop_reason = f"safetychip (d={dist:.3f}m)"
                return False, self.state.stop_reason
        return True, ""


class SELPGuard(GuardBase):
    """SELP: 속도 비례 마진 (고정 latency 가정)"""
    def __init__(self, robot_config: RobotConfig, safety_margin: float = 0.3,
                 assumed_latency: float = 0.1):
        super().__init__(robot_config, safety_margin)
        self.assumed_latency = assumed_latency

    def compute_margin(self, velocity: float) -> float:
        return self.safety_margin + velocity * self.assumed_latency

    def check_safety(self, pose, heading, velocity, zones) -> Tuple[bool, str]:
        robot_point = ShapelyPoint(pose)
        margin = self.compute_margin(velocity)
        for zone in zones:
            dist = robot_point.distance(zone)
            self.state.min_distance_observed = min(self.state.min_distance_observed, dist)
            if dist < margin:
                if not self.state.stop_issued:
                    self.state.stop_issued = True
                    self.state.stop_time = time.time()
                    self.state.stop_reason = f"selp (d={dist:.3f}m, margin={margin:.3f}m)"
                return False, self.state.stop_reason
        return True, ""


class GeofenceAdaptiveGuard(GuardBase):
    """
    Geofence Adaptive: 실측 기반 τ_total + a_eff 사용

    개선사항 (리뷰어 피드백 #1, #3):
    - τ를 가정하지 않고 실측
    - a_eff를 velocity log에서 추정
    """
    def __init__(self, robot_config: RobotConfig, safety_margin: float = 0.3,
                 initial_latency: float = 0.1):
        super().__init__(robot_config, safety_margin)
        self.latency_history = deque(maxlen=20)
        self.decel_history = deque(maxlen=20)
        self.estimated_latency = initial_latency
        self.estimated_a_eff = robot_config.nominal_deceleration * 0.8  # 초기값: 명목의 80%

    def update_latency_estimate(self, measured_latency: float):
        """실측 latency로 업데이트"""
        self.latency_history.append(measured_latency)
        if len(self.latency_history) >= 3:
            # 90th percentile (보수적)
            sorted_hist = sorted(self.latency_history)
            idx = int(len(sorted_hist) * 0.9)
            self.estimated_latency = sorted_hist[idx]

    def update_decel_estimate(self, measured_decel: float):
        """실측 감속도로 a_eff 업데이트"""
        if measured_decel > 0:
            self.decel_history.append(measured_decel)
            if len(self.decel_history) >= 3:
                # 10th percentile (보수적 - 가장 약한 감속 기준)
                sorted_hist = sorted(self.decel_history)
                idx = int(len(sorted_hist) * 0.1)
                self.estimated_a_eff = max(0.5, sorted_hist[idx])

    def compute_required_distance(self, velocity: float) -> float:
        """실측 기반 필요 거리 계산"""
        d_react = velocity * self.estimated_latency
        d_brake = velocity ** 2 / (2 * self.estimated_a_eff)
        return d_react + d_brake + self.safety_margin

    def check_safety(self, pose, heading, velocity, zones) -> Tuple[bool, str]:
        robot_point = ShapelyPoint(pose)
        required_dist = self.compute_required_distance(velocity)
        self.state.projection_distance = required_dist

        proj_x = pose[0] + required_dist * math.cos(heading)
        proj_y = pose[1] + required_dist * math.sin(heading)
        projection_line = LineString([pose, (proj_x, proj_y)])

        for zone in zones:
            dist = robot_point.distance(zone)
            self.state.min_distance_observed = min(self.state.min_distance_observed, dist)

            if projection_line.intersects(zone) or dist < required_dist:
                if not self.state.stop_issued:
                    self.state.stop_issued = True
                    self.state.stop_time = time.time()
                    self.state.stop_reason = (f"geofence_adaptive (d={dist:.3f}m, "
                                             f"req={required_dist:.3f}m, "
                                             f"τ={self.estimated_latency*1000:.0f}ms, "
                                             f"a_eff={self.estimated_a_eff:.2f})")
                return False, self.state.stop_reason
        return True, ""


def create_guard(mode: GuardMode, robot_config: RobotConfig,
                 safety_margin: float, estimated_latency: float = 0.1) -> GuardBase:
    if mode == GuardMode.NO_GUARD:
        return NoGuard(robot_config, safety_margin)
    elif mode == GuardMode.SAFETYCHIP:
        return SafetyChipGuard(robot_config, safety_margin=0.5)
    elif mode == GuardMode.SELP:
        return SELPGuard(robot_config, safety_margin, assumed_latency=estimated_latency)
    elif mode == GuardMode.GEOFENCE_ADAPTIVE:
        return GeofenceAdaptiveGuard(robot_config, safety_margin, initial_latency=estimated_latency)
    else:
        raise ValueError(f"Unknown guard mode: {mode}")


# =============================================================================
# Central Safety Manager (리뷰어 피드백 #2)
# =============================================================================

@dataclass
class HazardEvent:
    """
    Central Safety Manager에서 생성하는 Hazard Event

    시나리오: 작업자 감지 → 관제 서버 → temporary_worker_zone 생성 → 로봇 배포
    """
    event_id: str
    event_type: str  # "worker_detected", "equipment_malfunction", "emergency_stop"
    zone_center: Tuple[float, float]
    zone_radius: float

    # 타임스탬프 (τ_total 측정용)
    t_detected: float       # 환경에서 실제 감지된 시간 (ground truth)
    t_published: float      # 서버에서 퍼블리시한 시간
    t_received: Optional[float] = None   # 로봇이 수신한 시간
    t_costmap_updated: Optional[float] = None  # costmap에 반영된 시간
    t_stop_issued: Optional[float] = None  # 정지 명령 발행 시간

    def get_polygon(self) -> Polygon:
        return ShapelyPoint(self.zone_center).buffer(self.zone_radius)

    def get_latency_breakdown(self) -> Optional[LatencyBreakdown]:
        """실측 τ_total 계산"""
        if self.t_received is None or self.t_stop_issued is None:
            return None

        breakdown = LatencyBreakdown()
        breakdown.tau_net = self.t_received - self.t_published

        if self.t_costmap_updated:
            breakdown.tau_rx = self.t_costmap_updated - self.t_received

        if self.t_stop_issued:
            # τ_total,meas = t_stop_issued - t_published
            total_measured = self.t_stop_issued - self.t_published
            # 나머지를 plan+ctrl로 배분
            breakdown.tau_plan = max(0, total_measured - breakdown.tau_net - breakdown.tau_rx - breakdown.tau_ctrl)

        return breakdown


class CentralSafetyManager:
    """
    Central Safety Manager 모델

    "작업자 감지/관제 서버가 구역을 생성하고 로봇에 배포한다"
    즉, zone ground truth는 환경/작업자 기준 즉시, 로봇 반영은 지연 가능.
    """

    def __init__(self, network_config: NetworkConfig):
        self.network_config = network_config
        self.pending_events: List[Tuple[float, HazardEvent]] = []  # (delivery_time, event)
        self.active_events: List[HazardEvent] = []
        self.lost_events: List[HazardEvent] = []
        self._event_counter = 0

        # Burst loss state
        self._burst_loss_until = 0.0

    def detect_hazard(self, robot_pose: Tuple[float, float], robot_heading: float,
                      distance_L: float, current_time: float,
                      event_type: str = "worker_detected") -> HazardEvent:
        """
        Hazard 감지 및 이벤트 생성

        시나리오: 로봇 전방 L 미터에서 작업자 감지됨
        """
        self._event_counter += 1

        # 구역 중심: 로봇 전방 L 미터
        center_x = robot_pose[0] + distance_L * math.cos(robot_heading)
        center_y = robot_pose[1] + distance_L * math.sin(robot_heading)

        event = HazardEvent(
            event_id=f"hazard_{self._event_counter}",
            event_type=event_type,
            zone_center=(center_x, center_y),
            zone_radius=1.5,
            t_detected=current_time,
            t_published=current_time,  # 서버 처리 지연 무시 (또는 추가 가능)
        )

        # 패킷 손실 체크
        if self._is_lost(current_time):
            self.lost_events.append(event)
            return event

        # 지연 계산
        actual_delay = self.network_config.get_actual_delay()
        delivery_time = current_time + actual_delay

        self.pending_events.append((delivery_time, event))
        return event

    def _is_lost(self, current_time: float) -> bool:
        """패킷 손실 여부 결정 (burst loss 포함)"""
        # Burst loss 구간
        if current_time < self._burst_loss_until:
            return True

        # 일반 패킷 손실
        if self.network_config.is_packet_lost():
            # Burst loss 시작?
            if random.random() < self.network_config.burst_loss_prob:
                self._burst_loss_until = current_time + self.network_config.burst_loss_duration
            return True

        return False

    def update(self, current_time: float) -> List[HazardEvent]:
        """수신된 이벤트 처리"""
        received = []
        still_pending = []

        for delivery_time, event in self.pending_events:
            if current_time >= delivery_time:
                event.t_received = current_time
                event.t_costmap_updated = current_time + 0.05  # costmap 반영 지연
                self.active_events.append(event)
                received.append(event)
            else:
                still_pending.append((delivery_time, event))

        self.pending_events = still_pending
        return received

    def get_known_zones(self) -> List[Polygon]:
        """로봇이 알고 있는 구역들"""
        return [e.get_polygon() for e in self.active_events]

    def get_ground_truth_zones(self) -> List[Polygon]:
        """실제 활성 구역 (pending 포함)"""
        all_events = self.active_events + [e for _, e in self.pending_events]
        return [e.get_polygon() for e in all_events]

    def reset(self):
        self.pending_events = []
        self.active_events = []
        self.lost_events = []
        self._burst_loss_until = 0.0


# =============================================================================
# Effective Deceleration Measurement (리뷰어 피드백 #3)
# =============================================================================

class VelocityLogger:
    """속도 로그로 유효 감속도 측정"""

    def __init__(self, dt: float):
        self.dt = dt
        self.velocity_log: List[Tuple[float, float]] = []  # (time, velocity)

    def record(self, t: float, velocity: float):
        self.velocity_log.append((t, velocity))

    def compute_effective_deceleration(self) -> float:
        """
        a_eff = max(-Δv/Δt) during braking

        정지 구간의 속도 변화를 분석하여 유효 감속도 계산
        """
        if len(self.velocity_log) < 3:
            return 2.0  # 기본값

        max_decel = 0.0

        for i in range(1, len(self.velocity_log)):
            t_prev, v_prev = self.velocity_log[i-1]
            t_curr, v_curr = self.velocity_log[i]

            dt = t_curr - t_prev
            if dt > 0 and v_prev > v_curr:  # 감속 중
                decel = (v_prev - v_curr) / dt
                max_decel = max(max_decel, decel)

        return max_decel if max_decel > 0 else 2.0

    def reset(self):
        self.velocity_log = []


# =============================================================================
# Cascade Attack Scenario (리뷰어 피드백 #5)
# =============================================================================

@dataclass
class CascadeScenario:
    """
    Cascade 공격 시나리오

    동적 구역은 "회피를 강제하는 트리거"
    회피 시 storage_racks 방향으로 유도
    실제 피해: storage_racks 침입
    """
    name: str
    description: str

    # 로봇 설정 (storage_racks와 분리된 경로)
    start: Tuple[float, float]
    goal: Tuple[float, float]
    initial_heading: float
    robot_speed: float

    # 동적 구역 (회피 트리거)
    trigger_distance: float  # 로봇 전방 거리
    trigger_time: float      # 출발 후 시간

    # 기대 결과
    expected_evasion_direction: str  # "up" or "down"
    storage_racks_risk: bool  # True if evasion leads toward racks


def create_cascade_scenarios() -> List[CascadeScenario]:
    """Cascade 공격 시나리오 생성"""
    scenarios = []

    # 시나리오 1: 기본 cascade
    # 로봇이 x 방향으로 이동 중, 전방에 작업자 구역 출현
    # 위로 회피하면 storage_racks (y=9~13)에 접근
    scenarios.append(CascadeScenario(
        name="S7-C1: Worker Avoidance → Racks Risk",
        description="작업자 회피 시 storage_racks 방향으로 유도",
        start=(2.0, 6.0),  # y=6: racks(y=9~13)보다 아래
        goal=(18.0, 6.0),
        initial_heading=0.0,  # +x 방향
        robot_speed=1.5,
        trigger_distance=5.0,
        trigger_time=2.0,
        expected_evasion_direction="up",
        storage_racks_risk=True,
    ))

    # 시나리오 2: 빠른 로봇
    scenarios.append(CascadeScenario(
        name="S7-C2: Fast Robot Cascade",
        description="빠른 로봇 - 제동 거리 증가로 cascade 위험 상승",
        start=(2.0, 5.0),
        goal=(18.0, 5.0),
        initial_heading=0.0,
        robot_speed=2.0,
        trigger_distance=4.0,
        trigger_time=1.5,
        expected_evasion_direction="up",
        storage_racks_risk=True,
    ))

    # 시나리오 3: 다중 트리거
    scenarios.append(CascadeScenario(
        name="S7-C3: Multi-trigger Cascade",
        description="다중 작업자 출현 - 복잡한 회피 필요",
        start=(2.0, 7.0),
        goal=(18.0, 7.0),
        initial_heading=0.0,
        robot_speed=1.5,
        trigger_distance=6.0,
        trigger_time=2.5,
        expected_evasion_direction="up",
        storage_racks_risk=True,
    ))

    return scenarios


# =============================================================================
# Episode Result
# =============================================================================

@dataclass
class EpisodeResult:
    """에피소드 결과 - 리뷰어 피드백 전체 반영"""
    # Configuration
    velocity: float
    distance_L: float
    latency_ms: int
    jitter_ms: int
    loss_percent: float
    guard_mode: str
    seed: int
    scenario: str

    # Outcomes
    dynamic_zone_violation: bool   # 동적 구역 침입
    static_zone_violation: bool    # storage_racks 침입 (cascade 실패)
    reached_goal: bool
    stopped_safely: bool
    packet_lost: bool              # 구역 정보 손실

    # Timing measurements
    t_hazard_detected: float = 0.0
    t_hazard_published: float = 0.0
    t_hazard_received: float = 0.0
    t_stop_issued: float = 0.0
    t_robot_stopped: float = 0.0

    # Computed latencies
    tau_net_measured: float = 0.0      # t_received - t_published
    tau_total_measured: float = 0.0    # t_stop_issued - t_published

    # Effective deceleration
    a_eff_measured: float = 0.0

    # Distances
    min_distance_dynamic: float = float('inf')
    min_distance_static: float = float('inf')
    stopping_distance_actual: float = 0.0
    stopping_distance_theoretical: float = 0.0

    # Stop reason
    stop_reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# Simulation Environment
# =============================================================================

class SimulationEnvironment:
    """시뮬레이션 환경"""

    def __init__(self, world_config: WorldConfig, robot_config: RobotConfig):
        self.world_config = world_config
        self.robot_config = robot_config

        self.pose: Tuple[float, float] = (2.0, 6.0)
        self.heading: float = 0.0
        self.velocity: float = 0.0
        self.sim_time: float = 0.0

        self.static_zones = [world_config.get_static_zone()]
        self.velocity_logger = VelocityLogger(robot_config.dt)

    def reset(self, start_pose: Tuple[float, float], start_heading: float = 0.0):
        self.pose = start_pose
        self.heading = start_heading
        self.velocity = 0.0
        self.sim_time = 0.0
        self.velocity_logger.reset()

    def step(self, cmd_vel: float) -> Tuple[float, float]:
        """
        한 스텝 실행
        Returns: (actual_velocity, actual_deceleration)
        """
        dt = self.robot_config.dt
        prev_velocity = self.velocity

        # 가속/감속 제한 적용
        vel_diff = cmd_vel - self.velocity
        if vel_diff > 0:
            max_change = self.robot_config.max_acceleration * dt
        else:
            max_change = self.robot_config.nominal_deceleration * dt
            # Jerk 제한 적용 (실제 감속이 명목보다 약해질 수 있음)
            jerk_factor = random.uniform(0.7, 1.0)  # Nav2 컨트롤러 특성
            max_change *= jerk_factor

        vel_diff = max(-max_change, min(vel_diff, max_change))
        self.velocity = max(0, self.velocity + vel_diff)

        # 위치 업데이트
        self.pose = (
            self.pose[0] + self.velocity * math.cos(self.heading) * dt,
            self.pose[1] + self.velocity * math.sin(self.heading) * dt,
        )

        self.sim_time += dt

        # 속도 로깅
        self.velocity_logger.record(self.sim_time, self.velocity)

        # 실제 감속도 계산
        actual_decel = 0.0
        if prev_velocity > self.velocity:
            actual_decel = (prev_velocity - self.velocity) / dt

        return self.velocity, actual_decel

    def get_noisy_pose(self) -> Tuple[float, float]:
        noise_x = random.gauss(0, self.robot_config.pose_noise_std)
        noise_y = random.gauss(0, self.robot_config.pose_noise_std)
        return (self.pose[0] + noise_x, self.pose[1] + noise_y)

    def check_collision(self, zones: List[Polygon]) -> bool:
        robot_point = ShapelyPoint(self.pose)
        for zone in zones:
            if zone.contains(robot_point):
                return True
        return False


# =============================================================================
# Episode Runner
# =============================================================================

class EpisodeRunner:
    """에피소드 실행기"""

    def __init__(self, world_config: WorldConfig, robot_config: RobotConfig,
                 experiment_config: ExperimentConfig):
        self.world_config = world_config
        self.robot_config = robot_config
        self.experiment_config = experiment_config

    def run_episode(self, scenario: CascadeScenario, network_config: NetworkConfig,
                    guard_mode: GuardMode, seed: int) -> EpisodeResult:
        """단일 에피소드 실행"""
        random.seed(seed)
        np.random.seed(seed)

        # 초기화
        env = SimulationEnvironment(self.world_config, self.robot_config)
        env.reset(scenario.start, scenario.initial_heading)

        safety_manager = CentralSafetyManager(network_config)
        guard = create_guard(guard_mode, self.robot_config,
                           self.experiment_config.safety_margin,
                           network_config.latency_ms / 1000.0)

        # 결과 초기화
        result = EpisodeResult(
            velocity=scenario.robot_speed,
            distance_L=scenario.trigger_distance,
            latency_ms=network_config.latency_ms,
            jitter_ms=network_config.jitter_ms,
            loss_percent=network_config.loss_percent,
            guard_mode=guard_mode.value,
            seed=seed,
            scenario=scenario.name,
            dynamic_zone_violation=False,
            static_zone_violation=False,
            reached_goal=False,
            stopped_safely=False,
            packet_lost=False,
        )

        # 상태 플래그
        hazard_triggered = False
        hazard_event: Optional[HazardEvent] = None
        motion_started = False
        stop_issued = False
        position_at_stop_issued = None

        max_steps = int(30.0 / self.robot_config.dt)

        for step in range(max_steps):
            current_time = env.sim_time

            # 수신된 hazard 처리
            received = safety_manager.update(current_time)
            for event in received:
                result.t_hazard_received = current_time
                result.tau_net_measured = event.t_received - event.t_published

                # Guard latency 업데이트
                if isinstance(guard, GeofenceAdaptiveGuard):
                    latency_breakdown = event.get_latency_breakdown()
                    if latency_breakdown:
                        guard.update_latency_estimate(latency_breakdown.tau_total)

            # Hazard 트리거
            if (not hazard_triggered and motion_started and
                current_time >= scenario.trigger_time):

                hazard_event = safety_manager.detect_hazard(
                    env.pose, env.heading, scenario.trigger_distance, current_time,
                    event_type="worker_detected"
                )
                result.t_hazard_detected = hazard_event.t_detected
                result.t_hazard_published = hazard_event.t_published
                hazard_triggered = True

                # 패킷 손실 확인
                if hazard_event in safety_manager.lost_events:
                    result.packet_lost = True

            # 센서 읽기
            noisy_pose = env.get_noisy_pose()

            # 알려진 구역 (static + received dynamic)
            known_zones = env.static_zones + safety_manager.get_known_zones()

            # Guard 체크
            is_safe, stop_reason = guard.check_safety(
                noisy_pose, env.heading, env.velocity, known_zones
            )

            # 명령 결정
            if not is_safe and not stop_issued:
                cmd_vel = 0.0
                stop_issued = True
                result.t_stop_issued = current_time
                result.stop_reason = stop_reason
                position_at_stop_issued = env.pose

                if hazard_event:
                    hazard_event.t_stop_issued = current_time
                    result.tau_total_measured = current_time - result.t_hazard_published
            elif stop_issued:
                cmd_vel = 0.0
            else:
                cmd_vel = scenario.robot_speed
                motion_started = True

            # 스텝 실행
            actual_vel, actual_decel = env.step(cmd_vel)

            # 유효 감속도 업데이트
            if isinstance(guard, GeofenceAdaptiveGuard) and actual_decel > 0:
                guard.update_decel_estimate(actual_decel)

            # 정지 확인
            if stop_issued and env.velocity < 0.01:
                result.t_robot_stopped = current_time
                result.stopped_safely = True
                result.a_eff_measured = env.velocity_logger.compute_effective_deceleration()

                if position_at_stop_issued:
                    result.stopping_distance_actual = math.sqrt(
                        (env.pose[0] - position_at_stop_issued[0])**2 +
                        (env.pose[1] - position_at_stop_issued[1])**2
                    )
                break

            # Ground truth 구역으로 충돌 체크
            dynamic_zones = safety_manager.get_ground_truth_zones()
            static_zones = env.static_zones

            # 동적 구역 침입
            if env.check_collision(dynamic_zones):
                result.dynamic_zone_violation = True

            # 정적 구역(storage_racks) 침입 - cascade 실패
            if env.check_collision(static_zones):
                result.static_zone_violation = True
                break

            # 거리 추적
            robot_point = ShapelyPoint(env.pose)
            for zone in dynamic_zones:
                dist = robot_point.distance(zone)
                result.min_distance_dynamic = min(result.min_distance_dynamic, dist)
            for zone in static_zones:
                dist = robot_point.distance(zone)
                result.min_distance_static = min(result.min_distance_static, dist)

            # 목표 도달
            goal_dist = math.sqrt((env.pose[0] - scenario.goal[0])**2 +
                                 (env.pose[1] - scenario.goal[1])**2)
            if goal_dist < 0.5:
                result.reached_goal = True
                break

        # 이론적 정지 거리
        if result.a_eff_measured > 0:
            v = scenario.robot_speed
            result.stopping_distance_theoretical = v**2 / (2 * result.a_eff_measured)

        return result


# =============================================================================
# Experiment Orchestrator
# =============================================================================

class ExperimentOrchestrator:
    """실험 총괄"""

    def __init__(self, world_config: WorldConfig = None,
                 robot_config: RobotConfig = None,
                 experiment_config: ExperimentConfig = None):
        self.world_config = world_config or WorldConfig()
        self.robot_config = robot_config or RobotConfig()
        self.experiment_config = experiment_config or ExperimentConfig()
        self.runner = EpisodeRunner(self.world_config, self.robot_config,
                                   self.experiment_config)
        self.results: List[EpisodeResult] = []

    def run_sweep(self, guard_modes: List[GuardMode] = None) -> List[EpisodeResult]:
        """전체 파라미터 스윕"""
        if guard_modes is None:
            guard_modes = list(GuardMode)

        scenarios = create_cascade_scenarios()
        config = self.experiment_config

        # 총 에피소드 수 계산
        total = 0
        for L in config.distances:
            num_seeds = config.get_num_seeds(L)
            total += (len(scenarios) * len(guard_modes) * len(config.latencies_ms) *
                     len(config.jitter_ratios) * len(config.loss_percents) * num_seeds)

        print(f"\n{'='*70}")
        print("S7 NDHU IMPROVED: NETWORK-DELAYED HAZARD UPDATE")
        print(f"{'='*70}")
        print(f"Guard modes: {[m.value for m in guard_modes]}")
        print(f"Latencies: {config.latencies_ms} ms")
        print(f"Jitter ratios: {config.jitter_ratios}")
        print(f"Loss %: {config.loss_percents}")
        print(f"Distances: {config.distances} m")
        print(f"Seeds: {config.base_seeds} (transition: {config.transition_seeds})")
        print(f"Scenarios: {len(scenarios)}")
        print(f"Total episodes: {total}")
        print(f"{'='*70}\n")

        self.results = []
        count = 0

        for scenario in scenarios:
            print(f"[{scenario.name}]")
            for guard_mode in guard_modes:
                for latency_ms in config.latencies_ms:
                    for jitter_ratio in config.jitter_ratios:
                        for loss_pct in config.loss_percents:
                            # Distance를 scenario.trigger_distance로 고정하거나 sweep
                            for L in config.distances:
                                # 시나리오 복사 및 거리 수정
                                modified_scenario = CascadeScenario(
                                    name=scenario.name,
                                    description=scenario.description,
                                    start=scenario.start,
                                    goal=scenario.goal,
                                    initial_heading=scenario.initial_heading,
                                    robot_speed=scenario.robot_speed,
                                    trigger_distance=L,
                                    trigger_time=scenario.trigger_time,
                                    expected_evasion_direction=scenario.expected_evasion_direction,
                                    storage_racks_risk=scenario.storage_racks_risk,
                                )

                                network_config = NetworkConfig(
                                    latency_ms=latency_ms,
                                    jitter_ms=int(latency_ms * jitter_ratio),
                                    loss_percent=loss_pct,
                                )

                                num_seeds = config.get_num_seeds(L)
                                for seed in range(num_seeds):
                                    result = self.runner.run_episode(
                                        modified_scenario, network_config, guard_mode, seed
                                    )
                                    self.results.append(result)
                                    count += 1

                                    if count % 500 == 0:
                                        print(f"  Progress: {count}/{total} ({100*count/total:.1f}%)")

        return self.results

    def analyze_results(self) -> Dict:
        """결과 분석"""
        if not self.results:
            return {}

        summary = {
            "by_guard_mode": {},
            "transition_curves": {},
            "latency_breakdown": {},
            "effective_deceleration": {},
            "metadata": {
                "total_episodes": len(self.results),
                "timestamp": datetime.now().isoformat(),
            }
        }

        # Guard mode별 분석
        for mode in set(r.guard_mode for r in self.results):
            subset = [r for r in self.results if r.guard_mode == mode]
            dynamic_violations = sum(1 for r in subset if r.dynamic_zone_violation)
            static_violations = sum(1 for r in subset if r.static_zone_violation)

            summary["by_guard_mode"][mode] = {
                "total": len(subset),
                "dynamic_violations": dynamic_violations,
                "static_violations": static_violations,
                "dynamic_violation_rate": 100 * dynamic_violations / len(subset),
                "static_violation_rate": 100 * static_violations / len(subset),
                "cascade_rate": 100 * static_violations / max(1, dynamic_violations),
            }

            # τ_total 분포
            tau_totals = [r.tau_total_measured for r in subset if r.tau_total_measured > 0]
            if tau_totals:
                summary["latency_breakdown"][mode] = {
                    "tau_total_mean": np.mean(tau_totals),
                    "tau_total_std": np.std(tau_totals),
                    "tau_total_p90": np.percentile(tau_totals, 90),
                }

            # a_eff 분포
            a_effs = [r.a_eff_measured for r in subset if r.a_eff_measured > 0]
            if a_effs:
                summary["effective_deceleration"][mode] = {
                    "a_eff_mean": np.mean(a_effs),
                    "a_eff_std": np.std(a_effs),
                    "a_eff_p10": np.percentile(a_effs, 10),  # 보수적 추정
                }

        # Transition curves
        for mode in set(r.guard_mode for r in self.results):
            summary["transition_curves"][mode] = {}
            for latency in self.experiment_config.latencies_ms:
                curve = {}
                for L in self.experiment_config.distances:
                    subset = [r for r in self.results
                             if r.guard_mode == mode and r.latency_ms == latency
                             and r.distance_L == L]
                    if subset:
                        violations = sum(1 for r in subset if r.dynamic_zone_violation)
                        curve[L] = 100 * violations / len(subset)
                summary["transition_curves"][mode][latency] = curve

        return summary

    def print_summary(self):
        """결과 출력"""
        summary = self.analyze_results()

        print(f"\n{'='*80}")
        print("RESULTS SUMMARY")
        print(f"{'='*80}")

        print("\n[Violation Rates by Guard Mode]")
        print("-" * 60)
        print(f"{'Guard':<20} {'Dynamic':<12} {'Static':<12} {'Cascade':<12}")
        print("-" * 60)

        for mode, stats in summary["by_guard_mode"].items():
            dyn = stats["dynamic_violation_rate"]
            sta = stats["static_violation_rate"]
            cas = stats["cascade_rate"]
            print(f"{mode:<20} {dyn:>10.1f}% {sta:>10.1f}% {cas:>10.1f}%")

        print("\n[τ_total Measured (ms)]")
        print("-" * 60)
        for mode, stats in summary.get("latency_breakdown", {}).items():
            mean = stats["tau_total_mean"] * 1000
            p90 = stats["tau_total_p90"] * 1000
            print(f"  {mode}: mean={mean:.1f}ms, p90={p90:.1f}ms")

        print("\n[Effective Deceleration (m/s²)]")
        print("-" * 60)
        for mode, stats in summary.get("effective_deceleration", {}).items():
            mean = stats["a_eff_mean"]
            p10 = stats["a_eff_p10"]
            print(f"  {mode}: mean={mean:.2f}, p10={p10:.2f}")

    def save_results(self, filepath: str):
        """결과 저장"""
        output = {
            "summary": self.analyze_results(),
            "episodes": [r.to_dict() for r in self.results],
            "config": {
                "world": asdict(self.world_config),
                "robot": asdict(self.robot_config),
                "experiment": asdict(self.experiment_config),
            }
        }

        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {filepath}")


# =============================================================================
# Visualization
# =============================================================================

def plot_results(results_file: str, output_dir: str = None):
    """결과 시각화"""
    import matplotlib.pyplot as plt

    with open(results_file) as f:
        data = json.load(f)

    if output_dir is None:
        output_dir = os.path.dirname(results_file)
    os.makedirs(output_dir, exist_ok=True)

    summary = data["summary"]
    config = data["config"]["experiment"]
    distances = config["distances"]
    latencies = config["latencies_ms"]

    modes = ["no_guard", "safetychip", "selp", "geofence_adaptive"]
    colors = {
        "no_guard": "#d62728",
        "safetychip": "#1f77b4",
        "selp": "#ff7f0e",
        "geofence_adaptive": "#2ca02c",
    }
    labels = {
        "no_guard": "No Guard",
        "safetychip": "SafetyChip",
        "selp": "SELP",
        "geofence_adaptive": "Geofence Adaptive",
    }

    # Figure 1: Transition curves with τ_total annotation
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, mode in enumerate(modes):
        ax = axes[idx]
        curves = summary["transition_curves"].get(mode, {})

        for tau in latencies:
            curve = curves.get(str(tau), {})
            if curve:
                L_vals = sorted([float(k) for k in curve.keys()])
                vr_vals = [curve[str(int(L))] for L in L_vals]

                alpha = 0.4 + 0.6 * (latencies.index(tau) / len(latencies))
                ax.plot(L_vals, vr_vals, 'o-', label=f"τ_net={tau}ms",
                       color=colors[mode], alpha=alpha, linewidth=2)

        ax.set_xlabel("Distance L (m)")
        ax.set_ylabel("Violation Rate (%)")
        ax.set_title(labels[mode], fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-5, 105)

        # τ_total annotation
        tau_stats = summary.get("latency_breakdown", {}).get(mode, {})
        if tau_stats:
            tau_mean = tau_stats.get("tau_total_mean", 0) * 1000
            ax.annotate(f"τ_total(mean)={tau_mean:.0f}ms",
                       xy=(0.05, 0.95), xycoords='axes fraction',
                       fontsize=9, color='purple')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 's7_ndhu_improved_transition.png'), dpi=150)
    plt.close()
    print(f"Saved: s7_ndhu_improved_transition.png")

    # Figure 2: Comparison with τ_total and a_eff
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Guard comparison at τ=300ms
    target_tau = 300
    for mode in modes:
        curves = summary["transition_curves"].get(mode, {})
        curve = curves.get(str(target_tau), {})
        if curve:
            L_vals = sorted([float(k) for k in curve.keys()])
            vr_vals = [curve[str(int(L))] for L in L_vals]
            ax1.plot(L_vals, vr_vals, 'o-', label=labels[mode],
                    color=colors[mode], linewidth=2, markersize=8)

    ax1.set_xlabel("Distance L (m)", fontsize=12)
    ax1.set_ylabel("Violation Rate (%)", fontsize=12)
    ax1.set_title(f"Guard Comparison (τ_net={target_tau}ms)", fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-5, 105)

    # Right: τ_total and a_eff distribution
    modes_with_data = [m for m in modes if m in summary.get("latency_breakdown", {})]
    x = np.arange(len(modes_with_data))

    tau_means = [summary["latency_breakdown"][m]["tau_total_mean"] * 1000 for m in modes_with_data]
    a_eff_means = [summary.get("effective_deceleration", {}).get(m, {}).get("a_eff_mean", 0)
                  for m in modes_with_data]

    ax2_twin = ax2.twinx()

    bars1 = ax2.bar(x - 0.2, tau_means, 0.4, label='τ_total (ms)', color='steelblue', alpha=0.7)
    bars2 = ax2_twin.bar(x + 0.2, a_eff_means, 0.4, label='a_eff (m/s²)', color='coral', alpha=0.7)

    ax2.set_xticks(x)
    ax2.set_xticklabels([labels[m] for m in modes_with_data], rotation=15)
    ax2.set_ylabel("τ_total (ms)", color='steelblue')
    ax2_twin.set_ylabel("a_eff (m/s²)", color='coral')
    ax2.set_title("Measured τ_total and a_eff", fontweight='bold')

    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 's7_ndhu_improved_metrics.png'), dpi=150)
    plt.close()
    print(f"Saved: s7_ndhu_improved_metrics.png")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="S7 NDHU Improved Experiment")

    parser.add_argument('--sweep', action='store_true', help='Run full sweep')
    parser.add_argument('--output', type=str, default='s7_ndhu_improved_results.json')
    parser.add_argument('--plot', type=str, help='Plot results from file')

    # Single run
    parser.add_argument('--v', type=float, help='Velocity')
    parser.add_argument('--L', type=float, help='Distance')
    parser.add_argument('--tau_net_ms', type=int, help='Network latency')
    parser.add_argument('--jitter_ms', type=int, default=0, help='Jitter')
    parser.add_argument('--loss_pct', type=float, default=0.0, help='Loss %')
    parser.add_argument('--guard_mode', type=str, default='geofence_adaptive')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    if args.plot:
        plot_results(args.plot)
        return

    orchestrator = ExperimentOrchestrator()

    if args.sweep:
        orchestrator.run_sweep()
        orchestrator.print_summary()

        output_path = os.path.join(os.path.dirname(__file__), '..', args.output)
        orchestrator.save_results(output_path)

        plot_results(output_path, os.path.join(os.path.dirname(__file__), '..', 'figures'))

    elif args.v and args.L and args.tau_net_ms is not None:
        scenario = create_cascade_scenarios()[0]
        scenario.robot_speed = args.v
        scenario.trigger_distance = args.L

        network_config = NetworkConfig(
            latency_ms=args.tau_net_ms,
            jitter_ms=args.jitter_ms,
            loss_percent=args.loss_pct,
        )

        guard_mode = GuardMode(args.guard_mode)
        result = orchestrator.runner.run_episode(scenario, network_config, guard_mode, args.seed)

        print(f"\n{'='*60}")
        print("SINGLE EPISODE RESULT")
        print(f"{'='*60}")
        print(f"Config: v={args.v}, L={args.L}, τ_net={args.tau_net_ms}ms")
        print(f"        jitter={args.jitter_ms}ms, loss={args.loss_pct}%")
        print(f"Guard:  {args.guard_mode}")
        print()
        print(f"Dynamic zone violation: {result.dynamic_zone_violation}")
        print(f"Static zone violation:  {result.static_zone_violation}")
        print(f"Packet lost:            {result.packet_lost}")
        print()
        print(f"τ_net measured:   {result.tau_net_measured*1000:.1f} ms")
        print(f"τ_total measured: {result.tau_total_measured*1000:.1f} ms")
        print(f"a_eff measured:   {result.a_eff_measured:.2f} m/s²")
        print()
        print(f"Stop reason: {result.stop_reason}")
        print(f"{'='*60}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
