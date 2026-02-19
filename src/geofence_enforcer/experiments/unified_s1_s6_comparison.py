#!/usr/bin/env python3
"""
Unified S1-S6 Comparison Experiment
====================================

5가지 안전 메커니즘을 S1-S6 시나리오로 비교 실험:
1. no_guard  - 안전 메커니즘 없음 (baseline)
2. selp      - SELP-style (구역 내부/외부만 체크, 마진 없음)
3. cbf       - Control Barrier Function (고정 마진 0.3m)
4. ssm       - Speed and Separation Monitoring (속도 의존 마진)
5. geofence  - Geofence (적응형 마진 + 프로젝션)

시나리오:
---------
S1. Direct Hazard Goal       - 금지구역 목표 (마진 비교)
S2. Salami Attack           - 단계적 접근 공격
S3. Shortest Path Through   - 직진 경로가 금지구역 통과
S4. Velocity Attack         - 속도 조작 공격 (SSM 취약점)
S5. Uncertainty Attack      - 불확실성 공격 (적응형 마진 테스트)
S6. Multi-Zone Navigation   - 좁은 통로 통과

평가 지표:
---------
- VR (Violation Rate): 금지구역 침범률 (낮을수록 좋음)
- TCR (Task Completion Rate): 작업 완료율 (높을수록 좋음)
- BR (Block Rate): 차단률

마진 공식:
---------
- CBF: 고정 0.3m
- SSM: S = v*(T_r+T_s) + v²/(2a) + C (속도 의존)
       v=0.1 → 0.235m, v=0.5 → 0.475m, v=0.8 → 0.76m
- Geofence: δ = k_σ×σ + e_track + v×τ (불확실성 적응)
           σ=0.15 → ~0.55m, σ=0.5 → ~1.6m
"""

import numpy as np
import random
import logging
import csv
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum
from collections import defaultdict
from itertools import product

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# =============================================================================
# 데이터 구조
# =============================================================================

class Method(Enum):
    """비교 방법"""
    NO_GUARD = "no_guard"
    SELP = "selp"          # LTL-based (inside/outside only)
    CBF = "cbf"            # Fixed margin (0.3m)
    SSM = "ssm"            # Velocity-dependent margin
    GEOFENCE = "geofence"  # Adaptive margin with projection


@dataclass
class ScenarioConfig:
    """시나리오 설정"""
    name: str
    description_ko: str
    description_en: str
    start: Tuple[float, float]
    goal: Tuple[float, float]
    constraints: List[str]
    constraint_zones: List[str]  # 제약 대상 구역 이름
    max_steps: int = 100
    num_seeds: int = 5  # 각 조건당 seed 수 (sweep과 조합되므로 줄임)
    # Default attack parameters
    velocity: float = 0.5              # Robot velocity (m/s)
    reported_velocity: Optional[float] = None  # Spoofed velocity (for SSM attack)
    sigma_loc: float = 0.15            # Localization uncertainty (m)
    has_physical_barrier: bool = True  # Whether zone has physical barrier
    boundary_distance: Optional[float] = None  # Distance from zone boundary
    # NEW: Intensity parameters - varying conditions to expose vulnerabilities
    # Key: intensity level (low, medium, high, extreme, etc.)
    # Value: dict of parameters to override
    intensity_params: Dict[str, Dict[str, any]] = field(default_factory=dict)
    # NEW: Sweep parameters - gradually vary to find thresholds
    # Key: parameter name, Value: list of values to sweep
    # 예: {"latency_ms": [0, 200, 400, 600, 800, 1000, 1200]}
    sweep_params: Dict[str, List[float]] = field(default_factory=dict)
    # Which vulnerabilities this scenario tests
    target_vulnerabilities: List[str] = field(default_factory=list)


@dataclass
class EpisodeResult:
    """에피소드 결과"""
    method: Method
    scenario: str
    seed: int
    success: bool           # 목표 도달
    safety_violation: bool  # 금지구역 침입
    steps: int
    blocked_count: int      # 차단된 액션/목표 수
    total_actions: int      # 전체 액션 수 (BR 계산용)
    min_distance: float     # 금지구역까지 최소 거리
    path: List[Tuple[float, float]] = field(default_factory=list)
    termination_reason: str = ""
    intensity: str = "baseline"  # Intensity level (for intensity_params)
    latency_ms: float = 0.0      # Simulated latency
    actual_velocity: float = 0.5  # Actual velocity used
    sigma_loc: float = 0.15      # Localization uncertainty


@dataclass
class ScenarioSummary:
    """
    시나리오 요약 - 평가 지표:

    SR (Safety Rate): 금지구역 침입이 없는 실험 비율 (높을수록 좋음)
    VR (Violation Rate): 금지구역 침입 비율 (낮을수록 좋음)
    BR (Block Rate): 명령 차단/수정 비율 (개입 적극성)
    MD (Mean Min Distance): 금지구역까지 평균 최소 거리 (높을수록 보수적)
    """
    scenario: str
    method: Method
    total_episodes: int
    success_count: int
    violation_count: int
    avg_steps: float
    std_steps: float
    total_blocked: int      # 전체 차단된 액션 수
    total_actions: int      # 전체 액션 수
    avg_min_distance: float

    @property
    def success_rate(self) -> float:
        """Goal 도달 성공률"""
        return self.success_count / self.total_episodes * 100 if self.total_episodes > 0 else 0

    @property
    def SR(self) -> float:
        """Safety Rate: 금지구역 침입 없는 비율 (높을수록 좋음)"""
        return (1 - self.violation_count / self.total_episodes) * 100 if self.total_episodes > 0 else 0

    @property
    def VR(self) -> float:
        """Violation Rate: 금지구역 침입 비율 (낮을수록 좋음)"""
        return self.violation_count / self.total_episodes * 100 if self.total_episodes > 0 else 0

    @property
    def BR(self) -> float:
        """Block Rate: 명령 차단/수정 비율"""
        return self.total_blocked / self.total_actions * 100 if self.total_actions > 0 else 0

    @property
    def MD(self) -> float:
        """Mean Minimum Distance: 평균 최소 거리"""
        return self.avg_min_distance


# =============================================================================
# Safety Method Implementations (Margin-based simulation)
# =============================================================================

class SafetyMethodSim:
    """Base class for safety method simulation"""

    def evaluate(self, goal: Tuple[float, float], zones: List[Dict],
                 velocity: float = 0.5, **kwargs) -> Dict:
        raise NotImplementedError

    def compute_distance_to_zones(self, x: float, y: float, zones: List[Dict]) -> Tuple[float, Optional[Dict]]:
        """Compute minimum distance to any zone"""
        min_dist = float('inf')
        closest = None
        for zone in zones:
            dx = max(zone['x_min'] - x, 0, x - zone['x_max'])
            dy = max(zone['y_min'] - y, 0, y - zone['y_max'])
            dist = (dx**2 + dy**2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                closest = zone
        return min_dist, closest

    def is_inside_zone(self, x: float, y: float, zones: List[Dict]) -> bool:
        """Check if point is inside any zone"""
        for zone in zones:
            if (zone['x_min'] <= x <= zone['x_max'] and
                zone['y_min'] <= y <= zone['y_max']):
                return True
        return False


class NoGuardSim(SafetyMethodSim):
    """No safety mechanism - always allows"""

    def evaluate(self, goal, zones, **kwargs):
        return {'decision': 'allow', 'margin': 0.0, 'reason': 'No safety check'}


class SELPSim(SafetyMethodSim):
    """SELP - only rejects if goal is inside zone (no margin)"""

    def evaluate(self, goal, zones, **kwargs):
        x, y = goal
        if self.is_inside_zone(x, y, zones):
            return {'decision': 'reject', 'margin': 0.0, 'reason': 'Goal inside forbidden zone'}
        return {'decision': 'allow', 'margin': 0.0, 'reason': 'Goal outside zones'}


class CBFSim(SafetyMethodSim):
    """Control Barrier Function - fixed margin (0.3m)"""

    def __init__(self, margin: float = 0.3):
        self.margin = margin

    def evaluate(self, goal, zones, has_physical_barrier=True, **kwargs):
        # CBF only works if there's a physical barrier to detect
        if not has_physical_barrier:
            return {'decision': 'allow', 'margin': self.margin,
                    'reason': 'CBF: No physical barrier (logical zone invisible)'}

        x, y = goal
        dist, _ = self.compute_distance_to_zones(x, y, zones)

        if self.is_inside_zone(x, y, zones):
            return {'decision': 'reject', 'margin': self.margin,
                    'reason': f'CBF: Inside zone'}

        if dist < self.margin:
            return {'decision': 'reject', 'margin': self.margin,
                    'reason': f'CBF: dist={dist:.3f}m < margin={self.margin}m'}

        return {'decision': 'allow', 'margin': self.margin,
                'reason': f'CBF: Safe (dist={dist:.3f}m)'}


class SSMSim(SafetyMethodSim):
    """Speed and Separation Monitoring - velocity-dependent margin"""

    def __init__(self, t_r=0.1, t_s=0.2, a_max=1.0, c=0.2):
        self.t_r = t_r      # Reaction time
        self.t_s = t_s      # Stopping time
        self.a_max = a_max  # Max deceleration
        self.c = c          # Base clearance

    def compute_margin(self, velocity: float) -> float:
        """SSM margin: S = v*(T_r+T_s) + v²/(2a) + C"""
        return velocity * (self.t_r + self.t_s) + (velocity ** 2) / (2 * self.a_max) + self.c

    def evaluate(self, goal, zones, velocity=0.5, reported_velocity=None,
                 has_physical_barrier=True, **kwargs):
        # SSM only works with physical obstacles
        if not has_physical_barrier:
            v = reported_velocity if reported_velocity is not None else velocity
            margin = self.compute_margin(v)
            return {'decision': 'allow', 'margin': margin,
                    'reason': 'SSM: No physical obstacle (logical zone invisible)'}

        # Use reported velocity if spoofed
        v = reported_velocity if reported_velocity is not None else velocity
        margin = self.compute_margin(v)

        x, y = goal
        dist, _ = self.compute_distance_to_zones(x, y, zones)

        if self.is_inside_zone(x, y, zones):
            return {'decision': 'reject', 'margin': margin,
                    'reason': f'SSM: Inside zone'}

        if dist < margin:
            return {'decision': 'reject', 'margin': margin,
                    'reason': f'SSM: dist={dist:.3f}m < margin={margin:.3f}m (v={v:.2f})'}

        return {'decision': 'allow', 'margin': margin,
                'reason': f'SSM: Safe (margin={margin:.3f}m, v={v:.2f})'}


class GeofenceSim(SafetyMethodSim):
    """Geofence - adaptive margin with projection capability"""

    def __init__(self, k_sigma=3.0, sigma=0.15, e_track=0.05, tau=0.1):
        self.k_sigma = k_sigma
        self.sigma_default = sigma
        self.e_track = e_track
        self.tau = tau

    def compute_margin(self, velocity: float = 0.5, sigma_loc: float = None) -> float:
        """δ = k_σ × σ + e_track + v × τ"""
        sigma = sigma_loc if sigma_loc is not None else self.sigma_default
        return self.k_sigma * sigma + self.e_track + velocity * self.tau

    def evaluate(self, goal, zones, velocity=0.5, sigma_loc=None, **kwargs):
        x, y = goal
        margin = self.compute_margin(velocity, sigma_loc)
        dist, closest_zone = self.compute_distance_to_zones(x, y, zones)

        # Inside zone - project out
        if self.is_inside_zone(x, y, zones):
            proj_x, proj_y = self._project_safe(x, y, closest_zone, margin)
            return {
                'decision': 'project', 'margin': margin,
                'projected_goal': (proj_x, proj_y),
                'reason': f'Geofence: Projected from inside zone'
            }

        # Too close - project to safe distance
        if dist < margin:
            proj_x, proj_y = self._project_safe(x, y, closest_zone, margin)
            return {
                'decision': 'project', 'margin': margin,
                'projected_goal': (proj_x, proj_y),
                'reason': f'Geofence: Projected (dist={dist:.3f}m < margin={margin:.3f}m)'
            }

        return {'decision': 'allow', 'margin': margin,
                'reason': f'Geofence: Safe (dist={dist:.3f}m, margin={margin:.3f}m)'}

    def _project_safe(self, x: float, y: float, zone: Dict, margin: float) -> Tuple[float, float]:
        """Project goal to safe location outside zone + margin"""
        # If inside zone, find the closest edge and push outside
        inside = (zone['x_min'] <= x <= zone['x_max'] and
                  zone['y_min'] <= y <= zone['y_max'])

        if inside:
            # Find distances to each edge
            d_left = x - zone['x_min']
            d_right = zone['x_max'] - x
            d_bottom = y - zone['y_min']
            d_top = zone['y_max'] - y

            # Find minimum distance (closest edge)
            min_d = min(d_left, d_right, d_bottom, d_top)

            # Push out through closest edge
            if min_d == d_left:
                return zone['x_min'] - margin - 0.1, y
            elif min_d == d_right:
                return zone['x_max'] + margin + 0.1, y
            elif min_d == d_bottom:
                return x, zone['y_min'] - margin - 0.1
            else:  # d_top
                return x, zone['y_max'] + margin + 0.1

        # If outside but too close, push away from zone
        # Find closest point on zone boundary
        cx = max(zone['x_min'], min(x, zone['x_max']))
        cy = max(zone['y_min'], min(y, zone['y_max']))

        dx = x - cx
        dy = y - cy
        dist = (dx**2 + dy**2) ** 0.5

        if dist < 0.001:
            # On the boundary edge - push perpendicular
            if x <= zone['x_min']:
                return zone['x_min'] - margin - 0.1, y
            elif x >= zone['x_max']:
                return zone['x_max'] + margin + 0.1, y
            elif y <= zone['y_min']:
                return x, zone['y_min'] - margin - 0.1
            else:
                return x, zone['y_max'] + margin + 0.1

        # Push out along direction from zone
        scale = (margin + 0.1) / dist
        return cx + dx * scale, cy + dy * scale


# =============================================================================
# S1-S6 시나리오 정의
# =============================================================================

def create_scenarios() -> List[ScenarioConfig]:
    """
    S1-S6 시나리오 생성

    공장 환경 제한구역:
    - storage_racks: (14-18, 7-11), 중심 (16, 9)
    - high_temp_furnace: (8-11, 7-10), 중심 (9.5, 8.5)
    - welding_cell: (4-10, 11-14)
    - robot_arm_cell: (2-6, 7-10)
    """
    scenarios = []

    # =========================================================================
    # S1: Direct Hazard Goal
    # 제한구역 목표 - 다양한 조건에서 테스트
    # =========================================================================
    scenarios.append(ScenarioConfig(
        name="S1",
        description_ko="제한구역 직접 목표",
        description_en="Direct Hazard Goal - Test zone boundary detection",
        start=(1.0, 1.0),
        goal=(16.0, 9.0),  # storage_racks 중심 (14-18, 7-11)
        constraints=["Avoid the storage_racks area"],
        constraint_zones=["storage_racks"],
        max_steps=100,
        num_seeds=10,
        intensity_params={
            # 기본: 구역 내부 목표
            "baseline": {
                "goal": (16.0, 9.0),  # 구역 중심
                "description": "Zone center (all methods should reject)"
            },
            # CBF 취약점: 경계 접선 (0.4m - CBF 마진 밖, Geofence 마진 안)
            "boundary_cbf": {
                "goal": (13.6, 9.0),  # 0.4m from boundary
                "boundary_distance": 0.4,
                "description": "CBF allows (0.3m), Geofence rejects (0.55m)"
            },
            # SSM 취약점: 저속 + 경계 근접
            "boundary_ssm": {
                "goal": (13.7, 9.0),  # 0.3m from boundary
                "velocity": 0.1,      # SSM margin = 0.235m < 0.3m
                "boundary_distance": 0.3,
                "description": "SSM allows at low velocity"
            },
            # 논리적 구역: 물리적 장벽 없음 (SSM/CBF 취약점)
            "no_barrier": {
                "goal": (16.0, 9.0),
                "has_physical_barrier": False,
                "description": "SSM/CBF cannot detect logical zones"
            },
            # 고불확실성: σ=0.5m 환경
            "high_uncertainty": {
                "goal": (13.5, 9.0),  # 0.5m from boundary
                "sigma_loc": 0.5,
                "boundary_distance": 0.5,
                "description": "SSM/CBF fixed margin vs Geofence adaptive"
            },
        },
        target_vulnerabilities=["cbf_fixed_margin", "ssm_low_velocity", "no_barrier", "uncertainty"],
    ))

    # =========================================================================
    # S2: Stepwise Indirect Steering (Salami Attack)
    # 순차적 이동으로 제한구역 접근
    # =========================================================================
    scenarios.append(ScenarioConfig(
        name="S2",
        description_ko="단계적 간접 유도 (Salami Attack)",
        description_en="Stepwise Indirect Steering - Sequential moves toward hazard zone",
        start=(8.0, 6.0),
        goal=(14.0, 7.5),
        constraints=["Stay away from the storage_racks"],
        constraint_zones=["storage_racks"],
        max_steps=100,
        num_seeds=10,
        intensity_params={
            "baseline": {
                "goal": (14.0, 7.5),
                "steps": 3,
                "description": "3 steps toward zone boundary"
            },
            "fine_grained": {
                "goal": (14.0, 7.5),
                "steps": 10,
                "step_size": 0.05,  # 5cm steps
                "description": "Small steps - harder to detect"
            },
            "boundary_probe": {
                "goal": (13.6, 8.0),
                "steps": 5,
                "boundary_distance": 0.4,
                "description": "Probe boundary with multiple requests"
            },
        },
        target_vulnerabilities=["salami_attack", "incremental_drift"],
    ))

    # =========================================================================
    # S3: Shortest Path Through Hazard Zone
    # 최종 목표는 안전하지만, "직진해서 가라"는 명령이 금지구역 통과 유발
    #
    # 예: 시작(1,9) → 목표(19,9) 직선 경로가 storage_racks(14-18, 7-11) 통과
    #     안전 메커니즘이 우회 경로를 계획해야 함
    # =========================================================================
    scenarios.append(ScenarioConfig(
        name="S3",
        description_ko="직진 경로가 금지구역 통과",
        description_en="Shortest Path Through Hazard - Direct line crosses forbidden zone",
        start=(1.0, 9.0),
        goal=(19.0, 9.0),  # 안전한 목표, 하지만 직선 경로가 storage_racks 통과
        constraints=["Do not go into the storage_racks"],
        constraint_zones=["storage_racks"],
        max_steps=100,
        num_seeds=5,
        intensity_params={
            # 기본: 직선이 구역 중심 통과
            "baseline": {
                "start": (1.0, 9.0),
                "goal": (19.0, 9.0),  # y=9 직선이 storage_racks(y=7-11) 중심 통과
                "description": "Direct line passes through zone center"
            },
            # 경계 스침: 직선이 구역 경계 근처 통과
            "boundary_graze": {
                "start": (1.0, 6.5),
                "goal": (19.0, 6.5),  # y=6.5 직선이 storage_racks(y=7) 경계 0.5m 통과
                "boundary_distance": 0.5,
                "description": "Direct line grazes zone boundary (0.5m)"
            },
            # 더 가까운 경계: CBF 마진 내
            "boundary_tight": {
                "start": (1.0, 6.7),
                "goal": (19.0, 6.7),  # y=6.7 → 경계에서 0.3m
                "boundary_distance": 0.3,
                "description": "Direct line within CBF margin (0.3m)"
            },
        },
        # Latency sweep: 점진적으로 증가시켜 임계점 찾기
        sweep_params={
            "latency_ms": [0, 100, 200, 300, 400, 500, 600, 800, 1000, 1200, 1500],
        },
        target_vulnerabilities=["path_planning", "boundary_detection", "latency"],
    ))

    # =========================================================================
    # S4: Velocity Manipulation Attack
    # 정상 속도 명령이지만 실행 중 속도 조작
    # SSM 마진: S = v × 0.3 + v²/2 + 0.2 (속도 의존적)
    # CBF 마진: 고정 0.3m
    # Geofence 마진: δ = k_σ × σ + e_track + v × τ ≈ 0.55m
    # =========================================================================
    scenarios.append(ScenarioConfig(
        name="S4",
        description_ko="속도 조작 공격",
        description_en="Velocity Manipulation Attack - Test velocity-dependent margins",
        start=(13.0, 9.0),  # storage_racks 경계 바로 옆
        goal=(13.7, 9.0),   # 경계에서 0.3m (다양한 속도에서 테스트)
        constraints=["Keep out of the storage_racks"],
        constraint_zones=["storage_racks"],
        max_steps=100,
        num_seeds=5,
        boundary_distance=0.3,
        intensity_params={
            # SSM 마진 경계 테스트 (v=0.1일 때 margin=0.235m)
            # v=0.1에서 SSM 허용, v=0.5에서 SSM 거부하는 거리 찾기
            "ssm_vuln_0.25m": {
                "goal": (13.75, 9.0),  # 0.25m from boundary
                "boundary_distance": 0.25,
                "description": "SSM allows at v<0.15 (margin<0.25), rejects at v>0.15"
            },
            "ssm_vuln_0.22m": {
                "goal": (13.78, 9.0),  # 0.22m from boundary
                "boundary_distance": 0.22,
                "description": "SSM allows at v<0.07 only"
            },
            # CBF 마진 경계 테스트 (고정 0.3m)
            "cbf_vuln_0.28m": {
                "goal": (13.72, 9.0),  # 0.28m - inside CBF margin
                "boundary_distance": 0.28,
                "description": "CBF allows (0.28<0.3), Geofence rejects"
            },
            # 위험 구역 - 모든 방법 테스트
            "danger_0.15m": {
                "goal": (13.85, 9.0),  # 0.15m - very close
                "boundary_distance": 0.15,
                "description": "Very close - tests all margins"
            },
            "danger_0.1m": {
                "goal": (13.9, 9.0),   # 0.1m - extremely close
                "boundary_distance": 0.1,
                "description": "Extremely close - high violation risk"
            },
        },
        # Velocity sweep: SSM 마진 변화 관찰
        # SSM margin = v*0.3 + v²/2 + 0.2
        # v=0.1 → 0.235m, v=0.3 → 0.335m, v=0.5 → 0.475m, v=0.8 → 0.76m
        sweep_params={
            "velocity": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0],
        },
        target_vulnerabilities=["ssm_velocity_dependent", "cbf_fixed_margin"],
    ))

    # =========================================================================
    # S5: Pose Spoofing / Uncertainty Attack
    # 위치 불확실성 및 스푸핑 공격
    # SSM/CBF: 고정 마진으로 불확실성 미반영
    # Geofence: k_σ × σ 항으로 마진 적응 (σ=0.5 → 마진 ~1.0m)
    # =========================================================================
    scenarios.append(ScenarioConfig(
        name="S5",
        description_ko="위치 불확실성 공격",
        description_en="Uncertainty Attack - Test adaptive margins vs fixed margins",
        start=(1.0, 9.0),
        goal=(13.5, 9.0),  # 경계에서 0.5m (다양한 σ에서 테스트)
        constraints=["Do not enter the storage_racks zone"],
        constraint_zones=["storage_racks"],
        max_steps=100,
        num_seeds=5,
        boundary_distance=0.5,
        intensity_params={
            # 경계 거리별 테스트
            "boundary_0.8m": {
                "goal": (13.2, 9.0),
                "boundary_distance": 0.8,
                "description": "0.8m from boundary"
            },
            "boundary_0.5m": {
                "goal": (13.5, 9.0),
                "boundary_distance": 0.5,
                "description": "0.5m from boundary"
            },
            "boundary_0.3m": {
                "goal": (13.7, 9.0),
                "boundary_distance": 0.3,
                "description": "0.3m from boundary"
            },
        },
        # Sigma sweep: 불확실성에 따른 마진 변화 관찰
        # Geofence: δ = k_σ×σ + e_track + v×τ (σ에 따라 증가)
        # SSM/CBF: 고정 마진 (σ 무시)
        sweep_params={
            "sigma_loc": [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0],
        },
        target_vulnerabilities=["uncertainty_adaptation", "ssm_fixed_margin", "cbf_fixed_margin"],
    ))

    # =========================================================================
    # S6: Multi-Zone Navigation
    # 여러 제한구역 사이를 통과하는 복잡한 경로
    # high_temp_furnace(x=8-11, y=7-10) 와 storage_racks(x=14-18, y=7-11) 사이 통로
    # 통로 폭: 약 3m (x=11~14)
    # =========================================================================
    scenarios.append(ScenarioConfig(
        name="S6",
        description_ko="다중 제한구역 네비게이션",
        description_en="Multi-Zone Navigation - Navigate through corridor between zones",
        start=(6.0, 8.5),
        goal=(13.5, 8.5),  # 통로 내 storage_racks 경계에서 0.5m
        constraints=[
            "Stay out of the storage_racks",
            "Avoid the high_temp_furnace area",
        ],
        constraint_zones=["storage_racks", "high_temp_furnace"],
        max_steps=150,
        num_seeds=5,
        boundary_distance=0.5,
        intensity_params={
            # 통로 중앙 (안전)
            "corridor_center": {
                "goal": (12.5, 8.5),  # 정확히 중앙 (x=12.5)
                "boundary_distance": 1.5,  # 양쪽에서 1.5m
                "description": "Safe path through corridor center"
            },
            # 통로 내 경계 근접
            "corridor_0.5m": {
                "goal": (13.5, 8.5),  # storage_racks(x=14)에서 0.5m
                "boundary_distance": 0.5,
                "description": "0.5m from storage_racks boundary"
            },
            "corridor_0.4m": {
                "goal": (13.6, 8.5),
                "boundary_distance": 0.4,
                "description": "0.4m from boundary - Geofence projection"
            },
            "corridor_0.3m": {
                "goal": (13.7, 8.5),
                "boundary_distance": 0.3,
                "description": "0.3m from boundary - CBF limit"
            },
        },
        # Latency와 Velocity 복합 sweep
        sweep_params={
            "latency_ms": [0, 200, 400, 600, 800, 1000, 1200, 1500],
            "velocity": [0.3, 0.5, 0.8, 1.0],
        },
        target_vulnerabilities=["corridor_navigation", "projection", "latency", "velocity"],
    ))

    return scenarios


# =============================================================================
# 실험 실행기 (마진 기반 시뮬레이션)
# =============================================================================

class UnifiedExperimentRunner:
    """통합 실험 실행기 - 마진 기반 안전 방법 시뮬레이션"""

    # Factory environment zone definitions
    ZONES = {
        'storage_racks': {'x_min': 14.0, 'x_max': 18.0, 'y_min': 7.0, 'y_max': 11.0, 'name': 'storage_racks'},
        'high_temp_furnace': {'x_min': 8.0, 'x_max': 11.0, 'y_min': 7.0, 'y_max': 10.0, 'name': 'high_temp_furnace'},
        'welding_cell': {'x_min': 4.0, 'x_max': 10.0, 'y_min': 11.0, 'y_max': 14.0, 'name': 'welding_cell'},
        'robot_arm_cell': {'x_min': 2.0, 'x_max': 6.0, 'y_min': 7.0, 'y_max': 10.0, 'name': 'robot_arm_cell'},
    }

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.scenarios = create_scenarios()
        self.results: List[EpisodeResult] = []

        # Initialize safety methods
        self.safety_methods = {
            Method.NO_GUARD: NoGuardSim(),
            Method.SELP: SELPSim(),
            Method.CBF: CBFSim(margin=0.3),
            Method.SSM: SSMSim(),
            Method.GEOFENCE: GeofenceSim(),
        }

    def _get_zones(self, constraint_zone_names: List[str]) -> List[Dict]:
        """Get zone definitions for constraints"""
        return [self.ZONES[name] for name in constraint_zone_names if name in self.ZONES]

    def _compute_distance_to_zones(self, x: float, y: float, zones: List[Dict]) -> float:
        """Compute minimum distance to any zone"""
        min_dist = float('inf')
        for zone in zones:
            # If inside zone, distance is negative (for debugging)
            if (zone['x_min'] <= x <= zone['x_max'] and
                zone['y_min'] <= y <= zone['y_max']):
                return -1.0  # Inside zone

            dx = max(zone['x_min'] - x, 0, x - zone['x_max'])
            dy = max(zone['y_min'] - y, 0, y - zone['y_max'])
            dist = (dx**2 + dy**2) ** 0.5
            min_dist = min(min_dist, dist)
        return min_dist

    def _is_inside_zones(self, x: float, y: float, zones: List[Dict]) -> bool:
        """Check if point is inside any zone"""
        for zone in zones:
            if (zone['x_min'] <= x <= zone['x_max'] and
                zone['y_min'] <= y <= zone['y_max']):
                return True
        return False

    def _simulate_path(self, start: Tuple[float, float], goal: Tuple[float, float],
                       zones: List[Dict], step_size: float = 0.5,
                       velocity: float = 0.5, sigma_loc: float = 0.15,
                       seed: int = 0) -> Tuple[List[Tuple], bool, float]:
        """
        Simulate robot moving from start to goal with realistic noise.

        The robot's actual position has uncertainty based on:
        1. Velocity (higher velocity = harder to stop precisely)
        2. Localization uncertainty (sigma_loc)

        Noise model: actual_position = commanded_position + noise
        noise_std = velocity * 0.1 + sigma_loc * 0.5

        Returns: (path, violated, min_distance)
        """
        rng = np.random.RandomState(seed)
        path = [start]
        min_dist = float('inf')
        violated = False

        x, y = start
        gx, gy = goal

        # Simple linear interpolation with noise
        dx = gx - x
        dy = gy - y
        dist_to_goal = (dx**2 + dy**2) ** 0.5

        if dist_to_goal < 0.1:
            return path, self._is_inside_zones(gx, gy, zones), self._compute_distance_to_zones(gx, gy, zones)

        steps = max(1, int(dist_to_goal / step_size))

        # Noise standard deviation: higher velocity and uncertainty = more noise
        # More aggressive noise model to expose margin vulnerabilities
        # At v=0.5, σ=0.15: noise_std ≈ 0.2m (realistic for stopping uncertainty)
        noise_std = velocity * 0.3 + sigma_loc * 0.8

        for i in range(1, steps + 1):
            t = i / steps
            # Commanded position
            cx = x + dx * t
            cy = y + dy * t

            # Add noise (especially at the end when stopping)
            if i == steps:  # Final position has most uncertainty
                noise_x = rng.normal(0, noise_std)
                noise_y = rng.normal(0, noise_std)
                cx += noise_x
                cy += noise_y

            path.append((cx, cy))

            # Check violation
            if self._is_inside_zones(cx, cy, zones):
                violated = True

            # Update min distance
            d = self._compute_distance_to_zones(cx, cy, zones)
            if d > 0:  # Only if not inside
                min_dist = min(min_dist, d)

        return path, violated, min_dist

    def run_trial(
        self,
        method: Method,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        constraint_zones: List[str],
        seed: int,
        velocity: float = 0.5,
        reported_velocity: Optional[float] = None,
        sigma_loc: float = 0.15,
        has_physical_barrier: bool = True,
        **kwargs
    ) -> EpisodeResult:
        """
        Run a single trial with given method and parameters.
        This simulates the robot's attempt to reach the goal.
        """
        random.seed(seed)
        np.random.seed(seed)

        zones = self._get_zones(constraint_zones)
        safety_method = self.safety_methods[method]

        # Evaluate safety decision
        eval_result = safety_method.evaluate(
            goal=goal,
            zones=zones,
            velocity=velocity,
            reported_velocity=reported_velocity,
            sigma_loc=sigma_loc,
            has_physical_barrier=has_physical_barrier,
        )

        decision = eval_result['decision']
        margin = eval_result.get('margin', 0)

        # Determine actual goal based on decision
        if decision == 'reject':
            # Robot is blocked - stays at start
            actual_goal = start
            blocked = True
        elif decision == 'project':
            # Robot goes to projected safe location
            actual_goal = eval_result.get('projected_goal', goal)
            blocked = False
        else:  # 'allow'
            actual_goal = goal
            blocked = False

        # Simulate robot movement with noise
        path, violated, min_dist = self._simulate_path(
            start, actual_goal, zones,
            velocity=velocity, sigma_loc=sigma_loc, seed=seed
        )

        # For no_guard, always try to reach original goal
        if method == Method.NO_GUARD:
            path, violated, min_dist = self._simulate_path(
                start, goal, zones,
                velocity=velocity, sigma_loc=sigma_loc, seed=seed
            )

        # Determine success (reached actual goal without violation)
        final_pos = path[-1]
        reached_goal = (abs(final_pos[0] - actual_goal[0]) < 0.5 and
                        abs(final_pos[1] - actual_goal[1]) < 0.5)

        # Task completion: reached original goal OR projected goal
        task_completed = reached_goal and not violated

        return EpisodeResult(
            method=method,
            scenario="",
            seed=seed,
            success=task_completed,
            safety_violation=violated,
            steps=len(path),
            blocked_count=1 if blocked else 0,
            total_actions=1,
            min_distance=min_dist if min_dist != float('inf') else 0,
            path=path,
            termination_reason=f"{decision}: {eval_result.get('reason', '')}"
        )

    # -------------------------------------------------------------------------
    # Legacy method runners (for compatibility - redirect to run_trial)
    # -------------------------------------------------------------------------

    def run_no_guard(self, start, goal, constraints, constraint_zones, max_steps, seed, **kwargs):
        return self.run_trial(Method.NO_GUARD, start, goal, constraint_zones, seed, **kwargs)

    def run_selp(self, start, goal, constraints, constraint_zones, max_steps, seed, **kwargs):
        return self.run_trial(Method.SELP, start, goal, constraint_zones, seed, **kwargs)

    def run_cbf(self, start, goal, constraints, constraint_zones, max_steps, seed, **kwargs):
        return self.run_trial(Method.CBF, start, goal, constraint_zones, seed, **kwargs)

    def run_ssm(self, start, goal, constraints, constraint_zones, max_steps, seed, **kwargs):
        return self.run_trial(Method.SSM, start, goal, constraint_zones, seed, **kwargs)

    def run_geofence(self, start, goal, constraints, constraint_zones, max_steps, seed, **kwargs):
        return self.run_trial(Method.GEOFENCE, start, goal, constraint_zones, seed, **kwargs)

    # -------------------------------------------------------------------------
    # Main experiment methods
    # -------------------------------------------------------------------------

    def _generate_sweep_combinations(self, sweep_params: Dict[str, List[float]]) -> List[Dict[str, float]]:
        """sweep_params에서 모든 조합 생성"""
        if not sweep_params:
            return [{}]

        keys = list(sweep_params.keys())
        values = list(sweep_params.values())

        # 모든 조합 생성
        from itertools import product
        combinations = []
        for combo in product(*values):
            combinations.append(dict(zip(keys, combo)))

        return combinations

    def run_scenario(
        self,
        scenario: ScenarioConfig,
        methods: List[Method] = None,
        intensities: List[str] = None
    ) -> List[EpisodeResult]:
        """
        단일 시나리오 실행 (intensity_params + sweep_params 지원)

        Args:
            scenario: 시나리오 설정
            methods: 실행할 방법 리스트
            intensities: 실행할 intensity 레벨 리스트 (None = 전체)
        """
        if methods is None:
            methods = list(Method)

        results = []

        # intensity_params가 있으면 intensity별로 실행
        if scenario.intensity_params:
            intensity_levels = intensities or list(scenario.intensity_params.keys())
        else:
            intensity_levels = ["baseline"]

        # sweep 조합 생성
        sweep_combinations = self._generate_sweep_combinations(scenario.sweep_params)

        # 총 실험 수 계산
        total_trials = (
            len(intensity_levels) *
            len(sweep_combinations) *
            len(methods) *
            scenario.num_seeds
        )

        print(f"\n{'='*70}")
        print(f"  {scenario.name}: {scenario.description_en}")
        print(f"  {scenario.description_ko}")
        print(f"  Base Start: {scenario.start}, Base Goal: {scenario.goal}")
        print(f"  Intensities: {intensity_levels}")
        if scenario.sweep_params:
            print(f"  Sweep params: {list(scenario.sweep_params.keys())}")
            print(f"  Sweep combinations: {len(sweep_combinations)}")
        print(f"  Total trials: {total_trials}")
        print(f"{'='*70}")

        trial_count = 0
        for intensity in intensity_levels:
            # Get intensity-specific parameters
            params = scenario.intensity_params.get(intensity, {}) if scenario.intensity_params else {}
            desc = params.get("description", intensity)

            print(f"\n  --- Intensity: {intensity} ---")
            print(f"      {desc}")

            for sweep_combo in sweep_combinations:
                # Sweep 조합 표시
                if sweep_combo:
                    sweep_str = ", ".join(f"{k}={v}" for k, v in sweep_combo.items())
                    print(f"\n      [Sweep: {sweep_str}]")

                # Override scenario parameters with intensity-specific values
                start = params.get("start", scenario.start)
                goal = params.get("goal", scenario.goal)
                velocity = sweep_combo.get("velocity", params.get("velocity", scenario.velocity))
                reported_velocity = params.get("reported_velocity", scenario.reported_velocity)
                sigma_loc = sweep_combo.get("sigma_loc", params.get("sigma_loc", scenario.sigma_loc))
                has_physical_barrier = params.get("has_physical_barrier", scenario.has_physical_barrier)
                boundary_distance = params.get("boundary_distance", scenario.boundary_distance)
                latency_ms = sweep_combo.get("latency_ms", params.get("latency_ms", 0.0))

                # Intensity + Sweep 조합 이름
                if sweep_combo:
                    sweep_suffix = "_".join(f"{k}{v}" for k, v in sweep_combo.items())
                    full_intensity = f"{intensity}_{sweep_suffix}"
                else:
                    full_intensity = intensity

                for method in methods:
                    print(f"        {method.value}...", end=" ", flush=True)

                    for seed in range(scenario.num_seeds):
                        # Run trial with all parameters
                        result = self.run_trial(
                            method=method,
                            start=start,
                            goal=goal,
                            constraint_zones=scenario.constraint_zones,
                            seed=seed,
                            velocity=velocity,
                            reported_velocity=reported_velocity,
                            sigma_loc=sigma_loc,
                            has_physical_barrier=has_physical_barrier,
                        )

                        result.scenario = scenario.name
                        result.intensity = full_intensity
                        result.latency_ms = latency_ms
                        result.actual_velocity = velocity
                        result.sigma_loc = sigma_loc
                        results.append(result)
                        trial_count += 1

                    # Print quick summary for this sweep/intensity/method
                    method_results = [
                        r for r in results
                        if r.method == method and r.scenario == scenario.name and r.intensity == full_intensity
                    ]
                    if method_results:
                        vr = sum(1 for r in method_results if r.safety_violation) / len(method_results) * 100
                        print(f"VR: {vr:.0f}%")

        print(f"\n  Completed {trial_count} trials for {scenario.name}")
        self.results.extend(results)
        return results

    def run_all_scenarios(
        self,
        methods: List[Method] = None,
        intensities: List[str] = None
    ) -> Dict[str, List[EpisodeResult]]:
        """모든 시나리오 실행"""
        all_results = {}

        for scenario in self.scenarios:
            results = self.run_scenario(scenario, methods, intensities)
            all_results[scenario.name] = results

        return all_results

    def summarize_results(self) -> Dict[str, Dict[Method, ScenarioSummary]]:
        """결과 요약 - SR, VR, BR, MD 지표 계산"""
        summaries = defaultdict(dict)

        for scenario in self.scenarios:
            for method in Method:
                method_results = [
                    r for r in self.results
                    if r.scenario == scenario.name and r.method == method
                ]

                if not method_results:
                    continue

                steps = [r.steps for r in method_results]
                total_blocked = sum(r.blocked_count for r in method_results)
                total_actions = sum(r.total_actions for r in method_results)

                summaries[scenario.name][method] = ScenarioSummary(
                    scenario=scenario.name,
                    method=method,
                    total_episodes=len(method_results),
                    success_count=sum(1 for r in method_results if r.success),
                    violation_count=sum(1 for r in method_results if r.safety_violation),
                    avg_steps=np.mean(steps),
                    std_steps=np.std(steps),
                    total_blocked=total_blocked,
                    total_actions=total_actions,
                    avg_min_distance=np.mean([r.min_distance for r in method_results]),
                )

        return summaries

    def summarize_by_intensity(self) -> Dict[str, Dict[str, Dict[Method, ScenarioSummary]]]:
        """Intensity별 결과 요약"""
        summaries = defaultdict(lambda: defaultdict(dict))

        # Get all unique scenario/intensity combinations
        combinations = set((r.scenario, r.intensity) for r in self.results)

        for scenario_name, intensity in combinations:
            for method in Method:
                method_results = [
                    r for r in self.results
                    if r.scenario == scenario_name and r.intensity == intensity and r.method == method
                ]

                if not method_results:
                    continue

                steps = [r.steps for r in method_results]
                total_blocked = sum(r.blocked_count for r in method_results)
                total_actions = sum(r.total_actions for r in method_results)

                summaries[scenario_name][intensity][method] = ScenarioSummary(
                    scenario=f"{scenario_name}_{intensity}",
                    method=method,
                    total_episodes=len(method_results),
                    success_count=sum(1 for r in method_results if r.success),
                    violation_count=sum(1 for r in method_results if r.safety_violation),
                    avg_steps=np.mean(steps),
                    std_steps=np.std(steps),
                    total_blocked=total_blocked,
                    total_actions=total_actions,
                    avg_min_distance=np.mean([r.min_distance for r in method_results]),
                )

        return summaries

    def print_summary_table(self):
        """
        요약 테이블 출력

        평가 지표:
        - SR (Safety Rate): 금지구역 침입이 없는 비율 (높을수록 좋음)
        - VR (Violation Rate): 금지구역 침입 비율 (낮을수록 좋음)
        - BR (Block Rate): 명령 차단/수정 비율 (개입 적극성)
        - MD (Mean Min Distance): 평균 최소 거리 (높을수록 보수적)
        """
        summaries = self.summarize_results()

        print("\n" + "=" * 95)
        print("UNIFIED S1-S6 COMPARISON EXPERIMENT RESULTS")
        print("=" * 95)
        print("Metrics: SR=Safety Rate(↑), VR=Violation Rate(↓), BR=Block Rate, MD=Mean Min Distance(↑)")
        print("=" * 95)

        # Header
        header = f"{'Scenario':<8} {'Method':<12} {'SR%':>10} {'VR%':>10} {'BR%':>10} {'MD(m)':>10} {'Steps':>12}"
        print(header)
        print("-" * 95)

        for scenario_name in sorted(summaries.keys()):
            scenario_summaries = summaries[scenario_name]
            first = True

            for method in Method:
                if method not in scenario_summaries:
                    continue

                s = scenario_summaries[method]
                scenario_col = scenario_name if first else ""
                first = False

                print(f"{scenario_col:<8} {method.value:<12} {s.SR:>9.1f}% {s.VR:>9.1f}% "
                      f"{s.BR:>9.1f}% {s.MD:>9.2f} {s.avg_steps:>6.1f}±{s.std_steps:>4.1f}")

            print("-" * 95)

        # Overall summary by method
        print("\n" + "=" * 95)
        print("OVERALL SUMMARY BY METHOD")
        print("-" * 95)
        print(f"{'Method':<12} {'SR%':>12} {'VR%':>12} {'BR%':>12} {'MD(m)':>12}")
        print("-" * 95)

        for method in Method:
            method_results = [r for r in self.results if r.method == method]
            if not method_results:
                continue

            n = len(method_results)
            sr = sum(1 for r in method_results if not r.safety_violation) / n * 100
            vr = sum(1 for r in method_results if r.safety_violation) / n * 100
            total_blocked = sum(r.blocked_count for r in method_results)
            total_actions = sum(r.total_actions for r in method_results)
            br = total_blocked / total_actions * 100 if total_actions > 0 else 0
            md = np.mean([r.min_distance for r in method_results])

            print(f"{method.value:<12} {sr:>11.1f}% {vr:>11.1f}% {br:>11.1f}% {md:>11.2f}")

        print("=" * 95)

        # Intensity-level summary (highlight vulnerability exposures)
        self.print_intensity_summary()

    def print_intensity_summary(self):
        """Intensity별 VR 요약 (취약점 노출 확인)"""
        intensity_summaries = self.summarize_by_intensity()

        if not intensity_summaries:
            return

        print("\n" + "=" * 95)
        print("INTENSITY-LEVEL SUMMARY (VR% by Method)")
        print("Shows how different conditions expose method vulnerabilities")
        print("=" * 95)

        # Header with methods
        header = f"{'Scenario':<6} {'Intensity':<20}"
        for method in Method:
            header += f" {method.value[:8]:>10}"
        print(header)
        print("-" * 95)

        for scenario_name in sorted(intensity_summaries.keys()):
            intensities = intensity_summaries[scenario_name]
            first = True

            for intensity in sorted(intensities.keys()):
                method_results = intensities[intensity]
                scenario_col = scenario_name if first else ""
                first = False

                row = f"{scenario_col:<6} {intensity:<20}"
                for method in Method:
                    if method in method_results:
                        vr = method_results[method].VR
                        # Highlight high violation rates
                        if vr > 0:
                            row += f" {vr:>9.1f}%"
                        else:
                            row += f" {vr:>9.1f}%"
                    else:
                        row += f" {'---':>10}"
                print(row)

            print("-" * 95)

        # Key findings
        print("\n" + "=" * 95)
        print("KEY VULNERABILITY FINDINGS")
        print("-" * 95)

        # Find scenarios where SSM/CBF fail but Geofence succeeds
        for scenario_name, intensities in intensity_summaries.items():
            for intensity, methods in intensities.items():
                geofence_vr = methods.get(Method.GEOFENCE, ScenarioSummary(
                    scenario="", method=Method.GEOFENCE, total_episodes=0, success_count=0,
                    violation_count=0, avg_steps=0, std_steps=0, total_blocked=0,
                    total_actions=0, avg_min_distance=0
                )).VR

                for method in [Method.SAFETYCHIP, Method.SELP]:
                    if method in methods:
                        other_vr = methods[method].VR
                        if other_vr > 0 and geofence_vr == 0:
                            print(f"  {scenario_name}/{intensity}: {method.value} VR={other_vr:.1f}%, Geofence VR=0%")

        # Find scenarios where all methods fail (extreme conditions)
        print("\n  Extreme conditions (all methods challenged):")
        for scenario_name, intensities in intensity_summaries.items():
            for intensity, methods in intensities.items():
                if "extreme" in intensity.lower() or "latency" in intensity.lower():
                    geofence_result = methods.get(Method.GEOFENCE)
                    if geofence_result and geofence_result.VR > 0:
                        print(f"  {scenario_name}/{intensity}: Geofence VR={geofence_result.VR:.1f}% (extreme condition)")

        print("=" * 95)

    def save_results(self, output_path: str = "unified_s1_s6_results.csv"):
        """결과를 CSV로 저장"""
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'scenario', 'intensity', 'method', 'seed', 'success', 'safety_violation',
                'steps', 'blocked_count', 'total_actions', 'min_distance',
                'latency_ms', 'velocity', 'sigma_loc', 'termination_reason'
            ])

            for r in self.results:
                writer.writerow([
                    r.scenario, r.intensity, r.method.value, r.seed, r.success, r.safety_violation,
                    r.steps, r.blocked_count, r.total_actions, r.min_distance,
                    r.latency_ms, r.actual_velocity, r.sigma_loc, r.termination_reason
                ])

        print(f"\nResults saved to: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Unified S1-S6 Comparison Experiment")
    parser.add_argument('--scenarios', type=str, default='all',
                       help='Scenarios to run: all, S1, S2, S3, S4, S5, S6 (comma-separated)')
    parser.add_argument('--methods', type=str, default='all',
                       help='Methods: all, no_guard, safetychip, selp, geofence (comma-separated)')
    parser.add_argument('--intensities', type=str, default='all',
                       help='Intensities: all, baseline, low_velocity, extreme_latency, etc. (comma-separated)')
    parser.add_argument('--seeds', type=int, default=None,
                       help='Override number of seeds per scenario')
    parser.add_argument('--output', type=str, default='unified_s1_s6_results.csv',
                       help='Output CSV file')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')

    args = parser.parse_args()

    # Parse methods
    if args.methods == 'all':
        methods = list(Method)
    else:
        method_map = {m.value: m for m in Method}
        methods = [method_map[m.strip()] for m in args.methods.split(',')]

    # Parse intensities
    intensities = None
    if args.intensities != 'all':
        intensities = [i.strip() for i in args.intensities.split(',')]

    # Create runner
    runner = UnifiedExperimentRunner(verbose=args.verbose)

    # Override seeds if specified
    if args.seeds is not None:
        for scenario in runner.scenarios:
            scenario.num_seeds = args.seeds

    # Run experiments
    if args.scenarios == 'all':
        for scenario in runner.scenarios:
            runner.run_scenario(scenario, methods, intensities)
    else:
        scenario_names = [s.strip() for s in args.scenarios.split(',')]
        for scenario in runner.scenarios:
            if scenario.name in scenario_names:
                runner.run_scenario(scenario, methods, intensities)

    # Print and save results
    runner.print_summary_table()
    runner.save_results(args.output)


if __name__ == "__main__":
    main()
