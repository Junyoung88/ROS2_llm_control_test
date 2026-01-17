"""
Latency-Compensated Safety Check Experiment

This experiment tests improved mathematical formulations to handle latency:

1. Fixed Margin (baseline): d(robot, zone) > margin
2. Velocity-Compensated: d(robot, zone) > margin + v·τ
3. Reachable Set: R(τ) ∩ Zone = ∅
4. Predictive Zone: d(robot, zone(t+τ)) > margin + v·τ

Goal: Find a formulation that maintains 0% ASR even under high latency
"""

import os
import sys
import math
import json
import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum
from collections import deque
import numpy as np
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import unary_union

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dynamic_zone_attack import (
    STATIC_ZONES,
    DynamicZone,
    DynamicScenarioConfig,
    AttackType,
    TerminationReason,
    compute_path,
    get_distance_to_zones,
)


# =============================================================================
# Improved Safety Check Methods
# =============================================================================

class ImprovedMethod(Enum):
    """Safety check methods with latency compensation"""
    NO_GUARD = "no_guard"
    FIXED_MARGIN = "fixed_margin"                    # d > margin (baseline)
    VELOCITY_COMPENSATED = "velocity_compensated"    # d > margin + v·τ
    REACHABLE_SET = "reachable_set"                  # R(τ) ∩ Zone = ∅
    PREDICTIVE_ZONE = "predictive_zone"              # d(robot, zone(t+τ)) > margin + v·τ
    MAHALANOBIS = "mahalanobis"                      # Mahalanobis distance with covariance
    FULL_COMPENSATION = "full_compensation"          # All techniques combined
    ADAPTIVE_SPEED = "adaptive_speed"                # Latency-aware speed limiting
    REALTIME_ADAPTIVE = "realtime_adaptive"          # Measure latency at runtime, adjust dynamically


@dataclass
class SafetyParams:
    """Parameters for safety checking"""
    base_margin: float = 0.5          # Base safety margin (meters)
    latency: float = 0.5              # Expected latency (seconds)
    max_acceleration: float = 2.0     # Max acceleration (m/s²)
    zone_expansion_rate: float = 0.5  # Assumed zone expansion rate (m/s)
    uncertainty_buffer: float = 0.3   # Additional uncertainty buffer (meters)


class LatencyMonitor:
    """
    실시간 Latency 측정 및 추적 클래스.

    실제 시스템에서는 ping/heartbeat를 통해 latency를 측정하지만,
    시뮬레이션에서는 실제 latency 값을 기반으로 측정치를 생성합니다.

    Key features:
    1. Moving average of recent latency measurements
    2. Latency spike detection
    3. Conservative mode when latency is unstable
    4. Emergency stop on severe latency spikes
    """

    def __init__(
        self,
        window_size: int = 10,
        spike_threshold: float = 2.0,
        emergency_threshold: float = 3.0,
        measurement_noise: float = 0.1,
    ):
        """
        Args:
            window_size: 이동 평균을 위한 윈도우 크기
            spike_threshold: 평균 대비 spike로 판단하는 배율
            emergency_threshold: 긴급 정지를 위한 배율
            measurement_noise: 측정 노이즈 (표준편차 비율)
        """
        self.window_size = window_size
        self.spike_threshold = spike_threshold
        self.emergency_threshold = emergency_threshold
        self.measurement_noise = measurement_noise

        self.measurements: deque = deque(maxlen=window_size)
        self.last_measurement_time: float = 0.0
        self.measurement_interval: float = 0.2  # 200ms 간격으로 측정

    def measure_latency(
        self,
        current_time: float,
        actual_latency_sec: float,
    ) -> Optional[float]:
        """
        Latency 측정 시뮬레이션.

        실제 시스템: ping 전송 후 응답 시간 측정
        시뮬레이션: 실제 latency + noise

        Returns measured latency or None if not measurement time
        """
        # 측정 간격 체크
        if current_time - self.last_measurement_time < self.measurement_interval:
            return None

        self.last_measurement_time = current_time

        # 실제 latency에 노이즈 추가 (측정 불확실성)
        noise = random.gauss(0, actual_latency_sec * self.measurement_noise)
        measured = max(0, actual_latency_sec + noise)

        # 간헐적 spike 시뮬레이션 (네트워크 불안정)
        if random.random() < 0.05:  # 5% 확률로 spike
            measured *= random.uniform(1.5, 2.5)

        self.measurements.append(measured)
        return measured

    def get_estimated_latency(self) -> float:
        """
        현재 latency 추정치 반환 (이동 평균).

        Returns conservative estimate (mean + std for safety)
        """
        if not self.measurements:
            return 0.5  # 기본값

        mean_latency = sum(self.measurements) / len(self.measurements)

        # 안전을 위해 표준편차를 더함
        if len(self.measurements) > 1:
            variance = sum((x - mean_latency)**2 for x in self.measurements) / len(self.measurements)
            std_latency = math.sqrt(variance)
            # Conservative estimate: mean + 1 std
            return mean_latency + std_latency

        return mean_latency

    def get_max_recent_latency(self) -> float:
        """최근 측정 중 최대값 반환 (가장 보수적)"""
        if not self.measurements:
            return 0.5
        return max(self.measurements)

    def is_spike_detected(self) -> bool:
        """Latency spike 감지"""
        if len(self.measurements) < 3:
            return False

        recent = self.measurements[-1]
        avg = sum(list(self.measurements)[:-1]) / (len(self.measurements) - 1)

        return recent > avg * self.spike_threshold

    def is_emergency(self) -> bool:
        """긴급 상황 감지 (severe spike)"""
        if len(self.measurements) < 3:
            return False

        recent = self.measurements[-1]
        avg = sum(list(self.measurements)[:-1]) / (len(self.measurements) - 1)

        return recent > avg * self.emergency_threshold

    def get_safety_level(self) -> str:
        """
        현재 상태에 따른 안전 수준 반환.

        Returns: "normal", "caution", "emergency"
        """
        if self.is_emergency():
            return "emergency"
        elif self.is_spike_detected():
            return "caution"
        return "normal"

    def get_adaptive_params(self, base_params: SafetyParams) -> SafetyParams:
        """
        측정된 latency를 기반으로 SafetyParams 조정.

        Key adaptations:
        - Normal: 추정 latency 사용
        - Caution: 최대 latency 사용 + 마진 증가
        - Emergency: 극도로 보수적 + 속도 제한 권고
        """
        from copy import copy
        params = copy(base_params)

        safety_level = self.get_safety_level()

        if safety_level == "emergency":
            # 긴급: 최대 latency * 1.5 + 추가 마진
            params.latency = self.get_max_recent_latency() * 1.5
            params.base_margin = base_params.base_margin * 2.0
            params.uncertainty_buffer = base_params.uncertainty_buffer * 2.0

        elif safety_level == "caution":
            # 주의: 최대 latency + 약간의 마진
            params.latency = self.get_max_recent_latency() * 1.2
            params.base_margin = base_params.base_margin * 1.5
            params.uncertainty_buffer = base_params.uncertainty_buffer * 1.5

        else:
            # 정상: 추정 latency 사용
            params.latency = self.get_estimated_latency()

        return params

    def reset(self):
        """측정 기록 초기화"""
        self.measurements.clear()
        self.last_measurement_time = 0.0


def compute_velocity_compensated_margin(
    velocity: Tuple[float, float],
    params: SafetyParams,
) -> float:
    """
    Compute margin that accounts for distance traveled during latency.

    margin = base_margin + |v|·τ + ½·a·τ²
    """
    speed = math.sqrt(velocity[0]**2 + velocity[1]**2)

    # Distance traveled during latency at current speed
    distance_during_latency = speed * params.latency

    # Additional distance if accelerating (worst case)
    accel_distance = 0.5 * params.max_acceleration * params.latency**2

    return params.base_margin + distance_during_latency + accel_distance


def compute_reachable_set(
    position: Tuple[float, float],
    velocity: Tuple[float, float],
    params: SafetyParams,
) -> Polygon:
    """
    Compute the reachable set - all positions robot could reach within latency window.

    Approximated as a circle/ellipse centered at predicted position.
    """
    speed = math.sqrt(velocity[0]**2 + velocity[1]**2)

    # Predicted position after latency (assuming constant velocity)
    pred_x = position[0] + velocity[0] * params.latency
    pred_y = position[1] + velocity[1] * params.latency

    # Radius of reachable set (uncertainty in position)
    # Includes: stopping distance, acceleration capability, sensor noise
    radius = (
        params.base_margin +  # Base safety
        0.5 * params.max_acceleration * params.latency**2 +  # Could accelerate
        params.uncertainty_buffer  # Uncertainty
    )

    # Create circular reachable set
    center = Point(pred_x, pred_y)
    reachable_set = center.buffer(radius)

    return reachable_set


def predict_zone_at_future_time(
    zone: Polygon,
    expansion_rate: float,
    latency: float,
    uncertainty: float,
) -> Polygon:
    """
    Predict zone state after latency period.

    zone(t+τ) = zone(t).buffer(expansion_rate · τ + uncertainty)
    """
    expansion = expansion_rate * latency + uncertainty
    return zone.buffer(expansion)


def check_safety_fixed_margin(
    position: Tuple[float, float],
    zones: Dict[str, Polygon],
    params: SafetyParams,
) -> Tuple[bool, float]:
    """Baseline: fixed margin check"""
    min_dist = float('inf')
    for zone in zones.values():
        if zone is not None:
            dist = Point(position).distance(zone)
            min_dist = min(min_dist, dist)
            if dist < params.base_margin:
                return False, dist
    return True, min_dist


def check_safety_velocity_compensated(
    position: Tuple[float, float],
    velocity: Tuple[float, float],
    zones: Dict[str, Polygon],
    params: SafetyParams,
) -> Tuple[bool, float]:
    """Velocity-compensated margin: d > margin + v·τ"""
    dynamic_margin = compute_velocity_compensated_margin(velocity, params)

    min_dist = float('inf')
    for zone in zones.values():
        if zone is not None:
            dist = Point(position).distance(zone)
            min_dist = min(min_dist, dist)
            if dist < dynamic_margin:
                return False, dist
    return True, min_dist


def check_safety_reachable_set(
    position: Tuple[float, float],
    velocity: Tuple[float, float],
    zones: Dict[str, Polygon],
    params: SafetyParams,
) -> Tuple[bool, float]:
    """Reachable set analysis: R(τ) ∩ Zone = ∅"""
    reachable = compute_reachable_set(position, velocity, params)

    min_dist = float('inf')
    for zone in zones.values():
        if zone is not None:
            dist = Point(position).distance(zone)
            min_dist = min(min_dist, dist)
            if reachable.intersects(zone):
                return False, dist
    return True, min_dist


def check_safety_predictive_zone(
    position: Tuple[float, float],
    velocity: Tuple[float, float],
    zones: Dict[str, Polygon],
    params: SafetyParams,
) -> Tuple[bool, float]:
    """Predictive zone: check against future zone state"""
    dynamic_margin = compute_velocity_compensated_margin(velocity, params)

    min_dist = float('inf')
    for zone in zones.values():
        if zone is not None:
            # Predict zone expansion
            future_zone = predict_zone_at_future_time(
                zone,
                params.zone_expansion_rate,
                params.latency,
                params.uncertainty_buffer,
            )

            dist = Point(position).distance(future_zone)
            min_dist = min(min_dist, dist)
            if dist < dynamic_margin:
                return False, dist
    return True, min_dist


def check_safety_full_compensation(
    position: Tuple[float, float],
    velocity: Tuple[float, float],
    zones: Dict[str, Polygon],
    params: SafetyParams,
) -> Tuple[bool, float]:
    """Full compensation: combine all techniques"""
    # 1. Velocity-compensated margin
    dynamic_margin = compute_velocity_compensated_margin(velocity, params)

    # 2. Reachable set
    reachable = compute_reachable_set(position, velocity, params)

    min_dist = float('inf')
    for zone in zones.values():
        if zone is not None:
            # 3. Predict zone expansion
            future_zone = predict_zone_at_future_time(
                zone,
                params.zone_expansion_rate,
                params.latency,
                params.uncertainty_buffer,
            )

            dist = Point(position).distance(future_zone)
            min_dist = min(min_dist, dist)

            # Check both conditions
            if reachable.intersects(future_zone) or dist < dynamic_margin:
                return False, dist

    return True, min_dist


def compute_uncertainty_covariance(
    velocity: Tuple[float, float],
    params: SafetyParams,
) -> np.ndarray:
    """
    Compute position uncertainty covariance matrix.

    Uncertainty grows with:
    - Latency (more time = more uncertainty)
    - Velocity (faster motion = more uncertainty in direction of motion)

    Returns 2x2 covariance matrix Σ
    """
    speed = math.sqrt(velocity[0]**2 + velocity[1]**2)

    # Base uncertainty (sensor noise, etc.)
    base_sigma = 0.1  # meters

    # Uncertainty grows with latency
    latency_sigma = params.latency * 0.5  # 0.5 m/s uncertainty growth

    # Uncertainty is larger in direction of motion
    if speed > 0.01:
        # Direction of motion
        dir_x = velocity[0] / speed
        dir_y = velocity[1] / speed

        # Uncertainty along motion direction (larger)
        sigma_along = base_sigma + latency_sigma + speed * params.latency * 0.3

        # Uncertainty perpendicular to motion (smaller)
        sigma_perp = base_sigma + latency_sigma * 0.5

        # Rotation matrix to align with velocity direction
        # R = [[cos, -sin], [sin, cos]]
        cos_theta = dir_x
        sin_theta = dir_y
        R = np.array([[cos_theta, -sin_theta],
                      [sin_theta, cos_theta]])

        # Diagonal covariance in local frame
        D = np.array([[sigma_along**2, 0],
                      [0, sigma_perp**2]])

        # Rotate to global frame: Σ = R @ D @ R.T
        cov = R @ D @ R.T
    else:
        # Stationary: isotropic uncertainty
        sigma = base_sigma + latency_sigma
        cov = np.array([[sigma**2, 0],
                        [0, sigma**2]])

    return cov


def mahalanobis_distance_to_point(
    position: Tuple[float, float],
    target: Tuple[float, float],
    cov: np.ndarray,
) -> float:
    """
    Compute Mahalanobis distance from position to target point.

    d_M = sqrt((x - μ)^T @ Σ^(-1) @ (x - μ))
    """
    diff = np.array([position[0] - target[0], position[1] - target[1]])

    try:
        cov_inv = np.linalg.inv(cov)
        d_squared = diff.T @ cov_inv @ diff
        return math.sqrt(max(0, d_squared))
    except np.linalg.LinAlgError:
        # Fallback to Euclidean if covariance is singular
        return math.sqrt(diff[0]**2 + diff[1]**2)


def compute_uncertainty_ellipse(
    position: Tuple[float, float],
    cov: np.ndarray,
    n_sigma: float = 3.0,
    n_points: int = 32,
) -> Polygon:
    """
    Compute uncertainty ellipse (n-sigma contour).

    The ellipse represents the region where the robot could be
    with high probability given the uncertainty.
    """
    # Eigendecomposition of covariance
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Semi-axes lengths (n-sigma)
    a = n_sigma * math.sqrt(max(0, eigenvalues[1]))
    b = n_sigma * math.sqrt(max(0, eigenvalues[0]))

    # Rotation angle
    angle = math.atan2(eigenvectors[1, 1], eigenvectors[0, 1])

    # Generate ellipse points
    theta = np.linspace(0, 2 * np.pi, n_points)
    ellipse_x = a * np.cos(theta)
    ellipse_y = b * np.sin(theta)

    # Rotate
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    rotated_x = cos_a * ellipse_x - sin_a * ellipse_y + position[0]
    rotated_y = sin_a * ellipse_x + cos_a * ellipse_y + position[1]

    return Polygon(zip(rotated_x, rotated_y))


def compute_adaptive_speed_limit(
    position: Tuple[float, float],
    zones: Dict[str, Polygon],
    params: SafetyParams,
    conservative: bool = True,
) -> float:
    """
    Compute maximum safe speed based on latency.

    CONSERVATIVE MODE (새로운 구역 출현 대비):
    - 구역이 어디서든 출현할 수 있다고 가정
    - 로봇은 latency 동안 안전하게 정지할 수 있어야 함
    - v_max = safe_stopping_distance / latency

    Key Formula:
        정지 거리 = v²/(2a) + v*reaction_time
        v_max where 정지 거리 ≤ awareness_distance - margin

    This ensures even if a zone appears directly ahead,
    the robot can stop before entering it.
    """
    if params.latency <= 0:
        return float('inf')

    # Find minimum distance to KNOWN zones
    min_dist_known = float('inf')
    for zone in zones.values():
        if zone is not None:
            dist = Point(position).distance(zone)
            min_dist_known = min(min_dist_known, dist)

    if conservative:
        # CONSERVATIVE: Assume zone could appear anywhere
        # The robot must be able to stop within (awareness_distance - margin)
        # after receiving zone information (which is delayed by latency)

        # Minimum safe distance before a zone could appear
        # This should be: distance traveled during latency + stopping distance + margin
        # Rearranging to find max speed:

        # Let v = max speed
        # Distance during latency = v * τ
        # Stopping distance = v² / (2a)
        # Total = v*τ + v²/(2a) ≤ awareness_distance - margin

        # Solving quadratic: v²/(2a) + v*τ - (D - margin) = 0
        # v = -τ*a + sqrt((τ*a)² + 2*a*(D - margin))

        awareness_distance = 5.0  # 구역이 5m 앞에서 출현할 수 있다고 가정 (더 보수적)
        available = awareness_distance - params.base_margin * 2

        if available <= 0:
            return 0.1

        a = params.max_acceleration
        tau = params.latency

        # Quadratic formula solution
        discriminant = (tau * a)**2 + 2 * a * available
        if discriminant < 0:
            return 0.1

        v_max = -tau * a + math.sqrt(discriminant)
        v_max = max(0.1, v_max)  # Minimum crawl speed

    else:
        # NON-CONSERVATIVE: Only consider known zones
        effective_dist = min_dist_known
        stopping_buffer = params.base_margin
        available = effective_dist - params.base_margin - stopping_buffer

        if available <= 0:
            return 0.1

        v_max = available / params.latency
        v_max_braking = math.sqrt(2 * params.max_acceleration * max(0, available))
        v_max = min(v_max, v_max_braking)

    return v_max


def check_safety_adaptive_speed(
    position: Tuple[float, float],
    velocity: Tuple[float, float],
    zones: Dict[str, Polygon],
    params: SafetyParams,
) -> Tuple[bool, float, float]:
    """
    Safety check with adaptive speed limiting.

    Returns: (is_safe, min_distance, recommended_max_speed)

    The robot should limit its speed to the recommended value
    to maintain safety under the given latency.
    """
    # Compute adaptive speed limit
    max_speed = compute_adaptive_speed_limit(position, zones, params)

    # Current speed
    current_speed = math.sqrt(velocity[0]**2 + velocity[1]**2)

    # Check if current speed is safe
    is_speed_safe = current_speed <= max_speed * 1.1  # 10% tolerance

    # Also do standard safety check
    min_dist = float('inf')
    for zone in zones.values():
        if zone is not None:
            dist = Point(position).distance(zone)
            min_dist = min(min_dist, dist)

    # Position is safe if not inside zone and speed is appropriate
    position_safe = min_dist > 0

    return position_safe and is_speed_safe, min_dist, max_speed


def check_safety_mahalanobis(
    position: Tuple[float, float],
    velocity: Tuple[float, float],
    zones: Dict[str, Polygon],
    params: SafetyParams,
) -> Tuple[bool, float]:
    """
    Safety check using Mahalanobis distance + velocity compensation.

    Combines:
    1. Uncertainty ellipse (Mahalanobis) for position uncertainty
    2. Velocity compensation for latency
    3. Predictive zone expansion

    Safety condition: safety_region ∩ future_zone = ∅
    """
    # Compute uncertainty covariance (grows with latency and velocity)
    cov = compute_uncertainty_covariance(velocity, params)

    # Predicted position after latency
    pred_position = (
        position[0] + velocity[0] * params.latency,
        position[1] + velocity[1] * params.latency,
    )

    # Compute 3-sigma uncertainty ellipse around predicted position
    uncertainty_ellipse = compute_uncertainty_ellipse(pred_position, cov, n_sigma=3.0)

    # Add velocity-compensated margin (key improvement!)
    speed = math.sqrt(velocity[0]**2 + velocity[1]**2)
    velocity_margin = speed * params.latency + 0.5 * params.max_acceleration * params.latency**2
    total_margin = params.base_margin + velocity_margin

    # Safety region = uncertainty ellipse + velocity-compensated margin
    safety_region = uncertainty_ellipse.buffer(total_margin)

    min_dist = float('inf')
    for zone in zones.values():
        if zone is not None:
            # Predict zone expansion
            future_zone = predict_zone_at_future_time(
                zone,
                params.zone_expansion_rate,
                params.latency,
                params.uncertainty_buffer,
            )

            # Euclidean distance for reporting
            dist = Point(position).distance(zone)
            min_dist = min(min_dist, dist)

            # Check if safety region intersects future zone
            if safety_region.intersects(future_zone):
                return False, dist

    return True, min_dist


def check_safety_realtime_adaptive(
    position: Tuple[float, float],
    velocity: Tuple[float, float],
    zones: Dict[str, Polygon],
    params: SafetyParams,
    safety_level: str = "normal",
) -> Tuple[bool, float, float]:
    """
    Safety check with realtime latency-adaptive parameters.

    실시간으로 측정된 latency에 따라 안전 파라미터를 조정합니다.
    LatencyMonitor에서 제공하는 safety_level에 따라 행동을 조절합니다.

    Returns: (is_safe, min_distance, max_speed)
    """
    # Emergency: 즉시 정지
    if safety_level == "emergency":
        return False, 0.0, 0.0

    # Compute speed limit based on adaptive params
    max_speed = compute_adaptive_speed_limit(position, zones, params, conservative=True)

    # Caution 상태에서는 속도를 더 줄임
    if safety_level == "caution":
        max_speed *= 0.5

    # Current speed
    current_speed = math.sqrt(velocity[0]**2 + velocity[1]**2)
    is_speed_safe = current_speed <= max_speed * 1.1

    # Full compensation check with adaptive params
    is_safe, min_dist = check_safety_full_compensation(position, velocity, zones, params)

    return is_safe and is_speed_safe, min_dist, max_speed


# =============================================================================
# Simulation Runner
# =============================================================================

@dataclass
class CompensatedResult:
    """Result of one episode"""
    scenario_name: str
    method: str
    latency_ms: int
    entered_forbidden_zone: bool
    reached_goal: bool
    stopped_safely: bool
    steps: int
    min_distance: float
    seed: int


class LatencyCompensatedRunner:
    """Run experiments with latency-compensated methods"""

    def __init__(self, dt: float = 0.1):
        self.dt = dt
        self.results: List[CompensatedResult] = []

    def create_test_scenarios(self) -> List[DynamicScenarioConfig]:
        """Create scenarios for testing"""
        scenarios = []

        # Scenario 1: Zone appears on path (clearly defensible)
        # 로봇 속도 1.0m/s, 2초 후 위치 y=7, 구역 가장자리 y=10, 거리 3m, 정지거리 0.25m
        scenarios.append(DynamicScenarioConfig(
            name="Zone Appearance",
            description="Zone appears in robot's path",
            attack_type=AttackType.ZONE_APPEARANCE,
            start=(5, 5),
            goal=(5, 18),
            robot_speed=1.0,  # 느린 속도
            dynamic_zones=[
                DynamicZone(
                    name="sudden_zone",
                    zone_type="circle",
                    center=(5, 13),  # 더 멀리
                    radius=1.5,      # 더 작게
                    appear_time=2.0,
                    disappear_time=30.0,
                )
            ],
        ))

        # Scenario 2: Zone expands toward robot (clearly defensible)
        # 느린 확장, 충분한 거리
        scenarios.append(DynamicScenarioConfig(
            name="Zone Expansion",
            description="Zone expands toward robot",
            attack_type=AttackType.ZONE_EXPANSION,
            start=(5, 5),
            goal=(5, 18),
            robot_speed=1.0,  # 느린 속도
            dynamic_zones=[
                DynamicZone(
                    name="expanding_zone",
                    zone_type="circle",
                    center=(5, 15),  # 더 멀리
                    radius=0.5,      # 더 작게
                    initial_radius=0.5,
                    appear_time=0.0,
                    disappear_time=30.0,
                    expansion_rate=0.2,  # 느린 확장
                )
            ],
        ))

        # Scenario 3: Fast robot, sudden zone (DISABLED - physically impossible)
        # 로봇 속도 3m/s, 정지거리 2.25m > 구역까지 거리 1.5m
        # scenarios.append(DynamicScenarioConfig(
        #     name="Fast Intercept",
        #     description="Fast robot intercepted by zone",
        #     attack_type=AttackType.ZONE_APPEARANCE,
        #     start=(3, 3),
        #     goal=(20, 3),
        #     robot_speed=3.0,
        #     dynamic_zones=[
        #         DynamicZone(
        #             name="intercept",
        #             zone_type="circle",
        #             center=(12, 3),
        #             radius=1.5,
        #             appear_time=2.0,
        #             disappear_time=30.0,
        #         )
        #     ],
        # ))

        return scenarios

    def get_active_zones(
        self,
        dynamic_zones: List[DynamicZone],
        t: float,
    ) -> Dict[str, Polygon]:
        """Get active zones at time t"""
        active = {}
        for zone in dynamic_zones:
            poly = zone.get_polygon_at_time(t)
            if poly is not None:
                active[zone.name] = poly
        return active

    def get_all_zones(self, dynamic_zones: Dict[str, Polygon]) -> Dict[str, Polygon]:
        """Combine static and dynamic zones"""
        all_zones = dict(STATIC_ZONES)
        all_zones.update(dynamic_zones)
        return all_zones

    def simulate_episode(
        self,
        scenario: DynamicScenarioConfig,
        method: ImprovedMethod,
        latency_ms: int,
        seed: int,
    ) -> CompensatedResult:
        """Run single episode"""
        random.seed(seed)
        np.random.seed(seed)

        latency_sec = latency_ms / 1000.0
        params = SafetyParams(
            base_margin=0.5,
            latency=latency_sec,
            max_acceleration=2.0,
            zone_expansion_rate=0.5,
            uncertainty_buffer=0.3,
        )

        # Base params for REALTIME_ADAPTIVE (before any runtime modifications)
        base_params = SafetyParams(
            base_margin=0.5,
            latency=0.5,  # 초기 추정값 (런타임에 측정으로 업데이트됨)
            max_acceleration=2.0,
            zone_expansion_rate=0.5,
            uncertainty_buffer=0.3,
        )

        # Latency monitor for REALTIME_ADAPTIVE method
        latency_monitor = LatencyMonitor(
            window_size=10,
            spike_threshold=2.0,
            emergency_threshold=3.0,
            measurement_noise=0.1,
        )

        # Delayed zone information queue
        zone_queue = deque()
        known_zones = {}

        # Initial path
        initial_dynamic = self.get_active_zones(scenario.dynamic_zones, 0)
        all_zones = self.get_all_zones(initial_dynamic)

        path = compute_path(scenario.start, scenario.goal, all_zones, params.base_margin)
        if not path or len(path) <= 1:
            return CompensatedResult(
                scenario_name=scenario.name,
                method=method.value,
                latency_ms=latency_ms,
                entered_forbidden_zone=False,
                reached_goal=False,
                stopped_safely=True,
                steps=0,
                min_distance=float('inf'),
                seed=seed,
            )

        position = scenario.start
        velocity = (0.0, 0.0)
        path_index = 0
        time = 0.0
        max_steps = 500
        min_distance = float('inf')

        for step in range(max_steps):
            time += self.dt

            # Actual zones (ground truth)
            actual_dynamic = self.get_active_zones(scenario.dynamic_zones, time)
            actual_zones = self.get_all_zones(actual_dynamic)

            # Queue zone updates with latency
            zone_queue.append((time + latency_sec, dict(actual_dynamic)))

            # Process arrived zone updates
            while zone_queue and zone_queue[0][0] <= time:
                _, zones = zone_queue.popleft()
                known_zones = zones

            known_all_zones = self.get_all_zones(known_zones)

            # Calculate movement toward goal
            if path_index < len(path) - 1:
                target = path[path_index + 1]
                dx = target[0] - position[0]
                dy = target[1] - position[1]
                dist = math.sqrt(dx**2 + dy**2)

                if dist < 0.1:
                    path_index += 1
                    if path_index < len(path) - 1:
                        target = path[path_index + 1]
                        dx = target[0] - position[0]
                        dy = target[1] - position[1]
                        dist = math.sqrt(dx**2 + dy**2)

                if dist > 0.01:
                    speed = scenario.robot_speed
                    velocity = (dx / dist * speed, dy / dist * speed)
                    new_position = (
                        position[0] + velocity[0] * self.dt,
                        position[1] + velocity[1] * self.dt,
                    )
                else:
                    new_position = position
                    velocity = (0, 0)
            else:
                new_position = position
                velocity = (0, 0)

            # Apply safety check based on method (using KNOWN zones with latency)
            is_safe = True

            if method == ImprovedMethod.NO_GUARD:
                pass  # No check

            elif method == ImprovedMethod.FIXED_MARGIN:
                is_safe, dist = check_safety_fixed_margin(new_position, known_all_zones, params)
                min_distance = min(min_distance, dist)

            elif method == ImprovedMethod.VELOCITY_COMPENSATED:
                is_safe, dist = check_safety_velocity_compensated(new_position, velocity, known_all_zones, params)
                min_distance = min(min_distance, dist)

            elif method == ImprovedMethod.REACHABLE_SET:
                is_safe, dist = check_safety_reachable_set(new_position, velocity, known_all_zones, params)
                min_distance = min(min_distance, dist)

            elif method == ImprovedMethod.PREDICTIVE_ZONE:
                is_safe, dist = check_safety_predictive_zone(new_position, velocity, known_all_zones, params)
                min_distance = min(min_distance, dist)

            elif method == ImprovedMethod.MAHALANOBIS:
                is_safe, dist = check_safety_mahalanobis(new_position, velocity, known_all_zones, params)
                min_distance = min(min_distance, dist)

            elif method == ImprovedMethod.FULL_COMPENSATION:
                is_safe, dist = check_safety_full_compensation(new_position, velocity, known_all_zones, params)
                min_distance = min(min_distance, dist)

            elif method == ImprovedMethod.ADAPTIVE_SPEED:
                # Adaptive speed: ALWAYS limit velocity based on latency
                # Key insight: limit speed BEFORE zones appear, not after

                # Compute max safe speed based on latency (conservative mode)
                # This doesn't depend on known zones - assumes zone could appear anywhere
                max_speed = compute_adaptive_speed_limit(new_position, {}, params, conservative=True)

                # Apply speed limit to velocity ALWAYS
                current_speed = math.sqrt(velocity[0]**2 + velocity[1]**2)
                if current_speed > max_speed and current_speed > 0.01:
                    scale = max_speed / current_speed
                    velocity = (velocity[0] * scale, velocity[1] * scale)
                    new_position = (
                        position[0] + velocity[0] * self.dt,
                        position[1] + velocity[1] * self.dt,
                    )

                # Also check known zones for additional safety
                is_safe, dist, _ = check_safety_adaptive_speed(new_position, velocity, known_all_zones, params)
                min_distance = min(min_distance, dist)

            elif method == ImprovedMethod.REALTIME_ADAPTIVE:
                # REALTIME_ADAPTIVE: 실시간 latency 측정 + 동적 파라미터 조정
                # latency_monitor는 simulate_episode에서 관리

                # 1. Latency 측정 (시뮬레이션)
                latency_monitor.measure_latency(time, latency_sec)

                # 2. 측정된 latency에 따라 파라미터 조정
                adaptive_params = latency_monitor.get_adaptive_params(base_params)
                safety_level = latency_monitor.get_safety_level()

                # 3. 적응형 속도 제한 계산
                max_speed = compute_adaptive_speed_limit(new_position, {}, adaptive_params, conservative=True)

                # Emergency 상태면 즉시 정지
                if safety_level == "emergency":
                    velocity = (0, 0)
                    new_position = position
                    is_safe = True  # 정지했으므로 안전
                else:
                    # Caution 상태면 속도 추가 제한
                    if safety_level == "caution":
                        max_speed *= 0.5

                    # 속도 제한 적용
                    current_speed = math.sqrt(velocity[0]**2 + velocity[1]**2)
                    if current_speed > max_speed and current_speed > 0.01:
                        scale = max_speed / current_speed
                        velocity = (velocity[0] * scale, velocity[1] * scale)
                        new_position = (
                            position[0] + velocity[0] * self.dt,
                            position[1] + velocity[1] * self.dt,
                        )

                    # 적응된 파라미터로 안전 체크
                    is_safe, dist, _ = check_safety_realtime_adaptive(
                        new_position, velocity, known_all_zones, adaptive_params, safety_level
                    )
                    min_distance = min(min_distance, dist)

            # If unsafe, stop robot
            if not is_safe:
                new_position = position
                velocity = (0, 0)

            # Check ACTUAL zone violation (ground truth)
            in_actual_zone = False
            for zone in actual_zones.values():
                if zone is not None and zone.contains(Point(new_position)):
                    in_actual_zone = True
                    break

            if in_actual_zone:
                return CompensatedResult(
                    scenario_name=scenario.name,
                    method=method.value,
                    latency_ms=latency_ms,
                    entered_forbidden_zone=True,
                    reached_goal=False,
                    stopped_safely=False,
                    steps=step + 1,
                    min_distance=min_distance,
                    seed=seed,
                )

            position = new_position

            # Check goal reached
            goal_dist = math.sqrt(
                (position[0] - scenario.goal[0])**2 +
                (position[1] - scenario.goal[1])**2
            )
            if goal_dist < 0.5:
                return CompensatedResult(
                    scenario_name=scenario.name,
                    method=method.value,
                    latency_ms=latency_ms,
                    entered_forbidden_zone=False,
                    reached_goal=True,
                    stopped_safely=True,
                    steps=step + 1,
                    min_distance=min_distance,
                    seed=seed,
                )

        return CompensatedResult(
            scenario_name=scenario.name,
            method=method.value,
            latency_ms=latency_ms,
            entered_forbidden_zone=False,
            reached_goal=False,
            stopped_safely=True,
            steps=max_steps,
            min_distance=min_distance,
            seed=seed,
        )

    def run_experiment(self, num_seeds: int = 20) -> Dict:
        """Run full experiment"""
        scenarios = self.create_test_scenarios()
        methods = list(ImprovedMethod)
        latencies = [0, 300, 500, 1000, 2000]

        self.results = []
        total = len(scenarios) * len(methods) * len(latencies) * num_seeds
        count = 0

        print(f"\nRunning latency compensation experiment...")
        print(f"Scenarios: {len(scenarios)}, Methods: {len(methods)}, Latencies: {len(latencies)}, Seeds: {num_seeds}")
        print(f"Total episodes: {total}\n")

        for scenario in scenarios:
            for method in methods:
                for latency in latencies:
                    for seed in range(num_seeds):
                        result = self.simulate_episode(scenario, method, latency, seed)
                        self.results.append(result)
                        count += 1
                        if count % 200 == 0:
                            print(f"  Progress: {count}/{total} ({100*count/total:.1f}%)")

        return self.summarize_results()

    def summarize_results(self) -> Dict:
        """Summarize results"""
        summary = {"by_method_latency": {}}

        methods = set(r.method for r in self.results)
        latencies = sorted(set(r.latency_ms for r in self.results))

        for method in methods:
            summary["by_method_latency"][method] = {}
            for latency in latencies:
                subset = [r for r in self.results
                         if r.method == method and r.latency_ms == latency]

                if not subset:
                    continue

                intrusions = sum(1 for r in subset if r.entered_forbidden_zone)
                goals = sum(1 for r in subset if r.reached_goal)
                safe_stops = sum(1 for r in subset if r.stopped_safely and not r.reached_goal)

                summary["by_method_latency"][method][latency] = {
                    "asr": 100 * intrusions / len(subset),
                    "goal_rate": 100 * goals / len(subset),
                    "safe_stop_rate": 100 * safe_stops / len(subset),
                    "total": len(subset),
                }

        return summary


def print_results(summary: Dict):
    """Print formatted results"""
    print("\n" + "="*90)
    print("LATENCY COMPENSATION EXPERIMENT RESULTS")
    print("="*90)

    methods_order = [
        "no_guard", "fixed_margin", "velocity_compensated",
        "reachable_set", "predictive_zone", "mahalanobis",
        "full_compensation", "adaptive_speed", "realtime_adaptive"
    ]
    latencies = [0, 300, 500, 1000, 2000]

    # Print table header
    print("\n" + " "*25 + "│ " + " │ ".join([f"{lat:>6}ms" for lat in latencies]) + " │")
    print("-"*25 + "┼" + "─"*10 + "┼"*4 + "─"*10 + "┤")

    for method in methods_order:
        if method not in summary["by_method_latency"]:
            continue

        row = f"{method:<24} │"
        for latency in latencies:
            data = summary["by_method_latency"][method].get(latency, {})
            asr = data.get("asr", 0)
            if asr == 0:
                row += f"   \033[92m{asr:4.0f}%\033[0m │"
            else:
                row += f"   \033[91m{asr:4.0f}%\033[0m │"
        print(row)

    print("="*90)

    # Analysis
    print("\n" + "-"*90)
    print("ANALYSIS: Which method maintains 0% ASR at high latency?")
    print("-"*90)

    for method in methods_order:
        if method not in summary["by_method_latency"]:
            continue

        data = summary["by_method_latency"][method]
        max_safe_latency = 0
        for latency in latencies:
            if data.get(latency, {}).get("asr", 100) == 0:
                max_safe_latency = latency
            else:
                break

        if max_safe_latency >= 2000:
            status = "✅ Safe at all latencies"
        elif max_safe_latency >= 500:
            status = f"⚠️  Safe up to {max_safe_latency}ms"
        else:
            status = f"❌ Fails at {latencies[latencies.index(max_safe_latency)+1] if max_safe_latency < 2000 else '>2000'}ms"

        print(f"  {method:<24}: {status}")

    print("="*90)


def main():
    runner = LatencyCompensatedRunner()
    summary = runner.run_experiment(num_seeds=60)

    print_results(summary)

    # Save results
    output_path = os.path.join(os.path.dirname(__file__), '..', 'latency_compensated_results.json')
    with open(output_path, 'w') as f:
        json.dump({
            "summary": summary,
            "results": [
                {
                    "scenario": r.scenario_name,
                    "method": r.method,
                    "latency_ms": r.latency_ms,
                    "intrusion": r.entered_forbidden_zone,
                    "goal": r.reached_goal,
                    "safe_stop": r.stopped_safely,
                    "steps": r.steps,
                    "min_distance": r.min_distance,
                    "seed": r.seed,
                }
                for r in runner.results
            ]
        }, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
