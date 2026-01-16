#!/usr/bin/env python3
"""
Sensor Integrity Monitor - 센서 스푸핑 공격 방어

센서 데이터의 무결성을 검증하여 스푸핑 공격을 탐지합니다.

공격 유형:
1. AMCL 공분산 조작: 불확실성을 낮게 보고 → 안전 마진 축소
2. Odometry 스푸핑: 속도 과소보고 → v_max·τ 항 축소
3. LiDAR/카메라 스푸핑: 위치 추정 오염

방어 전략:
1. 최소 공분산 경계 (Covariance Floor)
2. 시간적 일관성 검사 (Temporal Consistency)
3. 센서 융합 교차 검증 (Cross-Validation)
4. 물리적 타당성 검사 (Physical Plausibility)
5. 이상 탐지 (Anomaly Detection)

핵심 원칙:
    "센서가 비정상적으로 '좋은' 데이터를 보고하면 의심하라"
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Deque
from collections import deque
from enum import IntEnum
import time


class AlertLevel(IntEnum):
    """경고 수준"""
    NORMAL = 0
    WARNING = 1    # 의심스러운 데이터
    CRITICAL = 2   # 명백한 스푸핑 시도
    EMERGENCY = 3  # 즉시 조치 필요


@dataclass
class SensorReading:
    """센서 데이터 구조"""
    timestamp: float
    source: str
    position: Optional[Tuple[float, float]] = None
    velocity: Optional[Tuple[float, float]] = None
    covariance: Optional[np.ndarray] = None
    raw_data: Optional[dict] = None


@dataclass
class IntegrityReport:
    """무결성 검사 결과"""
    timestamp: float
    sensor_source: str
    is_valid: bool
    alert_level: AlertLevel
    violations: List[str] = field(default_factory=list)
    original_covariance: Optional[np.ndarray] = None
    corrected_covariance: Optional[np.ndarray] = None
    confidence_score: float = 1.0  # 0.0 ~ 1.0


class CovarianceValidator:
    """
    공분산 검증기 - 최소 경계 강제

    공격 시나리오:
        공격자가 AMCL 공분산을 [0.001, 0, 0, 0.001]로 조작
        → 시스템이 "위치 매우 정확"으로 판단
        → 안전 마진이 λ_max 기반으로 축소됨

    방어:
        물리적으로 불가능한 낮은 공분산은 거부하고
        최소 경계(floor)를 강제 적용
    """

    # 물리적 한계 기반 최소 공분산
    # 일반적인 AMCL + LiDAR 시스템의 최소 불확실성
    MIN_POSITION_VARIANCE = 0.01  # 10cm (제곱) - 최고 수준 센서도 이 이하 불가
    MIN_ORIENTATION_VARIANCE = 0.001  # ~1.8도 (제곱)

    # 비정상적으로 낮은 공분산 임계값 (스푸핑 의심)
    SUSPICIOUS_VARIANCE_THRESHOLD = 0.005  # 5mm (제곱) - 실제로 불가능

    def __init__(
        self,
        min_position_var: float = MIN_POSITION_VARIANCE,
        min_orientation_var: float = MIN_ORIENTATION_VARIANCE,
        suspicious_threshold: float = SUSPICIOUS_VARIANCE_THRESHOLD
    ):
        self._min_pos_var = min_position_var
        self._min_orient_var = min_orientation_var
        self._suspicious_threshold = suspicious_threshold

        # 통계
        self._total_checks = 0
        self._suspicious_count = 0
        self._forced_floor_count = 0

    def validate(self, covariance: np.ndarray) -> Tuple[bool, np.ndarray, List[str]]:
        """
        공분산 검증 및 교정

        Args:
            covariance: 6x6 또는 36-element 공분산 (x, y, z, roll, pitch, yaw)

        Returns:
            (is_valid, corrected_covariance, violations)
        """
        self._total_checks += 1
        violations = []

        # 공분산 행렬 정규화
        if covariance.size == 36:
            cov_matrix = covariance.reshape(6, 6)
        elif covariance.shape == (6, 6):
            cov_matrix = covariance.copy()
        else:
            violations.append(f"Invalid covariance shape: {covariance.shape}")
            return False, self._get_safe_covariance(), violations

        # 위치 공분산 추출 (x, y)
        pos_var_x = cov_matrix[0, 0]
        pos_var_y = cov_matrix[1, 1]
        orient_var = cov_matrix[5, 5]

        # 1. 비정상적으로 낮은 값 탐지 (스푸핑 의심)
        if pos_var_x < self._suspicious_threshold or pos_var_y < self._suspicious_threshold:
            self._suspicious_count += 1
            violations.append(
                f"SUSPICIOUS: Position variance too low "
                f"(x={pos_var_x:.6f}, y={pos_var_y:.6f}) - possible spoofing"
            )

        # 2. 최소 경계 강제 적용 (Covariance Floor)
        corrected = cov_matrix.copy()
        floor_applied = False

        if pos_var_x < self._min_pos_var:
            corrected[0, 0] = self._min_pos_var
            floor_applied = True
            violations.append(f"Floor applied to x variance: {pos_var_x:.6f} → {self._min_pos_var}")

        if pos_var_y < self._min_pos_var:
            corrected[1, 1] = self._min_pos_var
            floor_applied = True
            violations.append(f"Floor applied to y variance: {pos_var_y:.6f} → {self._min_pos_var}")

        if orient_var < self._min_orient_var:
            corrected[5, 5] = self._min_orient_var
            floor_applied = True
            violations.append(f"Floor applied to orientation variance: {orient_var:.6f} → {self._min_orient_var}")

        if floor_applied:
            self._forced_floor_count += 1

        # 3. 양정치 행렬 확인
        try:
            eigenvalues = np.linalg.eigvalsh(corrected[:2, :2])
            if np.any(eigenvalues < 0):
                violations.append("Non-positive-definite covariance matrix")
                return False, self._get_safe_covariance(), violations
        except np.linalg.LinAlgError:
            violations.append("Covariance matrix decomposition failed")
            return False, self._get_safe_covariance(), violations

        is_valid = len([v for v in violations if "SUSPICIOUS" in v]) == 0
        return is_valid, corrected.flatten(), violations

    def _get_safe_covariance(self) -> np.ndarray:
        """안전한 기본 공분산 반환"""
        safe_cov = np.eye(6) * 0.1  # 보수적인 값
        safe_cov[0, 0] = self._min_pos_var * 10
        safe_cov[1, 1] = self._min_pos_var * 10
        safe_cov[5, 5] = self._min_orient_var * 10
        return safe_cov.flatten()

    def get_statistics(self) -> dict:
        return {
            'total_checks': self._total_checks,
            'suspicious_count': self._suspicious_count,
            'floor_applied_count': self._forced_floor_count,
            'suspicious_rate': self._suspicious_count / max(1, self._total_checks)
        }


class TemporalConsistencyChecker:
    """
    시간적 일관성 검사기

    공격 시나리오:
        공격자가 갑자기 매우 정확한 공분산으로 전환
        또는 위치가 물리적으로 불가능한 속도로 점프

    방어:
        - 공분산의 급격한 감소 탐지
        - 위치 변화율이 물리적 한계 초과 시 거부
        - 이동 평균과의 편차 검사
    """

    # 물리적 한계
    MAX_VELOCITY = 2.0  # m/s (일반적인 모바일 로봇)
    MAX_ACCELERATION = 3.0  # m/s² (급가속)

    # 공분산 변화 한계
    MAX_COVARIANCE_DECREASE_RATE = 0.5  # 1초당 최대 50% 감소

    def __init__(
        self,
        history_size: int = 50,
        max_velocity: float = MAX_VELOCITY,
        max_acceleration: float = MAX_ACCELERATION
    ):
        self._history_size = history_size
        self._max_velocity = max_velocity
        self._max_acceleration = max_acceleration

        # 히스토리 저장
        self._position_history: Deque[Tuple[float, float, float]] = deque(maxlen=history_size)
        self._covariance_history: Deque[Tuple[float, float]] = deque(maxlen=history_size)
        self._velocity_history: Deque[Tuple[float, float, float]] = deque(maxlen=history_size)

        # 통계
        self._total_checks = 0
        self._position_violations = 0
        self._covariance_violations = 0

    def check_position(
        self,
        x: float,
        y: float,
        timestamp: float
    ) -> Tuple[bool, List[str]]:
        """위치의 시간적 일관성 검사"""
        self._total_checks += 1
        violations = []

        if len(self._position_history) > 0:
            prev_x, prev_y, prev_time = self._position_history[-1]
            dt = timestamp - prev_time

            if dt > 0:
                # 속도 계산
                vx = (x - prev_x) / dt
                vy = (y - prev_y) / dt
                speed = np.sqrt(vx**2 + vy**2)

                # 속도 한계 검사
                if speed > self._max_velocity:
                    self._position_violations += 1
                    violations.append(
                        f"CRITICAL: Impossible velocity detected: {speed:.2f} m/s "
                        f"(max: {self._max_velocity} m/s) - possible position spoofing"
                    )

                # 가속도 검사 (이전 속도가 있는 경우)
                if len(self._velocity_history) > 0:
                    prev_vx, prev_vy, prev_vt = self._velocity_history[-1]
                    dt_v = timestamp - prev_vt

                    if dt_v > 0:
                        ax = (vx - prev_vx) / dt_v
                        ay = (vy - prev_vy) / dt_v
                        accel = np.sqrt(ax**2 + ay**2)

                        if accel > self._max_acceleration:
                            violations.append(
                                f"WARNING: High acceleration: {accel:.2f} m/s² "
                                f"(threshold: {self._max_acceleration} m/s²)"
                            )

                self._velocity_history.append((vx, vy, timestamp))

        self._position_history.append((x, y, timestamp))
        return len([v for v in violations if "CRITICAL" in v]) == 0, violations

    def check_covariance_change(
        self,
        new_variance: float,
        timestamp: float
    ) -> Tuple[bool, List[str]]:
        """공분산 변화율 검사"""
        violations = []

        if len(self._covariance_history) > 0:
            prev_var, prev_time = self._covariance_history[-1]
            dt = timestamp - prev_time

            if dt > 0 and prev_var > 0:
                # 변화율 계산
                change_rate = (prev_var - new_variance) / prev_var / dt

                # 급격한 감소 탐지
                if change_rate > self.MAX_COVARIANCE_DECREASE_RATE:
                    self._covariance_violations += 1
                    violations.append(
                        f"SUSPICIOUS: Covariance decreased too fast: "
                        f"{change_rate*100:.1f}%/s (max: {self.MAX_COVARIANCE_DECREASE_RATE*100}%/s)"
                    )

        self._covariance_history.append((new_variance, timestamp))
        return len(violations) == 0, violations

    def get_statistics(self) -> dict:
        return {
            'total_checks': self._total_checks,
            'position_violations': self._position_violations,
            'covariance_violations': self._covariance_violations,
            'history_size': len(self._position_history)
        }


class CrossSensorValidator:
    """
    센서 융합 교차 검증

    공격 시나리오:
        단일 센서(예: AMCL)만 스푸핑 → 다른 센서와 불일치

    방어:
        - 여러 센서 소스 비교
        - 불일치 시 보수적 추정 사용
        - 센서 신뢰도 가중치 동적 조정
    """

    # 센서 간 허용 편차
    POSITION_TOLERANCE = 0.5  # meters
    VELOCITY_TOLERANCE = 0.3  # m/s

    def __init__(
        self,
        position_tolerance: float = POSITION_TOLERANCE,
        velocity_tolerance: float = VELOCITY_TOLERANCE
    ):
        self._pos_tolerance = position_tolerance
        self._vel_tolerance = velocity_tolerance

        # 센서별 최신 데이터
        self._sensor_data: Dict[str, SensorReading] = {}

        # 센서 신뢰도 점수
        self._sensor_trust: Dict[str, float] = {}

        # 통계
        self._total_validations = 0
        self._inconsistencies = 0

    def update_sensor(self, reading: SensorReading):
        """센서 데이터 업데이트"""
        self._sensor_data[reading.source] = reading

        # 초기 신뢰도 설정
        if reading.source not in self._sensor_trust:
            self._sensor_trust[reading.source] = 1.0

    def validate_cross_sensors(self) -> Tuple[bool, List[str], Dict[str, float]]:
        """모든 센서 간 교차 검증"""
        self._total_validations += 1
        violations = []
        trust_updates = {}

        sensors = list(self._sensor_data.keys())

        for i, sensor_a in enumerate(sensors):
            for sensor_b in sensors[i+1:]:
                data_a = self._sensor_data[sensor_a]
                data_b = self._sensor_data[sensor_b]

                # 위치 비교
                if data_a.position and data_b.position:
                    dist = np.sqrt(
                        (data_a.position[0] - data_b.position[0])**2 +
                        (data_a.position[1] - data_b.position[1])**2
                    )

                    if dist > self._pos_tolerance:
                        self._inconsistencies += 1
                        violations.append(
                            f"INCONSISTENCY: {sensor_a} vs {sensor_b} "
                            f"position diff: {dist:.2f}m (tolerance: {self._pos_tolerance}m)"
                        )

                        # 신뢰도 감소 (둘 중 하나가 스푸핑됨)
                        trust_updates[sensor_a] = self._sensor_trust.get(sensor_a, 1.0) * 0.9
                        trust_updates[sensor_b] = self._sensor_trust.get(sensor_b, 1.0) * 0.9

                # 속도 비교
                if data_a.velocity and data_b.velocity:
                    vel_diff = np.sqrt(
                        (data_a.velocity[0] - data_b.velocity[0])**2 +
                        (data_a.velocity[1] - data_b.velocity[1])**2
                    )

                    if vel_diff > self._vel_tolerance:
                        violations.append(
                            f"INCONSISTENCY: {sensor_a} vs {sensor_b} "
                            f"velocity diff: {vel_diff:.2f}m/s"
                        )

        # 신뢰도 업데이트
        for sensor, new_trust in trust_updates.items():
            self._sensor_trust[sensor] = max(0.1, new_trust)  # 최소 0.1 유지

        return len(violations) == 0, violations, self._sensor_trust.copy()

    def get_most_trusted_position(self) -> Optional[Tuple[float, float, float]]:
        """가장 신뢰도 높은 위치 반환 (신뢰도 가중 평균)"""
        if not self._sensor_data:
            return None

        total_weight = 0.0
        weighted_x = 0.0
        weighted_y = 0.0

        for source, data in self._sensor_data.items():
            if data.position:
                trust = self._sensor_trust.get(source, 1.0)
                weighted_x += data.position[0] * trust
                weighted_y += data.position[1] * trust
                total_weight += trust

        if total_weight > 0:
            return (
                weighted_x / total_weight,
                weighted_y / total_weight,
                total_weight / len(self._sensor_data)  # 평균 신뢰도
            )

        return None

    def get_statistics(self) -> dict:
        return {
            'total_validations': self._total_validations,
            'inconsistencies': self._inconsistencies,
            'sensor_trust_scores': self._sensor_trust.copy()
        }


class SensorIntegrityMonitor:
    """
    통합 센서 무결성 모니터

    모든 검증 모듈을 통합하여 센서 스푸핑을 방어합니다.

    사용:
        monitor = SensorIntegrityMonitor()

        # AMCL 데이터 검증
        report = monitor.validate_amcl_pose(x, y, covariance, timestamp)

        if report.alert_level >= AlertLevel.CRITICAL:
            # 긴급 조치
            use_safe_fallback()
        elif report.alert_level >= AlertLevel.WARNING:
            # 경고 로깅
            log_warning(report)

        # 교정된 공분산 사용
        safe_covariance = report.corrected_covariance
    """

    def __init__(
        self,
        enable_covariance_floor: bool = True,
        enable_temporal_check: bool = True,
        enable_cross_validation: bool = True,
        min_position_variance: float = 0.01,
        max_velocity: float = 2.0
    ):
        # 검증 모듈 초기화
        self._covariance_validator = CovarianceValidator(
            min_position_var=min_position_variance
        ) if enable_covariance_floor else None

        self._temporal_checker = TemporalConsistencyChecker(
            max_velocity=max_velocity
        ) if enable_temporal_check else None

        self._cross_validator = CrossSensorValidator() if enable_cross_validation else None

        # 최근 보고서
        self._recent_reports: Deque[IntegrityReport] = deque(maxlen=100)

        # 연속 위반 카운터 (Emergency 판단용)
        self._consecutive_violations = 0
        self._emergency_threshold = 5

    def validate_amcl_pose(
        self,
        x: float,
        y: float,
        covariance: np.ndarray,
        timestamp: Optional[float] = None
    ) -> IntegrityReport:
        """
        AMCL pose 데이터 검증

        Args:
            x, y: 위치
            covariance: 36-element 공분산
            timestamp: 타임스탬프 (None이면 현재 시간)

        Returns:
            IntegrityReport with validation results
        """
        if timestamp is None:
            timestamp = time.time()

        violations = []
        corrected_cov = covariance.copy()
        alert_level = AlertLevel.NORMAL
        confidence = 1.0

        # 1. 공분산 검증
        if self._covariance_validator:
            is_valid, corrected_cov, cov_violations = self._covariance_validator.validate(covariance)
            violations.extend(cov_violations)

            if not is_valid:
                alert_level = AlertLevel.CRITICAL
                confidence *= 0.5

            if any("SUSPICIOUS" in v for v in cov_violations):
                alert_level = max(alert_level, AlertLevel.WARNING)
                confidence *= 0.7

        # 2. 시간적 일관성 검사
        if self._temporal_checker:
            pos_valid, pos_violations = self._temporal_checker.check_position(x, y, timestamp)
            violations.extend(pos_violations)

            if not pos_valid:
                alert_level = AlertLevel.CRITICAL
                confidence *= 0.3

            # 공분산 변화율 검사
            if covariance.size >= 1:
                cov_var = covariance.reshape(-1)[0]  # x variance
                _, cov_change_violations = self._temporal_checker.check_covariance_change(
                    cov_var, timestamp
                )
                violations.extend(cov_change_violations)

                if cov_change_violations:
                    alert_level = max(alert_level, AlertLevel.WARNING)
                    confidence *= 0.8

        # 3. 교차 센서 검증 (데이터 업데이트)
        if self._cross_validator:
            reading = SensorReading(
                timestamp=timestamp,
                source='amcl',
                position=(x, y),
                covariance=covariance
            )
            self._cross_validator.update_sensor(reading)

        # 4. 연속 위반 체크
        if alert_level >= AlertLevel.CRITICAL:
            self._consecutive_violations += 1
        else:
            self._consecutive_violations = max(0, self._consecutive_violations - 1)

        if self._consecutive_violations >= self._emergency_threshold:
            alert_level = AlertLevel.EMERGENCY

        # 보고서 생성
        report = IntegrityReport(
            timestamp=timestamp,
            sensor_source='amcl',
            is_valid=alert_level < AlertLevel.CRITICAL,
            alert_level=alert_level,
            violations=violations,
            original_covariance=covariance,
            corrected_covariance=corrected_cov,
            confidence_score=confidence
        )

        self._recent_reports.append(report)
        return report

    def validate_odometry(
        self,
        vx: float,
        vy: float,
        timestamp: Optional[float] = None
    ) -> IntegrityReport:
        """
        Odometry 데이터 검증

        주로 속도 과소보고 탐지
        """
        if timestamp is None:
            timestamp = time.time()

        violations = []
        alert_level = AlertLevel.NORMAL
        confidence = 1.0

        # 속도가 비정상적으로 작은지 검사
        speed = np.sqrt(vx**2 + vy**2)

        if self._temporal_checker and len(self._temporal_checker._velocity_history) > 2:
            # 이전 속도들과 비교
            recent_speeds = [
                np.sqrt(v[0]**2 + v[1]**2)
                for v in list(self._temporal_checker._velocity_history)[-5:]
            ]
            avg_speed = np.mean(recent_speeds)

            # 속도가 갑자기 매우 낮아졌는지 검사
            if avg_speed > 0.5 and speed < avg_speed * 0.1:
                violations.append(
                    f"SUSPICIOUS: Sudden velocity drop: "
                    f"{avg_speed:.2f} → {speed:.2f} m/s - possible odometry spoofing"
                )
                alert_level = AlertLevel.WARNING
                confidence *= 0.7

        # 교차 검증용 데이터 업데이트
        if self._cross_validator:
            reading = SensorReading(
                timestamp=timestamp,
                source='odometry',
                velocity=(vx, vy)
            )
            self._cross_validator.update_sensor(reading)

        report = IntegrityReport(
            timestamp=timestamp,
            sensor_source='odometry',
            is_valid=alert_level < AlertLevel.CRITICAL,
            alert_level=alert_level,
            violations=violations,
            confidence_score=confidence
        )

        self._recent_reports.append(report)
        return report

    def perform_cross_validation(self) -> IntegrityReport:
        """모든 센서 간 교차 검증 수행"""
        if not self._cross_validator:
            return IntegrityReport(
                timestamp=time.time(),
                sensor_source='cross_validation',
                is_valid=True,
                alert_level=AlertLevel.NORMAL,
                confidence_score=1.0
            )

        is_valid, violations, trust_scores = self._cross_validator.validate_cross_sensors()

        alert_level = AlertLevel.NORMAL
        if violations:
            alert_level = AlertLevel.WARNING
        if any("CRITICAL" in v for v in violations):
            alert_level = AlertLevel.CRITICAL

        # 평균 신뢰도 계산
        avg_trust = np.mean(list(trust_scores.values())) if trust_scores else 1.0

        return IntegrityReport(
            timestamp=time.time(),
            sensor_source='cross_validation',
            is_valid=is_valid,
            alert_level=alert_level,
            violations=violations,
            confidence_score=avg_trust
        )

    def get_safe_margin_multiplier(self) -> float:
        """
        안전 마진 배수 반환

        센서 무결성이 의심스러우면 안전 마진을 증가시킴
        """
        if not self._recent_reports:
            return 1.0

        # 최근 보고서의 신뢰도 평균
        recent_confidence = np.mean([
            r.confidence_score for r in list(self._recent_reports)[-10:]
        ])

        # 신뢰도가 낮으면 마진 배수 증가
        # confidence 1.0 → multiplier 1.0
        # confidence 0.5 → multiplier 2.0
        # confidence 0.1 → multiplier 10.0
        multiplier = 1.0 / max(0.1, recent_confidence)

        return min(multiplier, 5.0)  # 최대 5배

    def get_statistics(self) -> dict:
        """통계 반환"""
        stats = {
            'consecutive_violations': self._consecutive_violations,
            'recent_reports_count': len(self._recent_reports)
        }

        if self._covariance_validator:
            stats['covariance_validator'] = self._covariance_validator.get_statistics()

        if self._temporal_checker:
            stats['temporal_checker'] = self._temporal_checker.get_statistics()

        if self._cross_validator:
            stats['cross_validator'] = self._cross_validator.get_statistics()

        return stats


# =============================================================================
# 안전 마진 계산기 통합
# =============================================================================

class SecureMarginCalculator:
    """
    보안 강화 안전 마진 계산기

    기존 MarginCalculator에 센서 무결성 검증을 통합
    """

    def __init__(
        self,
        base_margin: float = 0.5,
        integrity_monitor: Optional[SensorIntegrityMonitor] = None
    ):
        self._base_margin = base_margin
        self._integrity_monitor = integrity_monitor or SensorIntegrityMonitor()

    def calculate_secure_margin(
        self,
        covariance: np.ndarray,
        velocity: Tuple[float, float] = (0.0, 0.0),
        timestamp: Optional[float] = None
    ) -> Tuple[float, IntegrityReport]:
        """
        보안 강화 마진 계산

        Returns:
            (secure_margin, integrity_report)
        """
        if timestamp is None:
            timestamp = time.time()

        # 1. 센서 무결성 검증
        report = self._integrity_monitor.validate_amcl_pose(
            x=0.0, y=0.0,  # 위치는 마진 계산에 불필요
            covariance=covariance,
            timestamp=timestamp
        )

        # 2. 교정된 공분산 사용
        safe_covariance = report.corrected_covariance
        if safe_covariance is None:
            safe_covariance = covariance

        # 3. λ_max 계산 (교정된 공분산 사용)
        cov_matrix = safe_covariance.reshape(6, 6) if safe_covariance.size == 36 else np.eye(6) * 0.1
        xy_cov = cov_matrix[:2, :2]

        try:
            lambda_max = np.max(np.linalg.eigvalsh(xy_cov))
        except:
            lambda_max = 0.1

        # 4. 기본 마진 계산 (χ²₂(0.99) ≈ 9.21)
        chi2_99 = 9.21
        position_margin = np.sqrt(chi2_99 * lambda_max)

        # 5. 속도 항 추가
        speed = np.sqrt(velocity[0]**2 + velocity[1]**2)
        velocity_margin = speed * 0.5  # τ = 0.5초

        # 6. 기본 마진
        base_margin = self._base_margin

        # 7. 무결성 배수 적용
        integrity_multiplier = self._integrity_monitor.get_safe_margin_multiplier()

        # 최종 마진
        total_margin = (position_margin + velocity_margin + base_margin) * integrity_multiplier

        return total_margin, report
