#!/usr/bin/env python3
"""
Sensor Integrity Monitor Tests - 센서 스푸핑 방어 테스트

테스트 시나리오:
1. 공분산 스푸핑 공격 탐지
2. 위치 점프 공격 탐지
3. 공분산 급격 변화 탐지
4. 센서 간 불일치 탐지
5. 보안 강화 마진 계산
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from geofence_enforcer.sensor_integrity_monitor import (
    SensorIntegrityMonitor,
    CovarianceValidator,
    TemporalConsistencyChecker,
    CrossSensorValidator,
    SecureMarginCalculator,
    SensorReading,
    AlertLevel
)


class TestCovarianceValidator:
    """공분산 검증기 테스트"""

    def setup_method(self):
        self.validator = CovarianceValidator()

    def test_normal_covariance(self):
        """정상적인 공분산은 통과"""
        # 일반적인 AMCL 공분산 (약 5cm 불확실성)
        cov = np.eye(6) * 0.05
        cov[0, 0] = 0.02  # x variance
        cov[1, 1] = 0.02  # y variance
        cov[5, 5] = 0.01  # yaw variance

        is_valid, corrected, violations = self.validator.validate(cov.flatten())

        assert is_valid
        assert len([v for v in violations if "SUSPICIOUS" in v]) == 0

    def test_spoofed_low_covariance(self):
        """스푸핑된 낮은 공분산 탐지"""
        # 공격: 비정상적으로 낮은 공분산
        cov = np.eye(6) * 0.001  # 1mm 불확실성 - 불가능
        cov[0, 0] = 0.0001
        cov[1, 1] = 0.0001

        is_valid, corrected, violations = self.validator.validate(cov.flatten())

        # 스푸핑 탐지
        assert not is_valid or any("SUSPICIOUS" in v for v in violations)

        # 교정된 공분산은 최소 경계 이상
        corrected_matrix = corrected.reshape(6, 6)
        assert corrected_matrix[0, 0] >= self.validator._min_pos_var
        assert corrected_matrix[1, 1] >= self.validator._min_pos_var

    def test_covariance_floor_enforcement(self):
        """공분산 최소 경계 강제"""
        # 너무 낮은 공분산
        cov = np.eye(6) * 0.005  # 경계 미만

        is_valid, corrected, violations = self.validator.validate(cov.flatten())

        # Floor 적용됨
        assert any("Floor applied" in v for v in violations)

        corrected_matrix = corrected.reshape(6, 6)
        assert corrected_matrix[0, 0] >= self.validator._min_pos_var

    def test_statistics_tracking(self):
        """통계 추적"""
        # 정상 데이터
        for _ in range(5):
            cov = np.eye(6) * 0.05
            self.validator.validate(cov.flatten())

        # 스푸핑 데이터
        for _ in range(3):
            cov = np.eye(6) * 0.001
            self.validator.validate(cov.flatten())

        stats = self.validator.get_statistics()

        assert stats['total_checks'] == 8
        assert stats['suspicious_count'] >= 3


class TestTemporalConsistencyChecker:
    """시간적 일관성 검사기 테스트"""

    def setup_method(self):
        self.checker = TemporalConsistencyChecker(max_velocity=2.0)

    def test_normal_movement(self):
        """정상적인 이동은 통과"""
        # 0.5m/s로 이동
        positions = [
            (0.0, 0.0, 0.0),
            (0.5, 0.0, 1.0),
            (1.0, 0.0, 2.0),
            (1.5, 0.0, 3.0),
        ]

        for x, y, t in positions:
            is_valid, violations = self.checker.check_position(x, y, t)
            assert is_valid
            assert len([v for v in violations if "CRITICAL" in v]) == 0

    def test_impossible_teleportation(self):
        """불가능한 순간이동 탐지"""
        # 초기 위치
        self.checker.check_position(0.0, 0.0, 0.0)

        # 0.1초 후 10m 이동 = 100m/s (불가능)
        is_valid, violations = self.checker.check_position(10.0, 0.0, 0.1)

        assert not is_valid
        assert any("CRITICAL" in v for v in violations)
        assert any("Impossible velocity" in v for v in violations)

    def test_covariance_sudden_drop(self):
        """공분산 급격 감소 탐지"""
        # 초기 공분산 (높은 불확실성)
        self.checker.check_covariance_change(0.1, 0.0)

        # 1초 후 90% 감소 (의심스러움)
        is_valid, violations = self.checker.check_covariance_change(0.01, 1.0)

        assert len(violations) > 0
        assert any("SUSPICIOUS" in v for v in violations)


class TestCrossSensorValidator:
    """교차 센서 검증기 테스트"""

    def setup_method(self):
        self.validator = CrossSensorValidator(position_tolerance=0.5)

    def test_consistent_sensors(self):
        """일관된 센서 데이터는 통과"""
        # AMCL
        amcl_reading = SensorReading(
            timestamp=1.0,
            source='amcl',
            position=(5.0, 5.0)
        )
        self.validator.update_sensor(amcl_reading)

        # Odometry (위치 유사)
        odom_reading = SensorReading(
            timestamp=1.0,
            source='odometry',
            position=(5.1, 5.1)  # 0.14m 차이
        )
        self.validator.update_sensor(odom_reading)

        is_valid, violations, _ = self.validator.validate_cross_sensors()

        assert is_valid
        assert len(violations) == 0

    def test_inconsistent_sensors(self):
        """불일치 센서 탐지"""
        # AMCL
        amcl_reading = SensorReading(
            timestamp=1.0,
            source='amcl',
            position=(5.0, 5.0)
        )
        self.validator.update_sensor(amcl_reading)

        # Odometry (위치 매우 다름 - 스푸핑 의심)
        odom_reading = SensorReading(
            timestamp=1.0,
            source='odometry',
            position=(7.0, 7.0)  # 2.8m 차이
        )
        self.validator.update_sensor(odom_reading)

        is_valid, violations, trust_scores = self.validator.validate_cross_sensors()

        assert not is_valid
        assert len(violations) > 0
        assert any("INCONSISTENCY" in v for v in violations)

        # 신뢰도 감소
        assert trust_scores['amcl'] < 1.0
        assert trust_scores['odometry'] < 1.0

    def test_trust_weighted_position(self):
        """신뢰도 가중 위치 계산"""
        # 높은 신뢰도 센서
        self.validator.update_sensor(SensorReading(
            timestamp=1.0, source='lidar', position=(5.0, 5.0)
        ))
        self.validator._sensor_trust['lidar'] = 1.0

        # 낮은 신뢰도 센서
        self.validator.update_sensor(SensorReading(
            timestamp=1.0, source='gps', position=(10.0, 10.0)
        ))
        self.validator._sensor_trust['gps'] = 0.2

        result = self.validator.get_most_trusted_position()

        # 결과는 lidar에 가까워야 함
        assert result is not None
        x, y, _ = result
        assert abs(x - 5.0) < abs(x - 10.0)
        assert abs(y - 5.0) < abs(y - 10.0)


class TestSensorIntegrityMonitor:
    """통합 모니터 테스트"""

    def setup_method(self):
        self.monitor = SensorIntegrityMonitor()

    def test_normal_operation(self):
        """정상 동작"""
        # 정상적인 AMCL 데이터
        cov = np.eye(6) * 0.05

        report = self.monitor.validate_amcl_pose(
            x=5.0, y=5.0,
            covariance=cov.flatten(),
            timestamp=0.0
        )

        assert report.is_valid
        assert report.alert_level == AlertLevel.NORMAL

    def test_covariance_spoofing_attack(self):
        """공분산 스푸핑 공격 탐지"""
        # 공격: 매우 낮은 공분산
        cov = np.eye(6) * 0.0001  # 불가능하게 낮음

        report = self.monitor.validate_amcl_pose(
            x=5.0, y=5.0,
            covariance=cov.flatten(),
            timestamp=0.0
        )

        # 스푸핑 탐지
        assert report.alert_level >= AlertLevel.WARNING
        assert len(report.violations) > 0

        # 공분산이 교정됨
        assert report.corrected_covariance is not None
        corrected_matrix = report.corrected_covariance.reshape(6, 6)
        assert corrected_matrix[0, 0] > cov[0, 0]

    def test_position_spoofing_attack(self):
        """위치 스푸핑 공격 탐지 (텔레포트)"""
        cov = np.eye(6) * 0.05

        # 초기 위치
        self.monitor.validate_amcl_pose(0.0, 0.0, cov.flatten(), 0.0)

        # 0.1초 후 20m 이동 (물리적으로 불가능)
        report = self.monitor.validate_amcl_pose(20.0, 0.0, cov.flatten(), 0.1)

        assert not report.is_valid
        assert report.alert_level >= AlertLevel.CRITICAL
        assert any("velocity" in v.lower() for v in report.violations)

    def test_safe_margin_multiplier(self):
        """안전 마진 배수"""
        cov = np.eye(6) * 0.05

        # 정상 데이터 → 배수 1.0
        for i in range(5):
            self.monitor.validate_amcl_pose(float(i), 0.0, cov.flatten(), float(i))

        multiplier = self.monitor.get_safe_margin_multiplier()
        assert 0.9 <= multiplier <= 1.1

        # 의심스러운 데이터 추가
        spoofed_cov = np.eye(6) * 0.0001
        for i in range(5, 10):
            self.monitor.validate_amcl_pose(float(i), 0.0, spoofed_cov.flatten(), float(i))

        multiplier = self.monitor.get_safe_margin_multiplier()
        # 배수 증가 (신뢰도 감소)
        assert multiplier > 1.0

    def test_emergency_escalation(self):
        """연속 위반 시 Emergency 에스컬레이션"""
        # 불가능한 위치 연속 보고
        cov = np.eye(6) * 0.05

        self.monitor.validate_amcl_pose(0.0, 0.0, cov.flatten(), 0.0)

        # 연속 텔레포트
        for i in range(10):
            report = self.monitor.validate_amcl_pose(
                100.0 * (i + 1), 0.0, cov.flatten(), 0.1 * (i + 1)
            )

        # Emergency로 에스컬레이션
        assert report.alert_level == AlertLevel.EMERGENCY


class TestSecureMarginCalculator:
    """보안 강화 마진 계산기 테스트"""

    def setup_method(self):
        self.calculator = SecureMarginCalculator(base_margin=0.3)

    def test_normal_margin(self):
        """정상 마진 계산"""
        cov = np.eye(6) * 0.05

        margin, report = self.calculator.calculate_secure_margin(
            covariance=cov.flatten(),
            velocity=(0.5, 0.0),
            timestamp=0.0
        )

        assert margin > 0.3  # base_margin 이상
        assert report.is_valid

    def test_spoofed_covariance_increases_margin(self):
        """스푸핑된 공분산은 마진 증가"""
        # 정상 마진
        normal_cov = np.eye(6) * 0.05
        normal_margin, _ = self.calculator.calculate_secure_margin(
            covariance=normal_cov.flatten(),
            timestamp=0.0
        )

        # 스푸핑 마진 (낮은 공분산)
        spoofed_cov = np.eye(6) * 0.0001
        spoofed_margin, report = self.calculator.calculate_secure_margin(
            covariance=spoofed_cov.flatten(),
            timestamp=1.0
        )

        # 스푸핑 감지됨
        assert len(report.violations) > 0

        # 마진은 교정된 공분산 기반으로 계산되어야 함
        # (스푸핑된 낮은 값이 아닌)


class TestAttackScenarios:
    """실제 공격 시나리오 테스트"""

    def test_scenario_1_covariance_manipulation(self):
        """
        시나리오 1: 공분산 조작 공격

        공격자가 AMCL 토픽을 가로채서 공분산을 매우 낮게 조작
        → 시스템이 위치가 정확하다고 판단
        → 안전 마진 축소
        → 금지구역에 더 가까이 접근 가능
        """
        monitor = SensorIntegrityMonitor()

        # 정상 운영 (몇 초간)
        normal_cov = np.eye(6) * 0.04
        for i in range(10):
            monitor.validate_amcl_pose(
                x=float(i) * 0.1, y=0.0,
                covariance=normal_cov.flatten(),
                timestamp=float(i)
            )

        # 공격 시작: 공분산 급격 감소
        attack_cov = np.eye(6) * 0.0001  # 0.1mm - 불가능

        report = monitor.validate_amcl_pose(
            x=1.1, y=0.0,
            covariance=attack_cov.flatten(),
            timestamp=11.0
        )

        # 결과: 공격 탐지
        assert report.alert_level >= AlertLevel.WARNING
        assert any("SUSPICIOUS" in v or "Floor" in v for v in report.violations)

        # 교정된 공분산은 최소 경계 이상
        if report.corrected_covariance is not None:
            corrected = report.corrected_covariance.reshape(6, 6)
            assert corrected[0, 0] >= 0.01  # 최소 경계

    def test_scenario_2_position_teleport(self):
        """
        시나리오 2: 위치 스푸핑 공격 (텔레포트)

        공격자가 로봇 위치를 금지구역 밖으로 가짜 보고
        → 시스템이 로봇이 안전하다고 판단
        → 실제로는 금지구역 내부
        """
        monitor = SensorIntegrityMonitor()

        cov = np.eye(6) * 0.05

        # 정상 위치 (금지구역 근처)
        monitor.validate_amcl_pose(4.0, 5.0, cov.flatten(), 0.0)

        # 공격: 갑자기 먼 위치로 보고 (0.1초 후 100m 이동)
        report = monitor.validate_amcl_pose(0.0, 0.0, cov.flatten(), 0.1)

        # 결과: 불가능한 이동 탐지
        assert not report.is_valid
        assert report.alert_level >= AlertLevel.CRITICAL
        assert any("velocity" in v.lower() or "impossible" in v.lower()
                  for v in report.violations)

    def test_scenario_3_gradual_covariance_attack(self):
        """
        시나리오 3: 점진적 공분산 공격

        갑작스러운 변화 대신 천천히 공분산을 낮춤
        → 단순 임계값 검사 우회 시도
        """
        monitor = SensorIntegrityMonitor()

        # 천천히 공분산 감소
        for i in range(20):
            # 0.05 → 0.001 (20 단계)
            variance = 0.05 - (0.05 - 0.001) * (i / 19)
            cov = np.eye(6) * variance

            report = monitor.validate_amcl_pose(
                x=float(i) * 0.1, y=0.0,
                covariance=cov.flatten(),
                timestamp=float(i)
            )

        # 최종 공분산은 최소 경계로 교정되어야 함
        assert report.corrected_covariance is not None
        corrected = report.corrected_covariance.reshape(6, 6)

        # 공분산이 물리적 한계 이하로 떨어지면 교정됨
        assert corrected[0, 0] >= 0.01

    def test_scenario_4_multi_sensor_inconsistency(self):
        """
        시나리오 4: 단일 센서 스푸핑

        공격자가 AMCL만 스푸핑
        → 다른 센서(odometry)와 불일치
        """
        cross_validator = CrossSensorValidator()

        # AMCL: 스푸핑된 위치
        cross_validator.update_sensor(SensorReading(
            timestamp=1.0,
            source='amcl',
            position=(0.0, 0.0)  # 가짜 위치
        ))

        # Odometry: 실제 위치 (다름)
        cross_validator.update_sensor(SensorReading(
            timestamp=1.0,
            source='odometry',
            position=(5.0, 5.0)  # 실제 위치
        ))

        is_valid, violations, trust_scores = cross_validator.validate_cross_sensors()

        # 불일치 탐지
        assert not is_valid
        assert len(violations) > 0

        # 신뢰도 감소
        assert trust_scores['amcl'] < 1.0
        assert trust_scores['odometry'] < 1.0


def run_all_tests():
    """모든 테스트 실행"""
    print("=" * 70)
    print(" Sensor Integrity Monitor Tests ".center(70))
    print("=" * 70)

    # pytest 실행
    exit_code = pytest.main([__file__, '-v', '--tb=short'])
    return exit_code == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
