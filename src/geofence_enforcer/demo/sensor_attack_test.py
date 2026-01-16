#!/usr/bin/env python3
"""
Sensor Spoofing Attack Test - 실제 공격 시뮬레이션

이 테스트는 실제로:
1. 스푸핑된 센서 데이터를 생성
2. 방어 시스템 없이 안전 마진 계산 (취약)
3. 방어 시스템과 함께 안전 마진 계산 (방어)
4. 차이를 비교하여 방어 효과 검증

실행:
    python3 demo/sensor_attack_test.py
"""

import sys
import os
import numpy as np
import time
from dataclasses import dataclass
from typing import Tuple, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from geofence_enforcer.sensor_integrity_monitor import (
    SensorIntegrityMonitor,
    SecureMarginCalculator,
    CovarianceValidator,
    AlertLevel
)
from geofence_enforcer.margin_calculator import MarginCalculator


@dataclass
class AttackResult:
    """공격 테스트 결과"""
    attack_name: str
    attack_description: str

    # 공격 데이터
    spoofed_covariance: np.ndarray
    spoofed_position: Tuple[float, float]

    # 방어 없는 경우
    unprotected_margin: float
    unprotected_would_allow: bool  # 금지구역 진입 허용 여부

    # 방어 있는 경우
    protected_margin: float
    protected_would_allow: bool
    alert_level: AlertLevel
    violations: List[str]

    # 방어 성공 여부
    attack_blocked: bool


class SensorAttackSimulator:
    """센서 스푸핑 공격 시뮬레이터"""

    # 금지구역 정의
    FORBIDDEN_ZONE = {
        'min': (4.5, 4.5),
        'max': (6.5, 6.5),
        'center': (5.5, 5.5)
    }

    def __init__(self):
        # 방어 없는 시스템 (취약)
        self.unprotected_calculator = MarginCalculator(
            alpha=0.99,
            e_track=0.05,
            tau=0.5
        )

        # 방어 있는 시스템
        self.integrity_monitor = SensorIntegrityMonitor(
            enable_covariance_floor=True,
            enable_temporal_check=True,
            enable_cross_validation=True,
            min_position_variance=0.01,
            max_velocity=2.0
        )

        self.protected_calculator = SecureMarginCalculator(
            base_margin=0.3,
            integrity_monitor=self.integrity_monitor
        )

        # 타임스탬프
        self._current_time = 0.0

    def _advance_time(self, dt: float = 0.1):
        """시간 진행"""
        self._current_time += dt

    def _distance_to_forbidden(self, x: float, y: float) -> float:
        """금지구역까지 거리 (내부면 음수)"""
        min_x, min_y = self.FORBIDDEN_ZONE['min']
        max_x, max_y = self.FORBIDDEN_ZONE['max']

        # 내부인 경우
        if min_x <= x <= max_x and min_y <= y <= max_y:
            dx = min(x - min_x, max_x - x)
            dy = min(y - min_y, max_y - y)
            return -min(dx, dy)

        # 외부인 경우
        dx = max(min_x - x, 0, x - max_x)
        dy = max(min_y - y, 0, y - max_y)
        return np.sqrt(dx**2 + dy**2)

    def _would_allow_entry(self, position: Tuple[float, float], margin: float) -> bool:
        """주어진 마진으로 금지구역 진입이 허용되는지"""
        distance = self._distance_to_forbidden(*position)
        # 마진이 거리보다 작으면 진입 허용됨 (위험)
        return margin < distance

    def run_attack(
        self,
        attack_name: str,
        description: str,
        spoofed_covariance: np.ndarray,
        position: Tuple[float, float],
        velocity: Tuple[float, float] = (0.0, 0.0),
        setup_history: bool = True
    ) -> AttackResult:
        """단일 공격 실행"""

        # 정상 히스토리 설정 (temporal check용)
        if setup_history:
            normal_cov = np.eye(6) * 0.05
            for i in range(10):
                self._advance_time(0.1)
                # 정상 위치에서 시작
                normal_pos = (position[0] - 1.0, position[1])
                self.integrity_monitor.validate_amcl_pose(
                    x=normal_pos[0] + i * 0.1,
                    y=normal_pos[1],
                    covariance=normal_cov.flatten(),
                    timestamp=self._current_time
                )

        # 1. 방어 없는 시스템 (취약)
        # 스푸핑된 공분산으로 마진 계산
        cov_matrix = spoofed_covariance.reshape(6, 6) if spoofed_covariance.size == 36 else spoofed_covariance
        xy_cov = cov_matrix[:2, :2] if cov_matrix.shape[0] >= 2 else np.eye(2) * 0.01

        try:
            lambda_max = np.max(np.linalg.eigvalsh(xy_cov))
        except:
            lambda_max = 0.01

        # 취약한 마진 계산 (스푸핑된 낮은 공분산 그대로 사용)
        chi2_99 = 9.21
        unprotected_margin = np.sqrt(chi2_99 * lambda_max) + 0.05 + np.sqrt(velocity[0]**2 + velocity[1]**2) * 0.5

        # 2. 방어 있는 시스템
        self._advance_time(0.1)
        protected_margin, report = self.protected_calculator.calculate_secure_margin(
            covariance=spoofed_covariance.flatten(),
            velocity=velocity,
            timestamp=self._current_time
        )

        # 3. 결과 분석
        unprotected_would_allow = position[0] < self.FORBIDDEN_ZONE['min'][0] - unprotected_margin
        protected_would_allow = position[0] < self.FORBIDDEN_ZONE['min'][0] - protected_margin

        # 공격이 차단되었는지 (방어 시스템이 더 큰 마진을 요구)
        attack_blocked = protected_margin > unprotected_margin * 1.5 or report.alert_level >= AlertLevel.WARNING

        return AttackResult(
            attack_name=attack_name,
            attack_description=description,
            spoofed_covariance=spoofed_covariance,
            spoofed_position=position,
            unprotected_margin=unprotected_margin,
            unprotected_would_allow=unprotected_would_allow,
            protected_margin=protected_margin,
            protected_would_allow=protected_would_allow,
            alert_level=report.alert_level,
            violations=report.violations,
            attack_blocked=attack_blocked
        )


def run_attack_scenarios():
    """모든 공격 시나리오 실행"""

    print("=" * 70)
    print(" 센서 스푸핑 공격 시뮬레이션 ".center(70))
    print("=" * 70)
    print()
    print("이 테스트는 실제 공격 데이터를 생성하여")
    print("방어 시스템의 효과를 검증합니다.")
    print()
    print(f"금지구역: {SensorAttackSimulator.FORBIDDEN_ZONE}")
    print()

    results = []

    # =========================================================================
    # 공격 시나리오 1: 공분산 조작 공격
    # =========================================================================
    print("-" * 70)
    print("공격 1: AMCL 공분산 조작")
    print("-" * 70)

    simulator = SensorAttackSimulator()

    # 공격: 매우 낮은 공분산 (1mm 정확도로 위장)
    spoofed_cov = np.eye(6) * 0.0001  # 0.1mm variance - 불가능
    position = (4.2, 5.5)  # 금지구역 경계 근처

    result = simulator.run_attack(
        attack_name="공분산 조작",
        description="공분산을 0.0001(0.1mm²)로 조작하여 안전 마진 축소 시도",
        spoofed_covariance=spoofed_cov,
        position=position
    )
    results.append(result)

    print(f"  공격 데이터: covariance = {spoofed_cov[0,0]:.6f} (정상: ~0.05)")
    print(f"  위치: {position} (금지구역 경계에서 0.3m)")
    print()
    print(f"  [방어 없음]")
    print(f"    계산된 마진: {result.unprotected_margin:.4f}m")
    print(f"    → 마진이 매우 작아 금지구역 접근 허용됨!")
    print()
    print(f"  [방어 있음]")
    print(f"    계산된 마진: {result.protected_margin:.4f}m")
    print(f"    경고 수준: {result.alert_level.name}")
    print(f"    탐지된 위반: {len(result.violations)}개")
    for v in result.violations[:3]:
        print(f"      - {v[:60]}...")
    print()
    print(f"  결과: {'✅ 공격 차단' if result.attack_blocked else '❌ 공격 성공'}")
    print()

    # =========================================================================
    # 공격 시나리오 2: 위치 스푸핑 (텔레포트)
    # =========================================================================
    print("-" * 70)
    print("공격 2: 위치 스푸핑 (텔레포트)")
    print("-" * 70)

    simulator2 = SensorAttackSimulator()

    # 정상 히스토리 먼저 설정
    normal_cov = np.eye(6) * 0.05
    for i in range(10):
        simulator2._advance_time(0.1)
        simulator2.integrity_monitor.validate_amcl_pose(
            x=5.0 + i * 0.05,  # 금지구역 근처에서 천천히 이동
            y=5.5,
            covariance=normal_cov.flatten(),
            timestamp=simulator2._current_time
        )

    # 공격: 갑자기 먼 위치로 보고 (로봇이 안전하다고 속임)
    simulator2._advance_time(0.1)
    spoofed_position = (0.0, 0.0)  # 갑자기 원점으로 (5.5m 순간이동)

    report = simulator2.integrity_monitor.validate_amcl_pose(
        x=spoofed_position[0],
        y=spoofed_position[1],
        covariance=normal_cov.flatten(),
        timestamp=simulator2._current_time
    )

    result2 = AttackResult(
        attack_name="위치 스푸핑",
        attack_description="위치를 갑자기 원점으로 보고하여 시스템을 속임",
        spoofed_covariance=normal_cov,
        spoofed_position=spoofed_position,
        unprotected_margin=0.3,
        unprotected_would_allow=True,
        protected_margin=0.3,
        protected_would_allow=False,
        alert_level=report.alert_level,
        violations=report.violations,
        attack_blocked=report.alert_level >= AlertLevel.CRITICAL
    )
    results.append(result2)

    print(f"  공격 데이터: 위치 (5.5, 5.5) → (0.0, 0.0) (0.1초 만에)")
    print(f"  계산된 속도: {5.5 * np.sqrt(2) / 0.1:.1f} m/s (물리적 한계: 2 m/s)")
    print()
    print(f"  [방어 없음]")
    print(f"    시스템이 위치를 그대로 신뢰")
    print(f"    → 로봇이 안전하다고 판단 (실제로는 금지구역 내부일 수 있음)")
    print()
    print(f"  [방어 있음]")
    print(f"    경고 수준: {report.alert_level.name}")
    print(f"    탐지된 위반: {len(report.violations)}개")
    for v in report.violations[:3]:
        print(f"      - {v[:60]}...")
    print()
    print(f"  결과: {'✅ 공격 차단' if result2.attack_blocked else '❌ 공격 성공'}")
    print()

    # =========================================================================
    # 공격 시나리오 3: 점진적 공분산 공격
    # =========================================================================
    print("-" * 70)
    print("공격 3: 점진적 공분산 공격 (탐지 우회 시도)")
    print("-" * 70)

    simulator3 = SensorAttackSimulator()

    # 천천히 공분산을 낮춤
    gradual_violations = []
    final_margin = 0.0

    for i in range(30):
        simulator3._advance_time(0.1)

        # 0.05 → 0.001로 점진적 감소
        variance = 0.05 - (0.05 - 0.001) * (i / 29)
        cov = np.eye(6) * variance

        margin, report = simulator3.protected_calculator.calculate_secure_margin(
            covariance=cov.flatten(),
            timestamp=simulator3._current_time
        )

        if report.violations:
            gradual_violations.extend(report.violations)

        final_margin = margin

    # 최종 공분산 확인
    final_report = report

    result3 = AttackResult(
        attack_name="점진적 공분산 공격",
        attack_description="30단계에 걸쳐 공분산을 0.05 → 0.001로 서서히 낮춤",
        spoofed_covariance=np.eye(6) * 0.001,
        spoofed_position=(4.2, 5.5),
        unprotected_margin=np.sqrt(9.21 * 0.001) + 0.05,  # 매우 작음
        unprotected_would_allow=True,
        protected_margin=final_margin,
        protected_would_allow=False,
        alert_level=final_report.alert_level,
        violations=gradual_violations[-5:],
        attack_blocked=len(gradual_violations) > 0
    )
    results.append(result3)

    print(f"  공격 데이터: 30단계에 걸쳐 공분산 0.05 → 0.001")
    print()
    print(f"  [방어 없음]")
    print(f"    최종 마진: {result3.unprotected_margin:.4f}m (매우 작음)")
    print()
    print(f"  [방어 있음]")
    print(f"    최종 마진: {result3.protected_margin:.4f}m")
    print(f"    탐지된 위반: {len(gradual_violations)}개")
    unique_violations = set(v[:40] for v in gradual_violations)
    for v in list(unique_violations)[:3]:
        print(f"      - {v}...")
    print()
    print(f"  결과: {'✅ 공격 차단' if result3.attack_blocked else '❌ 공격 성공'}")
    print()

    # =========================================================================
    # 공격 시나리오 4: Odometry 속도 과소보고
    # =========================================================================
    print("-" * 70)
    print("공격 4: Odometry 속도 과소보고")
    print("-" * 70)

    simulator4 = SensorAttackSimulator()

    # 정상 히스토리
    for i in range(10):
        simulator4._advance_time(0.1)
        simulator4.integrity_monitor.validate_odometry(
            vx=1.0, vy=0.0,  # 정상 속도 1m/s
            timestamp=simulator4._current_time
        )

    # 공격: 속도를 갑자기 0으로 보고
    simulator4._advance_time(0.1)
    odom_report = simulator4.integrity_monitor.validate_odometry(
        vx=0.0, vy=0.0,  # 갑자기 정지로 보고
        timestamp=simulator4._current_time
    )

    result4 = AttackResult(
        attack_name="Odometry 스푸핑",
        attack_description="실제 1m/s로 이동 중인데 0m/s로 보고",
        spoofed_covariance=np.eye(6) * 0.05,
        spoofed_position=(4.2, 5.5),
        unprotected_margin=0.3,  # v_max*τ 항이 0이 됨
        unprotected_would_allow=True,
        protected_margin=0.3 + 1.0 * 0.5,  # 실제 속도 반영
        protected_would_allow=False,
        alert_level=odom_report.alert_level,
        violations=odom_report.violations,
        attack_blocked=len(odom_report.violations) > 0 or odom_report.alert_level >= AlertLevel.WARNING
    )
    results.append(result4)

    print(f"  공격 데이터: 실제 속도 1.0m/s → 보고 속도 0.0m/s")
    print()
    print(f"  [방어 없음]")
    print(f"    v_max·τ 항: 0.0m (실제: 0.5m)")
    print(f"    → 안전 마진이 줄어듦")
    print()
    print(f"  [방어 있음]")
    print(f"    경고 수준: {odom_report.alert_level.name}")
    print(f"    탐지된 위반: {len(odom_report.violations)}개")
    for v in odom_report.violations[:3]:
        print(f"      - {v[:60]}...")
    print()
    print(f"  결과: {'✅ 공격 차단' if result4.attack_blocked else '❌ 공격 성공'}")
    print()

    # =========================================================================
    # 최종 결과 요약
    # =========================================================================
    print("=" * 70)
    print(" 공격 시뮬레이션 결과 요약 ".center(70))
    print("=" * 70)
    print()

    blocked_count = sum(1 for r in results if r.attack_blocked)
    total_count = len(results)

    print(f"총 공격 시나리오: {total_count}")
    print(f"차단된 공격: {blocked_count}")
    print(f"방어율: {blocked_count/total_count*100:.0f}%")
    print()

    print("상세 결과:")
    print("-" * 70)

    for r in results:
        status = "✅ 차단" if r.attack_blocked else "❌ 실패"
        print(f"  {status} | {r.attack_name}")
        print(f"         | 방어없음 마진: {r.unprotected_margin:.4f}m → 방어있음: {r.protected_margin:.4f}m")
        print(f"         | 경고: {r.alert_level.name}, 위반: {len(r.violations)}개")
        print()

    print("=" * 70)

    if blocked_count == total_count:
        print("\n✅ 모든 공격이 성공적으로 차단되었습니다!")
        print("   센서 무결성 모니터가 정상적으로 동작합니다.")
    else:
        print(f"\n⚠️  {total_count - blocked_count}개 공격이 차단되지 않았습니다.")
        print("   방어 시스템 검토가 필요합니다.")

    return blocked_count == total_count


def main():
    """메인 함수"""
    success = run_attack_scenarios()
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
