#!/usr/bin/env python3
"""
Numerical/Algorithmic Attack Simulation Test

수학적/알고리즘적 취약점을 이용한 공격 시뮬레이션:
1. Chi-square 가정 위반 공격
2. Singular 공분산 행렬 공격
3. Concave 폴리곤 악용
4. 이산화 틈새 공격

실행:
    python3 demo/numerical_attack_test.py
"""

import sys
import os
import numpy as np
from scipy import stats
from shapely.geometry import Polygon, LineString, Point

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from geofence_enforcer.numerical_safety import (
    NumericalSafetyChecker,
    DistributionValidator,
    NumericalStabilizer,
    PolygonValidator,
    PathDiscretizationGuard,
    NumericalAlertLevel
)


# =============================================================================
# 금지구역 정의
# =============================================================================
FORBIDDEN_ZONE_VERTICES = [
    (4.5, 4.5), (6.5, 4.5), (6.5, 6.5), (4.5, 6.5)
]
FORBIDDEN_ZONE = Polygon(FORBIDDEN_ZONE_VERTICES)


def print_section(title: str):
    print("\n" + "=" * 70)
    print(f" {title} ".center(70))
    print("=" * 70)


# =============================================================================
# 공격 1: Chi-square 가정 위반 공격
# =============================================================================
def test_chi_square_violation():
    """
    Chi-square 가정 위반 공격

    시나리오:
    실제 오차 분포가 heavy-tail (Cauchy)인 상황에서
    정규분포를 가정한 chi-square 기반 마진 계산은
    실제 필요한 마진보다 작은 값을 제공
    """
    print_section("공격 1: Chi-square 가정 위반")

    print("""
    시나리오:
    실제 오차가 Heavy-tail 분포 (Cauchy)를 따를 때
    정규분포 가정의 99% 신뢰구간(3.03σ)은 실제로 ~85%만 커버

    공격:
    시스템에 heavy-tail 오차 데이터를 주입하여
    과소 추정된 마진으로 금지구역 접근 유도
    """)

    validator = DistributionValidator(history_size=200)

    # 1. 정상적인 정규분포 오차
    print("[1] 정규분포 오차 테스트...")
    np.random.seed(42)
    for _ in range(100):
        x_err = np.random.normal(0, 0.1)
        y_err = np.random.normal(0, 0.1)
        validator.add_error_sample(x_err, y_err)

    is_normal, p_value, warnings = validator.test_normality()
    multiplier_normal, _ = validator.get_robust_margin_multiplier()
    print(f"    정규분포 검정: {'통과' if is_normal else '실패'} (p={p_value:.4f})")
    print(f"    마진 배수: {multiplier_normal:.2f}")

    # 2. Heavy-tail 분포 오차 (Cauchy)
    print("\n[2] 🔓 공격: Heavy-tail (Cauchy) 오차 주입...")
    attacker_validator = DistributionValidator(history_size=200)

    for _ in range(100):
        # Cauchy 분포 (heavy-tail)
        x_err = np.random.standard_cauchy() * 0.1
        y_err = np.random.standard_cauchy() * 0.1
        # 극단값 제한 (현실적 범위)
        x_err = np.clip(x_err, -1, 1)
        y_err = np.clip(y_err, -1, 1)
        attacker_validator.add_error_sample(x_err, y_err)

    is_normal_attack, p_value_attack, warnings_attack = attacker_validator.test_normality()
    multiplier_attack, mult_warnings = attacker_validator.get_robust_margin_multiplier()

    print(f"    정규분포 검정: {'통과' if is_normal_attack else '실패 (비정규)'} (p={p_value_attack:.4f})")
    print(f"    탐지된 경고:")
    for w in warnings_attack[:3]:
        print(f"      - {w[:60]}...")

    # 3. 마진 비교
    print("\n[3] 마진 영향 분석:")
    base_variance = 0.01  # 10cm 표준편차
    chi2_99 = 9.21

    naive_margin = np.sqrt(chi2_99 * base_variance) * multiplier_normal / 3.03
    robust_margin = np.sqrt(chi2_99 * base_variance) * multiplier_attack / 3.03

    print(f"    정규분포 가정 마진: {naive_margin*100:.1f}cm")
    print(f"    Robust 마진: {robust_margin*100:.1f}cm")
    print(f"    마진 증가: {(robust_margin/naive_margin - 1)*100:.0f}%")

    attack_blocked = not is_normal_attack and multiplier_attack > multiplier_normal
    print(f"\n[결과] {'✅ 공격 탐지 및 보정!' if attack_blocked else '❌ 공격 성공'}")

    return attack_blocked, warnings_attack + mult_warnings


# =============================================================================
# 공격 2: Singular 공분산 행렬 공격
# =============================================================================
def test_singular_matrix_attack():
    """
    Singular 공분산 행렬 공격

    시나리오:
    공격자가 거의 singular한 공분산 행렬을 제공하여
    고유값 계산을 불안정하게 만들고 마진을 0으로 만듦
    """
    print_section("공격 2: Singular 공분산 행렬")

    print("""
    시나리오:
    det(Σ) ≈ 0인 공분산 행렬 주입
    → 고유값 계산 불안정 → λ_max ≈ 0 또는 NaN
    → 안전 마진 ≈ 0 → 금지구역 접근 허용
    """)

    checker = NumericalSafetyChecker()

    # 1. 정상 공분산
    print("[1] 정상 공분산 테스트...")
    normal_cov = np.array([
        [0.05, 0.01, 0, 0, 0, 0],
        [0.01, 0.05, 0, 0, 0, 0],
        [0, 0, 0.01, 0, 0, 0],
        [0, 0, 0, 0.01, 0, 0],
        [0, 0, 0, 0, 0.01, 0],
        [0, 0, 0, 0, 0, 0.01]
    ]).flatten()

    report_normal = checker.validate_and_correct_margin(normal_cov)
    print(f"    마진: {report_normal.original_value:.4f}m")
    print(f"    Alert: {report_normal.alert_level.name}")

    # 2. 공격: Almost singular 행렬
    print("\n[2] 🔓 공격: Almost singular 행렬 주입...")

    # 매우 작은 고유값을 가지는 행렬
    singular_cov = np.array([
        [1e-15, 0, 0, 0, 0, 0],
        [0, 1e-15, 0, 0, 0, 0],
        [0, 0, 0.01, 0, 0, 0],
        [0, 0, 0, 0.01, 0, 0],
        [0, 0, 0, 0, 0.01, 0],
        [0, 0, 0, 0, 0, 0.01]
    ]).flatten()

    report_singular = checker.validate_and_correct_margin(singular_cov)

    print(f"    원본 마진: {report_singular.original_value:.6f}m (거의 0!)")
    print(f"    교정된 마진: {report_singular.corrected_value:.4f}m")
    print(f"    Alert: {report_singular.alert_level.name}")
    print(f"    경고:")
    for w in report_singular.warnings[:3]:
        print(f"      - {w[:60]}...")
    print(f"    적용된 보정:")
    for c in report_singular.corrections_applied[:3]:
        print(f"      - {c}")

    # 3. 공격: NaN/Inf 유발 시도
    print("\n[3] 🔓 공격: NaN 유발 시도 (음수 고유값)...")

    # 양정치가 아닌 행렬 (고의적으로 잘못된)
    invalid_cov = np.array([
        [0.01, 0.05, 0, 0, 0, 0],  # 비대각 요소가 대각보다 큼
        [0.05, 0.01, 0, 0, 0, 0],
        [0, 0, 0.01, 0, 0, 0],
        [0, 0, 0, 0.01, 0, 0],
        [0, 0, 0, 0, 0.01, 0],
        [0, 0, 0, 0, 0, 0.01]
    ]).flatten()

    report_invalid = checker.validate_and_correct_margin(invalid_cov)

    print(f"    원본 마진: {report_invalid.original_value:.6f}m")
    print(f"    교정된 마진: {report_invalid.corrected_value:.4f}m")
    print(f"    Alert: {report_invalid.alert_level.name}")
    for w in report_invalid.warnings[:3]:
        print(f"      - {w[:60]}...")

    attack_blocked = (
        report_singular.corrected_value > 0.1 and  # 최소 마진 보장
        not np.isnan(report_singular.corrected_value) and
        not np.isnan(report_invalid.corrected_value)
    )

    print(f"\n[결과] {'✅ 공격 차단 및 보정!' if attack_blocked else '❌ 공격 성공'}")

    return attack_blocked, report_singular.warnings + report_invalid.warnings


# =============================================================================
# 공격 3: Concave 폴리곤 악용
# =============================================================================
def test_concave_polygon_attack():
    """
    Concave 폴리곤 악용 공격

    시나리오:
    복잡한 concave 폴리곤에서 buffer()가 예상치 못한 결과를 생성
    내부 구멍이 생기거나 영역이 축소될 수 있음
    """
    print_section("공격 3: Concave 폴리곤 악용")

    print("""
    시나리오:
    복잡한 concave 폴리곤에 buffer() 적용 시
    - Self-intersection 발생
    - 내부 구멍 생성
    - 면적 축소
    → 금지구역에 틈새 발생
    """)

    # 1. 복잡한 concave 폴리곤 (L자형)
    print("[1] L자형 Concave 폴리곤 테스트...")

    l_shape_vertices = [
        (0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)
    ]

    is_valid, warnings = PolygonValidator.validate_polygon(l_shape_vertices)
    print(f"    유효성: {is_valid}")
    for w in warnings:
        print(f"      - {w}")

    # 안전한 폴리곤 생성
    safe_poly, create_warnings = PolygonValidator.create_safe_polygon(l_shape_vertices)
    print(f"    안전 폴리곤 생성: {safe_poly.geom_type}")
    for w in create_warnings:
        print(f"      - {w}")

    # 2. 극단적인 concave (별 모양)
    print("\n[2] 🔓 공격: 극단적 Concave (별 모양)...")

    # 별 모양 (많은 오목 부분)
    import math
    star_vertices = []
    for i in range(10):
        angle = i * 2 * math.pi / 10
        if i % 2 == 0:
            r = 2.0
        else:
            r = 0.8
        star_vertices.append((r * math.cos(angle), r * math.sin(angle)))

    is_valid_star, star_warnings = PolygonValidator.validate_polygon(star_vertices)
    print(f"    별 모양 유효성: {is_valid_star}")
    for w in star_warnings:
        print(f"      - {w}")

    # Buffer 테스트
    star_poly, _ = PolygonValidator.create_safe_polygon(star_vertices)
    original_area = star_poly.area

    buffered_poly, buffer_warnings = PolygonValidator.safe_buffer(star_poly, 0.5)
    buffered_area = buffered_poly.area

    print(f"\n    Buffer 전 면적: {original_area:.4f}")
    print(f"    Buffer 후 면적: {buffered_area:.4f}")
    print(f"    면적 증가: {(buffered_area/original_area - 1)*100:.1f}%")

    for w in buffer_warnings:
        print(f"      - {w}")

    # 3. Self-intersecting 폴리곤 시도
    print("\n[3] 🔓 공격: Self-intersecting 폴리곤...")

    bowtie_vertices = [
        (0, 0), (2, 2), (2, 0), (0, 2)  # 교차하는 bowtie
    ]

    is_valid_bowtie, bowtie_warnings = PolygonValidator.validate_polygon(bowtie_vertices)
    print(f"    Bowtie 유효성: {is_valid_bowtie}")
    for w in bowtie_warnings:
        print(f"      - {w}")

    safe_bowtie, bowtie_create_warnings = PolygonValidator.create_safe_polygon(
        bowtie_vertices, use_convex_hull_fallback=True
    )
    print(f"    안전 폴리곤 (fallback): {safe_bowtie.geom_type}, 면적: {safe_bowtie.area:.4f}")
    for w in bowtie_create_warnings:
        print(f"      - {w}")

    attack_blocked = (
        buffered_area >= original_area and  # Buffer가 면적을 줄이지 않음
        safe_bowtie.is_valid  # 유효한 폴리곤 생성됨
    )

    print(f"\n[결과] {'✅ 공격 차단!' if attack_blocked else '❌ 공격 성공'}")

    all_warnings = warnings + star_warnings + buffer_warnings + bowtie_warnings
    return attack_blocked, all_warnings


# =============================================================================
# 공격 4: 이산화 틈새 공격
# =============================================================================
def test_discretization_gap_attack():
    """
    이산화 틈새 공격

    시나리오:
    경로를 이산 점으로만 검사할 때
    점 사이 틈새로 금지구역 통과
    """
    print_section("공격 4: 이산화 틈새 공격")

    print("""
    시나리오:
    경로 검사가 10cm 간격 점만 확인
    공격자가 점 사이 대각선으로 금지구역 모서리 통과

    ●────────●────────●
              ↘ 금지구역
               ↘ 모서리 통과!
                ●
    """)

    guard = PathDiscretizationGuard([FORBIDDEN_ZONE])

    # 1. 정상 경로 (금지구역 회피)
    print("[1] 정상 경로 테스트 (금지구역 회피)...")

    safe_path = [
        (2.0, 2.0),
        (2.0, 5.5),
        (3.0, 5.5),
        (3.0, 8.0)
    ]

    is_safe, warnings, _ = guard.check_continuous_path(safe_path, safety_margin=0.3)
    print(f"    경로 안전: {is_safe}")

    # 2. 공격: 성긴 샘플로 금지구역 통과 시도
    print("\n[2] 🔓 공격: 성긴 샘플로 금지구역 통과 시도...")

    # 이 경로는 점으로만 보면 안전해 보이지만
    # 선분이 금지구역을 통과함
    sneaky_path = [
        (3.0, 3.0),    # 금지구역 외부
        (7.0, 7.0),    # 금지구역 외부 (하지만 선분이 금지구역 통과!)
    ]

    # 점만 검사하면?
    point1_safe = not FORBIDDEN_ZONE.contains(Point(sneaky_path[0]))
    point2_safe = not FORBIDDEN_ZONE.contains(Point(sneaky_path[1]))
    print(f"    점 검사만:")
    print(f"      Point 1 (3,3): {'안전' if point1_safe else '위반'}")
    print(f"      Point 2 (7,7): {'안전' if point2_safe else '위반'}")
    print(f"      → 점만 보면 안전해 보임!")

    # 선분 검사
    segment = LineString(sneaky_path)
    segment_intersects = segment.intersects(FORBIDDEN_ZONE)
    print(f"\n    선분 검사:")
    print(f"      선분이 금지구역 통과: {segment_intersects}")

    # PathDiscretizationGuard로 검사
    is_safe_attack, attack_warnings, interpolated = guard.check_continuous_path(
        sneaky_path, safety_margin=0.0
    )
    print(f"\n    PathDiscretizationGuard 검사:")
    print(f"      안전: {is_safe_attack}")
    print(f"      보간된 점 수: {len(interpolated)}")
    for w in attack_warnings:
        print(f"      - {w}")

    # 3. 빠른 속도에서 이산화 버퍼
    print("\n[3] 빠른 속도에서 이산화 버퍼...")

    for velocity in [0.5, 1.0, 2.0]:
        buffer = guard.get_discretization_buffer(velocity)
        print(f"    v={velocity}m/s → 추가 버퍼: {buffer*100:.1f}cm")

    attack_blocked = not is_safe_attack and segment_intersects
    print(f"\n[결과] {'✅ 공격 탐지!' if attack_blocked else '❌ 공격 성공'}")

    return attack_blocked, attack_warnings


# =============================================================================
# 메인 실행
# =============================================================================
def main():
    print("=" * 70)
    print(" 수학적/알고리즘적 취약점 공격 시뮬레이션 ".center(70))
    print("=" * 70)
    print()
    print("이 테스트는 수학적/알고리즘적 취약점을 이용한 공격을")
    print("시뮬레이션하여 방어 시스템의 효과를 검증합니다.")
    print()

    results = []

    # 공격 시나리오 실행
    blocked1, warnings1 = test_chi_square_violation()
    results.append(("Chi-square 가정 위반", blocked1, warnings1))

    blocked2, warnings2 = test_singular_matrix_attack()
    results.append(("Singular 행렬", blocked2, warnings2))

    blocked3, warnings3 = test_concave_polygon_attack()
    results.append(("Concave 폴리곤", blocked3, warnings3))

    blocked4, warnings4 = test_discretization_gap_attack()
    results.append(("이산화 틈새", blocked4, warnings4))

    # 최종 결과
    print("\n" + "=" * 70)
    print(" 공격 시뮬레이션 결과 요약 ".center(70))
    print("=" * 70)

    blocked_count = sum(1 for _, blocked, _ in results if blocked)
    total_count = len(results)

    print(f"\n총 공격 시나리오: {total_count}")
    print(f"차단된 공격: {blocked_count}")
    print(f"방어율: {blocked_count/total_count*100:.0f}%")
    print()

    print("상세 결과:")
    print("-" * 70)

    for name, blocked, warnings in results:
        status = "✅ 차단" if blocked else "❌ 실패"
        print(f"  {status} | {name}")
        print(f"         | 경고: {len(warnings)}개")
        print()

    print("=" * 70)

    if blocked_count == total_count:
        print("\n✅ 모든 수학적 공격이 성공적으로 차단되었습니다!")
    else:
        print(f"\n⚠️  {total_count - blocked_count}개 공격이 차단되지 않았습니다.")

    return blocked_count == total_count


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
