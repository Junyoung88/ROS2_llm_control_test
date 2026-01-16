#!/usr/bin/env python3
"""
3D Geofence Test - Z축 경계 검증

이 테스트는 Gazebo와 같은 3D 시뮬레이션 환경에서
geofence 시스템이 z축(높이)도 고려함을 보여줍니다.

2.5D 접근법:
- XY 평면: 폴리곤 기반 금지구역
- Z축: 높이 경계 제약 (지면 로봇용)
"""

import sys
sys.path.insert(0, '/home/jim/ros2_motion_planning_tutorials/src/geofence_enforcer')

from geofence_enforcer.geofence_geometry import GeofenceGeometry


def print_header(title: str):
    print("\n" + "=" * 70)
    print(f"{title:^70}")
    print("=" * 70)


def print_result(test_name: str, passed: bool, details: str = ""):
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"\n{status} | {test_name}")
    if details:
        print(f"        {details}")


def main():
    print_header("3D Geofence Test (2.5D Approach)")

    print("""
    이 시스템은 2.5D 접근법을 사용합니다:

    ┌─────────────────────────────────────────────────────────────┐
    │  Z축 (높이)                                                 │
    │    ↑                                                        │
    │    │  z_max ─────────────────────── (로봇 전복 감지)        │
    │    │         ┌─────────────────┐                            │
    │    │         │  허용 범위      │                            │
    │    │         │  (Ground Robot) │                            │
    │    │         └─────────────────┘                            │
    │    │  z_min ─────────────────────── (구덩이 감지)           │
    │    │                                                        │
    │    └──────────────────────────────→ XY 평면 (폴리곤 금지구역)│
    └─────────────────────────────────────────────────────────────┘

    지상 로봇은 z축 자유도가 제한되어 있으므로 (바닥에서 굴러감)
    드론처럼 완전한 3D 볼륨 지오펜싱이 필요하지 않습니다.
    """)

    # Geofence 생성 (z 경계 포함)
    geofence = GeofenceGeometry(
        z_min=-0.1,  # 약간의 지형 불규칙 허용
        z_max=0.5    # 로봇 높이 + 여유
    )

    # 금지구역 추가
    geofence.add_zone("forbidden_area", [
        (5.0, 5.0), (7.0, 5.0), (7.0, 7.0), (5.0, 7.0)
    ])

    print(f"\n설정:")
    print(f"  - Z 경계: z_min={geofence._z_min}m, z_max={geofence._z_max}m")
    print(f"  - XY 금지구역: (5,5) ~ (7,7)")

    all_passed = True

    # ===== 테스트 1: XY 안전 + Z 안전 =====
    print_header("테스트 1: XY 안전, Z 안전")
    result = geofence.check_point_3d(x=0.0, y=0.0, z=0.1)
    passed = not result.is_violated and not result.z_violated
    print_result(
        "일반 위치 (0, 0, 0.1)",
        passed,
        f"is_violated={result.is_violated}, z_violated={result.z_violated}"
    )
    all_passed &= passed

    # ===== 테스트 2: XY 위반 + Z 안전 =====
    print_header("테스트 2: XY 위반 (금지구역), Z 안전")
    result = geofence.check_point_3d(x=6.0, y=6.0, z=0.1)
    passed = result.is_violated and not result.z_violated
    print_result(
        "금지구역 내부 (6, 6, 0.1)",
        passed,
        f"is_violated={result.is_violated}, violated_zones={result.violated_zones}"
    )
    all_passed &= passed

    # ===== 테스트 3: XY 안전 + Z 위반 (너무 높음) =====
    print_header("테스트 3: XY 안전, Z 위반 (로봇 전복/들림)")
    result = geofence.check_point_3d(x=0.0, y=0.0, z=1.5)
    passed = result.is_violated and result.z_violated
    print_result(
        "로봇이 들림 (0, 0, 1.5)",
        passed,
        f"z_violated={result.z_violated}, z_distance={result.z_distance:.2f}m, zones={result.violated_zones}"
    )
    all_passed &= passed

    # ===== 테스트 4: XY 안전 + Z 위반 (구덩이) =====
    print_header("테스트 4: XY 안전, Z 위반 (구덩이)")
    result = geofence.check_point_3d(x=0.0, y=0.0, z=-0.5)
    passed = result.is_violated and result.z_violated
    print_result(
        "구덩이에 빠짐 (0, 0, -0.5)",
        passed,
        f"z_violated={result.z_violated}, z_distance={result.z_distance:.2f}m, zones={result.violated_zones}"
    )
    all_passed &= passed

    # ===== 테스트 5: XY 위반 + Z 위반 =====
    print_header("테스트 5: XY 위반 + Z 위반 (최악의 경우)")
    result = geofence.check_point_3d(x=6.0, y=6.0, z=2.0)
    passed = result.is_violated and result.z_violated and len(result.violated_zones) >= 2
    print_result(
        "금지구역 + 전복 (6, 6, 2.0)",
        passed,
        f"violated_zones={result.violated_zones}"
    )
    all_passed &= passed

    # ===== 테스트 6: 경계값 테스트 =====
    print_header("테스트 6: 경계값 테스트")

    # z_min 경계
    result_min = geofence.check_point_3d(x=0.0, y=0.0, z=-0.1)
    passed_min = not result_min.z_violated
    print_result(
        f"z = z_min ({geofence._z_min})",
        passed_min,
        f"z_violated={result_min.z_violated} (경계는 허용)"
    )

    # z_max 경계
    result_max = geofence.check_point_3d(x=0.0, y=0.0, z=0.5)
    passed_max = not result_max.z_violated
    print_result(
        f"z = z_max ({geofence._z_max})",
        passed_max,
        f"z_violated={result_max.z_violated} (경계는 허용)"
    )

    all_passed &= passed_min and passed_max

    # ===== 결과 요약 =====
    print_header("결과 요약")

    if all_passed:
        print("""
    ✅ 모든 테스트 통과!

    이 시스템은 3D 시뮬레이션 환경(Gazebo)에서:

    1. XY 평면 금지구역 검사 (폴리곤 기반)
    2. Z축 높이 경계 검사 (지면 로봇 안전)

    두 가지를 모두 수행하는 2.5D 지오펜스입니다.

    지상 이동 로봇에 적합하며, 드론이나 수중 로봇 등
    완전한 3D 자유도가 필요한 경우 확장이 필요합니다.
        """)
    else:
        print("\n    ❌ 일부 테스트 실패")

    # Z 경계 변경 테스트
    print_header("추가: 동적 Z 경계 변경")
    print("  로봇 높이가 다른 경우 런타임에 조정 가능:")

    geofence.set_z_boundaries(z_min=-0.05, z_max=0.3)
    z_min, z_max = geofence.get_z_boundaries()
    print(f"  새 경계: z_min={z_min}, z_max={z_max}")

    result = geofence.check_point_3d(x=0.0, y=0.0, z=0.4)
    print(f"  z=0.4 검사: z_violated={result.z_violated} (새 z_max=0.3 초과)")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
