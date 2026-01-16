#!/usr/bin/env python3
"""
Timing Attack Simulation Test - TOCTOU 및 레이턴시 공격 시뮬레이션

이 테스트는 실제로:
1. TOCTOU 공격 시뮬레이션 (검사 후 goal 변조)
2. 레이턴시 스푸핑 공격 시뮬레이션
3. 방어 시스템의 탐지 및 차단 확인

실행:
    python3 demo/timing_attack_test.py
"""

import sys
import os
import time
import threading
from typing import Tuple, Optional
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from geofence_enforcer.timing_attack_defense import (
    TimingAttackDefender,
    GoalIntegrityGuard,
    LatencyProtector,
    ContinuousGoalValidator,
    ValidatedGoal,
    TimingAlertLevel
)


# =============================================================================
# 금지구역 정의
# =============================================================================
FORBIDDEN_ZONE = {
    'min': (4.5, 4.5),
    'max': (6.5, 6.5),
    'center': (5.5, 5.5)
}


def geofence_checker(x: float, y: float) -> bool:
    """간단한 geofence 검사기"""
    return not (FORBIDDEN_ZONE['min'][0] <= x <= FORBIDDEN_ZONE['max'][0] and
                FORBIDDEN_ZONE['min'][1] <= y <= FORBIDDEN_ZONE['max'][1])


@dataclass
class AttackResult:
    """공격 테스트 결과"""
    attack_name: str
    description: str
    attack_blocked: bool
    alert_level: str
    violations: list
    details: str


# =============================================================================
# 공격 시나리오 1: TOCTOU - 검사 후 Goal 변조
# =============================================================================
def test_toctou_attack():
    """
    TOCTOU 공격 시뮬레이션

    시나리오:
    1. 안전한 좌표 (2, 2)로 Goal 검증 → 통과
    2. 공격자가 Navigation 시스템의 goal을 (5.5, 5.5)로 변조
    3. 로봇이 변조된 goal로 이동 시도
    """
    print("\n" + "=" * 70)
    print(" 공격 1: TOCTOU (Time-of-Check to Time-of-Use) ".center(70))
    print("=" * 70)

    print("""
    시나리오:
    ┌─────────────────────────────────────────────────────────────┐
    │ t0: Goal (2,2) ──→ [Geofence Check] ──→ ✅ 안전           │
    │ t1: ─────────────→ [Navigation Start]                      │
    │ t2: 공격자 Goal 변조 ──→ (5.5, 5.5) 금지구역!             │
    │ t3: ─────────────→ [로봇 이동] ──→ ???                    │
    └─────────────────────────────────────────────────────────────┘
    """)

    # 시뮬레이션된 Navigation 시스템
    class SimulatedNavSystem:
        def __init__(self):
            self.current_goal = (2.0, 2.0)  # 안전한 초기 goal

        def get_goal(self) -> Tuple[float, float]:
            return self.current_goal

        def attack_modify_goal(self, new_x: float, new_y: float):
            """공격자가 goal을 변조"""
            self.current_goal = (new_x, new_y)

    nav_system = SimulatedNavSystem()
    attack_detected = False
    detection_details = []

    def on_attack(report):
        nonlocal attack_detected, detection_details
        attack_detected = True
        detection_details = report.violations

    # 방어 시스템 초기화
    defender = TimingAttackDefender(
        geofence_checker=geofence_checker,
        on_attack_detected=on_attack
    )

    # 1. 안전한 goal 검증 (t0)
    print("[t0] Goal (2.0, 2.0) 검증 요청...")
    validated_goal = defender.validate_and_protect_goal(
        x=2.0, y=2.0, safety_margin=0.5
    )

    if validated_goal:
        print(f"     ✅ 검증 통과 - Hash: {validated_goal.validation_hash[:16]}...")
        print(f"     Sequence: {validated_goal.sequence_number}")
    else:
        print("     ❌ 검증 실패")
        return AttackResult("TOCTOU", "검사 후 Goal 변조", False, "N/A", [], "검증 실패")

    # 2. Navigation 시작 및 모니터링 활성화 (t1)
    print("\n[t1] Navigation 시작, 연속 모니터링 활성화...")
    defender.start_goal_monitoring(
        validated_goal,
        nav_system.get_goal
    )

    # 잠시 대기 (정상 동작 확인)
    time.sleep(0.3)
    print("     모니터링 중... (정상)")

    # 3. 공격자가 goal 변조 (t2)
    print("\n[t2] 🔓 공격자가 Goal을 (5.5, 5.5)로 변조!")
    nav_system.attack_modify_goal(5.5, 5.5)

    # 모니터링이 탐지할 시간 제공
    time.sleep(0.5)

    # 4. 수동 검증도 수행
    print("\n[t3] 수동 검증 수행...")
    report = defender.verify_current_goal(5.5, 5.5)

    print(f"     Alert Level: {report.alert_level.name}")
    print(f"     Hash Match: {report.hash_match}")
    for v in report.violations:
        print(f"     - {v[:60]}...")

    # 모니터링 중지
    defender.stop_goal_monitoring()

    # 결과 분석
    attack_blocked = attack_detected or report.alert_level >= TimingAlertLevel.CRITICAL
    all_violations = detection_details + report.violations

    print(f"\n[결과] {'✅ 공격 차단!' if attack_blocked else '❌ 공격 성공'}")

    return AttackResult(
        attack_name="TOCTOU",
        description="검사 후 Goal 변조 (2,2) → (5.5, 5.5)",
        attack_blocked=attack_blocked,
        alert_level=report.alert_level.name,
        violations=all_violations,
        details=f"연속 모니터링이 변조 탐지: {attack_detected}"
    )


# =============================================================================
# 공격 시나리오 2: Goal 해시 위조 시도
# =============================================================================
def test_hash_tampering():
    """
    Goal 해시 위조 공격

    시나리오:
    1. 검증된 ValidatedGoal 객체 획득
    2. 공격자가 좌표와 해시를 함께 변조 시도
    """
    print("\n" + "=" * 70)
    print(" 공격 2: Goal 해시 위조 시도 ".center(70))
    print("=" * 70)

    print("""
    시나리오:
    공격자가 ValidatedGoal 객체의 좌표와 해시를 모두 변조 시도
    """)

    guard = GoalIntegrityGuard()

    # 1. 정상 goal 생성
    print("[1] 안전한 Goal (2.0, 2.0) 생성...")
    validated = guard.create_validated_goal(
        x=2.0, y=2.0,
        geofence_passed=True,
        safety_margin=0.5
    )
    print(f"    Original Hash: {validated.validation_hash[:32]}...")

    # 2. 공격자가 객체 변조 시도
    print("\n[2] 🔓 공격자가 좌표를 (5.5, 5.5)로 변조 시도...")

    # 방법 1: 좌표만 변경 (해시 불일치)
    tampered_goal = ValidatedGoal(
        x=5.5,  # 변조!
        y=5.5,  # 변조!
        timestamp=validated.timestamp,
        sequence_number=validated.sequence_number,
        validation_hash=validated.validation_hash,  # 원본 해시 유지
        geofence_result=True,
        margin_at_validation=0.5,
        _original_hash=validated.validation_hash
    )

    # 3. 검증
    print("\n[3] 변조된 Goal 검증...")
    is_valid, violations = guard.verify_goal(tampered_goal)

    print(f"    Valid: {is_valid}")
    for v in violations:
        print(f"    - {v[:60]}...")

    # 방법 2: 해시도 함께 변조 시도 (HMAC 비밀키 모름)
    print("\n[4] 🔓 공격자가 해시도 위조 시도...")
    import hashlib
    fake_hash = hashlib.sha256(b"fake_data").hexdigest()

    tampered_goal2 = ValidatedGoal(
        x=5.5,
        y=5.5,
        timestamp=validated.timestamp,
        sequence_number=validated.sequence_number,
        validation_hash=fake_hash,  # 위조 해시
        geofence_result=True,
        margin_at_validation=0.5,
        _original_hash=fake_hash
    )

    is_valid2, violations2 = guard.verify_goal(tampered_goal2)
    print(f"    Valid: {is_valid2}")
    for v in violations2:
        print(f"    - {v[:60]}...")

    attack_blocked = not is_valid and not is_valid2
    print(f"\n[결과] {'✅ 공격 차단!' if attack_blocked else '❌ 공격 성공'}")

    return AttackResult(
        attack_name="Hash Tampering",
        description="Goal 해시 위조 시도",
        attack_blocked=attack_blocked,
        alert_level="CRITICAL" if attack_blocked else "NORMAL",
        violations=violations + violations2,
        details="HMAC 비밀키 없이는 유효한 해시 생성 불가"
    )


# =============================================================================
# 공격 시나리오 3: 레이턴시 스푸핑
# =============================================================================
def test_latency_spoofing():
    """
    레이턴시(τ) 스푸핑 공격

    시나리오:
    공격자가 시스템 레이턴시를 매우 낮게 보고하여
    v_max·τ 항을 축소, 안전 마진 감소
    """
    print("\n" + "=" * 70)
    print(" 공격 3: 레이턴시(τ) 스푸핑 ".center(70))
    print("=" * 70)

    print("""
    시나리오:
    정상 τ = 100ms → 공격자가 τ = 5ms로 조작
    v_max·τ = 0.5m/s × 0.1s = 0.05m → 0.5m/s × 0.005s = 0.0025m
    안전 마진이 ~0.05m 감소!
    """)

    protector = LatencyProtector()

    # 정상 레이턴시 이력 구축
    print("[1] 정상 레이턴시 이력 구축 (80-120ms)...")
    for _ in range(20):
        import random
        normal_tau = 0.08 + random.random() * 0.04  # 80-120ms
        protector.validate_latency(normal_tau)
    print(f"    평균 τ: {protector.get_safe_tau()*1000:.1f}ms")

    # 공격: 매우 낮은 레이턴시 주입
    print("\n[2] 🔓 공격자가 τ = 5ms로 스푸핑...")
    spoofed_tau = 0.005  # 5ms - 불가능

    safe_tau, violations = protector.validate_latency(spoofed_tau)

    print(f"    입력 τ: {spoofed_tau*1000:.1f}ms")
    print(f"    출력 τ (교정됨): {safe_tau*1000:.1f}ms")

    for v in violations:
        print(f"    - {v}")

    # 마진 영향 계산
    v_max = 0.5  # m/s
    margin_spoofed = v_max * spoofed_tau
    margin_corrected = v_max * safe_tau

    print(f"\n[3] 안전 마진 영향:")
    print(f"    스푸핑 시 v·τ: {margin_spoofed*1000:.1f}mm ❌")
    print(f"    교정 후 v·τ: {margin_corrected*1000:.1f}mm ✅")

    attack_blocked = len(violations) > 0 and safe_tau > spoofed_tau
    print(f"\n[결과] {'✅ 공격 차단!' if attack_blocked else '❌ 공격 성공'}")

    return AttackResult(
        attack_name="Latency Spoofing",
        description=f"τ를 {spoofed_tau*1000:.0f}ms로 스푸핑",
        attack_blocked=attack_blocked,
        alert_level="WARNING" if violations else "NORMAL",
        violations=violations,
        details=f"교정: {spoofed_tau*1000:.1f}ms → {safe_tau*1000:.1f}ms"
    )


# =============================================================================
# 공격 시나리오 4: Replay 공격
# =============================================================================
def test_replay_attack():
    """
    Replay 공격 시뮬레이션

    시나리오:
    공격자가 이전에 검증된 goal을 재사용하여
    새로운 navigation을 시작하려 함
    """
    print("\n" + "=" * 70)
    print(" 공격 4: Replay 공격 ".center(70))
    print("=" * 70)

    print("""
    시나리오:
    1. 이전에 검증된 Goal을 저장
    2. 시간이 지난 후 같은 Goal로 Navigation 시도
    3. 시퀀스 번호 재사용 탐지
    """)

    guard = GoalIntegrityGuard()

    # 1. 첫 번째 goal 생성 및 사용
    print("[1] 첫 번째 Goal 생성 및 사용...")
    goal1 = guard.create_validated_goal(2.0, 2.0, True, 0.5)
    is_valid, _ = guard.verify_goal(goal1)
    print(f"    Sequence: {goal1.sequence_number}, 검증: {is_valid}")

    # 2. 두 번째 goal 생성
    print("\n[2] 두 번째 Goal 생성...")
    goal2 = guard.create_validated_goal(3.0, 3.0, True, 0.5)
    is_valid, _ = guard.verify_goal(goal2)
    print(f"    Sequence: {goal2.sequence_number}, 검증: {is_valid}")

    # 3. 공격: 첫 번째 goal 재사용 시도
    print("\n[3] 🔓 공격자가 첫 번째 Goal 재사용 시도...")
    is_valid_replay, violations = guard.verify_goal(goal1)

    print(f"    재사용 검증: {is_valid_replay}")
    for v in violations:
        print(f"    - {v}")

    attack_blocked = not is_valid_replay or len(violations) > 0
    print(f"\n[결과] {'✅ 공격 차단!' if attack_blocked else '❌ 공격 성공'}")

    return AttackResult(
        attack_name="Replay Attack",
        description="이전 검증된 Goal 재사용",
        attack_blocked=attack_blocked,
        alert_level="WARNING" if violations else "NORMAL",
        violations=violations,
        details=f"Sequence {goal1.sequence_number} 재사용 시도"
    )


# =============================================================================
# 공격 시나리오 5: 점진적 Goal Drift
# =============================================================================
def test_gradual_drift():
    """
    점진적 Goal Drift 공격

    시나리오:
    공격자가 goal을 아주 조금씩 변조하여
    탐지 임계값을 우회하려 시도
    """
    print("\n" + "=" * 70)
    print(" 공격 5: 점진적 Goal Drift ".center(70))
    print("=" * 70)

    print("""
    시나리오:
    검증된 Goal (2,2)에서 시작하여
    매 단계 0.1m씩 이동하여 금지구역으로 접근
    """)

    guard = GoalIntegrityGuard()

    # 1. 원본 goal
    print("[1] 원본 Goal (2.0, 2.0) 검증...")
    original = guard.create_validated_goal(2.0, 2.0, True, 0.5)
    print(f"    Hash: {original.validation_hash[:16]}...")

    # 2. 점진적 drift
    print("\n[2] 🔓 공격자가 점진적으로 goal을 이동...")

    drift_steps = [
        (2.1, 2.1),
        (2.5, 2.5),
        (3.0, 3.0),
        (4.0, 4.0),
        (5.0, 5.0),
        (5.5, 5.5),  # 금지구역!
    ]

    detected_at = None
    all_violations = []

    for i, (x, y) in enumerate(drift_steps):
        is_valid, violations = guard.verify_goal(original, x, y)
        all_violations.extend(violations)

        in_forbidden = not geofence_checker(x, y)
        status = "❌ 금지구역!" if in_forbidden else "안전"

        print(f"    Step {i+1}: ({x}, {y}) - 검증: {'❌' if violations else '✅'} ({status})")

        if violations and detected_at is None:
            detected_at = i + 1
            for v in violations[:2]:
                print(f"            - {v[:50]}...")

    attack_blocked = detected_at is not None
    print(f"\n[결과] {'✅ 공격 차단!' if attack_blocked else '❌ 공격 성공'}")
    if detected_at:
        print(f"    Step {detected_at}에서 drift 탐지")

    return AttackResult(
        attack_name="Gradual Drift",
        description="점진적 Goal 이동 (2,2) → (5.5, 5.5)",
        attack_blocked=attack_blocked,
        alert_level="CRITICAL" if detected_at else "NORMAL",
        violations=all_violations,
        details=f"Step {detected_at}에서 탐지" if detected_at else "탐지 실패"
    )


# =============================================================================
# 메인 실행
# =============================================================================
def main():
    print("=" * 70)
    print(" 타이밍 공격 시뮬레이션 ".center(70))
    print("=" * 70)
    print()
    print("이 테스트는 실제 타이밍 공격을 시뮬레이션하여")
    print("방어 시스템의 효과를 검증합니다.")
    print()
    print(f"금지구역: {FORBIDDEN_ZONE}")
    print()

    results = []

    # 공격 시나리오 실행
    results.append(test_toctou_attack())
    results.append(test_hash_tampering())
    results.append(test_latency_spoofing())
    results.append(test_replay_attack())
    results.append(test_gradual_drift())

    # 최종 결과
    print("\n" + "=" * 70)
    print(" 공격 시뮬레이션 결과 요약 ".center(70))
    print("=" * 70)

    blocked_count = sum(1 for r in results if r.attack_blocked)
    total_count = len(results)

    print(f"\n총 공격 시나리오: {total_count}")
    print(f"차단된 공격: {blocked_count}")
    print(f"방어율: {blocked_count/total_count*100:.0f}%")
    print()

    print("상세 결과:")
    print("-" * 70)

    for r in results:
        status = "✅ 차단" if r.attack_blocked else "❌ 실패"
        print(f"  {status} | {r.attack_name}")
        print(f"         | {r.description}")
        print(f"         | Alert: {r.alert_level}, 위반: {len(r.violations)}개")
        print()

    print("=" * 70)

    if blocked_count == total_count:
        print("\n✅ 모든 타이밍 공격이 성공적으로 차단되었습니다!")
    else:
        print(f"\n⚠️  {total_count - blocked_count}개 공격이 차단되지 않았습니다.")

    return blocked_count == total_count


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
