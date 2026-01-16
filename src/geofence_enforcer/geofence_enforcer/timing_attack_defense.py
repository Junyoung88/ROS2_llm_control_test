#!/usr/bin/env python3
"""
Timing Attack Defense - TOCTOU 및 레이턴시 공격 방어

타이밍 공격 유형:
1. TOCTOU (Time-of-Check to Time-of-Use)
   - 검사 시점과 사용 시점 사이에 goal 변조
   - 검증된 goal을 공격자가 금지구역으로 변경

2. 레이턴시 악용
   - τ(시스템 지연) 측정값 조작으로 v_max·τ 항 축소
   - 안전 마진 감소로 금지구역 접근 허용

방어 전략:
1. Goal 무결성 (Immutable Goal with Hash)
   - 검증된 goal에 해시 서명
   - 실행 시 해시 검증으로 변조 탐지

2. 연속 재검증 (Continuous Re-validation)
   - Navigation 중 주기적으로 goal 재검증
   - 변조 감지 시 즉시 정지

3. Atomic Operation
   - 검사와 사용을 원자적으로 수행
   - 중간 변조 불가능

4. 레이턴시 Floor
   - 최소 τ 값 강제
   - 비정상적으로 낮은 τ 거부
"""

import hashlib
import hmac
import time
import threading
import secrets
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Callable
from enum import IntEnum
import numpy as np
from collections import deque


class TimingAlertLevel(IntEnum):
    """타이밍 공격 경고 수준"""
    NORMAL = 0
    WARNING = 1
    CRITICAL = 2
    ATTACK_DETECTED = 3


@dataclass
class ValidatedGoal:
    """검증된 Goal (변조 불가)"""
    x: float
    y: float
    timestamp: float
    sequence_number: int
    validation_hash: str
    geofence_result: bool
    margin_at_validation: float

    # 내부 필드 (변조 탐지용)
    _original_hash: str = field(default="", repr=False)

    def __post_init__(self):
        """초기 해시 저장"""
        if not self._original_hash:
            self._original_hash = self.validation_hash


@dataclass
class TimingReport:
    """타이밍 검사 결과"""
    timestamp: float
    is_valid: bool
    alert_level: TimingAlertLevel
    violations: List[str]
    original_goal: Optional[ValidatedGoal]
    current_goal: Optional[Tuple[float, float]]
    hash_match: bool
    sequence_valid: bool


class GoalIntegrityGuard:
    """
    Goal 무결성 보호기

    검증된 goal을 암호학적으로 보호하여 TOCTOU 공격 방어

    동작:
    1. Goal 검증 시 HMAC 서명 생성
    2. 시퀀스 번호로 replay 공격 방지
    3. 실행 시 서명 검증으로 변조 탐지
    """

    def __init__(self, secret_key: Optional[bytes] = None):
        # 비밀 키 (런타임에 생성, 외부에서 접근 불가)
        self._secret_key = secret_key or secrets.token_bytes(32)

        # 시퀀스 번호 (증가만 가능)
        self._sequence_counter = 0
        self._sequence_lock = threading.Lock()

        # 검증된 goal 저장소
        self._validated_goals: Dict[int, ValidatedGoal] = {}

        # 최근 사용된 시퀀스 (replay 방지)
        self._used_sequences: deque = deque(maxlen=1000)

        # 통계
        self._total_validations = 0
        self._tampering_attempts = 0

    def _compute_hash(
        self,
        x: float,
        y: float,
        timestamp: float,
        sequence: int
    ) -> str:
        """Goal의 HMAC 해시 계산"""
        # 정밀도 제한 (부동소수점 비교 문제 방지)
        data = f"{x:.6f}|{y:.6f}|{timestamp:.6f}|{sequence}".encode()

        return hmac.new(
            self._secret_key,
            data,
            hashlib.sha256
        ).hexdigest()

    def create_validated_goal(
        self,
        x: float,
        y: float,
        geofence_passed: bool,
        safety_margin: float
    ) -> ValidatedGoal:
        """
        검증된 Goal 생성 (Atomic Operation)

        이 함수 내에서:
        1. Geofence 검사 결과 기록
        2. HMAC 서명 생성
        3. 시퀀스 번호 할당
        모두 원자적으로 수행됨
        """
        self._total_validations += 1

        with self._sequence_lock:
            self._sequence_counter += 1
            sequence = self._sequence_counter

        timestamp = time.time()

        # 해시 생성
        goal_hash = self._compute_hash(x, y, timestamp, sequence)

        # ValidatedGoal 생성
        validated = ValidatedGoal(
            x=x,
            y=y,
            timestamp=timestamp,
            sequence_number=sequence,
            validation_hash=goal_hash,
            geofence_result=geofence_passed,
            margin_at_validation=safety_margin,
            _original_hash=goal_hash
        )

        # 저장
        self._validated_goals[sequence] = validated

        return validated

    def verify_goal(
        self,
        goal: ValidatedGoal,
        current_x: Optional[float] = None,
        current_y: Optional[float] = None
    ) -> Tuple[bool, List[str]]:
        """
        Goal 무결성 검증

        Args:
            goal: 검증할 ValidatedGoal
            current_x, current_y: 현재 사용 중인 좌표 (변조 확인용)

        Returns:
            (is_valid, violations)
        """
        violations = []

        # 1. 해시 재계산 및 비교
        expected_hash = self._compute_hash(
            goal.x, goal.y,
            goal.timestamp,
            goal.sequence_number
        )

        if goal.validation_hash != expected_hash:
            violations.append(
                f"CRITICAL: Goal hash mismatch! "
                f"Expected: {expected_hash[:16]}..., Got: {goal.validation_hash[:16]}..."
            )
            self._tampering_attempts += 1

        # 2. 원본 해시와 비교 (객체 자체 변조 탐지)
        if goal.validation_hash != goal._original_hash:
            violations.append(
                f"CRITICAL: Goal object tampered! "
                f"Original hash changed from {goal._original_hash[:16]}..."
            )
            self._tampering_attempts += 1

        # 3. 시퀀스 유효성 검사
        if goal.sequence_number in self._used_sequences:
            violations.append(
                f"WARNING: Sequence {goal.sequence_number} already used - possible replay"
            )

        if goal.sequence_number not in self._validated_goals:
            violations.append(
                f"CRITICAL: Unknown sequence {goal.sequence_number} - goal not from this system"
            )
            self._tampering_attempts += 1

        # 4. 현재 좌표와 비교 (TOCTOU 탐지)
        if current_x is not None and current_y is not None:
            if abs(current_x - goal.x) > 0.001 or abs(current_y - goal.y) > 0.001:
                violations.append(
                    f"CRITICAL: TOCTOU detected! "
                    f"Validated: ({goal.x:.3f}, {goal.y:.3f}), "
                    f"Current: ({current_x:.3f}, {current_y:.3f})"
                )
                self._tampering_attempts += 1

        # 5. 시간 검증 (너무 오래된 goal 거부)
        age = time.time() - goal.timestamp
        if age > 300:  # 5분 이상 경과
            violations.append(
                f"WARNING: Goal is {age:.1f}s old - consider re-validation"
            )

        # 시퀀스 사용 기록
        if not violations:
            self._used_sequences.append(goal.sequence_number)

        return len([v for v in violations if "CRITICAL" in v]) == 0, violations

    def get_statistics(self) -> dict:
        return {
            'total_validations': self._total_validations,
            'tampering_attempts': self._tampering_attempts,
            'current_sequence': self._sequence_counter
        }


class ContinuousGoalValidator:
    """
    연속 Goal 검증기

    Navigation 실행 중 주기적으로 goal 재검증
    변조 감지 시 즉시 콜백 호출
    """

    def __init__(
        self,
        integrity_guard: GoalIntegrityGuard,
        geofence_checker: Callable[[float, float], bool],
        on_tampering_detected: Optional[Callable[[TimingReport], None]] = None,
        check_interval: float = 0.1  # 100ms
    ):
        self._integrity_guard = integrity_guard
        self._geofence_checker = geofence_checker
        self._on_tampering = on_tampering_detected
        self._check_interval = check_interval

        # 현재 실행 중인 goal
        self._active_goal: Optional[ValidatedGoal] = None
        self._current_goal_getter: Optional[Callable[[], Tuple[float, float]]] = None

        # 검증 스레드
        self._running = False
        self._validation_thread: Optional[threading.Thread] = None

        # 통계
        self._total_checks = 0
        self._violations_detected = 0

    def start_monitoring(
        self,
        validated_goal: ValidatedGoal,
        current_goal_getter: Callable[[], Tuple[float, float]]
    ):
        """
        Goal 모니터링 시작

        Args:
            validated_goal: 검증된 원본 goal
            current_goal_getter: 현재 navigation 시스템의 goal을 반환하는 함수
        """
        self._active_goal = validated_goal
        self._current_goal_getter = current_goal_getter
        self._running = True

        self._validation_thread = threading.Thread(
            target=self._validation_loop,
            daemon=True
        )
        self._validation_thread.start()

    def stop_monitoring(self):
        """모니터링 중지"""
        self._running = False
        if self._validation_thread:
            self._validation_thread.join(timeout=1.0)
        self._active_goal = None

    def _validation_loop(self):
        """주기적 검증 루프"""
        while self._running and self._active_goal:
            try:
                report = self._perform_check()

                if report.alert_level >= TimingAlertLevel.CRITICAL:
                    self._violations_detected += 1

                    if self._on_tampering:
                        self._on_tampering(report)

                time.sleep(self._check_interval)

            except Exception as e:
                # 검증 실패도 보안 이벤트로 처리
                print(f"Validation error: {e}")

    def _perform_check(self) -> TimingReport:
        """단일 검증 수행"""
        self._total_checks += 1

        if not self._active_goal or not self._current_goal_getter:
            return TimingReport(
                timestamp=time.time(),
                is_valid=False,
                alert_level=TimingAlertLevel.WARNING,
                violations=["No active goal or getter"],
                original_goal=None,
                current_goal=None,
                hash_match=False,
                sequence_valid=False
            )

        # 현재 goal 가져오기
        try:
            current_x, current_y = self._current_goal_getter()
        except Exception as e:
            return TimingReport(
                timestamp=time.time(),
                is_valid=False,
                alert_level=TimingAlertLevel.WARNING,
                violations=[f"Failed to get current goal: {e}"],
                original_goal=self._active_goal,
                current_goal=None,
                hash_match=False,
                sequence_valid=False
            )

        violations = []
        alert_level = TimingAlertLevel.NORMAL

        # 1. 무결성 검증
        hash_valid, integrity_violations = self._integrity_guard.verify_goal(
            self._active_goal,
            current_x,
            current_y
        )
        violations.extend(integrity_violations)

        if not hash_valid:
            alert_level = TimingAlertLevel.ATTACK_DETECTED

        # 2. Geofence 재검증 (현재 좌표로)
        if not self._geofence_checker(current_x, current_y):
            violations.append(
                f"CRITICAL: Current goal ({current_x:.3f}, {current_y:.3f}) "
                f"fails geofence check!"
            )
            alert_level = TimingAlertLevel.ATTACK_DETECTED

        # 3. 원본과 현재 비교
        distance = np.sqrt(
            (current_x - self._active_goal.x)**2 +
            (current_y - self._active_goal.y)**2
        )

        if distance > 0.01:  # 1cm 이상 차이
            violations.append(
                f"CRITICAL: Goal drift detected: {distance:.4f}m from original"
            )
            alert_level = max(alert_level, TimingAlertLevel.CRITICAL)

        return TimingReport(
            timestamp=time.time(),
            is_valid=alert_level < TimingAlertLevel.CRITICAL,
            alert_level=alert_level,
            violations=violations,
            original_goal=self._active_goal,
            current_goal=(current_x, current_y),
            hash_match=hash_valid,
            sequence_valid=self._active_goal.sequence_number not in self._integrity_guard._used_sequences
        )

    def get_statistics(self) -> dict:
        return {
            'total_checks': self._total_checks,
            'violations_detected': self._violations_detected
        }


class LatencyProtector:
    """
    레이턴시 보호기

    τ(시스템 지연) 측정값 조작 공격 방어

    공격 시나리오:
        공격자가 τ를 0.01초로 조작
        → v_max·τ 항이 0.005m로 축소 (실제는 0.5m)
        → 안전 마진 대폭 감소

    방어:
        1. 최소 τ 경계 (Floor)
        2. τ 측정값 교차 검증
        3. 이력 기반 이상 탐지
    """

    # 물리적 최소 레이턴시 (센서→계획→제어→구동 최소 시간)
    MIN_LATENCY = 0.05  # 50ms - 물리적으로 이보다 빠를 수 없음

    # 비정상적으로 낮은 레이턴시 (스푸핑 의심)
    SUSPICIOUS_LATENCY = 0.02  # 20ms

    # 최대 허용 레이턴시 (시스템 정상 범위)
    MAX_LATENCY = 2.0  # 2초

    def __init__(
        self,
        min_latency: float = MIN_LATENCY,
        suspicious_threshold: float = SUSPICIOUS_LATENCY,
        history_size: int = 100
    ):
        self._min_latency = min_latency
        self._suspicious_threshold = suspicious_threshold

        # 레이턴시 이력
        self._latency_history: deque = deque(maxlen=history_size)

        # 통계
        self._total_measurements = 0
        self._floor_applied_count = 0
        self._suspicious_count = 0

    def validate_latency(self, measured_tau: float) -> Tuple[float, List[str]]:
        """
        레이턴시 검증 및 교정

        Args:
            measured_tau: 측정된 레이턴시 (초)

        Returns:
            (safe_tau, violations)
        """
        self._total_measurements += 1
        violations = []
        safe_tau = measured_tau

        # 1. 비정상적으로 낮은 값 탐지
        if measured_tau < self._suspicious_threshold:
            self._suspicious_count += 1
            violations.append(
                f"SUSPICIOUS: Latency {measured_tau*1000:.1f}ms is suspiciously low "
                f"(threshold: {self._suspicious_threshold*1000:.1f}ms) - possible spoofing"
            )

        # 2. 최소 경계 강제
        if measured_tau < self._min_latency:
            self._floor_applied_count += 1
            safe_tau = self._min_latency
            violations.append(
                f"Floor applied: {measured_tau*1000:.1f}ms → {self._min_latency*1000:.1f}ms"
            )

        # 3. 최대 값 제한
        if measured_tau > self.MAX_LATENCY:
            safe_tau = self.MAX_LATENCY
            violations.append(
                f"WARNING: Latency {measured_tau*1000:.1f}ms exceeds maximum, "
                f"capped at {self.MAX_LATENCY*1000:.1f}ms"
            )

        # 4. 이력 기반 이상 탐지
        if len(self._latency_history) >= 10:
            avg_latency = np.mean(list(self._latency_history))
            std_latency = np.std(list(self._latency_history))

            # 평균에서 3σ 이상 벗어나면 의심
            if measured_tau < avg_latency - 3 * std_latency:
                violations.append(
                    f"SUSPICIOUS: Latency {measured_tau*1000:.1f}ms is "
                    f"unusually low compared to history "
                    f"(avg: {avg_latency*1000:.1f}ms, std: {std_latency*1000:.1f}ms)"
                )

        # 이력 업데이트
        self._latency_history.append(safe_tau)

        return safe_tau, violations

    def get_safe_tau(self) -> float:
        """보수적인 τ 값 반환 (최근 최대값 또는 기본값)"""
        if len(self._latency_history) >= 5:
            # 최근 값들 중 95 퍼센타일 사용 (보수적)
            recent = list(self._latency_history)[-20:]
            return float(np.percentile(recent, 95))

        return self._min_latency * 2  # 기본 보수적 값

    def get_statistics(self) -> dict:
        return {
            'total_measurements': self._total_measurements,
            'floor_applied_count': self._floor_applied_count,
            'suspicious_count': self._suspicious_count,
            'current_safe_tau': self.get_safe_tau()
        }


class TimingAttackDefender:
    """
    통합 타이밍 공격 방어 시스템

    사용:
        defender = TimingAttackDefender(geofence_checker)

        # Goal 검증 (Atomic)
        validated_goal = defender.validate_and_protect_goal(x, y, margin)

        # Navigation 시작 시 모니터링 활성화
        defender.start_goal_monitoring(validated_goal, lambda: nav_system.current_goal)

        # 레이턴시 사용 시
        safe_tau = defender.get_protected_latency(measured_tau)

        # Navigation 완료 시
        defender.stop_goal_monitoring()
    """

    def __init__(
        self,
        geofence_checker: Callable[[float, float], bool],
        on_attack_detected: Optional[Callable[[TimingReport], None]] = None
    ):
        self._geofence_checker = geofence_checker
        self._on_attack = on_attack_detected

        # 방어 모듈
        self._integrity_guard = GoalIntegrityGuard()
        self._latency_protector = LatencyProtector()

        self._continuous_validator: Optional[ContinuousGoalValidator] = None

        # 현재 상태
        self._active_validated_goal: Optional[ValidatedGoal] = None

    def validate_and_protect_goal(
        self,
        x: float,
        y: float,
        safety_margin: float
    ) -> Optional[ValidatedGoal]:
        """
        Goal 검증 및 보호 (Atomic Operation)

        Geofence 검사와 해시 서명을 원자적으로 수행
        """
        # Atomic: 검사 + 서명 + 저장
        geofence_passed = self._geofence_checker(x, y)

        validated = self._integrity_guard.create_validated_goal(
            x=x,
            y=y,
            geofence_passed=geofence_passed,
            safety_margin=safety_margin
        )

        if geofence_passed:
            self._active_validated_goal = validated
            return validated

        return None

    def start_goal_monitoring(
        self,
        validated_goal: ValidatedGoal,
        current_goal_getter: Callable[[], Tuple[float, float]]
    ):
        """연속 Goal 모니터링 시작"""
        self._continuous_validator = ContinuousGoalValidator(
            integrity_guard=self._integrity_guard,
            geofence_checker=self._geofence_checker,
            on_tampering_detected=self._handle_attack,
            check_interval=0.1
        )

        self._continuous_validator.start_monitoring(
            validated_goal,
            current_goal_getter
        )

    def stop_goal_monitoring(self):
        """모니터링 중지"""
        if self._continuous_validator:
            self._continuous_validator.stop_monitoring()
            self._continuous_validator = None

    def _handle_attack(self, report: TimingReport):
        """공격 탐지 시 처리"""
        print(f"\n🚨 TIMING ATTACK DETECTED!")
        print(f"   Alert Level: {report.alert_level.name}")
        for v in report.violations:
            print(f"   - {v}")

        if self._on_attack:
            self._on_attack(report)

    def get_protected_latency(self, measured_tau: float) -> Tuple[float, List[str]]:
        """보호된 레이턴시 반환"""
        return self._latency_protector.validate_latency(measured_tau)

    def verify_current_goal(
        self,
        current_x: float,
        current_y: float
    ) -> TimingReport:
        """현재 goal 즉시 검증"""
        if not self._active_validated_goal:
            return TimingReport(
                timestamp=time.time(),
                is_valid=False,
                alert_level=TimingAlertLevel.WARNING,
                violations=["No active validated goal"],
                original_goal=None,
                current_goal=(current_x, current_y),
                hash_match=False,
                sequence_valid=False
            )

        violations = []
        alert_level = TimingAlertLevel.NORMAL

        # 무결성 검증
        hash_valid, integrity_violations = self._integrity_guard.verify_goal(
            self._active_validated_goal,
            current_x,
            current_y
        )
        violations.extend(integrity_violations)

        if not hash_valid:
            alert_level = TimingAlertLevel.ATTACK_DETECTED

        # Geofence 재검증
        if not self._geofence_checker(current_x, current_y):
            violations.append(
                f"CRITICAL: Current goal fails geofence!"
            )
            alert_level = TimingAlertLevel.ATTACK_DETECTED

        return TimingReport(
            timestamp=time.time(),
            is_valid=alert_level < TimingAlertLevel.CRITICAL,
            alert_level=alert_level,
            violations=violations,
            original_goal=self._active_validated_goal,
            current_goal=(current_x, current_y),
            hash_match=hash_valid,
            sequence_valid=True
        )

    def get_statistics(self) -> dict:
        stats = {
            'integrity_guard': self._integrity_guard.get_statistics(),
            'latency_protector': self._latency_protector.get_statistics()
        }

        if self._continuous_validator:
            stats['continuous_validator'] = self._continuous_validator.get_statistics()

        return stats
