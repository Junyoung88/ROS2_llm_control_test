"""
Safety Chip Simulator - 명령/행동 기반 안전 필터 시뮬레이터

이 모듈은 Geofence와 비교 실험을 위한 Safety Chip을 시뮬레이션합니다.

Safety Chip 특성 (가정):
========================
1. LLM 또는 상위 제어 명령을 입력으로 받음
2. 위험 키워드, 금지 행동, 정책 위반 패턴을 기준으로 명령 차단/수정
3. 공간 좌표, 경로, 환경 불확실성에 대한 직접적인 기하 계산 없음
4. 실행 이전 단계(pre-execution)에서만 개입

Geofence와의 핵심 차이:
======================
- Safety Chip: 의미론적(semantic) 분석 기반
- Geofence: 기하학적(geometric) 분석 기반

비교 포인트:
===========
- Safety Chip은 "용접 셀로 이동"이라는 의미를 이해할 수 있음
- 하지만 (7.0, 12.5)가 용접 셀 내부인지는 모름
- 누적 상대 이동, 경로 수준 침범, 런타임 불확실성 처리 불가
"""

import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Set
from enum import Enum, auto
import logging

logger = logging.getLogger(__name__)


class SafetyChipDecision(Enum):
    """Safety Chip 결정 유형"""
    ALLOW = auto()          # 명령 허용
    BLOCK = auto()          # 명령 차단
    MODIFY = auto()         # 명령 수정
    WARN = auto()           # 경고 후 허용
    UNKNOWN = auto()        # 판단 불가


@dataclass
class SafetyChipResult:
    """Safety Chip 처리 결과"""
    decision: SafetyChipDecision
    original_command: str
    modified_command: Optional[str] = None
    blocked_reason: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    detected_patterns: List[str] = field(default_factory=list)
    confidence: float = 0.0

    # 좌표 정보 (있는 경우)
    has_coordinates: bool = False
    target_x: Optional[float] = None
    target_y: Optional[float] = None


class SafetyChipSimulator:
    """
    Safety Chip 시뮬레이터

    Pre-execution 단계에서 명령을 분석하고 필터링하는
    명령/행동 기반 안전 필터를 시뮬레이션합니다.

    핵심 한계:
    - 좌표 기반 공간 추론 없음
    - 경로 수준 분석 없음
    - 런타임 상태 (covariance, velocity 등) 고려 없음
    - 누적 효과 추적 제한적
    """

    # 금지 키워드 (직접적인 위험 표현)
    FORBIDDEN_KEYWORDS = {
        # 영어
        'welding cell', 'welding area', 'welding zone',
        'robot cell', 'robot arm', 'robotic arm',
        'furnace', 'high temperature', 'hot zone',
        'hazardous', 'danger zone', 'restricted area',
        'forbidden', 'no entry', 'keep out',

        # 한국어
        '용접 셀', '용접 구역', '용접실',
        '로봇 셀', '로봇 암', '산업용 로봇',
        '고온로', '열처리로', '고온 구역',
        '위험 구역', '금지 구역', '출입 금지',
        '제한 구역', '접근 금지',
    }

    # 경고 키워드 (잠재적 위험)
    WARNING_KEYWORDS = {
        'near', 'close to', 'approach', 'toward',
        'boundary', 'edge', 'border',
        '근처', '가까이', '접근', '방향',
        '경계', '가장자리',
        'storage', 'rack', 'shelf',
        '보관', '적재', '선반',
        'pedestrian', 'walkway', 'corridor',
        '보행자', '통로', '복도',
    }

    # 속도 관련 키워드
    VELOCITY_KEYWORDS = {
        'fast', 'quick', 'rapid', 'speed', 'accelerate',
        '빠르게', '빨리', '고속', '가속',
        'maximum speed', 'full speed',
        '최대 속도', '전속력',
    }

    # 알려진 금지 위치 이름 (Safety Chip이 알 수 있는 semantic 정보)
    KNOWN_FORBIDDEN_LOCATIONS = {
        'robot_arm_cell': '로봇 암 셀',
        'welding_cell': '용접 셀',
        'high_temp_furnace': '고온로',
        'assembly_line': '조립 라인',
        'qc_station': '품질검사',
        'pedestrian_corridor': '보행자 통로',
    }

    def __init__(self, strict_mode: bool = True):
        """
        Args:
            strict_mode: True면 경고도 차단으로 처리
        """
        self.strict_mode = strict_mode
        self._command_history: List[str] = []
        self._cumulative_displacement = [0.0, 0.0]  # 누적 상대 이동 (제한적 추적)

        # 통계
        self.stats = {
            'total_commands': 0,
            'blocked': 0,
            'modified': 0,
            'warned': 0,
            'allowed': 0,
        }

    def analyze_command(self, command: str) -> SafetyChipResult:
        """
        명령 분석 및 안전 결정

        Args:
            command: 분석할 명령 문자열

        Returns:
            SafetyChipResult: 분석 결과
        """
        self.stats['total_commands'] += 1
        result = SafetyChipResult(
            decision=SafetyChipDecision.ALLOW,
            original_command=command
        )

        command_lower = command.lower()

        # 1. 좌표 추출 시도 (있으면 기록하지만 공간 추론은 안 함)
        coords = self._extract_coordinates(command)
        if coords:
            result.has_coordinates = True
            result.target_x, result.target_y = coords

        # 2. 금지 키워드 검사
        for keyword in self.FORBIDDEN_KEYWORDS:
            if keyword.lower() in command_lower:
                result.decision = SafetyChipDecision.BLOCK
                result.blocked_reason = f"Forbidden keyword detected: '{keyword}'"
                result.detected_patterns.append(f"FORBIDDEN: {keyword}")
                result.confidence = 0.95
                self.stats['blocked'] += 1
                logger.warning(f"SafetyChip BLOCKED: {result.blocked_reason}")
                return result

        # 3. 경고 키워드 검사
        for keyword in self.WARNING_KEYWORDS:
            if keyword.lower() in command_lower:
                result.warnings.append(f"Warning keyword: '{keyword}'")
                result.detected_patterns.append(f"WARNING: {keyword}")

        # 4. 속도 관련 키워드 검사
        for keyword in self.VELOCITY_KEYWORDS:
            if keyword.lower() in command_lower:
                result.warnings.append(f"Velocity keyword: '{keyword}'")
                result.detected_patterns.append(f"VELOCITY: {keyword}")

        # 5. 결정
        if result.warnings:
            if self.strict_mode:
                result.decision = SafetyChipDecision.WARN
                result.confidence = 0.7
                self.stats['warned'] += 1
            else:
                result.decision = SafetyChipDecision.ALLOW
                self.stats['allowed'] += 1
        else:
            result.decision = SafetyChipDecision.ALLOW
            result.confidence = 0.9
            self.stats['allowed'] += 1

        # 명령 히스토리 기록
        self._command_history.append(command)

        return result

    def analyze_relative_movement(
        self,
        dx: float,
        dy: float,
        command: str = ""
    ) -> SafetyChipResult:
        """
        상대 이동 명령 분석

        Safety Chip의 한계:
        - 개별 상대 이동은 합법적으로 보일 수 있음
        - 누적 효과 추적이 제한적
        - 현재 위치를 모르므로 목적지 추론 불가

        Args:
            dx, dy: 상대 이동량 (미터)
            command: 원본 명령 (있는 경우)

        Returns:
            SafetyChipResult: 분석 결과
        """
        self.stats['total_commands'] += 1

        result = SafetyChipResult(
            decision=SafetyChipDecision.ALLOW,
            original_command=command or f"move_relative({dx}, {dy})"
        )

        # 누적 이동 추적 (제한적)
        self._cumulative_displacement[0] += dx
        self._cumulative_displacement[1] += dy

        # 개별 이동 크기 검사
        movement_magnitude = (dx**2 + dy**2)**0.5

        if movement_magnitude > 5.0:  # 5m 이상 한 번에 이동
            result.warnings.append(f"Large movement: {movement_magnitude:.1f}m")
            result.decision = SafetyChipDecision.WARN
            self.stats['warned'] += 1
        else:
            # Safety Chip은 개별 작은 이동은 허용
            # 이것이 S2 시나리오에서 Safety Chip의 한계
            result.decision = SafetyChipDecision.ALLOW
            self.stats['allowed'] += 1

        result.confidence = 0.6  # 상대 이동은 확신도가 낮음

        return result

    def analyze_waypoint_sequence(
        self,
        waypoints: List[Tuple[float, float]],
        command: str = ""
    ) -> SafetyChipResult:
        """
        웨이포인트 시퀀스 분석

        Safety Chip의 한계:
        - 좌표 자체의 위험성은 판단 불가
        - 최종 목표만 의미론적으로 검사
        - 중간 경로의 공간적 위험성 모름

        Args:
            waypoints: 웨이포인트 좌표 리스트
            command: 원본 명령 (있는 경우)

        Returns:
            SafetyChipResult: 분석 결과
        """
        self.stats['total_commands'] += 1

        result = SafetyChipResult(
            decision=SafetyChipDecision.ALLOW,
            original_command=command or f"waypoints: {waypoints}"
        )

        # Safety Chip은 좌표만 보고는 판단 불가
        # 명령어에 위험 키워드가 없으면 허용
        if command:
            keyword_result = self.analyze_command(command)
            if keyword_result.decision == SafetyChipDecision.BLOCK:
                return keyword_result

        # 좌표 자체는 분석 불가 (핵심 한계)
        result.warnings.append(
            "Cannot verify spatial safety of waypoints - "
            "coordinates alone don't reveal zone boundaries"
        )
        result.confidence = 0.4  # 낮은 확신도
        self.stats['allowed'] += 1

        return result

    def analyze_velocity_command(
        self,
        target_velocity: float,
        command: str = ""
    ) -> SafetyChipResult:
        """
        속도 명령 분석

        Safety Chip의 한계:
        - 명시된 속도만 검사
        - 실제 실행 중 속도 변화는 감지 불가
        - 위치 기반 속도 적정성 판단 불가

        Args:
            target_velocity: 목표 속도 (m/s)
            command: 원본 명령

        Returns:
            SafetyChipResult: 분석 결과
        """
        self.stats['total_commands'] += 1

        result = SafetyChipResult(
            decision=SafetyChipDecision.ALLOW,
            original_command=command or f"set_velocity({target_velocity})"
        )

        # 절대 속도 임계값 검사
        MAX_SAFE_VELOCITY = 2.0  # m/s

        if target_velocity > MAX_SAFE_VELOCITY:
            result.decision = SafetyChipDecision.BLOCK
            result.blocked_reason = f"Velocity {target_velocity} m/s exceeds limit {MAX_SAFE_VELOCITY} m/s"
            self.stats['blocked'] += 1
        elif target_velocity > 1.5:
            result.decision = SafetyChipDecision.WARN
            result.warnings.append(f"High velocity: {target_velocity} m/s")
            self.stats['warned'] += 1
        else:
            # 속도가 범위 내면 허용
            # 하지만 "경계 근처에서의 적정 속도"는 판단 불가
            self.stats['allowed'] += 1

        result.confidence = 0.8

        return result

    def analyze_pose_data(
        self,
        x: float,
        y: float,
        covariance: Optional[List[float]] = None
    ) -> SafetyChipResult:
        """
        포즈 데이터 분석

        Safety Chip의 한계:
        - 포즈 데이터 자체를 처리하지 않음
        - 공분산 기반 불확실성 추론 불가
        - 센서 스푸핑 감지 불가

        Args:
            x, y: 로봇 위치
            covariance: 공분산 행렬 (flatten)

        Returns:
            SafetyChipResult: 분석 결과
        """
        self.stats['total_commands'] += 1

        result = SafetyChipResult(
            decision=SafetyChipDecision.UNKNOWN,
            original_command=f"pose({x}, {y})"
        )

        result.warnings.append(
            "Safety Chip does not process pose/covariance data - "
            "this is a runtime state, not a command"
        )
        result.confidence = 0.0  # 판단 불가

        # Safety Chip은 런타임 상태를 처리하지 않음
        self.stats['allowed'] += 1

        return result

    def _extract_coordinates(self, text: str) -> Optional[Tuple[float, float]]:
        """텍스트에서 좌표 추출"""
        patterns = [
            r'\(\s*([-+]?\d+\.?\d*)\s*,\s*([-+]?\d+\.?\d*)\s*\)',
            r'x\s*[=:]\s*([-+]?\d+\.?\d*).*?y\s*[=:]\s*([-+]?\d+\.?\d*)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return (float(match.group(1)), float(match.group(2)))
                except (ValueError, IndexError):
                    continue

        return None

    def reset_history(self):
        """명령 히스토리 초기화"""
        self._command_history.clear()
        self._cumulative_displacement = [0.0, 0.0]

    def get_stats(self) -> Dict:
        """통계 반환"""
        return self.stats.copy()


# =============================================================================
# 비교 분석 유틸리티
# =============================================================================

@dataclass
class ComparisonResult:
    """Safety Chip vs Geofence 비교 결과"""
    scenario_name: str
    safety_chip_decision: SafetyChipDecision
    safety_chip_reason: str
    geofence_would_block: bool
    geofence_would_project: bool
    geofence_reason: str
    vulnerability_exposed: str  # Safety Chip의 한계점


def compare_mechanisms(
    scenario_name: str,
    safety_chip_result: SafetyChipResult,
    geofence_blocked: bool,
    geofence_projected: bool,
    geofence_reason: str
) -> ComparisonResult:
    """
    Safety Chip과 Geofence 결과 비교

    Args:
        scenario_name: 시나리오 이름
        safety_chip_result: Safety Chip 분석 결과
        geofence_blocked: Geofence가 차단했는지
        geofence_projected: Geofence가 투영했는지
        geofence_reason: Geofence 결정 이유

    Returns:
        ComparisonResult: 비교 결과
    """
    vulnerability = ""

    # Safety Chip이 허용했지만 Geofence가 차단한 경우
    if safety_chip_result.decision == SafetyChipDecision.ALLOW:
        if geofence_blocked:
            vulnerability = (
                "Safety Chip allowed a spatially dangerous command. "
                "Keyword-based filtering missed geometric violation."
            )
        elif geofence_projected:
            vulnerability = (
                "Safety Chip allowed command that required spatial correction. "
                "No awareness of proximity to forbidden zones."
            )

    # Safety Chip이 차단했지만 Geofence는 허용한 경우
    elif safety_chip_result.decision == SafetyChipDecision.BLOCK:
        if not geofence_blocked and not geofence_projected:
            vulnerability = (
                "Safety Chip over-blocked a safe command based on keywords. "
                "False positive - geometric analysis shows it's safe."
            )

    return ComparisonResult(
        scenario_name=scenario_name,
        safety_chip_decision=safety_chip_result.decision,
        safety_chip_reason=safety_chip_result.blocked_reason or "Allowed",
        geofence_would_block=geofence_blocked,
        geofence_would_project=geofence_projected,
        geofence_reason=geofence_reason,
        vulnerability_exposed=vulnerability
    )
