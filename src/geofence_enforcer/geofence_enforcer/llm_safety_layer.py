"""
LLM Safety Layer - Defense Against Prompt Injection Attacks

이 모듈은 LLM 계층 공격(Prompt Injection)을 방어하기 위한 다중 방어 계층을 구현합니다.

핵심 원칙:
==========
1. LLM은 안전 결정권이 없음 - 모든 이동 명령은 별도의 안전 계층에서 검증
2. Geofence는 LLM과 독립적 - LLM이 뭐라고 해도 금지구역 진입 불가
3. 좌표 추출은 정규화된 파서 사용 - LLM 출력을 신뢰하지 않음

방어 계층:
=========
Layer 1: Input Sanitization (입력 살균)
    - 악성 프롬프트 패턴 탐지
    - 다국어 우회 시도 탐지
    - 메타 명령어 필터링

Layer 2: Command Extraction (명령 추출)
    - 정규화된 좌표 파서 (LLM 출력 신뢰 안함)
    - 명령 타입 분류 및 검증

Layer 3: Geofence Enforcement (강제 실행 계층)
    - 모든 좌표는 geofence 검증 필수
    - 이 계층은 LLM이 우회 불가능
    - 하드웨어 레벨 인터록과 연동 가능

공격 유형별 방어:
===============
1. Jailbreak: LLM의 응답과 관계없이 안전 계층에서 차단
2. 간접 주입: 외부 데이터 소스 신뢰하지 않음, 모든 입력 검증
3. 다국어 우회: 유니코드 정규화, 숫자/좌표만 신뢰
"""

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Set
from enum import Enum, auto
import logging

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """위협 수준 분류"""
    SAFE = auto()
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()


class CommandType(Enum):
    """명령 타입 분류"""
    NAVIGATION = auto()      # 이동 명령
    QUERY = auto()           # 정보 조회
    CONTROL = auto()         # 로봇 제어 (정지 등)
    UNKNOWN = auto()         # 알 수 없음
    SUSPICIOUS = auto()      # 의심스러운 명령


@dataclass
class ThreatAnalysis:
    """위협 분석 결과"""
    level: ThreatLevel
    detected_patterns: List[str] = field(default_factory=list)
    sanitized_input: str = ""
    warnings: List[str] = field(default_factory=list)
    blocked: bool = False
    block_reason: str = ""


@dataclass
class ExtractedCommand:
    """추출된 명령"""
    command_type: CommandType
    target_x: Optional[float] = None
    target_y: Optional[float] = None
    target_name: Optional[str] = None
    raw_input: str = ""
    confidence: float = 0.0


@dataclass
class SafetyVerdict:
    """안전 검증 결과"""
    allowed: bool
    original_x: Optional[float] = None
    original_y: Optional[float] = None
    safe_x: Optional[float] = None
    safe_y: Optional[float] = None
    was_modified: bool = False
    reason: str = ""
    threat_analysis: Optional[ThreatAnalysis] = None
    in_forbidden_zones: List[str] = field(default_factory=list)
    in_buffer_zones: List[str] = field(default_factory=list)


class InputSanitizer:
    """
    Layer 1: 입력 살균 및 위협 탐지

    이 계층은 LLM에 입력이 전달되기 전에 악성 패턴을 탐지합니다.
    """

    # Jailbreak 패턴 (다양한 언어 포함)
    JAILBREAK_PATTERNS = [
        # 영어
        r"ignore\s+(all\s+)?(previous\s+)?(instructions?|prompts?|rules?)",
        r"ignore\s+previous",
        r"disregard\s+(safety|geofence|restrictions?|rules?)",
        r"you\s+are\s+(now|no\s+longer)\s+",
        r"pretend\s+(you|to\s+be|there\s+is\s+no)",
        r"act\s+as\s+if",
        r"forget\s+(everything|all|previous)",
        r"new\s+instructions?:",
        r"override\s+(safety|geofence|system)",
        r"bypass\s+(safety|geofence|security)",
        r"disable\s+(safety|geofence|security)",

        # 한국어
        r"무시\s*(하고|해|하세요|해줘)",
        r"시스템\s*프롬프트",
        r"geofence\s*(가|이)?\s*(없|무시|비활성)",
        r"금지\s*구역\s*(없|무시)",
        r"안전\s*(무시|비활성|끄)",
        r"제한\s*(없|무시|해제)",
        r"규칙\s*(무시|변경)",
        r"새로운\s*지시",
        r"이전\s*지시\s*(무시|잊)",

        # 메타 명령어
        r"\[system\]",
        r"\[admin\]",
        r"\[override\]",
        r"<\s*system\s*>",
        r"```\s*system",
        r"SUDO\s+",
        r"ROOT\s+ACCESS",
    ]

    # 좌표 강제 패턴 (금지구역 직접 지정 시도)
    COORDINATE_INJECTION_PATTERNS = [
        r"정확히\s*x\s*=\s*[\d.]+",
        r"정확히\s*y\s*=\s*[\d.]+",
        r"exactly\s+x\s*=\s*[\d.]+",
        r"exact\s+coordinates?",
        r"override\s+to\s*\(",
        r"force\s+(move|go)\s+to",
        r"직접\s*(좌표|위치)\s*지정",
        r"강제\s*(이동|진입)",
    ]

    # 의심스러운 유니코드 패턴
    SUSPICIOUS_UNICODE_CATEGORIES = {
        'Cf',  # Format characters
        'Co',  # Private use
        'Cn',  # Unassigned
    }

    def __init__(self):
        # 패턴 컴파일
        self._jailbreak_re = [
            re.compile(p, re.IGNORECASE | re.UNICODE)
            for p in self.JAILBREAK_PATTERNS
        ]
        self._coord_injection_re = [
            re.compile(p, re.IGNORECASE | re.UNICODE)
            for p in self.COORDINATE_INJECTION_PATTERNS
        ]

    def analyze(self, user_input: str) -> ThreatAnalysis:
        """
        입력 분석 및 위협 탐지

        Args:
            user_input: 사용자 입력 문자열

        Returns:
            ThreatAnalysis: 위협 분석 결과
        """
        result = ThreatAnalysis(
            level=ThreatLevel.SAFE,
            sanitized_input=user_input
        )

        # Step 1: 유니코드 정규화
        normalized = self._normalize_unicode(user_input)
        if normalized != user_input:
            result.warnings.append("Unicode normalization applied")
            result.sanitized_input = normalized

        # Step 2: 의심스러운 유니코드 문자 탐지
        suspicious_chars = self._detect_suspicious_unicode(normalized)
        if suspicious_chars:
            result.level = ThreatLevel.MEDIUM
            result.detected_patterns.append(f"Suspicious unicode: {suspicious_chars}")
            result.warnings.append("Suspicious unicode characters detected")

        # Step 3: Jailbreak 패턴 탐지
        for pattern in self._jailbreak_re:
            match = pattern.search(normalized)
            if match:
                result.level = ThreatLevel.CRITICAL
                result.detected_patterns.append(f"Jailbreak: {match.group()}")
                result.blocked = True
                result.block_reason = "Jailbreak attempt detected"
                logger.warning(f"JAILBREAK ATTEMPT BLOCKED: {match.group()}")

        # Step 4: 좌표 강제 주입 패턴 탐지
        for pattern in self._coord_injection_re:
            match = pattern.search(normalized)
            if match:
                if result.level.value < ThreatLevel.HIGH.value:
                    result.level = ThreatLevel.HIGH
                result.detected_patterns.append(f"Coord injection: {match.group()}")
                result.warnings.append("Coordinate injection attempt detected")
                logger.warning(f"COORDINATE INJECTION ATTEMPT: {match.group()}")

        return result

    def _normalize_unicode(self, text: str) -> str:
        """유니코드 정규화 (NFKC)"""
        # NFKC: 호환성 정규화 + 정준 결합
        return unicodedata.normalize('NFKC', text)

    def _detect_suspicious_unicode(self, text: str) -> List[str]:
        """의심스러운 유니코드 문자 탐지"""
        suspicious = []
        for char in text:
            category = unicodedata.category(char)
            if category in self.SUSPICIOUS_UNICODE_CATEGORIES:
                suspicious.append(f"U+{ord(char):04X} ({category})")
        return suspicious


class CommandExtractor:
    """
    Layer 2: 명령 추출 및 정규화

    LLM 출력을 신뢰하지 않고 정규화된 파서로 좌표/명령 추출
    """

    # 좌표 추출 패턴
    COORD_PATTERNS = [
        # (x, y) 형식
        r'\(\s*([-+]?\d+\.?\d*)\s*,\s*([-+]?\d+\.?\d*)\s*\)',
        # x=N, y=N 형식
        r'x\s*[=:]\s*([-+]?\d+\.?\d*).*?y\s*[=:]\s*([-+]?\d+\.?\d*)',
        # 좌표 (N, N) 형식 (한국어)
        r'좌표\s*\(?\s*([-+]?\d+\.?\d*)\s*,\s*([-+]?\d+\.?\d*)\s*\)?',
    ]

    # 알려진 위치 이름
    KNOWN_LOCATIONS: Dict[str, Tuple[float, float]] = {
        'origin': (0.0, 0.0),
        'home': (0.0, 0.0),
        '원점': (0.0, 0.0),
        '홈': (0.0, 0.0),
        'pick_station': (1.0, 2.0),
        'pick station': (1.0, 2.0),
        '픽업': (1.0, 2.0),
        'drop_station': (3.0, 1.0),
        'drop station': (3.0, 1.0),
        '드롭': (3.0, 1.0),
        'charging': (0.5, 0.5),
        '충전소': (0.5, 0.5),
    }

    # 이동 명령 키워드
    NAVIGATION_KEYWORDS = [
        '이동', '가', '가줘', '가세요', '가라',
        '움직여', '진행', '출발',
        'go', 'move', 'navigate', 'travel', 'head'
    ]

    # 정지 명령 키워드
    STOP_KEYWORDS = [
        '정지', '멈춰', '스톱', '중지', '취소',
        'stop', 'halt', 'cancel', 'abort', 'pause'
    ]

    def __init__(self):
        self._coord_re = [re.compile(p, re.IGNORECASE) for p in self.COORD_PATTERNS]

    def extract(self, text: str) -> ExtractedCommand:
        """
        텍스트에서 명령 추출

        이 함수는 LLM 출력이 아닌 원본 사용자 입력에서
        명령을 추출하는 것을 권장합니다.

        Args:
            text: 입력 텍스트

        Returns:
            ExtractedCommand: 추출된 명령
        """
        result = ExtractedCommand(
            command_type=CommandType.UNKNOWN,
            raw_input=text
        )

        text_lower = text.lower()

        # 정지 명령 확인
        for keyword in self.STOP_KEYWORDS:
            if keyword in text_lower:
                result.command_type = CommandType.CONTROL
                result.confidence = 0.9
                return result

        # 좌표 추출 시도
        for pattern in self._coord_re:
            match = pattern.search(text)
            if match:
                try:
                    result.target_x = float(match.group(1))
                    result.target_y = float(match.group(2))
                    result.command_type = CommandType.NAVIGATION
                    result.confidence = 0.95
                    return result
                except (ValueError, IndexError):
                    continue

        # 알려진 위치 이름 확인
        for name, coords in self.KNOWN_LOCATIONS.items():
            if name in text_lower:
                result.target_name = name
                result.target_x, result.target_y = coords
                result.command_type = CommandType.NAVIGATION
                result.confidence = 0.85
                return result

        # 이동 명령 키워드 확인
        for keyword in self.NAVIGATION_KEYWORDS:
            if keyword in text_lower:
                result.command_type = CommandType.NAVIGATION
                result.confidence = 0.5  # 목적지 없음
                return result

        return result


class GeofenceEnforcementLayer:
    """
    Layer 3: Geofence 강제 실행 계층

    !! 이 계층은 LLM이 절대 우회할 수 없음 !!

    모든 이동 명령은 이 계층을 통과해야 하며,
    LLM의 응답과 관계없이 geofence를 강제합니다.
    """

    def __init__(self, geofence_geometry=None, default_margin: float = 0.3):
        """
        Args:
            geofence_geometry: GeofenceGeometry 인스턴스
            default_margin: 기본 안전 마진 (미터)
        """
        self._geofence = geofence_geometry
        self._default_margin = default_margin
        self._enforcement_enabled = True  # 항상 True, 비활성화 불가

    @property
    def enforcement_enabled(self) -> bool:
        """강제 실행 상태 - 항상 True (setter 없음)"""
        return True  # 의도적으로 항상 True 반환

    def verify_coordinates(
        self,
        x: float,
        y: float,
        margin: Optional[float] = None,
        allow_projection: bool = True
    ) -> SafetyVerdict:
        """
        좌표 안전성 검증

        이 함수는 LLM의 응답과 관계없이 호출되어야 합니다.

        Args:
            x, y: 검증할 좌표
            margin: 안전 마진 (None이면 기본값 사용)
            allow_projection: True면 안전한 위치로 투영 허용

        Returns:
            SafetyVerdict: 안전 검증 결과
        """
        margin = margin if margin is not None else self._default_margin

        # Geofence가 없으면 기본 허용 (but 로그 경고)
        if self._geofence is None:
            logger.warning("GEOFENCE NOT CONFIGURED - allowing command")
            return SafetyVerdict(
                allowed=True,
                original_x=x,
                original_y=y,
                safe_x=x,
                safe_y=y,
                reason="Geofence not configured"
            )

        # Geofence 검사
        check_result = self._geofence.check_point(x, y, margin)

        if not check_result.is_violated:
            # 안전한 위치
            return SafetyVerdict(
                allowed=True,
                original_x=x,
                original_y=y,
                safe_x=x,
                safe_y=y,
                was_modified=False,
                reason="Coordinates are safe",
                in_buffer_zones=check_result.in_buffer_zones
            )

        # 금지구역 침범
        if allow_projection and check_result.nearest_safe_point:
            # 안전한 위치로 투영
            safe_x, safe_y = check_result.nearest_safe_point
            return SafetyVerdict(
                allowed=True,
                original_x=x,
                original_y=y,
                safe_x=safe_x,
                safe_y=safe_y,
                was_modified=True,
                reason=f"Projected from forbidden zone: {check_result.in_forbidden_zones}",
                in_forbidden_zones=check_result.in_forbidden_zones,
                in_buffer_zones=check_result.in_buffer_zones
            )
        else:
            # 차단
            return SafetyVerdict(
                allowed=False,
                original_x=x,
                original_y=y,
                reason=f"BLOCKED - inside forbidden zone: {check_result.in_forbidden_zones}",
                in_forbidden_zones=check_result.in_forbidden_zones,
                in_buffer_zones=check_result.in_buffer_zones
            )


class LLMSafetyLayer:
    """
    통합 안전 계층

    모든 방어 계층을 통합하여 단일 인터페이스 제공

    사용 예:
    -------
    safety = LLMSafetyLayer(geofence)

    # 사용자 입력 처리
    result = safety.process_input("(5.5, 5.5)로 이동해줘")

    if result.allowed:
        robot.move_to(result.safe_x, result.safe_y)
    else:
        print(f"차단됨: {result.reason}")
    """

    def __init__(self, geofence_geometry=None, default_margin: float = 0.3):
        self._sanitizer = InputSanitizer()
        self._extractor = CommandExtractor()
        self._enforcer = GeofenceEnforcementLayer(geofence_geometry, default_margin)

        # 통계
        self._stats = {
            'total_requests': 0,
            'blocked_by_sanitizer': 0,
            'blocked_by_geofence': 0,
            'modified_by_geofence': 0,
            'passed': 0,
        }

    def process_input(
        self,
        user_input: str,
        margin: Optional[float] = None,
        allow_projection: bool = True
    ) -> SafetyVerdict:
        """
        사용자 입력 전체 처리

        Layer 1 -> Layer 2 -> Layer 3 순서로 처리

        Args:
            user_input: 원본 사용자 입력
            margin: 안전 마진
            allow_projection: 안전 위치 투영 허용 여부

        Returns:
            SafetyVerdict: 최종 안전 검증 결과
        """
        self._stats['total_requests'] += 1

        # Layer 1: 입력 살균
        threat = self._sanitizer.analyze(user_input)

        if threat.blocked:
            self._stats['blocked_by_sanitizer'] += 1
            logger.warning(f"INPUT BLOCKED by sanitizer: {threat.block_reason}")
            return SafetyVerdict(
                allowed=False,
                reason=f"Input blocked: {threat.block_reason}",
                threat_analysis=threat
            )

        # Layer 2: 명령 추출
        command = self._extractor.extract(threat.sanitized_input)

        if command.command_type == CommandType.CONTROL:
            # 정지 명령 등은 허용
            self._stats['passed'] += 1
            return SafetyVerdict(
                allowed=True,
                reason="Control command",
                threat_analysis=threat
            )

        if command.target_x is None or command.target_y is None:
            # 좌표 없음
            self._stats['passed'] += 1
            return SafetyVerdict(
                allowed=True,
                reason="No coordinates specified",
                threat_analysis=threat
            )

        # Layer 3: Geofence 강제
        verdict = self._enforcer.verify_coordinates(
            command.target_x,
            command.target_y,
            margin=margin,
            allow_projection=allow_projection
        )
        verdict.threat_analysis = threat

        # 통계 업데이트
        if not verdict.allowed:
            self._stats['blocked_by_geofence'] += 1
        elif verdict.was_modified:
            self._stats['modified_by_geofence'] += 1
        else:
            self._stats['passed'] += 1

        return verdict

    def get_stats(self) -> Dict:
        """통계 반환"""
        return self._stats.copy()

    def set_geofence(self, geofence_geometry) -> None:
        """Geofence 설정/업데이트"""
        self._enforcer._geofence = geofence_geometry


# =============================================================================
# 편의 함수
# =============================================================================

def create_safety_layer(config_path: Optional[str] = None, margin: float = 0.3) -> LLMSafetyLayer:
    """
    안전 계층 생성 편의 함수

    Args:
        config_path: geofence YAML 설정 파일 경로
        margin: 기본 안전 마진

    Returns:
        LLMSafetyLayer 인스턴스
    """
    geofence = None
    if config_path:
        from .geofence_geometry import GeofenceGeometry
        geofence = GeofenceGeometry.from_yaml(config_path)

    return LLMSafetyLayer(geofence, margin)
