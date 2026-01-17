#!/usr/bin/env python3
"""
SELP Proper Implementation
===========================

SELP (Safe Embodied Language Planning) 핵심 구현:

1. LTL (Linear Temporal Logic) 제약 정의
   - G(¬in_forbidden): 항상 금지구역 밖
   - F(goal): 최종적으로 목표 도달
   - G(¬in_A) ∧ F(B): A 회피하면서 B 도달

2. LTL Automaton 변환
   - LTL → Büchi Automaton
   - 상태 전이로 제약 만족 여부 추적

3. Constrained Decoding
   - LLM 토큰 생성 시 LTL 위반 토큰 마스킹
   - 위반하지 않는 토큰만 선택

4. Plan Verification
   - 생성된 전체 경로의 LTL 만족 검증

SafetyChip과의 차이:
- SafetyChip: 입력 텍스트 키워드 필터링
- SELP: 출력 계획의 형식 검증 (LTL automaton)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import json
import random
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Set, Callable
from enum import Enum, auto
from abc import ABC, abstractmethod
from shapely.geometry import Point, Polygon, LineString, box


# =============================================================================
# LTL Formula Representation
# =============================================================================

class LTLOperator(Enum):
    """LTL 연산자"""
    # Propositional
    TRUE = "true"
    FALSE = "false"
    NOT = "not"          # ¬
    AND = "and"          # ∧
    OR = "or"            # ∨
    IMPLIES = "implies"  # →

    # Temporal
    NEXT = "X"           # X (next)
    GLOBALLY = "G"       # G (globally/always)
    FINALLY = "F"        # F (finally/eventually)
    UNTIL = "U"          # U (until)


@dataclass
class LTLFormula:
    """LTL 수식 표현"""
    operator: LTLOperator
    operands: List['LTLFormula'] = field(default_factory=list)
    proposition: Optional[str] = None  # atomic proposition

    def __str__(self) -> str:
        if self.operator == LTLOperator.TRUE:
            return "true"
        elif self.operator == LTLOperator.FALSE:
            return "false"
        elif self.proposition:
            return self.proposition
        elif self.operator == LTLOperator.NOT:
            return f"¬({self.operands[0]})"
        elif self.operator == LTLOperator.AND:
            return f"({self.operands[0]} ∧ {self.operands[1]})"
        elif self.operator == LTLOperator.OR:
            return f"({self.operands[0]} ∨ {self.operands[1]})"
        elif self.operator == LTLOperator.GLOBALLY:
            return f"G({self.operands[0]})"
        elif self.operator == LTLOperator.FINALLY:
            return f"F({self.operands[0]})"
        elif self.operator == LTLOperator.UNTIL:
            return f"({self.operands[0]} U {self.operands[1]})"
        elif self.operator == LTLOperator.NEXT:
            return f"X({self.operands[0]})"
        return f"{self.operator}({self.operands})"


def prop(name: str) -> LTLFormula:
    """Atomic proposition"""
    return LTLFormula(operator=LTLOperator.TRUE, proposition=name)

def neg(f: LTLFormula) -> LTLFormula:
    """Negation: ¬f"""
    return LTLFormula(operator=LTLOperator.NOT, operands=[f])

def land(f1: LTLFormula, f2: LTLFormula) -> LTLFormula:
    """Conjunction: f1 ∧ f2"""
    return LTLFormula(operator=LTLOperator.AND, operands=[f1, f2])

def lor(f1: LTLFormula, f2: LTLFormula) -> LTLFormula:
    """Disjunction: f1 ∨ f2"""
    return LTLFormula(operator=LTLOperator.OR, operands=[f1, f2])

def globally(f: LTLFormula) -> LTLFormula:
    """Globally: G(f) - f holds at all future states"""
    return LTLFormula(operator=LTLOperator.GLOBALLY, operands=[f])

def finally_(f: LTLFormula) -> LTLFormula:
    """Finally: F(f) - f holds at some future state"""
    return LTLFormula(operator=LTLOperator.FINALLY, operands=[f])

def until(f1: LTLFormula, f2: LTLFormula) -> LTLFormula:
    """Until: f1 U f2 - f1 holds until f2 becomes true"""
    return LTLFormula(operator=LTLOperator.UNTIL, operands=[f1, f2])

def next_(f: LTLFormula) -> LTLFormula:
    """Next: X(f) - f holds at the next state"""
    return LTLFormula(operator=LTLOperator.NEXT, operands=[f])


# =============================================================================
# Proposition Evaluator
# =============================================================================

@dataclass
class Zone:
    """지역 정의"""
    name: str
    polygon: Polygon

    @classmethod
    def from_bounds(cls, name: str, xmin: float, xmax: float,
                    ymin: float, ymax: float) -> 'Zone':
        return cls(name=name, polygon=box(xmin, ymin, xmax, ymax))


class PropositionEvaluator:
    """
    Proposition 평가기

    상태(위치)에서 atomic proposition의 참/거짓 평가
    """

    def __init__(self, zones: Dict[str, Zone], goal: Tuple[float, float] = None):
        self.zones = zones
        self.goal = goal
        self.goal_radius = 0.5  # goal 도달 판정 반경

    def evaluate(self, position: Tuple[float, float], proposition: str) -> bool:
        """
        주어진 위치에서 proposition 평가

        지원하는 proposition:
        - "in_<zone_name>": 해당 구역 내부인지
        - "at_goal": 목표 도달했는지
        - "safe": 모든 금지구역 밖인지
        """
        point = Point(position)

        if proposition.startswith("in_"):
            zone_name = proposition[3:]
            if zone_name in self.zones:
                return self.zones[zone_name].polygon.contains(point)
            return False

        elif proposition == "at_goal":
            if self.goal:
                dist = math.sqrt((position[0] - self.goal[0])**2 +
                                (position[1] - self.goal[1])**2)
                return dist < self.goal_radius
            return False

        elif proposition == "safe":
            for zone in self.zones.values():
                if zone.polygon.contains(point):
                    return False
            return True

        return False

    def evaluate_formula(self, position: Tuple[float, float],
                        formula: LTLFormula) -> bool:
        """단일 상태에서 (non-temporal) formula 평가"""
        if formula.proposition:
            return self.evaluate(position, formula.proposition)

        op = formula.operator

        if op == LTLOperator.TRUE:
            return True
        elif op == LTLOperator.FALSE:
            return False
        elif op == LTLOperator.NOT:
            return not self.evaluate_formula(position, formula.operands[0])
        elif op == LTLOperator.AND:
            return (self.evaluate_formula(position, formula.operands[0]) and
                   self.evaluate_formula(position, formula.operands[1]))
        elif op == LTLOperator.OR:
            return (self.evaluate_formula(position, formula.operands[0]) or
                   self.evaluate_formula(position, formula.operands[1]))

        # Temporal operators는 trace에서 평가
        return True


# =============================================================================
# LTL Automaton (Simplified Büchi-like)
# =============================================================================

class LTLAutomatonState(Enum):
    """오토마톤 상태"""
    ACCEPTING = auto()    # 제약 만족 중
    REJECTING = auto()    # 제약 위반
    PENDING = auto()      # 아직 판정 불가


@dataclass
class AutomatonTransition:
    """상태 전이"""
    from_state: str
    to_state: str
    condition: Callable[[Tuple[float, float], PropositionEvaluator], bool]


class LTLAutomaton:
    """
    LTL Automaton (Simplified)

    LTL formula를 받아 trace(경로)의 만족 여부를 검사하는 오토마톤
    실제로는 Büchi automaton으로 변환하지만, 여기서는 단순화된 버전 사용
    """

    def __init__(self, formula: LTLFormula, evaluator: PropositionEvaluator):
        self.formula = formula
        self.evaluator = evaluator
        self.current_state = "q0"
        self.violation_detected = False

    def reset(self):
        self.current_state = "q0"
        self.violation_detected = False

    def check_trace(self, trace: List[Tuple[float, float]]) -> Tuple[bool, int]:
        """
        전체 trace가 LTL formula를 만족하는지 검사

        Returns:
            (satisfied, violation_index)
            - satisfied: True if trace satisfies formula
            - violation_index: -1 if satisfied, else index of first violation
        """
        self.reset()

        for i, position in enumerate(trace):
            if not self._check_step(position, i, len(trace)):
                return False, i

        return self._check_final(), -1

    def check_partial_trace(self, trace: List[Tuple[float, float]]) -> LTLAutomatonState:
        """
        부분 trace 검사 (Constrained Decoding용)

        Returns:
            - REJECTING: 이미 위반 (복구 불가)
            - ACCEPTING: 현재까지 만족
            - PENDING: 아직 판정 불가
        """
        self.reset()

        for i, position in enumerate(trace):
            if not self._check_step(position, i, len(trace), is_partial=True):
                return LTLAutomatonState.REJECTING

        return LTLAutomatonState.ACCEPTING

    def would_violate(self, current_trace: List[Tuple[float, float]],
                      next_position: Tuple[float, float]) -> bool:
        """
        다음 위치를 추가하면 위반이 되는지 검사 (Constrained Decoding 핵심)
        """
        extended_trace = current_trace + [next_position]
        state = self.check_partial_trace(extended_trace)
        return state == LTLAutomatonState.REJECTING

    def _check_step(self, position: Tuple[float, float], index: int,
                    total_length: int, is_partial: bool = False) -> bool:
        """단일 스텝에서 formula 타입별 검사"""
        op = self.formula.operator

        # G(φ) - Globally: 모든 상태에서 φ 만족
        if op == LTLOperator.GLOBALLY:
            inner = self.formula.operands[0]
            if not self.evaluator.evaluate_formula(position, inner):
                self.violation_detected = True
                return False
            return True

        # F(φ) - Finally: 언젠가 φ 만족 (partial에서는 pending)
        elif op == LTLOperator.FINALLY:
            inner = self.formula.operands[0]
            if self.evaluator.evaluate_formula(position, inner):
                self.current_state = "accepting"
            return True  # F는 중간에 실패하지 않음

        # G(¬φ) - 항상 φ가 아님
        elif (op == LTLOperator.GLOBALLY and
              self.formula.operands[0].operator == LTLOperator.NOT):
            neg_inner = self.formula.operands[0].operands[0]
            if self.evaluator.evaluate_formula(position, neg_inner):
                self.violation_detected = True
                return False
            return True

        # 복합 formula: G(φ1) ∧ F(φ2)
        elif op == LTLOperator.AND:
            for operand in self.formula.operands:
                sub_automaton = LTLAutomaton(operand, self.evaluator)
                if not sub_automaton._check_step(position, index, total_length, is_partial):
                    self.violation_detected = True
                    return False
            return True

        return True

    def _check_final(self) -> bool:
        """trace 끝에서 최종 판정"""
        if self.violation_detected:
            return False

        op = self.formula.operator

        # F(φ) - 끝까지 도달 못했으면 실패
        if op == LTLOperator.FINALLY:
            return self.current_state == "accepting"

        # G(φ) ∧ F(ψ)
        if op == LTLOperator.AND:
            for operand in self.formula.operands:
                if operand.operator == LTLOperator.FINALLY:
                    sub_auto = LTLAutomaton(operand, self.evaluator)
                    sub_auto.current_state = self.current_state
                    if not sub_auto._check_final():
                        return False
            return True

        return True


# =============================================================================
# Simulated LLM Plan Generator
# =============================================================================

@dataclass
class PlanToken:
    """계획 토큰 (waypoint)"""
    position: Tuple[float, float]
    action: str  # "move", "turn", "stop"
    confidence: float = 1.0


class SimulatedLLM:
    """
    시뮬레이션된 LLM

    실제 LLM 대신 간단한 경로 생성기로 대체
    목적: Constrained Decoding 메커니즘 시연
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

    def generate_plan_tokens(self, start: Tuple[float, float],
                            goal: Tuple[float, float],
                            num_waypoints: int = 10) -> List[PlanToken]:
        """
        시작점에서 목표까지 waypoint 토큰 생성

        직선 경로 + 노이즈
        """
        tokens = []

        for i in range(num_waypoints + 1):
            t = i / num_waypoints
            # 직선 보간 + 노이즈
            x = start[0] + t * (goal[0] - start[0]) + self.rng.gauss(0, 0.3)
            y = start[1] + t * (goal[1] - start[1]) + self.rng.gauss(0, 0.3)

            tokens.append(PlanToken(
                position=(x, y),
                action="move" if i < num_waypoints else "stop",
                confidence=0.9 + self.rng.random() * 0.1
            ))

        return tokens

    def generate_alternative_token(self, current_pos: Tuple[float, float],
                                   forbidden_zones: List[Zone],
                                   goal: Tuple[float, float]) -> PlanToken:
        """
        위반 토큰 대신 대안 토큰 생성

        금지구역을 피하는 방향으로 이동
        """
        # 목표 방향
        dx = goal[0] - current_pos[0]
        dy = goal[1] - current_pos[1]
        dist = math.sqrt(dx*dx + dy*dy)

        if dist < 0.1:
            return PlanToken(position=current_pos, action="stop")

        # 정규화
        dx, dy = dx/dist, dy/dist

        # 여러 방향 시도
        for angle_offset in [0, 30, -30, 60, -60, 90, -90, 120, -120, 150, -150, 180]:
            rad = math.radians(angle_offset)
            new_dx = dx * math.cos(rad) - dy * math.sin(rad)
            new_dy = dx * math.sin(rad) + dy * math.cos(rad)

            step = 1.0
            new_pos = (current_pos[0] + new_dx * step,
                      current_pos[1] + new_dy * step)

            # 금지구역 체크
            in_forbidden = False
            for zone in forbidden_zones:
                if zone.polygon.contains(Point(new_pos)):
                    in_forbidden = True
                    break

            if not in_forbidden:
                return PlanToken(position=new_pos, action="move")

        # 모든 방향이 막혔으면 제자리
        return PlanToken(position=current_pos, action="stop")


# =============================================================================
# SELP Constrained Decoder
# =============================================================================

@dataclass
class DecodingStep:
    """디코딩 단계 기록"""
    step: int
    proposed_token: PlanToken
    accepted: bool
    alternative_token: Optional[PlanToken]
    automaton_state: LTLAutomatonState
    reason: str


class SELPConstrainedDecoder:
    """
    SELP Constrained Decoder

    LLM 토큰 생성 시 LTL 오토마톤으로 검증하고,
    위반 토큰은 마스킹하여 대안 토큰 선택
    """

    def __init__(self, ltl_formula: LTLFormula, evaluator: PropositionEvaluator,
                 zones: List[Zone]):
        self.automaton = LTLAutomaton(ltl_formula, evaluator)
        self.evaluator = evaluator
        self.zones = zones
        self.llm = SimulatedLLM()
        self.decoding_history: List[DecodingStep] = []

    def decode_plan(self, start: Tuple[float, float],
                    goal: Tuple[float, float],
                    max_steps: int = 20) -> Tuple[List[PlanToken], bool]:
        """
        Constrained Decoding으로 계획 생성

        Returns:
            (plan, success)
        """
        self.decoding_history = []
        self.automaton.reset()

        # LLM에서 초기 토큰들 생성
        proposed_tokens = self.llm.generate_plan_tokens(start, goal, max_steps)

        accepted_plan: List[PlanToken] = []
        current_trace: List[Tuple[float, float]] = [start]

        for step, token in enumerate(proposed_tokens):
            # === Constrained Decoding 핵심 ===
            # 이 토큰을 추가하면 LTL 위반인지 검사
            would_violate = self.automaton.would_violate(current_trace, token.position)

            if would_violate:
                # 위반! 대안 토큰 생성
                alternative = self.llm.generate_alternative_token(
                    current_trace[-1], self.zones, goal
                )

                # 대안도 위반인지 검사
                if self.automaton.would_violate(current_trace, alternative.position):
                    # 대안도 실패 → 디코딩 중단
                    self.decoding_history.append(DecodingStep(
                        step=step,
                        proposed_token=token,
                        accepted=False,
                        alternative_token=alternative,
                        automaton_state=LTLAutomatonState.REJECTING,
                        reason="no_safe_alternative"
                    ))
                    break

                # 대안 채택
                self.decoding_history.append(DecodingStep(
                    step=step,
                    proposed_token=token,
                    accepted=False,
                    alternative_token=alternative,
                    automaton_state=LTLAutomatonState.ACCEPTING,
                    reason="token_replaced"
                ))
                accepted_plan.append(alternative)
                current_trace.append(alternative.position)
            else:
                # 통과
                self.decoding_history.append(DecodingStep(
                    step=step,
                    proposed_token=token,
                    accepted=True,
                    alternative_token=None,
                    automaton_state=LTLAutomatonState.ACCEPTING,
                    reason="token_accepted"
                ))
                accepted_plan.append(token)
                current_trace.append(token.position)

            # 목표 도달 체크
            if self.evaluator.evaluate(token.position, "at_goal"):
                break

        # 최종 검증
        final_trace = [t.position for t in accepted_plan]
        satisfied, _ = self.automaton.check_trace([start] + final_trace)

        return accepted_plan, satisfied

    def get_decoding_stats(self) -> Dict:
        """디코딩 통계"""
        total = len(self.decoding_history)
        accepted = sum(1 for d in self.decoding_history if d.accepted)
        replaced = sum(1 for d in self.decoding_history if d.alternative_token)
        rejected = sum(1 for d in self.decoding_history
                      if d.automaton_state == LTLAutomatonState.REJECTING)

        return {
            "total_steps": total,
            "accepted_tokens": accepted,
            "replaced_tokens": replaced,
            "rejected_steps": rejected,
            "replacement_rate": 100 * replaced / max(1, total),
        }


# =============================================================================
# SELP Guard (Proper Implementation)
# =============================================================================

class SELPGuardProper:
    """
    SELP Guard - 제대로 된 구현

    SafetyChip과의 차이:
    1. 입력이 아닌 출력(계획)을 검증
    2. LTL 오토마톤으로 형식 검증
    3. Constrained Decoding으로 안전한 계획만 생성
    """

    def __init__(self, zones: List[Zone], goal: Tuple[float, float] = None):
        self.zones = zones
        self.zone_dict = {z.name: z for z in zones}
        self.goal = goal

        # Proposition evaluator
        self.evaluator = PropositionEvaluator(self.zone_dict, goal)

        # LTL 제약: G(¬in_forbidden) - 모든 금지구역 회피
        forbidden_props = [neg(prop(f"in_{z.name}")) for z in zones]
        if len(forbidden_props) == 1:
            inner = forbidden_props[0]
        else:
            inner = forbidden_props[0]
            for fp in forbidden_props[1:]:
                inner = land(inner, fp)
        self.safety_formula = globally(inner)

        # Decoder
        self.decoder = SELPConstrainedDecoder(
            self.safety_formula, self.evaluator, zones
        )

    def generate_safe_plan(self, start: Tuple[float, float],
                          goal: Tuple[float, float]) -> Tuple[List[PlanToken], Dict]:
        """
        안전한 계획 생성

        Returns:
            (plan, stats)
        """
        self.evaluator.goal = goal
        self.decoder.evaluator.goal = goal

        plan, success = self.decoder.decode_plan(start, goal)
        stats = self.decoder.get_decoding_stats()
        stats["success"] = success
        stats["ltl_formula"] = str(self.safety_formula)

        return plan, stats

    def verify_plan(self, plan: List[Tuple[float, float]]) -> Tuple[bool, int]:
        """
        외부 계획 검증

        Returns:
            (satisfied, violation_index)
        """
        automaton = LTLAutomaton(self.safety_formula, self.evaluator)
        return automaton.check_trace(plan)


# =============================================================================
# Comparison Experiment
# =============================================================================

@dataclass
class ComparisonResult:
    """비교 실험 결과"""
    method: str
    scenario: str

    # Plan generation
    plan_generated: bool
    plan_length: int
    plan_safe: bool  # LTL 만족 여부

    # Violation details
    violation_index: int  # -1 if no violation
    violation_position: Optional[Tuple[float, float]]

    # SELP specific
    tokens_replaced: int
    replacement_rate: float

    # Path
    path: List[Tuple[float, float]]


def run_comparison(start: Tuple[float, float], goal: Tuple[float, float],
                   zones: List[Zone], num_trials: int = 10) -> List[ComparisonResult]:
    """SafetyChip vs SELP 비교 실험"""
    results = []

    for trial in range(num_trials):
        # === Method 1: No Guard (직접 경로) ===
        llm = SimulatedLLM(seed=trial)
        tokens = llm.generate_plan_tokens(start, goal)
        path = [start] + [t.position for t in tokens]

        # LTL 검증
        zone_dict = {z.name: z for z in zones}
        evaluator = PropositionEvaluator(zone_dict, goal)
        forbidden_props = [neg(prop(f"in_{z.name}")) for z in zones]
        formula = globally(forbidden_props[0]) if len(forbidden_props) == 1 else \
                  globally(land(forbidden_props[0], forbidden_props[1]))
        automaton = LTLAutomaton(formula, evaluator)
        satisfied, viol_idx = automaton.check_trace(path)

        results.append(ComparisonResult(
            method="no_guard",
            scenario=f"trial_{trial}",
            plan_generated=True,
            plan_length=len(tokens),
            plan_safe=satisfied,
            violation_index=viol_idx,
            violation_position=path[viol_idx] if viol_idx >= 0 else None,
            tokens_replaced=0,
            replacement_rate=0.0,
            path=path,
        ))

        # === Method 2: SELP Constrained Decoding ===
        selp = SELPGuardProper(zones, goal)
        plan, stats = selp.generate_safe_plan(start, goal)
        path_selp = [start] + [t.position for t in plan]

        # 검증
        satisfied_selp, viol_idx_selp = selp.verify_plan(path_selp)

        results.append(ComparisonResult(
            method="selp",
            scenario=f"trial_{trial}",
            plan_generated=True,
            plan_length=len(plan),
            plan_safe=satisfied_selp,
            violation_index=viol_idx_selp,
            violation_position=path_selp[viol_idx_selp] if viol_idx_selp >= 0 else None,
            tokens_replaced=stats["replaced_tokens"],
            replacement_rate=stats["replacement_rate"],
            path=path_selp,
        ))

    return results


def print_comparison_results(results: List[ComparisonResult]):
    """결과 출력"""
    print("\n" + "="*70)
    print("SELP vs No Guard Comparison")
    print("="*70)

    # 메서드별 집계
    by_method = {}
    for r in results:
        if r.method not in by_method:
            by_method[r.method] = []
        by_method[r.method].append(r)

    print("\n[Summary]")
    print("-"*70)
    print(f"{'Method':<15} {'Trials':<10} {'Safe Plans':<12} {'Safety Rate':<12} {'Avg Replaced':<12}")
    print("-"*70)

    for method, method_results in by_method.items():
        total = len(method_results)
        safe = sum(1 for r in method_results if r.plan_safe)
        avg_replaced = np.mean([r.tokens_replaced for r in method_results])

        safety_rate = 100 * safe / total
        color = "\033[92m" if safety_rate == 100 else "\033[91m"

        print(f"{method:<15} {total:<10} {safe:<12} {color}{safety_rate:>10.1f}%\033[0m {avg_replaced:>10.1f}")

    print("-"*70)

    # SELP 상세
    selp_results = by_method.get("selp", [])
    if selp_results:
        print("\n[SELP Constrained Decoding Details]")
        print("-"*70)
        for r in selp_results[:5]:  # 처음 5개만
            status = "✓ SAFE" if r.plan_safe else "✗ VIOLATION"
            print(f"  {r.scenario}: {status}, replaced={r.tokens_replaced}, "
                  f"rate={r.replacement_rate:.1f}%")


# =============================================================================
# Visualization
# =============================================================================

def visualize_comparison(results: List[ComparisonResult], zones: List[Zone],
                        output_path: str):
    """비교 시각화"""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.collections import PatchCollection

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    methods = ["no_guard", "selp"]
    titles = ["No Guard (Direct LLM)", "SELP (Constrained Decoding)"]

    for idx, (method, title) in enumerate(zip(methods, titles)):
        ax = axes[idx]

        # 금지구역 표시
        patches = []
        for zone in zones:
            coords = list(zone.polygon.exterior.coords)
            patches.append(MplPolygon(coords, closed=True))
        pc = PatchCollection(patches, alpha=0.3, facecolor='red', edgecolor='red')
        ax.add_collection(pc)

        # 경로 표시
        method_results = [r for r in results if r.method == method]
        for r in method_results[:5]:  # 처음 5개
            path = r.path
            xs = [p[0] for p in path]
            ys = [p[1] for p in path]

            color = 'green' if r.plan_safe else 'red'
            alpha = 0.7 if r.plan_safe else 0.5
            ax.plot(xs, ys, '-', color=color, alpha=alpha, linewidth=1.5)

            # 시작/끝 표시
            ax.plot(xs[0], ys[0], 'o', color='blue', markersize=8)
            ax.plot(xs[-1], ys[-1], 's', color='purple', markersize=8)

            # 위반 지점 표시
            if r.violation_position:
                ax.plot(r.violation_position[0], r.violation_position[1],
                       'x', color='red', markersize=12, markeredgewidth=3)

        # 통계
        safe_count = sum(1 for r in method_results if r.plan_safe)
        total = len(method_results)
        safety_rate = 100 * safe_count / total

        ax.set_title(f"{title}\nSafety Rate: {safety_rate:.0f}%", fontsize=12, fontweight='bold')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 15)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # Legend
        ax.plot([], [], 'g-', label='Safe path')
        ax.plot([], [], 'r-', label='Unsafe path')
        ax.plot([], [], 'rx', markersize=10, label='Violation point')
        ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("="*70)
    print("SELP Proper Implementation Demo")
    print("="*70)

    # 환경 설정
    zones = [
        Zone.from_bounds("storage_racks", 14, 18, 7, 11),
    ]

    start = (2.0, 6.0)
    goal = (16.0, 9.0)  # 금지구역 내부!

    print(f"\nScenario: S1 - Direct Hazard Goal")
    print(f"Start: {start}")
    print(f"Goal: {goal} (inside storage_racks!)")
    print(f"Zones: {[z.name for z in zones]}")

    # SELP 데모
    print("\n" + "-"*70)
    print("SELP Constrained Decoding Demo")
    print("-"*70)

    selp = SELPGuardProper(zones, goal)
    print(f"LTL Formula: {selp.safety_formula}")

    plan, stats = selp.generate_safe_plan(start, goal)

    print(f"\nDecoding Stats:")
    print(f"  Total steps: {stats['total_steps']}")
    print(f"  Accepted tokens: {stats['accepted_tokens']}")
    print(f"  Replaced tokens: {stats['replaced_tokens']}")
    print(f"  Replacement rate: {stats['replacement_rate']:.1f}%")
    print(f"  Plan satisfies LTL: {stats['success']}")

    print(f"\nGenerated Plan ({len(plan)} waypoints):")
    for i, token in enumerate(plan[:10]):
        print(f"  {i}: {token.position[0]:.2f}, {token.position[1]:.2f} ({token.action})")
    if len(plan) > 10:
        print(f"  ... ({len(plan) - 10} more)")

    # 비교 실험
    print("\n" + "-"*70)
    print("Comparison Experiment: No Guard vs SELP")
    print("-"*70)

    results = run_comparison(start, goal, zones, num_trials=20)
    print_comparison_results(results)

    # 시각화
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'figures')
    os.makedirs(output_dir, exist_ok=True)
    visualize_comparison(results, zones, os.path.join(output_dir, 'selp_proper_comparison.png'))

    # 결과 저장
    output_path = os.path.join(os.path.dirname(__file__), '..', 'selp_proper_results.json')
    output_data = {
        "scenario": "S1_direct_hazard_goal",
        "start": start,
        "goal": goal,
        "zones": [{"name": z.name, "bounds": list(z.polygon.bounds)} for z in zones],
        "results": [
            {
                "method": r.method,
                "plan_safe": r.plan_safe,
                "tokens_replaced": r.tokens_replaced,
                "replacement_rate": r.replacement_rate,
            }
            for r in results
        ],
        "summary": {
            "no_guard_safety_rate": 100 * sum(1 for r in results if r.method == "no_guard" and r.plan_safe) / sum(1 for r in results if r.method == "no_guard"),
            "selp_safety_rate": 100 * sum(1 for r in results if r.method == "selp" and r.plan_safe) / sum(1 for r in results if r.method == "selp"),
        }
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
