#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RoboGuard-style Baseline for Mobile Robot Navigation Safety

Faithful reproduction of the RoboGuard architecture
(Ravichandran et al., 2024, arXiv:2503.07885, KumarRobotics/UPenn) adapted
for mobile robot navigation with forbidden zones.

RoboGuard Architecture (original paper):
  Stage 1 — Safety Reasoning Module:
    A root-of-trust LLM (GPT-4o, shielded from user input) receives:
      (a) robot API as Python function signatures
      (b) user-defined safety rules in natural language
      (c) current world model as a semantic graph (objects, regions, connections)
    → Outputs grounded LTL constraints via chain-of-thought reasoning.
    Example: rule "Stay away from hazardous regions" + region "construction_area_1"
             → G(!goto(construction_area_1))

  Stage 2 — Control Synthesis Module:
    LTL constraints are conjuncted: φ_safe = φ_1 ∧ φ_2 ∧ ... ∧ φ_n
    Converted to Büchi automaton via Spot library.
    Candidate plan (action sequence) is validated step-by-step through the automaton.
    If final state is accepting → ALLOW. Otherwise → REJECT (binary replacement).

Adaptation for our navigation experiment:
  - Robot actions = {goto(x,y)} — single navigation goal per trial
  - Semantic graph = zone polygon metadata
  - LTL constraints = G(!goto_zone(Z)) for each forbidden zone Z
  - Automaton = simplified 2-state Büchi (accepting ↔ rejecting)
  - No repair/projection — binary ALLOW/REJECT (faithful to paper)

Approximations clearly marked:
  [FAITHFUL]  Two-stage architecture (ground rules → check plan)
  [FAITHFUL]  LTL constraint generation from rules + environment
  [FAITHFUL]  Büchi automaton state-transition checking (simplified)
  [FAITHFUL]  Binary ALLOW/REJECT decision (no repair — paper uses replacement)
  [FAITHFUL]  Action-level checking (goal destination, not execution path)
  [FAITHFUL]  No safety margin (operates on semantic actions, not geometry)
  [APPROX]   Root-of-trust LLM replaced with deterministic rule grounding
             (equivalent for our fixed-zone scenario — LLM output is deterministic
              given identical rules + environment)
  [APPROX]   Spot library replaced with equivalent 2-state automaton
             (for G(!p) formulas, full Büchi reduces to 2 states)
  [APPROX]   Semantic graph replaced with polygon zone definitions
  [NOT IMPL] No path-through detection (paper checks actions, not paths)
  [NOT IMPL] No runtime monitoring (planning-stage guardrail only)

Reference:
    Ravichandran, Z., Robey, A., Kumar, V., Pappas, G.J., Hassani, H.
    "Safety Guardrails for LLM-Enabled Robots", 2024.
    arXiv:2503.07885  |  github.com/KumarRobotics/RoboGuard
"""

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set
from enum import Enum, auto

from .safety_baselines import SafetyDecision, SafetyResult


# =============================================================================
# Stage 1: Safety Reasoning Module — Rules + Environment → LTL Specifications
# =============================================================================

class LTLOperator(Enum):
    """Standard LTL operators used in RoboGuard specifications."""
    GLOBALLY = "G"       # G(φ) — always φ
    FINALLY = "F"        # F(φ) — eventually φ
    NEXT = "X"           # X(φ) — next step φ
    UNTIL = "U"          # φ U ψ — φ until ψ
    NEGATION = "!"       # ¬φ
    CONJUNCTION = "&"    # φ ∧ ψ
    DISJUNCTION = "|"    # φ ∨ ψ
    IMPLICATION = "->"   # φ → ψ


@dataclass
class GroundedLTLConstraint:
    """
    A single grounded LTL constraint from Stage 1.

    In the original paper, the root-of-trust LLM generates these as JSON:
      {"constraint": "G(!goto(construction_area_1))", "reasoning": "..."}

    We generate them deterministically from zone definitions (equivalent for
    our fixed-zone scenario).
    """
    constraint_id: str
    ltl_formula: str           # e.g., "G(!goto_zone(forbidden_zone))"
    natural_language_rule: str  # The safety rule this was grounded from
    reasoning: str             # Chain-of-thought reasoning (simulated)
    zone_name: str             # Which zone this constraint refers to
    zone_vertices: List[Tuple[float, float]]
    atomic_propositions: Set[str] = field(default_factory=set)


@dataclass
class SafetySpecification:
    """
    Complete safety specification φ_safe = φ_1 ∧ φ_2 ∧ ... ∧ φ_n

    Stage 1 output, ready for Stage 2 automaton construction.
    """
    constraints: List[GroundedLTLConstraint]
    conjuncted_formula: str    # φ_safe as a single LTL string
    environment_summary: str


class SafetyReasoningModule:
    """
    Stage 1: Safety Reasoning Module.

    [FAITHFUL] Receives safety rules + environment → produces grounded LTL constraints.
    [APPROX]   Root-of-trust LLM (GPT-4o) replaced with deterministic grounding.
               For our fixed-zone scenario, the LLM output is deterministic given
               identical rules + environment, so this is functionally equivalent.

    Original paper's LLM prompt structure:
      - System: "You are a safety reasoning module..."
      - Robot API: function signatures (goto, inspect, map_region, ...)
      - Safety rules: natural language (e.g., "Respect privacy", "Avoid hazards")
      - World model: semantic graph JSON (objects, regions, connections)
      → Output: {rule: [{constraint: LTL, reasoning: str}, ...]}
    """

    # Safety rules for our navigation scenario
    # Equivalent to the paper's user-defined rules
    DEFAULT_RULES = [
        "The robot must never navigate to a location inside a forbidden zone.",
    ]

    def __init__(self, rules: Optional[List[str]] = None):
        self.rules = rules or self.DEFAULT_RULES

    def ground_constraints(self, zones: List) -> SafetySpecification:
        """
        Ground safety rules in the current environment.

        [FAITHFUL] For each (rule, zone) pair, generate a grounded LTL constraint.
        [APPROX]   Deterministic grounding instead of LLM chain-of-thought.
                   The reasoning field simulates what GPT-4o would produce.

        In the paper's implementation (generator.py):
          ContextualGrounding.generate() → calls GPT-4o → parses JSON response

        Args:
            zones: Forbidden zones from the environment (our "semantic graph").

        Returns:
            SafetySpecification with all grounded constraints.
        """
        constraints = []

        for zone in zones:
            zone_name = zone.name.replace(' ', '_').lower()
            # Atomic proposition: goto_zone_<name> is true when robot navigates to zone
            ap = f"goto_zone_{zone_name}"

            for i, rule in enumerate(self.rules):
                constraint = GroundedLTLConstraint(
                    constraint_id=f"R{i+1}_{zone_name}",
                    ltl_formula=f"G(!{ap})",
                    natural_language_rule=rule,
                    reasoning=(
                        f"The zone '{zone.name}' is designated as forbidden. "
                        f"Rule '{rule}' requires that the robot never targets a location "
                        f"inside this zone. Therefore, the action goto_zone({zone_name}) must "
                        f"be globally avoided. This produces the LTL constraint G(!{ap})."
                    ),
                    zone_name=zone.name,
                    zone_vertices=list(zone.vertices),
                    atomic_propositions={ap},
                )
                constraints.append(constraint)

        # Conjunct all constraints: φ_safe = φ_1 & φ_2 & ... & φ_n
        if constraints:
            conjuncted = " & ".join(c.ltl_formula for c in constraints)
        else:
            conjuncted = "true"

        return SafetySpecification(
            constraints=constraints,
            conjuncted_formula=conjuncted,
            environment_summary=(
                f"Navigation environment with {len(zones)} forbidden zone(s). "
                f"Total constraints: {len(constraints)}."
            ),
        )


# =============================================================================
# Stage 2: Control Synthesis Module — Büchi Automaton Plan Validation
# =============================================================================

class AutomatonState(Enum):
    """Büchi automaton states for plan validation."""
    ACCEPTING = auto()   # Safe state (plan satisfies φ_safe so far)
    REJECTING = auto()   # Violation state (plan violates φ_safe)


@dataclass
class AutomatonTransition:
    """A transition in the Büchi automaton."""
    from_state: AutomatonState
    to_state: AutomatonState
    guard: str  # Propositional formula that triggers this transition
    description: str


@dataclass
class PlanAction:
    """
    A single action in a candidate plan.

    In the original paper, plans are sequences of (function, argument) tuples:
      [("goto", "region_1"), ("inspect", "person_1"), ("replan", "")]

    In our navigation context, each trial has one action:
      [("goto", (x, y))]
    """
    function: str                           # "goto"
    argument: object                        # (x, y) coordinates
    propositions: Dict[str, bool] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """
    Result of validating a plan through the Büchi automaton.

    Matches the paper's validate_action_sequence() return format:
    per-step safety state + final decision.
    """
    is_safe: bool
    final_state: AutomatonState
    step_results: List[Tuple[str, AutomatonState]]  # [(action_str, state_after)]
    violated_constraints: List[str]                   # Which LTL formulas were violated
    explanation: str


class ControlSynthesisModule:
    """
    Stage 2: Control Synthesis Module.

    [FAITHFUL] Constructs Büchi automaton from φ_safe, validates plan step-by-step.
    [APPROX]   Spot library replaced with equivalent manual automaton construction.
               For formulas of the form G(!p1) & G(!p2) & ..., the Büchi automaton
               has exactly 2 states (accepting, rejecting) with transitions:
                 accepting → accepting : ¬p1 ∧ ¬p2 ∧ ...  (all safe)
                 accepting → rejecting : p1 ∨ p2 ∨ ...     (any violation)
                 rejecting → rejecting : true               (sink state)
               This is provably equivalent to Spot's output for this formula class.

    Original paper's implementation (synthesis.py):
      self.automaton = spot.translate(ltl_formula, "Buchi", "complete", "state-based")
      Then validate_action_sequence() walks the automaton per action.
    """

    def __init__(self, specification: SafetySpecification):
        self.specification = specification
        self._build_automaton()

    def _build_automaton(self):
        """
        Build Büchi automaton from φ_safe.

        For G(!p) formulas, the automaton is:
          States: {q_accept, q_reject}
          Initial: q_accept
          Accepting: {q_accept}
          Transitions:
            q_accept --[¬p]--> q_accept
            q_accept --[p]---> q_reject
            q_reject --[true]-> q_reject  (sink)

        [APPROX] For our formula class G(!p1) & G(!p2) & ..., this 2-state
        automaton is equivalent to what Spot produces. Full Spot would handle
        arbitrary LTL but is overkill for G(!p) conjunction.
        """
        self.initial_state = AutomatonState.ACCEPTING
        self.accepting_states = {AutomatonState.ACCEPTING}

        # Collect all forbidden atomic propositions
        self.forbidden_aps: Set[str] = set()
        for constraint in self.specification.constraints:
            self.forbidden_aps.update(constraint.atomic_propositions)

        # Build transitions
        safe_guard = " & ".join(f"!{ap}" for ap in sorted(self.forbidden_aps)) or "true"
        violation_guard = " | ".join(sorted(self.forbidden_aps)) or "false"

        self.transitions = [
            AutomatonTransition(
                AutomatonState.ACCEPTING, AutomatonState.ACCEPTING,
                safe_guard, "All propositions safe"
            ),
            AutomatonTransition(
                AutomatonState.ACCEPTING, AutomatonState.REJECTING,
                violation_guard, "Forbidden proposition became true"
            ),
            AutomatonTransition(
                AutomatonState.REJECTING, AutomatonState.REJECTING,
                "true", "Sink state — violation is permanent"
            ),
        ]

    def _evaluate_propositions(self, action: PlanAction) -> Dict[str, bool]:
        """
        Convert an action to atomic propositions.

        [FAITHFUL] Paper uses closed-world assumption: the action's AP is true,
        all other APs are false.

        For goto(x, y): check if (x, y) is inside each forbidden zone.
        If inside zone Z → goto_zone_Z = true.
        """
        propositions = {}
        goal = action.argument  # (x, y)

        for constraint in self.specification.constraints:
            ap = next(iter(constraint.atomic_propositions))
            # Check if goal is inside this zone
            inside = _point_in_polygon(goal, constraint.zone_vertices)
            propositions[ap] = inside

        return propositions

    def validate_plan(self, plan: List[PlanAction]) -> ValidationResult:
        """
        Validate a candidate plan through the Büchi automaton.

        [FAITHFUL] Walks automaton step-by-step, checking each action.
        Matches paper's validate_action_sequence() method.

        Args:
            plan: Sequence of PlanActions to validate.

        Returns:
            ValidationResult with per-step states and final decision.
        """
        current_state = self.initial_state
        step_results = []
        violated = []

        for action in plan:
            # Get propositions for this action
            props = self._evaluate_propositions(action)

            # Check if any forbidden AP is true
            any_violation = any(props.get(ap, False) for ap in self.forbidden_aps)

            if any_violation and current_state == AutomatonState.ACCEPTING:
                current_state = AutomatonState.REJECTING
                # Identify which constraints were violated
                for constraint in self.specification.constraints:
                    ap = next(iter(constraint.atomic_propositions))
                    if props.get(ap, False):
                        violated.append(constraint.ltl_formula)

            action_str = f"{action.function}({action.argument})"
            step_results.append((action_str, current_state))

        is_safe = current_state in self.accepting_states

        if is_safe:
            explanation = (
                f"Plan validated: {len(plan)} action(s) checked, "
                f"automaton ends in accepting state."
            )
        else:
            explanation = (
                f"Plan rejected: automaton reached rejecting state. "
                f"Violated constraints: {violated}"
            )

        return ValidationResult(
            is_safe=is_safe,
            final_state=current_state,
            step_results=step_results,
            violated_constraints=violated,
            explanation=explanation,
        )


# =============================================================================
# RoboGuard Baseline — Complete Two-Stage Pipeline
# =============================================================================

class RoboGuardBaseline:
    """
    Complete RoboGuard baseline combining Stage 1 + Stage 2.

    Behavior:
      - Goal inside forbidden zone → REJECT (automaton reaches rejecting state)
      - Goal outside forbidden zone → ALLOW (even if path would cross zone)
      - No safety margin (action-level check, not geometric margin)
      - No path-through detection (not in original RoboGuard)
      - No runtime monitoring (planning-stage guardrail only)
      - Binary decision: ALLOW or REJECT (no repair — paper uses replacement)

    This means RoboGuard will behave similarly to SELP in our experiment:
      - S1 inside_zone: REJECT (goal in zone)
      - S1 through_zone: ALLOW → FN (goal outside zone, path crosses)
      - S1 near/mid_boundary: ALLOW (no margin)
      - S3: ALLOW → FN (goal outside zone, path crosses)
      - S4: FN (no runtime guard)
      - S5: FN (planning-stage only)

    The key experimental insight: both RoboGuard and SELP operate at the
    action/planning level without geometric safety margins or path awareness.
    PETSE's advantage comes from execution-layer enforcement with uncertainty-
    aware margins and runtime monitoring.

    Usage:
        zones = [GeofenceZone(...), ...]
        guard = RoboGuardBaseline(zones)
        result = guard.evaluate(goal=(5.0, 0.0), start=(0.0, 0.0))
        # result.decision == SafetyDecision.REJECT
    """

    def __init__(self, zones: List, rules: Optional[List[str]] = None):
        """
        Initialize RoboGuard baseline.

        Args:
            zones: Forbidden zones from the environment.
            rules: Safety rules in natural language (default: zone avoidance).
        """
        self.zones = zones

        # Stage 1: Safety Reasoning Module
        reasoning_module = SafetyReasoningModule(rules=rules)
        self.specification = reasoning_module.ground_constraints(zones)

        # Stage 2: Control Synthesis Module
        self.synthesis = ControlSynthesisModule(self.specification)

    def evaluate(self, goal: Tuple[float, float],
                 start: Tuple[float, float] = (0.0, 0.0)) -> SafetyResult:
        """
        Evaluate a candidate navigation goal using the full RoboGuard pipeline.

        Stage 1: specification is pre-computed in __init__.
        Stage 2: construct plan [goto(goal)] → validate through automaton.

        [FAITHFUL] Binary ALLOW/REJECT (no repair).
        [FAITHFUL] Action-level check (goal point only, not path).

        Args:
            goal: Candidate goal (x, y).
            start: Robot's current position (unused — RoboGuard checks actions,
                   not paths, so start position is irrelevant).

        Returns:
            SafetyResult compatible with the existing baseline framework.
        """
        # Construct candidate plan: a single goto action
        plan = [PlanAction(function="goto", argument=goal)]

        # Stage 2: Validate through Büchi automaton
        result = self.synthesis.validate_plan(plan)

        # Compute distance to nearest zone boundary (for metrics, not for decision)
        min_dist = float('inf')
        closest_zone = ""
        for constraint in self.specification.constraints:
            dist = _distance_to_polygon(goal, constraint.zone_vertices)
            if dist < min_dist:
                min_dist = dist
                closest_zone = constraint.zone_name

        if result.is_safe:
            return SafetyResult(
                decision=SafetyDecision.ALLOW,
                reason=(
                    f"RoboGuard: plan approved — automaton accepts. "
                    f"Action goto({goal[0]:.2f},{goal[1]:.2f}) satisfies φ_safe."
                ),
                distance_to_boundary=min_dist,
                margin_used=0.0,
                extra_info={
                    "method": "roboguard",
                    "stage1_constraints": len(self.specification.constraints),
                    "stage2_decision": "ALLOW",
                    "automaton_state": result.final_state.name,
                    "phi_safe": self.specification.conjuncted_formula,
                    "closest_zone": closest_zone,
                    "reference": "Ravichandran et al., 2024 (arXiv:2503.07885)",
                },
            )
        else:
            violated_zones = []
            for constraint in self.specification.constraints:
                if constraint.ltl_formula in result.violated_constraints:
                    violated_zones.append(constraint.zone_name)

            return SafetyResult(
                decision=SafetyDecision.REJECT,
                reason=(
                    f"RoboGuard: plan rejected — automaton reaches rejecting state. "
                    f"Action goto({goal[0]:.2f},{goal[1]:.2f}) violates "
                    f"{result.violated_constraints}."
                ),
                distance_to_boundary=min_dist,
                margin_used=0.0,
                extra_info={
                    "method": "roboguard",
                    "stage1_constraints": len(self.specification.constraints),
                    "stage2_decision": "REJECT",
                    "automaton_state": result.final_state.name,
                    "violated_constraints": result.violated_constraints,
                    "violated_zones": violated_zones,
                    "phi_safe": self.specification.conjuncted_formula,
                    "explanation": result.explanation,
                    "reference": "Ravichandran et al., 2024 (arXiv:2503.07885)",
                },
            )

    def evaluate_multi_step(self, goals: List[Tuple[float, float]]) -> SafetyResult:
        """
        Evaluate a multi-step plan (e.g., S2 salami sequence).

        [FAITHFUL] RoboGuard validates action SEQUENCES, not just single actions.
        Each step is checked through the automaton in order.

        Args:
            goals: Sequence of (x, y) goals forming the plan.

        Returns:
            SafetyResult for the overall plan.
        """
        plan = [PlanAction(function="goto", argument=g) for g in goals]
        result = self.synthesis.validate_plan(plan)

        min_dist = float('inf')
        for g in goals:
            for constraint in self.specification.constraints:
                dist = _distance_to_polygon(g, constraint.zone_vertices)
                if dist < min_dist:
                    min_dist = dist

        if result.is_safe:
            return SafetyResult(
                decision=SafetyDecision.ALLOW,
                reason=f"RoboGuard: multi-step plan approved ({len(goals)} steps).",
                distance_to_boundary=min_dist,
                margin_used=0.0,
                extra_info={
                    "method": "roboguard",
                    "stage2_decision": "ALLOW",
                    "step_results": [(s, st.name) for s, st in result.step_results],
                },
            )
        else:
            violated_zones = []
            for constraint in self.specification.constraints:
                if constraint.ltl_formula in result.violated_constraints:
                    violated_zones.append(constraint.zone_name)

            return SafetyResult(
                decision=SafetyDecision.REJECT,
                reason=(
                    f"RoboGuard: multi-step plan rejected at step "
                    f"that violates {result.violated_constraints}."
                ),
                distance_to_boundary=min_dist,
                margin_used=0.0,
                extra_info={
                    "method": "roboguard",
                    "stage2_decision": "REJECT",
                    "violated_constraints": result.violated_constraints,
                    "violated_zones": violated_zones,
                    "step_results": [(s, st.name) for s, st in result.step_results],
                },
            )


# =============================================================================
# Geometry Utilities
# =============================================================================

def _point_in_polygon(point: Tuple[float, float],
                      vertices: List[Tuple[float, float]]) -> bool:
    """Ray casting algorithm for point-in-polygon test."""
    x, y = point
    n = len(vertices)
    inside = False

    j = n - 1
    for i in range(n):
        xi, yi = vertices[i]
        xj, yj = vertices[j]

        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i

    return inside


def _point_to_segment_distance(point: Tuple[float, float],
                                seg_start: Tuple[float, float],
                                seg_end: Tuple[float, float]) -> float:
    """Distance from point to line segment."""
    px, py = point
    x1, y1 = seg_start
    x2, y2 = seg_end

    dx = x2 - x1
    dy = y2 - y1

    if dx == 0 and dy == 0:
        return math.sqrt((px - x1) ** 2 + (py - y1) ** 2)

    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    nearest_x = x1 + t * dx
    nearest_y = y1 + t * dy

    return math.sqrt((px - nearest_x) ** 2 + (py - nearest_y) ** 2)


def _distance_to_polygon(point: Tuple[float, float],
                          vertices: List[Tuple[float, float]]) -> float:
    """Minimum distance from point to polygon boundary."""
    if _point_in_polygon(point, vertices):
        return 0.0

    min_dist = float('inf')
    n = len(vertices)

    for i in range(n):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n]
        dist = _point_to_segment_distance(point, p1, p2)
        if dist < min_dist:
            min_dist = dist

    return min_dist
