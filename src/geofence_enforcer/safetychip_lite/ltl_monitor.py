"""
SafetyChip-lite: LTL Constraint Monitor

Progression-based LTL monitor for runtime constraint checking.
Supports the three main SafetyChip patterns without requiring spot library:
1. Always avoid: G(!p)
2. Precedence: (!q) U p
3. Trigger-response: G(p -> F(q))

Key features:
- Check if an action would violate constraints BEFORE execution
- Generate diagnostic information explaining WHY a violation occurs
- Track constraint satisfaction progress over time
"""

import re
import logging
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod

from .nl2ltl import LTLConstraint, ConstraintType

logger = logging.getLogger(__name__)


class MonitorStatus(Enum):
    """Status of the constraint monitor"""
    SATISFIED = auto()      # Constraint is satisfied (for safety constraints: not violated)
    VIOLATED = auto()       # Constraint has been violated
    PENDING = auto()        # Constraint not yet satisfied but not violated (for liveness)
    UNKNOWN = auto()


@dataclass
class ViolationInfo:
    """Information about a constraint violation"""
    constraint: LTLConstraint         # The violated constraint
    violated_proposition: str         # Which proposition caused violation
    proposition_change: Tuple[bool, bool]  # (before, after) values
    explanation: str                  # Human-readable explanation
    suggested_action: str             # Suggested alternative


@dataclass
class MonitorState:
    """Current state of the monitor"""
    status: MonitorStatus
    violated_constraints: List[ViolationInfo] = field(default_factory=list)
    pending_obligations: List[str] = field(default_factory=list)
    step_count: int = 0


class ConstraintMonitor(ABC):
    """Abstract base class for constraint monitors"""

    @abstractmethod
    def reset(self) -> None:
        """Reset monitor to initial state"""
        pass

    @abstractmethod
    def step(self, props: Dict[str, bool]) -> MonitorState:
        """Process one step with given propositions"""
        pass

    @abstractmethod
    def would_violate(self, props: Dict[str, bool]) -> Tuple[bool, List[ViolationInfo]]:
        """Check if these propositions would cause a violation"""
        pass


class PatternMonitor(ConstraintMonitor):
    """
    Pattern-based LTL monitor using progression semantics.

    For each supported pattern, we track a simple state machine:

    G(!p) - Always Avoid:
        State: {violated: bool}
        Transition: if p becomes true -> violated
        Once violated, stays violated

    (!q) U p - Precedence (p before q):
        State: {satisfied: bool, violated: bool}
        Transition: if p becomes true -> satisfied
                    if q becomes true before p -> violated

    G(p -> F(q)) - Trigger Response:
        State: {pending_responses: Set[trigger_instance]}
        Transition: if p becomes true -> add pending q
                    if q becomes true -> clear pending
                    Violation only checked at trace end or timeout
    """

    def __init__(self, constraint: LTLConstraint):
        self.constraint = constraint
        self.pattern_type = constraint.constraint_type
        self.params = constraint.parameters

        # State variables
        self._violated = False
        self._satisfied = False  # For precedence/eventually
        self._pending_responses: List[int] = []  # For trigger-response (step numbers)
        self._step_count = 0

    def reset(self) -> None:
        """Reset monitor to initial state"""
        self._violated = False
        self._satisfied = False
        self._pending_responses = []
        self._step_count = 0

    def step(self, props: Dict[str, bool]) -> MonitorState:
        """Process one step with given propositions"""
        self._step_count += 1
        violations = []

        if self.pattern_type == ConstraintType.ALWAYS_AVOID:
            violations = self._step_always_avoid(props)
        elif self.pattern_type == ConstraintType.PRECEDENCE:
            violations = self._step_precedence(props)
        elif self.pattern_type == ConstraintType.TRIGGER_RESPONSE:
            violations = self._step_trigger_response(props)
        elif self.pattern_type == ConstraintType.EVENTUALLY:
            violations = self._step_eventually(props)
        elif self.pattern_type == ConstraintType.ALWAYS_MAINTAIN:
            violations = self._step_always_maintain(props)

        status = MonitorStatus.VIOLATED if self._violated else (
            MonitorStatus.SATISFIED if self._satisfied else MonitorStatus.PENDING
        )

        return MonitorState(
            status=status,
            violated_constraints=violations,
            pending_obligations=self._get_pending_obligations(),
            step_count=self._step_count
        )

    def would_violate(self, props: Dict[str, bool]) -> Tuple[bool, List[ViolationInfo]]:
        """Check if these propositions would cause a violation WITHOUT advancing state"""
        if self._violated:
            return True, [self._make_violation_info("already_violated", props)]

        if self.pattern_type == ConstraintType.ALWAYS_AVOID:
            return self._check_always_avoid(props)
        elif self.pattern_type == ConstraintType.PRECEDENCE:
            return self._check_precedence(props)
        elif self.pattern_type == ConstraintType.TRIGGER_RESPONSE:
            return self._check_trigger_response(props)
        elif self.pattern_type == ConstraintType.EVENTUALLY:
            return False, []  # Eventually can't be violated by a single step
        elif self.pattern_type == ConstraintType.ALWAYS_MAINTAIN:
            return self._check_always_maintain(props)

        return False, []

    # =========================================================================
    # Always Avoid: G(!p)
    # =========================================================================

    def _step_always_avoid(self, props: Dict[str, bool]) -> List[ViolationInfo]:
        """Step for always-avoid pattern"""
        avoid_prop = f"in_{self.params['avoid_zone']}"
        if props.get(avoid_prop, False):
            self._violated = True
            return [self._make_violation_info(avoid_prop, props)]
        return []

    def _check_always_avoid(self, props: Dict[str, bool]) -> Tuple[bool, List[ViolationInfo]]:
        """Check always-avoid without advancing state"""
        avoid_prop = f"in_{self.params['avoid_zone']}"
        if props.get(avoid_prop, False):
            return True, [self._make_violation_info(avoid_prop, props)]
        return False, []

    # =========================================================================
    # Precedence: (!q) U p (p before q)
    # =========================================================================

    def _step_precedence(self, props: Dict[str, bool]) -> List[ViolationInfo]:
        """Step for precedence pattern"""
        if self._satisfied:
            return []  # Already satisfied

        first_prop = f"in_{self.params['first']}"
        second_prop = f"in_{self.params['second']}"

        # Check if first (required) is reached
        if props.get(first_prop, False):
            self._satisfied = True
            return []

        # Check if second is reached before first -> violation
        if props.get(second_prop, False):
            self._violated = True
            return [self._make_violation_info(second_prop, props,
                reason=f"Entered {self.params['second']} before visiting {self.params['first']}")]

        return []

    def _check_precedence(self, props: Dict[str, bool]) -> Tuple[bool, List[ViolationInfo]]:
        """Check precedence without advancing state"""
        if self._satisfied:
            return False, []

        first_prop = f"in_{self.params['first']}"
        second_prop = f"in_{self.params['second']}"

        # If entering second before first -> violation
        if props.get(second_prop, False) and not props.get(first_prop, False):
            return True, [self._make_violation_info(second_prop, props,
                reason=f"Would enter {self.params['second']} before visiting {self.params['first']}")]

        return False, []

    # =========================================================================
    # Trigger Response: G(p -> F(q))
    # =========================================================================

    def _step_trigger_response(self, props: Dict[str, bool]) -> List[ViolationInfo]:
        """Step for trigger-response pattern"""
        trigger_prop = f"in_{self.params['trigger']}"
        response_prop = f"in_{self.params['response']}"

        # Check if response satisfies pending obligations
        if props.get(response_prop, False):
            self._pending_responses = []  # Clear all pending

        # Check if trigger creates new obligation
        if props.get(trigger_prop, False):
            if self._step_count not in self._pending_responses:
                self._pending_responses.append(self._step_count)

        return []  # Trigger-response violations are checked at trace end

    def _check_trigger_response(self, props: Dict[str, bool]) -> Tuple[bool, List[ViolationInfo]]:
        """Check trigger-response - always safe for single step"""
        # Trigger-response can't be violated by a single action
        # (it's a liveness property)
        return False, []

    # =========================================================================
    # Eventually: F(p)
    # =========================================================================

    def _step_eventually(self, props: Dict[str, bool]) -> List[ViolationInfo]:
        """Step for eventually pattern"""
        goal_prop = f"in_{self.params['goal']}"
        if props.get(goal_prop, False):
            self._satisfied = True
        return []

    # =========================================================================
    # Always Maintain: G(p)
    # =========================================================================

    def _step_always_maintain(self, props: Dict[str, bool]) -> List[ViolationInfo]:
        """Step for always-maintain pattern"""
        maintain_prop = f"in_{self.params['maintain']}"
        if maintain_prop.startswith("in_in_"):
            maintain_prop = maintain_prop[3:]  # Remove double "in_"

        if not props.get(maintain_prop, True):  # Default to True to avoid false violations
            self._violated = True
            return [self._make_violation_info(maintain_prop, props,
                reason=f"Failed to maintain {maintain_prop}")]
        return []

    def _check_always_maintain(self, props: Dict[str, bool]) -> Tuple[bool, List[ViolationInfo]]:
        """Check always-maintain without advancing state"""
        maintain_prop = f"in_{self.params['maintain']}"
        if maintain_prop.startswith("in_in_"):
            maintain_prop = maintain_prop[3:]

        if not props.get(maintain_prop, True):
            return True, [self._make_violation_info(maintain_prop, props,
                reason=f"Would fail to maintain {maintain_prop}")]
        return False, []

    # =========================================================================
    # Helper methods
    # =========================================================================

    def _make_violation_info(self, prop: str, props: Dict[str, bool],
                             reason: str = None) -> ViolationInfo:
        """Create violation info with explanation"""
        prop_value = props.get(prop, False)

        if reason is None:
            if self.pattern_type == ConstraintType.ALWAYS_AVOID:
                reason = f"Entered forbidden zone: {self.params['avoid_zone']}"
            elif self.pattern_type == ConstraintType.PRECEDENCE:
                reason = f"Visited {self.params['second']} before {self.params['first']}"
            else:
                reason = f"Constraint violated: {self.constraint.ltl_formula}"

        # Generate suggested action
        if self.pattern_type == ConstraintType.ALWAYS_AVOID:
            suggestion = f"Move away from {self.params['avoid_zone']} zone"
        elif self.pattern_type == ConstraintType.PRECEDENCE:
            suggestion = f"Visit {self.params['first']} first"
        else:
            suggestion = "Choose a different action"

        return ViolationInfo(
            constraint=self.constraint,
            violated_proposition=prop,
            proposition_change=(not prop_value, prop_value),
            explanation=reason,
            suggested_action=suggestion
        )

    def _get_pending_obligations(self) -> List[str]:
        """Get list of pending obligations"""
        pending = []
        if self.pattern_type == ConstraintType.TRIGGER_RESPONSE and self._pending_responses:
            response = self.params.get('response', 'response')
            pending.append(f"Must eventually visit {response}")
        if self.pattern_type == ConstraintType.EVENTUALLY and not self._satisfied:
            goal = self.params.get('goal', 'goal')
            pending.append(f"Must eventually reach {goal}")
        if self.pattern_type == ConstraintType.PRECEDENCE and not self._satisfied:
            first = self.params.get('first', 'first')
            pending.append(f"Must visit {first} before continuing")
        return pending


class CompositeMonitor(ConstraintMonitor):
    """
    Composite monitor that combines multiple pattern monitors.
    All constraints must be satisfied (conjunction).
    """

    def __init__(self, constraints: List[LTLConstraint]):
        self.monitors: List[PatternMonitor] = []
        for c in constraints:
            self.monitors.append(PatternMonitor(c))

    def reset(self) -> None:
        """Reset all monitors"""
        for m in self.monitors:
            m.reset()

    def step(self, props: Dict[str, bool]) -> MonitorState:
        """Process step through all monitors"""
        all_violations = []
        all_pending = []
        any_violated = False

        for m in self.monitors:
            state = m.step(props)
            all_violations.extend(state.violated_constraints)
            all_pending.extend(state.pending_obligations)
            if state.status == MonitorStatus.VIOLATED:
                any_violated = True

        status = MonitorStatus.VIOLATED if any_violated else MonitorStatus.PENDING

        return MonitorState(
            status=status,
            violated_constraints=all_violations,
            pending_obligations=all_pending,
            step_count=self.monitors[0]._step_count if self.monitors else 0
        )

    def would_violate(self, props: Dict[str, bool]) -> Tuple[bool, List[ViolationInfo]]:
        """Check if props would violate any constraint"""
        all_violations = []

        for m in self.monitors:
            would_viol, violations = m.would_violate(props)
            if would_viol:
                all_violations.extend(violations)

        return len(all_violations) > 0, all_violations


def create_monitor(constraints: List[LTLConstraint]) -> ConstraintMonitor:
    """Factory function to create appropriate monitor"""
    if len(constraints) == 0:
        raise ValueError("No constraints provided")
    elif len(constraints) == 1:
        return PatternMonitor(constraints[0])
    else:
        return CompositeMonitor(constraints)


# =============================================================================
# Explanation Generator
# =============================================================================

@dataclass
class ExplanationResult:
    """Result of violation explanation"""
    summary: str                      # Short summary
    details: List[str]                # Detailed explanation lines
    violated_constraints_nl: List[str]  # Original NL constraints that were violated
    proposition_changes: Dict[str, Tuple[bool, bool]]  # Prop -> (before, after)
    suggestions: List[str]            # Suggested alternatives
    reprompt_text: str               # Text to add to LLM reprompt


def explain_violation(violations: List[ViolationInfo],
                      current_props: Dict[str, bool],
                      proposed_props: Dict[str, bool]) -> ExplanationResult:
    """
    Generate detailed explanation for constraint violation.

    This is a key SafetyChip feature: explaining WHY an action is unsafe
    to enable informed reprompting.
    """
    if not violations:
        return ExplanationResult(
            summary="No violation",
            details=[],
            violated_constraints_nl=[],
            proposition_changes={},
            suggestions=[],
            reprompt_text=""
        )

    details = []
    nl_constraints = []
    prop_changes = {}
    suggestions = []

    for v in violations:
        details.append(f"- {v.explanation}")
        nl_constraints.append(v.constraint.original_nl)
        prop_changes[v.violated_proposition] = v.proposition_change
        if v.suggested_action not in suggestions:
            suggestions.append(v.suggested_action)

    # Create summary
    if len(violations) == 1:
        summary = violations[0].explanation
    else:
        summary = f"Multiple constraint violations ({len(violations)})"

    # Create reprompt text
    reprompt_lines = [
        "SAFETY VIOLATION DETECTED:",
        f"The proposed action would violate the following constraint(s):",
    ]
    for nl in nl_constraints:
        reprompt_lines.append(f"  - \"{nl}\"")

    reprompt_lines.append("\nProposition changes that cause violation:")
    for prop, (before, after) in prop_changes.items():
        reprompt_lines.append(f"  - {prop}: {before} -> {after}")

    reprompt_lines.append("\nSuggested alternatives:")
    for s in suggestions:
        reprompt_lines.append(f"  - {s}")

    reprompt_lines.append("\nPlease propose a different action that does not violate these constraints.")

    return ExplanationResult(
        summary=summary,
        details=details,
        violated_constraints_nl=nl_constraints,
        proposition_changes=prop_changes,
        suggestions=suggestions,
        reprompt_text="\n".join(reprompt_lines)
    )


def format_monitor_log(state: MonitorState) -> str:
    """Format monitor state for logging"""
    lines = [f"Monitor Status: {state.status.name}"]
    lines.append(f"Step: {state.step_count}")

    if state.violated_constraints:
        lines.append("Violations:")
        for v in state.violated_constraints:
            lines.append(f"  - {v.explanation}")

    if state.pending_obligations:
        lines.append("Pending obligations:")
        for p in state.pending_obligations:
            lines.append(f"  - {p}")

    return "\n".join(lines)
