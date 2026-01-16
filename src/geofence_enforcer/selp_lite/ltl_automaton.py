"""
LTL Automaton Module - LTL to Automaton conversion and runtime monitoring

This module converts LTL formulas to automata for:
1. Equivalence checking (via trace sampling)
2. Runtime monitoring during constrained decoding
3. Action masking based on LTL constraint satisfaction

Supported LTL Operators:
- G (Globally/Always): G(phi) - phi must hold in all future states
- F (Finally/Eventually): F(phi) - phi must hold at some future state
- U (Until): phi U psi - phi holds until psi becomes true
- X (Next): X(phi) - phi holds in the next state
- ! (Not): !phi - negation
- & (And): phi & psi - conjunction
- | (Or): phi | psi - disjunction
- -> (Implies): phi -> psi - implication

Implementation Notes:
- Uses spot library if available (recommended)
- Falls back to pattern-based automaton for common safety patterns
"""

import re
import logging
from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# Try to import spot for proper LTL handling
SPOT_AVAILABLE = False
try:
    import spot
    SPOT_AVAILABLE = True
    logger.info("spot library available - using full LTL support")
except ImportError:
    logger.warning("spot library not available - using pattern-based fallback")


class AutomatonState(Enum):
    """State of LTL automaton monitoring"""
    ACCEPTING = auto()      # Currently in accepting state
    NON_ACCEPTING = auto()  # Not in accepting state but can still reach one
    VIOLATED = auto()       # Constraint permanently violated
    UNKNOWN = auto()        # Cannot determine


@dataclass
class LTLFormula:
    """Parsed LTL formula representation"""
    raw: str                          # Original formula string
    normalized: str                   # Normalized formula
    propositions: Set[str] = field(default_factory=set)  # Atomic propositions used

    def __hash__(self):
        return hash(self.normalized)

    def __eq__(self, other):
        if isinstance(other, LTLFormula):
            return self.normalized == other.normalized
        return False


@dataclass
class MonitorState:
    """State of the runtime monitor"""
    automaton_state: Any              # Current automaton state
    is_accepting: bool                # Whether current state is accepting
    can_reach_accepting: bool         # Whether accepting state is reachable
    status: AutomatonState            # Overall status
    visited_states: Set = field(default_factory=set)  # For cycle detection


class LTLAutomaton(ABC):
    """Abstract base class for LTL automaton"""

    @abstractmethod
    def reset(self) -> MonitorState:
        """Reset automaton to initial state"""
        pass

    @abstractmethod
    def step(self, propositions: Dict[str, bool]) -> MonitorState:
        """
        Advance automaton by one step given current propositions.

        Args:
            propositions: Dictionary mapping proposition names to truth values

        Returns:
            Updated monitor state
        """
        pass

    @abstractmethod
    def would_violate(self, propositions: Dict[str, bool]) -> bool:
        """
        Check if taking a step with these propositions would violate the constraint.
        Does not actually advance the automaton state.
        """
        pass

    @abstractmethod
    def can_satisfy(self) -> bool:
        """Check if the constraint can still be satisfied from current state"""
        pass

    def evaluate_trace(self, trace: List[Dict[str, bool]]) -> bool:
        """
        Evaluate whether a complete trace satisfies the LTL formula.

        Args:
            trace: List of proposition dictionaries (one per step)

        Returns:
            True if trace satisfies the formula
        """
        self.reset()
        for props in trace:
            state = self.step(props)
            if state.status == AutomatonState.VIOLATED:
                return False
        return state.is_accepting or state.can_reach_accepting


# =============================================================================
# Pattern-based Automaton (Fallback without spot)
# =============================================================================

class PatternAutomaton(LTLAutomaton):
    """
    Pattern-based automaton for common LTL safety/liveness patterns.

    Supported patterns:
    - G(!p): Always avoid p (safety)
    - G(p): Always maintain p (invariant)
    - F(p): Eventually reach p (reachability)
    - G(p -> F(q)): If p then eventually q (response)
    - p U q: p until q (until)
    - G(!p & !q): Always avoid multiple propositions
    """

    def __init__(self, formula: LTLFormula):
        self.formula = formula
        self.pattern_type: str = ""
        self.pattern_props: Dict[str, str] = {}
        self._parse_pattern()
        self._current_state: Dict = {}
        self.reset()

    def _parse_pattern(self):
        """Parse formula to identify pattern type"""
        f = self.formula.normalized.strip()

        # Remove outer parentheses
        while f.startswith('(') and f.endswith(')'):
            # Check if they match
            depth = 0
            matched = True
            for i, c in enumerate(f):
                if c == '(':
                    depth += 1
                elif c == ')':
                    depth -= 1
                if depth == 0 and i < len(f) - 1:
                    matched = False
                    break
            if matched:
                f = f[1:-1].strip()
            else:
                break

        # Pattern: G(!p) - Always avoid p
        match = re.match(r'^G\s*\(\s*!\s*(\w+)\s*\)$', f)
        if match:
            self.pattern_type = "always_avoid"
            self.pattern_props["avoid"] = match.group(1)
            return

        # Pattern: !F(p) - Never eventually p (equivalent to G(!p) for safety)
        match = re.match(r'^!\s*F\s*\(\s*(\w+)\s*\)$', f)
        if match:
            self.pattern_type = "always_avoid"  # Treat as equivalent
            self.pattern_props["avoid"] = match.group(1)
            return

        # Pattern: G(p) - Always maintain p
        match = re.match(r'^G\s*\(\s*(\w+)\s*\)$', f)
        if match:
            self.pattern_type = "always_maintain"
            self.pattern_props["maintain"] = match.group(1)
            return

        # Pattern: F(p) - Eventually reach p
        match = re.match(r'^F\s*\(\s*(\w+)\s*\)$', f)
        if match:
            self.pattern_type = "eventually"
            self.pattern_props["goal"] = match.group(1)
            return

        # Pattern: G(!p & !q) or G(!(p | q)) - Avoid multiple
        match = re.match(r'^G\s*\(\s*!\s*(\w+)\s*&\s*!\s*(\w+)\s*\)$', f)
        if match:
            self.pattern_type = "avoid_multiple"
            self.pattern_props["avoid1"] = match.group(1)
            self.pattern_props["avoid2"] = match.group(2)
            return

        # Pattern: G(p -> F(q)) - Response
        match = re.match(r'^G\s*\(\s*(\w+)\s*->\s*F\s*\(\s*(\w+)\s*\)\s*\)$', f)
        if match:
            self.pattern_type = "response"
            self.pattern_props["trigger"] = match.group(1)
            self.pattern_props["response"] = match.group(2)
            return

        # Pattern: p U q - Until
        match = re.match(r'^(\w+)\s*U\s*(\w+)$', f)
        if match:
            self.pattern_type = "until"
            self.pattern_props["hold"] = match.group(1)
            self.pattern_props["goal"] = match.group(2)
            return

        # Pattern: F(p) & G(!q) - Reach p while avoiding q
        match = re.match(r'^F\s*\(\s*(\w+)\s*\)\s*&\s*G\s*\(\s*!\s*(\w+)\s*\)$', f)
        if match:
            self.pattern_type = "reach_avoid"
            self.pattern_props["goal"] = match.group(1)
            self.pattern_props["avoid"] = match.group(2)
            return

        # Complex pattern - use generic monitoring
        self.pattern_type = "generic"
        logger.warning(f"Using generic pattern for: {f}")

    def reset(self) -> MonitorState:
        """Reset to initial state"""
        self._current_state = {
            "violated": False,
            "satisfied": False,
            "pending_responses": [],  # For response pattern
            "steps": 0,
        }
        return self._get_monitor_state()

    def step(self, propositions: Dict[str, bool]) -> MonitorState:
        """Advance automaton state"""
        self._current_state["steps"] += 1

        if self._current_state["violated"]:
            return self._get_monitor_state()

        if self.pattern_type == "always_avoid":
            prop = self.pattern_props["avoid"]
            if propositions.get(prop, False):
                self._current_state["violated"] = True
                logger.debug(f"VIOLATED: {prop} became true (should avoid)")

        elif self.pattern_type == "always_maintain":
            prop = self.pattern_props["maintain"]
            if not propositions.get(prop, True):
                self._current_state["violated"] = True
                logger.debug(f"VIOLATED: {prop} became false (should maintain)")

        elif self.pattern_type == "eventually":
            prop = self.pattern_props["goal"]
            if propositions.get(prop, False):
                self._current_state["satisfied"] = True

        elif self.pattern_type == "avoid_multiple":
            prop1 = self.pattern_props["avoid1"]
            prop2 = self.pattern_props["avoid2"]
            if propositions.get(prop1, False) or propositions.get(prop2, False):
                self._current_state["violated"] = True

        elif self.pattern_type == "response":
            trigger = self.pattern_props["trigger"]
            response = self.pattern_props["response"]
            # Add pending response if trigger is true
            if propositions.get(trigger, False):
                self._current_state["pending_responses"].append(self._current_state["steps"])
            # Clear pending if response is true
            if propositions.get(response, False):
                self._current_state["pending_responses"] = []

        elif self.pattern_type == "until":
            hold = self.pattern_props["hold"]
            goal = self.pattern_props["goal"]
            if propositions.get(goal, False):
                self._current_state["satisfied"] = True
            elif not self._current_state["satisfied"] and not propositions.get(hold, True):
                self._current_state["violated"] = True

        elif self.pattern_type == "reach_avoid":
            goal = self.pattern_props["goal"]
            avoid = self.pattern_props["avoid"]
            if propositions.get(avoid, False):
                self._current_state["violated"] = True
            elif propositions.get(goal, False):
                self._current_state["satisfied"] = True

        return self._get_monitor_state()

    def would_violate(self, propositions: Dict[str, bool]) -> bool:
        """Check if these propositions would violate (without advancing state)"""
        if self._current_state["violated"]:
            return True

        if self.pattern_type == "always_avoid":
            return propositions.get(self.pattern_props["avoid"], False)

        elif self.pattern_type == "always_maintain":
            return not propositions.get(self.pattern_props["maintain"], True)

        elif self.pattern_type == "avoid_multiple":
            return (propositions.get(self.pattern_props["avoid1"], False) or
                    propositions.get(self.pattern_props["avoid2"], False))

        elif self.pattern_type == "until":
            goal = self.pattern_props["goal"]
            hold = self.pattern_props["hold"]
            if self._current_state["satisfied"]:
                return False
            return not propositions.get(goal, False) and not propositions.get(hold, True)

        elif self.pattern_type == "reach_avoid":
            return propositions.get(self.pattern_props["avoid"], False)

        return False

    def can_satisfy(self) -> bool:
        """Check if constraint can still be satisfied"""
        if self._current_state["violated"]:
            return False
        if self._current_state["satisfied"]:
            return True
        # For safety patterns (always_avoid, etc.), can always satisfy if not violated
        if self.pattern_type in ["always_avoid", "always_maintain", "avoid_multiple"]:
            return True
        return True  # Optimistic for other patterns

    def _get_monitor_state(self) -> MonitorState:
        """Get current monitor state"""
        if self._current_state["violated"]:
            status = AutomatonState.VIOLATED
            is_accepting = False
            can_reach = False
        elif self._current_state["satisfied"]:
            status = AutomatonState.ACCEPTING
            is_accepting = True
            can_reach = True
        else:
            status = AutomatonState.NON_ACCEPTING
            is_accepting = False
            can_reach = True

        return MonitorState(
            automaton_state=self._current_state.copy(),
            is_accepting=is_accepting,
            can_reach_accepting=can_reach,
            status=status
        )


# =============================================================================
# Spot-based Automaton (when spot library is available)
# =============================================================================

class SpotAutomaton(LTLAutomaton):
    """
    Full LTL automaton using the spot library.
    Supports arbitrary LTL formulas.
    """

    def __init__(self, formula: LTLFormula):
        if not SPOT_AVAILABLE:
            raise ImportError("spot library not available")

        self.formula = formula
        self._build_automaton()
        self.reset()

    def _build_automaton(self):
        """Build Büchi automaton from LTL formula"""
        # Parse formula
        f = spot.formula(self.formula.normalized)

        # Convert to automaton
        self.automaton = spot.translate(f, 'BA', 'High', 'SBAcc')

        # Get BDD dictionary for proposition evaluation
        self.bdd_dict = self.automaton.get_dict()

        # Get initial state
        self.initial_state = self.automaton.get_init_state_number()

        logger.debug(f"Built automaton with {self.automaton.num_states()} states")

    def reset(self) -> MonitorState:
        """Reset to initial state"""
        self._current_state = self.initial_state
        self._visited = {self._current_state}
        return self._get_monitor_state()

    def step(self, propositions: Dict[str, bool]) -> MonitorState:
        """Advance automaton by one step"""
        # Build BDD for current propositions
        bdd = self._props_to_bdd(propositions)

        # Find matching transition
        next_state = None
        for edge in self.automaton.out(self._current_state):
            if spot.bdd_to_formula(edge.cond, self.bdd_dict).is_tt() or \
               self._bdd_matches(edge.cond, bdd):
                next_state = edge.dst
                break

        if next_state is None:
            # No valid transition - violated
            self._current_state = -1  # Invalid state
        else:
            self._current_state = next_state
            self._visited.add(next_state)

        return self._get_monitor_state()

    def would_violate(self, propositions: Dict[str, bool]) -> bool:
        """Check if these propositions would violate"""
        if self._current_state == -1:
            return True

        bdd = self._props_to_bdd(propositions)

        # Check if any valid transition exists
        for edge in self.automaton.out(self._current_state):
            if self._bdd_matches(edge.cond, bdd):
                return False

        return True  # No valid transition

    def can_satisfy(self) -> bool:
        """Check if accepting state is reachable"""
        if self._current_state == -1:
            return False

        # BFS to find accepting state
        visited = set()
        queue = [self._current_state]

        while queue:
            state = queue.pop(0)
            if state in visited:
                continue
            visited.add(state)

            # Check if accepting
            if self.automaton.state_is_accepting(state):
                return True

            # Add successors
            for edge in self.automaton.out(state):
                if edge.dst not in visited:
                    queue.append(edge.dst)

        return False

    def _props_to_bdd(self, propositions: Dict[str, bool]) -> Any:
        """Convert proposition dict to BDD"""
        bdd = spot.bddtrue
        for name, value in propositions.items():
            try:
                var = spot.formula_ap(name)
                var_bdd = spot.formula_to_bdd(var, self.bdd_dict, self.automaton)
                if value:
                    bdd &= var_bdd
                else:
                    bdd &= -var_bdd
            except:
                pass  # Unknown proposition
        return bdd

    def _bdd_matches(self, cond, props_bdd) -> bool:
        """Check if condition BDD is satisfied by propositions BDD"""
        return (cond & props_bdd) != spot.bddfalse

    def _get_monitor_state(self) -> MonitorState:
        """Get current monitor state"""
        if self._current_state == -1:
            return MonitorState(
                automaton_state=None,
                is_accepting=False,
                can_reach_accepting=False,
                status=AutomatonState.VIOLATED
            )

        is_accepting = self.automaton.state_is_accepting(self._current_state)
        can_reach = self.can_satisfy()

        if is_accepting:
            status = AutomatonState.ACCEPTING
        elif can_reach:
            status = AutomatonState.NON_ACCEPTING
        else:
            status = AutomatonState.VIOLATED

        return MonitorState(
            automaton_state=self._current_state,
            is_accepting=is_accepting,
            can_reach_accepting=can_reach,
            status=status
        )


# =============================================================================
# Factory and Utilities
# =============================================================================

def parse_ltl(formula_str: str) -> LTLFormula:
    """
    Parse LTL formula string.

    Args:
        formula_str: LTL formula (e.g., "G(!in_storage)")

    Returns:
        Parsed LTL formula
    """
    # Normalize formula
    normalized = formula_str.strip()
    normalized = re.sub(r'\s+', ' ', normalized)
    normalized = normalized.replace('always', 'G')
    normalized = normalized.replace('eventually', 'F')
    normalized = normalized.replace('until', 'U')
    normalized = normalized.replace('next', 'X')
    normalized = normalized.replace('not ', '!')
    normalized = normalized.replace(' and ', ' & ')
    normalized = normalized.replace(' or ', ' | ')
    normalized = normalized.replace('implies', '->')

    # Extract propositions (identifiers that aren't operators)
    props = set(re.findall(r'\b([a-z_][a-z0-9_]*)\b', normalized.lower()))
    props -= {'g', 'f', 'u', 'x'}  # Remove operators

    return LTLFormula(
        raw=formula_str,
        normalized=normalized,
        propositions=props
    )


def create_automaton(formula: LTLFormula) -> LTLAutomaton:
    """
    Create appropriate automaton for the formula.
    Uses spot if available, otherwise pattern-based fallback.
    """
    if SPOT_AVAILABLE:
        try:
            return SpotAutomaton(formula)
        except Exception as e:
            logger.warning(f"spot automaton failed, using pattern fallback: {e}")

    return PatternAutomaton(formula)


def check_trace_equivalence(
    formula1: LTLFormula,
    formula2: LTLFormula,
    num_traces: int = 100,
    trace_length: int = 20,
    propositions: List[str] = None
) -> bool:
    """
    Check approximate equivalence by sampling traces.

    Args:
        formula1, formula2: Formulas to compare
        num_traces: Number of random traces to sample
        trace_length: Length of each trace
        propositions: List of proposition names (auto-detected if None)

    Returns:
        True if formulas agree on all sampled traces
    """
    import random

    # Get propositions
    props = propositions or list(formula1.propositions | formula2.propositions)

    # Create automata
    auto1 = create_automaton(formula1)
    auto2 = create_automaton(formula2)

    # Sample traces
    for _ in range(num_traces):
        trace = []
        for _ in range(trace_length):
            step_props = {p: random.random() > 0.5 for p in props}
            trace.append(step_props)

        # Evaluate both formulas
        result1 = auto1.evaluate_trace(trace)
        result2 = auto2.evaluate_trace(trace)

        if result1 != result2:
            return False

    return True


# =============================================================================
# Common Safety Patterns
# =============================================================================

def always_avoid(proposition: str) -> LTLFormula:
    """Create G(!p) formula - always avoid proposition"""
    return parse_ltl(f"G(!{proposition})")


def always_maintain(proposition: str) -> LTLFormula:
    """Create G(p) formula - always maintain proposition"""
    return parse_ltl(f"G({proposition})")


def eventually_reach(proposition: str) -> LTLFormula:
    """Create F(p) formula - eventually reach proposition"""
    return parse_ltl(f"F({proposition})")


def reach_while_avoiding(goal: str, avoid: str) -> LTLFormula:
    """Create F(goal) & G(!avoid) formula"""
    return parse_ltl(f"F({goal}) & G(!{avoid})")


def visit_in_order(first: str, second: str) -> LTLFormula:
    """Create (!second) U first & F(second) formula - visit first before second"""
    return parse_ltl(f"(!{second}) U {first} & F({second})")
