"""
SafetyChip-lite: NL to LTL Translator

Rule-based natural language to LTL conversion with support for:
1. Always avoid: "never enter X", "avoid X" -> G(!in_X)
2. Precedence: "visit A before B" -> (!in_B) U in_A
3. Trigger-response: "if A then eventually B" -> G(in_A -> F(in_B))

This module focuses on deterministic rule-based translation for reproducibility.
LLM-based translation is available as an optional feature.
"""

import re
import logging
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum, auto

logger = logging.getLogger(__name__)


class ConstraintType(Enum):
    """Supported constraint pattern types"""
    ALWAYS_AVOID = auto()      # G(!p) - always avoid p
    ALWAYS_MAINTAIN = auto()   # G(p) - always maintain p
    EVENTUALLY = auto()        # F(p) - eventually reach p
    PRECEDENCE = auto()        # (!q) U p - p before q
    TRIGGER_RESPONSE = auto()  # G(p -> F(q)) - if p then eventually q
    CONJUNCTION = auto()       # Multiple constraints ANDed together
    UNKNOWN = auto()


@dataclass
class LTLConstraint:
    """A single LTL constraint with metadata"""
    original_nl: str                  # Original natural language
    ltl_formula: str                  # LTL formula string
    constraint_type: ConstraintType   # Pattern type
    propositions: Set[str] = field(default_factory=set)  # Involved propositions
    parameters: Dict = field(default_factory=dict)  # Pattern-specific params

    def __str__(self):
        return f"{self.original_nl} -> {self.ltl_formula}"


@dataclass
class TranslationResult:
    """Result of NL to LTL translation"""
    constraints: List[LTLConstraint]  # Individual constraints
    combined_ltl: str                 # Combined formula (AND of all)
    all_propositions: Set[str]        # All propositions used
    log: List[str] = field(default_factory=list)  # Translation log


class NL2LTLTranslator:
    """
    Rule-based Natural Language to LTL Translator.

    Supports three main patterns as specified in SafetyChip:
    1. Always avoid: G(!p)
    2. Precedence: (!q) U p
    3. Trigger-response: G(p -> F(q))

    Usage:
        translator = NL2LTLTranslator()
        result = translator.translate([
            "Never enter the storage area",
            "Visit checkpoint before goal"
        ])
        print(result.combined_ltl)
    """

    # ==========================================================================
    # Pattern definitions (English)
    # ==========================================================================

    AVOID_PATTERNS = [
        # "never enter X", "don't go to X", "avoid X"
        (r"(?:never|don't|do not|must not)\s+(?:enter|go to|visit|go into)\s+(?:the\s+)?(\w+)",
         ConstraintType.ALWAYS_AVOID),
        (r"(?:always\s+)?avoid\s+(?:the\s+)?(\w+)",
         ConstraintType.ALWAYS_AVOID),
        (r"(\w+)\s+(?:is\s+)?(?:forbidden|prohibited|off-limits|restricted)",
         ConstraintType.ALWAYS_AVOID),
        (r"stay\s+(?:away\s+)?(?:from|out of)\s+(?:the\s+)?(\w+)",
         ConstraintType.ALWAYS_AVOID),
        (r"do\s+not\s+(?:enter|visit)\s+(?:the\s+)?(\w+)",
         ConstraintType.ALWAYS_AVOID),
    ]

    PRECEDENCE_PATTERNS = [
        # "visit A before B", "go to A first then B"
        (r"(?:visit|go to)\s+(?:the\s+)?(\w+)\s+(?:before|prior to)\s+(?:the\s+)?(\w+)",
         ConstraintType.PRECEDENCE),
        (r"(?:the\s+)?(\w+)\s+(?:before|then)\s+(?:the\s+)?(\w+)",
         ConstraintType.PRECEDENCE),
        (r"(?:first|firstly)\s+(?:visit|go to)\s+(?:the\s+)?(\w+)(?:,)?\s+(?:then|and then)\s+(?:the\s+)?(\w+)",
         ConstraintType.PRECEDENCE),
        (r"(\w+)\s+must\s+(?:be\s+)?(?:visited|reached)\s+before\s+(\w+)",
         ConstraintType.PRECEDENCE),
    ]

    TRIGGER_RESPONSE_PATTERNS = [
        # "if A then eventually B", "whenever A, must B"
        (r"if\s+(?:you\s+)?(?:enter|visit|reach|go to)\s+(?:the\s+)?(\w+)(?:,)?\s+(?:then\s+)?(?:you\s+)?(?:must|should)\s+(?:eventually\s+)?(?:go to|visit|reach)\s+(?:the\s+)?(\w+)",
         ConstraintType.TRIGGER_RESPONSE),
        (r"whenever\s+(?:in|at|entering)\s+(?:the\s+)?(\w+)(?:,)?\s+(?:must|should)\s+(?:eventually\s+)?(?:visit|reach|go to)\s+(?:the\s+)?(\w+)",
         ConstraintType.TRIGGER_RESPONSE),
        (r"(?:entering|visiting)\s+(?:the\s+)?(\w+)\s+(?:requires|means)\s+(?:eventually\s+)?(?:visiting|reaching)\s+(?:the\s+)?(\w+)",
         ConstraintType.TRIGGER_RESPONSE),
    ]

    EVENTUALLY_PATTERNS = [
        # "eventually reach X", "must visit X"
        (r"(?:eventually|finally)\s+(?:reach|visit|go to)\s+(?:the\s+)?(\w+)",
         ConstraintType.EVENTUALLY),
        (r"must\s+(?:eventually\s+)?(?:reach|visit|go to)\s+(?:the\s+)?(\w+)",
         ConstraintType.EVENTUALLY),
        (r"(?:reach|get to|arrive at)\s+(?:the\s+)?(\w+)\s+(?:eventually|in the end)",
         ConstraintType.EVENTUALLY),
    ]

    MAINTAIN_PATTERNS = [
        # "always stay in X", "maintain X"
        (r"(?:always\s+)?(?:stay|remain)\s+(?:in|at|within)\s+(?:the\s+)?(\w+)",
         ConstraintType.ALWAYS_MAINTAIN),
        (r"(?:always\s+)?maintain\s+(\w+)",
         ConstraintType.ALWAYS_MAINTAIN),
        (r"keep\s+(\w+)\s+(?:true|active)",
         ConstraintType.ALWAYS_MAINTAIN),
    ]

    # ==========================================================================
    # Korean patterns
    # ==========================================================================

    AVOID_PATTERNS_KO = [
        (r"(\w+)(?:에|로|을|를)?\s*(?:들어가|진입하|가)(?:면|지)\s*(?:안|않)",
         ConstraintType.ALWAYS_AVOID),
        (r"(\w+)\s*(?:구역|영역|지역)(?:에|은|는)?\s*(?:금지|제한|출입금지)",
         ConstraintType.ALWAYS_AVOID),
        (r"(\w+)(?:을|를)?\s*(?:피하|회피)",
         ConstraintType.ALWAYS_AVOID),
    ]

    PRECEDENCE_PATTERNS_KO = [
        (r"(\w+)(?:을|를)?\s*(?:먼저|우선)\s*(?:방문|들른|가)\s*(?:뒤|후|다음)(?:에)?\s*(\w+)",
         ConstraintType.PRECEDENCE),
        (r"(\w+)\s*(?:전에|이전에)\s*(\w+)(?:을|를)?\s*(?:방문|가)",
         ConstraintType.PRECEDENCE),
    ]

    TRIGGER_RESPONSE_PATTERNS_KO = [
        (r"(\w+)(?:에|를)?\s*(?:가면|방문하면|들어가면)\s*(?:반드시|꼭)\s*(\w+)(?:에|를)?\s*(?:가|방문)",
         ConstraintType.TRIGGER_RESPONSE),
    ]

    def __init__(self, use_llm: bool = False, llm_model: str = "gpt-3.5-turbo"):
        """
        Args:
            use_llm: If True, use LLM for translation (requires API)
            llm_model: LLM model to use if use_llm is True
        """
        self.use_llm = use_llm
        self.llm_model = llm_model
        self.proposition_vocabulary: Set[str] = set()

    def set_proposition_vocabulary(self, propositions: List[str]) -> None:
        """Set the allowed proposition vocabulary"""
        self.proposition_vocabulary = set(propositions)

    def translate(self, constraints: List[str]) -> TranslationResult:
        """
        Translate a list of natural language constraints to LTL.

        Args:
            constraints: List of NL constraint strings

        Returns:
            TranslationResult with individual and combined LTL formulas
        """
        log = [f"Translating {len(constraints)} constraints..."]
        translated = []
        all_props = set()

        for nl in constraints:
            log.append(f"\nInput: {nl}")

            if self.use_llm:
                constraint = self._translate_with_llm(nl)
            else:
                constraint = self._translate_with_rules(nl)

            if constraint:
                translated.append(constraint)
                all_props.update(constraint.propositions)
                log.append(f"  -> LTL: {constraint.ltl_formula}")
                log.append(f"  -> Type: {constraint.constraint_type.name}")
                log.append(f"  -> Props: {constraint.propositions}")
            else:
                log.append(f"  -> FAILED: Could not translate")

        # Combine all constraints with AND
        if len(translated) == 0:
            combined = "true"
        elif len(translated) == 1:
            combined = translated[0].ltl_formula
        else:
            combined = " & ".join(f"({c.ltl_formula})" for c in translated)

        log.append(f"\nCombined LTL: {combined}")
        log.append(f"All propositions: {all_props}")

        return TranslationResult(
            constraints=translated,
            combined_ltl=combined,
            all_propositions=all_props,
            log=log
        )

    def _translate_with_rules(self, nl: str) -> Optional[LTLConstraint]:
        """Rule-based translation"""
        nl_lower = nl.lower().strip()

        # Try all pattern categories in order
        pattern_groups = [
            (self.AVOID_PATTERNS, self._make_avoid_constraint),
            (self.PRECEDENCE_PATTERNS, self._make_precedence_constraint),
            (self.TRIGGER_RESPONSE_PATTERNS, self._make_trigger_response_constraint),
            (self.EVENTUALLY_PATTERNS, self._make_eventually_constraint),
            (self.MAINTAIN_PATTERNS, self._make_maintain_constraint),
            # Korean patterns
            (self.AVOID_PATTERNS_KO, self._make_avoid_constraint),
            (self.PRECEDENCE_PATTERNS_KO, self._make_precedence_constraint),
            (self.TRIGGER_RESPONSE_PATTERNS_KO, self._make_trigger_response_constraint),
        ]

        for patterns, make_constraint in pattern_groups:
            for pattern, constraint_type in patterns:
                match = re.search(pattern, nl_lower, re.IGNORECASE)
                if match:
                    try:
                        return make_constraint(nl, match, constraint_type)
                    except Exception as e:
                        logger.debug(f"Pattern {pattern} failed: {e}")
                        continue

        logger.warning(f"Could not translate: {nl}")
        return None

    def _make_avoid_constraint(self, nl: str, match: re.Match,
                               ctype: ConstraintType) -> LTLConstraint:
        """Create always-avoid constraint: G(!in_X)"""
        zone = match.group(1).lower()
        prop = f"in_{zone}"
        ltl = f"G(!{prop})"

        return LTLConstraint(
            original_nl=nl,
            ltl_formula=ltl,
            constraint_type=ctype,
            propositions={prop},
            parameters={"avoid_zone": zone}
        )

    def _make_precedence_constraint(self, nl: str, match: re.Match,
                                    ctype: ConstraintType) -> LTLConstraint:
        """Create precedence constraint: (!in_B) U in_A (A before B)"""
        first = match.group(1).lower()
        second = match.group(2).lower()
        prop_first = f"in_{first}"
        prop_second = f"in_{second}"
        ltl = f"(!{prop_second}) U {prop_first}"

        return LTLConstraint(
            original_nl=nl,
            ltl_formula=ltl,
            constraint_type=ctype,
            propositions={prop_first, prop_second},
            parameters={"first": first, "second": second}
        )

    def _make_trigger_response_constraint(self, nl: str, match: re.Match,
                                          ctype: ConstraintType) -> LTLConstraint:
        """Create trigger-response constraint: G(in_A -> F(in_B))"""
        trigger = match.group(1).lower()
        response = match.group(2).lower()
        prop_trigger = f"in_{trigger}"
        prop_response = f"in_{response}"
        ltl = f"G({prop_trigger} -> F({prop_response}))"

        return LTLConstraint(
            original_nl=nl,
            ltl_formula=ltl,
            constraint_type=ctype,
            propositions={prop_trigger, prop_response},
            parameters={"trigger": trigger, "response": response}
        )

    def _make_eventually_constraint(self, nl: str, match: re.Match,
                                    ctype: ConstraintType) -> LTLConstraint:
        """Create eventually constraint: F(in_X)"""
        goal = match.group(1).lower()
        prop = f"in_{goal}"
        ltl = f"F({prop})"

        return LTLConstraint(
            original_nl=nl,
            ltl_formula=ltl,
            constraint_type=ctype,
            propositions={prop},
            parameters={"goal": goal}
        )

    def _make_maintain_constraint(self, nl: str, match: re.Match,
                                  ctype: ConstraintType) -> LTLConstraint:
        """Create always-maintain constraint: G(X)"""
        prop_name = match.group(1).lower()
        # Check if it's a zone or a boolean property
        if prop_name.startswith("in_"):
            prop = prop_name
        else:
            prop = f"in_{prop_name}"
        ltl = f"G({prop})"

        return LTLConstraint(
            original_nl=nl,
            ltl_formula=ltl,
            constraint_type=ctype,
            propositions={prop},
            parameters={"maintain": prop_name}
        )

    def _translate_with_llm(self, nl: str) -> Optional[LTLConstraint]:
        """LLM-based translation (placeholder)"""
        logger.warning("LLM translation not implemented, falling back to rules")
        return self._translate_with_rules(nl)


def format_translation_log(result: TranslationResult) -> str:
    """Format translation result for logging"""
    lines = ["=" * 60]
    lines.append("SafetyChip-lite NL->LTL Translation Log")
    lines.append("=" * 60)
    lines.extend(result.log)
    lines.append("=" * 60)
    return "\n".join(lines)
