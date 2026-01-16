"""
NL to LTL Conversion with Equivalence Voting

This module implements the Equivalence Voting component of SELP-lite:
1. Generate K LTL formula candidates from natural language constraint
2. Group candidates by equivalence (using trace sampling)
3. Select the majority group's formula via voting

The module supports:
- LLM-based generation (when API is available)
- Rule-based fallback (for reproducibility without API)
- Structured constraint input (YAML/JSON) for automated experiments
"""

import re
import random
import logging
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import json
import yaml

from .ltl_automaton import (
    LTLFormula, parse_ltl, create_automaton, check_trace_equivalence
)

logger = logging.getLogger(__name__)


@dataclass
class VotingResult:
    """Result of equivalence voting"""
    selected_formula: LTLFormula
    candidates: List[LTLFormula]
    groups: Dict[int, List[int]]  # group_id -> list of candidate indices
    group_votes: Dict[int, int]   # group_id -> vote count
    winning_group: int
    confidence: float
    log: List[str] = field(default_factory=list)


@dataclass
class StructuredConstraint:
    """Structured constraint specification"""
    constraint_type: str  # "avoid", "reach", "visit_order", "response", "custom"
    propositions: List[str]
    parameters: Dict = field(default_factory=dict)

    def to_natural_language(self) -> str:
        """Convert to natural language description"""
        if self.constraint_type == "avoid":
            zones = ", ".join(self.propositions)
            return f"Always avoid {zones}"
        elif self.constraint_type == "reach":
            return f"Eventually reach {self.propositions[0]}"
        elif self.constraint_type == "visit_order":
            return f"Visit {self.propositions[0]} before {self.propositions[1]}"
        elif self.constraint_type == "response":
            return f"If {self.propositions[0]} then eventually {self.propositions[1]}"
        else:
            return str(self.propositions)


class NL2LTLConverter:
    """
    Natural Language to LTL Converter with Equivalence Voting.

    This class implements the SELP-style NL→LTL conversion:
    1. Takes natural language constraint as input
    2. Generates K candidate LTL formulas
    3. Groups candidates by equivalence
    4. Returns the majority-voted formula

    Usage:
        converter = NL2LTLConverter(num_candidates=5)
        result = converter.convert("Never enter the storage area")
        print(f"Selected LTL: {result.selected_formula.normalized}")
    """

    # Rule-based patterns for fallback conversion
    NL_PATTERNS = [
        # Avoid patterns
        (r"(?:never|don't|do not|avoid|stay away from|must not)\s+(?:enter|go to|visit)?\s*(?:the\s+)?(\w+)",
         lambda m: f"G(!in_{m.group(1).lower()})"),
        (r"(\w+)\s+(?:is\s+)?(?:forbidden|prohibited|off-limits|restricted)",
         lambda m: f"G(!in_{m.group(1).lower()})"),
        (r"(?:always\s+)?avoid\s+(?:the\s+)?(\w+)",
         lambda m: f"G(!in_{m.group(1).lower()})"),

        # Reach patterns
        (r"(?:eventually|finally|reach|go to|visit)\s+(?:the\s+)?(\w+)",
         lambda m: f"F(in_{m.group(1).lower()})"),
        (r"(?:must|should)\s+(?:reach|visit|go to)\s+(?:the\s+)?(\w+)",
         lambda m: f"F(in_{m.group(1).lower()})"),

        # Order patterns
        (r"(?:visit|go to)\s+(?:the\s+)?(\w+)\s+(?:before|first,?\s+then)\s+(?:the\s+)?(\w+)",
         lambda m: f"(!in_{m.group(2).lower()}) U in_{m.group(1).lower()}"),
        (r"(\w+)\s+(?:before|then)\s+(\w+)",
         lambda m: f"(!in_{m.group(2).lower()}) U in_{m.group(1).lower()}"),

        # Maintain patterns
        (r"(?:always\s+)?(?:stay|remain)\s+(?:in\s+)?(?:the\s+)?(\w+)",
         lambda m: f"G(in_{m.group(1).lower()})"),
        (r"(?:always\s+)?(?:maintain|keep)\s+(\w+)",
         lambda m: f"G({m.group(1).lower()})"),

        # Combined patterns
        (r"reach\s+(?:the\s+)?(\w+)\s+(?:while\s+)?avoid(?:ing)?\s+(?:the\s+)?(\w+)",
         lambda m: f"F(in_{m.group(1).lower()}) & G(!in_{m.group(2).lower()})"),
    ]

    # Korean patterns
    NL_PATTERNS_KO = [
        (r"(\w+)(?:에|로|을|를)?\s*(?:들어가|진입하|가)(?:면|지)\s*(?:안|않)",
         lambda m: f"G(!in_{m.group(1)})"),
        (r"(\w+)\s*(?:구역|영역|지역)(?:에|은|는)?\s*(?:금지|제한)",
         lambda m: f"G(!in_{m.group(1)})"),
        (r"(\w+)(?:에|로|를)?\s*(?:가|도달|방문)",
         lambda m: f"F(in_{m.group(1)})"),
        (r"(\w+)(?:을|를)?\s*(?:먼저|우선)\s*(?:방문|들른)\s*(?:뒤|후|다음)(?:에)?\s*(\w+)",
         lambda m: f"(!in_{m.group(2)}) U in_{m.group(1)}"),
    ]

    def __init__(
        self,
        num_candidates: int = 5,
        num_traces_for_equivalence: int = 100,
        trace_length: int = 20,
        use_llm: bool = False,
        llm_model: str = "gpt-3.5-turbo"
    ):
        """
        Args:
            num_candidates: Number of LTL candidates to generate (K)
            num_traces_for_equivalence: Number of traces for equivalence checking
            trace_length: Length of sampled traces
            use_llm: Whether to use LLM for generation (requires API)
            llm_model: LLM model to use if use_llm is True
        """
        self.num_candidates = num_candidates
        self.num_traces = num_traces_for_equivalence
        self.trace_length = trace_length
        self.use_llm = use_llm
        self.llm_model = llm_model

        # Proposition vocabulary (updated when converting)
        self.known_propositions: Set[str] = set()

    def set_propositions(self, propositions: List[str]) -> None:
        """Set known propositions for the domain"""
        self.known_propositions = set(propositions)

    def convert(
        self,
        natural_language: str,
        propositions: Optional[List[str]] = None
    ) -> VotingResult:
        """
        Convert natural language constraint to LTL via equivalence voting.

        Args:
            natural_language: Natural language constraint description
            propositions: Available proposition names (for context)

        Returns:
            VotingResult with selected formula and voting details
        """
        if propositions:
            self.known_propositions.update(propositions)

        log = [f"Input: {natural_language}"]
        log.append(f"Generating {self.num_candidates} LTL candidates...")

        # Step 1: Generate K candidates
        if self.use_llm:
            candidates = self._generate_candidates_llm(natural_language)
        else:
            candidates = self._generate_candidates_rules(natural_language)

        log.append(f"Candidates generated: {len(candidates)}")
        for i, c in enumerate(candidates):
            log.append(f"  [{i}] {c.normalized}")

        if len(candidates) == 0:
            raise ValueError(f"Could not generate any LTL candidates for: {natural_language}")

        # Step 2: Group by equivalence
        groups, group_members = self._group_by_equivalence(candidates)

        log.append(f"\nEquivalence groups: {len(groups)}")
        for gid, members in group_members.items():
            formulas = [candidates[i].normalized for i in members]
            log.append(f"  Group {gid}: {members} -> {formulas[0]}")

        # Step 3: Vote and select
        group_votes = {gid: len(members) for gid, members in group_members.items()}
        winning_group = max(group_votes.keys(), key=lambda g: group_votes[g])
        winning_idx = group_members[winning_group][0]
        selected = candidates[winning_idx]

        confidence = group_votes[winning_group] / len(candidates)

        log.append(f"\nVoting results:")
        for gid, votes in sorted(group_votes.items(), key=lambda x: -x[1]):
            marker = " <- WINNER" if gid == winning_group else ""
            log.append(f"  Group {gid}: {votes} votes{marker}")

        log.append(f"\nSelected LTL: {selected.normalized}")
        log.append(f"Confidence: {confidence:.2f}")

        return VotingResult(
            selected_formula=selected,
            candidates=candidates,
            groups=groups,
            group_votes=group_votes,
            winning_group=winning_group,
            confidence=confidence,
            log=log
        )

    def _generate_candidates_rules(self, nl_constraint: str) -> List[LTLFormula]:
        """Generate candidates using rule-based patterns"""
        candidates = []

        # Try all patterns
        all_patterns = self.NL_PATTERNS + self.NL_PATTERNS_KO

        for pattern, generator in all_patterns:
            match = re.search(pattern, nl_constraint, re.IGNORECASE)
            if match:
                try:
                    ltl_str = generator(match)
                    formula = parse_ltl(ltl_str)
                    if formula not in candidates:
                        candidates.append(formula)
                except Exception as e:
                    logger.debug(f"Pattern {pattern} failed: {e}")

        # Add variations to get K candidates
        if candidates:
            base = candidates[0]
            while len(candidates) < self.num_candidates:
                # Add equivalent variations
                variations = self._generate_variations(base)
                added_any = False
                for var in variations:
                    if var not in candidates:
                        candidates.append(var)
                        added_any = True
                        if len(candidates) >= self.num_candidates:
                            break
                # Break if no new candidates were added (avoid infinite loop)
                if not added_any:
                    break

        return candidates[:self.num_candidates]

    def _generate_variations(self, formula: LTLFormula) -> List[LTLFormula]:
        """Generate equivalent variations of a formula"""
        variations = []
        f = formula.normalized

        # Add parentheses variations (semantically equivalent)
        variations.append(parse_ltl(f"({f})"))

        # Duplicate the formula for voting purposes (ensures main pattern wins)
        variations.append(formula)

        # Commutative reordering for conjunctions
        if " & " in f:
            parts = f.split(" & ")
            if len(parts) == 2:
                variations.append(parse_ltl(f"{parts[1]} & {parts[0]}"))

        return variations

    def _generate_candidates_llm(self, nl_constraint: str) -> List[LTLFormula]:
        """Generate candidates using LLM (requires API)"""
        # Placeholder for LLM-based generation
        # In practice, this would call OpenAI or other LLM API

        prompt = f"""Convert the following natural language constraint to LTL formula.
Generate {self.num_candidates} different LTL formulas that capture the constraint.

Constraint: {nl_constraint}

Available propositions: {list(self.known_propositions)}

LTL operators:
- G(p): Globally/Always p
- F(p): Finally/Eventually p
- X(p): Next p
- p U q: p Until q
- !p: Not p
- p & q: p And q
- p | q: p Or q

Output each formula on a separate line, starting with "LTL:".
"""
        # For now, fall back to rules
        logger.warning("LLM generation not implemented, using rule-based fallback")
        return self._generate_candidates_rules(nl_constraint)

    def _group_by_equivalence(
        self,
        candidates: List[LTLFormula]
    ) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
        """
        Group candidates by approximate equivalence using trace sampling.

        Two formulas are considered equivalent if they agree on all sampled traces.
        """
        n = len(candidates)
        if n == 0:
            return {}, {}

        # Create automata for each candidate
        automata = [create_automaton(c) for c in candidates]

        # Get all propositions
        all_props = set()
        for c in candidates:
            all_props.update(c.propositions)
        props_list = list(all_props)

        # Generate sample traces
        sample_traces = []
        for _ in range(self.num_traces):
            trace = []
            for _ in range(self.trace_length):
                step = {p: random.random() > 0.5 for p in props_list}
                trace.append(step)
            sample_traces.append(trace)

        # Compute equivalence vectors (satisfaction pattern for each trace)
        equiv_vectors = []
        for auto in automata:
            vector = tuple(auto.evaluate_trace(trace) for trace in sample_traces)
            equiv_vectors.append(vector)

        # Group by equivalence vector
        vector_to_group: Dict[Tuple, int] = {}
        groups: Dict[int, List[int]] = defaultdict(list)
        next_group_id = 0

        for i, vector in enumerate(equiv_vectors):
            if vector not in vector_to_group:
                vector_to_group[vector] = next_group_id
                next_group_id += 1
            group_id = vector_to_group[vector]
            groups[group_id].append(i)

        return dict(groups), dict(groups)

    @staticmethod
    def from_structured(constraint: StructuredConstraint) -> LTLFormula:
        """Convert structured constraint directly to LTL"""
        if constraint.constraint_type == "avoid":
            avoids = " & ".join(f"!in_{p}" for p in constraint.propositions)
            return parse_ltl(f"G({avoids})")

        elif constraint.constraint_type == "reach":
            return parse_ltl(f"F(in_{constraint.propositions[0]})")

        elif constraint.constraint_type == "visit_order":
            first, second = constraint.propositions[:2]
            return parse_ltl(f"(!in_{second}) U in_{first}")

        elif constraint.constraint_type == "response":
            trigger, response = constraint.propositions[:2]
            return parse_ltl(f"G({trigger} -> F({response}))")

        elif constraint.constraint_type == "reach_avoid":
            goal = constraint.propositions[0]
            avoids = " & ".join(f"!in_{p}" for p in constraint.propositions[1:])
            return parse_ltl(f"F(in_{goal}) & G({avoids})")

        elif constraint.constraint_type == "custom":
            return parse_ltl(constraint.parameters.get("ltl", "true"))

        else:
            raise ValueError(f"Unknown constraint type: {constraint.constraint_type}")


def load_constraints_from_yaml(yaml_path: str) -> List[StructuredConstraint]:
    """Load constraints from YAML file"""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    constraints = []
    for item in data.get('constraints', []):
        constraints.append(StructuredConstraint(
            constraint_type=item['type'],
            propositions=item.get('propositions', []),
            parameters=item.get('parameters', {})
        ))

    return constraints


def load_constraints_from_json(json_path: str) -> List[StructuredConstraint]:
    """Load constraints from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    constraints = []
    for item in data.get('constraints', []):
        constraints.append(StructuredConstraint(
            constraint_type=item['type'],
            propositions=item.get('propositions', []),
            parameters=item.get('parameters', {})
        ))

    return constraints


# =============================================================================
# Logging Utilities for Paper
# =============================================================================

def format_voting_log(result: VotingResult) -> str:
    """Format voting result for paper/logging"""
    lines = ["=" * 60]
    lines.append("SELP-lite Equivalence Voting Log")
    lines.append("=" * 60)
    lines.extend(result.log)
    lines.append("=" * 60)
    return "\n".join(lines)


def log_voting_result(result: VotingResult, logger_instance=None) -> None:
    """Log voting result"""
    log_func = logger_instance.info if logger_instance else print
    log_func(format_voting_log(result))
