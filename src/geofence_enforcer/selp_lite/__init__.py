"""
SELP-lite: LTL-based Constrained Decoding for Safe Robot Navigation

A simplified implementation of SELP (Safe Execution of LLM-Generated Plans)
for comparison with geometric Geofence-based safety mechanisms.

Components:
- propositions: Shared evaluation functions for forbidden zones
- env_grid: Grid world navigation environment
- ltl_automaton: LTL formula to automaton conversion
- nl2ltl: Natural language to LTL with equivalence voting
- planner_constrained: Constrained decoding planner
"""

from .propositions import (
    PropositionEvaluator,
    PropositionState,
    Zone,
    create_factory_propositions,
    create_simple_grid_propositions,
)

from .env_grid import (
    GridWorld,
    GridState,
    Action,
    StepResult,
    create_factory_environment,
    create_simple_environment,
)

from .ltl_automaton import (
    LTLFormula,
    LTLAutomaton,
    AutomatonState,
    MonitorState,
    parse_ltl,
    create_automaton,
    always_avoid,
    always_maintain,
    eventually_reach,
    reach_while_avoiding,
)

from .nl2ltl import (
    NL2LTLConverter,
    VotingResult,
    StructuredConstraint,
    format_voting_log,
)

from .planner_constrained import (
    ConstrainedPlanner,
    GeofencePlanner,
    PlanResult,
    PlanStatus,
    format_plan_log,
)

__version__ = "1.0.0"
__all__ = [
    # Propositions
    "PropositionEvaluator",
    "PropositionState",
    "Zone",
    "create_factory_propositions",
    "create_simple_grid_propositions",

    # Environment
    "GridWorld",
    "GridState",
    "Action",
    "StepResult",
    "create_factory_environment",
    "create_simple_environment",

    # LTL
    "LTLFormula",
    "LTLAutomaton",
    "AutomatonState",
    "MonitorState",
    "parse_ltl",
    "create_automaton",
    "always_avoid",
    "always_maintain",
    "eventually_reach",
    "reach_while_avoiding",

    # NL2LTL
    "NL2LTLConverter",
    "VotingResult",
    "StructuredConstraint",
    "format_voting_log",

    # Planner
    "ConstrainedPlanner",
    "GeofencePlanner",
    "PlanResult",
    "PlanStatus",
    "format_plan_log",
]
