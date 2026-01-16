"""
SafetyChip-lite: LTL-based Safety Enforcement for Robot Navigation

A simplified implementation of SafetyChip for comparison with
geometric Geofence-based safety mechanisms.

Key Components:
1. NL→LTL Translator: Converts natural language constraints to LTL
2. Constraint Monitor: Progression-based LTL monitoring
3. Action Pruning: Blocks unsafe actions BEFORE execution
4. Reprompting: Explains violations and requests alternatives

SafetyChip Architecture:
- LLM/Planner proposes actions
- SafetyChip (external module) validates actions
- Unsafe actions are pruned with explanations
- Planner is reprompted for alternatives

Key Guarantee: 100% safety (no unsafe actions executed)
"""

from .nl2ltl import (
    NL2LTLTranslator,
    LTLConstraint,
    ConstraintType,
    TranslationResult,
    format_translation_log,
)

from .ltl_monitor import (
    ConstraintMonitor,
    PatternMonitor,
    CompositeMonitor,
    MonitorStatus,
    MonitorState,
    ViolationInfo,
    ExplanationResult,
    create_monitor,
    explain_violation,
    format_monitor_log,
)

from .planner import (
    Action,
    BasePlanner,
    HeuristicPlanner,
    RandomPlanner,
    LLMPlanner,
    RepromptingPlanner,
    PlannerContext,
    PlannerProposal,
    create_planner,
)

from .agent_loop import (
    SafetyChipAgent,
    EpisodeResult,
    LoopStatus,
    StepLog,
    run_safetychip_agent,
    format_episode_log,
)

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
    StepResult,
    create_factory_environment,
    create_simple_environment,
)

__version__ = "1.0.0"
__all__ = [
    # NL2LTL
    "NL2LTLTranslator",
    "LTLConstraint",
    "ConstraintType",
    "TranslationResult",
    "format_translation_log",

    # Monitor
    "ConstraintMonitor",
    "PatternMonitor",
    "CompositeMonitor",
    "MonitorStatus",
    "MonitorState",
    "ViolationInfo",
    "ExplanationResult",
    "create_monitor",
    "explain_violation",
    "format_monitor_log",

    # Planner
    "Action",
    "BasePlanner",
    "HeuristicPlanner",
    "RandomPlanner",
    "LLMPlanner",
    "RepromptingPlanner",
    "PlannerContext",
    "PlannerProposal",
    "create_planner",

    # Agent Loop
    "SafetyChipAgent",
    "EpisodeResult",
    "LoopStatus",
    "StepLog",
    "run_safetychip_agent",
    "format_episode_log",

    # Propositions
    "PropositionEvaluator",
    "PropositionState",
    "Zone",
    "create_factory_propositions",
    "create_simple_grid_propositions",

    # Environment
    "GridWorld",
    "GridState",
    "StepResult",
    "create_factory_environment",
    "create_simple_environment",
]
