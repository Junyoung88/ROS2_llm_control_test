"""
SafetyChip-lite: Agent Execution Loop

This is the main SafetyChip execution loop that integrates:
1. NL→LTL translation
2. Constraint monitoring
3. Action pruning
4. Violation explanation
5. Reprompting

Key SafetyChip principle:
- Unsafe actions are BLOCKED BEFORE execution (not detected after)
- 100% safety is the goal

The loop structure:
1. Translate NL constraints to LTL
2. Create constraint monitor
3. For each step:
   a. Planner proposes action
   b. Monitor checks if action would violate constraints
   c. If safe: execute
   d. If unsafe: prune, explain, reprompt
4. Continue until goal reached or max steps exceeded
"""

import logging
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum, auto
import copy

from .nl2ltl import NL2LTLTranslator, TranslationResult, format_translation_log
from .ltl_monitor import (
    ConstraintMonitor, create_monitor, MonitorStatus, ViolationInfo,
    explain_violation, ExplanationResult, format_monitor_log
)
from .planner import (
    BasePlanner, HeuristicPlanner, RepromptingPlanner, PlannerContext,
    PlannerProposal, create_planner
)
from .env_grid import Action  # Planner now uses env_grid.Action directly

logger = logging.getLogger(__name__)


class LoopStatus(Enum):
    """Status of the agent loop"""
    SUCCESS = auto()           # Goal reached safely
    MAX_STEPS_EXCEEDED = auto()  # Ran out of steps
    ALL_ACTIONS_UNSAFE = auto()  # Deadlock - all actions violate constraints
    SAFETY_VIOLATION = auto()    # Should not happen - violation occurred
    ERROR = auto()              # Unexpected error


@dataclass
class StepLog:
    """Log entry for one execution step"""
    step_number: int
    position_before: Tuple[float, float]
    position_after: Tuple[float, float]
    proposed_action: Action
    action_executed: Optional[Action]
    was_pruned: bool
    pruning_reason: str
    reprompt_count: int
    propositions_before: Dict[str, bool]
    propositions_after: Dict[str, bool]
    violation_info: Optional[ExplanationResult] = None


@dataclass
class EpisodeResult:
    """Result of a complete episode"""
    status: LoopStatus
    success: bool                    # Goal reached
    safety_violation: bool           # Any violation occurred (should be False)
    steps: int                       # Total steps taken
    path: List[Tuple[float, float]]  # Trajectory
    actions: List[Action]            # Executed actions
    pruned_count: int               # Number of pruned actions
    reprompt_count: int             # Number of reprompts
    min_distance_to_forbidden: float  # Minimum distance to forbidden zone
    step_logs: List[StepLog] = field(default_factory=list)
    translation_log: str = ""
    termination_reason: str = ""


class SafetyChipAgent:
    """
    SafetyChip-lite Agent: Safe execution with LTL constraint monitoring.

    This agent implements the SafetyChip architecture:
    1. LTL constraints define safety requirements
    2. Monitor checks actions BEFORE execution
    3. Unsafe actions are pruned with explanations
    4. Planner is reprompted to find safe alternatives

    Key guarantee: No unsafe actions are ever executed.
    """

    def __init__(
        self,
        env,  # GridWorld environment
        planner: Optional[BasePlanner] = None,
        max_reprompts: int = 5,
        verbose: bool = True
    ):
        """
        Args:
            env: Grid world environment with propositions
            planner: Action planner (default: HeuristicPlanner)
            max_reprompts: Maximum reprompts per step
            verbose: Enable verbose logging
        """
        self.env = env
        self.base_planner = planner or HeuristicPlanner()
        self.planner = RepromptingPlanner(self.base_planner, max_reprompts)
        self.verbose = verbose

        # Components initialized during run
        self.translator: Optional[NL2LTLTranslator] = None
        self.monitor: Optional[ConstraintMonitor] = None
        self.translation_result: Optional[TranslationResult] = None

    def run(
        self,
        constraints: List[str],
        max_steps: int = 100
    ) -> EpisodeResult:
        """
        Run the SafetyChip agent with given constraints.

        Args:
            constraints: List of natural language constraints
            max_steps: Maximum steps before termination

        Returns:
            EpisodeResult with trajectory and metrics
        """
        # Step 1: Translate NL constraints to LTL
        logger.info("=" * 70)
        logger.info("SafetyChip-lite Agent Starting")
        logger.info("=" * 70)

        self.translator = NL2LTLTranslator()
        self.translation_result = self.translator.translate(constraints)

        if self.verbose:
            logger.info(format_translation_log(self.translation_result))

        if not self.translation_result.constraints:
            logger.error("No constraints could be translated!")
            return EpisodeResult(
                status=LoopStatus.ERROR,
                success=False,
                safety_violation=False,
                steps=0,
                path=[],
                actions=[],
                pruned_count=0,
                reprompt_count=0,
                min_distance_to_forbidden=float('inf'),
                termination_reason="Failed to translate constraints"
            )

        # Step 2: Create constraint monitor
        self.monitor = create_monitor(self.translation_result.constraints)

        # Step 3: Initialize episode
        state = self.env.reset()
        path = [state.position]
        actions = []
        step_logs = []
        total_pruned = 0
        total_reprompts = 0
        min_dist = float('inf')

        logger.info(f"\nStarting execution: {state.position} -> {self.env.goal_position}")
        logger.info(f"Constraints: {[c.original_nl for c in self.translation_result.constraints]}")
        logger.info("-" * 70)

        # Step 4: Main execution loop
        for step in range(max_steps):
            # Check if goal reached
            if self.env.props.is_at_goal(state.position):
                logger.info(f"\nStep {step}: GOAL REACHED!")
                return EpisodeResult(
                    status=LoopStatus.SUCCESS,
                    success=True,
                    safety_violation=False,
                    steps=step,
                    path=path,
                    actions=actions,
                    pruned_count=total_pruned,
                    reprompt_count=total_reprompts,
                    min_distance_to_forbidden=min_dist,
                    step_logs=step_logs,
                    translation_log=format_translation_log(self.translation_result),
                    termination_reason="Goal reached"
                )

            # Get current propositions
            current_props = self.env.get_propositions(state)

            # Update min distance
            dist = self.env.props.get_distance_to_nearest_forbidden(state.position)
            min_dist = min(min_dist, dist)

            # Create planner context
            context = PlannerContext(
                current_position=state.position,
                goal_position=self.env.goal_position,
                current_propositions=current_props,
                step_number=step,
                history=[a for a in actions]
            )

            # Get action through pruning + reprompt loop
            executed_action, step_log = self._get_safe_action(
                context, current_props, step
            )

            if executed_action is None:
                # All actions unsafe - deadlock
                logger.warning(f"\nStep {step}: DEADLOCK - All actions unsafe!")
                step_logs.append(step_log)
                return EpisodeResult(
                    status=LoopStatus.ALL_ACTIONS_UNSAFE,
                    success=False,
                    safety_violation=False,
                    steps=step,
                    path=path,
                    actions=actions,
                    pruned_count=total_pruned,
                    reprompt_count=total_reprompts,
                    min_distance_to_forbidden=min_dist,
                    step_logs=step_logs,
                    translation_log=format_translation_log(self.translation_result),
                    termination_reason="All actions violate constraints (deadlock)"
                )

            # Execute safe action
            result = self.env.step(executed_action)
            state = result.next_state
            path.append(state.position)
            actions.append(executed_action)

            # Update monitor state
            new_props = self.env.get_propositions(state)
            self.monitor.step(new_props)

            # Update step log
            step_log.position_after = state.position
            step_log.propositions_after = new_props
            step_log.action_executed = executed_action
            step_logs.append(step_log)

            # Update counters
            total_pruned += step_log.pruning_reason != ""
            total_reprompts += step_log.reprompt_count

            if self.verbose:
                self._log_step(step_log)

        # Max steps exceeded
        logger.warning(f"\nMax steps ({max_steps}) exceeded")
        return EpisodeResult(
            status=LoopStatus.MAX_STEPS_EXCEEDED,
            success=False,
            safety_violation=False,
            steps=max_steps,
            path=path,
            actions=actions,
            pruned_count=total_pruned,
            reprompt_count=total_reprompts,
            min_distance_to_forbidden=min_dist,
            step_logs=step_logs,
            translation_log=format_translation_log(self.translation_result),
            termination_reason="Maximum steps exceeded"
        )

    def _get_safe_action(
        self,
        context: PlannerContext,
        current_props: Dict[str, bool],
        step: int
    ) -> Tuple[Optional[Action], StepLog]:
        """
        Get a safe action through pruning and reprompting.

        Returns:
            Tuple of (safe_action or None, step_log)
        """
        self.planner.reset()
        pruned_actions = []
        all_violations = []

        step_log = StepLog(
            step_number=step,
            position_before=context.current_position,
            position_after=context.current_position,  # Updated later
            proposed_action=Action.WAIT,  # Updated below
            action_executed=None,
            was_pruned=False,
            pruning_reason="",
            reprompt_count=0,
            propositions_before=current_props.copy(),
            propositions_after={}
        )

        while True:
            # Get proposal from planner
            proposal, _ = self.planner.propose_action(context)
            step_log.proposed_action = proposal.action

            if self.verbose:
                logger.info(f"\nStep {step}: Planner proposes {proposal.action.name}")
                logger.info(f"  Reasoning: {proposal.reasoning}")

            # Simulate action to get resulting propositions
            sim_state = self.env.simulate_action(
                self.env.state, proposal.action
            )
            sim_props = self.env.get_propositions(sim_state)

            # Check with monitor
            would_violate, violations = self.monitor.would_violate(sim_props)

            if not would_violate:
                # Action is safe!
                if self.verbose:
                    logger.info(f"  -> SAFE: Action approved")
                return proposal.action, step_log

            # Action would violate - PRUNE
            pruned_actions.append(proposal.action)
            all_violations.extend(violations)

            # Generate explanation
            explanation = explain_violation(violations, current_props, sim_props)

            if self.verbose:
                logger.info(f"  -> UNSAFE: Action PRUNED")
                logger.info(f"  Reason: {explanation.summary}")
                for detail in explanation.details:
                    logger.info(f"    {detail}")

            step_log.was_pruned = True
            step_log.pruning_reason = explanation.summary
            step_log.violation_info = explanation

            # Check if all actions have been tried
            if len(pruned_actions) >= len(Action.all_actions()):
                logger.warning("All actions have been pruned!")
                return None, step_log

            # Reprompt planner
            new_proposal = self.planner.handle_rejection(
                context, proposal.action, explanation.reprompt_text
            )

            if new_proposal is None:
                # Max reprompts exceeded
                logger.warning("Max reprompts exceeded")
                return None, step_log

            step_log.reprompt_count = self.planner.reprompt_count

    def _log_step(self, step_log: StepLog) -> None:
        """Log step details"""
        logger.info("-" * 50)
        logger.info(f"Step {step_log.step_number}:")
        logger.info(f"  Position: {step_log.position_before} -> {step_log.position_after}")
        logger.info(f"  Proposed: {step_log.proposed_action.name}")
        if step_log.action_executed:
            logger.info(f"  Executed: {step_log.action_executed.name}")
        if step_log.was_pruned:
            logger.info(f"  Pruned: Yes ({step_log.pruning_reason})")
            logger.info(f"  Reprompts: {step_log.reprompt_count}")


def run_safetychip_agent(
    env,
    constraints: List[str],
    max_steps: int = 100,
    planner_type: str = "heuristic",
    verbose: bool = False
) -> EpisodeResult:
    """
    Convenience function to run SafetyChip agent.

    Args:
        env: Grid world environment
        constraints: Natural language constraints
        max_steps: Maximum steps
        planner_type: "heuristic", "random", or "llm"
        verbose: Enable verbose logging

    Returns:
        EpisodeResult
    """
    planner = create_planner(planner_type)
    agent = SafetyChipAgent(env, planner, verbose=verbose)
    return agent.run(constraints, max_steps)


# =============================================================================
# Logging Utilities
# =============================================================================

def format_episode_log(result: EpisodeResult) -> str:
    """Format complete episode log for paper documentation"""
    lines = ["=" * 70]
    lines.append("SafetyChip-lite Episode Log")
    lines.append("=" * 70)

    lines.append(f"\nResult: {result.status.name}")
    lines.append(f"Success: {result.success}")
    lines.append(f"Safety Violation: {result.safety_violation}")
    lines.append(f"Steps: {result.steps}")
    lines.append(f"Pruned Actions: {result.pruned_count}")
    lines.append(f"Reprompts: {result.reprompt_count}")
    lines.append(f"Min Distance to Forbidden: {result.min_distance_to_forbidden:.2f}")

    lines.append("\n" + "-" * 70)
    lines.append("Translation Log:")
    lines.append(result.translation_log)

    lines.append("\n" + "-" * 70)
    lines.append("Step-by-Step Log:")

    for log in result.step_logs:
        lines.append(f"\n--- Step {log.step_number} ---")
        lines.append(f"Position: {log.position_before} -> {log.position_after}")
        lines.append(f"Proposed: {log.proposed_action.name}")
        if log.action_executed:
            lines.append(f"Executed: {log.action_executed.name}")
        if log.was_pruned:
            lines.append(f"PRUNED: {log.pruning_reason}")
            lines.append(f"Reprompts: {log.reprompt_count}")
            if log.violation_info:
                lines.append("Violation details:")
                for detail in log.violation_info.details:
                    lines.append(f"  {detail}")
                lines.append("Reprompt text:")
                for line in log.violation_info.reprompt_text.split('\n')[:5]:
                    lines.append(f"  {line}")

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)
