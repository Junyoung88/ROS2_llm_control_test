"""
Constrained Decoding Planner for SELP-lite

This module implements the Constrained Decoding component of SELP-lite:
1. Generate plan one action at a time
2. For each candidate action, simulate and check LTL constraint
3. Mask actions that would violate the constraint
4. Select from remaining safe actions

The key difference from Geofence:
- Geofence: Projects/blocks goals based on geometric computation
- SELP-lite: Masks actions based on LTL automaton progress

Logging Requirements (for paper):
- Each step: which actions were masked and why
- Which proposition changes caused LTL violation
"""

import logging
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum, auto
import copy

from .env_grid import GridWorld, GridState, Action, StepResult
from .ltl_automaton import LTLAutomaton, AutomatonState, MonitorState, create_automaton
from .propositions import PropositionEvaluator

logger = logging.getLogger(__name__)


class PlanStatus(Enum):
    """Status of planning result"""
    SUCCESS = auto()          # Goal reached
    CONSTRAINT_DEADLOCK = auto()  # All actions violate constraint
    MAX_STEPS_EXCEEDED = auto()   # Ran out of steps
    PLANNING_FAILED = auto()      # Other failure


@dataclass
class ActionMaskInfo:
    """Information about why an action was masked"""
    action: Action
    masked: bool
    reason: str
    proposition_changes: Dict[str, Tuple[bool, bool]] = field(default_factory=dict)  # prop -> (before, after)
    automaton_status_before: AutomatonState = AutomatonState.UNKNOWN
    automaton_status_after: AutomatonState = AutomatonState.UNKNOWN


@dataclass
class StepLog:
    """Log entry for one planning step"""
    step_number: int
    current_position: Tuple[float, float]
    current_propositions: Dict[str, bool]
    automaton_state: MonitorState
    action_evaluations: List[ActionMaskInfo]
    safe_actions: List[Action]
    selected_action: Optional[Action]
    selection_reason: str


@dataclass
class PlanResult:
    """Result of constrained planning"""
    status: PlanStatus
    trajectory: List[Tuple[float, float]]
    actions: List[Action]
    total_steps: int
    masked_action_count: int
    safety_violations: int
    reached_goal: bool
    step_logs: List[StepLog] = field(default_factory=list)
    termination_reason: str = ""


class ConstrainedPlanner:
    """
    SELP-lite Constrained Decoding Planner.

    This planner generates safe plans by:
    1. Evaluating each candidate action
    2. Simulating the action's effect on propositions
    3. Checking if the LTL automaton would be violated
    4. Masking unsafe actions
    5. Selecting from remaining safe actions

    The selection heuristic prioritizes:
    1. Actions that move toward the goal
    2. Random tie-breaking among equally good actions
    """

    def __init__(
        self,
        env: GridWorld,
        automaton: LTLAutomaton,
        verbose_logging: bool = True
    ):
        """
        Args:
            env: Grid world environment
            automaton: LTL automaton for constraint checking
            verbose_logging: Whether to log detailed step information
        """
        self.env = env
        self.automaton = automaton
        self.verbose_logging = verbose_logging

        # Planning state
        self._monitor_state: Optional[MonitorState] = None
        self._step_logs: List[StepLog] = []
        self._masked_count: int = 0

    def plan(
        self,
        start_state: Optional[GridState] = None,
        max_steps: Optional[int] = None
    ) -> PlanResult:
        """
        Generate a plan from start to goal using constrained decoding.

        Args:
            start_state: Starting state (uses env's start if None)
            max_steps: Maximum planning steps

        Returns:
            PlanResult with trajectory, actions, and detailed logs
        """
        # Initialize
        max_steps = max_steps or self.env.max_steps
        state = self.env.reset() if start_state is None else start_state
        self.automaton.reset()
        self._monitor_state = self.automaton.step(self.env.get_propositions(state))
        self._step_logs = []
        self._masked_count = 0

        trajectory = [state.position]
        actions = []
        safety_violations = 0

        logger.info("=" * 60)
        logger.info("SELP-lite Constrained Decoding Started")
        logger.info(f"Start: {state.position}, Goal: {self.env.goal_position}")
        logger.info("=" * 60)

        for step in range(max_steps):
            # Check if already at goal
            if self.env.props.is_at_goal(state.position):
                logger.info(f"Step {step}: Goal reached!")
                return PlanResult(
                    status=PlanStatus.SUCCESS,
                    trajectory=trajectory,
                    actions=actions,
                    total_steps=step,
                    masked_action_count=self._masked_count,
                    safety_violations=safety_violations,
                    reached_goal=True,
                    step_logs=self._step_logs,
                    termination_reason="Goal reached"
                )

            # Check automaton status
            if self._monitor_state.status == AutomatonState.VIOLATED:
                logger.warning(f"Step {step}: LTL constraint violated!")
                return PlanResult(
                    status=PlanStatus.PLANNING_FAILED,
                    trajectory=trajectory,
                    actions=actions,
                    total_steps=step,
                    masked_action_count=self._masked_count,
                    safety_violations=safety_violations + 1,
                    reached_goal=False,
                    step_logs=self._step_logs,
                    termination_reason="LTL constraint violated"
                )

            # Evaluate all actions and create mask
            current_props = self.env.get_propositions(state)
            action_evals = self._evaluate_actions(state, current_props)

            # Get safe actions
            safe_actions = [ae.action for ae in action_evals if not ae.masked]

            # Log step
            step_log = StepLog(
                step_number=step,
                current_position=state.position,
                current_propositions=current_props,
                automaton_state=self._monitor_state,
                action_evaluations=action_evals,
                safe_actions=safe_actions,
                selected_action=None,
                selection_reason=""
            )

            if self.verbose_logging:
                self._log_step_evaluation(step_log)

            # Check for deadlock
            if not safe_actions:
                logger.warning(f"Step {step}: DEADLOCK - All actions masked!")
                step_log.selection_reason = "DEADLOCK: No safe actions"
                self._step_logs.append(step_log)

                return PlanResult(
                    status=PlanStatus.CONSTRAINT_DEADLOCK,
                    trajectory=trajectory,
                    actions=actions,
                    total_steps=step,
                    masked_action_count=self._masked_count,
                    safety_violations=safety_violations,
                    reached_goal=False,
                    step_logs=self._step_logs,
                    termination_reason="All actions violate constraint"
                )

            # Select best action from safe actions
            selected_action, reason = self._select_action(state, safe_actions)
            step_log.selected_action = selected_action
            step_log.selection_reason = reason
            self._step_logs.append(step_log)

            # Execute action
            result = self.env.step(selected_action)
            state = result.next_state
            actions.append(selected_action)
            trajectory.append(state.position)

            # Update automaton
            new_props = self.env.get_propositions(state)
            self._monitor_state = self.automaton.step(new_props)

            # Track safety
            if result.entered_forbidden or result.violated_margin:
                safety_violations += 1

            if self.verbose_logging:
                logger.info(f"  -> Executed {selected_action.name}, "
                           f"new pos: {state.position}")

        # Max steps exceeded
        return PlanResult(
            status=PlanStatus.MAX_STEPS_EXCEEDED,
            trajectory=trajectory,
            actions=actions,
            total_steps=max_steps,
            masked_action_count=self._masked_count,
            safety_violations=safety_violations,
            reached_goal=False,
            step_logs=self._step_logs,
            termination_reason="Maximum steps exceeded"
        )

    def _evaluate_actions(
        self,
        state: GridState,
        current_props: Dict[str, bool]
    ) -> List[ActionMaskInfo]:
        """
        Evaluate all candidate actions for safety.

        For each action:
        1. Simulate the action
        2. Get resulting propositions
        3. Check if LTL automaton would be violated
        4. Mark as masked if violated
        """
        evaluations = []
        valid_actions = self.env.get_valid_actions(state)

        for action in Action.all_actions():
            info = ActionMaskInfo(
                action=action,
                masked=False,
                reason="",
                automaton_status_before=self._monitor_state.status
            )

            # Check if action is valid (in bounds)
            if action not in valid_actions:
                info.masked = True
                info.reason = "Out of bounds"
                evaluations.append(info)
                continue

            # Simulate action
            simulated_state = self.env.simulate_action(state, action)
            simulated_props = self.env.get_propositions(simulated_state)

            # Track proposition changes
            for prop in set(current_props.keys()) | set(simulated_props.keys()):
                before = current_props.get(prop, False)
                after = simulated_props.get(prop, False)
                if before != after:
                    info.proposition_changes[prop] = (before, after)

            # Check if automaton would be violated
            would_violate = self.automaton.would_violate(simulated_props)

            if would_violate:
                info.masked = True
                info.reason = self._explain_violation(current_props, simulated_props)
                info.automaton_status_after = AutomatonState.VIOLATED
                self._masked_count += 1
            else:
                info.reason = "Safe"
                info.automaton_status_after = AutomatonState.NON_ACCEPTING

            evaluations.append(info)

        return evaluations

    def _explain_violation(
        self,
        before_props: Dict[str, bool],
        after_props: Dict[str, bool]
    ) -> str:
        """Generate explanation for why an action violates the constraint"""
        changes = []
        for prop in set(before_props.keys()) | set(after_props.keys()):
            b = before_props.get(prop, False)
            a = after_props.get(prop, False)
            if b != a:
                if a:
                    changes.append(f"{prop}: False->True")
                else:
                    changes.append(f"{prop}: True->False")

        if changes:
            return f"LTL violation due to: {', '.join(changes)}"
        else:
            return "LTL violation (no proposition change detected)"

    def _select_action(
        self,
        state: GridState,
        safe_actions: List[Action]
    ) -> Tuple[Action, str]:
        """
        Select best action from safe actions using heuristic.

        Heuristic: Minimize distance to goal
        Tie-breaking: Random selection
        """
        import random

        if len(safe_actions) == 1:
            return safe_actions[0], "Only safe action"

        # Compute distances to goal for each action
        goal = self.env.goal_position
        action_distances = []

        for action in safe_actions:
            simulated = self.env.simulate_action(state, action)
            dx = simulated.position[0] - goal[0]
            dy = simulated.position[1] - goal[1]
            dist = (dx**2 + dy**2)**0.5
            action_distances.append((action, dist))

        # Sort by distance (ascending)
        action_distances.sort(key=lambda x: x[1])

        # Get all actions with minimum distance (for tie-breaking)
        min_dist = action_distances[0][1]
        best_actions = [a for a, d in action_distances if abs(d - min_dist) < 0.01]

        # Random tie-break
        selected = random.choice(best_actions)

        reason = f"Heuristic: distance to goal = {min_dist:.2f}"
        if len(best_actions) > 1:
            reason += f" (tie-break among {len(best_actions)} actions)"

        return selected, reason

    def _log_step_evaluation(self, step_log: StepLog) -> None:
        """Log detailed step evaluation for paper"""
        logger.info("-" * 50)
        logger.info(f"Step {step_log.step_number}: Position {step_log.current_position}")
        logger.info(f"  Automaton status: {step_log.automaton_state.status.name}")

        # Log action evaluations
        logger.info("  Action evaluations:")
        for ae in step_log.action_evaluations:
            status = "MASKED" if ae.masked else "SAFE"
            logger.info(f"    {ae.action.name}: {status}")
            if ae.masked:
                logger.info(f"      Reason: {ae.reason}")
                if ae.proposition_changes:
                    for prop, (before, after) in ae.proposition_changes.items():
                        logger.info(f"      {prop}: {before} -> {after}")

        logger.info(f"  Safe actions: {[a.name for a in step_log.safe_actions]}")


# =============================================================================
# Geofence Baseline Planner (for comparison)
# =============================================================================

class GeofencePlanner:
    """
    Geofence-based planner for comparison.

    This planner uses geometric projection/blocking instead of LTL-based masking.
    It provides a fair comparison baseline for SELP-lite.
    """

    def __init__(
        self,
        env: GridWorld,
        safety_margin: float = 0.3,
        use_projection: bool = True
    ):
        """
        Args:
            env: Grid world environment
            safety_margin: Safety margin distance
            use_projection: If True, project unsafe goals; if False, block
        """
        self.env = env
        self.safety_margin = safety_margin
        self.use_projection = use_projection
        self._masked_count = 0

    def plan(
        self,
        start_state: Optional[GridState] = None,
        max_steps: Optional[int] = None
    ) -> PlanResult:
        """Plan using geometric geofence checking"""
        max_steps = max_steps or self.env.max_steps
        state = self.env.reset() if start_state is None else start_state
        self._masked_count = 0

        trajectory = [state.position]
        actions = []
        safety_violations = 0
        step_logs = []

        for step in range(max_steps):
            # Check goal
            if self.env.props.is_at_goal(state.position):
                return PlanResult(
                    status=PlanStatus.SUCCESS,
                    trajectory=trajectory,
                    actions=actions,
                    total_steps=step,
                    masked_action_count=self._masked_count,
                    safety_violations=safety_violations,
                    reached_goal=True,
                    termination_reason="Goal reached"
                )

            # Evaluate actions with geofence
            safe_actions = []
            for action in self.env.get_valid_actions(state):
                simulated = self.env.simulate_action(state, action)

                # Check if within margin of forbidden zone
                if not self.env.props.is_within_margin(simulated.position, self.safety_margin):
                    safe_actions.append(action)
                else:
                    self._masked_count += 1

            if not safe_actions:
                return PlanResult(
                    status=PlanStatus.CONSTRAINT_DEADLOCK,
                    trajectory=trajectory,
                    actions=actions,
                    total_steps=step,
                    masked_action_count=self._masked_count,
                    safety_violations=safety_violations,
                    reached_goal=False,
                    termination_reason="All actions blocked by geofence"
                )

            # Select action (same heuristic as SELP-lite for fairness)
            selected = self._select_action(state, safe_actions)

            # Execute
            result = self.env.step(selected)
            state = result.next_state
            actions.append(selected)
            trajectory.append(state.position)

            if result.entered_forbidden or result.violated_margin:
                safety_violations += 1

        return PlanResult(
            status=PlanStatus.MAX_STEPS_EXCEEDED,
            trajectory=trajectory,
            actions=actions,
            total_steps=max_steps,
            masked_action_count=self._masked_count,
            safety_violations=safety_violations,
            reached_goal=False,
            termination_reason="Maximum steps exceeded"
        )

    def _select_action(self, state: GridState, safe_actions: List[Action]) -> Action:
        """Select action using same heuristic as SELP-lite"""
        import random

        if len(safe_actions) == 1:
            return safe_actions[0]

        goal = self.env.goal_position
        action_distances = []

        for action in safe_actions:
            simulated = self.env.simulate_action(state, action)
            dx = simulated.position[0] - goal[0]
            dy = simulated.position[1] - goal[1]
            dist = (dx**2 + dy**2)**0.5
            action_distances.append((action, dist))

        action_distances.sort(key=lambda x: x[1])
        min_dist = action_distances[0][1]
        best = [a for a, d in action_distances if abs(d - min_dist) < 0.01]

        return random.choice(best)


# =============================================================================
# Logging Utilities
# =============================================================================

def format_plan_log(result: PlanResult) -> str:
    """Format plan result for detailed logging"""
    lines = ["=" * 70]
    lines.append("SELP-lite Constrained Decoding Plan Log")
    lines.append("=" * 70)
    lines.append(f"Status: {result.status.name}")
    lines.append(f"Steps: {result.total_steps}")
    lines.append(f"Masked actions: {result.masked_action_count}")
    lines.append(f"Safety violations: {result.safety_violations}")
    lines.append(f"Reached goal: {result.reached_goal}")
    lines.append(f"Termination: {result.termination_reason}")
    lines.append("")

    for log in result.step_logs:
        lines.append(f"--- Step {log.step_number} ---")
        lines.append(f"Position: {log.current_position}")
        lines.append(f"Safe actions: {[a.name for a in log.safe_actions]}")

        masked = [ae for ae in log.action_evaluations if ae.masked]
        if masked:
            lines.append("Masked actions:")
            for ae in masked:
                lines.append(f"  {ae.action.name}: {ae.reason}")
                for prop, (before, after) in ae.proposition_changes.items():
                    lines.append(f"    {prop}: {before} -> {after}")

        lines.append(f"Selected: {log.selected_action.name if log.selected_action else 'None'}")
        lines.append(f"Reason: {log.selection_reason}")
        lines.append("")

    lines.append("=" * 70)
    return "\n".join(lines)
