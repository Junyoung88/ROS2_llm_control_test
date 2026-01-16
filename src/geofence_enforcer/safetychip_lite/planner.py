"""
SafetyChip-lite: Action Planner

This module provides action planners that propose actions for the agent.
In the SafetyChip architecture, the planner (simulating LLM) proposes actions,
and the SafetyChip monitor validates them.

Planners:
1. HeuristicPlanner: Simple goal-directed heuristic (default)
2. RandomPlanner: Random action selection (baseline)
3. LLMPlanner: Uses actual LLM API (optional, requires API key)

The planner supports reprompting: when an action is rejected,
the planner receives feedback and proposes a different action.
"""

import random
import logging
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

# Import Action from environment for consistency
from .env_grid import Action

logger = logging.getLogger(__name__)


@dataclass
class PlannerContext:
    """Context information for the planner"""
    current_position: Tuple[float, float]
    goal_position: Tuple[float, float]
    current_propositions: Dict[str, bool]
    step_number: int
    history: List[Action] = field(default_factory=list)
    rejected_actions: List[Action] = field(default_factory=list)  # Actions rejected this step
    rejection_feedback: str = ""  # Feedback from SafetyChip


@dataclass
class PlannerProposal:
    """A proposed action from the planner"""
    action: Action
    confidence: float  # 0-1 confidence score
    reasoning: str     # Why this action was chosen


class BasePlanner(ABC):
    """Abstract base class for planners"""

    @abstractmethod
    def propose_action(self, context: PlannerContext) -> PlannerProposal:
        """Propose a single action given the current context"""
        pass

    @abstractmethod
    def propose_actions(self, context: PlannerContext, k: int = 5) -> List[PlannerProposal]:
        """Propose k candidate actions (for batch pruning)"""
        pass

    def handle_rejection(self, context: PlannerContext, feedback: str) -> None:
        """Handle rejection feedback (for reprompting)"""
        context.rejection_feedback = feedback


class HeuristicPlanner(BasePlanner):
    """
    Simple goal-directed heuristic planner.

    Simulates an LLM that proposes actions based on:
    1. Distance to goal (greedy)
    2. Avoidance of recently rejected actions
    3. Random tie-breaking
    """

    def __init__(self, randomness: float = 0.1):
        """
        Args:
            randomness: Probability of random action (exploration)
        """
        self.randomness = randomness

    def propose_action(self, context: PlannerContext) -> PlannerProposal:
        """Propose best action toward goal, avoiding rejected actions"""
        proposals = self.propose_actions(context, k=len(Action.all_actions()))

        # Filter out rejected actions
        available = [p for p in proposals if p.action not in context.rejected_actions]

        if not available:
            # All actions rejected - return WAIT as last resort
            return PlannerProposal(
                action=Action.WAIT,
                confidence=0.1,
                reasoning="All movement actions rejected, waiting"
            )

        # Random exploration with small probability
        if random.random() < self.randomness and len(available) > 1:
            return random.choice(available)

        return available[0]

    def propose_actions(self, context: PlannerContext, k: int = 5) -> List[PlannerProposal]:
        """Propose k actions ranked by goal distance"""
        proposals = []
        goal = context.goal_position
        pos = context.current_position

        for action in Action.all_actions():
            delta = action.to_delta()
            new_pos = (pos[0] + delta[0], pos[1] + delta[1])

            # Calculate distance to goal
            dx = new_pos[0] - goal[0]
            dy = new_pos[1] - goal[1]
            dist = (dx**2 + dy**2)**0.5

            # Higher confidence for actions that reduce distance
            current_dist = ((pos[0] - goal[0])**2 + (pos[1] - goal[1])**2)**0.5
            improvement = current_dist - dist

            if improvement > 0:
                confidence = min(1.0, 0.5 + improvement * 0.2)
                reasoning = f"Moves toward goal (distance: {dist:.2f})"
            elif improvement == 0:
                confidence = 0.3
                reasoning = f"Maintains distance to goal (distance: {dist:.2f})"
            else:
                confidence = max(0.1, 0.3 - abs(improvement) * 0.1)
                reasoning = f"Moves away from goal (distance: {dist:.2f})"

            proposals.append(PlannerProposal(
                action=action,
                confidence=confidence,
                reasoning=reasoning
            ))

        # Sort by confidence (descending)
        proposals.sort(key=lambda p: p.confidence, reverse=True)
        return proposals[:k]


class RandomPlanner(BasePlanner):
    """Random action planner (baseline)"""

    def propose_action(self, context: PlannerContext) -> PlannerProposal:
        """Propose random action"""
        available = [a for a in Action.all_actions() if a not in context.rejected_actions]
        if not available:
            available = [Action.WAIT]

        action = random.choice(available)
        return PlannerProposal(
            action=action,
            confidence=1.0 / len(Action.all_actions()),
            reasoning="Random selection"
        )

    def propose_actions(self, context: PlannerContext, k: int = 5) -> List[PlannerProposal]:
        """Propose k random actions"""
        actions = list(Action.all_actions())
        random.shuffle(actions)
        return [
            PlannerProposal(
                action=a,
                confidence=1.0 / len(actions),
                reasoning="Random selection"
            )
            for a in actions[:k]
        ]


class LLMPlanner(BasePlanner):
    """
    LLM-based planner (optional, requires API key).

    Uses actual LLM to propose actions based on:
    - Current state description
    - Goal description
    - Safety constraints
    - Rejection feedback (reprompting)
    """

    def __init__(self, model: str = "gpt-3.5-turbo", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key
        self._fallback = HeuristicPlanner()

    def propose_action(self, context: PlannerContext) -> PlannerProposal:
        """Propose action using LLM"""
        # Placeholder - falls back to heuristic
        logger.warning("LLM planner not fully implemented, using heuristic fallback")
        return self._fallback.propose_action(context)

    def propose_actions(self, context: PlannerContext, k: int = 5) -> List[PlannerProposal]:
        """Propose k actions using LLM"""
        return self._fallback.propose_actions(context, k)

    def _build_prompt(self, context: PlannerContext) -> str:
        """Build prompt for LLM"""
        prompt = f"""You are a robot navigation planner. Given the current state, propose the best action.

Current position: {context.current_position}
Goal position: {context.goal_position}
Current step: {context.step_number}

Available actions: MOVE_NORTH, MOVE_SOUTH, MOVE_EAST, MOVE_WEST, WAIT

"""
        if context.rejection_feedback:
            prompt += f"""
IMPORTANT: Your previous action was REJECTED for safety reasons:
{context.rejection_feedback}

Please propose a DIFFERENT action that does not violate the safety constraints.
"""

        if context.rejected_actions:
            rejected_names = [a.name for a in context.rejected_actions]
            prompt += f"\nActions already rejected this step: {rejected_names}\n"

        prompt += "\nPropose the best action and explain your reasoning."
        return prompt


class RepromptingPlanner:
    """
    Wrapper that handles the reprompting logic.

    When an action is rejected:
    1. Records the rejected action
    2. Updates context with rejection feedback
    3. Requests new proposal from base planner
    """

    def __init__(self, base_planner: BasePlanner, max_reprompts: int = 5):
        self.base_planner = base_planner
        self.max_reprompts = max_reprompts
        self._reprompt_count = 0

    def reset(self) -> None:
        """Reset reprompt counter"""
        self._reprompt_count = 0

    def propose_action(self, context: PlannerContext) -> Tuple[PlannerProposal, int]:
        """
        Propose action, handling rejections internally.

        Returns:
            Tuple of (proposal, reprompt_count)
        """
        context.rejected_actions = []
        proposal = self.base_planner.propose_action(context)
        return proposal, self._reprompt_count

    def handle_rejection(self, context: PlannerContext, rejected_action: Action,
                        feedback: str) -> Optional[PlannerProposal]:
        """
        Handle rejection and get new proposal.

        Returns:
            New proposal, or None if max reprompts exceeded
        """
        self._reprompt_count += 1

        if self._reprompt_count > self.max_reprompts:
            logger.warning(f"Max reprompts ({self.max_reprompts}) exceeded")
            return None

        context.rejected_actions.append(rejected_action)
        context.rejection_feedback = feedback

        logger.info(f"Reprompting ({self._reprompt_count}/{self.max_reprompts})")
        logger.info(f"  Feedback: {feedback[:100]}...")

        return self.base_planner.propose_action(context)

    @property
    def reprompt_count(self) -> int:
        return self._reprompt_count


def create_planner(planner_type: str = "heuristic", **kwargs) -> BasePlanner:
    """Factory function to create planner"""
    if planner_type == "heuristic":
        return HeuristicPlanner(**kwargs)
    elif planner_type == "random":
        return RandomPlanner()
    elif planner_type == "llm":
        return LLMPlanner(**kwargs)
    else:
        raise ValueError(f"Unknown planner type: {planner_type}")
