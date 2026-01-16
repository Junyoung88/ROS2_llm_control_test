"""
Grid World Environment for SELP-lite experiments

A discrete grid world environment for robot navigation with:
- Discrete actions: MOVE_NORTH, MOVE_SOUTH, MOVE_EAST, MOVE_WEST, WAIT
- State: robot position + visited zones (boolean propositions)
- Forbidden zones with safety margins
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass, field
from enum import Enum, auto
import logging
import copy

from .propositions import PropositionEvaluator, PropositionState, create_factory_propositions

logger = logging.getLogger(__name__)


class Action(Enum):
    """Discrete actions in grid world"""
    MOVE_NORTH = auto()  # +Y
    MOVE_SOUTH = auto()  # -Y
    MOVE_EAST = auto()   # +X
    MOVE_WEST = auto()   # -X
    WAIT = auto()        # Stay in place

    @classmethod
    def all_actions(cls) -> List['Action']:
        return list(cls)

    def to_delta(self, cell_size: float = 1.0) -> Tuple[float, float]:
        """Convert action to position delta"""
        deltas = {
            Action.MOVE_NORTH: (0, cell_size),
            Action.MOVE_SOUTH: (0, -cell_size),
            Action.MOVE_EAST: (cell_size, 0),
            Action.MOVE_WEST: (-cell_size, 0),
            Action.WAIT: (0, 0),
        }
        return deltas[self]

    def __str__(self):
        return self.name


@dataclass
class GridState:
    """State in the grid world"""
    position: Tuple[float, float]
    visited_zones: Set[str] = field(default_factory=set)
    step_count: int = 0
    trajectory: List[Tuple[float, float]] = field(default_factory=list)

    def copy(self) -> 'GridState':
        return GridState(
            position=self.position,
            visited_zones=self.visited_zones.copy(),
            step_count=self.step_count,
            trajectory=self.trajectory.copy()
        )


@dataclass
class StepResult:
    """Result of taking a step in the environment"""
    next_state: GridState
    reward: float
    done: bool
    info: Dict

    # Detailed safety information
    entered_forbidden: bool = False
    violated_margin: bool = False
    reached_goal: bool = False
    out_of_bounds: bool = False


class GridWorld:
    """
    Grid world environment for navigation experiments.

    The environment supports:
    - Discrete actions (NSEW + WAIT)
    - Forbidden zones with safety margins
    - Goal-directed navigation
    - Proposition tracking for LTL evaluation
    """

    def __init__(
        self,
        width: int = 20,
        height: int = 15,
        cell_size: float = 1.0,
        max_steps: int = 100,
        propositions: Optional[PropositionEvaluator] = None
    ):
        """
        Args:
            width: Grid width in cells
            height: Grid height in cells
            cell_size: Size of each cell in continuous units
            max_steps: Maximum steps per episode
            propositions: Proposition evaluator (shared with Geofence)
        """
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.max_steps = max_steps

        # Proposition evaluator (shared module)
        self.props = propositions or PropositionEvaluator(grid_resolution=cell_size)

        # Episode state
        self.state: Optional[GridState] = None
        self.start_position: Tuple[float, float] = (1.0, 1.0)
        self.goal_position: Tuple[float, float] = (18.0, 13.0)

        # Metrics tracking
        self.episode_violations = 0
        self.episode_margin_violations = 0
        self.min_distance_to_forbidden = float('inf')

    def set_start(self, position: Tuple[float, float]) -> None:
        """Set the starting position"""
        self.start_position = position

    def set_goal(self, position: Tuple[float, float], threshold: float = 0.5) -> None:
        """Set the goal position"""
        self.goal_position = position
        self.props.set_goal(position, threshold)

    def reset(self, start_position: Optional[Tuple[float, float]] = None,
              seed: Optional[int] = None) -> GridState:
        """
        Reset the environment to initial state.

        Args:
            start_position: Override starting position
            seed: Random seed for reproducibility

        Returns:
            Initial state
        """
        if seed is not None:
            np.random.seed(seed)

        start = start_position or self.start_position

        self.state = GridState(
            position=start,
            visited_zones=set(),
            step_count=0,
            trajectory=[start]
        )

        # Reset metrics
        self.episode_violations = 0
        self.episode_margin_violations = 0
        self.min_distance_to_forbidden = float('inf')

        # Update initial proposition state
        prop_state = self.props.evaluate(start)
        self.state.visited_zones = prop_state.visited_zones.copy()

        logger.debug(f"Environment reset: start={start}, goal={self.goal_position}")

        return self.state.copy()

    def step(self, action: Action) -> StepResult:
        """
        Take a step in the environment.

        Args:
            action: Action to take

        Returns:
            StepResult with next state, reward, done flag, and info
        """
        if self.state is None:
            raise RuntimeError("Environment not reset. Call reset() first.")

        # Compute new position
        dx, dy = action.to_delta(self.cell_size)
        new_x = self.state.position[0] + dx
        new_y = self.state.position[1] + dy

        # Check bounds
        out_of_bounds = False
        if new_x < 0 or new_x >= self.width * self.cell_size:
            new_x = self.state.position[0]
            out_of_bounds = True
        if new_y < 0 or new_y >= self.height * self.cell_size:
            new_y = self.state.position[1]
            out_of_bounds = True

        new_position = (new_x, new_y)

        # Evaluate propositions at new position
        prop_state = self.props.evaluate(new_position, self.state.visited_zones)

        # Check safety violations
        entered_forbidden = self.props.is_in_forbidden_zone(new_position)
        violated_margin = self.props.is_within_margin(new_position)
        distance_to_forbidden = self.props.get_distance_to_nearest_forbidden(new_position)

        # Update metrics
        if entered_forbidden:
            self.episode_violations += 1
        if violated_margin and not entered_forbidden:
            self.episode_margin_violations += 1
        self.min_distance_to_forbidden = min(
            self.min_distance_to_forbidden,
            distance_to_forbidden
        )

        # Check goal
        reached_goal = self.props.is_at_goal(new_position)

        # Create new state
        new_state = GridState(
            position=new_position,
            visited_zones=prop_state.visited_zones,
            step_count=self.state.step_count + 1,
            trajectory=self.state.trajectory + [new_position]
        )

        # Compute reward
        reward = self._compute_reward(
            entered_forbidden, violated_margin, reached_goal, out_of_bounds
        )

        # Check termination
        done = (
            reached_goal or
            new_state.step_count >= self.max_steps or
            entered_forbidden  # Terminate on forbidden zone entry
        )

        # Update state
        self.state = new_state

        # Build info dict
        info = {
            'proposition_state': prop_state,
            'propositions': self.props.get_proposition_dict(prop_state),
            'distance_to_forbidden': distance_to_forbidden,
            'distance_to_goal': self._distance_to_goal(new_position),
            'step_count': new_state.step_count,
            'action_taken': action.name,
        }

        return StepResult(
            next_state=new_state.copy(),
            reward=reward,
            done=done,
            info=info,
            entered_forbidden=entered_forbidden,
            violated_margin=violated_margin,
            reached_goal=reached_goal,
            out_of_bounds=out_of_bounds
        )

    def _compute_reward(self, entered_forbidden: bool, violated_margin: bool,
                        reached_goal: bool, out_of_bounds: bool) -> float:
        """Compute step reward"""
        if reached_goal:
            return 100.0
        if entered_forbidden:
            return -100.0
        if violated_margin:
            return -10.0
        if out_of_bounds:
            return -5.0
        return -1.0  # Small penalty for each step (encourage efficiency)

    def _distance_to_goal(self, position: Tuple[float, float]) -> float:
        """Compute Euclidean distance to goal"""
        dx = position[0] - self.goal_position[0]
        dy = position[1] - self.goal_position[1]
        return (dx**2 + dy**2)**0.5

    def get_valid_actions(self, state: Optional[GridState] = None) -> List[Action]:
        """
        Get list of valid actions (those that don't go out of bounds).
        Note: This does NOT check safety - that's done by SELP-lite's constrained decoder.
        """
        state = state or self.state
        if state is None:
            return []

        valid = []
        for action in Action.all_actions():
            dx, dy = action.to_delta(self.cell_size)
            new_x = state.position[0] + dx
            new_y = state.position[1] + dy

            # Check bounds only
            if 0 <= new_x < self.width * self.cell_size:
                if 0 <= new_y < self.height * self.cell_size:
                    valid.append(action)

        return valid

    def simulate_action(self, state: GridState, action: Action) -> GridState:
        """
        Simulate taking an action from a given state without modifying environment.
        Used by SELP-lite's constrained decoder to evaluate action safety.
        """
        dx, dy = action.to_delta(self.cell_size)
        new_x = state.position[0] + dx
        new_y = state.position[1] + dy

        # Clamp to bounds
        new_x = max(0, min(new_x, (self.width - 1) * self.cell_size))
        new_y = max(0, min(new_y, (self.height - 1) * self.cell_size))

        new_position = (new_x, new_y)

        # Evaluate propositions
        prop_state = self.props.evaluate(new_position, state.visited_zones)

        return GridState(
            position=new_position,
            visited_zones=prop_state.visited_zones,
            step_count=state.step_count + 1,
            trajectory=state.trajectory + [new_position]
        )

    def get_proposition_state(self, state: Optional[GridState] = None) -> PropositionState:
        """Get current proposition state"""
        state = state or self.state
        if state is None:
            raise RuntimeError("No state available")
        return self.props.evaluate(state.position, state.visited_zones)

    def get_propositions(self, state: Optional[GridState] = None) -> Dict[str, bool]:
        """Get current propositions as dictionary"""
        prop_state = self.get_proposition_state(state)
        return self.props.get_proposition_dict(prop_state)

    def get_episode_metrics(self) -> Dict:
        """Get metrics for current episode"""
        return {
            'violations': self.episode_violations,
            'margin_violations': self.episode_margin_violations,
            'min_distance_to_forbidden': self.min_distance_to_forbidden,
            'steps': self.state.step_count if self.state else 0,
            'reached_goal': self.props.is_at_goal(self.state.position) if self.state else False,
        }

    def render_ascii(self, state: Optional[GridState] = None) -> str:
        """Render environment as ASCII art for debugging"""
        state = state or self.state
        if state is None:
            return "Environment not initialized"

        # Create grid
        grid = [['.' for _ in range(self.width)] for _ in range(self.height)]

        # Mark forbidden zones
        for y in range(self.height):
            for x in range(self.width):
                pos = (x * self.cell_size + self.cell_size/2,
                       y * self.cell_size + self.cell_size/2)
                if self.props.is_in_forbidden_zone(pos):
                    grid[self.height - 1 - y][x] = '#'
                elif self.props.is_within_margin(pos):
                    grid[self.height - 1 - y][x] = '~'

        # Mark goal
        gx = int(self.goal_position[0] / self.cell_size)
        gy = int(self.goal_position[1] / self.cell_size)
        if 0 <= gx < self.width and 0 <= gy < self.height:
            grid[self.height - 1 - gy][gx] = 'G'

        # Mark robot
        rx = int(state.position[0] / self.cell_size)
        ry = int(state.position[1] / self.cell_size)
        if 0 <= rx < self.width and 0 <= ry < self.height:
            grid[self.height - 1 - ry][rx] = 'R'

        # Convert to string
        lines = [''.join(row) for row in grid]
        return '\n'.join(lines)


# =============================================================================
# Factory for creating environments
# =============================================================================

def create_factory_environment(
    start: Tuple[float, float] = (1.0, 1.0),
    goal: Tuple[float, float] = (18.0, 13.0),
    max_steps: int = 100
) -> GridWorld:
    """Create grid world with factory environment"""
    props = create_factory_propositions(grid_size=20, cell_size=1.0)
    env = GridWorld(
        width=20,
        height=15,
        cell_size=1.0,
        max_steps=max_steps,
        propositions=props
    )
    env.set_start(start)
    env.set_goal(goal)
    return env


def create_simple_environment(
    start: Tuple[float, float] = (0.5, 0.5),
    goal: Tuple[float, float] = (9.5, 9.5),
    max_steps: int = 50
) -> GridWorld:
    """Create simple grid world for testing"""
    from .propositions import create_simple_grid_propositions
    props = create_simple_grid_propositions(grid_size=10, cell_size=1.0)
    env = GridWorld(
        width=10,
        height=10,
        cell_size=1.0,
        max_steps=max_steps,
        propositions=props
    )
    env.set_start(start)
    env.set_goal(goal)
    return env
