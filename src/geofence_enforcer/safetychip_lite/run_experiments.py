#!/usr/bin/env python3
"""
SafetyChip-lite vs Geofence Comparison Experiment Runner

This script runs batch experiments comparing:
1. SafetyChip-lite (LTL monitor + pruning + reprompt)
2. Geofence (geometric projection/blocking)
3. No Guard (baseline without safety)

Metrics collected (as specified in the requirements):
- success: Goal reached
- safety_violation: Unsafe state entry (should be False for SafetyChip)
- pruned_count: Number of pruned actions
- reprompt_count: Number of reprompts
- path_length / steps: Total steps
- mean_min_distance_to_forbidden: Closest approach to forbidden zone

Usage:
    python run_experiments.py --config configs/factory_experiment.yaml --seeds 0-199
    python run_experiments.py --scenario factory --seeds 50 --output results.csv
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import yaml
import numpy as np

# Add parent path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safetychip_lite.env_grid import (
    GridWorld, Action, create_factory_environment, create_simple_environment
)
from safetychip_lite.propositions import PropositionEvaluator, create_factory_propositions
from safetychip_lite.nl2ltl import NL2LTLTranslator, format_translation_log
from safetychip_lite.ltl_monitor import create_monitor
from safetychip_lite.planner import create_planner, HeuristicPlanner
from safetychip_lite.agent_loop import (
    SafetyChipAgent, EpisodeResult, LoopStatus, format_episode_log
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class EpisodeMetrics:
    """Metrics for a single episode (matches paper requirements)"""
    seed: int
    method: str  # "safetychip", "geofence", "no_guard"
    constraint: str

    # Success metrics
    success: bool
    reached_goal: bool

    # Safety metrics (key SafetyChip metric)
    safety_violation: bool  # Should be False for SafetyChip

    # SafetyChip-specific metrics
    pruned_count: int       # Number of pruned actions
    reprompt_count: int     # Number of reprompts

    # Efficiency metrics
    steps: int
    path_length: int

    # Distance metrics
    min_distance_to_forbidden: float
    min_distance_to_constraint_zone: float

    # Status
    status: str
    termination_reason: str


@dataclass
class ExperimentConfig:
    """Configuration for an experiment"""
    name: str
    environment: str  # "factory" or "simple"
    start_position: Tuple[float, float]
    goal_position: Tuple[float, float]
    constraints: List[str]  # Natural language constraints
    max_steps: int = 100
    safety_margin: float = 0.3
    num_seeds: int = 50
    planner_type: str = "heuristic"


def extract_constraint_zone(constraints: List[str]) -> Optional[str]:
    """Extract zone name from constraint for metrics"""
    import re
    for c in constraints:
        # Look for "avoid X", "never enter X", etc.
        match = re.search(r'(?:avoid|enter|visit)\s+(?:the\s+)?(\w+)', c.lower())
        if match:
            return match.group(1)
    return None


class ExperimentRunner:
    """
    Runs comparison experiments between SafetyChip-lite and Geofence.
    """

    def __init__(self, config: ExperimentConfig, verbose: bool = False):
        self.config = config
        self.verbose = verbose
        self.results: List[EpisodeMetrics] = []

        # Set up environment
        if config.environment == "factory":
            self.env = create_factory_environment(
                start=config.start_position,
                goal=config.goal_position,
                max_steps=config.max_steps
            )
        else:
            self.env = create_simple_environment(
                start=config.start_position,
                goal=config.goal_position,
                max_steps=config.max_steps
            )

        self.env.props.set_safety_margin(config.safety_margin)

        # Extract constraint zone for metrics
        self.constraint_zone = extract_constraint_zone(config.constraints)

        # Translate constraints (for logging)
        self.translator = NL2LTLTranslator()
        self.translation_result = self.translator.translate(config.constraints)

        logger.info("=" * 60)
        logger.info("SafetyChip-lite Experiment Setup")
        logger.info("=" * 60)
        logger.info(f"Environment: {config.environment}")
        logger.info(f"Constraints: {config.constraints}")
        logger.info(f"LTL: {self.translation_result.combined_ltl}")
        logger.info(f"Constraint zone: {self.constraint_zone}")

    def _check_constraint_zone_violation(self, position: Tuple[float, float]) -> Tuple[bool, float]:
        """Check if position is in the constraint-specific zone"""
        if self.constraint_zone and self.constraint_zone in self.env.props.zones:
            zone = self.env.props.zones[self.constraint_zone]
            from shapely.geometry import Point
            point = Point(position)
            is_inside = zone.polygon.contains(point)
            if is_inside:
                dist = -zone.polygon.exterior.distance(point)
            else:
                dist = zone.polygon.exterior.distance(point)
            return is_inside, dist
        return False, float('inf')

    def run_safetychip(self, seed: int) -> EpisodeMetrics:
        """Run SafetyChip-lite agent for one episode"""
        np.random.seed(seed)

        planner = create_planner(self.config.planner_type)
        agent = SafetyChipAgent(
            env=self.env,
            planner=planner,
            verbose=self.verbose
        )

        result = agent.run(self.config.constraints, self.config.max_steps)

        # Track constraint-specific violations
        constraint_violations = 0
        min_dist_to_constraint = float('inf')
        for pos in result.path:
            in_zone, dist = self._check_constraint_zone_violation(pos)
            min_dist_to_constraint = min(min_dist_to_constraint, dist)
            if in_zone:
                constraint_violations += 1

        return EpisodeMetrics(
            seed=seed,
            method="safetychip",
            constraint=self.config.constraints[0] if self.config.constraints else "",
            success=result.success and constraint_violations == 0,
            reached_goal=result.success,
            safety_violation=constraint_violations > 0,
            pruned_count=result.pruned_count,
            reprompt_count=result.reprompt_count,
            steps=result.steps,
            path_length=len(result.path),
            min_distance_to_forbidden=result.min_distance_to_forbidden,
            min_distance_to_constraint_zone=min_dist_to_constraint,
            status=result.status.name,
            termination_reason=result.termination_reason
        )

    def run_geofence(self, seed: int) -> EpisodeMetrics:
        """Run Geofence planner for one episode (constraint-specific blocking)"""
        np.random.seed(seed)
        import random

        state = self.env.reset(seed=seed)
        trajectory = [state.position]
        actions_list = []
        pruned_count = 0
        all_violations = 0

        for step in range(self.config.max_steps):
            if self.env.props.is_at_goal(state.position):
                break

            # Evaluate actions - only block if entering constraint zone
            safe_actions = []
            for action in self.env.get_valid_actions(state):
                simulated = self.env.simulate_action(state, action)

                # Check if would enter constraint zone (with margin)
                if self.constraint_zone and self.constraint_zone in self.env.props.zones:
                    from shapely.geometry import Point
                    zone = self.env.props.zones[self.constraint_zone]
                    point = Point(simulated.position)
                    dist = zone.polygon.exterior.distance(point)
                    if zone.polygon.contains(point):
                        dist = -dist
                    if dist < self.config.safety_margin:
                        pruned_count += 1
                        continue  # Block this action

                safe_actions.append(action)

            if not safe_actions:
                break

            # Select action (same heuristic as SafetyChip)
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
            selected = random.choice(best)

            result = self.env.step(selected)
            state = result.next_state
            actions_list.append(selected)
            trajectory.append(state.position)

            if result.entered_forbidden or result.violated_margin:
                all_violations += 1

        # Track constraint-specific violations
        constraint_violations = 0
        min_dist_to_constraint = float('inf')
        for pos in trajectory:
            in_zone, dist = self._check_constraint_zone_violation(pos)
            min_dist_to_constraint = min(min_dist_to_constraint, dist)
            if in_zone:
                constraint_violations += 1

        reached_goal = self.env.props.is_at_goal(state.position)

        return EpisodeMetrics(
            seed=seed,
            method="geofence",
            constraint=self.config.constraints[0] if self.config.constraints else "",
            success=reached_goal and constraint_violations == 0,
            reached_goal=reached_goal,
            safety_violation=constraint_violations > 0,
            pruned_count=pruned_count,
            reprompt_count=0,  # Geofence doesn't reprompt
            steps=len(actions_list),
            path_length=len(trajectory),
            min_distance_to_forbidden=self.env.min_distance_to_forbidden,
            min_distance_to_constraint_zone=min_dist_to_constraint,
            status="SUCCESS" if reached_goal else "MAX_STEPS",
            termination_reason="Goal reached" if reached_goal else "Max steps"
        )

    def run_no_guard(self, seed: int) -> EpisodeMetrics:
        """Run without any safety mechanism (baseline)"""
        np.random.seed(seed)
        import random

        state = self.env.reset(seed=seed)
        trajectory = [state.position]
        steps = 0
        all_violations = 0

        for step in range(self.config.max_steps):
            if self.env.props.is_at_goal(state.position):
                break

            valid_actions = self.env.get_valid_actions(state)
            if not valid_actions:
                break

            # Heuristic: move toward goal
            goal = self.env.goal_position
            action_distances = []
            for action in valid_actions:
                simulated = self.env.simulate_action(state, action)
                dx = simulated.position[0] - goal[0]
                dy = simulated.position[1] - goal[1]
                dist = (dx**2 + dy**2)**0.5
                action_distances.append((action, dist))

            action_distances.sort(key=lambda x: x[1])
            selected = action_distances[0][0]

            result = self.env.step(selected)
            state = result.next_state
            trajectory.append(state.position)
            steps += 1

            if result.entered_forbidden or result.violated_margin:
                all_violations += 1

        # Track constraint-specific violations
        constraint_violations = 0
        min_dist_to_constraint = float('inf')
        for pos in trajectory:
            in_zone, dist = self._check_constraint_zone_violation(pos)
            min_dist_to_constraint = min(min_dist_to_constraint, dist)
            if in_zone:
                constraint_violations += 1

        reached_goal = self.env.props.is_at_goal(state.position)

        return EpisodeMetrics(
            seed=seed,
            method="no_guard",
            constraint=self.config.constraints[0] if self.config.constraints else "",
            success=reached_goal and constraint_violations == 0,
            reached_goal=reached_goal,
            safety_violation=constraint_violations > 0,
            pruned_count=0,
            reprompt_count=0,
            steps=steps,
            path_length=len(trajectory),
            min_distance_to_forbidden=self.env.min_distance_to_forbidden,
            min_distance_to_constraint_zone=min_dist_to_constraint,
            status="SUCCESS" if reached_goal else "MAX_STEPS",
            termination_reason="Goal reached" if reached_goal else "Max steps"
        )

    def run_all(self, seeds: List[int]) -> List[EpisodeMetrics]:
        """Run all methods for all seeds"""
        self.results = []

        total = len(seeds) * 3  # 3 methods
        completed = 0

        logger.info("=" * 60)
        logger.info(f"Running experiments: {len(seeds)} seeds x 3 methods = {total} episodes")
        logger.info("=" * 60)

        for seed in seeds:
            # SafetyChip-lite
            try:
                metrics = self.run_safetychip(seed)
                self.results.append(metrics)
            except Exception as e:
                logger.error(f"SafetyChip seed {seed} failed: {e}")
            completed += 1

            # Geofence
            try:
                metrics = self.run_geofence(seed)
                self.results.append(metrics)
            except Exception as e:
                logger.error(f"Geofence seed {seed} failed: {e}")
            completed += 1

            # No Guard
            try:
                metrics = self.run_no_guard(seed)
                self.results.append(metrics)
            except Exception as e:
                logger.error(f"No Guard seed {seed} failed: {e}")
            completed += 1

            if completed % 30 == 0:
                logger.info(f"Progress: {completed}/{total} ({100*completed/total:.1f}%)")

        logger.info(f"Progress: {completed}/{total} (100.0%)")
        return self.results

    def compute_summary(self) -> Dict[str, Dict]:
        """Compute summary statistics by method"""
        summary = {}

        for method in ["safetychip", "geofence", "no_guard"]:
            method_results = [r for r in self.results if r.method == method]

            if not method_results:
                continue

            n = len(method_results)
            success_rate = sum(1 for r in method_results if r.success) / n * 100
            safety_rate = sum(1 for r in method_results if not r.safety_violation) / n * 100
            goal_rate = sum(1 for r in method_results if r.reached_goal) / n * 100
            avg_steps = np.mean([r.steps for r in method_results])
            std_steps = np.std([r.steps for r in method_results])
            avg_pruned = np.mean([r.pruned_count for r in method_results])
            avg_reprompts = np.mean([r.reprompt_count for r in method_results])
            avg_constraint_dist = np.mean([r.min_distance_to_constraint_zone for r in method_results])

            summary[method] = {
                'n_episodes': n,
                'success_rate': success_rate,
                'safety_rate': safety_rate,
                'goal_rate': goal_rate,
                'avg_steps': avg_steps,
                'std_steps': std_steps,
                'avg_pruned': avg_pruned,
                'avg_reprompts': avg_reprompts,
                'avg_constraint_distance': avg_constraint_dist,
            }

        return summary

    def save_results(self, output_path: str) -> None:
        """Save results to CSV"""
        if not self.results:
            logger.warning("No results to save")
            return

        fieldnames = list(asdict(self.results[0]).keys())

        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in self.results:
                writer.writerow(asdict(result))

        logger.info(f"Results saved to {output_path}")

    def print_summary(self) -> None:
        """Print summary statistics"""
        summary = self.compute_summary()

        print("\n" + "=" * 95)
        print("SAFETYCHIP-LITE vs GEOFENCE EXPERIMENT SUMMARY")
        print("=" * 95)
        print(f"Constraint: {self.config.constraints[0] if self.config.constraints else 'None'}")
        print(f"Constraint zone: {self.constraint_zone}")
        print(f"LTL: {self.translation_result.combined_ltl}")
        print(f"Environment: {self.config.environment}")
        print(f"Start: {self.config.start_position}, Goal: {self.config.goal_position}")
        print("-" * 95)
        print(f"{'Method':<15} {'Success%':>10} {'Safety%':>10} {'Goal%':>10} "
              f"{'Steps':>12} {'Pruned':>8} {'Reprompt':>8} {'ConstDist':>10}")
        print("-" * 95)

        for method, stats in summary.items():
            print(f"{method:<15} "
                  f"{stats['success_rate']:>9.1f}% "
                  f"{stats['safety_rate']:>9.1f}% "
                  f"{stats['goal_rate']:>9.1f}% "
                  f"{stats['avg_steps']:>7.1f}+/-{stats['std_steps']:<4.1f}"
                  f"{stats['avg_pruned']:>8.1f} "
                  f"{stats['avg_reprompts']:>8.1f} "
                  f"{stats['avg_constraint_distance']:>10.2f}")

        print("=" * 95)
        print("Key SafetyChip metrics:")
        print("  - Safety%: Percentage of episodes with NO constraint zone violations")
        print("  - Pruned: Actions blocked BEFORE execution (SafetyChip guarantee)")
        print("  - Reprompt: Times planner was asked to propose alternative action")
        print("=" * 95)


def load_config(config_path: str) -> ExperimentConfig:
    """Load experiment configuration from YAML"""
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)

    return ExperimentConfig(
        name=data.get('name', 'experiment'),
        environment=data.get('environment', 'factory'),
        start_position=tuple(data.get('start', [1.0, 1.0])),
        goal_position=tuple(data.get('goal', [18.0, 13.0])),
        constraints=data.get('constraints', ["Never enter the storage_racks"]),
        max_steps=data.get('max_steps', 100),
        safety_margin=data.get('safety_margin', 0.3),
        num_seeds=data.get('num_seeds', 50),
        planner_type=data.get('planner_type', 'heuristic')
    )


def parse_seeds(seed_str: str) -> List[int]:
    """Parse seed string like '0-199' or '50'"""
    if '-' in seed_str:
        start, end = seed_str.split('-')
        return list(range(int(start), int(end) + 1))
    else:
        return list(range(int(seed_str)))


def main():
    parser = argparse.ArgumentParser(description='SafetyChip-lite vs Geofence Comparison')
    parser.add_argument('--config', type=str, help='Path to config YAML')
    parser.add_argument('--scenario', type=str, default='factory',
                        choices=['factory', 'simple'], help='Predefined scenario')
    parser.add_argument('--seeds', type=str, default='50', help='Seeds (e.g., "0-199" or "50")')
    parser.add_argument('--output', type=str, default='safetychip_results.csv', help='Output CSV')
    parser.add_argument('--constraint', type=str, default='Never enter the storage_racks',
                        help='Natural language constraint')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')

    args = parser.parse_args()

    # Load or create config
    if args.config and os.path.exists(args.config):
        config = load_config(args.config)
    else:
        if args.scenario == 'factory':
            config = ExperimentConfig(
                name="factory_comparison",
                environment="factory",
                start_position=(1.0, 1.0),
                goal_position=(18.0, 13.0),
                constraints=[args.constraint],
                max_steps=100,
                safety_margin=0.3
            )
        else:
            config = ExperimentConfig(
                name="simple_comparison",
                environment="simple",
                start_position=(0.5, 0.5),
                goal_position=(9.5, 9.5),
                constraints=[args.constraint],
                max_steps=50,
                safety_margin=0.3
            )

    # Parse seeds
    seeds = parse_seeds(args.seeds)
    config.num_seeds = len(seeds)

    # Run experiments
    runner = ExperimentRunner(config, verbose=args.verbose)

    start_time = time.time()
    runner.run_all(seeds)
    elapsed = time.time() - start_time

    # Save and print results
    runner.save_results(args.output)
    runner.print_summary()

    logger.info(f"\nTotal time: {elapsed:.1f} seconds")
    logger.info(f"Results saved to: {args.output}")

    # Save translation log
    log_path = args.output.replace('.csv', '_translation.txt')
    with open(log_path, 'w') as f:
        f.write(format_translation_log(runner.translation_result))
    logger.info(f"Translation log saved to: {log_path}")


if __name__ == "__main__":
    main()
