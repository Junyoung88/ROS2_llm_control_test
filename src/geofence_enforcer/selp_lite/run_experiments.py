#!/usr/bin/env python3
"""
SELP-lite vs Geofence Comparison Experiment Runner

This script runs batch experiments comparing:
1. SELP-lite (LTL-based constrained decoding)
2. Geofence (geometric projection/blocking)
3. No Guard (baseline without safety)

Metrics collected:
- Success Rate: goal reached
- Safety Rate: no forbidden zone entry
- Block/Mask Count: number of masked actions
- Plan Length: total steps
- Min Distance to Forbidden: minimum distance to forbidden zone boundary

Usage:
    python run_experiments.py --config configs/exp1.yaml --seeds 0-199
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

from selp_lite.env_grid import GridWorld, Action, create_factory_environment, create_simple_environment
from selp_lite.propositions import PropositionEvaluator, create_factory_propositions
from selp_lite.ltl_automaton import parse_ltl, create_automaton, LTLFormula
from selp_lite.nl2ltl import NL2LTLConverter, VotingResult, format_voting_log
from selp_lite.planner_constrained import (
    ConstrainedPlanner, GeofencePlanner, PlanResult, PlanStatus, format_plan_log
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class EpisodeMetrics:
    """Metrics for a single episode"""
    seed: int
    method: str  # "selp_lite", "geofence", "no_guard"
    constraint: str

    # Success metrics
    success: bool
    reached_goal: bool

    # Safety metrics
    safe: bool              # No violations of constraint-specific zone
    violations: int         # Violations of constraint-specific zone
    all_zone_violations: int  # Violations of ANY forbidden zone
    min_distance_to_forbidden: float
    min_distance_to_constraint_zone: float  # Distance to constraint-specific zone

    # Efficiency metrics
    steps: int
    masked_actions: int
    mask_ratio: float

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


def extract_constraint_zone(formula: LTLFormula) -> Optional[str]:
    """Extract zone name from LTL formula like G(!in_storage_racks)"""
    import re
    f = formula.normalized
    # Pattern: G(!in_<zone>) or similar
    match = re.search(r'in_(\w+)', f)
    if match:
        return match.group(1)
    return None


class ExperimentRunner:
    """
    Runs comparison experiments between SELP-lite and Geofence.
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

        # NL to LTL converter
        self.nl2ltl = NL2LTLConverter(num_candidates=5)
        self.nl2ltl.set_propositions(self.env.props.get_all_proposition_names())

        # Convert constraints to LTL
        self.ltl_formulas: List[LTLFormula] = []
        self.voting_results: List[VotingResult] = []
        self.constraint_zones: List[str] = []  # Zone names from constraints

        logger.info("=" * 60)
        logger.info("Converting constraints to LTL...")
        for constraint in config.constraints:
            result = self.nl2ltl.convert(constraint)
            self.ltl_formulas.append(result.selected_formula)
            self.voting_results.append(result)

            # Extract constraint zone name
            zone_name = extract_constraint_zone(result.selected_formula)
            self.constraint_zones.append(zone_name)

            if verbose:
                print(format_voting_log(result))

            logger.info(f"  '{constraint}' -> {result.selected_formula.normalized}")

    def _check_constraint_zone_violation(self, position: Tuple[float, float]) -> Tuple[bool, float]:
        """Check if position is in the constraint-specific zone.
        Returns: (is_in_zone, distance_to_zone)
        """
        constraint_zone = self.constraint_zones[0] if self.constraint_zones else None
        if constraint_zone and constraint_zone in self.env.props.zones:
            zone = self.env.props.zones[constraint_zone]
            from shapely.geometry import Point
            point = Point(position)
            is_inside = zone.polygon.contains(point)
            if is_inside:
                dist = -zone.polygon.exterior.distance(point)
            else:
                dist = zone.polygon.exterior.distance(point)
            return is_inside, dist
        return False, float('inf')

    def run_selp_lite(self, seed: int) -> EpisodeMetrics:
        """Run SELP-lite planner for one episode"""
        np.random.seed(seed)

        # Create combined automaton for all constraints
        # For simplicity, we use the first constraint (extend for multiple)
        automaton = create_automaton(self.ltl_formulas[0])

        # Create planner
        planner = ConstrainedPlanner(
            env=self.env,
            automaton=automaton,
            verbose_logging=self.verbose
        )

        # Run planning
        state = self.env.reset(seed=seed)
        result = planner.plan(start_state=state)

        # Track constraint-specific violations
        constraint_violations = 0
        min_dist_to_constraint = float('inf')
        for pos in result.trajectory:
            in_zone, dist = self._check_constraint_zone_violation(pos)
            min_dist_to_constraint = min(min_dist_to_constraint, dist)
            if in_zone:
                constraint_violations += 1

        # Compute metrics
        total_actions = result.total_steps * len(Action.all_actions())
        mask_ratio = result.masked_action_count / max(total_actions, 1)

        return EpisodeMetrics(
            seed=seed,
            method="selp_lite",
            constraint=self.config.constraints[0],
            success=result.status == PlanStatus.SUCCESS and constraint_violations == 0,
            reached_goal=result.reached_goal,
            safe=constraint_violations == 0,
            violations=constraint_violations,
            all_zone_violations=result.safety_violations,
            min_distance_to_forbidden=self.env.min_distance_to_forbidden,
            min_distance_to_constraint_zone=min_dist_to_constraint,
            steps=result.total_steps,
            masked_actions=result.masked_action_count,
            mask_ratio=mask_ratio,
            status=result.status.name,
            termination_reason=result.termination_reason
        )

    def run_geofence(self, seed: int) -> EpisodeMetrics:
        """Run Geofence planner for one episode - constraint-specific blocking"""
        np.random.seed(seed)

        # Run planning with constraint-specific blocking
        # (only block actions entering the constraint zone, not all zones)
        state = self.env.reset(seed=seed)
        constraint_zone = self.constraint_zones[0] if self.constraint_zones else None

        trajectory = [state.position]
        actions_list = []
        masked_count = 0
        all_violations = 0

        for step in range(self.config.max_steps):
            if self.env.props.is_at_goal(state.position):
                break

            # Evaluate actions - only block if entering constraint zone
            safe_actions = []
            for action in self.env.get_valid_actions(state):
                simulated = self.env.simulate_action(state, action)

                # Check if would enter constraint zone (with margin)
                if constraint_zone and constraint_zone in self.env.props.zones:
                    from shapely.geometry import Point
                    zone = self.env.props.zones[constraint_zone]
                    point = Point(simulated.position)
                    dist = zone.polygon.exterior.distance(point)
                    if zone.polygon.contains(point):
                        dist = -dist
                    if dist < self.config.safety_margin:
                        masked_count += 1
                        continue  # Block this action

                safe_actions.append(action)

            if not safe_actions:
                # Deadlock
                result_status = PlanStatus.CONSTRAINT_DEADLOCK
                break

            # Select action (same heuristic as SELP-lite)
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
            import random
            selected = random.choice(best)

            result = self.env.step(selected)
            state = result.next_state
            actions_list.append(selected)
            trajectory.append(state.position)

            if result.entered_forbidden or result.violated_margin:
                all_violations += 1
        else:
            result_status = PlanStatus.MAX_STEPS_EXCEEDED

        if self.env.props.is_at_goal(state.position):
            result_status = PlanStatus.SUCCESS

        # Create result object
        from selp_lite.planner_constrained import PlanResult
        result = PlanResult(
            status=result_status,
            trajectory=trajectory,
            actions=actions_list,
            total_steps=len(actions_list),
            masked_action_count=masked_count,
            safety_violations=all_violations,
            reached_goal=self.env.props.is_at_goal(state.position),
            termination_reason="Goal reached" if result_status == PlanStatus.SUCCESS else "Other"
        )

        # Track constraint-specific violations
        constraint_violations = 0
        min_dist_to_constraint = float('inf')
        for pos in result.trajectory:
            in_zone, dist = self._check_constraint_zone_violation(pos)
            min_dist_to_constraint = min(min_dist_to_constraint, dist)
            if in_zone:
                constraint_violations += 1

        # Compute metrics
        total_actions = result.total_steps * len(Action.all_actions())
        mask_ratio = result.masked_action_count / max(total_actions, 1)

        return EpisodeMetrics(
            seed=seed,
            method="geofence",
            constraint=self.config.constraints[0],
            success=result.status == PlanStatus.SUCCESS and constraint_violations == 0,
            reached_goal=result.reached_goal,
            safe=constraint_violations == 0,
            violations=constraint_violations,
            all_zone_violations=result.safety_violations,
            min_distance_to_forbidden=self.env.min_distance_to_forbidden,
            min_distance_to_constraint_zone=min_dist_to_constraint,
            steps=result.total_steps,
            masked_actions=result.masked_action_count,
            mask_ratio=mask_ratio,
            status=result.status.name,
            termination_reason=result.termination_reason
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

            # Random action selection (no safety check)
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
            constraint=self.config.constraints[0],
            success=reached_goal and constraint_violations == 0,
            reached_goal=reached_goal,
            safe=constraint_violations == 0,
            violations=constraint_violations,
            all_zone_violations=all_violations,
            min_distance_to_forbidden=self.env.min_distance_to_forbidden,
            min_distance_to_constraint_zone=min_dist_to_constraint,
            steps=steps,
            masked_actions=0,
            mask_ratio=0.0,
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
            # SELP-lite
            try:
                metrics = self.run_selp_lite(seed)
                self.results.append(metrics)
            except Exception as e:
                logger.error(f"SELP-lite seed {seed} failed: {e}")

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

        return self.results

    def compute_summary(self) -> Dict[str, Dict]:
        """Compute summary statistics by method"""
        summary = {}

        for method in ["selp_lite", "geofence", "no_guard"]:
            method_results = [r for r in self.results if r.method == method]

            if not method_results:
                continue

            n = len(method_results)
            success_rate = sum(1 for r in method_results if r.success) / n * 100
            safety_rate = sum(1 for r in method_results if r.safe) / n * 100
            goal_rate = sum(1 for r in method_results if r.reached_goal) / n * 100
            avg_steps = np.mean([r.steps for r in method_results])
            std_steps = np.std([r.steps for r in method_results])
            avg_violations = np.mean([r.violations for r in method_results])
            avg_all_violations = np.mean([r.all_zone_violations for r in method_results])
            avg_masked = np.mean([r.masked_actions for r in method_results])
            avg_mask_ratio = np.mean([r.mask_ratio for r in method_results])
            avg_min_dist = np.mean([r.min_distance_to_forbidden for r in method_results])
            avg_constraint_dist = np.mean([r.min_distance_to_constraint_zone for r in method_results])

            summary[method] = {
                'n_episodes': n,
                'success_rate': success_rate,
                'safety_rate': safety_rate,
                'goal_rate': goal_rate,
                'avg_steps': avg_steps,
                'std_steps': std_steps,
                'avg_violations': avg_violations,
                'avg_all_zone_violations': avg_all_violations,
                'avg_masked_actions': avg_masked,
                'avg_mask_ratio': avg_mask_ratio,
                'avg_min_distance': avg_min_dist,
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
        constraint_zone = self.constraint_zones[0] if self.constraint_zones else "N/A"

        print("\n" + "=" * 90)
        print("EXPERIMENT SUMMARY")
        print("=" * 90)
        print(f"Constraint: {self.config.constraints[0]}")
        print(f"Constraint zone: {constraint_zone}")
        print(f"Environment: {self.config.environment}")
        print(f"Start: {self.config.start_position}, Goal: {self.config.goal_position}")
        print("-" * 90)
        print(f"{'Method':<15} {'Success%':>10} {'Safety%':>10} {'Goal%':>10} "
              f"{'Steps':>12} {'Masked':>10} {'ConstraintDist':>14}")
        print("-" * 90)

        for method, stats in summary.items():
            print(f"{method:<15} "
                  f"{stats['success_rate']:>9.1f}% "
                  f"{stats['safety_rate']:>9.1f}% "
                  f"{stats['goal_rate']:>9.1f}% "
                  f"{stats['avg_steps']:>7.1f}+/-{stats['std_steps']:<4.1f}"
                  f"{stats['avg_masked_actions']:>10.1f} "
                  f"{stats['avg_constraint_distance']:>14.2f}")

        print("=" * 90)
        print("Note: Safety% = no violations of constraint zone (not all forbidden zones)")
        print(f"      ConstraintDist = min distance to {constraint_zone} (negative = inside)")
        print("=" * 90)


def load_config(config_path: str) -> ExperimentConfig:
    """Load experiment configuration from YAML"""
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)

    return ExperimentConfig(
        name=data.get('name', 'experiment'),
        environment=data.get('environment', 'factory'),
        start_position=tuple(data.get('start', [1.0, 1.0])),
        goal_position=tuple(data.get('goal', [18.0, 13.0])),
        constraints=data.get('constraints', ["Always avoid storage_racks"]),
        max_steps=data.get('max_steps', 100),
        safety_margin=data.get('safety_margin', 0.3),
        num_seeds=data.get('num_seeds', 50)
    )


def parse_seeds(seed_str: str) -> List[int]:
    """Parse seed string like '0-199' or '50'"""
    if '-' in seed_str:
        start, end = seed_str.split('-')
        return list(range(int(start), int(end) + 1))
    else:
        return list(range(int(seed_str)))


def main():
    parser = argparse.ArgumentParser(description='SELP-lite vs Geofence Comparison')
    parser.add_argument('--config', type=str, help='Path to config YAML')
    parser.add_argument('--scenario', type=str, default='factory',
                        choices=['factory', 'simple'], help='Predefined scenario')
    parser.add_argument('--seeds', type=str, default='50', help='Seeds (e.g., "0-199" or "50")')
    parser.add_argument('--output', type=str, default='results.csv', help='Output CSV path')
    parser.add_argument('--constraint', type=str, default='Always avoid storage_racks',
                        help='Natural language constraint')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')

    args = parser.parse_args()

    # Load or create config
    if args.config and os.path.exists(args.config):
        config = load_config(args.config)
    else:
        # Default config based on scenario
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

    # Save voting logs
    voting_log_path = args.output.replace('.csv', '_voting.txt')
    with open(voting_log_path, 'w') as f:
        for i, vr in enumerate(runner.voting_results):
            f.write(f"Constraint {i+1}: {config.constraints[i]}\n")
            f.write(format_voting_log(vr))
            f.write("\n\n")
    logger.info(f"Voting logs saved to: {voting_log_path}")


if __name__ == "__main__":
    main()
