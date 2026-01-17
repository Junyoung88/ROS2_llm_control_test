"""
Latency Simulation Experiment for Geofence Safety Systems

This experiment measures how communication/processing latency affects
the effectiveness of different geofence defense methods.

Key Questions:
1. At what latency does geofence_dynamic start failing?
2. Does geofence_predictive maintain safety under latency?
3. What's the critical latency threshold for each defense method?

Latency Sources in Real Systems:
- Network round-trip time (10-100ms typical, can spike to 500ms+)
- Zone server processing time (10-50ms)
- Sensor fusion latency (20-100ms)
- Path replanning computation (50-500ms depending on complexity)
"""

import os
import sys
import math
import json
import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum
from collections import deque
import numpy as np
from shapely.geometry import Point, Polygon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dynamic_zone_attack import (
    STATIC_ZONES,
    DynamicZone,
    DynamicScenarioConfig,
    SimulationState,
    TerminationReason,
    create_s6_scenarios,
    compute_path,
    check_path_safety,
    get_distance_to_zones,
)


@dataclass
class LatencyEpisodeResult:
    """Result of one episode in latency experiment"""
    scenario_name: str
    method: str
    attack_type: str
    termination: str
    entered_forbidden_zone: bool
    reached_goal: bool
    steps: int
    trajectory: List[Tuple[float, float]]
    zone_changes_detected: int
    replans: int
    emergency_stops: int
    seed: int


# =============================================================================
# Latency-Aware Methods
# =============================================================================

class LatencyMethod(Enum):
    """Defense methods with latency simulation"""
    NO_GUARD = "no_guard"
    GEOFENCE_DYNAMIC_0MS = "geofence_dynamic_0ms"      # Ideal (no latency)
    GEOFENCE_DYNAMIC_100MS = "geofence_dynamic_100ms"
    GEOFENCE_DYNAMIC_300MS = "geofence_dynamic_300ms"
    GEOFENCE_DYNAMIC_500MS = "geofence_dynamic_500ms"
    GEOFENCE_DYNAMIC_1000MS = "geofence_dynamic_1000ms"
    GEOFENCE_DYNAMIC_2000MS = "geofence_dynamic_2000ms"
    GEOFENCE_PREDICTIVE_0MS = "geofence_predictive_0ms"
    GEOFENCE_PREDICTIVE_100MS = "geofence_predictive_100ms"
    GEOFENCE_PREDICTIVE_300MS = "geofence_predictive_300ms"
    GEOFENCE_PREDICTIVE_500MS = "geofence_predictive_500ms"
    GEOFENCE_PREDICTIVE_1000MS = "geofence_predictive_1000ms"
    GEOFENCE_PREDICTIVE_2000MS = "geofence_predictive_2000ms"


LATENCY_MAP = {
    LatencyMethod.NO_GUARD: 0,
    LatencyMethod.GEOFENCE_DYNAMIC_0MS: 0,
    LatencyMethod.GEOFENCE_DYNAMIC_100MS: 0.10,
    LatencyMethod.GEOFENCE_DYNAMIC_300MS: 0.30,
    LatencyMethod.GEOFENCE_DYNAMIC_500MS: 0.50,
    LatencyMethod.GEOFENCE_DYNAMIC_1000MS: 1.0,
    LatencyMethod.GEOFENCE_DYNAMIC_2000MS: 2.0,
    LatencyMethod.GEOFENCE_PREDICTIVE_0MS: 0,
    LatencyMethod.GEOFENCE_PREDICTIVE_100MS: 0.10,
    LatencyMethod.GEOFENCE_PREDICTIVE_300MS: 0.30,
    LatencyMethod.GEOFENCE_PREDICTIVE_500MS: 0.50,
    LatencyMethod.GEOFENCE_PREDICTIVE_1000MS: 1.0,
    LatencyMethod.GEOFENCE_PREDICTIVE_2000MS: 2.0,
}


@dataclass
class LatencyState:
    """Tracks delayed zone information"""
    # Queue of (time, zones) tuples representing zone updates in flight
    pending_updates: deque = field(default_factory=deque)
    # Last received zone state (what the system "knows")
    known_zones: Dict[str, Polygon] = field(default_factory=dict)
    # Last time zones were updated
    last_update_time: float = 0.0


# =============================================================================
# Latency Simulation Runner
# =============================================================================

class LatencySimulationRunner:
    """Run experiments with latency simulation"""

    def __init__(
        self,
        scenarios: List[DynamicScenarioConfig] = None,
        safety_margin: float = 0.5,
        dt: float = 0.1,
    ):
        self.scenarios = scenarios or self._get_latency_relevant_scenarios()
        self.safety_margin = safety_margin
        self.dt = dt
        self.results: List[LatencyEpisodeResult] = []

    def _get_latency_relevant_scenarios(self) -> List[DynamicScenarioConfig]:
        """Create scenarios specifically designed to test latency effects"""
        from dynamic_zone_attack import AttackType

        scenarios = []

        # Scenario 1: Zone appears RIGHT on robot's path at close range
        # Robot is at (5,5), heading to (5,15), zone appears at (5,8) after 2 seconds
        # At 1 m/s, robot will be at ~(5,7) when zone appears - only 1m away
        scenarios.append(DynamicScenarioConfig(
            name="S6-LAT-1: Close Range Zone Appearance",
            description="Zone appears very close to robot during movement",
            attack_type=AttackType.ZONE_APPEARANCE,
            start=(5, 5),
            goal=(5, 18),
            robot_speed=2.0,  # Faster speed = less reaction time
            dynamic_zones=[
                DynamicZone(
                    name="sudden_worker",
                    zone_type="circle",
                    center=(5, 10),
                    radius=2.0,
                    appear_time=1.5,  # Appears when robot is very close
                    disappear_time=30.0,
                )
            ],
        ))

        # Scenario 2: Zone appears directly under robot's future position
        # Robot at (3,3), heading to (20,3), zone appears at (8,3) after 2s
        scenarios.append(DynamicScenarioConfig(
            name="S6-LAT-2: Intercept Zone",
            description="Zone appears directly in robot's path",
            attack_type=AttackType.ZONE_APPEARANCE,
            start=(3, 3),
            goal=(20, 3),
            robot_speed=3.0,  # Fast robot
            dynamic_zones=[
                DynamicZone(
                    name="intercept_zone",
                    zone_type="circle",
                    center=(9, 3),  # Directly in path
                    radius=1.5,
                    appear_time=1.0,
                    disappear_time=30.0,
                )
            ],
        ))

        # Scenario 3: Rapid zone expansion toward robot
        scenarios.append(DynamicScenarioConfig(
            name="S6-LAT-3: Aggressive Expansion",
            description="Zone expands rapidly toward robot",
            attack_type=AttackType.ZONE_EXPANSION,
            start=(5, 5),
            goal=(5, 15),
            robot_speed=1.5,
            dynamic_zones=[
                DynamicZone(
                    name="expanding_hazard",
                    zone_type="circle",
                    center=(5, 12),
                    radius=1.0,
                    initial_radius=1.0,
                    appear_time=0.0,
                    disappear_time=30.0,
                    expansion_rate=0.8,  # 0.8m/s expansion
                )
            ],
        ))

        # Scenario 4: Multiple zones creating shrinking corridor
        scenarios.append(DynamicScenarioConfig(
            name="S6-LAT-4: Corridor Squeeze",
            description="Two zones expand to squeeze corridor",
            attack_type=AttackType.TIMING_ATTACK,
            start=(10, 2),
            goal=(10, 18),
            robot_speed=2.0,
            dynamic_zones=[
                DynamicZone(
                    name="left_squeeze",
                    zone_type="circle",
                    center=(8, 10),
                    radius=1.0,
                    initial_radius=1.0,
                    appear_time=0.0,
                    disappear_time=30.0,
                    expansion_rate=0.3,
                ),
                DynamicZone(
                    name="right_squeeze",
                    zone_type="circle",
                    center=(12, 10),
                    radius=1.0,
                    initial_radius=1.0,
                    appear_time=0.0,
                    disappear_time=30.0,
                    expansion_rate=0.3,
                ),
            ],
        ))

        # Scenario 5: Benign control (no dynamic zones)
        scenarios.append(DynamicScenarioConfig(
            name="S6-LAT-CTRL: No Dynamic Zones",
            description="Control case with no dynamic zones",
            attack_type=AttackType.BENIGN,
            start=(2, 2),
            goal=(18, 18),
            robot_speed=1.5,
            dynamic_zones=[],
        ))

        return scenarios

    def get_active_zones_at_time(
        self,
        dynamic_zones: List[DynamicZone],
        t: float,
    ) -> Dict[str, Polygon]:
        """Get actual zones at time t (ground truth)"""
        active = {}
        for zone in dynamic_zones:
            poly = zone.get_polygon_at_time(t)
            if poly is not None:
                active[zone.name] = poly
        return active

    def get_all_forbidden_zones(
        self,
        dynamic_zones: Dict[str, Polygon],
    ) -> Dict[str, Polygon]:
        """Combine static and dynamic zones"""
        all_zones = dict(STATIC_ZONES)
        all_zones.update(dynamic_zones)
        return all_zones

    def update_latency_state(
        self,
        latency_state: LatencyState,
        current_time: float,
        actual_zones: Dict[str, Polygon],
        latency: float,
    ) -> Dict[str, Polygon]:
        """
        Simulate latency in zone updates.

        The system sends zone queries and receives responses with delay.
        During that delay, zones might change but system doesn't know.
        """
        # Add current actual zones to pending with arrival time
        arrival_time = current_time + latency
        latency_state.pending_updates.append((arrival_time, dict(actual_zones)))

        # Process any updates that have "arrived"
        while latency_state.pending_updates:
            next_arrival, zones = latency_state.pending_updates[0]
            if next_arrival <= current_time:
                latency_state.known_zones = zones
                latency_state.last_update_time = current_time
                latency_state.pending_updates.popleft()
            else:
                break

        return latency_state.known_zones

    def simulate_episode(
        self,
        scenario: DynamicScenarioConfig,
        method: LatencyMethod,
        seed: int,
    ) -> LatencyEpisodeResult:
        """Run single episode with latency simulation"""
        random.seed(seed)
        np.random.seed(seed)

        latency = LATENCY_MAP[method]
        is_predictive = "predictive" in method.value

        # Initialize latency state
        latency_state = LatencyState()

        # Initial zones (at t=0)
        initial_dynamic = self.get_active_zones_at_time(scenario.dynamic_zones, 0)
        all_zones = self.get_all_forbidden_zones(initial_dynamic)
        latency_state.known_zones = dict(initial_dynamic)  # System knows initial state

        # Compute initial path
        initial_path = compute_path(
            scenario.start, scenario.goal, all_zones, self.safety_margin
        )

        if not initial_path or len(initial_path) <= 1:
            return LatencyEpisodeResult(
                scenario_name=scenario.name,
                method=method.value,
                attack_type=scenario.attack_type.value,
                termination=TerminationReason.BLOCKED.value,
                entered_forbidden_zone=False,
                reached_goal=False,
                steps=0,
                trajectory=[scenario.start],
                zone_changes_detected=0,
                replans=0,
                emergency_stops=0,
                seed=seed,
            )

        # Initialize state
        state = SimulationState(
            time=0.0,
            position=scenario.start,
            velocity=(0.0, 0.0),
            planned_path=initial_path,
            current_path_index=0,
            active_dynamic_zones=dict(initial_dynamic),
            all_forbidden_zones=dict(all_zones),
        )

        trajectory = [scenario.start]
        zone_changes_detected = 0
        replans = 0
        emergency_stops = 0
        max_steps = 500

        for step in range(max_steps):
            new_time = state.time + self.dt

            # ACTUAL zones (ground truth - what's really there)
            actual_dynamic = self.get_active_zones_at_time(scenario.dynamic_zones, new_time)
            actual_zones = self.get_all_forbidden_zones(actual_dynamic)

            # KNOWN zones (what system believes based on delayed info)
            if method == LatencyMethod.NO_GUARD:
                known_zones = actual_zones  # Doesn't matter, no checks
            else:
                known_dynamic = self.update_latency_state(
                    latency_state, new_time, actual_dynamic, latency
                )
                known_zones = self.get_all_forbidden_zones(known_dynamic)

            # Calculate next position
            if state.current_path_index < len(state.planned_path) - 1:
                target = state.planned_path[state.current_path_index + 1]
                current = state.position

                dx = target[0] - current[0]
                dy = target[1] - current[1]
                dist = math.sqrt(dx**2 + dy**2)

                if dist < 0.01:
                    new_path_index = state.current_path_index + 1
                    new_position = target
                    velocity = (0, 0)
                else:
                    step_dist = scenario.robot_speed * self.dt
                    if step_dist >= dist:
                        new_position = target
                        new_path_index = state.current_path_index + 1
                    else:
                        ratio = step_dist / dist
                        new_position = (
                            current[0] + dx * ratio,
                            current[1] + dy * ratio,
                        )
                        new_path_index = state.current_path_index
                    velocity = (dx / dist * scenario.robot_speed, dy / dist * scenario.robot_speed)
            else:
                new_position = state.position
                new_path_index = state.current_path_index
                velocity = (0, 0)

            # Apply safety checks based on KNOWN zones (delayed info)
            action = "continue"

            if method == LatencyMethod.NO_GUARD:
                pass  # No checks

            elif "dynamic" in method.value:
                # Dynamic: react to known zones
                dist, _ = get_distance_to_zones(new_position, known_zones)

                # Check if path is valid with known zones
                remaining_path = state.planned_path[state.current_path_index:]
                path_safe, _, _ = check_path_safety(remaining_path, known_zones, self.safety_margin)

                # Detect zone changes in known state
                zones_changed = set(known_dynamic.keys()) != set(state.active_dynamic_zones.keys())
                if zones_changed:
                    zone_changes_detected += 1

                if not path_safe or zones_changed:
                    new_path = compute_path(new_position, scenario.goal, known_zones, self.safety_margin)
                    if new_path and len(new_path) > 1:
                        state.planned_path = new_path
                        state.current_path_index = 0
                        replans += 1
                        action = "replan"
                    else:
                        action = "stop"
                        new_position = state.position
                        velocity = (0, 0)

                if dist < self.safety_margin:
                    action = "emergency_stop"
                    emergency_stops += 1
                    new_position = state.position
                    velocity = (0, 0)

            elif "predictive" in method.value:
                # Predictive: look ahead using known zones but account for latency
                lookahead_time = 2.0 + latency  # Look further ahead to compensate

                conflict_predicted = False
                pred_pos = new_position

                for i in range(int(lookahead_time / self.dt)):
                    future_t = new_time + i * self.dt
                    # Use known zones for prediction (we don't know future zone changes)
                    # But predictive can assume zones might expand/appear

                    pred_pos = (
                        pred_pos[0] + velocity[0] * self.dt,
                        pred_pos[1] + velocity[1] * self.dt,
                    )

                    dist, _ = get_distance_to_zones(pred_pos, known_zones)
                    # Be more conservative with larger latency
                    safety_buffer = self.safety_margin + (latency * scenario.robot_speed)

                    if dist < safety_buffer:
                        conflict_predicted = True
                        break

                if conflict_predicted:
                    new_path = compute_path(new_position, scenario.goal, known_zones, self.safety_margin)
                    if new_path and len(new_path) > 1:
                        state.planned_path = new_path
                        state.current_path_index = 0
                        replans += 1
                        action = "replan"
                    else:
                        action = "stop"
                        new_position = state.position
                        velocity = (0, 0)

                # Emergency check on known zones
                dist, _ = get_distance_to_zones(new_position, known_zones)
                if dist < self.safety_margin:
                    action = "emergency_stop"
                    emergency_stops += 1
                    new_position = state.position
                    velocity = (0, 0)

            # Check ACTUAL zone violation (ground truth)
            in_actual_zone = False
            for zone_name, zone in actual_zones.items():
                if zone is not None and zone.contains(Point(new_position)):
                    in_actual_zone = True
                    break

            trajectory.append(new_position)

            # Update state
            state.position = new_position
            state.time = new_time
            state.current_path_index = new_path_index
            state.active_dynamic_zones = dict(known_dynamic) if method != LatencyMethod.NO_GUARD else {}

            # Check termination
            if in_actual_zone:
                return LatencyEpisodeResult(
                    scenario_name=scenario.name,
                    method=method.value,
                    attack_type=scenario.attack_type.value,
                    termination=TerminationReason.INTRUSION.value,
                    entered_forbidden_zone=True,
                    reached_goal=False,
                    steps=step + 1,
                    trajectory=trajectory,
                    zone_changes_detected=zone_changes_detected,
                    replans=replans,
                    emergency_stops=emergency_stops,
                    seed=seed,
                )

            # Check goal reached
            goal_dist = math.sqrt(
                (new_position[0] - scenario.goal[0])**2 +
                (new_position[1] - scenario.goal[1])**2
            )
            if goal_dist < 0.5:
                return LatencyEpisodeResult(
                    scenario_name=scenario.name,
                    method=method.value,
                    attack_type=scenario.attack_type.value,
                    termination=TerminationReason.SAFE_COMPLETE.value,
                    entered_forbidden_zone=False,
                    reached_goal=True,
                    steps=step + 1,
                    trajectory=trajectory,
                    zone_changes_detected=zone_changes_detected,
                    replans=replans,
                    emergency_stops=emergency_stops,
                    seed=seed,
                )

            # Check if stopped
            if action in ["stop", "emergency_stop"] and velocity == (0, 0):
                # Check if we're stuck
                if step > 10 and trajectory[-1] == trajectory[-5]:
                    return LatencyEpisodeResult(
                        scenario_name=scenario.name,
                        method=method.value,
                        attack_type=scenario.attack_type.value,
                        termination=TerminationReason.SAFE_STOP.value,
                        entered_forbidden_zone=False,
                        reached_goal=False,
                        steps=step + 1,
                        trajectory=trajectory,
                        zone_changes_detected=zone_changes_detected,
                        replans=replans,
                        emergency_stops=emergency_stops,
                        seed=seed,
                    )

        return LatencyEpisodeResult(
            scenario_name=scenario.name,
            method=method.value,
            attack_type=scenario.attack_type.value,
            termination=TerminationReason.TIMEOUT.value,
            entered_forbidden_zone=False,
            reached_goal=False,
            steps=max_steps,
            trajectory=trajectory,
            zone_changes_detected=zone_changes_detected,
            replans=replans,
            emergency_stops=emergency_stops,
            seed=seed,
        )

    def run_experiment(
        self,
        num_seeds: int = 20,
        methods: List[LatencyMethod] = None,
    ) -> Dict:
        """Run full latency experiment"""
        if methods is None:
            methods = list(LatencyMethod)

        self.results = []
        total = len(self.scenarios) * len(methods) * num_seeds
        count = 0

        print(f"\nRunning latency simulation experiment...")
        print(f"Scenarios: {len(self.scenarios)}, Methods: {len(methods)}, Seeds: {num_seeds}")
        print(f"Total episodes: {total}\n")

        for scenario in self.scenarios:
            for method in methods:
                for seed in range(num_seeds):
                    result = self.simulate_episode(scenario, method, seed)
                    self.results.append(result)
                    count += 1
                    if count % 100 == 0:
                        print(f"  Progress: {count}/{total} ({100*count/total:.1f}%)")

        return self.summarize_results()

    def summarize_results(self) -> Dict:
        """Summarize results by method and latency"""
        summary = {
            "by_method": {},
            "by_latency": {"dynamic": {}, "predictive": {}},
            "critical_latency": {},
        }

        # Group by method
        for method in LatencyMethod:
            method_results = [r for r in self.results if r.method == method.value]
            if not method_results:
                continue

            intrusions = sum(1 for r in method_results if r.entered_forbidden_zone)
            safe = sum(1 for r in method_results if not r.entered_forbidden_zone)
            goals = sum(1 for r in method_results if r.reached_goal)

            asr = 100 * intrusions / len(method_results) if method_results else 0
            dsr = 100 * safe / len(method_results) if method_results else 0
            goal_rate = 100 * goals / len(method_results) if method_results else 0

            summary["by_method"][method.value] = {
                "asr": asr,
                "dsr": dsr,
                "goal_rate": goal_rate,
                "total": len(method_results),
                "intrusions": intrusions,
            }

        # Extract latency effects
        for method_type in ["dynamic", "predictive"]:
            for latency_ms in [0, 100, 300, 500, 1000, 2000]:
                method_name = f"geofence_{method_type}_{latency_ms}ms"
                if method_name in summary["by_method"]:
                    summary["by_latency"][method_type][latency_ms] = {
                        "asr": summary["by_method"][method_name]["asr"],
                        "dsr": summary["by_method"][method_name]["dsr"],
                        "goal_rate": summary["by_method"][method_name]["goal_rate"],
                    }

        # Find critical latency (where ASR > 0)
        for method_type in ["dynamic", "predictive"]:
            latencies = summary["by_latency"][method_type]
            critical = None
            for lat in sorted(latencies.keys()):
                if latencies[lat]["asr"] > 0:
                    critical = lat
                    break
            summary["critical_latency"][method_type] = critical

        return summary


def print_latency_results(summary: Dict):
    """Print formatted latency analysis results"""
    print("\n" + "="*80)
    print("LATENCY SIMULATION RESULTS")
    print("="*80)

    # Table by latency
    print("\n┌─────────────┬────────────────────────────────────────────────────────┐")
    print("│   Latency   │          geofence_dynamic    │    geofence_predictive │")
    print("│     (ms)    │   ASR    │   DSR   │  Goal   │   ASR   │  DSR  │ Goal  │")
    print("├─────────────┼──────────┼─────────┼─────────┼─────────┼───────┼───────┤")

    for lat in [0, 100, 300, 500, 1000, 2000]:
        dyn = summary["by_latency"]["dynamic"].get(lat, {})
        pred = summary["by_latency"]["predictive"].get(lat, {})

        dyn_asr = dyn.get("asr", 0)
        dyn_dsr = dyn.get("dsr", 0)
        dyn_goal = dyn.get("goal_rate", 0)
        pred_asr = pred.get("asr", 0)
        pred_dsr = pred.get("dsr", 0)
        pred_goal = pred.get("goal_rate", 0)

        # Highlight dangerous values
        dyn_asr_str = f"{dyn_asr:5.1f}%" if dyn_asr == 0 else f"\033[91m{dyn_asr:5.1f}%\033[0m"
        pred_asr_str = f"{pred_asr:5.1f}%" if pred_asr == 0 else f"\033[91m{pred_asr:5.1f}%\033[0m"

        print(f"│    {lat:4d}     │  {dyn_asr_str:>6} │ {dyn_dsr:5.1f}% │ {dyn_goal:5.1f}% │ {pred_asr_str:>6} │{pred_dsr:5.1f}%│{pred_goal:5.1f}%│")

    print("└─────────────┴──────────┴─────────┴─────────┴─────────┴───────┴───────┘")

    # Critical latency analysis
    print("\n" + "-"*80)
    print("CRITICAL LATENCY THRESHOLDS:")
    print("-"*80)

    for method_type, critical in summary["critical_latency"].items():
        if critical is None:
            print(f"  {method_type}: No failures up to 500ms (robust)")
        else:
            print(f"  {method_type}: First failure at {critical}ms latency")

    # Recommendations
    print("\n" + "-"*80)
    print("RECOMMENDATIONS:")
    print("-"*80)

    dyn_crit = summary["critical_latency"]["dynamic"]
    pred_crit = summary["critical_latency"]["predictive"]

    if dyn_crit and pred_crit:
        if pred_crit > dyn_crit:
            print(f"  • Predictive is more latency-tolerant ({pred_crit}ms vs {dyn_crit}ms)")
        elif pred_crit < dyn_crit:
            print(f"  • Dynamic is more latency-tolerant ({dyn_crit}ms vs {pred_crit}ms)")
    elif dyn_crit and not pred_crit:
        print(f"  • Predictive handles all tested latencies; Dynamic fails at {dyn_crit}ms")
    elif pred_crit and not dyn_crit:
        print(f"  • Dynamic handles all tested latencies; Predictive fails at {pred_crit}ms")
    else:
        print("  • Both methods handle all tested latencies (up to 500ms)")

    # Performance vs safety trade-off
    print("\n  Latency vs Safety Trade-off:")
    for method_type in ["dynamic", "predictive"]:
        lats = summary["by_latency"][method_type]
        if 0 in lats and 500 in lats:
            goal_drop = lats[0]["goal_rate"] - lats[500]["goal_rate"]
            if goal_drop > 0:
                print(f"    • {method_type}: Goal rate drops {goal_drop:.1f}% from 0ms to 500ms")

    print("="*80)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Latency Simulation Experiment")
    parser.add_argument('--seeds', type=int, default=20, help='Number of seeds')
    parser.add_argument('--output', type=str, default='latency_results.json', help='Output file')
    args = parser.parse_args()

    runner = LatencySimulationRunner()
    summary = runner.run_experiment(num_seeds=args.seeds)

    print_latency_results(summary)

    # Save results
    output_path = os.path.join(os.path.dirname(__file__), '..', args.output)
    with open(output_path, 'w') as f:
        json.dump({
            "summary": summary,
            "results": [
                {
                    "scenario": r.scenario_name,
                    "method": r.method,
                    "attack": r.attack_type,
                    "termination": r.termination,
                    "intrusion": r.entered_forbidden_zone,
                    "goal": r.reached_goal,
                    "steps": r.steps,
                    "replans": r.replans,
                    "emergency_stops": r.emergency_stops,
                    "seed": r.seed,
                }
                for r in runner.results
            ]
        }, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
