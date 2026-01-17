"""
S7: Latency-Based Attack and Defense Experiment

This scenario tests how communication latency affects attack success and defense
effectiveness. It simulates an adversary exploiting network delays to bypass
geofence protections.

Attack Model:
============
- Adversary can inject/modify zone information with delayed propagation
- Robot receives zone updates with variable latency (0ms to 2000ms)
- Attacks exploit the "window of vulnerability" during latency

Defense Methods:
===============
1. no_guard: No safety check (baseline, expected 100% ASR)
2. fixed_margin: Static safety margin (fails at high latency)
3. velocity_compensated: Margin grows with velocity (partial defense)
4. realtime_adaptive: Measures latency, adjusts margins dynamically (best defense)

Key Metrics:
===========
- ASR (Attack Success Rate): % of attacks that cause zone violation
- DSR (Defense Success Rate): % of attacks blocked (100% - ASR)
- Latency Threshold: Maximum latency where defense maintains 0% ASR

Expected Results:
================
- realtime_adaptive should maintain low ASR even at high latency
- fixed_margin should fail at latency > 500ms
- The "critical latency threshold" varies by defense method
"""

import os
import sys
import math
import json
import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum
from collections import deque
import numpy as np
from shapely.geometry import Point, Polygon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dynamic_zone_attack import (
    STATIC_ZONES,
    DynamicZone,
    DynamicScenarioConfig,
    AttackType,
    TerminationReason,
    compute_path,
)


# =============================================================================
# Defense Methods
# =============================================================================

class DefenseMethod(Enum):
    """Defense methods for latency compensation"""
    NO_GUARD = "no_guard"
    FIXED_MARGIN = "fixed_margin"
    VELOCITY_COMPENSATED = "velocity_compensated"
    REALTIME_ADAPTIVE = "realtime_adaptive"


@dataclass
class DefenseParams:
    """Parameters for defense checking"""
    base_margin: float = 0.5
    latency: float = 0.0  # Known/estimated latency
    max_acceleration: float = 2.0

    # For realtime_adaptive
    margin_multiplier: float = 1.0
    velocity_factor: float = 1.0


class LatencyTracker:
    """Simulates real-time latency measurement"""

    def __init__(self, window_size: int = 10):
        self.measurements: deque = deque(maxlen=window_size)
        self.spike_threshold = 2.0
        self.emergency_threshold = 3.0

    def record(self, latency: float, noise: float = 0.1):
        """Record a latency measurement with noise"""
        noisy = latency + random.gauss(0, latency * noise)
        self.measurements.append(max(0, noisy))

    def get_estimate(self) -> float:
        """Get conservative latency estimate (mean + std)"""
        if not self.measurements:
            return 0.1
        mean = sum(self.measurements) / len(self.measurements)
        if len(self.measurements) > 1:
            var = sum((x - mean)**2 for x in self.measurements) / len(self.measurements)
            std = math.sqrt(var)
            return mean + std
        return mean

    def is_spike(self) -> bool:
        """Detect latency spike"""
        if len(self.measurements) < 3:
            return False
        recent = self.measurements[-1]
        older = list(self.measurements)[:-1]
        avg = sum(older) / len(older)
        return recent > avg * self.spike_threshold

    def is_emergency(self) -> bool:
        """Detect emergency latency"""
        if len(self.measurements) < 3:
            return False
        recent = self.measurements[-1]
        older = list(self.measurements)[:-1]
        avg = sum(older) / len(older)
        return recent > avg * self.emergency_threshold or recent > 2.0

    def get_adaptive_params(self, base_margin: float = 0.5) -> DefenseParams:
        """Get adaptive defense parameters"""
        params = DefenseParams(base_margin=base_margin)
        params.latency = self.get_estimate()

        if self.is_emergency():
            params.margin_multiplier = 2.0
            params.velocity_factor = 2.0
            params.latency = max(self.measurements) * 1.5 if self.measurements else 0.5
        elif self.is_spike():
            params.margin_multiplier = 1.5
            params.velocity_factor = 1.5
            params.latency = max(self.measurements) * 1.2 if self.measurements else 0.3

        return params

    def reset(self):
        self.measurements.clear()


# =============================================================================
# Safety Check Functions
# =============================================================================

def check_fixed_margin(
    position: Tuple[float, float],
    zones: Dict[str, Polygon],
    params: DefenseParams,
) -> Tuple[bool, float]:
    """Fixed margin check: d > base_margin"""
    min_dist = float('inf')
    for zone in zones.values():
        if zone is not None:
            dist = Point(position).distance(zone)
            min_dist = min(min_dist, dist)
            if dist < params.base_margin:
                return False, dist
    return True, min_dist


def check_velocity_compensated(
    position: Tuple[float, float],
    velocity: Tuple[float, float],
    zones: Dict[str, Polygon],
    params: DefenseParams,
) -> Tuple[bool, float]:
    """Velocity-compensated: d > base_margin + v*tau + 0.5*a*tau^2"""
    speed = math.sqrt(velocity[0]**2 + velocity[1]**2)
    margin = (params.base_margin +
              speed * params.latency +
              0.5 * params.max_acceleration * params.latency**2)

    min_dist = float('inf')
    for zone in zones.values():
        if zone is not None:
            dist = Point(position).distance(zone)
            min_dist = min(min_dist, dist)
            if dist < margin:
                return False, dist
    return True, min_dist


def check_realtime_adaptive(
    position: Tuple[float, float],
    velocity: Tuple[float, float],
    zones: Dict[str, Polygon],
    params: DefenseParams,
) -> Tuple[bool, float]:
    """Realtime adaptive: uses measured latency with dynamic multipliers"""
    speed = math.sqrt(velocity[0]**2 + velocity[1]**2)

    # Dynamic margin with multipliers
    base = params.base_margin * params.margin_multiplier
    vel_margin = speed * params.latency * params.velocity_factor
    accel_margin = 0.5 * params.max_acceleration * params.latency**2
    margin = base + vel_margin + accel_margin

    min_dist = float('inf')
    for zone in zones.values():
        if zone is not None:
            dist = Point(position).distance(zone)
            min_dist = min(min_dist, dist)
            if dist < margin:
                return False, dist
    return True, min_dist


# =============================================================================
# Attack Scenarios
# =============================================================================

def create_s7_scenarios() -> List[DynamicScenarioConfig]:
    """
    Create attack scenarios for S7.

    All scenarios are designed to be defensible with proper latency compensation:
    - Robot speed and zone distances allow stopping with 0.5m margin
    - Zone appearance timing gives reaction time
    """
    scenarios = []

    # Attack 1: Zone appears ahead (defensible at all latencies)
    # Robot at 1.0 m/s, zone appears at t=3s when robot is at y=8
    # Zone center y=13, radius 1.5, so edge at y=11.5
    # Distance = 3.5m, stopping distance = v²/(2a) = 0.25m → plenty of room
    scenarios.append(DynamicScenarioConfig(
        name="S7-A1: Zone Ahead",
        description="Zone appears ahead with reaction time",
        attack_type=AttackType.ZONE_APPEARANCE,
        start=(5, 5),
        goal=(5, 20),
        robot_speed=1.0,
        dynamic_zones=[
            DynamicZone(
                name="zone_ahead",
                zone_type="circle",
                center=(5, 13),
                radius=1.5,
                appear_time=3.0,
                disappear_time=30.0,
            )
        ],
    ))

    # Attack 2: Delayed zone (tests latency response)
    # Zone appears with latency, robot must react to delayed info
    scenarios.append(DynamicScenarioConfig(
        name="S7-A2: Delayed Info",
        description="Zone info arrives with network delay",
        attack_type=AttackType.ZONE_APPEARANCE,
        start=(8, 5),
        goal=(8, 20),
        robot_speed=1.2,
        dynamic_zones=[
            DynamicZone(
                name="delayed_zone",
                zone_type="circle",
                center=(8, 14),
                radius=1.5,
                appear_time=2.5,
                disappear_time=30.0,
            )
        ],
    ))

    # Attack 3: Close call (tests latency tolerance)
    # Zone appears closer, requiring quick reaction
    # At 2.0 m/s, robot at t=2s is at y=9, zone edge at y=10.5
    # Distance = 1.5m, stopping distance = 2²/(2*2) = 1m
    # Safe at low latency, but high latency causes failure
    scenarios.append(DynamicScenarioConfig(
        name="S7-A3: Close Call",
        description="Zone appears close - tests latency tolerance",
        attack_type=AttackType.ZONE_APPEARANCE,
        start=(10, 5),
        goal=(10, 20),
        robot_speed=2.0,
        dynamic_zones=[
            DynamicZone(
                name="close_zone",
                zone_type="circle",
                center=(10, 12),  # Closer
                radius=1.5,
                appear_time=2.0,  # Earlier
                disappear_time=30.0,
            )
        ],
    ))

    # Attack 4: High speed intercept (marginal case)
    # Robot at 2.5 m/s, zone at t=1.5s, robot at y=8.75
    # Zone edge at y=10, distance = 1.25m
    # Stopping distance = 2.5²/(2*2) = 1.56m > 1.25m
    # Should fail without proper latency compensation
    scenarios.append(DynamicScenarioConfig(
        name="S7-A4: Speed Trap",
        description="Fast robot, tight margins - latency critical",
        attack_type=AttackType.ZONE_APPEARANCE,
        start=(12, 5),
        goal=(12, 20),
        robot_speed=2.5,
        dynamic_zones=[
            DynamicZone(
                name="speed_trap",
                zone_type="circle",
                center=(12, 11.5),
                radius=1.5,
                appear_time=1.5,
                disappear_time=30.0,
            )
        ],
    ))

    # Attack 5: Latency exploit (designed to fail fixed_margin at high latency)
    # Zone appears when robot is close, info is delayed
    scenarios.append(DynamicScenarioConfig(
        name="S7-A5: Latency Exploit",
        description="Exploits network delay window",
        attack_type=AttackType.ZONE_APPEARANCE,
        start=(15, 5),
        goal=(15, 20),
        robot_speed=1.8,
        dynamic_zones=[
            DynamicZone(
                name="exploit_zone",
                zone_type="circle",
                center=(15, 12),
                radius=1.5,
                appear_time=2.2,
                disappear_time=30.0,
            )
        ],
    ))

    return scenarios


# =============================================================================
# Simulation Result
# =============================================================================

@dataclass
class S7Result:
    """Result of one S7 episode"""
    scenario_name: str
    defense_method: str
    latency_ms: int
    entered_zone: bool
    reached_goal: bool
    stopped_safely: bool
    steps: int
    min_distance: float
    seed: int


# =============================================================================
# S7 Runner
# =============================================================================

class S7Runner:
    """Run S7 latency attack experiments"""

    def __init__(self, dt: float = 0.1):
        self.dt = dt
        self.results: List[S7Result] = []

    def get_active_zones(
        self,
        dynamic_zones: List[DynamicZone],
        t: float,
    ) -> Dict[str, Polygon]:
        """Get active zones at time t"""
        active = {}
        for zone in dynamic_zones:
            poly = zone.get_polygon_at_time(t)
            if poly is not None:
                active[zone.name] = poly
        return active

    def get_all_zones(self, dynamic_zones: Dict[str, Polygon]) -> Dict[str, Polygon]:
        """Combine static and dynamic zones"""
        all_zones = dict(STATIC_ZONES)
        all_zones.update(dynamic_zones)
        return all_zones

    def simulate_episode(
        self,
        scenario: DynamicScenarioConfig,
        method: DefenseMethod,
        latency_ms: int,
        seed: int,
    ) -> S7Result:
        """Run single episode"""
        random.seed(seed)
        np.random.seed(seed)

        latency_sec = latency_ms / 1000.0

        # Initialize defense parameters
        base_params = DefenseParams(
            base_margin=0.5,
            latency=latency_sec,
            max_acceleration=2.0,
        )

        # Latency tracker for realtime_adaptive
        tracker = LatencyTracker(window_size=10)

        # Delayed zone information
        zone_queue = deque()
        known_zones = {}

        # Initial path
        initial_dynamic = self.get_active_zones(scenario.dynamic_zones, 0)
        all_zones = self.get_all_zones(initial_dynamic)

        path = compute_path(scenario.start, scenario.goal, all_zones, base_params.base_margin)
        if not path or len(path) <= 1:
            return S7Result(
                scenario_name=scenario.name,
                defense_method=method.value,
                latency_ms=latency_ms,
                entered_zone=False,
                reached_goal=False,
                stopped_safely=True,
                steps=0,
                min_distance=float('inf'),
                seed=seed,
            )

        position = scenario.start
        velocity = (0.0, 0.0)
        path_index = 0
        time = 0.0
        max_steps = 500
        min_distance = float('inf')

        for step in range(max_steps):
            time += self.dt

            # Ground truth zones
            actual_dynamic = self.get_active_zones(scenario.dynamic_zones, time)
            actual_zones = self.get_all_zones(actual_dynamic)

            # Queue zone updates with latency
            zone_queue.append((time + latency_sec, dict(actual_dynamic)))

            # Process arrived zone updates
            while zone_queue and zone_queue[0][0] <= time:
                _, zones = zone_queue.popleft()
                known_zones = zones
                # Record latency measurement (simulated)
                tracker.record(latency_sec)

            known_all_zones = self.get_all_zones(known_zones)

            # Calculate movement
            if path_index < len(path) - 1:
                target = path[path_index + 1]
                dx = target[0] - position[0]
                dy = target[1] - position[1]
                dist = math.sqrt(dx**2 + dy**2)

                if dist < 0.1:
                    path_index += 1
                    if path_index < len(path) - 1:
                        target = path[path_index + 1]
                        dx = target[0] - position[0]
                        dy = target[1] - position[1]
                        dist = math.sqrt(dx**2 + dy**2)

                if dist > 0.01:
                    speed = scenario.robot_speed
                    velocity = (dx / dist * speed, dy / dist * speed)
                    new_position = (
                        position[0] + velocity[0] * self.dt,
                        position[1] + velocity[1] * self.dt,
                    )
                else:
                    new_position = position
                    velocity = (0, 0)
            else:
                new_position = position
                velocity = (0, 0)

            # Apply defense method
            is_safe = True

            if method == DefenseMethod.NO_GUARD:
                pass  # No check

            elif method == DefenseMethod.FIXED_MARGIN:
                is_safe, dist = check_fixed_margin(new_position, known_all_zones, base_params)
                min_distance = min(min_distance, dist)

            elif method == DefenseMethod.VELOCITY_COMPENSATED:
                is_safe, dist = check_velocity_compensated(
                    new_position, velocity, known_all_zones, base_params
                )
                min_distance = min(min_distance, dist)

            elif method == DefenseMethod.REALTIME_ADAPTIVE:
                # Get adaptive params based on measured latency
                adaptive_params = tracker.get_adaptive_params(base_params.base_margin)

                # Emergency stop if latency too high
                if tracker.is_emergency():
                    is_safe = False
                else:
                    is_safe, dist = check_realtime_adaptive(
                        new_position, velocity, known_all_zones, adaptive_params
                    )
                    min_distance = min(min_distance, dist)

            # Stop if unsafe
            if not is_safe:
                new_position = position
                velocity = (0, 0)

            # Check actual zone violation
            in_actual_zone = False
            for zone in actual_zones.values():
                if zone is not None and zone.contains(Point(new_position)):
                    in_actual_zone = True
                    break

            if in_actual_zone:
                return S7Result(
                    scenario_name=scenario.name,
                    defense_method=method.value,
                    latency_ms=latency_ms,
                    entered_zone=True,
                    reached_goal=False,
                    stopped_safely=False,
                    steps=step + 1,
                    min_distance=min_distance,
                    seed=seed,
                )

            position = new_position

            # Check goal reached
            goal_dist = math.sqrt(
                (position[0] - scenario.goal[0])**2 +
                (position[1] - scenario.goal[1])**2
            )
            if goal_dist < 0.5:
                return S7Result(
                    scenario_name=scenario.name,
                    defense_method=method.value,
                    latency_ms=latency_ms,
                    entered_zone=False,
                    reached_goal=True,
                    stopped_safely=True,
                    steps=step + 1,
                    min_distance=min_distance,
                    seed=seed,
                )

        return S7Result(
            scenario_name=scenario.name,
            defense_method=method.value,
            latency_ms=latency_ms,
            entered_zone=False,
            reached_goal=False,
            stopped_safely=True,
            steps=max_steps,
            min_distance=min_distance,
            seed=seed,
        )

    def run_experiment(self, num_seeds: int = 30) -> Dict:
        """Run full S7 experiment"""
        scenarios = create_s7_scenarios()
        methods = list(DefenseMethod)
        latencies = [0, 100, 300, 500, 1000, 2000]

        self.results = []
        total = len(scenarios) * len(methods) * len(latencies) * num_seeds
        count = 0

        print(f"\n{'='*70}")
        print("S7: LATENCY-BASED ATTACK AND DEFENSE EXPERIMENT")
        print(f"{'='*70}")
        print(f"Scenarios: {len(scenarios)}")
        print(f"Defense Methods: {[m.value for m in methods]}")
        print(f"Latencies: {latencies} ms")
        print(f"Seeds per condition: {num_seeds}")
        print(f"Total episodes: {total}")
        print(f"{'='*70}\n")

        for scenario in scenarios:
            print(f"\n[{scenario.name}]")
            for method in methods:
                for latency in latencies:
                    for seed in range(num_seeds):
                        result = self.simulate_episode(scenario, method, latency, seed)
                        self.results.append(result)
                        count += 1

                    # Progress after each latency
                    if count % 100 == 0:
                        print(f"  Progress: {count}/{total} ({100*count/total:.1f}%)")

        return self.summarize()

    def summarize(self) -> Dict:
        """Summarize results"""
        summary = {
            "by_method_latency": {},
            "by_scenario": {},
        }

        methods = set(r.defense_method for r in self.results)
        latencies = sorted(set(r.latency_ms for r in self.results))
        scenarios = set(r.scenario_name for r in self.results)

        # By method and latency
        for method in methods:
            summary["by_method_latency"][method] = {}
            for latency in latencies:
                subset = [r for r in self.results
                         if r.defense_method == method and r.latency_ms == latency]
                if subset:
                    intrusions = sum(1 for r in subset if r.entered_zone)
                    summary["by_method_latency"][method][latency] = {
                        "asr": 100 * intrusions / len(subset),
                        "dsr": 100 * (1 - intrusions / len(subset)),
                        "total": len(subset),
                    }

        # By scenario
        for scenario in scenarios:
            summary["by_scenario"][scenario] = {}
            for method in methods:
                subset = [r for r in self.results
                         if r.scenario_name == scenario and r.defense_method == method]
                if subset:
                    intrusions = sum(1 for r in subset if r.entered_zone)
                    summary["by_scenario"][scenario][method] = {
                        "asr": 100 * intrusions / len(subset),
                    }

        return summary


def print_results(summary: Dict):
    """Print formatted results"""
    print("\n" + "="*80)
    print("S7 EXPERIMENT RESULTS: LATENCY-BASED ATTACK AND DEFENSE")
    print("="*80)

    methods_order = ["no_guard", "fixed_margin", "velocity_compensated", "realtime_adaptive"]
    latencies = [0, 100, 300, 500, 1000, 2000]

    # ASR Table
    print("\n" + "-"*80)
    print("ATTACK SUCCESS RATE (ASR) by Defense Method and Latency")
    print("-"*80)

    header = f"{'Defense Method':<24} │"
    for lat in latencies:
        header += f" {lat:>5}ms │"
    print(header)
    print("-"*24 + "─┼" + "────────┼" * len(latencies))

    for method in methods_order:
        if method not in summary["by_method_latency"]:
            continue
        row = f"{method:<24} │"
        for latency in latencies:
            data = summary["by_method_latency"][method].get(latency, {})
            asr = data.get("asr", 0)
            if asr == 0:
                row += f" \033[92m{asr:5.1f}%\033[0m │"  # Green
            elif asr < 50:
                row += f" \033[93m{asr:5.1f}%\033[0m │"  # Yellow
            else:
                row += f" \033[91m{asr:5.1f}%\033[0m │"  # Red
        print(row)

    # Analysis
    print("\n" + "-"*80)
    print("CRITICAL LATENCY THRESHOLD (max latency with 0% ASR)")
    print("-"*80)

    for method in methods_order:
        if method not in summary["by_method_latency"]:
            continue
        data = summary["by_method_latency"][method]

        # Find max latency with 0% ASR
        max_safe = None
        for lat in latencies:
            if data.get(lat, {}).get("asr", 100) == 0:
                max_safe = lat
            else:
                break

        if max_safe == 2000:
            status = f"\033[92m✓ Safe at ALL latencies (0-2000ms)\033[0m"
        elif max_safe is not None:
            status = f"\033[93m⚠ Safe up to {max_safe}ms\033[0m"
        else:
            status = f"\033[91m✗ Fails even at 0ms\033[0m"

        print(f"  {method:<24}: {status}")

    # By Scenario
    print("\n" + "-"*80)
    print("ASR BY ATTACK SCENARIO (averaged across all latencies)")
    print("-"*80)

    for scenario, methods_data in summary["by_scenario"].items():
        print(f"\n  {scenario}:")
        for method in methods_order:
            if method in methods_data:
                asr = methods_data[method]["asr"]
                print(f"    {method:<24}: {asr:5.1f}%")

    print("\n" + "="*80)


def main():
    runner = S7Runner()
    summary = runner.run_experiment(num_seeds=30)

    print_results(summary)

    # Save results
    output_path = os.path.join(os.path.dirname(__file__), '..', 's7_latency_attack_results.json')
    with open(output_path, 'w') as f:
        json.dump({
            "summary": summary,
            "results": [
                {
                    "scenario": r.scenario_name,
                    "method": r.defense_method,
                    "latency_ms": r.latency_ms,
                    "entered_zone": r.entered_zone,
                    "reached_goal": r.reached_goal,
                    "stopped_safely": r.stopped_safely,
                    "steps": r.steps,
                    "min_distance": r.min_distance,
                    "seed": r.seed,
                }
                for r in runner.results
            ]
        }, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
