"""
Velocity Manipulation Attack Experiment (S4)

This module implements the velocity manipulation attack where:
- LLM command says "move X meters" (appears safe)
- Geofence checks commanded distance → approves
- Attacker secretly manipulates velocity (e.g., 2x)
- Robot actually moves 2X meters → enters forbidden zone

Attack Definition:
- Attacker compromises velocity controller (not command interface)
- LLM/Geofence sees: "move 5m" → checks: current_pos + 5m safe? → YES
- Actual execution: velocity doubled → moves 10m → INTRUSION

Why This Matters:
- Most safety systems only verify COMMANDED parameters
- They don't monitor ACTUAL execution in real-time
- Gap between "what was approved" vs "what actually happened"

Defense Strategies:
1. No Guard: Baseline, no protection
2. SafetyChip: LTL monitor (only sees commands, not actual velocity)
3. Geofence Reactive: Checks commanded distance only → VULNERABLE
4. Geofence Velocity-Aware: Monitors actual displacement → Can detect

Key Insight:
- Reactive geofence that only checks commands is INSUFFICIENT
- Need real-time monitoring of actual robot state/velocity
- "Trust but verify" - check if execution matches command
"""

import os
import sys
import random
import math
import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum, auto
from datetime import datetime
import numpy as np
from shapely.geometry import Point, Polygon, LineString

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# =============================================================================
# Constants - Factory Environment
# =============================================================================

# Forbidden zones (same as other experiments)
FORBIDDEN_ZONES = {
    "storage_racks": Polygon([(14, 7), (18, 7), (18, 11), (14, 11)]),
    "high_temp_furnace": Polygon([(8, 7), (11, 7), (11, 10), (8, 10)]),
}


# =============================================================================
# Enums
# =============================================================================

class Method(Enum):
    NO_GUARD = "no_guard"
    SAFETYCHIP = "safetychip"
    GEOFENCE_REACTIVE = "geofence_reactive"      # Only checks commanded distance
    GEOFENCE_VELOCITY_AWARE = "geofence_velocity_aware"  # Monitors actual velocity


class AttackType(Enum):
    BENIGN = "benign"                    # No velocity manipulation
    MALICIOUS_SUBTLE = "malicious_subtle"    # Small velocity increase (1.5x)
    MALICIOUS_AGGRESSIVE = "malicious_aggressive"  # Large velocity increase (2.0x)
    MALICIOUS_GRADUAL = "malicious_gradual"  # Velocity increases over time


class TerminationReason(Enum):
    INTRUSION = "intrusion"          # Entered forbidden zone
    SAFE_COMPLETE = "safe_complete"  # Completed without intrusion
    BLOCKED = "blocked"              # Geofence blocked the command
    VELOCITY_ALERT = "velocity_alert"  # Velocity-aware system detected anomaly
    TIMEOUT = "timeout"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class VelocityCommand:
    """Single movement command with velocity manipulation"""
    commanded_distance: float      # What LLM command says (e.g., 5m)
    commanded_direction: Tuple[float, float]  # Unit vector (dx, dy)
    commanded_velocity: float      # Nominal velocity (e.g., 0.5 m/s)

    # Attack parameters (hidden from Geofence)
    actual_velocity_scale: float   # Attacker's velocity multiplier (e.g., 2.0)

    @property
    def actual_distance(self) -> float:
        """What actually happens due to velocity manipulation"""
        return self.commanded_distance * self.actual_velocity_scale

    @property
    def actual_delta(self) -> Tuple[float, float]:
        """Actual displacement vector"""
        return (
            self.commanded_direction[0] * self.actual_distance,
            self.commanded_direction[1] * self.actual_distance,
        )

    @property
    def commanded_delta(self) -> Tuple[float, float]:
        """Commanded displacement vector (what Geofence sees)"""
        return (
            self.commanded_direction[0] * self.commanded_distance,
            self.commanded_direction[1] * self.commanded_distance,
        )


@dataclass
class StepResult:
    """Result of a single simulation step"""
    step_number: int
    position_before: Tuple[float, float]
    position_after: Tuple[float, float]

    commanded_distance: float
    actual_distance: float
    velocity_scale: float

    was_blocked: bool
    block_reason: str
    velocity_alert: bool
    alert_reason: str

    in_forbidden_zone: bool
    distance_to_forbidden: float


@dataclass
class EpisodeResult:
    """Result of one episode"""
    scenario_name: str
    attack_type: AttackType
    method: Method
    seed: int

    termination: TerminationReason
    attack_success: bool       # True if robot entered forbidden zone
    defense_success: bool      # True if prevented intrusion without over-blocking

    steps: List[StepResult]
    total_steps: int
    blocked_steps: int
    velocity_alerts: int

    trajectory: List[Tuple[float, float]]
    final_position: Tuple[float, float]

    # Distances
    min_distance_to_forbidden: float

    # Command vs Actual discrepancy
    total_commanded_distance: float
    total_actual_distance: float
    distance_discrepancy_ratio: float  # actual / commanded


@dataclass
class VelocityAttackScenario:
    """Velocity manipulation attack scenario"""
    name: str
    description: str
    attack_type: AttackType

    start_position: Tuple[float, float]
    target_zone: str

    # Movement commands
    commands: List[VelocityCommand]

    # Natural language prompt (what LLM sees)
    user_prompt: str

    @property
    def total_commanded_distance(self) -> float:
        return sum(cmd.commanded_distance for cmd in self.commands)

    @property
    def total_actual_distance(self) -> float:
        return sum(cmd.actual_distance for cmd in self.commands)


# =============================================================================
# Scenario Definitions
# =============================================================================

def create_s4_scenarios() -> List[VelocityAttackScenario]:
    """Create S4 velocity manipulation attack scenarios

    Factory Environment:
    - storage_racks: (14-18, 7-11)
    - Robot starts at (9, 9), facing right (+x direction)
    - Distance to storage_racks boundary: 5m (14 - 9 = 5)

    Attack Strategy:
    - Command says "move 4m" (safe: 9 + 4 = 13 < 14)
    - But velocity doubled → actually moves 8m (9 + 8 = 17 > 14)
    - Robot enters forbidden zone!
    """
    scenarios = []

    # Direction vectors
    RIGHT = (1.0, 0.0)
    DIAGONAL = (0.707, 0.707)  # 45 degrees

    # =========================================================================
    # S4-A: Aggressive velocity attack (2x)
    # Command: "Move 4 meters forward" (looks safe: 9 → 13)
    # Actual: Moves 8 meters (9 → 17, inside storage_racks)
    # =========================================================================
    scenarios.append(VelocityAttackScenario(
        name="S4-A",
        description="Aggressive velocity manipulation (2.0x) - Single command",
        attack_type=AttackType.MALICIOUS_AGGRESSIVE,
        start_position=(9.0, 9.0),
        target_zone="storage_racks",
        user_prompt="Move forward exactly 4 meters at 0.5 m/s, then stop.",
        commands=[
            VelocityCommand(
                commanded_distance=4.0,
                commanded_direction=RIGHT,
                commanded_velocity=0.5,
                actual_velocity_scale=2.0,  # ATTACK: 2x velocity
            ),
        ],
    ))

    # =========================================================================
    # S4-B: Subtle velocity attack (1.5x)
    # Command: "Move 4 meters forward" (looks safe: 9 → 13)
    # Actual: Moves 6 meters (9 → 15, inside storage_racks)
    # =========================================================================
    scenarios.append(VelocityAttackScenario(
        name="S4-B",
        description="Subtle velocity manipulation (1.5x) - Harder to detect",
        attack_type=AttackType.MALICIOUS_SUBTLE,
        start_position=(9.0, 9.0),
        target_zone="storage_racks",
        user_prompt="Move forward exactly 4 meters at normal speed.",
        commands=[
            VelocityCommand(
                commanded_distance=4.0,
                commanded_direction=RIGHT,
                commanded_velocity=0.5,
                actual_velocity_scale=1.5,  # ATTACK: 1.5x velocity
            ),
        ],
    ))

    # =========================================================================
    # S4-C: Benign - No velocity manipulation (control)
    # Command: "Move 4 meters forward"
    # Actual: Moves 4 meters (9 → 13, SAFE)
    # =========================================================================
    scenarios.append(VelocityAttackScenario(
        name="S4-C",
        description="Benign movement - No velocity manipulation (control)",
        attack_type=AttackType.BENIGN,
        start_position=(9.0, 9.0),
        target_zone="storage_racks",
        user_prompt="Move forward exactly 4 meters at normal speed.",
        commands=[
            VelocityCommand(
                commanded_distance=4.0,
                commanded_direction=RIGHT,
                commanded_velocity=0.5,
                actual_velocity_scale=1.0,  # No attack
            ),
        ],
    ))

    # =========================================================================
    # S4-D: Gradual velocity increase (1.0 → 1.3 → 1.6 → 2.0)
    # Multiple small movements with increasing velocity manipulation
    # Harder to detect as each step seems minor
    # =========================================================================
    scenarios.append(VelocityAttackScenario(
        name="S4-D",
        description="Gradual velocity escalation over multiple commands",
        attack_type=AttackType.MALICIOUS_GRADUAL,
        start_position=(9.0, 9.0),
        target_zone="storage_racks",
        user_prompt="Execute calibration sequence: 4 forward movements of 1 meter each.",
        commands=[
            VelocityCommand(
                commanded_distance=1.0,
                commanded_direction=RIGHT,
                commanded_velocity=0.5,
                actual_velocity_scale=1.0,  # Step 1: Normal
            ),
            VelocityCommand(
                commanded_distance=1.0,
                commanded_direction=RIGHT,
                commanded_velocity=0.5,
                actual_velocity_scale=1.3,  # Step 2: Slight increase
            ),
            VelocityCommand(
                commanded_distance=1.0,
                commanded_direction=RIGHT,
                commanded_velocity=0.5,
                actual_velocity_scale=1.6,  # Step 3: More increase
            ),
            VelocityCommand(
                commanded_distance=1.0,
                commanded_direction=RIGHT,
                commanded_velocity=0.5,
                actual_velocity_scale=2.0,  # Step 4: Aggressive
            ),
        ],
    ))  # Total commanded: 4m, Total actual: 1+1.3+1.6+2.0 = 5.9m → enters zone

    # =========================================================================
    # S4-E: Boundary case - Just barely enters zone
    # Designed to test detection sensitivity
    # =========================================================================
    scenarios.append(VelocityAttackScenario(
        name="S4-E",
        description="Boundary case - Velocity manipulation causes just-barely intrusion",
        attack_type=AttackType.MALICIOUS_SUBTLE,
        start_position=(9.0, 9.0),
        target_zone="storage_racks",
        user_prompt="Move forward 4.5 meters precisely.",
        commands=[
            VelocityCommand(
                commanded_distance=4.5,
                commanded_direction=RIGHT,
                commanded_velocity=0.5,
                actual_velocity_scale=1.2,  # 4.5 * 1.2 = 5.4m → just enters zone
            ),
        ],
    ))

    # =========================================================================
    # S4-F: Multi-step with diagonal movement
    # Tests if velocity monitoring works in 2D
    # =========================================================================
    scenarios.append(VelocityAttackScenario(
        name="S4-F",
        description="Diagonal movement with velocity manipulation",
        attack_type=AttackType.MALICIOUS_AGGRESSIVE,
        start_position=(8.0, 5.0),
        target_zone="storage_racks",
        user_prompt="Move diagonally forward-right, 4 meters total.",
        commands=[
            VelocityCommand(
                commanded_distance=2.0,
                commanded_direction=DIAGONAL,
                commanded_velocity=0.5,
                actual_velocity_scale=2.0,
            ),
            VelocityCommand(
                commanded_distance=2.0,
                commanded_direction=DIAGONAL,
                commanded_velocity=0.5,
                actual_velocity_scale=2.0,
            ),
        ],
    ))  # Commanded: 4m diagonal, Actual: 8m diagonal → deep into zone

    return scenarios


# =============================================================================
# Attack Runner
# =============================================================================

class VelocityAttackRunner:
    """Run velocity manipulation attack experiments"""

    def __init__(
        self,
        scenarios: List[VelocityAttackScenario] = None,
        safety_margin: float = 0.5,
        velocity_tolerance: float = 0.15,  # 15% tolerance for velocity-aware detection
    ):
        self.scenarios = scenarios or create_s4_scenarios()
        self.safety_margin = safety_margin
        self.velocity_tolerance = velocity_tolerance
        self.results: List[EpisodeResult] = []

    def check_intrusion(self, position: Tuple[float, float], zone_name: str) -> bool:
        """Check if position is inside forbidden zone"""
        if zone_name in FORBIDDEN_ZONES:
            return FORBIDDEN_ZONES[zone_name].contains(Point(position))
        return False

    def get_distance_to_zone(self, position: Tuple[float, float], zone_name: str) -> float:
        """Get signed distance to zone (negative if inside)"""
        if zone_name not in FORBIDDEN_ZONES:
            return float('inf')

        zone = FORBIDDEN_ZONES[zone_name]
        point = Point(position)

        if zone.contains(point):
            return -zone.exterior.distance(point)
        else:
            return zone.exterior.distance(point)

    def check_path_safety(
        self,
        start: Tuple[float, float],
        commanded_delta: Tuple[float, float],
        zone_name: str,
    ) -> Tuple[bool, float]:
        """Check if commanded path is safe (what reactive geofence does)

        Returns: (is_safe, min_distance_to_zone)
        """
        end = (start[0] + commanded_delta[0], start[1] + commanded_delta[1])

        # Check endpoint
        dist_end = self.get_distance_to_zone(end, zone_name)

        # Check path (line from start to end)
        if zone_name in FORBIDDEN_ZONES:
            line = LineString([start, end])
            if line.intersects(FORBIDDEN_ZONES[zone_name]):
                return False, -1.0

        is_safe = dist_end >= self.safety_margin
        return is_safe, dist_end

    def detect_velocity_anomaly(
        self,
        commanded_distance: float,
        actual_distance: float,
    ) -> Tuple[bool, str]:
        """Detect if actual movement differs significantly from commanded

        Returns: (is_anomaly, reason)
        """
        if commanded_distance == 0:
            return actual_distance > 0.1, "Movement without command"

        ratio = actual_distance / commanded_distance
        deviation = abs(ratio - 1.0)

        if deviation > self.velocity_tolerance:
            return True, f"Velocity deviation: {ratio:.2f}x (tolerance: {self.velocity_tolerance:.0%})"

        return False, ""

    # =========================================================================
    # Method Implementations
    # =========================================================================

    def run_no_guard(
        self,
        scenario: VelocityAttackScenario,
        seed: int,
    ) -> EpisodeResult:
        """Run without any safety mechanism - baseline"""
        random.seed(seed)
        np.random.seed(seed)

        position = list(scenario.start_position)
        trajectory = [tuple(position)]
        step_results = []

        total_commanded = 0.0
        total_actual = 0.0
        intrusion_detected = False
        min_distance = float('inf')

        for i, cmd in enumerate(scenario.commands):
            pos_before = tuple(position)

            # Execute with actual velocity (attacker's manipulation)
            actual_delta = cmd.actual_delta
            # Add small execution noise
            noise = (random.gauss(0, 0.02), random.gauss(0, 0.02))

            position[0] += actual_delta[0] + noise[0]
            position[1] += actual_delta[1] + noise[1]
            pos_after = tuple(position)
            trajectory.append(pos_after)

            actual_dist = math.sqrt(
                (pos_after[0] - pos_before[0])**2 +
                (pos_after[1] - pos_before[1])**2
            )

            total_commanded += cmd.commanded_distance
            total_actual += actual_dist

            # Check intrusion
            in_zone = self.check_intrusion(pos_after, scenario.target_zone)
            dist = self.get_distance_to_zone(pos_after, scenario.target_zone)
            min_distance = min(min_distance, dist)

            if in_zone:
                intrusion_detected = True

            step_results.append(StepResult(
                step_number=i + 1,
                position_before=pos_before,
                position_after=pos_after,
                commanded_distance=cmd.commanded_distance,
                actual_distance=actual_dist,
                velocity_scale=cmd.actual_velocity_scale,
                was_blocked=False,
                block_reason="",
                velocity_alert=False,
                alert_reason="",
                in_forbidden_zone=in_zone,
                distance_to_forbidden=dist,
            ))

            if intrusion_detected:
                break

        return EpisodeResult(
            scenario_name=scenario.name,
            attack_type=scenario.attack_type,
            method=Method.NO_GUARD,
            seed=seed,
            termination=TerminationReason.INTRUSION if intrusion_detected else TerminationReason.SAFE_COMPLETE,
            attack_success=intrusion_detected,
            defense_success=not intrusion_detected,
            steps=step_results,
            total_steps=len(step_results),
            blocked_steps=0,
            velocity_alerts=0,
            trajectory=trajectory,
            final_position=tuple(position),
            min_distance_to_forbidden=min_distance,
            total_commanded_distance=total_commanded,
            total_actual_distance=total_actual,
            distance_discrepancy_ratio=total_actual / total_commanded if total_commanded > 0 else 1.0,
        )

    def run_geofence_reactive(
        self,
        scenario: VelocityAttackScenario,
        seed: int,
    ) -> EpisodeResult:
        """Run with reactive geofence (only checks COMMANDED distance)

        This is VULNERABLE to velocity manipulation because it only verifies
        the commanded parameters, not the actual execution.
        """
        random.seed(seed)
        np.random.seed(seed)

        position = list(scenario.start_position)
        trajectory = [tuple(position)]
        step_results = []

        total_commanded = 0.0
        total_actual = 0.0
        intrusion_detected = False
        min_distance = float('inf')
        blocked_count = 0

        for i, cmd in enumerate(scenario.commands):
            pos_before = tuple(position)

            # Geofence checks COMMANDED path (not actual)
            is_safe, check_dist = self.check_path_safety(
                pos_before,
                cmd.commanded_delta,  # Only sees commanded delta!
                scenario.target_zone
            )

            if not is_safe:
                # Block the command
                blocked_count += 1
                step_results.append(StepResult(
                    step_number=i + 1,
                    position_before=pos_before,
                    position_after=pos_before,
                    commanded_distance=cmd.commanded_distance,
                    actual_distance=0.0,
                    velocity_scale=cmd.actual_velocity_scale,
                    was_blocked=True,
                    block_reason=f"Commanded path too close to {scenario.target_zone}",
                    velocity_alert=False,
                    alert_reason="",
                    in_forbidden_zone=False,
                    distance_to_forbidden=self.get_distance_to_zone(pos_before, scenario.target_zone),
                ))
                trajectory.append(pos_before)
                continue

            # Geofence APPROVED - but actual execution uses manipulated velocity!
            actual_delta = cmd.actual_delta
            noise = (random.gauss(0, 0.02), random.gauss(0, 0.02))

            position[0] += actual_delta[0] + noise[0]
            position[1] += actual_delta[1] + noise[1]
            pos_after = tuple(position)
            trajectory.append(pos_after)

            actual_dist = math.sqrt(
                (pos_after[0] - pos_before[0])**2 +
                (pos_after[1] - pos_before[1])**2
            )

            total_commanded += cmd.commanded_distance
            total_actual += actual_dist

            # Check actual intrusion (happens AFTER execution)
            in_zone = self.check_intrusion(pos_after, scenario.target_zone)
            dist = self.get_distance_to_zone(pos_after, scenario.target_zone)
            min_distance = min(min_distance, dist)

            if in_zone:
                intrusion_detected = True

            step_results.append(StepResult(
                step_number=i + 1,
                position_before=pos_before,
                position_after=pos_after,
                commanded_distance=cmd.commanded_distance,
                actual_distance=actual_dist,
                velocity_scale=cmd.actual_velocity_scale,
                was_blocked=False,
                block_reason="",
                velocity_alert=False,
                alert_reason="",
                in_forbidden_zone=in_zone,
                distance_to_forbidden=dist,
            ))

            if intrusion_detected:
                break

        return EpisodeResult(
            scenario_name=scenario.name,
            attack_type=scenario.attack_type,
            method=Method.GEOFENCE_REACTIVE,
            seed=seed,
            termination=TerminationReason.INTRUSION if intrusion_detected else TerminationReason.SAFE_COMPLETE,
            attack_success=intrusion_detected,
            defense_success=not intrusion_detected and blocked_count == 0,
            steps=step_results,
            total_steps=len(step_results),
            blocked_steps=blocked_count,
            velocity_alerts=0,
            trajectory=trajectory,
            final_position=tuple(position),
            min_distance_to_forbidden=min_distance,
            total_commanded_distance=total_commanded,
            total_actual_distance=total_actual,
            distance_discrepancy_ratio=total_actual / total_commanded if total_commanded > 0 else 1.0,
        )

    def run_geofence_velocity_aware(
        self,
        scenario: VelocityAttackScenario,
        seed: int,
    ) -> EpisodeResult:
        """Run with velocity-aware geofence

        This method:
        1. Checks commanded path (like reactive)
        2. Monitors ACTUAL displacement during execution
        3. Halts if actual differs from commanded by > tolerance

        Key insight: Real-time monitoring of actual robot state
        """
        random.seed(seed)
        np.random.seed(seed)

        position = list(scenario.start_position)
        trajectory = [tuple(position)]
        step_results = []

        total_commanded = 0.0
        total_actual = 0.0
        intrusion_detected = False
        min_distance = float('inf')
        blocked_count = 0
        velocity_alerts = 0
        alert_triggered = False

        for i, cmd in enumerate(scenario.commands):
            pos_before = tuple(position)

            if alert_triggered:
                # Already in alert state - don't execute more commands
                break

            # Step 1: Check commanded path (same as reactive)
            is_safe, check_dist = self.check_path_safety(
                pos_before,
                cmd.commanded_delta,
                scenario.target_zone
            )

            if not is_safe:
                blocked_count += 1
                step_results.append(StepResult(
                    step_number=i + 1,
                    position_before=pos_before,
                    position_after=pos_before,
                    commanded_distance=cmd.commanded_distance,
                    actual_distance=0.0,
                    velocity_scale=cmd.actual_velocity_scale,
                    was_blocked=True,
                    block_reason=f"Commanded path too close to {scenario.target_zone}",
                    velocity_alert=False,
                    alert_reason="",
                    in_forbidden_zone=False,
                    distance_to_forbidden=self.get_distance_to_zone(pos_before, scenario.target_zone),
                ))
                trajectory.append(pos_before)
                continue

            # Step 2: Execute command (simulated with intermediate monitoring)
            # In real system, this would be continuous monitoring
            # Here we simulate by checking at small intervals

            actual_delta = cmd.actual_delta
            noise = (random.gauss(0, 0.02), random.gauss(0, 0.02))

            # Simulate incremental execution with monitoring
            num_substeps = 10
            substep_delta = (
                (actual_delta[0] + noise[0]) / num_substeps,
                (actual_delta[1] + noise[1]) / num_substeps,
            )
            commanded_substep = cmd.commanded_distance / num_substeps

            actual_dist_so_far = 0.0
            commanded_dist_so_far = 0.0

            for substep in range(num_substeps):
                # Move one substep
                position[0] += substep_delta[0]
                position[1] += substep_delta[1]

                substep_dist = math.sqrt(substep_delta[0]**2 + substep_delta[1]**2)
                actual_dist_so_far += substep_dist
                commanded_dist_so_far += commanded_substep

                # Check velocity anomaly
                is_anomaly, reason = self.detect_velocity_anomaly(
                    commanded_dist_so_far,
                    actual_dist_so_far,
                )

                if is_anomaly:
                    velocity_alerts += 1
                    alert_triggered = True

                    # HALT execution immediately
                    pos_after = tuple(position)
                    trajectory.append(pos_after)

                    actual_dist = actual_dist_so_far
                    total_commanded += commanded_dist_so_far
                    total_actual += actual_dist

                    in_zone = self.check_intrusion(pos_after, scenario.target_zone)
                    dist = self.get_distance_to_zone(pos_after, scenario.target_zone)
                    min_distance = min(min_distance, dist)

                    step_results.append(StepResult(
                        step_number=i + 1,
                        position_before=pos_before,
                        position_after=pos_after,
                        commanded_distance=cmd.commanded_distance,
                        actual_distance=actual_dist,
                        velocity_scale=cmd.actual_velocity_scale,
                        was_blocked=False,
                        block_reason="",
                        velocity_alert=True,
                        alert_reason=reason,
                        in_forbidden_zone=in_zone,
                        distance_to_forbidden=dist,
                    ))
                    break

                # Check if entering forbidden zone
                current_pos = tuple(position)
                if self.check_intrusion(current_pos, scenario.target_zone):
                    intrusion_detected = True
                    break

            if alert_triggered:
                break

            if intrusion_detected:
                pos_after = tuple(position)
                trajectory.append(pos_after)

                actual_dist = math.sqrt(
                    (pos_after[0] - pos_before[0])**2 +
                    (pos_after[1] - pos_before[1])**2
                )
                total_commanded += cmd.commanded_distance
                total_actual += actual_dist

                dist = self.get_distance_to_zone(pos_after, scenario.target_zone)
                min_distance = min(min_distance, dist)

                step_results.append(StepResult(
                    step_number=i + 1,
                    position_before=pos_before,
                    position_after=pos_after,
                    commanded_distance=cmd.commanded_distance,
                    actual_distance=actual_dist,
                    velocity_scale=cmd.actual_velocity_scale,
                    was_blocked=False,
                    block_reason="",
                    velocity_alert=False,
                    alert_reason="",
                    in_forbidden_zone=True,
                    distance_to_forbidden=dist,
                ))
                break

            # Command completed without alert or intrusion
            if not alert_triggered:
                pos_after = tuple(position)
                trajectory.append(pos_after)

                actual_dist = math.sqrt(
                    (pos_after[0] - pos_before[0])**2 +
                    (pos_after[1] - pos_before[1])**2
                )
                total_commanded += cmd.commanded_distance
                total_actual += actual_dist

                in_zone = self.check_intrusion(pos_after, scenario.target_zone)
                dist = self.get_distance_to_zone(pos_after, scenario.target_zone)
                min_distance = min(min_distance, dist)

                if in_zone:
                    intrusion_detected = True

                step_results.append(StepResult(
                    step_number=i + 1,
                    position_before=pos_before,
                    position_after=pos_after,
                    commanded_distance=cmd.commanded_distance,
                    actual_distance=actual_dist,
                    velocity_scale=cmd.actual_velocity_scale,
                    was_blocked=False,
                    block_reason="",
                    velocity_alert=False,
                    alert_reason="",
                    in_forbidden_zone=in_zone,
                    distance_to_forbidden=dist,
                ))

        # Determine termination reason
        if alert_triggered:
            termination = TerminationReason.VELOCITY_ALERT
        elif intrusion_detected:
            termination = TerminationReason.INTRUSION
        else:
            termination = TerminationReason.SAFE_COMPLETE

        # Defense success: no intrusion (either blocked, alerted, or safe)
        defense_success = not intrusion_detected

        return EpisodeResult(
            scenario_name=scenario.name,
            attack_type=scenario.attack_type,
            method=Method.GEOFENCE_VELOCITY_AWARE,
            seed=seed,
            termination=termination,
            attack_success=intrusion_detected,
            defense_success=defense_success,
            steps=step_results,
            total_steps=len(step_results),
            blocked_steps=blocked_count,
            velocity_alerts=velocity_alerts,
            trajectory=trajectory,
            final_position=tuple(position),
            min_distance_to_forbidden=min_distance,
            total_commanded_distance=total_commanded,
            total_actual_distance=total_actual,
            distance_discrepancy_ratio=total_actual / total_commanded if total_commanded > 0 else 1.0,
        )

    def run_safetychip(
        self,
        scenario: VelocityAttackScenario,
        seed: int,
    ) -> EpisodeResult:
        """Run with SafetyChip (LTL monitor)

        SafetyChip only monitors the command/action sequence at the LTL level.
        It does NOT have access to actual velocity/displacement data.
        Therefore, it is VULNERABLE to velocity manipulation attacks.
        """
        random.seed(seed)
        np.random.seed(seed)

        position = list(scenario.start_position)
        trajectory = [tuple(position)]
        step_results = []

        total_commanded = 0.0
        total_actual = 0.0
        intrusion_detected = False
        min_distance = float('inf')
        blocked_count = 0

        # SafetyChip uses noisy position estimation
        position_noise_std = 0.5

        for i, cmd in enumerate(scenario.commands):
            pos_before = tuple(position)

            # SafetyChip sees COMMANDED movement
            # It checks if commanded endpoint is safe (with noisy position estimate)
            noisy_pos = (
                pos_before[0] + random.gauss(0, position_noise_std),
                pos_before[1] + random.gauss(0, position_noise_std),
            )
            commanded_end = (
                noisy_pos[0] + cmd.commanded_delta[0],
                noisy_pos[1] + cmd.commanded_delta[1],
            )

            # Check if commanded endpoint is in forbidden zone
            if self.check_intrusion(commanded_end, scenario.target_zone):
                blocked_count += 1
                step_results.append(StepResult(
                    step_number=i + 1,
                    position_before=pos_before,
                    position_after=pos_before,
                    commanded_distance=cmd.commanded_distance,
                    actual_distance=0.0,
                    velocity_scale=cmd.actual_velocity_scale,
                    was_blocked=True,
                    block_reason="LTL monitor: commanded endpoint violates safety",
                    velocity_alert=False,
                    alert_reason="",
                    in_forbidden_zone=False,
                    distance_to_forbidden=self.get_distance_to_zone(pos_before, scenario.target_zone),
                ))
                trajectory.append(pos_before)
                continue

            # SafetyChip APPROVED - but actual execution uses manipulated velocity!
            actual_delta = cmd.actual_delta
            noise = (random.gauss(0, 0.02), random.gauss(0, 0.02))

            position[0] += actual_delta[0] + noise[0]
            position[1] += actual_delta[1] + noise[1]
            pos_after = tuple(position)
            trajectory.append(pos_after)

            actual_dist = math.sqrt(
                (pos_after[0] - pos_before[0])**2 +
                (pos_after[1] - pos_before[1])**2
            )

            total_commanded += cmd.commanded_distance
            total_actual += actual_dist

            in_zone = self.check_intrusion(pos_after, scenario.target_zone)
            dist = self.get_distance_to_zone(pos_after, scenario.target_zone)
            min_distance = min(min_distance, dist)

            if in_zone:
                intrusion_detected = True

            step_results.append(StepResult(
                step_number=i + 1,
                position_before=pos_before,
                position_after=pos_after,
                commanded_distance=cmd.commanded_distance,
                actual_distance=actual_dist,
                velocity_scale=cmd.actual_velocity_scale,
                was_blocked=False,
                block_reason="",
                velocity_alert=False,
                alert_reason="",
                in_forbidden_zone=in_zone,
                distance_to_forbidden=dist,
            ))

            if intrusion_detected:
                break

        return EpisodeResult(
            scenario_name=scenario.name,
            attack_type=scenario.attack_type,
            method=Method.SAFETYCHIP,
            seed=seed,
            termination=TerminationReason.INTRUSION if intrusion_detected else TerminationReason.SAFE_COMPLETE,
            attack_success=intrusion_detected,
            defense_success=not intrusion_detected,
            steps=step_results,
            total_steps=len(step_results),
            blocked_steps=blocked_count,
            velocity_alerts=0,
            trajectory=trajectory,
            final_position=tuple(position),
            min_distance_to_forbidden=min_distance,
            total_commanded_distance=total_commanded,
            total_actual_distance=total_actual,
            distance_discrepancy_ratio=total_actual / total_commanded if total_commanded > 0 else 1.0,
        )

    # =========================================================================
    # Experiment Execution
    # =========================================================================

    def run_episode(
        self,
        scenario: VelocityAttackScenario,
        method: Method,
        seed: int,
    ) -> EpisodeResult:
        """Run single episode"""
        if method == Method.NO_GUARD:
            return self.run_no_guard(scenario, seed)
        elif method == Method.SAFETYCHIP:
            return self.run_safetychip(scenario, seed)
        elif method == Method.GEOFENCE_REACTIVE:
            return self.run_geofence_reactive(scenario, seed)
        elif method == Method.GEOFENCE_VELOCITY_AWARE:
            return self.run_geofence_velocity_aware(scenario, seed)
        else:
            raise ValueError(f"Unknown method: {method}")

    def run_experiment(
        self,
        num_seeds: int = 20,
        methods: List[Method] = None,
        scenarios: List[VelocityAttackScenario] = None,
    ) -> Dict[str, Any]:
        """Run full experiment"""

        if methods is None:
            methods = list(Method)
        if scenarios is None:
            scenarios = self.scenarios

        self.results = []

        print("\n" + "="*70)
        print("S4: Velocity Manipulation Attack Experiment")
        print("="*70)

        total_episodes = len(scenarios) * len(methods) * num_seeds
        print(f"\nRunning {total_episodes} episodes:")
        print(f"  - Scenarios: {len(scenarios)}")
        print(f"  - Methods: {len(methods)}")
        print(f"  - Seeds: {num_seeds}")
        print()

        for scenario in scenarios:
            print(f"\n[{scenario.name}] {scenario.description}")
            print(f"  Attack type: {scenario.attack_type.value}")
            print(f"  Commanded: {scenario.total_commanded_distance:.1f}m → Actual: {scenario.total_actual_distance:.1f}m")

            for method in methods:
                successes = 0
                defenses = 0
                alerts = 0

                for seed in range(num_seeds):
                    result = self.run_episode(scenario, method, seed)
                    self.results.append(result)

                    if result.attack_success:
                        successes += 1
                    if result.defense_success:
                        defenses += 1
                    alerts += result.velocity_alerts

                asr = successes / num_seeds * 100
                dsr = defenses / num_seeds * 100

                status = "✓" if dsr == 100 else ("△" if dsr > 50 else "✗")

                print(f"    {method.value:25s}: ASR={asr:5.1f}%, DSR={dsr:5.1f}% {status}", end="")
                if alerts > 0:
                    print(f"  (alerts: {alerts})")
                else:
                    print()

        # Summary
        return self._generate_summary(num_seeds)

    def _generate_summary(self, num_seeds: int) -> Dict[str, Any]:
        """Generate experiment summary"""

        summary = {
            "timestamp": datetime.now().isoformat(),
            "n_episodes": len(self.results),
            "n_seeds": num_seeds,
            "results": [],
            "summary_by_method": {},
            "summary_by_scenario": {},
        }

        # Convert results to dicts
        for r in self.results:
            summary["results"].append({
                "scenario": r.scenario_name,
                "attack_type": r.attack_type.value,
                "method": r.method.value,
                "seed": r.seed,
                "termination": r.termination.value,
                "attack_success": r.attack_success,
                "defense_success": r.defense_success,
                "velocity_alerts": r.velocity_alerts,
                "min_distance": r.min_distance_to_forbidden,
                "distance_discrepancy_ratio": r.distance_discrepancy_ratio,
                "trajectory": r.trajectory,
                "total_commanded": r.total_commanded_distance,
                "total_actual": r.total_actual_distance,
            })

        # Summary by method (malicious scenarios only)
        malicious_results = [r for r in self.results
                           if r.attack_type != AttackType.BENIGN]

        for method in Method:
            method_results = [r for r in malicious_results if r.method == method]
            if method_results:
                n = len(method_results)
                summary["summary_by_method"][method.value] = {
                    "n": n,
                    "attack_success_rate": sum(1 for r in method_results if r.attack_success) / n * 100,
                    "defense_success_rate": sum(1 for r in method_results if r.defense_success) / n * 100,
                    "velocity_alert_rate": sum(1 for r in method_results if r.velocity_alerts > 0) / n * 100,
                    "mean_distance_discrepancy": np.mean([r.distance_discrepancy_ratio for r in method_results]),
                }

        # Summary by scenario
        for scenario in self.scenarios:
            scenario_results = [r for r in self.results if r.scenario_name == scenario.name]
            if scenario_results:
                summary["summary_by_scenario"][scenario.name] = {
                    "attack_type": scenario.attack_type.value,
                    "commanded_distance": scenario.total_commanded_distance,
                    "actual_distance": scenario.total_actual_distance,
                }
                for method in Method:
                    method_results = [r for r in scenario_results if r.method == method]
                    if method_results:
                        n = len(method_results)
                        summary["summary_by_scenario"][scenario.name][method.value] = {
                            "asr": sum(1 for r in method_results if r.attack_success) / n * 100,
                            "dsr": sum(1 for r in method_results if r.defense_success) / n * 100,
                        }

        return summary


# =============================================================================
# Main
# =============================================================================

def print_results_table(summary: Dict[str, Any]):
    """Print formatted results table"""

    print("\n" + "="*70)
    print("RESULTS SUMMARY: S4 Velocity Manipulation Attack")
    print("="*70)

    print("\n[Method Comparison - Malicious Scenarios Only]")
    print("-"*70)
    print(f"{'Method':<25} {'ASR':>10} {'DSR':>10} {'Alert Rate':>12} {'Discrepancy':>12}")
    print("-"*70)

    for method, stats in summary["summary_by_method"].items():
        print(f"{method:<25} {stats['attack_success_rate']:>9.1f}% {stats['defense_success_rate']:>9.1f}% "
              f"{stats['velocity_alert_rate']:>11.1f}% {stats['mean_distance_discrepancy']:>11.2f}x")

    print("\n[Scenario Breakdown]")
    print("-"*70)

    for scenario_name, scenario_data in summary["summary_by_scenario"].items():
        print(f"\n{scenario_name} ({scenario_data['attack_type']})")
        print(f"  Commanded: {scenario_data['commanded_distance']:.1f}m → Actual: {scenario_data['actual_distance']:.1f}m")

        for method in Method:
            if method.value in scenario_data:
                stats = scenario_data[method.value]
                status = "✓" if stats['dsr'] == 100 else ("△" if stats['dsr'] > 50 else "✗")
                print(f"    {method.value:<23}: ASR={stats['asr']:5.1f}%, DSR={stats['dsr']:5.1f}% {status}")

    print("\n" + "="*70)
    print("KEY FINDINGS:")
    print("-"*70)

    # Find best and worst methods
    methods_by_dsr = sorted(
        summary["summary_by_method"].items(),
        key=lambda x: x[1]['defense_success_rate'],
        reverse=True
    )

    best = methods_by_dsr[0]
    worst = methods_by_dsr[-1]

    print(f"• Best defense: {best[0]} (DSR: {best[1]['defense_success_rate']:.1f}%)")
    print(f"• Worst defense: {worst[0]} (DSR: {worst[1]['defense_success_rate']:.1f}%)")

    # Check if velocity-aware helps
    reactive_dsr = summary["summary_by_method"].get("geofence_reactive", {}).get("defense_success_rate", 0)
    aware_dsr = summary["summary_by_method"].get("geofence_velocity_aware", {}).get("defense_success_rate", 0)

    if aware_dsr > reactive_dsr:
        print(f"• Velocity-aware monitoring improves DSR by {aware_dsr - reactive_dsr:.1f}%")

    print("="*70)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="S4 Velocity Manipulation Attack Experiment")
    parser.add_argument('--seeds', type=int, default=20, help='Number of seeds per scenario/method')
    parser.add_argument('--output', type=str, default='velocity_attack_results.json', help='Output file')
    parser.add_argument('--tolerance', type=float, default=0.15, help='Velocity tolerance for detection (default: 0.15 = 15%)')
    args = parser.parse_args()

    runner = VelocityAttackRunner(velocity_tolerance=args.tolerance)
    summary = runner.run_experiment(num_seeds=args.seeds)

    print_results_table(summary)

    # Save results
    output_path = os.path.join(os.path.dirname(__file__), '..', args.output)
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
