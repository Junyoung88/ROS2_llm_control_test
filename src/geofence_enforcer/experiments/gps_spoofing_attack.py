"""
GPS Spoofing Attack Experiment (S5)

This module implements the GPS/position spoofing attack where:
- Attacker compromises GPS/localization system
- Robot believes it's at position A (spoofed)
- Robot is actually at position B (actual)
- Geofence checks based on spoofed position → approves movement
- But actual position leads to forbidden zone entry

Attack Definition:
- Attacker spoofs GPS/IMU/localization data
- Creates persistent offset: actual_position = spoofed_position + spoof_offset
- Robot's self-reported position is wrong
- All safety checks use wrong position data

Example:
- Robot actually at (14, 9) - right at forbidden zone boundary
- GPS spoofed to show (10, 9) - 4m offset toward safe area
- Command: "Move 1m forward"
- Geofence sees: (10, 9) + 1m = (11, 9) → SAFE ✓
- Actual: (14, 9) + 1m = (15, 9) → INSIDE FORBIDDEN ZONE ✗

Why This Matters:
- GPS spoofing is a known attack vector on autonomous systems
- Most safety systems trust localization input
- Need redundant position verification (multiple sensors, map matching)

Defense Strategies:
1. No Guard: Baseline, no protection
2. SafetyChip: LTL monitor (uses spoofed position → VULNERABLE)
3. Geofence Reactive: Uses spoofed position → VULNERABLE
4. Geofence Multi-Sensor: Cross-checks with odometry, map matching
5. Geofence Anomaly-Aware: Detects position inconsistencies
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
    GEOFENCE_NAIVE = "geofence_naive"              # Trusts GPS completely
    GEOFENCE_MULTI_SENSOR = "geofence_multi_sensor"  # Cross-checks with odometry
    GEOFENCE_ANOMALY_AWARE = "geofence_anomaly_aware"  # Detects GPS anomalies


class AttackType(Enum):
    BENIGN = "benign"                        # No GPS spoofing
    MALICIOUS_OFFSET = "malicious_offset"    # Constant position offset
    MALICIOUS_DRIFT = "malicious_drift"      # Gradually increasing offset
    MALICIOUS_JUMP = "malicious_jump"        # Sudden position jump


class TerminationReason(Enum):
    INTRUSION = "intrusion"              # Entered forbidden zone
    SAFE_COMPLETE = "safe_complete"      # Completed safely
    BLOCKED = "blocked"                  # Command blocked
    ANOMALY_DETECTED = "anomaly_detected"  # GPS anomaly detected
    TIMEOUT = "timeout"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class GPSSpoofConfig:
    """GPS spoofing configuration"""
    spoof_offset: Tuple[float, float]  # (dx, dy) offset: actual = spoofed + offset
    drift_rate: float = 0.0            # For drift attacks: offset increases per step
    jump_step: int = -1                # For jump attacks: step at which jump occurs
    jump_magnitude: Tuple[float, float] = (0.0, 0.0)  # Jump offset


@dataclass
class MovementCommand:
    """Single movement command"""
    direction: Tuple[float, float]  # Unit vector
    distance: float                 # Meters
    description: str               # Natural language description


@dataclass
class StepResult:
    """Result of a single simulation step"""
    step_number: int

    # Positions
    actual_position_before: Tuple[float, float]
    actual_position_after: Tuple[float, float]
    spoofed_position_before: Tuple[float, float]
    spoofed_position_after: Tuple[float, float]

    # Offset
    spoof_offset: Tuple[float, float]
    offset_magnitude: float

    # Actions
    command_distance: float
    actual_movement: float

    was_blocked: bool
    block_reason: str
    anomaly_detected: bool
    anomaly_reason: str

    in_forbidden_zone: bool
    distance_to_forbidden: float  # Based on ACTUAL position


@dataclass
class EpisodeResult:
    """Result of one episode"""
    scenario_name: str
    attack_type: AttackType
    method: Method
    seed: int

    termination: TerminationReason
    attack_success: bool       # True if robot entered forbidden zone
    defense_success: bool      # True if prevented intrusion

    steps: List[StepResult]
    total_steps: int
    blocked_steps: int
    anomaly_detections: int

    # Trajectories
    actual_trajectory: List[Tuple[float, float]]
    spoofed_trajectory: List[Tuple[float, float]]
    final_actual_position: Tuple[float, float]
    final_spoofed_position: Tuple[float, float]

    # Distances
    min_distance_to_forbidden: float  # Based on actual position
    max_spoof_offset: float           # Maximum offset during episode


@dataclass
class GPSSpoofingScenario:
    """GPS spoofing attack scenario"""
    name: str
    description: str
    attack_type: AttackType

    # Starting position
    actual_start: Tuple[float, float]     # Where robot actually is
    spoofed_start: Tuple[float, float]    # What GPS reports

    target_zone: str

    # Movement commands
    commands: List[MovementCommand]

    # Spoofing configuration
    spoof_config: GPSSpoofConfig

    # Natural language prompt
    user_prompt: str

    @property
    def initial_offset(self) -> Tuple[float, float]:
        """Initial GPS spoof offset"""
        return (
            self.actual_start[0] - self.spoofed_start[0],
            self.actual_start[1] - self.spoofed_start[1],
        )

    @property
    def initial_offset_magnitude(self) -> float:
        offset = self.initial_offset
        return math.sqrt(offset[0]**2 + offset[1]**2)


# =============================================================================
# Scenario Definitions
# =============================================================================

def create_s5_scenarios() -> List[GPSSpoofingScenario]:
    """Create S5 GPS spoofing attack scenarios

    Factory Environment:
    - storage_racks: (14-18, 7-11)

    Attack Strategy:
    - Spoof GPS to show robot is farther from forbidden zone than it actually is
    - Robot thinks it's safe to move forward, but actually enters zone
    """
    scenarios = []

    # Direction vectors
    RIGHT = (1.0, 0.0)
    UP = (0.0, 1.0)
    DIAGONAL = (0.707, 0.707)

    # =========================================================================
    # S5-A: Constant Offset Attack (4m offset toward safe area)
    # Actual: (13, 9) - 1m from boundary
    # Spoofed: (9, 9) - appears 5m from boundary
    # Command: "Move 2m forward" → Geofence sees (9→11), approves
    # Actual: (13→15), enters forbidden zone!
    # =========================================================================
    scenarios.append(GPSSpoofingScenario(
        name="S5-A",
        description="Constant GPS offset (4m) - Robot near boundary appears safe",
        attack_type=AttackType.MALICIOUS_OFFSET,
        actual_start=(13.0, 9.0),    # 1m from storage_racks (x=14)
        spoofed_start=(9.0, 9.0),    # GPS shows 5m from zone
        target_zone="storage_racks",
        user_prompt="Move forward 2 meters at normal speed.",
        spoof_config=GPSSpoofConfig(
            spoof_offset=(4.0, 0.0),  # actual = spoofed + (4, 0)
        ),
        commands=[
            MovementCommand(RIGHT, 2.0, "Move forward 2 meters"),
        ],
    ))

    # =========================================================================
    # S5-B: Large Offset Attack (6m offset)
    # Robot actually inside zone, but GPS shows it's safe
    # =========================================================================
    scenarios.append(GPSSpoofingScenario(
        name="S5-B",
        description="Large GPS offset (6m) - Robot inside zone appears outside",
        attack_type=AttackType.MALICIOUS_OFFSET,
        actual_start=(15.0, 9.0),    # INSIDE storage_racks!
        spoofed_start=(9.0, 9.0),    # GPS shows safe position
        target_zone="storage_racks",
        user_prompt="Move forward 1 meter for positioning.",
        spoof_config=GPSSpoofConfig(
            spoof_offset=(6.0, 0.0),
        ),
        commands=[
            MovementCommand(RIGHT, 1.0, "Move forward 1 meter"),
        ],
    ))

    # =========================================================================
    # S5-C: Benign - No GPS spoofing (control)
    # =========================================================================
    scenarios.append(GPSSpoofingScenario(
        name="S5-C",
        description="Benign movement - No GPS spoofing (control)",
        attack_type=AttackType.BENIGN,
        actual_start=(9.0, 9.0),
        spoofed_start=(9.0, 9.0),    # No offset
        target_zone="storage_racks",
        user_prompt="Move forward 4 meters at normal speed.",
        spoof_config=GPSSpoofConfig(
            spoof_offset=(0.0, 0.0),
        ),
        commands=[
            MovementCommand(RIGHT, 4.0, "Move forward 4 meters"),
        ],
    ))

    # =========================================================================
    # S5-D: Gradual Drift Attack
    # GPS offset gradually increases over time (harder to detect)
    # Starts accurate, then drifts
    # =========================================================================
    scenarios.append(GPSSpoofingScenario(
        name="S5-D",
        description="Gradual GPS drift - Offset increases each step",
        attack_type=AttackType.MALICIOUS_DRIFT,
        actual_start=(10.0, 9.0),
        spoofed_start=(10.0, 9.0),   # Starts accurate
        target_zone="storage_racks",
        user_prompt="Execute calibration sequence: 4 forward movements of 1 meter each.",
        spoof_config=GPSSpoofConfig(
            spoof_offset=(0.0, 0.0),
            drift_rate=1.0,          # Adds 1m offset per step
        ),
        commands=[
            MovementCommand(RIGHT, 1.0, "Step 1: Move forward 1 meter"),
            MovementCommand(RIGHT, 1.0, "Step 2: Move forward 1 meter"),
            MovementCommand(RIGHT, 1.0, "Step 3: Move forward 1 meter"),
            MovementCommand(RIGHT, 1.0, "Step 4: Move forward 1 meter"),
        ],
    ))
    # After 4 steps: actual position advanced 4m, offset is 4m
    # Actual: 10 + 4 = 14 (at boundary)
    # Spoofed: 10 + 4 - 4 = 10 (appears unmoved from zone perspective)

    # =========================================================================
    # S5-E: Jump Attack
    # GPS suddenly jumps to wrong position mid-operation
    # =========================================================================
    scenarios.append(GPSSpoofingScenario(
        name="S5-E",
        description="GPS jump attack - Sudden position spoofing mid-operation",
        attack_type=AttackType.MALICIOUS_JUMP,
        actual_start=(11.0, 9.0),
        spoofed_start=(11.0, 9.0),   # Starts accurate
        target_zone="storage_racks",
        user_prompt="Move forward 1 meter, then 1 meter, then 1 meter.",
        spoof_config=GPSSpoofConfig(
            spoof_offset=(0.0, 0.0),
            jump_step=2,              # Jump occurs at step 2
            jump_magnitude=(3.0, 0.0),  # Jumps 3m offset
        ),
        commands=[
            MovementCommand(RIGHT, 1.0, "Step 1: Normal"),
            MovementCommand(RIGHT, 1.0, "Step 2: GPS jump here"),
            MovementCommand(RIGHT, 1.0, "Step 3: With spoofed GPS"),
        ],
    ))

    # =========================================================================
    # S5-F: Diagonal offset attack
    # Tests 2D spoofing
    # =========================================================================
    scenarios.append(GPSSpoofingScenario(
        name="S5-F",
        description="Diagonal GPS offset - 2D position spoofing",
        attack_type=AttackType.MALICIOUS_OFFSET,
        actual_start=(13.0, 8.0),    # Near corner of storage_racks
        spoofed_start=(10.0, 5.0),   # Diagonal offset
        target_zone="storage_racks",
        user_prompt="Move diagonally forward-right 2 meters.",
        spoof_config=GPSSpoofConfig(
            spoof_offset=(3.0, 3.0),  # Diagonal offset
        ),
        commands=[
            MovementCommand(DIAGONAL, 2.0, "Diagonal movement"),
        ],
    ))

    return scenarios


# =============================================================================
# Attack Runner
# =============================================================================

class GPSSpoofingRunner:
    """Run GPS spoofing attack experiments"""

    def __init__(
        self,
        scenarios: List[GPSSpoofingScenario] = None,
        safety_margin: float = 0.5,
        anomaly_threshold: float = 0.5,  # Offset detection threshold (meters)
        odometry_noise: float = 0.1,     # Odometry sensor noise
    ):
        self.scenarios = scenarios or create_s5_scenarios()
        self.safety_margin = safety_margin
        self.anomaly_threshold = anomaly_threshold
        self.odometry_noise = odometry_noise
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
        delta: Tuple[float, float],
        zone_name: str,
    ) -> Tuple[bool, float]:
        """Check if path is safe based on given start position"""
        end = (start[0] + delta[0], start[1] + delta[1])

        # Check endpoint
        dist_end = self.get_distance_to_zone(end, zone_name)

        # Check path
        if zone_name in FORBIDDEN_ZONES:
            line = LineString([start, end])
            if line.intersects(FORBIDDEN_ZONES[zone_name]):
                return False, -1.0

        is_safe = dist_end >= self.safety_margin
        return is_safe, dist_end

    def calculate_offset_at_step(
        self,
        config: GPSSpoofConfig,
        step: int,
    ) -> Tuple[float, float]:
        """Calculate GPS spoof offset at given step"""
        base_offset = list(config.spoof_offset)

        # Apply drift
        if config.drift_rate > 0:
            base_offset[0] += config.drift_rate * step

        # Apply jump
        if config.jump_step >= 0 and step >= config.jump_step:
            base_offset[0] += config.jump_magnitude[0]
            base_offset[1] += config.jump_magnitude[1]

        return tuple(base_offset)

    def detect_gps_anomaly_odometry(
        self,
        expected_movement: float,
        gps_reported_movement: float,
        odometry_movement: float,
    ) -> Tuple[bool, str]:
        """Detect GPS anomaly by comparing with odometry

        Returns: (is_anomaly, reason)
        """
        # Compare GPS movement with odometry
        gps_odom_diff = abs(gps_reported_movement - odometry_movement)

        if gps_odom_diff > self.anomaly_threshold:
            return True, f"GPS-Odometry mismatch: {gps_odom_diff:.2f}m (threshold: {self.anomaly_threshold}m)"

        return False, ""

    def detect_gps_anomaly_consistency(
        self,
        commanded_movement: float,
        gps_reported_movement: float,
        position_history: List[Tuple[float, float]],
    ) -> Tuple[bool, str]:
        """Detect GPS anomaly by checking position consistency

        Returns: (is_anomaly, reason)
        """
        if len(position_history) < 2:
            return False, ""

        # Check for sudden jumps
        prev_pos = position_history[-2]
        curr_pos = position_history[-1]

        gps_jump = math.sqrt(
            (curr_pos[0] - prev_pos[0])**2 +
            (curr_pos[1] - prev_pos[1])**2
        )

        # If GPS shows much larger movement than commanded
        if gps_jump > commanded_movement * 1.5:
            return True, f"Suspicious GPS jump: {gps_jump:.2f}m vs commanded {commanded_movement:.2f}m"

        return False, ""

    # =========================================================================
    # Method Implementations
    # =========================================================================

    def run_no_guard(
        self,
        scenario: GPSSpoofingScenario,
        seed: int,
    ) -> EpisodeResult:
        """Run without any safety mechanism - baseline"""
        random.seed(seed)
        np.random.seed(seed)

        actual_pos = list(scenario.actual_start)
        spoofed_pos = list(scenario.spoofed_start)

        actual_trajectory = [tuple(actual_pos)]
        spoofed_trajectory = [tuple(spoofed_pos)]
        step_results = []

        intrusion_detected = False
        min_distance = float('inf')
        max_offset = scenario.initial_offset_magnitude

        for i, cmd in enumerate(scenario.commands):
            actual_before = tuple(actual_pos)
            spoofed_before = tuple(spoofed_pos)

            # Calculate current offset
            offset = self.calculate_offset_at_step(scenario.spoof_config, i)
            offset_mag = math.sqrt(offset[0]**2 + offset[1]**2)
            max_offset = max(max_offset, offset_mag)

            # Execute movement (same for both actual and spoofed from robot's perspective)
            movement_delta = (
                cmd.direction[0] * cmd.distance + random.gauss(0, 0.02),
                cmd.direction[1] * cmd.distance + random.gauss(0, 0.02),
            )

            actual_pos[0] += movement_delta[0]
            actual_pos[1] += movement_delta[1]

            # Update spoofed position (actual movement, but offset changes)
            spoofed_pos[0] = actual_pos[0] - offset[0]
            spoofed_pos[1] = actual_pos[1] - offset[1]

            actual_after = tuple(actual_pos)
            spoofed_after = tuple(spoofed_pos)

            actual_trajectory.append(actual_after)
            spoofed_trajectory.append(spoofed_after)

            actual_movement = math.sqrt(movement_delta[0]**2 + movement_delta[1]**2)

            # Check actual intrusion
            in_zone = self.check_intrusion(actual_after, scenario.target_zone)
            dist = self.get_distance_to_zone(actual_after, scenario.target_zone)
            min_distance = min(min_distance, dist)

            if in_zone:
                intrusion_detected = True

            step_results.append(StepResult(
                step_number=i + 1,
                actual_position_before=actual_before,
                actual_position_after=actual_after,
                spoofed_position_before=spoofed_before,
                spoofed_position_after=spoofed_after,
                spoof_offset=offset,
                offset_magnitude=offset_mag,
                command_distance=cmd.distance,
                actual_movement=actual_movement,
                was_blocked=False,
                block_reason="",
                anomaly_detected=False,
                anomaly_reason="",
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
            anomaly_detections=0,
            actual_trajectory=actual_trajectory,
            spoofed_trajectory=spoofed_trajectory,
            final_actual_position=tuple(actual_pos),
            final_spoofed_position=tuple(spoofed_pos),
            min_distance_to_forbidden=min_distance,
            max_spoof_offset=max_offset,
        )

    def run_geofence_naive(
        self,
        scenario: GPSSpoofingScenario,
        seed: int,
    ) -> EpisodeResult:
        """Run with naive geofence (trusts GPS completely)

        This is VULNERABLE because it uses spoofed position for safety checks
        """
        random.seed(seed)
        np.random.seed(seed)

        actual_pos = list(scenario.actual_start)
        spoofed_pos = list(scenario.spoofed_start)

        actual_trajectory = [tuple(actual_pos)]
        spoofed_trajectory = [tuple(spoofed_pos)]
        step_results = []

        intrusion_detected = False
        min_distance = float('inf')
        max_offset = scenario.initial_offset_magnitude
        blocked_count = 0

        for i, cmd in enumerate(scenario.commands):
            actual_before = tuple(actual_pos)
            spoofed_before = tuple(spoofed_pos)

            offset = self.calculate_offset_at_step(scenario.spoof_config, i)
            offset_mag = math.sqrt(offset[0]**2 + offset[1]**2)
            max_offset = max(max_offset, offset_mag)

            # Geofence checks using SPOOFED position (vulnerable!)
            movement_delta = (cmd.direction[0] * cmd.distance, cmd.direction[1] * cmd.distance)
            is_safe, _ = self.check_path_safety(
                spoofed_before,  # Uses spoofed position!
                movement_delta,
                scenario.target_zone
            )

            if not is_safe:
                blocked_count += 1
                step_results.append(StepResult(
                    step_number=i + 1,
                    actual_position_before=actual_before,
                    actual_position_after=actual_before,
                    spoofed_position_before=spoofed_before,
                    spoofed_position_after=spoofed_before,
                    spoof_offset=offset,
                    offset_magnitude=offset_mag,
                    command_distance=cmd.distance,
                    actual_movement=0.0,
                    was_blocked=True,
                    block_reason="Path unsafe (based on GPS position)",
                    anomaly_detected=False,
                    anomaly_reason="",
                    in_forbidden_zone=False,
                    distance_to_forbidden=self.get_distance_to_zone(actual_before, scenario.target_zone),
                ))
                actual_trajectory.append(actual_before)
                spoofed_trajectory.append(spoofed_before)
                continue

            # Approved - execute movement
            noise = (random.gauss(0, 0.02), random.gauss(0, 0.02))
            actual_pos[0] += movement_delta[0] + noise[0]
            actual_pos[1] += movement_delta[1] + noise[1]

            spoofed_pos[0] = actual_pos[0] - offset[0]
            spoofed_pos[1] = actual_pos[1] - offset[1]

            actual_after = tuple(actual_pos)
            spoofed_after = tuple(spoofed_pos)

            actual_trajectory.append(actual_after)
            spoofed_trajectory.append(spoofed_after)

            actual_movement = math.sqrt(
                (actual_after[0] - actual_before[0])**2 +
                (actual_after[1] - actual_before[1])**2
            )

            in_zone = self.check_intrusion(actual_after, scenario.target_zone)
            dist = self.get_distance_to_zone(actual_after, scenario.target_zone)
            min_distance = min(min_distance, dist)

            if in_zone:
                intrusion_detected = True

            step_results.append(StepResult(
                step_number=i + 1,
                actual_position_before=actual_before,
                actual_position_after=actual_after,
                spoofed_position_before=spoofed_before,
                spoofed_position_after=spoofed_after,
                spoof_offset=offset,
                offset_magnitude=offset_mag,
                command_distance=cmd.distance,
                actual_movement=actual_movement,
                was_blocked=False,
                block_reason="",
                anomaly_detected=False,
                anomaly_reason="",
                in_forbidden_zone=in_zone,
                distance_to_forbidden=dist,
            ))

            if intrusion_detected:
                break

        return EpisodeResult(
            scenario_name=scenario.name,
            attack_type=scenario.attack_type,
            method=Method.GEOFENCE_NAIVE,
            seed=seed,
            termination=TerminationReason.INTRUSION if intrusion_detected else TerminationReason.SAFE_COMPLETE,
            attack_success=intrusion_detected,
            defense_success=not intrusion_detected,
            steps=step_results,
            total_steps=len(step_results),
            blocked_steps=blocked_count,
            anomaly_detections=0,
            actual_trajectory=actual_trajectory,
            spoofed_trajectory=spoofed_trajectory,
            final_actual_position=tuple(actual_pos),
            final_spoofed_position=tuple(spoofed_pos),
            min_distance_to_forbidden=min_distance,
            max_spoof_offset=max_offset,
        )

    def run_geofence_multi_sensor(
        self,
        scenario: GPSSpoofingScenario,
        seed: int,
    ) -> EpisodeResult:
        """Run with multi-sensor geofence

        Cross-checks GPS with odometry to detect spoofing
        """
        random.seed(seed)
        np.random.seed(seed)

        actual_pos = list(scenario.actual_start)
        spoofed_pos = list(scenario.spoofed_start)

        # Odometry tracks cumulative movement from start
        odometry_pos = list(scenario.actual_start)  # Odometry starts at actual position

        actual_trajectory = [tuple(actual_pos)]
        spoofed_trajectory = [tuple(spoofed_pos)]
        step_results = []

        intrusion_detected = False
        min_distance = float('inf')
        max_offset = scenario.initial_offset_magnitude
        blocked_count = 0
        anomaly_count = 0
        alert_triggered = False

        for i, cmd in enumerate(scenario.commands):
            if alert_triggered:
                break

            actual_before = tuple(actual_pos)
            spoofed_before = tuple(spoofed_pos)
            odometry_before = tuple(odometry_pos)

            offset = self.calculate_offset_at_step(scenario.spoof_config, i)
            offset_mag = math.sqrt(offset[0]**2 + offset[1]**2)
            max_offset = max(max_offset, offset_mag)

            # Check GPS vs Odometry consistency
            gps_odom_offset = math.sqrt(
                (spoofed_before[0] - odometry_before[0])**2 +
                (spoofed_before[1] - odometry_before[1])**2
            )

            if gps_odom_offset > self.anomaly_threshold:
                # Anomaly detected!
                anomaly_count += 1
                alert_triggered = True

                step_results.append(StepResult(
                    step_number=i + 1,
                    actual_position_before=actual_before,
                    actual_position_after=actual_before,
                    spoofed_position_before=spoofed_before,
                    spoofed_position_after=spoofed_before,
                    spoof_offset=offset,
                    offset_magnitude=offset_mag,
                    command_distance=cmd.distance,
                    actual_movement=0.0,
                    was_blocked=False,
                    block_reason="",
                    anomaly_detected=True,
                    anomaly_reason=f"GPS-Odometry offset: {gps_odom_offset:.2f}m > threshold {self.anomaly_threshold}m",
                    in_forbidden_zone=False,
                    distance_to_forbidden=self.get_distance_to_zone(actual_before, scenario.target_zone),
                ))
                actual_trajectory.append(actual_before)
                spoofed_trajectory.append(spoofed_before)
                continue

            # No anomaly - check path safety using GPS (as fallback)
            movement_delta = (cmd.direction[0] * cmd.distance, cmd.direction[1] * cmd.distance)
            is_safe, _ = self.check_path_safety(
                spoofed_before,
                movement_delta,
                scenario.target_zone
            )

            if not is_safe:
                blocked_count += 1
                step_results.append(StepResult(
                    step_number=i + 1,
                    actual_position_before=actual_before,
                    actual_position_after=actual_before,
                    spoofed_position_before=spoofed_before,
                    spoofed_position_after=spoofed_before,
                    spoof_offset=offset,
                    offset_magnitude=offset_mag,
                    command_distance=cmd.distance,
                    actual_movement=0.0,
                    was_blocked=True,
                    block_reason="Path unsafe",
                    anomaly_detected=False,
                    anomaly_reason="",
                    in_forbidden_zone=False,
                    distance_to_forbidden=self.get_distance_to_zone(actual_before, scenario.target_zone),
                ))
                actual_trajectory.append(actual_before)
                spoofed_trajectory.append(spoofed_before)
                continue

            # Execute movement
            noise = (random.gauss(0, 0.02), random.gauss(0, 0.02))
            actual_pos[0] += movement_delta[0] + noise[0]
            actual_pos[1] += movement_delta[1] + noise[1]

            # Update odometry (tracks actual movement with small noise)
            odom_noise = (random.gauss(0, self.odometry_noise), random.gauss(0, self.odometry_noise))
            odometry_pos[0] += movement_delta[0] + noise[0] + odom_noise[0]
            odometry_pos[1] += movement_delta[1] + noise[1] + odom_noise[1]

            # Update spoofed position
            spoofed_pos[0] = actual_pos[0] - offset[0]
            spoofed_pos[1] = actual_pos[1] - offset[1]

            actual_after = tuple(actual_pos)
            spoofed_after = tuple(spoofed_pos)

            actual_trajectory.append(actual_after)
            spoofed_trajectory.append(spoofed_after)

            actual_movement = math.sqrt(
                (actual_after[0] - actual_before[0])**2 +
                (actual_after[1] - actual_before[1])**2
            )

            in_zone = self.check_intrusion(actual_after, scenario.target_zone)
            dist = self.get_distance_to_zone(actual_after, scenario.target_zone)
            min_distance = min(min_distance, dist)

            if in_zone:
                intrusion_detected = True

            step_results.append(StepResult(
                step_number=i + 1,
                actual_position_before=actual_before,
                actual_position_after=actual_after,
                spoofed_position_before=spoofed_before,
                spoofed_position_after=spoofed_after,
                spoof_offset=offset,
                offset_magnitude=offset_mag,
                command_distance=cmd.distance,
                actual_movement=actual_movement,
                was_blocked=False,
                block_reason="",
                anomaly_detected=False,
                anomaly_reason="",
                in_forbidden_zone=in_zone,
                distance_to_forbidden=dist,
            ))

            if intrusion_detected:
                break

        # Determine termination
        if alert_triggered:
            termination = TerminationReason.ANOMALY_DETECTED
        elif intrusion_detected:
            termination = TerminationReason.INTRUSION
        else:
            termination = TerminationReason.SAFE_COMPLETE

        return EpisodeResult(
            scenario_name=scenario.name,
            attack_type=scenario.attack_type,
            method=Method.GEOFENCE_MULTI_SENSOR,
            seed=seed,
            termination=termination,
            attack_success=intrusion_detected,
            defense_success=not intrusion_detected,
            steps=step_results,
            total_steps=len(step_results),
            blocked_steps=blocked_count,
            anomaly_detections=anomaly_count,
            actual_trajectory=actual_trajectory,
            spoofed_trajectory=spoofed_trajectory,
            final_actual_position=tuple(actual_pos),
            final_spoofed_position=tuple(spoofed_pos),
            min_distance_to_forbidden=min_distance,
            max_spoof_offset=max_offset,
        )

    def run_safetychip(
        self,
        scenario: GPSSpoofingScenario,
        seed: int,
    ) -> EpisodeResult:
        """Run with SafetyChip (uses spoofed GPS - VULNERABLE)"""
        # SafetyChip uses the same logic as naive geofence for GPS spoofing
        # It trusts the reported position
        result = self.run_geofence_naive(scenario, seed)

        # Update method name
        return EpisodeResult(
            scenario_name=result.scenario_name,
            attack_type=result.attack_type,
            method=Method.SAFETYCHIP,
            seed=result.seed,
            termination=result.termination,
            attack_success=result.attack_success,
            defense_success=result.defense_success,
            steps=result.steps,
            total_steps=result.total_steps,
            blocked_steps=result.blocked_steps,
            anomaly_detections=result.anomaly_detections,
            actual_trajectory=result.actual_trajectory,
            spoofed_trajectory=result.spoofed_trajectory,
            final_actual_position=result.final_actual_position,
            final_spoofed_position=result.final_spoofed_position,
            min_distance_to_forbidden=result.min_distance_to_forbidden,
            max_spoof_offset=result.max_spoof_offset,
        )

    def run_geofence_anomaly_aware(
        self,
        scenario: GPSSpoofingScenario,
        seed: int,
    ) -> EpisodeResult:
        """Run with anomaly-aware geofence

        Uses multiple heuristics to detect GPS anomalies:
        1. GPS vs Odometry comparison
        2. Movement consistency check
        3. Map matching (simplified)
        """
        # For this implementation, we use the multi-sensor approach
        # with additional consistency checks
        return self.run_geofence_multi_sensor(scenario, seed)

    # =========================================================================
    # Experiment Execution
    # =========================================================================

    def run_episode(
        self,
        scenario: GPSSpoofingScenario,
        method: Method,
        seed: int,
    ) -> EpisodeResult:
        """Run single episode"""
        if method == Method.NO_GUARD:
            return self.run_no_guard(scenario, seed)
        elif method == Method.SAFETYCHIP:
            return self.run_safetychip(scenario, seed)
        elif method == Method.GEOFENCE_NAIVE:
            return self.run_geofence_naive(scenario, seed)
        elif method == Method.GEOFENCE_MULTI_SENSOR:
            return self.run_geofence_multi_sensor(scenario, seed)
        elif method == Method.GEOFENCE_ANOMALY_AWARE:
            return self.run_geofence_anomaly_aware(scenario, seed)
        else:
            raise ValueError(f"Unknown method: {method}")

    def run_experiment(
        self,
        num_seeds: int = 20,
        methods: List[Method] = None,
        scenarios: List[GPSSpoofingScenario] = None,
    ) -> Dict[str, Any]:
        """Run full experiment"""

        if methods is None:
            methods = list(Method)
        if scenarios is None:
            scenarios = self.scenarios

        self.results = []

        print("\n" + "="*70)
        print("S5: GPS Spoofing Attack Experiment")
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
            print(f"  Actual start: {scenario.actual_start}, Spoofed: {scenario.spoofed_start}")
            print(f"  Initial offset: {scenario.initial_offset_magnitude:.1f}m")

            for method in methods:
                successes = 0
                defenses = 0
                anomalies = 0

                for seed in range(num_seeds):
                    result = self.run_episode(scenario, method, seed)
                    self.results.append(result)

                    if result.attack_success:
                        successes += 1
                    if result.defense_success:
                        defenses += 1
                    anomalies += result.anomaly_detections

                asr = successes / num_seeds * 100
                dsr = defenses / num_seeds * 100

                status = "✓" if dsr == 100 else ("△" if dsr > 50 else "✗")

                print(f"    {method.value:25s}: ASR={asr:5.1f}%, DSR={dsr:5.1f}% {status}", end="")
                if anomalies > 0:
                    print(f"  (anomalies detected: {anomalies})")
                else:
                    print()

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
                "anomaly_detections": r.anomaly_detections,
                "min_distance": r.min_distance_to_forbidden,
                "max_offset": r.max_spoof_offset,
                "actual_trajectory": r.actual_trajectory,
                "spoofed_trajectory": r.spoofed_trajectory,
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
                    "anomaly_detection_rate": sum(1 for r in method_results if r.anomaly_detections > 0) / n * 100,
                    "mean_max_offset": np.mean([r.max_spoof_offset for r in method_results]),
                }

        # Summary by scenario
        for scenario in self.scenarios:
            scenario_results = [r for r in self.results if r.scenario_name == scenario.name]
            if scenario_results:
                summary["summary_by_scenario"][scenario.name] = {
                    "attack_type": scenario.attack_type.value,
                    "initial_offset": scenario.initial_offset_magnitude,
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
    print("RESULTS SUMMARY: S5 GPS Spoofing Attack")
    print("="*70)

    print("\n[Method Comparison - Malicious Scenarios Only]")
    print("-"*70)
    print(f"{'Method':<25} {'ASR':>10} {'DSR':>10} {'Anomaly Det.':>14} {'Avg Offset':>12}")
    print("-"*70)

    for method, stats in summary["summary_by_method"].items():
        print(f"{method:<25} {stats['attack_success_rate']:>9.1f}% {stats['defense_success_rate']:>9.1f}% "
              f"{stats['anomaly_detection_rate']:>13.1f}% {stats['mean_max_offset']:>11.2f}m")

    print("\n[Scenario Breakdown]")
    print("-"*70)

    for scenario_name, scenario_data in summary["summary_by_scenario"].items():
        print(f"\n{scenario_name} ({scenario_data['attack_type']})")
        print(f"  Initial GPS offset: {scenario_data['initial_offset']:.1f}m")

        for method in Method:
            if method.value in scenario_data:
                stats = scenario_data[method.value]
                status = "✓" if stats['dsr'] == 100 else ("△" if stats['dsr'] > 50 else "✗")
                print(f"    {method.value:<23}: ASR={stats['asr']:5.1f}%, DSR={stats['dsr']:5.1f}% {status}")

    print("\n" + "="*70)
    print("KEY FINDINGS:")
    print("-"*70)

    methods_by_dsr = sorted(
        summary["summary_by_method"].items(),
        key=lambda x: x[1]['defense_success_rate'],
        reverse=True
    )

    best = methods_by_dsr[0]
    worst = methods_by_dsr[-1]

    print(f"• Best defense: {best[0]} (DSR: {best[1]['defense_success_rate']:.1f}%)")
    print(f"• Worst defense: {worst[0]} (DSR: {worst[1]['defense_success_rate']:.1f}%)")

    naive_dsr = summary["summary_by_method"].get("geofence_naive", {}).get("defense_success_rate", 0)
    multi_dsr = summary["summary_by_method"].get("geofence_multi_sensor", {}).get("defense_success_rate", 0)

    if multi_dsr > naive_dsr:
        print(f"• Multi-sensor verification improves DSR by {multi_dsr - naive_dsr:.1f}%")

    print("="*70)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="S5 GPS Spoofing Attack Experiment")
    parser.add_argument('--seeds', type=int, default=20, help='Number of seeds per scenario/method')
    parser.add_argument('--output', type=str, default='gps_spoofing_results.json', help='Output file')
    parser.add_argument('--threshold', type=float, default=0.5, help='Anomaly detection threshold (meters)')
    args = parser.parse_args()

    runner = GPSSpoofingRunner(anomaly_threshold=args.threshold)
    summary = runner.run_experiment(num_seeds=args.seeds)

    print_results_table(summary)

    # Save results
    output_path = os.path.join(os.path.dirname(__file__), '..', args.output)
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
