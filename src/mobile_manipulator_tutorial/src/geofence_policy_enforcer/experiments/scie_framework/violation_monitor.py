"""
Violation Monitor - Tracks actual robot position and detects forbidden zone violations.

This module monitors the robot's real position during navigation trials
and records if/when the robot actually enters forbidden zones.
"""

import math
import subprocess
import json
import threading
import time
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ForbiddenZone:
    """Represents a rectangular forbidden zone"""
    name: str
    min_x: float
    max_x: float
    min_y: float
    max_y: float

    def contains_point(self, x: float, y: float, margin: float = 0.0) -> bool:
        """Check if a point is inside the zone (with optional margin)"""
        return (self.min_x - margin <= x <= self.max_x + margin and
                self.min_y - margin <= y <= self.max_y + margin)

    def distance_to_boundary(self, x: float, y: float) -> float:
        """Calculate minimum distance from point to zone boundary"""
        # If inside, return negative distance
        if self.contains_point(x, y):
            dx = min(x - self.min_x, self.max_x - x)
            dy = min(y - self.min_y, self.max_y - y)
            return -min(dx, dy)

        # If outside, calculate distance to nearest edge
        dx = max(self.min_x - x, 0, x - self.max_x)
        dy = max(self.min_y - y, 0, y - self.max_y)
        return math.sqrt(dx*dx + dy*dy)


@dataclass
class ViolationEvent:
    """Records a single violation event"""
    timestamp: float
    zone_name: str
    robot_x: float
    robot_y: float
    penetration_depth: float  # How far into the zone


@dataclass
class MonitoringResult:
    """Results from monitoring a trial"""
    trial_id: str
    method: str
    scenario: str

    # Violation tracking
    violated: bool = False
    violation_count: int = 0
    violation_events: List[ViolationEvent] = field(default_factory=list)
    max_penetration_depth: float = 0.0
    violated_zones: List[str] = field(default_factory=list)

    # Trajectory tracking
    min_distance_to_any_zone: float = float('inf')
    closest_zone: str = ""
    trajectory_points: List[Tuple[float, float]] = field(default_factory=list)

    # Timing
    total_time_in_violation: float = 0.0


class ViolationMonitor:
    """
    Monitors robot position and detects forbidden zone violations.

    Uses ROS2 CLI to subscribe to odometry/pose topics and checks
    against defined forbidden zones.
    """

    # Default forbidden zones (from geofence.yaml)
    DEFAULT_ZONES = [
        ForbiddenZone(
            name="pick_station_zone",
            min_x=-5.7, max_x=-3.7,
            min_y=4.6, max_y=6.6
        ),
        # Add more zones as needed
    ]

    def __init__(self, workspace_path: str, zones: List[ForbiddenZone] = None):
        self.workspace_path = Path(workspace_path)
        self.zones = zones or self.DEFAULT_ZONES

        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._current_result: Optional[MonitoringResult] = None
        self._lock = threading.Lock()

        # Position tracking
        self._last_x = 0.0
        self._last_y = 0.0
        self._last_violation_time = 0.0

    def start_monitoring(self, trial_id: str, method: str, scenario: str) -> None:
        """Start monitoring robot position for a trial"""
        with self._lock:
            self._current_result = MonitoringResult(
                trial_id=trial_id,
                method=method,
                scenario=scenario
            )
            self._monitoring = True

        self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitor_thread.start()

    def stop_monitoring(self) -> MonitoringResult:
        """Stop monitoring and return results"""
        with self._lock:
            self._monitoring = False
            result = self._current_result
            self._current_result = None

        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
            self._monitor_thread = None

        return result

    def _monitoring_loop(self):
        """Main monitoring loop - reads robot position periodically"""
        sample_interval = 0.1  # 10 Hz

        while self._monitoring:
            try:
                x, y = self._get_robot_position()
                if x is not None and y is not None:
                    self._check_position(x, y)
            except Exception as e:
                pass  # Ignore errors during monitoring

            time.sleep(sample_interval)

    def _get_robot_position(self) -> Tuple[Optional[float], Optional[float]]:
        """Get current robot position from ROS2 topic"""
        try:
            # Use ros2 topic echo with timeout
            cmd = [
                "ros2", "topic", "echo", "--once", "--no-arr",
                "/odom", "nav_msgs/msg/Odometry"
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1.0,
                cwd=str(self.workspace_path)
            )

            if result.returncode == 0:
                # Parse the output to extract position
                # This is a simplified parser - adjust based on actual output format
                output = result.stdout

                # Look for position values in YAML-like output
                lines = output.split('\n')
                x, y = None, None
                in_pose = False
                in_position = False

                for line in lines:
                    stripped = line.strip()
                    if 'pose:' in stripped:
                        in_pose = True
                    elif in_pose and 'position:' in stripped:
                        in_position = True
                    elif in_position:
                        if 'x:' in stripped:
                            x = float(stripped.split(':')[1].strip())
                        elif 'y:' in stripped:
                            y = float(stripped.split(':')[1].strip())
                            break

                return x, y
        except (subprocess.TimeoutExpired, Exception):
            pass

        return None, None

    def _check_position(self, x: float, y: float):
        """Check if current position violates any forbidden zone"""
        current_time = time.time()

        with self._lock:
            if self._current_result is None:
                return

            # Record trajectory point
            self._current_result.trajectory_points.append((x, y))

            # Check each zone
            for zone in self.zones:
                dist = zone.distance_to_boundary(x, y)

                # Track minimum distance
                if dist < self._current_result.min_distance_to_any_zone:
                    self._current_result.min_distance_to_any_zone = dist
                    self._current_result.closest_zone = zone.name

                # Check for violation (inside zone = negative distance)
                if dist < 0:
                    penetration = abs(dist)

                    # Record violation event
                    event = ViolationEvent(
                        timestamp=current_time,
                        zone_name=zone.name,
                        robot_x=x,
                        robot_y=y,
                        penetration_depth=penetration
                    )

                    self._current_result.violated = True
                    self._current_result.violation_count += 1
                    self._current_result.violation_events.append(event)

                    if penetration > self._current_result.max_penetration_depth:
                        self._current_result.max_penetration_depth = penetration

                    if zone.name not in self._current_result.violated_zones:
                        self._current_result.violated_zones.append(zone.name)

                    # Track time in violation
                    if self._last_violation_time > 0:
                        self._current_result.total_time_in_violation += (
                            current_time - self._last_violation_time
                        )
                    self._last_violation_time = current_time

            self._last_x = x
            self._last_y = y


class SimpleViolationTracker:
    """
    Simplified violation tracker that checks position against zones
    without continuous ROS2 monitoring.

    Use this when you want to check violation at specific points
    rather than continuous monitoring.
    """

    DEFAULT_ZONES = [
        ForbiddenZone(
            name="pick_station_zone",
            min_x=-5.7, max_x=-3.7,
            min_y=4.6, max_y=6.6
        ),
    ]

    def __init__(self, zones: List[ForbiddenZone] = None):
        self.zones = zones or self.DEFAULT_ZONES

    def check_violation(self, x: float, y: float) -> Tuple[bool, Optional[str], float]:
        """
        Check if a position violates any forbidden zone.

        Returns:
            (violated: bool, zone_name: str or None, min_distance: float)
        """
        min_dist = float('inf')
        closest_zone = None
        violated = False
        violated_zone = None

        for zone in self.zones:
            dist = zone.distance_to_boundary(x, y)

            if dist < min_dist:
                min_dist = dist
                closest_zone = zone.name

            if dist < 0:  # Inside zone
                violated = True
                violated_zone = zone.name

        return violated, violated_zone, min_dist

    def check_path(self, waypoints: List[Tuple[float, float]],
                   sample_resolution: float = 0.1) -> Tuple[bool, List[Tuple[float, float]]]:
        """
        Check if a path (list of waypoints) violates any zones.

        Args:
            waypoints: List of (x, y) points
            sample_resolution: Distance between sampled points on path segments

        Returns:
            (violated: bool, violation_points: list of (x, y) where violations occur)
        """
        violations = []

        for i in range(len(waypoints) - 1):
            x1, y1 = waypoints[i]
            x2, y2 = waypoints[i + 1]

            # Sample points along segment
            dist = math.sqrt((x2-x1)**2 + (y2-y1)**2)
            num_samples = max(2, int(dist / sample_resolution))

            for j in range(num_samples + 1):
                t = j / num_samples
                x = x1 + t * (x2 - x1)
                y = y1 + t * (y2 - y1)

                violated, zone, _ = self.check_violation(x, y)
                if violated:
                    violations.append((x, y))

        return len(violations) > 0, violations
