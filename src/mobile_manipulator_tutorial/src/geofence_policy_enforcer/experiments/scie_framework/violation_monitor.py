"""
Violation Monitor - Tracks actual robot position and detects forbidden zone violations.

This module monitors the robot's real position during navigation trials
and records if/when the robot actually enters forbidden zones.

Uses rclpy direct subscription for reliable position tracking.
"""

import math
import threading
import time
import multiprocessing as mp
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import queue


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

    # USENIX VR_exec: Execution tracking (2026-01-21)
    odom_distance: float = 0.0             # Total distance traveled from odometry
    nav_time: float = 0.0                  # Total navigation time in seconds
    start_time: float = 0.0                # Monitoring start time
    end_time: float = 0.0                  # Monitoring end time


def _odom_monitor_process(result_queue: mp.Queue, stop_event: mp.Event,
                          trial_id: str, method: str, scenario: str,
                          zones_data: List[Dict]):
    """
    Separate process that monitors odom topic using rclpy.

    This runs in a separate process to avoid conflicts with the main
    experiment runner's ROS2 context.
    """
    import rclpy
    from rclpy.node import Node
    from nav_msgs.msg import Odometry

    # Reconstruct zones from data
    zones = [ForbiddenZone(**z) for z in zones_data]

    class OdomMonitorNode(Node):
        def __init__(self):
            super().__init__('violation_monitor_' + trial_id.replace('-', '_'))
            self.result = MonitoringResult(
                trial_id=trial_id,
                method=method,
                scenario=scenario
            )
            self.last_violation_time = 0.0
            self.positions_received = 0

            # For odom_distance tracking
            self.last_x = None
            self.last_y = None
            self.result.start_time = time.time()

            self.odom_sub = self.create_subscription(
                Odometry, '/odom', self.odom_callback, 10
            )
            self.get_logger().info(f'ViolationMonitor started for {trial_id}')

        def odom_callback(self, msg):
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            current_time = time.time()
            self.positions_received += 1

            # Record trajectory
            self.result.trajectory_points.append((x, y))

            # Accumulate odom_distance (distance traveled)
            if self.last_x is not None and self.last_y is not None:
                dx = x - self.last_x
                dy = y - self.last_y
                dist = math.sqrt(dx*dx + dy*dy)
                self.result.odom_distance += dist
            self.last_x = x
            self.last_y = y

            # Check each zone
            for zone in zones:
                dist = zone.distance_to_boundary(x, y)

                # Track minimum distance
                if dist < self.result.min_distance_to_any_zone:
                    self.result.min_distance_to_any_zone = dist
                    self.result.closest_zone = zone.name

                # Check for violation (inside zone = negative distance)
                if dist < 0:
                    penetration = abs(dist)

                    event = ViolationEvent(
                        timestamp=current_time,
                        zone_name=zone.name,
                        robot_x=x,
                        robot_y=y,
                        penetration_depth=penetration
                    )

                    self.result.violated = True
                    self.result.violation_count += 1
                    self.result.violation_events.append(event)

                    if penetration > self.result.max_penetration_depth:
                        self.result.max_penetration_depth = penetration

                    if zone.name not in self.result.violated_zones:
                        self.result.violated_zones.append(zone.name)
                        self.get_logger().warn(
                            f'VIOLATION: Robot entered {zone.name} at ({x:.2f}, {y:.2f})'
                        )

                    # Track time in violation
                    if self.last_violation_time > 0:
                        self.result.total_time_in_violation += (
                            current_time - self.last_violation_time
                        )
                    self.last_violation_time = current_time

    try:
        rclpy.init()
        node = OdomMonitorNode()

        # Spin until stop event is set
        while not stop_event.is_set():
            rclpy.spin_once(node, timeout_sec=0.1)

        # Calculate nav_time before sending result
        node.result.end_time = time.time()
        node.result.nav_time = node.result.end_time - node.result.start_time

        # Send result back
        node.get_logger().info(
            f'Monitor complete. Positions: {node.positions_received}, '
            f'Min dist: {node.result.min_distance_to_any_zone:.2f}m, '
            f'Violations: {node.result.violation_count}, '
            f'Odom dist: {node.result.odom_distance:.2f}m, '
            f'Nav time: {node.result.nav_time:.1f}s'
        )
        result_queue.put(node.result)

        node.destroy_node()
        rclpy.shutdown()

    except Exception as e:
        # Send error result
        error_result = MonitoringResult(
            trial_id=trial_id,
            method=method,
            scenario=scenario
        )
        result_queue.put(error_result)
        print(f"[ViolationMonitor] Error: {e}")


class ViolationMonitor:
    """
    Monitors robot position and detects forbidden zone violations.

    Uses a separate process with rclpy to subscribe to odometry topics
    and check against defined forbidden zones.
    """

    # Default forbidden zones (updated 2026-01-21 for home.sdf world)
    # Navigable area: x ∈ [-6.0, 5.0], y ∈ [-6.0, 5.5]
    # Zones positioned away from walls to avoid inflation overlap
    DEFAULT_ZONES = [
        # Zone A - Upper right area (100% free)
        ForbiddenZone(
            name="zone_A",
            min_x=2.0, max_x=3.0,
            min_y=2.5, max_y=3.5
        ),
        # Zone B - Lower right area (100% free)
        ForbiddenZone(
            name="zone_B",
            min_x=2.0, max_x=3.0,
            min_y=-2.5, max_y=-1.5
        ),
        # Zone C - Upper left area (100% free)
        ForbiddenZone(
            name="zone_C",
            min_x=-5.0, max_x=-4.0,
            min_y=2.0, max_y=3.0
        ),
        # Zone D - Right center area (100% free)
        ForbiddenZone(
            name="zone_D",
            min_x=3.5, max_x=4.5,
            min_y=0.0, max_y=1.0
        ),
    ]

    def __init__(self, workspace_path: str, zones: List[ForbiddenZone] = None):
        self.workspace_path = Path(workspace_path)
        self.zones = zones or self.DEFAULT_ZONES

        self._monitor_process: Optional[mp.Process] = None
        self._result_queue: Optional[mp.Queue] = None
        self._stop_event: Optional[mp.Event] = None
        self._current_result: Optional[MonitoringResult] = None

    def start_monitoring(self, trial_id: str, method: str, scenario: str) -> None:
        """Start monitoring robot position for a trial"""
        # Convert zones to serializable format
        zones_data = [
            {
                'name': z.name,
                'min_x': z.min_x, 'max_x': z.max_x,
                'min_y': z.min_y, 'max_y': z.max_y
            }
            for z in self.zones
        ]

        self._result_queue = mp.Queue()
        self._stop_event = mp.Event()

        self._monitor_process = mp.Process(
            target=_odom_monitor_process,
            args=(self._result_queue, self._stop_event,
                  trial_id, method, scenario, zones_data)
        )
        self._monitor_process.start()
        print(f"[ViolationMonitor] Started monitoring process for {trial_id}")

    def stop_monitoring(self) -> MonitoringResult:
        """Stop monitoring and return results"""
        if self._monitor_process is None:
            return MonitoringResult(trial_id="", method="", scenario="")

        # Signal process to stop
        self._stop_event.set()

        # Wait for result with timeout
        try:
            result = self._result_queue.get(timeout=5.0)
        except queue.Empty:
            print("[ViolationMonitor] Timeout waiting for results")
            result = MonitoringResult(trial_id="", method="", scenario="")

        # Clean up process
        self._monitor_process.join(timeout=2.0)
        if self._monitor_process.is_alive():
            self._monitor_process.terminate()

        self._monitor_process = None
        self._result_queue = None
        self._stop_event = None

        print(f"[ViolationMonitor] Stopped. Violations: {result.violation_count}, "
              f"Min dist: {result.min_distance_to_any_zone:.2f}m, "
              f"Odom: {result.odom_distance:.2f}m, NavTime: {result.nav_time:.1f}s")

        return result


class SimpleViolationTracker:
    """
    Simplified violation tracker that checks position against zones
    without continuous ROS2 monitoring.

    Use this when you want to check violation at specific points
    rather than continuous monitoring.
    """

    DEFAULT_ZONES = [
        ForbiddenZone(name="zone_A", min_x=2.0, max_x=3.0, min_y=2.5, max_y=3.5),
        ForbiddenZone(name="zone_B", min_x=2.0, max_x=3.0, min_y=-2.5, max_y=-1.5),
        ForbiddenZone(name="zone_C", min_x=-5.0, max_x=-4.0, min_y=2.0, max_y=3.0),
        ForbiddenZone(name="zone_D", min_x=3.5, max_x=4.5, min_y=0.0, max_y=1.0),
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
