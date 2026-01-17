"""
Metrics Collector Module

Collects all metrics during experiment execution:
- Safety metrics from goal_gate audit logs
- Performance metrics from ROS2 topics
- Method-specific metrics from safety system outputs

Integrates with ROS2 for real-time data collection.
"""

import time
import json
import threading
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Any
from pathlib import Path
import subprocess


@dataclass
class MetricSample:
    """A single metric sample with timestamp"""
    name: str
    value: Any
    timestamp: float
    unit: str = ""


class MetricsCollector:
    """
    Collects metrics from various sources during experiment trials.

    Sources:
    - Audit log file (goal_gate decisions)
    - ROS2 topics (navigation feedback, robot state)
    - System metrics (CPU, memory)
    - Timing measurements
    """

    def __init__(self, audit_log_path: str = "/tmp/geofence_audit.jsonl"):
        """
        Initialize metrics collector.

        Args:
            audit_log_path: Path to goal_gate audit log file
        """
        self.audit_log_path = Path(audit_log_path)

        # Timing
        self._trial_start_time: float = 0.0
        self._decision_start_time: float = 0.0

        # Collected metrics
        self._samples: List[MetricSample] = []

        # Last known state
        self._last_audit_entry: Dict = {}
        self._audit_file_position: int = 0

    def reset(self):
        """Reset collector for new trial"""
        self._samples = []
        self._trial_start_time = 0.0
        self._decision_start_time = 0.0
        self._last_audit_entry = {}
        self._audit_file_position = 0

        # Clear audit log for clean measurement
        if self.audit_log_path.exists():
            self._audit_file_position = self.audit_log_path.stat().st_size

    def start_trial(self):
        """Mark trial start time"""
        self._trial_start_time = time.time()
        self._record("trial_start", self._trial_start_time, "timestamp")

    def end_trial(self):
        """Mark trial end time and compute duration"""
        end_time = time.time()
        duration = end_time - self._trial_start_time
        self._record("trial_end", end_time, "timestamp")
        self._record("trial_duration_s", duration, "s")
        return duration

    def start_decision_timer(self):
        """Start timing safety decision"""
        self._decision_start_time = time.time()

    def end_decision_timer(self) -> float:
        """End timing and return decision latency in ms"""
        if self._decision_start_time == 0:
            return 0.0
        latency_ms = (time.time() - self._decision_start_time) * 1000
        self._record("decision_latency_ms", latency_ms, "ms")
        return latency_ms

    def _record(self, name: str, value: Any, unit: str = ""):
        """Record a metric sample"""
        self._samples.append(MetricSample(
            name=name,
            value=value,
            timestamp=time.time(),
            unit=unit
        ))

    def read_latest_audit_entry(self) -> Optional[Dict]:
        """
        Read the latest entry from the audit log.

        Returns:
            Latest audit log entry as dict, or None if not available
        """
        if not self.audit_log_path.exists():
            return None

        try:
            with open(self.audit_log_path, 'r') as f:
                # Seek to last known position
                f.seek(self._audit_file_position)
                lines = f.readlines()

                if lines:
                    # Update position
                    self._audit_file_position = f.tell()

                    # Parse last non-empty line
                    for line in reversed(lines):
                        line = line.strip()
                        if line:
                            entry = json.loads(line)
                            self._last_audit_entry = entry
                            return entry

        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Failed to read audit log: {e}")

        return self._last_audit_entry if self._last_audit_entry else None

    def collect_from_audit(self) -> Dict:
        """
        Collect all metrics from the latest audit entry.

        Returns:
            Dict with all extracted metrics
        """
        entry = self.read_latest_audit_entry()

        if not entry:
            return {}

        metrics = {
            "decision": entry.get("decision", "unknown"),
            "reason": entry.get("reason", ""),
            "safety_method": entry.get("safety_method", ""),
            "min_distance_to_forbidden": entry.get("min_distance_to_forbidden", float('inf')),
            "violated_zone": entry.get("violated_zone"),
            "original_x": entry.get("original_x", 0.0),
            "original_y": entry.get("original_y", 0.0),
            "projected_x": entry.get("projected_x"),
            "projected_y": entry.get("projected_y"),
        }

        # Extract extra_info if present (Proper version features)
        extra_info = entry.get("extra_info", {})
        if extra_info:
            metrics["reprompt_text"] = extra_info.get("reprompt_text", "")
            metrics["suggestions"] = extra_info.get("suggestions", [])
            metrics["reprompt_count"] = extra_info.get("reprompt_count", 0)
            metrics["automaton_state"] = extra_info.get("automaton_state", "")
            metrics["noisy_perception"] = extra_info.get("noisy_perception", False)
            metrics["proposition_changes"] = extra_info.get("proposition_changes", {})

        # Record all metrics
        for key, value in metrics.items():
            if value is not None:
                self._record(key, value)

        return metrics

    def get_path_length(self, path_topic: str = "/plan") -> float:
        """
        Get the planned path length from Nav2.

        Note: This requires the nav stack to be running.
        """
        # Placeholder - would need ROS2 integration
        return 0.0

    def get_robot_position(self) -> Optional[tuple]:
        """
        Get current robot position from TF.

        Note: This requires the nav stack to be running.
        """
        # Placeholder - would need ROS2 integration
        return None

    def measure_goal_distance(self, goal_x: float, goal_y: float) -> float:
        """
        Measure distance from current robot position to goal.
        """
        position = self.get_robot_position()
        if position:
            dx = goal_x - position[0]
            dy = goal_y - position[1]
            return (dx**2 + dy**2)**0.5
        return 0.0

    def collect_system_metrics(self) -> Dict:
        """
        Collect system performance metrics (CPU, memory).
        """
        try:
            import psutil
            return {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "memory_used_gb": psutil.virtual_memory().used / (1024**3)
            }
        except ImportError:
            return {}

    def get_all_samples(self) -> List[MetricSample]:
        """Get all collected samples"""
        return self._samples.copy()

    def get_metric(self, name: str) -> Optional[Any]:
        """Get the latest value of a specific metric"""
        for sample in reversed(self._samples):
            if sample.name == name:
                return sample.value
        return None

    def to_dict(self) -> Dict:
        """Convert all collected metrics to dictionary"""
        result = {}
        for sample in self._samples:
            # Use latest value for each metric
            result[sample.name] = sample.value
        return result


class ROS2MetricsCollector(MetricsCollector):
    """
    Extended metrics collector with ROS2 integration.

    Requires rclpy to be initialized.
    """

    def __init__(self, audit_log_path: str = "/tmp/geofence_audit.jsonl"):
        super().__init__(audit_log_path)

        self._ros_initialized = False
        self._node = None
        self._tf_buffer = None

    def init_ros(self):
        """Initialize ROS2 node for metrics collection"""
        try:
            import rclpy
            from rclpy.node import Node
            from tf2_ros import Buffer, TransformListener

            if not rclpy.ok():
                rclpy.init()

            self._node = Node('metrics_collector')
            self._tf_buffer = Buffer()
            self._tf_listener = TransformListener(self._tf_buffer, self._node)
            self._ros_initialized = True

        except ImportError:
            print("Warning: rclpy not available, ROS2 metrics disabled")

    def get_robot_position(self) -> Optional[tuple]:
        """Get robot position from TF"""
        if not self._ros_initialized:
            return None

        try:
            from rclpy.time import Time

            transform = self._tf_buffer.lookup_transform(
                'map',
                'base_link',
                Time(),
                timeout=Duration(seconds=1.0)
            )

            return (
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z
            )
        except Exception:
            return None

    def subscribe_to_path(self, callback: Callable):
        """Subscribe to Nav2 path topic"""
        if not self._ros_initialized:
            return

        from nav_msgs.msg import Path

        self._node.create_subscription(
            Path,
            '/plan',
            callback,
            10
        )

    def cleanup(self):
        """Cleanup ROS2 resources"""
        if self._node:
            self._node.destroy_node()


class TimingContext:
    """Context manager for timing code blocks"""

    def __init__(self, collector: MetricsCollector, metric_name: str):
        self.collector = collector
        self.metric_name = metric_name
        self.start_time = 0.0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        self.collector._record(self.metric_name, duration * 1000, "ms")
        return False

    @property
    def elapsed_ms(self) -> float:
        return (time.time() - self.start_time) * 1000


def compute_path_length(waypoints: List[tuple]) -> float:
    """
    Compute total path length from list of waypoints.

    Args:
        waypoints: List of (x, y) tuples

    Returns:
        Total path length in meters
    """
    if len(waypoints) < 2:
        return 0.0

    total = 0.0
    for i in range(1, len(waypoints)):
        dx = waypoints[i][0] - waypoints[i-1][0]
        dy = waypoints[i][1] - waypoints[i-1][1]
        total += (dx**2 + dy**2)**0.5

    return total


def compute_min_distance_to_zones(point: tuple, zones: List[Dict]) -> float:
    """
    Compute minimum distance from point to any zone boundary.

    Args:
        point: (x, y) tuple
        zones: List of zone dicts with 'vertices' key

    Returns:
        Minimum distance to any zone boundary
    """
    min_dist = float('inf')

    for zone in zones:
        vertices = zone.get('vertices', [])
        if not vertices:
            continue

        # Check distance to each edge
        for i in range(len(vertices)):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % len(vertices)]

            dist = point_to_segment_distance(point, p1, p2)
            min_dist = min(min_dist, dist)

    return min_dist


def point_to_segment_distance(point: tuple, seg_start: tuple, seg_end: tuple) -> float:
    """
    Compute distance from point to line segment.

    Args:
        point: (x, y) tuple
        seg_start: Segment start (x, y)
        seg_end: Segment end (x, y)

    Returns:
        Distance from point to segment
    """
    px, py = point
    x1, y1 = seg_start
    x2, y2 = seg_end

    dx = x2 - x1
    dy = y2 - y1

    if dx == 0 and dy == 0:
        # Segment is a point
        return ((px - x1)**2 + (py - y1)**2)**0.5

    # Parameter t for closest point on line
    t = max(0, min(1, ((px - x1)*dx + (py - y1)*dy) / (dx*dx + dy*dy)))

    # Closest point
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy

    return ((px - closest_x)**2 + (py - closest_y)**2)**0.5
