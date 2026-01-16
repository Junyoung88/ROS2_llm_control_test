"""
Real-time Path Tracking Error Monitor

This module measures the execution/tracking error (e_track) by comparing:
1. Commanded path/trajectory from the planner
2. Actual robot position from localization

The tracking error is used in the safety margin formula:
    margin_α = √(χ²₂(α) · λ_max(Σ_xy)) + e_track + v_max · τ

Methods:
--------
1. Path Deviation Analysis:
   - Subscribe to planned path from Nav2
   - Compare actual position to nearest point on path
   - Compute cross-track error (perpendicular distance)

2. Goal Reaching Error:
   - Measure final position error when goal is "reached"
   - Accounts for controller tolerance settings

3. Statistical Aggregation:
   - Maintain running statistics of tracking errors
   - Compute conservative estimate (mean + k*std)
"""

import math
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional
from threading import Lock


@dataclass
class TrackingErrorStats:
    """Statistics for tracking error measurements."""
    current: float = 0.0          # Most recent cross-track error
    mean: float = 0.0             # Moving average
    std: float = 0.0              # Standard deviation
    max_val: float = 0.0          # Maximum observed error
    percentile_95: float = 0.0    # 95th percentile (conservative)
    sample_count: int = 0


@dataclass
class PathPoint:
    """A point on the planned path with metadata."""
    x: float
    y: float
    index: int
    cumulative_distance: float = 0.0


class TrackingErrorMonitor:
    """
    Real-time monitor for path tracking/execution error.

    Measures how accurately the robot follows planned paths,
    providing a dynamic e_track value for safety margin computation.

    Parameters:
        window_size: Number of samples for statistics (default: 100)
        default_error: Initial error estimate [m] (default: 0.05)
        min_path_distance: Minimum distance to consider path valid [m] (default: 0.5)
    """

    def __init__(
        self,
        window_size: int = 100,
        default_error: float = 0.05,
        min_path_distance: float = 0.5
    ):
        self._window_size = window_size
        self._default_error = default_error
        self._min_path_distance = min_path_distance

        # Current planned path
        self._current_path: List[PathPoint] = []
        self._path_total_length: float = 0.0

        # Error measurements
        self._cross_track_errors: deque = deque(maxlen=window_size)
        self._goal_errors: deque = deque(maxlen=window_size // 2)

        # Thread safety
        self._lock = Lock()

        # Statistics
        self._stats = TrackingErrorStats()
        self._stats.mean = default_error

        # Goal tracking
        self._current_goal: Optional[Tuple[float, float]] = None
        self._goal_start_time: Optional[float] = None

    def set_planned_path(self, path_points: List[Tuple[float, float]]) -> None:
        """
        Set the current planned path from Nav2.

        Args:
            path_points: List of (x, y) waypoints from the planner
        """
        if len(path_points) < 2:
            return

        with self._lock:
            self._current_path = []
            cumulative = 0.0

            for i, (x, y) in enumerate(path_points):
                if i > 0:
                    prev = path_points[i - 1]
                    dist = math.sqrt((x - prev[0])**2 + (y - prev[1])**2)
                    cumulative += dist

                self._current_path.append(PathPoint(
                    x=x,
                    y=y,
                    index=i,
                    cumulative_distance=cumulative
                ))

            self._path_total_length = cumulative

    def set_path_from_nav_path(self, nav_path) -> None:
        """
        Set path from nav_msgs/Path message.

        Args:
            nav_path: nav_msgs/Path message from Nav2
        """
        points = [(pose.pose.position.x, pose.pose.position.y)
                  for pose in nav_path.poses]
        self.set_planned_path(points)

    def set_goal(self, goal_x: float, goal_y: float) -> None:
        """
        Set the current navigation goal.

        Args:
            goal_x: Goal X coordinate
            goal_y: Goal Y coordinate
        """
        import time
        with self._lock:
            self._current_goal = (goal_x, goal_y)
            self._goal_start_time = time.time()

    def update_position(self, robot_x: float, robot_y: float) -> float:
        """
        Update with current robot position and compute tracking error.

        Args:
            robot_x: Current robot X position
            robot_y: Current robot Y position

        Returns:
            Current cross-track error (distance to planned path)
        """
        with self._lock:
            if not self._current_path or self._path_total_length < self._min_path_distance:
                return self._default_error

            # Find nearest point on path
            min_dist = float('inf')
            nearest_idx = 0

            for i, pt in enumerate(self._current_path):
                dist = math.sqrt((robot_x - pt.x)**2 + (robot_y - pt.y)**2)
                if dist < min_dist:
                    min_dist = dist
                    nearest_idx = i

            # Compute cross-track error (perpendicular distance to path segment)
            cross_track = self._compute_cross_track_error(
                robot_x, robot_y, nearest_idx
            )

            # Record measurement
            if cross_track < 2.0:  # Reject outliers > 2m
                self._cross_track_errors.append(cross_track)
                self._update_stats()

            return cross_track

    def _compute_cross_track_error(
        self,
        robot_x: float,
        robot_y: float,
        nearest_idx: int
    ) -> float:
        """
        Compute perpendicular distance to path segment.

        Uses the path segment around the nearest point for
        accurate cross-track error computation.
        """
        path = self._current_path

        # Get segment endpoints
        if nearest_idx == 0:
            p1 = path[0]
            p2 = path[1] if len(path) > 1 else path[0]
        elif nearest_idx >= len(path) - 1:
            p1 = path[-2] if len(path) > 1 else path[-1]
            p2 = path[-1]
        else:
            # Use segment with smaller error
            error_prev = self._point_to_segment_distance(
                robot_x, robot_y,
                path[nearest_idx - 1].x, path[nearest_idx - 1].y,
                path[nearest_idx].x, path[nearest_idx].y
            )
            error_next = self._point_to_segment_distance(
                robot_x, robot_y,
                path[nearest_idx].x, path[nearest_idx].y,
                path[nearest_idx + 1].x, path[nearest_idx + 1].y
            )
            return min(error_prev, error_next)

        return self._point_to_segment_distance(
            robot_x, robot_y, p1.x, p1.y, p2.x, p2.y
        )

    @staticmethod
    def _point_to_segment_distance(
        px: float, py: float,
        x1: float, y1: float,
        x2: float, y2: float
    ) -> float:
        """Calculate perpendicular distance from point to line segment."""
        dx = x2 - x1
        dy = y2 - y1

        if dx == 0 and dy == 0:
            return math.sqrt((px - x1)**2 + (py - y1)**2)

        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx*dx + dy*dy)))

        nearest_x = x1 + t * dx
        nearest_y = y1 + t * dy

        return math.sqrt((px - nearest_x)**2 + (py - nearest_y)**2)

    def record_goal_reached(self, final_x: float, final_y: float) -> float:
        """
        Record the error when a navigation goal is reached.

        Args:
            final_x: Final robot X position
            final_y: Final robot Y position

        Returns:
            Goal reaching error (distance from goal)
        """
        with self._lock:
            if self._current_goal is None:
                return 0.0

            goal_x, goal_y = self._current_goal
            error = math.sqrt((final_x - goal_x)**2 + (final_y - goal_y)**2)

            if error < 1.0:  # Reasonable goal error
                self._goal_errors.append(error)

            # Clear goal
            self._current_goal = None
            self._current_path = []

            self._update_stats()
            return error

    def _update_stats(self) -> None:
        """Update statistics from all measurements."""
        all_errors = list(self._cross_track_errors)

        # Include goal errors with higher weight
        all_errors.extend(list(self._goal_errors) * 2)

        if not all_errors:
            return

        arr = np.array(all_errors)

        self._stats.current = arr[-1] if len(arr) > 0 else 0.0
        self._stats.mean = float(np.mean(arr))
        self._stats.std = float(np.std(arr))
        self._stats.max_val = float(np.max(arr))
        self._stats.percentile_95 = float(np.percentile(arr, 95))
        self._stats.sample_count = len(arr)

    def get_tracking_error(self) -> float:
        """
        Get current tracking error estimate (e_track).

        Returns a conservative estimate using mean + 1 std.

        Returns:
            Estimated tracking error in meters
        """
        with self._lock:
            if self._stats.sample_count < 10:
                return self._default_error

            # Conservative: mean + 1 std, but cap at 95th percentile
            conservative = self._stats.mean + self._stats.std
            return min(conservative, self._stats.percentile_95 * 1.1)

    def get_tracking_error_mean(self) -> float:
        """Get mean tracking error without safety factor."""
        with self._lock:
            if self._stats.sample_count < 10:
                return self._default_error
            return self._stats.mean

    def get_stats(self) -> TrackingErrorStats:
        """Get full tracking error statistics."""
        with self._lock:
            return TrackingErrorStats(
                current=self._stats.current,
                mean=self._stats.mean,
                std=self._stats.std,
                max_val=self._stats.max_val,
                percentile_95=self._stats.percentile_95,
                sample_count=self._stats.sample_count
            )

    def clear_path(self) -> None:
        """Clear the current planned path."""
        with self._lock:
            self._current_path = []
            self._path_total_length = 0.0

    def reset(self) -> None:
        """Reset all measurements and state."""
        with self._lock:
            self._current_path = []
            self._path_total_length = 0.0
            self._cross_track_errors.clear()
            self._goal_errors.clear()
            self._stats = TrackingErrorStats()
            self._stats.mean = self._default_error
            self._current_goal = None


class TrackingErrorROSHelper:
    """
    ROS2 integration helper for tracking error measurement.

    Provides callbacks for common ROS messages.
    """

    def __init__(self, monitor: TrackingErrorMonitor):
        self._monitor = monitor

    def path_callback(self, msg) -> None:
        """Process planned path from Nav2."""
        self._monitor.set_path_from_nav_path(msg)

    def pose_callback(self, msg) -> float:
        """
        Process pose update and return current tracking error.

        Works with both PoseStamped and PoseWithCovarianceStamped.
        """
        if hasattr(msg, 'pose') and hasattr(msg.pose, 'pose'):
            # PoseWithCovarianceStamped
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
        elif hasattr(msg, 'pose'):
            # PoseStamped
            x = msg.pose.position.x
            y = msg.pose.position.y
        else:
            return self._monitor.get_tracking_error()

        return self._monitor.update_position(x, y)

    def goal_callback(self, msg) -> None:
        """Process navigation goal."""
        if hasattr(msg, 'pose'):
            x = msg.pose.position.x
            y = msg.pose.position.y
            self._monitor.set_goal(x, y)

    def result_callback(self, msg, final_pose) -> None:
        """
        Process navigation result when goal is reached.

        Args:
            msg: Navigation result message
            final_pose: Final robot pose
        """
        if hasattr(final_pose, 'pose'):
            x = final_pose.pose.position.x
            y = final_pose.pose.position.y
            self._monitor.record_goal_reached(x, y)


# =============================================================================
# STANDALONE TEST
# =============================================================================

def _test_tracking_error_monitor():
    """Test the tracking error monitor with simulated data."""
    monitor = TrackingErrorMonitor(window_size=50)

    # Create a simple straight path
    path = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)]
    monitor.set_planned_path(path)
    monitor.set_goal(5.0, 0.0)

    # Simulate robot following path with some error
    import random
    for x in np.linspace(0, 5, 50):
        # Add some cross-track error (perpendicular deviation)
        y_error = random.gauss(0, 0.03)  # ~3cm std deviation
        error = monitor.update_position(x, y_error)

    # Record goal reaching
    final_error = monitor.record_goal_reached(4.98, 0.02)

    stats = monitor.get_stats()
    print(f"Tracking Error Stats:")
    print(f"  Mean:    {stats.mean*100:.2f} cm")
    print(f"  Std:     {stats.std*100:.2f} cm")
    print(f"  95th %:  {stats.percentile_95*100:.2f} cm")
    print(f"  Max:     {stats.max_val*100:.2f} cm")
    print(f"  e_track: {monitor.get_tracking_error()*100:.2f} cm")

    assert stats.mean < 0.1, "Mean error should be < 10cm"
    print("✓ Tracking error monitor test passed")


if __name__ == "__main__":
    _test_tracking_error_monitor()
