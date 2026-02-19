#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Geofence Core - Geometry operations and policy logic.

Implements polygon-based geofencing with uncertainty-aware margins.
Uses shapely if available, otherwise falls back to manual implementations.
"""

import math
import yaml
import numpy as np
from collections import deque
from threading import Lock
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum

# Try to import shapely; fall back to manual implementation
try:
    from shapely.geometry import Point, Polygon, LineString
    from shapely.ops import nearest_points
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False


class ZoneType(Enum):
    """Types of geofence zones."""
    FORBIDDEN = "forbidden"      # Hard no-go zone
    BUFFER = "buffer"            # Soft keep-away zone (warning)
    SAFE = "safe"                # Explicitly safe zone


class PolicyAction(Enum):
    """Actions the policy can take."""
    ALLOW = "allow"
    REJECT = "reject"
    PROJECT = "project"          # Project goal to nearest safe point
    WARN = "warn"


@dataclass
class GeofenceZone:
    """Represents a single geofence zone."""
    name: str
    zone_type: ZoneType
    vertices: List[Tuple[float, float]]
    priority: int = 0
    _polygon: object = field(default=None, repr=False)

    def __post_init__(self):
        if SHAPELY_AVAILABLE:
            self._polygon = Polygon(self.vertices)


class VelocityMonitor:
    """Compute velocity statistics from observed odometry for dynamic v_max estimation."""

    def __init__(self, window_size: int = 100, percentile: int = 95):
        self._velocities: deque = deque(maxlen=window_size)
        self._percentile = percentile
        self._lock = Lock()

    def record_velocity(self, velocity: float) -> None:
        """Record observed velocity (filters out near-zero values)."""
        with self._lock:
            if velocity >= 0.001:  # Filter out stationary state
                self._velocities.append(velocity)

    def get_v_max_estimate(self, default: float = 1.0) -> float:
        """Get the estimated v_max based on percentile of observed velocities."""
        with self._lock:
            if len(self._velocities) < 20:
                return default
            return float(np.percentile(np.array(self._velocities), self._percentile))

    def get_stats(self) -> dict:
        """Get velocity statistics."""
        with self._lock:
            if len(self._velocities) < 5:
                return {'sample_count': 0}
            arr = np.array(self._velocities)
            return {
                'sample_count': len(arr),
                'v_max_95': float(np.percentile(arr, 95)),
                'mean': float(np.mean(arr)),
                'max_observed': float(np.max(arr))
            }

    def reset(self) -> None:
        """Reset velocity history."""
        with self._lock:
            self._velocities.clear()


class LatencyMonitor:
    """
    Measure system latency (τ) from cmd_vel publication to odom response.

    Latency is measured as the time between:
    1. Commanding a velocity change (cmd_vel publication)
    2. Observing the corresponding velocity change in odometry

    The 95th percentile of measured latencies is used as τ for margin calculation.
    """

    def __init__(self, window_size: int = 50, percentile: int = 95,
                 velocity_threshold: float = 0.05):
        """
        Args:
            window_size: Number of latency samples to keep
            percentile: Percentile to use for τ estimate (default 95)
            velocity_threshold: Minimum velocity change to consider as response (m/s)
        """
        self._latencies: deque = deque(maxlen=window_size)
        self._percentile = percentile
        self._velocity_threshold = velocity_threshold
        self._lock = Lock()

        # State for latency measurement
        self._cmd_vel_time: Optional[float] = None
        self._cmd_vel_magnitude: float = 0.0
        self._last_odom_velocity: float = 0.0
        self._waiting_for_response: bool = False

    def record_cmd_vel(self, timestamp: float, linear_x: float, linear_y: float = 0.0) -> None:
        """
        Record cmd_vel publication for latency measurement.

        Args:
            timestamp: Time of cmd_vel publication (seconds)
            linear_x, linear_y: Commanded velocities
        """
        cmd_magnitude = math.sqrt(linear_x**2 + linear_y**2)

        with self._lock:
            # Only start measurement if there's a significant velocity change
            velocity_change = abs(cmd_magnitude - self._last_odom_velocity)
            if velocity_change > self._velocity_threshold:
                self._cmd_vel_time = timestamp
                self._cmd_vel_magnitude = cmd_magnitude
                self._waiting_for_response = True

    def record_odom_velocity(self, timestamp: float, velocity: float) -> Optional[float]:
        """
        Record observed velocity from odometry.

        Args:
            timestamp: Time of odometry message (seconds)
            velocity: Observed velocity magnitude

        Returns:
            Measured latency if a response was detected, None otherwise
        """
        with self._lock:
            latency = None

            if self._waiting_for_response and self._cmd_vel_time is not None:
                # Check if velocity is responding to command
                velocity_change = abs(velocity - self._last_odom_velocity)
                expected_direction = (self._cmd_vel_magnitude > self._last_odom_velocity)
                actual_direction = (velocity > self._last_odom_velocity)

                if velocity_change > self._velocity_threshold * 0.5:
                    # Response detected
                    latency = timestamp - self._cmd_vel_time
                    if 0.001 < latency < 2.0:  # Sanity check: 1ms to 2s
                        self._latencies.append(latency)
                    self._waiting_for_response = False
                    self._cmd_vel_time = None

            self._last_odom_velocity = velocity
            return latency

    def get_tau_estimate(self, default: float = 0.1) -> float:
        """Get the estimated latency τ based on percentile of measured latencies."""
        with self._lock:
            if len(self._latencies) < 10:
                return default
            return float(np.percentile(np.array(self._latencies), self._percentile))

    def get_stats(self) -> dict:
        """Get latency statistics."""
        with self._lock:
            if len(self._latencies) < 5:
                return {'sample_count': 0}
            arr = np.array(self._latencies)
            return {
                'sample_count': len(arr),
                'tau_95': float(np.percentile(arr, 95)),
                'tau_50': float(np.percentile(arr, 50)),
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'min': float(np.min(arr)),
                'max': float(np.max(arr))
            }

    def reset(self) -> None:
        """Reset latency history."""
        with self._lock:
            self._latencies.clear()
            self._cmd_vel_time = None
            self._waiting_for_response = False


class TrackingErrorMonitor:
    """
    Measure tracking error (e_track) from planned path vs actual position.

    Tracking error is the perpendicular distance between:
    1. The robot's actual position (from odometry/localization)
    2. The nearest point on the planned path

    The 95th percentile of measured errors is used as e_track for margin calculation.
    """

    def __init__(self, window_size: int = 200, percentile: int = 95):
        """
        Args:
            window_size: Number of error samples to keep
            percentile: Percentile to use for e_track estimate (default 95)
        """
        self._errors: deque = deque(maxlen=window_size)
        self._percentile = percentile
        self._lock = Lock()

        # Current planned path (list of (x, y) tuples)
        self._current_path: List[Tuple[float, float]] = []
        self._path_valid: bool = False

    def update_path(self, path_points: List[Tuple[float, float]]) -> None:
        """
        Update the current planned path.

        Args:
            path_points: List of (x, y) waypoints from the planner
        """
        with self._lock:
            if len(path_points) >= 2:
                self._current_path = path_points.copy()
                self._path_valid = True
            else:
                self._path_valid = False

    def record_position(self, x: float, y: float) -> Optional[float]:
        """
        Record robot position and compute tracking error.

        Args:
            x, y: Robot's current position

        Returns:
            Computed tracking error if path is valid, None otherwise
        """
        with self._lock:
            if not self._path_valid or len(self._current_path) < 2:
                return None

            # Find minimum distance to any path segment
            min_dist = float('inf')
            for i in range(len(self._current_path) - 1):
                p1 = self._current_path[i]
                p2 = self._current_path[i + 1]
                dist = self._point_to_segment_distance((x, y), p1, p2)
                if dist < min_dist:
                    min_dist = dist

            # Only record if we're reasonably close to the path (< 0.3m)
            # to filter out cases where robot is far from path start/end
            # or during path transitions. 0.3m is a reasonable upper bound
            # for tracking error during normal navigation.
            if min_dist < 0.3:
                self._errors.append(min_dist)

            return min_dist

    def _point_to_segment_distance(self, point: Tuple[float, float],
                                    seg_start: Tuple[float, float],
                                    seg_end: Tuple[float, float]) -> float:
        """Calculate perpendicular distance from point to line segment."""
        px, py = point
        x1, y1 = seg_start
        x2, y2 = seg_end

        dx = x2 - x1
        dy = y2 - y1

        if dx == 0 and dy == 0:
            return math.sqrt((px - x1)**2 + (py - y1)**2)

        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx*dx + dy*dy)))

        nearest_x = x1 + t * dx
        nearest_y = y1 + t * dy

        return math.sqrt((px - nearest_x)**2 + (py - nearest_y)**2)

    def get_e_track_estimate(self, default: float = 0.05, max_value: float = 0.2) -> float:
        """Get the estimated tracking error based on percentile of measured errors.

        Args:
            default: Default value if not enough samples (default 0.05m)
            max_value: Maximum allowed e_track value (default 0.2m)

        Returns:
            Capped tracking error estimate
        """
        with self._lock:
            if len(self._errors) < 20:
                return default
            estimate = float(np.percentile(np.array(self._errors), self._percentile))
            # Cap the estimate to prevent unreasonably large margins
            return min(estimate, max_value)

    def get_stats(self) -> dict:
        """Get tracking error statistics."""
        with self._lock:
            if len(self._errors) < 5:
                return {'sample_count': 0}
            arr = np.array(self._errors)
            return {
                'sample_count': len(arr),
                'e_track_95': float(np.percentile(arr, 95)),
                'e_track_50': float(np.percentile(arr, 50)),
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'min': float(np.min(arr)),
                'max': float(np.max(arr))
            }

    def clear_path(self) -> None:
        """Clear the current path (e.g., when navigation completes)."""
        with self._lock:
            self._current_path = []
            self._path_valid = False

    def reset(self) -> None:
        """Reset tracking error history."""
        with self._lock:
            self._errors.clear()
            self._current_path = []
            self._path_valid = False


@dataclass
class UncertaintyParams:
    """Parameters for uncertainty-aware margin computation with ablation support."""
    k_sigma: float = 3.0              # Confidence multiplier (e.g., 3-sigma)
    localization_sigma: float = 0.1   # Localization uncertainty (meters)
    tracking_error: float = 0.05      # Path tracking error (meters)
    v_max: float = 1.0                # Maximum velocity (m/s)
    latency: float = 0.1              # System latency (seconds)

    # Ablation study flags
    enable_estimation_term: bool = True
    enable_tracking_term: bool = True
    enable_latency_term: bool = True

    # Dynamic v_max tracking
    use_dynamic_v_max: bool = False
    v_max_observed: float = 1.0
    velocity_samples: int = 0

    # Dynamic latency (τ) tracking
    use_dynamic_tau: bool = False
    tau_observed: float = 0.1
    tau_samples: int = 0

    # Dynamic tracking error (e_track) tracking
    use_dynamic_e_track: bool = False
    e_track_observed: float = 0.05
    e_track_samples: int = 0

    def compute_margin(self) -> float:
        """Compute total safety margin with ablation support."""
        margin = 0.0

        # Estimation uncertainty term
        if self.enable_estimation_term:
            margin += self.k_sigma * self.localization_sigma

        # Tracking error term (use dynamic if enabled)
        if self.enable_tracking_term:
            e_track = self.e_track_observed if self.use_dynamic_e_track else self.tracking_error
            margin += e_track

        # Latency compensation term (use dynamic values if enabled)
        if self.enable_latency_term:
            v = self.v_max_observed if self.use_dynamic_v_max else self.v_max
            tau = self.tau_observed if self.use_dynamic_tau else self.latency
            margin += v * tau

        return margin

    def get_margin_breakdown(self) -> Dict[str, float]:
        """Get individual margin components for logging."""
        v = self.v_max_observed if self.use_dynamic_v_max else self.v_max
        tau = self.tau_observed if self.use_dynamic_tau else self.latency
        e_track = self.e_track_observed if self.use_dynamic_e_track else self.tracking_error
        return {
            'estimation': self.k_sigma * self.localization_sigma if self.enable_estimation_term else 0.0,
            'tracking': e_track if self.enable_tracking_term else 0.0,
            'latency': v * tau if self.enable_latency_term else 0.0,
            'total': self.compute_margin(),
        }

    def get_ablation_state(self) -> Dict[str, bool]:
        """Get ablation flags for logging."""
        return {
            'estimation_enabled': self.enable_estimation_term,
            'tracking_enabled': self.enable_tracking_term,
            'latency_enabled': self.enable_latency_term,
        }

    def get_measured_params(self) -> Dict[str, float]:
        """Get measured parameters for logging."""
        return {
            'k_sigma': self.k_sigma,
            'localization_sigma': self.localization_sigma,
            # Tracking error
            'tracking_error': self.tracking_error,
            'e_track_observed': self.e_track_observed,
            'e_track_samples': self.e_track_samples,
            'use_dynamic_e_track': self.use_dynamic_e_track,
            # Velocity
            'v_max': self.v_max,
            'v_max_observed': self.v_max_observed,
            'velocity_samples': self.velocity_samples,
            'use_dynamic_v_max': self.use_dynamic_v_max,
            # Latency
            'latency': self.latency,
            'tau_observed': self.tau_observed,
            'tau_samples': self.tau_samples,
            'use_dynamic_tau': self.use_dynamic_tau,
        }


@dataclass
class PolicyDecision:
    """Result of a policy evaluation."""
    action: PolicyAction
    reason: str
    original_point: Tuple[float, float]
    projected_point: Optional[Tuple[float, float]] = None
    min_distance_to_forbidden: float = float('inf')
    violated_zone: Optional[str] = None
    extra_info: Optional[Dict] = None  # For SafetyChip reprompt_text, SELP automaton state, etc.


class GeofencePolicy:
    """
    Main geofence policy class.

    Manages zones and provides policy decisions for points and paths.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.zones: List[GeofenceZone] = []
        self.uncertainty = UncertaintyParams()
        self._config_path = config_path

        if config_path:
            self.load_config(config_path)

    def load_config(self, config_path: str) -> bool:
        """Load geofence configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            self.zones.clear()

            # Load uncertainty parameters
            if 'uncertainty' in config:
                u = config['uncertainty']
                self.uncertainty = UncertaintyParams(
                    k_sigma=u.get('k_sigma', 3.0),
                    localization_sigma=u.get('localization_sigma', 0.1),
                    tracking_error=u.get('tracking_error', 0.05),
                    v_max=u.get('v_max', 1.0),
                    latency=u.get('latency', 0.1)
                )

            # Load zones
            for zone_data in config.get('zones', []):
                zone = GeofenceZone(
                    name=zone_data['name'],
                    zone_type=ZoneType(zone_data['type']),
                    vertices=[(v['x'], v['y']) for v in zone_data['vertices']],
                    priority=zone_data.get('priority', 0)
                )
                self.zones.append(zone)

            # Sort by priority (higher priority evaluated first)
            self.zones.sort(key=lambda z: z.priority, reverse=True)

            self._config_path = config_path
            return True

        except Exception as e:
            print(f"Failed to load geofence config: {e}")
            return False

    def reload(self) -> bool:
        """Reload configuration from the original file."""
        if self._config_path:
            return self.load_config(self._config_path)
        return False

    def get_safety_margin(self) -> float:
        """Get current safety margin based on uncertainty parameters."""
        return self.uncertainty.compute_margin()

    def evaluate_point(self, x: float, y: float) -> PolicyDecision:
        """
        Evaluate if a point is allowed by the policy.

        Returns a PolicyDecision with action and details.
        """
        point = (x, y)
        margin = self.get_safety_margin()

        min_dist_to_forbidden = float('inf')
        in_buffer = False
        violated_zone = None

        for zone in self.zones:
            if zone.zone_type == ZoneType.FORBIDDEN:
                in_zone = self._point_in_polygon(point, zone.vertices)
                dist = self._distance_to_polygon(point, zone.vertices)

                if dist < min_dist_to_forbidden:
                    min_dist_to_forbidden = dist

                # Check if inside forbidden zone
                if in_zone:
                    projected = self._project_to_safe(point, zone.vertices, margin)
                    return PolicyDecision(
                        action=PolicyAction.REJECT,
                        reason=f"Point inside forbidden zone '{zone.name}'",
                        original_point=point,
                        projected_point=projected,
                        min_distance_to_forbidden=0.0,
                        violated_zone=zone.name
                    )

                # Check if within safety margin
                if dist < margin:
                    violated_zone = zone.name
                    projected = self._project_to_safe(point, zone.vertices, margin)
                    return PolicyDecision(
                        action=PolicyAction.PROJECT,
                        reason=f"Point within safety margin of '{zone.name}' (dist={dist:.3f}m < margin={margin:.3f}m)",
                        original_point=point,
                        projected_point=projected,
                        min_distance_to_forbidden=dist,
                        violated_zone=zone.name
                    )

            elif zone.zone_type == ZoneType.BUFFER:
                if self._point_in_polygon(point, zone.vertices):
                    in_buffer = True

        if in_buffer:
            return PolicyDecision(
                action=PolicyAction.WARN,
                reason="Point in buffer zone",
                original_point=point,
                min_distance_to_forbidden=min_dist_to_forbidden
            )

        return PolicyDecision(
            action=PolicyAction.ALLOW,
            reason="Point in safe area",
            original_point=point,
            min_distance_to_forbidden=min_dist_to_forbidden
        )

    def evaluate_path(self, points: List[Tuple[float, float]]) -> Tuple[bool, List[PolicyDecision]]:
        """
        Evaluate an entire path.

        Returns (is_safe, list of decisions for each point).
        """
        decisions = []
        is_safe = True

        for point in points:
            decision = self.evaluate_point(point[0], point[1])
            decisions.append(decision)
            if decision.action in (PolicyAction.REJECT, PolicyAction.PROJECT):
                is_safe = False

        return is_safe, decisions

    def evaluate_segment(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> PolicyDecision:
        """
        Evaluate if a line segment intersects any forbidden zone.
        """
        margin = self.get_safety_margin()

        for zone in self.zones:
            if zone.zone_type == ZoneType.FORBIDDEN:
                if self._segment_intersects_polygon(p1, p2, zone.vertices, margin):
                    return PolicyDecision(
                        action=PolicyAction.REJECT,
                        reason=f"Path segment intersects forbidden zone '{zone.name}'",
                        original_point=p1,
                        violated_zone=zone.name
                    )

        return PolicyDecision(
            action=PolicyAction.ALLOW,
            reason="Segment is safe",
            original_point=p1
        )

    def get_min_distance_to_forbidden(self, x: float, y: float) -> float:
        """Get minimum distance from point to any forbidden zone."""
        point = (x, y)
        min_dist = float('inf')

        for zone in self.zones:
            if zone.zone_type == ZoneType.FORBIDDEN:
                dist = self._distance_to_polygon(point, zone.vertices)
                if dist < min_dist:
                    min_dist = dist

        return min_dist

    def get_distance_to_zone(self, point: Tuple[float, float], zone: 'GeofenceZone') -> float:
        """Get distance from point to a specific zone's boundary."""
        return self._distance_to_polygon(point, zone.vertices)

    # ==================== Geometry Operations ====================

    def _point_in_polygon(self, point: Tuple[float, float],
                          vertices: List[Tuple[float, float]]) -> bool:
        """Check if point is inside polygon using ray casting."""
        if SHAPELY_AVAILABLE:
            return Polygon(vertices).contains(Point(point))

        # Manual ray casting algorithm
        x, y = point
        n = len(vertices)
        inside = False

        j = n - 1
        for i in range(n):
            xi, yi = vertices[i]
            xj, yj = vertices[j]

            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i

        return inside

    def _distance_to_polygon(self, point: Tuple[float, float],
                             vertices: List[Tuple[float, float]]) -> float:
        """Calculate minimum distance from point to polygon boundary."""
        if SHAPELY_AVAILABLE:
            poly = Polygon(vertices)
            pt = Point(point)
            if poly.contains(pt):
                return 0.0
            return pt.distance(poly.exterior)

        # Manual implementation: distance to each edge
        min_dist = float('inf')
        n = len(vertices)

        for i in range(n):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % n]
            dist = self._point_to_segment_distance(point, p1, p2)
            if dist < min_dist:
                min_dist = dist

        # If inside polygon, distance is 0
        if self._point_in_polygon(point, vertices):
            return 0.0

        return min_dist

    def _point_to_segment_distance(self, point: Tuple[float, float],
                                    seg_start: Tuple[float, float],
                                    seg_end: Tuple[float, float]) -> float:
        """Calculate distance from point to line segment."""
        px, py = point
        x1, y1 = seg_start
        x2, y2 = seg_end

        dx = x2 - x1
        dy = y2 - y1

        if dx == 0 and dy == 0:
            return math.sqrt((px - x1)**2 + (py - y1)**2)

        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx*dx + dy*dy)))

        nearest_x = x1 + t * dx
        nearest_y = y1 + t * dy

        return math.sqrt((px - nearest_x)**2 + (py - nearest_y)**2)

    def _segment_intersects_polygon(self, p1: Tuple[float, float],
                                     p2: Tuple[float, float],
                                     vertices: List[Tuple[float, float]],
                                     margin: float = 0.0) -> bool:
        """Check if line segment intersects polygon (with optional margin)."""
        if SHAPELY_AVAILABLE:
            line = LineString([p1, p2])
            poly = Polygon(vertices)
            if margin > 0:
                poly = poly.buffer(margin)
            return line.intersects(poly)

        # Check if either endpoint is inside
        if self._point_in_polygon(p1, vertices) or self._point_in_polygon(p2, vertices):
            return True

        # Check intersection with each edge
        n = len(vertices)
        for i in range(n):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % n]
            if self._segments_intersect(p1, p2, v1, v2):
                return True

        # Check if segment passes within margin
        if margin > 0:
            # Sample points along segment and check distance
            length = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            steps = max(2, int(length / (margin / 2)))
            for i in range(steps + 1):
                t = i / steps
                px = p1[0] + t * (p2[0] - p1[0])
                py = p1[1] + t * (p2[1] - p1[1])
                if self._distance_to_polygon((px, py), vertices) < margin:
                    return True

        return False

    def _segments_intersect(self, p1: Tuple[float, float], p2: Tuple[float, float],
                            p3: Tuple[float, float], p4: Tuple[float, float]) -> bool:
        """Check if two line segments intersect."""
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4))

    def _project_to_safe(self, point: Tuple[float, float],
                         forbidden_vertices: List[Tuple[float, float]],
                         margin: float) -> Tuple[float, float]:
        """Project a point to the nearest safe location outside the margin."""
        if SHAPELY_AVAILABLE:
            poly = Polygon(forbidden_vertices).buffer(margin)
            pt = Point(point)
            if not poly.contains(pt):
                return point
            # Find nearest point on boundary and move slightly outside
            boundary = poly.exterior
            nearest = boundary.interpolate(boundary.project(pt))
            # Move point away from polygon center
            centroid = poly.centroid
            dx = nearest.x - centroid.x
            dy = nearest.y - centroid.y
            dist = math.sqrt(dx*dx + dy*dy)
            if dist > 0:
                return (nearest.x + 0.1 * dx/dist, nearest.y + 0.1 * dy/dist)
            return (nearest.x + 0.1, nearest.y)

        # Manual projection: find nearest edge and project outward
        min_dist = float('inf')
        nearest_point = point
        n = len(forbidden_vertices)

        for i in range(n):
            p1 = forbidden_vertices[i]
            p2 = forbidden_vertices[(i + 1) % n]
            proj = self._project_to_segment(point, p1, p2)
            dist = math.sqrt((point[0] - proj[0])**2 + (point[1] - proj[1])**2)
            if dist < min_dist:
                min_dist = dist
                nearest_point = proj

        # Move outward from polygon center
        cx = sum(v[0] for v in forbidden_vertices) / len(forbidden_vertices)
        cy = sum(v[1] for v in forbidden_vertices) / len(forbidden_vertices)

        dx = nearest_point[0] - cx
        dy = nearest_point[1] - cy
        dist = math.sqrt(dx*dx + dy*dy)

        if dist > 0:
            # Project to margin distance from boundary
            return (nearest_point[0] + (margin + 0.1) * dx/dist,
                    nearest_point[1] + (margin + 0.1) * dy/dist)

        return (nearest_point[0] + margin + 0.1, nearest_point[1])

    def _project_to_segment(self, point: Tuple[float, float],
                            seg_start: Tuple[float, float],
                            seg_end: Tuple[float, float]) -> Tuple[float, float]:
        """Project point onto line segment."""
        px, py = point
        x1, y1 = seg_start
        x2, y2 = seg_end

        dx = x2 - x1
        dy = y2 - y1

        if dx == 0 and dy == 0:
            return seg_start

        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx*dx + dy*dy)))

        return (x1 + t * dx, y1 + t * dy)

    def get_forbidden_zones_as_marker_array(self):
        """Get visualization markers for forbidden zones."""
        from visualization_msgs.msg import Marker, MarkerArray
        from std_msgs.msg import ColorRGBA
        from geometry_msgs.msg import Point

        markers = MarkerArray()

        for i, zone in enumerate(self.zones):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.ns = "geofence"
            marker.id = i
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.1

            if zone.zone_type == ZoneType.FORBIDDEN:
                marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.8)
            elif zone.zone_type == ZoneType.BUFFER:
                marker.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.6)
            else:
                marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.4)

            for vertex in zone.vertices:
                p = Point()
                p.x, p.y, p.z = vertex[0], vertex[1], 0.1
                marker.points.append(p)
            # Close the polygon
            p = Point()
            p.x, p.y, p.z = zone.vertices[0][0], zone.vertices[0][1], 0.1
            marker.points.append(p)

            markers.markers.append(marker)

        return markers
