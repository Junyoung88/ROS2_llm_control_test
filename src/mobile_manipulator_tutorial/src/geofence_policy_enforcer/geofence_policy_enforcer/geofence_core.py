#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Geofence Core - Geometry operations and policy logic.

Implements polygon-based geofencing with uncertainty-aware margins.
Uses shapely if available, otherwise falls back to manual implementations.
"""

import math
import yaml
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
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


@dataclass
class UncertaintyParams:
    """Parameters for uncertainty-aware margin computation."""
    k_sigma: float = 3.0              # Confidence multiplier (e.g., 3-sigma)
    localization_sigma: float = 0.1   # Localization uncertainty (meters)
    tracking_error: float = 0.05      # Path tracking error (meters)
    v_max: float = 1.0                # Maximum velocity (m/s)
    latency: float = 0.1              # System latency (seconds)

    def compute_margin(self) -> float:
        """Compute total safety margin."""
        return (self.k_sigma * self.localization_sigma +
                self.tracking_error +
                self.v_max * self.latency)


@dataclass
class PolicyDecision:
    """Result of a policy evaluation."""
    action: PolicyAction
    reason: str
    original_point: Tuple[float, float]
    projected_point: Optional[Tuple[float, float]] = None
    min_distance_to_forbidden: float = float('inf')
    violated_zone: Optional[str] = None


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
