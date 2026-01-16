"""
Geofence Geometry Module

This module handles all geometric operations for the probabilistic geofence:
- Loading forbidden zone definitions from YAML
- Inflating polygons by the computed safety margin (Minkowski sum)
- Point-in-polygon containment tests
- Projection of unsafe goals to nearest safe boundary points

Mathematical Background:
------------------------
Polygon inflation is performed as a Minkowski sum with a disk of radius r:
    P_inflated = P ⊕ B(0, r)

Where B(0, r) is a ball (disk in 2D) centered at origin with radius r.
This ensures that any point within distance r of the original polygon
is contained in the inflated polygon.

Shapely's buffer() operation implements this efficiently using
approximate arc segments for the disk.

Usage:
------
    geometry = GeofenceGeometry.from_yaml("geofence.yaml")
    inflated = geometry.get_inflated_zones(margin=0.3)

    if geometry.is_point_in_forbidden(x, y, margin):
        # Handle violation

    safe_x, safe_y = geometry.project_to_safe(x, y, margin)
"""

import yaml
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field

from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import nearest_points, unary_union
from shapely.validation import make_valid


@dataclass
class ForbiddenZone:
    """
    Represents a single forbidden zone polygon.

    Attributes:
        name: Unique identifier for the zone
        vertices: List of (x, y) tuples defining the polygon boundary
        polygon: Shapely Polygon object
        metadata: Optional additional properties (e.g., zone type, priority)
    """
    name: str
    vertices: List[Tuple[float, float]]
    polygon: Polygon
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate and fix polygon geometry if necessary."""
        if not self.polygon.is_valid:
            self.polygon = make_valid(self.polygon)


@dataclass
class GeofenceCheckResult:
    """
    Result of a geofence containment check.

    Attributes:
        is_violated: True if point is inside a forbidden zone or z-boundary violated
        violated_zones: List of zone names that contain the point
        distance_to_boundary: Signed distance (negative = inside)
        nearest_safe_point: Projected safe point if violated
        in_buffer_zones: List of buffer zone names (warning only)
        in_forbidden_zones: List of forbidden zone names (blocking)
        z_violated: True if z-coordinate is outside allowed range
        z_distance: Distance to nearest z-boundary (0 if within range)
    """
    is_violated: bool
    violated_zones: List[str]
    distance_to_boundary: float
    nearest_safe_point: Optional[Tuple[float, float]] = None
    in_buffer_zones: List[str] = field(default_factory=list)
    in_forbidden_zones: List[str] = field(default_factory=list)
    z_violated: bool = False
    z_distance: float = 0.0


class GeofenceGeometry:
    """
    Manages forbidden zone geometry and spatial queries.

    This class handles:
    1. Loading zone definitions from YAML configuration
    2. Inflating polygons by dynamic safety margins
    3. Point-in-polygon tests with margin consideration
    4. Projection of violated points to safe locations
    5. Z-axis (height) boundary enforcement for 3D environments

    The system uses a 2.5D approach:
    - XY plane: Polygon-based forbidden zones with Minkowski sum inflation
    - Z axis: Height boundary constraints for ground robot safety

    This is appropriate for ground mobile robots operating in 3D simulation
    environments (e.g., Gazebo) where the robot moves on a planar surface
    but the environment is fully 3D.

    The inflation uses Shapely's buffer() which computes the
    Minkowski sum of the polygon with a disk of the specified radius.

    Parameters:
        resolution: Number of segments per quarter circle for buffer approximation
                   Higher values = smoother curves but more computation
        z_min: Minimum allowed z-coordinate (default: -0.1m, slight ground tolerance)
        z_max: Maximum allowed z-coordinate (default: 0.5m, robot height)
    """

    # Default buffer resolution (segments per quarter circle)
    DEFAULT_RESOLUTION = 16

    # Default z-axis boundaries for ground robots
    DEFAULT_Z_MIN = -0.1  # Slight tolerance for uneven terrain
    DEFAULT_Z_MAX = 0.5   # Maximum robot height from ground

    def __init__(
        self,
        resolution: int = DEFAULT_RESOLUTION,
        z_min: float = DEFAULT_Z_MIN,
        z_max: float = DEFAULT_Z_MAX
    ):
        """
        Initialize an empty geofence geometry manager.

        Args:
            resolution: Segments per quarter circle for polygon inflation
            z_min: Minimum allowed z-coordinate (meters)
            z_max: Maximum allowed z-coordinate (meters)
        """
        self._zones: Dict[str, ForbiddenZone] = {}
        self._resolution = resolution

        # Z-axis boundaries for 3D environment support
        self._z_min = z_min
        self._z_max = z_max

        # Cache for inflated polygons (keyed by margin value)
        self._inflation_cache: Dict[float, Dict[str, Polygon]] = {}
        self._union_cache: Dict[float, Union[Polygon, MultiPolygon]] = {}

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path], resolution: int = DEFAULT_RESOLUTION) -> 'GeofenceGeometry':
        """
        Load forbidden zones from a YAML configuration file.

        Expected YAML format:
        ```yaml
        # Optional: Z-axis boundaries for 3D environments (Gazebo, etc.)
        z_boundaries:
          z_min: -0.1   # Minimum allowed height (pit detection)
          z_max: 0.5    # Maximum allowed height (flip detection)

        forbidden_zones:
          - name: "zone_1"
            vertices:
              - [x1, y1]
              - [x2, y2]
              - [x3, y3]
            metadata:
              type: "restricted_area"

          - name: "zone_2"
            vertices:
              - [x1, y1]
              ...
        ```

        Args:
            yaml_path: Path to the YAML configuration file
            resolution: Buffer resolution for polygon inflation

        Returns:
            Configured GeofenceGeometry instance

        Raises:
            FileNotFoundError: If YAML file doesn't exist
            ValueError: If YAML format is invalid
        """
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Geofence config not found: {path}")

        with open(path, 'r') as f:
            config = yaml.safe_load(f)

        # Load z-axis boundaries if specified
        z_min = cls.DEFAULT_Z_MIN
        z_max = cls.DEFAULT_Z_MAX

        if 'z_boundaries' in config:
            z_config = config['z_boundaries']
            z_min = z_config.get('z_min', z_min)
            z_max = z_config.get('z_max', z_max)

        instance = cls(resolution=resolution, z_min=z_min, z_max=z_max)
        instance._load_zones_from_config(config)

        return instance

    def _load_zones_from_config(self, config: Dict) -> None:
        """
        Parse zone definitions from loaded YAML config.

        Supports two formats:
        1. Legacy format: 'forbidden_zones' with vertices as [x, y] lists
        2. New format: 'zones' with type field and vertices as {x: ..., y: ...} dicts

        Args:
            config: Parsed YAML dictionary

        Raises:
            ValueError: If required fields are missing
        """
        # Support both 'forbidden_zones' (legacy) and 'zones' (new) keys
        if 'forbidden_zones' in config:
            zones = config['forbidden_zones']
        elif 'zones' in config:
            zones = config['zones']
        else:
            raise ValueError("YAML must contain 'forbidden_zones' or 'zones' key")

        if not isinstance(zones, list):
            raise ValueError("zones must be a list")

        for zone_def in zones:
            self._add_zone_from_dict(zone_def)

    def _add_zone_from_dict(self, zone_def: Dict) -> None:
        """
        Create and add a forbidden zone from dictionary definition.

        Supports two vertex formats:
        1. Legacy: [[x1, y1], [x2, y2], ...]
        2. New: [{x: x1, y: y1}, {x: x2, y: y2}, ...]

        Args:
            zone_def: Dictionary with 'name' and 'vertices' keys

        Raises:
            ValueError: If zone definition is invalid
        """
        if 'name' not in zone_def:
            raise ValueError("Zone definition must have 'name' field")
        if 'vertices' not in zone_def:
            raise ValueError(f"Zone '{zone_def['name']}' must have 'vertices' field")

        name = zone_def['name']

        # Parse vertices - support both formats
        raw_vertices = zone_def['vertices']
        vertices = []
        for v in raw_vertices:
            if isinstance(v, dict):
                # New format: {x: ..., y: ...}
                vertices.append((v['x'], v['y']))
            else:
                # Legacy format: [x, y]
                vertices.append(tuple(v))

        # Handle metadata - support both 'metadata' dict and top-level fields
        metadata = zone_def.get('metadata', {})

        # Add type and priority to metadata if present at top level
        if 'type' in zone_def:
            metadata['type'] = zone_def['type']
        if 'priority' in zone_def:
            metadata['priority'] = zone_def['priority']

        # Validate polygon requirements
        if len(vertices) < 3:
            raise ValueError(f"Zone '{name}' must have at least 3 vertices")

        # Create Shapely polygon
        polygon = Polygon(vertices)

        # Store the zone
        self._zones[name] = ForbiddenZone(
            name=name,
            vertices=vertices,
            polygon=polygon,
            metadata=metadata
        )

        # Invalidate caches when zones change
        self._inflation_cache.clear()
        self._union_cache.clear()

    def add_zone(
        self,
        name: str,
        vertices: List[Tuple[float, float]],
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Add a forbidden zone programmatically.

        Args:
            name: Unique identifier for the zone
            vertices: List of (x, y) boundary points
            metadata: Optional additional properties

        Raises:
            ValueError: If zone with same name exists or vertices invalid
        """
        if name in self._zones:
            raise ValueError(f"Zone '{name}' already exists")

        zone_def = {
            'name': name,
            'vertices': vertices,
            'metadata': metadata or {}
        }
        self._add_zone_from_dict(zone_def)

    def remove_zone(self, name: str) -> bool:
        """
        Remove a forbidden zone by name.

        Args:
            name: Zone identifier to remove

        Returns:
            True if zone was removed, False if not found
        """
        if name in self._zones:
            del self._zones[name]
            self._inflation_cache.clear()
            self._union_cache.clear()
            return True
        return False

    def get_zone_names(self) -> List[str]:
        """Get list of all zone names."""
        return list(self._zones.keys())

    def get_zone(self, name: str) -> Optional[ForbiddenZone]:
        """Get a specific zone by name."""
        return self._zones.get(name)

    def get_zone_type(self, name: str) -> str:
        """Get zone type ('forbidden' or 'buffer'). Defaults to 'forbidden'."""
        zone = self._zones.get(name)
        if zone:
            return zone.metadata.get('type', 'forbidden')
        return 'forbidden'

    def get_forbidden_zone_names(self) -> List[str]:
        """Get list of zone names with type 'forbidden'."""
        return [name for name in self._zones.keys()
                if self.get_zone_type(name) == 'forbidden']

    def get_buffer_zone_names(self) -> List[str]:
        """Get list of zone names with type 'buffer'."""
        return [name for name in self._zones.keys()
                if self.get_zone_type(name) == 'buffer']

    def get_zones_by_type(self, zone_type: str) -> Dict[str, ForbiddenZone]:
        """Get all zones of a specific type."""
        return {name: zone for name, zone in self._zones.items()
                if zone.metadata.get('type', 'forbidden') == zone_type}

    def _inflate_polygon(self, polygon: Polygon, margin: float) -> Polygon:
        """
        Inflate a polygon by a specified margin (Minkowski sum with disk).

        The buffer operation creates a polygon that is `margin` distance
        away from the original boundary at all points.

        Args:
            polygon: Original Shapely polygon
            margin: Inflation distance in meters

        Returns:
            Inflated polygon

        Note:
            - Positive margin expands the polygon (makes forbidden area larger)
            - Zero margin returns the original polygon
            - Negative margin would shrink (not used in this application)
        """
        if margin <= 0:
            return polygon

        # buffer() performs Minkowski sum with approximate disk
        # join_style=1 is "round" for smooth corners
        # cap_style=1 is "round" for smooth line endpoints
        inflated = polygon.buffer(
            margin,
            resolution=self._resolution,
            join_style=1,  # Round
            cap_style=1    # Round
        )

        # Ensure valid geometry
        if not inflated.is_valid:
            inflated = make_valid(inflated)

        return inflated

    def get_inflated_zones(self, margin: float) -> Dict[str, Polygon]:
        """
        Get all forbidden zones inflated by the specified margin.

        Uses caching to avoid recomputation for the same margin value.

        Args:
            margin: Safety margin in meters

        Returns:
            Dictionary mapping zone names to inflated polygons
        """
        # Round margin for cache key (avoid floating point issues)
        cache_key = round(margin, 6)

        if cache_key in self._inflation_cache:
            return self._inflation_cache[cache_key]

        inflated = {}
        for name, zone in self._zones.items():
            inflated[name] = self._inflate_polygon(zone.polygon, margin)

        self._inflation_cache[cache_key] = inflated
        return inflated

    def get_inflated_union(self, margin: float) -> Union[Polygon, MultiPolygon]:
        """
        Get the union of all inflated forbidden zones.

        Useful for efficient containment tests when multiple zones
        may overlap after inflation.

        Args:
            margin: Safety margin in meters

        Returns:
            Union of all inflated zones (Polygon or MultiPolygon)
        """
        cache_key = round(margin, 6)

        if cache_key in self._union_cache:
            return self._union_cache[cache_key]

        inflated_zones = self.get_inflated_zones(margin)
        polygons = list(inflated_zones.values())

        if not polygons:
            # Return empty polygon if no zones defined
            union = Polygon()
        elif len(polygons) == 1:
            union = polygons[0]
        else:
            union = unary_union(polygons)

        self._union_cache[cache_key] = union
        return union

    def is_point_in_forbidden(
        self,
        x: float,
        y: float,
        margin: float = 0.0
    ) -> bool:
        """
        Check if a point is inside any inflated forbidden zone.

        Args:
            x: X coordinate
            y: Y coordinate
            margin: Safety margin for inflation

        Returns:
            True if point is inside any inflated forbidden zone
        """
        point = Point(x, y)
        union = self.get_inflated_union(margin)
        return union.contains(point) or union.touches(point)

    def check_point(
        self,
        x: float,
        y: float,
        margin: float = 0.0
    ) -> GeofenceCheckResult:
        """
        Comprehensive check of a point against the geofence.

        Returns detailed information about:
        - Whether the point violates the geofence (forbidden zones only)
        - Which specific zones are violated
        - Distance to the nearest boundary
        - Nearest safe point if violated
        - Separate tracking of buffer zones (warning only)

        Args:
            x: X coordinate
            y: Y coordinate
            margin: Safety margin for inflation

        Returns:
            GeofenceCheckResult with detailed violation information
        """
        point = Point(x, y)
        violated_zones = []
        in_forbidden_zones = []
        in_buffer_zones = []

        # Check each inflated zone individually
        inflated_zones = self.get_inflated_zones(margin)
        for name, inflated in inflated_zones.items():
            if inflated.contains(point) or inflated.touches(point):
                violated_zones.append(name)
                zone_type = self.get_zone_type(name)
                if zone_type == 'buffer':
                    in_buffer_zones.append(name)
                else:
                    in_forbidden_zones.append(name)

        # Only forbidden zones count as violations
        is_violated = len(in_forbidden_zones) > 0

        # Compute signed distance to boundary (forbidden zones only)
        forbidden_zones = self.get_zones_by_type('forbidden')
        if forbidden_zones:
            forbidden_polygons = [
                self._inflate_polygon(z.polygon, margin)
                for z in forbidden_zones.values()
            ]
            union = unary_union(forbidden_polygons) if len(forbidden_polygons) > 1 else forbidden_polygons[0]
        else:
            union = Polygon()

        if union.is_empty:
            distance = float('inf')
        else:
            boundary = union.boundary
            distance = point.distance(boundary)
            if is_violated:
                distance = -distance  # Negative for inside

        # Find nearest safe point if violated
        nearest_safe = None
        if is_violated and not union.is_empty:
            safe_point = self._project_to_boundary(point, union)
            nearest_safe = (safe_point.x, safe_point.y)

        return GeofenceCheckResult(
            is_violated=is_violated,
            violated_zones=violated_zones,
            distance_to_boundary=distance,
            in_buffer_zones=in_buffer_zones,
            in_forbidden_zones=in_forbidden_zones,
            nearest_safe_point=nearest_safe
        )

    def check_point_3d(
        self,
        x: float,
        y: float,
        z: float,
        margin: float = 0.0
    ) -> GeofenceCheckResult:
        """
        Comprehensive 3D check of a point against the geofence.

        This method extends the 2D check with z-axis (height) validation,
        making it suitable for 3D simulation environments like Gazebo.

        For ground mobile robots, the z-axis constraint ensures:
        - Robot hasn't fallen into a pit (z < z_min)
        - Robot hasn't been lifted/flipped (z > z_max)
        - Robot maintains ground contact within tolerance

        The approach is "2.5D":
        - XY plane: Full polygon-based forbidden zone checking
        - Z axis: Simple boundary constraint checking

        This is appropriate because ground robots have constrained
        z-motion (they roll on surfaces), unlike drones which need
        full 3D volumetric geofencing.

        Args:
            x: X coordinate (meters)
            y: Y coordinate (meters)
            z: Z coordinate (height, meters)
            margin: Safety margin for XY zone inflation

        Returns:
            GeofenceCheckResult with both XY and Z violation information
        """
        # First, perform standard 2D check
        result = self.check_point(x, y, margin)

        # Then, check z-axis boundaries
        z_violated = False
        z_distance = 0.0

        if z < self._z_min:
            z_violated = True
            z_distance = self._z_min - z
            result.violated_zones.append(f"z_floor (z={z:.3f} < {self._z_min})")
        elif z > self._z_max:
            z_violated = True
            z_distance = z - self._z_max
            result.violated_zones.append(f"z_ceiling (z={z:.3f} > {self._z_max})")

        # Update result with z-axis information
        result.z_violated = z_violated
        result.z_distance = z_distance

        # Overall violation is XY OR Z
        if z_violated:
            result.is_violated = True

        return result

    def set_z_boundaries(self, z_min: float, z_max: float) -> None:
        """
        Update z-axis boundaries at runtime.

        Useful for adapting to different robot heights or terrain.

        Args:
            z_min: Minimum allowed z-coordinate (meters)
            z_max: Maximum allowed z-coordinate (meters)

        Raises:
            ValueError: If z_min >= z_max
        """
        if z_min >= z_max:
            raise ValueError(f"z_min ({z_min}) must be less than z_max ({z_max})")

        self._z_min = z_min
        self._z_max = z_max

    def get_z_boundaries(self) -> Tuple[float, float]:
        """Get current z-axis boundaries (z_min, z_max)."""
        return (self._z_min, self._z_max)

    def _project_to_boundary(
        self,
        point: Point,
        polygon: Union[Polygon, MultiPolygon]
    ) -> Point:
        """
        Project a point to the nearest location on the polygon boundary.

        For a point inside the polygon, this finds the closest point
        on the boundary, which is the "safest" projection.

        Args:
            point: Point to project
            polygon: Polygon or MultiPolygon boundary

        Returns:
            Nearest point on the boundary
        """
        boundary = polygon.boundary

        # nearest_points returns (point_on_geom1, point_on_geom2)
        _, nearest = nearest_points(point, boundary)

        return nearest

    def project_to_safe(
        self,
        x: float,
        y: float,
        margin: float = 0.0,
        safety_buffer: float = 0.01
    ) -> Tuple[float, float]:
        """
        Project a point to the nearest safe location outside forbidden zones.

        If the point is already safe, returns the original coordinates.
        If violated, projects to the boundary plus a small safety buffer.

        The safety buffer ensures the projected point is definitively
        outside the inflated zone (accounts for numerical precision).

        Args:
            x: X coordinate
            y: Y coordinate
            margin: Safety margin for zone inflation
            safety_buffer: Additional outward offset from boundary [m]

        Returns:
            Tuple (safe_x, safe_y) of projected coordinates
        """
        check = self.check_point(x, y, margin)

        if not check.is_violated:
            return (x, y)

        # Get the nearest boundary point
        point = Point(x, y)
        union = self.get_inflated_union(margin)
        boundary_point = self._project_to_boundary(point, union)

        # Apply safety buffer: move outward from the center of the polygon
        # Direction: from polygon centroid through boundary point, extended
        if safety_buffer > 0:
            centroid = union.centroid
            dx = boundary_point.x - centroid.x
            dy = boundary_point.y - centroid.y
            dist = np.sqrt(dx * dx + dy * dy)

            if dist > 1e-10:
                # Normalize and extend by safety buffer
                safe_x = boundary_point.x + (dx / dist) * safety_buffer
                safe_y = boundary_point.y + (dy / dist) * safety_buffer
            else:
                # Fallback: just use boundary point
                safe_x = boundary_point.x
                safe_y = boundary_point.y
        else:
            safe_x = boundary_point.x
            safe_y = boundary_point.y

        return (safe_x, safe_y)

    def validate_goal(
        self,
        goal_x: float,
        goal_y: float,
        margin: float = 0.0,
        enable_projection: bool = True
    ) -> Tuple[bool, float, float, Optional[str]]:
        """
        Validate a navigation goal against the geofence.

        This is the primary interface for goal validation in the policy enforcer.

        Args:
            goal_x: Goal X coordinate
            goal_y: Goal Y coordinate
            margin: Current safety margin
            enable_projection: If True, project unsafe goals to safe locations

        Returns:
            Tuple of:
                - is_valid: True if goal is safe (or was successfully projected)
                - final_x: Safe X coordinate (original or projected)
                - final_y: Safe Y coordinate (original or projected)
                - message: Descriptive message about the result
        """
        check = self.check_point(goal_x, goal_y, margin)

        if not check.is_violated:
            return (
                True,
                goal_x,
                goal_y,
                f"Goal ({goal_x:.3f}, {goal_y:.3f}) is safe"
            )

        zones_str = ", ".join(check.violated_zones)

        if not enable_projection:
            return (
                False,
                goal_x,
                goal_y,
                f"Goal ({goal_x:.3f}, {goal_y:.3f}) BLOCKED - inside {zones_str}"
            )

        # Project to safe location
        safe_x, safe_y = self.project_to_safe(goal_x, goal_y, margin)
        return (
            True,
            safe_x,
            safe_y,
            f"Goal projected from ({goal_x:.3f}, {goal_y:.3f}) to "
            f"({safe_x:.3f}, {safe_y:.3f}) - was inside {zones_str}"
        )

    def get_zone_info(self) -> Dict:
        """
        Get information about all configured zones for debugging/logging.

        Returns:
            Dictionary with zone information
        """
        info = {}
        for name, zone in self._zones.items():
            info[name] = {
                'vertices': zone.vertices,
                'area': zone.polygon.area,
                'centroid': (zone.polygon.centroid.x, zone.polygon.centroid.y),
                'bounds': zone.polygon.bounds,  # (minx, miny, maxx, maxy)
                'metadata': zone.metadata
            }
        return info

    def clear_caches(self) -> None:
        """Clear all cached inflation results."""
        self._inflation_cache.clear()
        self._union_cache.clear()


# =============================================================================
# UNIT TESTS / VERIFICATION
# =============================================================================

def _verify_polygon_inflation():
    """Verify polygon inflation works correctly."""
    # Simple square
    square = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

    geo = GeofenceGeometry()
    geo.add_zone("test_square", [(0, 0), (1, 0), (1, 1), (0, 1)])

    # No inflation
    inflated_0 = geo.get_inflated_zones(0.0)["test_square"]
    assert abs(inflated_0.area - 1.0) < 0.01

    # Inflated by 0.1m
    # New area should be approximately (1 + 2*0.1)² ≈ 1.44 plus corner arcs
    inflated = geo.get_inflated_zones(0.1)["test_square"]
    assert inflated.area > 1.4  # Should be larger

    print("✓ Polygon inflation verification passed")


def _verify_containment():
    """Verify point containment tests."""
    geo = GeofenceGeometry()
    geo.add_zone("test", [(0, 0), (1, 0), (1, 1), (0, 1)])

    # Point inside
    assert geo.is_point_in_forbidden(0.5, 0.5, margin=0.0)

    # Point outside
    assert not geo.is_point_in_forbidden(2.0, 2.0, margin=0.0)

    # Point outside but within margin
    assert geo.is_point_in_forbidden(1.05, 0.5, margin=0.1)

    print("✓ Containment test verification passed")


def _verify_projection():
    """Verify projection to safe point."""
    geo = GeofenceGeometry()
    geo.add_zone("test", [(0, 0), (1, 0), (1, 1), (0, 1)])

    # Point inside should be projected out
    safe_x, safe_y = geo.project_to_safe(0.5, 0.5, margin=0.0)
    assert not geo.is_point_in_forbidden(safe_x, safe_y, margin=0.0)

    print("✓ Projection verification passed")


if __name__ == "__main__":
    _verify_polygon_inflation()
    _verify_containment()
    _verify_projection()
    print("\nAll geometry verifications passed!")
