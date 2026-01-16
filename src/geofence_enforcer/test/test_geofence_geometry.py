"""
Unit tests for the Geofence Geometry module.

These tests verify the correctness of:
1. Polygon loading and validation
2. Polygon inflation (Minkowski sum with disk)
3. Point-in-polygon containment tests
4. Projection to nearest safe point
"""

import pytest
import tempfile
import yaml
from pathlib import Path

from geofence_enforcer.geofence_geometry import (
    GeofenceGeometry,
    ForbiddenZone,
    GeofenceCheckResult
)


class TestPolygonLoading:
    """Tests for loading polygons from YAML and programmatically."""

    def test_add_zone_programmatically(self):
        """Add a zone via the API."""
        geo = GeofenceGeometry()
        geo.add_zone(
            name="test_zone",
            vertices=[(0, 0), (1, 0), (1, 1), (0, 1)]
        )

        assert "test_zone" in geo.get_zone_names()
        zone = geo.get_zone("test_zone")
        assert zone is not None
        assert len(zone.vertices) == 4

    def test_add_zone_with_metadata(self):
        """Add a zone with metadata."""
        geo = GeofenceGeometry()
        geo.add_zone(
            name="test_zone",
            vertices=[(0, 0), (1, 0), (1, 1), (0, 1)],
            metadata={"type": "equipment", "priority": "high"}
        )

        zone = geo.get_zone("test_zone")
        assert zone.metadata["type"] == "equipment"

    def test_load_from_yaml(self):
        """Load zones from a YAML file."""
        yaml_content = """
forbidden_zones:
  - name: "zone_a"
    vertices:
      - [0, 0]
      - [2, 0]
      - [2, 2]
      - [0, 2]
  - name: "zone_b"
    vertices:
      - [5, 5]
      - [7, 5]
      - [7, 7]
      - [5, 7]
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            yaml_path = f.name

        geo = GeofenceGeometry.from_yaml(yaml_path)

        assert len(geo.get_zone_names()) == 2
        assert "zone_a" in geo.get_zone_names()
        assert "zone_b" in geo.get_zone_names()

        Path(yaml_path).unlink()

    def test_load_invalid_yaml_missing_zones(self):
        """Reject YAML without forbidden_zones key."""
        yaml_content = """
other_key:
  - name: "test"
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            yaml_path = f.name

        with pytest.raises(ValueError, match="forbidden_zones"):
            GeofenceGeometry.from_yaml(yaml_path)

        Path(yaml_path).unlink()

    def test_duplicate_zone_name_rejected(self):
        """Reject duplicate zone names."""
        geo = GeofenceGeometry()
        geo.add_zone("test", [(0, 0), (1, 0), (1, 1)])

        with pytest.raises(ValueError, match="already exists"):
            geo.add_zone("test", [(2, 2), (3, 2), (3, 3)])

    def test_insufficient_vertices_rejected(self):
        """Reject polygons with fewer than 3 vertices."""
        geo = GeofenceGeometry()

        with pytest.raises(ValueError, match="at least 3 vertices"):
            geo.add_zone("invalid", [(0, 0), (1, 1)])

    def test_remove_zone(self):
        """Remove a zone by name."""
        geo = GeofenceGeometry()
        geo.add_zone("temp", [(0, 0), (1, 0), (1, 1)])

        assert geo.remove_zone("temp")
        assert "temp" not in geo.get_zone_names()
        assert not geo.remove_zone("nonexistent")


class TestPolygonInflation:
    """Tests for polygon inflation (Minkowski sum)."""

    def test_zero_inflation(self):
        """Zero margin returns original polygon area."""
        geo = GeofenceGeometry()
        geo.add_zone("square", [(0, 0), (1, 0), (1, 1), (0, 1)])

        inflated = geo.get_inflated_zones(margin=0.0)

        assert abs(inflated["square"].area - 1.0) < 0.01

    def test_positive_inflation(self):
        """Positive margin expands the polygon."""
        geo = GeofenceGeometry()
        geo.add_zone("square", [(0, 0), (1, 0), (1, 1), (0, 1)])

        inflated = geo.get_inflated_zones(margin=0.1)

        # Inflated area > original area
        assert inflated["square"].area > 1.0

        # Approximate expected area:
        # (1 + 2*0.1)² = 1.44 for corners, plus circular arcs at corners
        # Total should be around 1.44 + π*0.1² ≈ 1.47
        assert inflated["square"].area > 1.4

    def test_inflation_caching(self):
        """Inflation results are cached."""
        geo = GeofenceGeometry()
        geo.add_zone("square", [(0, 0), (1, 0), (1, 1), (0, 1)])

        inflated1 = geo.get_inflated_zones(margin=0.1)
        inflated2 = geo.get_inflated_zones(margin=0.1)

        # Same object should be returned (cached)
        assert inflated1 is inflated2

    def test_cache_invalidation(self):
        """Cache is cleared when zones change."""
        geo = GeofenceGeometry()
        geo.add_zone("square", [(0, 0), (1, 0), (1, 1), (0, 1)])

        geo.get_inflated_zones(margin=0.1)

        # Add new zone should invalidate cache
        geo.add_zone("triangle", [(2, 2), (3, 2), (2.5, 3)])

        inflated = geo.get_inflated_zones(margin=0.1)
        assert "triangle" in inflated

    def test_union_computation(self):
        """Union of overlapping inflated zones."""
        geo = GeofenceGeometry()
        geo.add_zone("zone1", [(0, 0), (1, 0), (1, 1), (0, 1)])
        geo.add_zone("zone2", [(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)])

        union = geo.get_inflated_union(margin=0.0)

        # Union area should be less than sum of individual areas
        assert union.area < 2.0


class TestContainmentChecks:
    """Tests for point-in-polygon containment."""

    def test_point_inside_original(self):
        """Point inside original polygon."""
        geo = GeofenceGeometry()
        geo.add_zone("square", [(0, 0), (2, 0), (2, 2), (0, 2)])

        assert geo.is_point_in_forbidden(1.0, 1.0, margin=0.0)

    def test_point_outside_original(self):
        """Point outside original polygon."""
        geo = GeofenceGeometry()
        geo.add_zone("square", [(0, 0), (2, 0), (2, 2), (0, 2)])

        assert not geo.is_point_in_forbidden(3.0, 3.0, margin=0.0)

    def test_point_in_inflated_region(self):
        """Point outside original but inside inflated region."""
        geo = GeofenceGeometry()
        geo.add_zone("square", [(0, 0), (2, 0), (2, 2), (0, 2)])

        # Point just outside original boundary
        assert not geo.is_point_in_forbidden(2.05, 1.0, margin=0.0)

        # Same point now inside inflated boundary
        assert geo.is_point_in_forbidden(2.05, 1.0, margin=0.1)

    def test_comprehensive_check(self):
        """Test detailed check result."""
        geo = GeofenceGeometry()
        geo.add_zone("square", [(0, 0), (2, 0), (2, 2), (0, 2)])

        # Point inside
        result = geo.check_point(1.0, 1.0, margin=0.0)
        assert result.is_violated
        assert "square" in result.violated_zones
        assert result.distance_to_boundary < 0  # Negative = inside
        assert result.nearest_safe_point is not None

        # Point outside
        result = geo.check_point(5.0, 5.0, margin=0.0)
        assert not result.is_violated
        assert len(result.violated_zones) == 0
        assert result.distance_to_boundary > 0  # Positive = outside

    def test_multiple_zone_violation(self):
        """Point violating multiple overlapping zones."""
        geo = GeofenceGeometry()
        geo.add_zone("zone1", [(0, 0), (2, 0), (2, 2), (0, 2)])
        geo.add_zone("zone2", [(1, 1), (3, 1), (3, 3), (1, 3)])

        # Point in overlap region
        result = geo.check_point(1.5, 1.5, margin=0.0)
        assert result.is_violated
        assert len(result.violated_zones) == 2


class TestProjection:
    """Tests for projecting points to safe locations."""

    def test_projection_from_inside(self):
        """Project point from inside to boundary."""
        geo = GeofenceGeometry()
        geo.add_zone("square", [(0, 0), (2, 0), (2, 2), (0, 2)])

        safe_x, safe_y = geo.project_to_safe(1.0, 1.0, margin=0.0)

        # Projected point should be outside
        assert not geo.is_point_in_forbidden(safe_x, safe_y, margin=0.0)

    def test_projection_already_safe(self):
        """Safe point returns unchanged."""
        geo = GeofenceGeometry()
        geo.add_zone("square", [(0, 0), (2, 0), (2, 2), (0, 2)])

        safe_x, safe_y = geo.project_to_safe(5.0, 5.0, margin=0.0)

        assert abs(safe_x - 5.0) < 1e-10
        assert abs(safe_y - 5.0) < 1e-10

    def test_projection_with_margin(self):
        """Projection considers margin."""
        geo = GeofenceGeometry()
        geo.add_zone("square", [(0, 0), (2, 0), (2, 2), (0, 2)])

        # Point at (2.05, 1.0) is outside original but inside inflated
        safe_x, safe_y = geo.project_to_safe(2.05, 1.0, margin=0.1)

        # Should be projected outside inflated boundary
        assert not geo.is_point_in_forbidden(safe_x, safe_y, margin=0.1)

    def test_projection_safety_buffer(self):
        """Projection includes safety buffer."""
        geo = GeofenceGeometry()
        geo.add_zone("square", [(0, 0), (2, 0), (2, 2), (0, 2)])

        # Project with explicit safety buffer
        safe_x, safe_y = geo.project_to_safe(1.0, 1.0, margin=0.0, safety_buffer=0.05)

        # Should be clearly outside
        result = geo.check_point(safe_x, safe_y, margin=0.0)
        assert not result.is_violated
        assert result.distance_to_boundary > 0


class TestGoalValidation:
    """Tests for goal validation interface."""

    def test_valid_goal(self):
        """Goal outside forbidden zones."""
        geo = GeofenceGeometry()
        geo.add_zone("square", [(0, 0), (2, 0), (2, 2), (0, 2)])

        is_valid, final_x, final_y, msg = geo.validate_goal(5.0, 5.0, margin=0.0)

        assert is_valid
        assert abs(final_x - 5.0) < 1e-10
        assert abs(final_y - 5.0) < 1e-10
        assert "safe" in msg.lower()

    def test_invalid_goal_blocked(self):
        """Goal inside forbidden zone, projection disabled."""
        geo = GeofenceGeometry()
        geo.add_zone("square", [(0, 0), (2, 0), (2, 2), (0, 2)])

        is_valid, final_x, final_y, msg = geo.validate_goal(
            1.0, 1.0,
            margin=0.0,
            enable_projection=False
        )

        assert not is_valid
        assert "BLOCKED" in msg

    def test_invalid_goal_projected(self):
        """Goal inside forbidden zone, projection enabled."""
        geo = GeofenceGeometry()
        geo.add_zone("square", [(0, 0), (2, 0), (2, 2), (0, 2)])

        is_valid, final_x, final_y, msg = geo.validate_goal(
            1.0, 1.0,
            margin=0.0,
            enable_projection=True
        )

        assert is_valid
        assert (final_x != 1.0 or final_y != 1.0)  # Should be different
        assert "projected" in msg.lower()

        # Verify projected point is safe
        assert not geo.is_point_in_forbidden(final_x, final_y, margin=0.0)


class TestZoneInfo:
    """Tests for zone information retrieval."""

    def test_zone_info(self):
        """Get information about zones."""
        geo = GeofenceGeometry()
        geo.add_zone(
            "square",
            [(0, 0), (2, 0), (2, 2), (0, 2)],
            metadata={"type": "test"}
        )

        info = geo.get_zone_info()

        assert "square" in info
        assert abs(info["square"]["area"] - 4.0) < 0.01
        assert info["square"]["metadata"]["type"] == "test"


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_empty_geofence(self):
        """Empty geofence allows all points."""
        geo = GeofenceGeometry()

        assert not geo.is_point_in_forbidden(0, 0, margin=0.0)

        result = geo.check_point(0, 0, margin=0.0)
        assert not result.is_violated

    def test_l_shaped_polygon(self):
        """L-shaped polygon handling."""
        geo = GeofenceGeometry()
        geo.add_zone("l_shape", [
            (0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)
        ])

        # Inside L
        assert geo.is_point_in_forbidden(0.5, 0.5, margin=0.0)

        # In the "notch" of the L (outside)
        assert not geo.is_point_in_forbidden(1.5, 1.5, margin=0.0)

    def test_point_on_boundary(self):
        """Point exactly on boundary."""
        geo = GeofenceGeometry()
        geo.add_zone("square", [(0, 0), (2, 0), (2, 2), (0, 2)])

        # On edge - should be considered inside/touching
        result = geo.check_point(1.0, 0.0, margin=0.0)
        assert result.is_violated or abs(result.distance_to_boundary) < 0.01


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
