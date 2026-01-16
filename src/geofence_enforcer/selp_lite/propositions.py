"""
Propositions Module - Shared evaluation functions for forbidden zones and safety margins

This module provides common proposition evaluation functions used by both:
- Geofence baseline (projection/blocking)
- SELP-lite (action masking in constrained decoding)

The separation ensures fair comparison between the two approaches.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass, field
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
import logging

logger = logging.getLogger(__name__)


@dataclass
class Zone:
    """A forbidden or special zone in the environment"""
    name: str
    vertices: List[Tuple[float, float]]
    zone_type: str = "forbidden"  # forbidden, warning, goal, etc.
    polygon: Polygon = field(init=False)

    def __post_init__(self):
        self.polygon = Polygon(self.vertices)
        if not self.polygon.is_valid:
            self.polygon = self.polygon.buffer(0)  # Fix invalid polygons


@dataclass
class PropositionState:
    """Current state of all propositions"""
    robot_position: Tuple[float, float]
    in_zones: Dict[str, bool] = field(default_factory=dict)  # zone_name -> is_inside
    zone_distances: Dict[str, float] = field(default_factory=dict)  # zone_name -> distance
    visited_zones: Set[str] = field(default_factory=set)  # zones visited so far
    margin_violated: Dict[str, bool] = field(default_factory=dict)  # zone_name -> margin violation


class PropositionEvaluator:
    """
    Evaluates propositions for both Geofence and SELP-lite systems.

    Propositions are atomic boolean conditions that can be used in LTL formulas:
    - in_zone_X: robot is inside zone X
    - near_zone_X: robot is within safety margin of zone X
    - visited_X: robot has visited zone X at some point
    - at_goal: robot has reached the goal

    This evaluator is used by:
    - Geofence: to determine if projection/blocking is needed
    - SELP-lite: to check if LTL constraints are satisfied
    """

    def __init__(self, grid_resolution: float = 1.0):
        """
        Args:
            grid_resolution: Size of each grid cell (for discrete grid worlds)
        """
        self.zones: Dict[str, Zone] = {}
        self.safety_margin: float = 0.3  # Default safety margin
        self.grid_resolution = grid_resolution
        self.goal_position: Optional[Tuple[float, float]] = None
        self.goal_threshold: float = 0.5  # Distance to consider goal reached

    def add_zone(self, name: str, vertices: List[Tuple[float, float]],
                 zone_type: str = "forbidden") -> None:
        """Add a zone to the evaluator"""
        self.zones[name] = Zone(name, vertices, zone_type)
        logger.debug(f"Added zone '{name}' with {len(vertices)} vertices")

    def set_safety_margin(self, margin: float) -> None:
        """Set the safety margin for all zones"""
        self.safety_margin = margin

    def set_goal(self, position: Tuple[float, float], threshold: float = 0.5) -> None:
        """Set the goal position"""
        self.goal_position = position
        self.goal_threshold = threshold

    def evaluate(self, position: Tuple[float, float],
                 visited: Optional[Set[str]] = None) -> PropositionState:
        """
        Evaluate all propositions for a given position.

        Args:
            position: Current robot position (x, y)
            visited: Set of previously visited zone names

        Returns:
            PropositionState with all evaluated propositions
        """
        state = PropositionState(
            robot_position=position,
            visited_zones=visited.copy() if visited else set()
        )

        point = Point(position)

        for name, zone in self.zones.items():
            # Check if inside zone
            is_inside = zone.polygon.contains(point)
            state.in_zones[name] = is_inside

            # Calculate distance to zone boundary
            if is_inside:
                distance = -zone.polygon.exterior.distance(point)  # Negative = inside
            else:
                distance = zone.polygon.exterior.distance(point)
            state.zone_distances[name] = distance

            # Check margin violation
            state.margin_violated[name] = distance < self.safety_margin

            # Update visited zones
            if is_inside:
                state.visited_zones.add(name)

        return state

    def is_in_forbidden_zone(self, position: Tuple[float, float]) -> bool:
        """Check if position is inside any forbidden zone"""
        point = Point(position)
        for zone in self.zones.values():
            if zone.zone_type == "forbidden" and zone.polygon.contains(point):
                return True
        return False

    def is_within_margin(self, position: Tuple[float, float],
                         margin: Optional[float] = None) -> bool:
        """Check if position is within safety margin of any forbidden zone"""
        margin = margin if margin is not None else self.safety_margin
        point = Point(position)

        for zone in self.zones.values():
            if zone.zone_type == "forbidden":
                distance = zone.polygon.exterior.distance(point)
                if not zone.polygon.contains(point) and distance < margin:
                    return True
                elif zone.polygon.contains(point):
                    return True
        return False

    def get_distance_to_nearest_forbidden(self, position: Tuple[float, float]) -> float:
        """Get distance to nearest forbidden zone boundary"""
        point = Point(position)
        min_distance = float('inf')

        for zone in self.zones.values():
            if zone.zone_type == "forbidden":
                if zone.polygon.contains(point):
                    return -zone.polygon.exterior.distance(point)
                distance = zone.polygon.exterior.distance(point)
                min_distance = min(min_distance, distance)

        return min_distance

    def is_at_goal(self, position: Tuple[float, float]) -> bool:
        """Check if robot has reached the goal"""
        if self.goal_position is None:
            return False
        dx = position[0] - self.goal_position[0]
        dy = position[1] - self.goal_position[1]
        return (dx**2 + dy**2)**0.5 <= self.goal_threshold

    def get_proposition_dict(self, state: PropositionState) -> Dict[str, bool]:
        """
        Convert proposition state to a dictionary of named propositions.
        This format is compatible with LTL automaton evaluation.

        Returns:
            Dictionary mapping proposition names to boolean values
            Example: {"in_storage": True, "visited_A": False, "at_goal": False}
        """
        props = {}

        # Zone-related propositions
        for zone_name in self.zones:
            # in_<zone>: currently inside zone
            props[f"in_{zone_name}"] = state.in_zones.get(zone_name, False)

            # near_<zone>: within safety margin
            props[f"near_{zone_name}"] = state.margin_violated.get(zone_name, False)

            # visited_<zone>: has been inside zone at some point
            props[f"visited_{zone_name}"] = zone_name in state.visited_zones

        # Goal proposition
        props["at_goal"] = self.is_at_goal(state.robot_position)

        return props

    def get_all_proposition_names(self) -> List[str]:
        """Get list of all proposition names"""
        names = []
        for zone_name in self.zones:
            names.extend([
                f"in_{zone_name}",
                f"near_{zone_name}",
                f"visited_{zone_name}"
            ])
        names.append("at_goal")
        return names


# =============================================================================
# Grid World Specific Utilities
# =============================================================================

def grid_to_continuous(grid_x: int, grid_y: int,
                       cell_size: float = 1.0) -> Tuple[float, float]:
    """Convert grid coordinates to continuous coordinates (cell center)"""
    return (grid_x * cell_size + cell_size / 2,
            grid_y * cell_size + cell_size / 2)


def continuous_to_grid(x: float, y: float,
                       cell_size: float = 1.0) -> Tuple[int, int]:
    """Convert continuous coordinates to grid coordinates"""
    return (int(x // cell_size), int(y // cell_size))


def create_grid_zone(grid_cells: List[Tuple[int, int]],
                     cell_size: float = 1.0) -> List[Tuple[float, float]]:
    """
    Create zone vertices from a list of grid cells.
    Returns the convex hull of the cells.
    """
    from shapely.geometry import MultiPoint
    from shapely.ops import unary_union

    # Create polygons for each cell
    cell_polygons = []
    for gx, gy in grid_cells:
        x, y = gx * cell_size, gy * cell_size
        cell_poly = Polygon([
            (x, y), (x + cell_size, y),
            (x + cell_size, y + cell_size), (x, y + cell_size)
        ])
        cell_polygons.append(cell_poly)

    # Union all cells
    merged = unary_union(cell_polygons)

    # Return exterior coordinates
    if hasattr(merged, 'exterior'):
        return list(merged.exterior.coords)[:-1]  # Remove duplicate last point
    else:
        # Multiple separate regions - return convex hull
        return list(merged.convex_hull.exterior.coords)[:-1]


# =============================================================================
# Factory Environment Preset (for comparison with existing experiments)
# =============================================================================

def create_factory_propositions(grid_size: int = 20,
                                cell_size: float = 1.0) -> PropositionEvaluator:
    """
    Create proposition evaluator with factory environment zones.
    Compatible with factory_geofence.yaml configuration.
    """
    evaluator = PropositionEvaluator(grid_resolution=cell_size)

    # Zone 1: Robot Arm Cell (grid cells 2-5, 7-9)
    evaluator.add_zone("robot_arm_cell", [
        (2.0, 7.0), (6.0, 7.0), (6.0, 10.0), (2.0, 10.0)
    ], zone_type="forbidden")

    # Zone 2: Welding Cell
    evaluator.add_zone("welding_cell", [
        (4.0, 11.0), (10.0, 11.0), (10.0, 14.0), (4.0, 14.0)
    ], zone_type="forbidden")

    # Zone 3: High-Temperature Furnace
    evaluator.add_zone("high_temp_furnace", [
        (8.0, 7.0), (11.0, 7.0), (11.0, 10.0), (8.0, 10.0)
    ], zone_type="forbidden")

    # Zone 4: Assembly Line
    evaluator.add_zone("assembly_line", [
        (2.0, 1.0), (7.0, 1.0), (7.0, 3.5), (2.0, 3.5)
    ], zone_type="forbidden")

    # Zone 5: QC Station
    evaluator.add_zone("qc_station", [
        (12.0, 1.0), (17.0, 1.0), (17.0, 4.0), (12.0, 4.0)
    ], zone_type="forbidden")

    # Zone 6: Storage Area
    evaluator.add_zone("storage_racks", [
        (14.0, 7.0), (18.0, 7.0), (18.0, 11.0), (14.0, 11.0)
    ], zone_type="forbidden")

    # Zone 7: Pedestrian Corridor
    evaluator.add_zone("pedestrian_corridor", [
        (0.0, 4.5), (20.0, 4.5), (20.0, 5.5), (0.0, 5.5)
    ], zone_type="forbidden")

    evaluator.set_safety_margin(0.3)  # 30cm margin

    return evaluator


def create_simple_grid_propositions(grid_size: int = 10,
                                    cell_size: float = 1.0) -> PropositionEvaluator:
    """
    Create a simpler grid environment for basic testing.
    """
    evaluator = PropositionEvaluator(grid_resolution=cell_size)

    # Simple forbidden zone in the middle
    evaluator.add_zone("obstacle_A", [
        (3.0, 3.0), (5.0, 3.0), (5.0, 5.0), (3.0, 5.0)
    ], zone_type="forbidden")

    # Another zone in corner
    evaluator.add_zone("obstacle_B", [
        (7.0, 7.0), (9.0, 7.0), (9.0, 9.0), (7.0, 9.0)
    ], zone_type="forbidden")

    evaluator.set_safety_margin(0.3)

    return evaluator
