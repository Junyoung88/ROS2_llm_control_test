#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Safety Baselines for Spatial Navigation Safety

This module implements various safety methods for robot navigation,
providing fair comparisons for the Geofence method.

Methods implemented:
1. SELP (ICRA 2025) - LTL automaton with zone membership propositions
2. CBF (Ames et al., 2017) - Control Barrier Functions
3. Static Margin (IROS 2014) - Fixed safety margin (costmap inflation)
4. SSM (ISO 15066) - Speed and Separation Monitoring

References:
- SELP: Wu et al., "SELP: Generating Safe and Efficient Task Plans for Robot
  Agents with Large Language Models", ICRA 2025
- CBF: Ames et al., "Control Barrier Function Based Quadratic Programs for
  Safety Critical Systems", TAC 2017
- Static Margin: Lu et al., "Layered Costmaps for Context-Sensitive Navigation",
  IROS 2014
- SSM: ISO/TS 15066:2016, "Robots and robotic devices - Collaborative robots"
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set
from enum import Enum, auto


class SafetyDecision(Enum):
    """Safety decision outcomes."""
    ALLOW = "allow"
    REJECT = "reject"
    PROJECT = "project"


@dataclass
class SafetyResult:
    """Result of a safety evaluation."""
    decision: SafetyDecision
    reason: str
    distance_to_boundary: float = float('inf')
    barrier_value: float = float('inf')  # For CBF
    automaton_state: str = ""  # For SELP
    margin_used: float = 0.0
    extra_info: Dict = field(default_factory=dict)


# =============================================================================
# SELP: LTL Automaton-based Safety (ICRA 2025)
# =============================================================================

class AutomatonState(Enum):
    """Büchi automaton states."""
    ACCEPTING = auto()      # Safe state
    NON_ACCEPTING = auto()  # Transitional state
    REJECTING = auto()      # Violation state (sink)


class SELPSpatialSafety:
    """
    SELP adapted for spatial safety.

    Original SELP uses LTL automata for task-level constraints like
    "visit A before B". We adapt it for spatial safety using zone
    membership propositions: G(!in_zone_A) ∧ G(!in_zone_B) ∧ ...

    The automaton monitors zone membership and transitions to a
    rejecting state if any forbidden zone is entered.

    Reference:
        Wu et al., "SELP: Generating Safe and Efficient Task Plans for Robot
        Agents with Large Language Models", ICRA 2025
        https://arxiv.org/abs/2409.19471
    """

    def __init__(self, zone_names: List[str]):
        """
        Initialize SELP with zone names.

        Args:
            zone_names: List of forbidden zone names (e.g., ["zone_A", "zone_B"])
        """
        self.zone_names = zone_names
        self._state = AutomatonState.ACCEPTING
        self._violation_history: List[str] = []

        # LTL formula: G(!in_zone_A) ∧ G(!in_zone_B) ∧ ...
        # This means "globally, never be in any forbidden zone"
        self._constraints = [f"G(!in_{name})" for name in zone_names]

    def reset(self):
        """Reset automaton to initial accepting state."""
        self._state = AutomatonState.ACCEPTING
        self._violation_history.clear()

    def get_propositions(self, point: Tuple[float, float],
                         zones: List['GeofenceZone']) -> Dict[str, bool]:
        """
        Get zone membership propositions for a point.

        Args:
            point: (x, y) coordinates
            zones: List of GeofenceZone objects

        Returns:
            Dictionary of propositions: {"in_zone_A": True/False, ...}
        """
        props = {}
        for zone in zones:
            prop_name = f"in_{zone.name}"
            props[prop_name] = self._point_in_zone(point, zone)
        return props

    def evaluate(self, point: Tuple[float, float],
                 zones: List['GeofenceZone']) -> SafetyResult:
        """
        Evaluate if a point would cause automaton to reject.

        The Büchi automaton transitions based on zone membership:
        - ACCEPTING -> ACCEPTING: if !in_any_zone (safe)
        - ACCEPTING -> REJECTING: if in_any_zone (violation)
        - REJECTING -> REJECTING: sink state (absorbing)

        Args:
            point: Goal point (x, y)
            zones: List of forbidden zones

        Returns:
            SafetyResult with decision and automaton state
        """
        # Get propositions for the point
        props = self.get_propositions(point, zones)

        # Check if any forbidden zone proposition is true
        violated_zones = []
        for zone in zones:
            prop_name = f"in_{zone.name}"
            if props.get(prop_name, False):
                violated_zones.append(zone.name)

        # Automaton transition
        if violated_zones:
            self._state = AutomatonState.REJECTING
            self._violation_history.extend(violated_zones)

            return SafetyResult(
                decision=SafetyDecision.REJECT,
                reason=f"SELP automaton rejects: proposition in_{violated_zones[0]} would become true",
                distance_to_boundary=0.0,
                automaton_state=self._state.name,
                extra_info={
                    "violated_zones": violated_zones,
                    "propositions": props,
                    "ltl_constraints": self._constraints
                }
            )

        # No violation - automaton accepts
        min_dist = self._get_min_distance(point, zones)

        return SafetyResult(
            decision=SafetyDecision.ALLOW,
            reason="SELP automaton accepts: no constraint violation",
            distance_to_boundary=min_dist,
            automaton_state=self._state.name,
            extra_info={
                "propositions": props,
                "ltl_constraints": self._constraints
            }
        )

    def _point_in_zone(self, point: Tuple[float, float],
                       zone: 'GeofenceZone') -> bool:
        """Check if point is inside zone using ray casting."""
        x, y = point
        vertices = zone.vertices
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

    def _get_min_distance(self, point: Tuple[float, float],
                          zones: List['GeofenceZone']) -> float:
        """Get minimum distance to any zone boundary."""
        min_dist = float('inf')
        for zone in zones:
            dist = self._distance_to_polygon(point, zone.vertices)
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def _distance_to_polygon(self, point: Tuple[float, float],
                             vertices: List[Tuple[float, float]]) -> float:
        """Calculate minimum distance from point to polygon boundary."""
        if self._point_in_polygon_vertices(point, vertices):
            return 0.0

        min_dist = float('inf')
        n = len(vertices)

        for i in range(n):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % n]
            dist = self._point_to_segment_distance(point, p1, p2)
            if dist < min_dist:
                min_dist = dist

        return min_dist

    def _point_in_polygon_vertices(self, point: Tuple[float, float],
                                    vertices: List[Tuple[float, float]]) -> bool:
        """Ray casting algorithm for point-in-polygon."""
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


# =============================================================================
# CBF: Control Barrier Functions (Ames et al., 2017)
# =============================================================================

class CBFSpatialSafety:
    """
    Control Barrier Function for spatial safety.

    CBF ensures forward invariance of a safe set by maintaining h(x) >= 0,
    where h(x) is the barrier function. For zone avoidance:

        h(x) = d(x, Zone) - δ

    where d(x, Zone) is the signed distance to the zone boundary and δ is
    a safety margin.

    The CBF condition for safety:
        ḣ(x) + α(h(x)) >= 0

    where α is a class-K function (we use α(h) = γ * h for simplicity).

    Reference:
        Ames et al., "Control Barrier Function Based Quadratic Programs for
        Safety Critical Systems", IEEE TAC 2017
        http://ames.caltech.edu/ames2017cbf.pdf
    """

    def __init__(self, gamma: float = 1.0, margin: float = 0.3):
        """
        Initialize CBF safety checker.

        Args:
            gamma: Class-K function parameter (α(h) = γ * h)
            margin: Minimum safety margin δ (meters)
        """
        self.gamma = gamma
        self.margin = margin

    def compute_barrier(self, point: Tuple[float, float],
                        zones: List['GeofenceZone']) -> Tuple[float, str]:
        """
        Compute barrier function value h(x) = min_i(d(x, Zone_i) - δ).

        Args:
            point: Current or goal position (x, y)
            zones: List of forbidden zones

        Returns:
            Tuple of (barrier_value, closest_zone_name)
        """
        min_h = float('inf')
        closest_zone = ""

        for zone in zones:
            # Signed distance: negative inside, positive outside
            dist = self._signed_distance_to_zone(point, zone)
            h = dist - self.margin

            if h < min_h:
                min_h = h
                closest_zone = zone.name

        return min_h, closest_zone

    def evaluate(self, point: Tuple[float, float],
                 zones: List['GeofenceZone'],
                 current_pos: Optional[Tuple[float, float]] = None,
                 velocity: Optional[Tuple[float, float]] = None) -> SafetyResult:
        """
        Evaluate safety using Control Barrier Function.

        For goal evaluation without dynamics:
            - ALLOW if h(goal) >= 0 (goal is in safe set)
            - REJECT if h(goal) < 0 (goal violates barrier)

        Args:
            point: Goal point (x, y)
            zones: List of forbidden zones
            current_pos: Current robot position (optional, for dynamics)
            velocity: Current velocity (optional, for dynamics)

        Returns:
            SafetyResult with barrier value and decision
        """
        h, closest_zone = self.compute_barrier(point, zones)

        # Basic CBF check: h(x) >= 0 means safe
        if h >= 0:
            return SafetyResult(
                decision=SafetyDecision.ALLOW,
                reason=f"CBF safe: h(x) = {h:.3f} >= 0",
                distance_to_boundary=h + self.margin,
                barrier_value=h,
                margin_used=self.margin,
                extra_info={
                    "closest_zone": closest_zone,
                    "gamma": self.gamma,
                    "cbf_condition": "h(x) >= 0 satisfied"
                }
            )
        else:
            return SafetyResult(
                decision=SafetyDecision.REJECT,
                reason=f"CBF violation: h(x) = {h:.3f} < 0 (within {self.margin}m margin of {closest_zone})",
                distance_to_boundary=h + self.margin,
                barrier_value=h,
                margin_used=self.margin,
                extra_info={
                    "closest_zone": closest_zone,
                    "gamma": self.gamma,
                    "cbf_condition": "h(x) >= 0 violated"
                }
            )

    def compute_barrier_derivative(self, point: Tuple[float, float],
                                    velocity: Tuple[float, float],
                                    zones: List['GeofenceZone']) -> float:
        """
        Compute time derivative of barrier function ḣ(x).

        ḣ(x) = ∇h(x) · ẋ

        For circular/convex zones, ∇h points outward from zone center.
        """
        h, closest_zone = self.compute_barrier(point, zones)

        # Find gradient direction (pointing away from zone)
        for zone in zones:
            if zone.name == closest_zone:
                grad = self._compute_gradient(point, zone)
                # ḣ = grad · velocity
                h_dot = grad[0] * velocity[0] + grad[1] * velocity[1]
                return h_dot

        return 0.0

    def _signed_distance_to_zone(self, point: Tuple[float, float],
                                  zone: 'GeofenceZone') -> float:
        """
        Compute signed distance to zone boundary.
        Negative inside, positive outside.
        """
        vertices = zone.vertices

        # Check if inside
        if self._point_in_polygon(point, vertices):
            # Inside: return negative distance to boundary
            return -self._distance_to_boundary(point, vertices)
        else:
            # Outside: return positive distance to boundary
            return self._distance_to_boundary(point, vertices)

    def _compute_gradient(self, point: Tuple[float, float],
                          zone: 'GeofenceZone') -> Tuple[float, float]:
        """
        Compute gradient of distance function (unit vector pointing away from zone).
        """
        # Use centroid approximation for gradient direction
        cx = sum(v[0] for v in zone.vertices) / len(zone.vertices)
        cy = sum(v[1] for v in zone.vertices) / len(zone.vertices)

        dx = point[0] - cx
        dy = point[1] - cy
        dist = math.sqrt(dx*dx + dy*dy)

        if dist < 1e-6:
            return (1.0, 0.0)  # Arbitrary direction

        return (dx / dist, dy / dist)

    def _point_in_polygon(self, point: Tuple[float, float],
                          vertices: List[Tuple[float, float]]) -> bool:
        """Ray casting algorithm."""
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

    def _distance_to_boundary(self, point: Tuple[float, float],
                               vertices: List[Tuple[float, float]]) -> float:
        """Calculate minimum distance to polygon boundary."""
        min_dist = float('inf')
        n = len(vertices)

        for i in range(n):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % n]
            dist = self._point_to_segment_distance(point, p1, p2)
            if dist < min_dist:
                min_dist = dist

        return min_dist

    def _point_to_segment_distance(self, point: Tuple[float, float],
                                    seg_start: Tuple[float, float],
                                    seg_end: Tuple[float, float]) -> float:
        """Distance from point to line segment."""
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


# =============================================================================
# Static Margin (Costmap Inflation - IROS 2014)
# =============================================================================

class StaticMarginSafety:
    """
    Static safety margin (costmap inflation approach).

    Uses a fixed safety margin around obstacles, similar to ROS Nav2's
    costmap inflation layer. This is the industry-standard baseline.

    Reference:
        Lu et al., "Layered Costmaps for Context-Sensitive Navigation",
        IROS 2014
    """

    def __init__(self, margin: float = 0.5):
        """
        Initialize with fixed safety margin.

        Args:
            margin: Fixed safety margin in meters (inflation_radius)
        """
        self.margin = margin

    def evaluate(self, point: Tuple[float, float],
                 zones: List['GeofenceZone']) -> SafetyResult:
        """
        Evaluate if point is safe with static margin.

        REJECT if distance to any zone < margin
        ALLOW otherwise

        Args:
            point: Goal point (x, y)
            zones: List of forbidden zones

        Returns:
            SafetyResult
        """
        min_dist = float('inf')
        closest_zone = ""
        inside_zone = False

        for zone in zones:
            # Check if inside
            if self._point_in_polygon(point, zone.vertices):
                return SafetyResult(
                    decision=SafetyDecision.REJECT,
                    reason=f"Static margin: point inside forbidden zone '{zone.name}'",
                    distance_to_boundary=0.0,
                    margin_used=self.margin,
                    extra_info={"method": "static_margin", "inside_zone": True}
                )

            # Get distance to boundary
            dist = self._distance_to_boundary(point, zone.vertices)
            if dist < min_dist:
                min_dist = dist
                closest_zone = zone.name

        # Check if within margin
        if min_dist < self.margin:
            return SafetyResult(
                decision=SafetyDecision.REJECT,
                reason=f"Static margin: point within {self.margin}m of '{closest_zone}' (dist={min_dist:.3f}m)",
                distance_to_boundary=min_dist,
                margin_used=self.margin,
                extra_info={"method": "static_margin", "closest_zone": closest_zone}
            )

        return SafetyResult(
            decision=SafetyDecision.ALLOW,
            reason=f"Static margin: point safe (dist={min_dist:.3f}m > margin={self.margin}m)",
            distance_to_boundary=min_dist,
            margin_used=self.margin,
            extra_info={"method": "static_margin", "closest_zone": closest_zone}
        )

    def _point_in_polygon(self, point: Tuple[float, float],
                          vertices: List[Tuple[float, float]]) -> bool:
        """Ray casting."""
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

    def _distance_to_boundary(self, point: Tuple[float, float],
                               vertices: List[Tuple[float, float]]) -> float:
        """Distance to polygon boundary."""
        min_dist = float('inf')
        n = len(vertices)

        for i in range(n):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % n]

            px, py = point
            x1, y1 = p1
            x2, y2 = p2

            dx = x2 - x1
            dy = y2 - y1

            if dx == 0 and dy == 0:
                dist = math.sqrt((px - x1)**2 + (py - y1)**2)
            else:
                t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx*dx + dy*dy)))
                nearest_x = x1 + t * dx
                nearest_y = y1 + t * dy
                dist = math.sqrt((px - nearest_x)**2 + (py - nearest_y)**2)

            if dist < min_dist:
                min_dist = dist

        return min_dist


# =============================================================================
# SSM: Speed and Separation Monitoring (ISO 15066)
# =============================================================================

class SSMSpatialSafety:
    """
    Speed and Separation Monitoring for spatial safety.

    Implements velocity-dependent safety distance based on ISO/TS 15066.
    The protective separation distance is:

        S = S_h + S_r + S_s + C + Z_d + Z_r

    Simplified for mobile robots:
        S = v_r * T_r + v_r * T_s + B + C

    where:
        v_r: Robot velocity
        T_r: Reaction time
        T_s: Robot stopping time
        B: Braking distance = v_r² / (2 * a_max)
        C: Intrusion distance (uncertainty)

    Reference:
        ISO/TS 15066:2016, "Robots and robotic devices - Collaborative robots"
        https://pmc.ncbi.nlm.nih.gov/articles/PMC5117641/
    """

    def __init__(self,
                 reaction_time: float = 0.1,
                 stopping_time: float = 0.2,
                 max_deceleration: float = 1.0,
                 intrusion_distance: float = 0.1,
                 base_margin: float = 0.2):
        """
        Initialize SSM safety checker.

        Args:
            reaction_time: System reaction time T_r (seconds)
            stopping_time: Time to initiate stop T_s (seconds)
            max_deceleration: Maximum deceleration a_max (m/s²)
            intrusion_distance: Uncertainty/intrusion distance C (meters)
            base_margin: Minimum margin when stationary (meters)
        """
        self.reaction_time = reaction_time
        self.stopping_time = stopping_time
        self.max_deceleration = max_deceleration
        self.intrusion_distance = intrusion_distance
        self.base_margin = base_margin

    def compute_protective_distance(self, velocity: float) -> float:
        """
        Compute protective separation distance per ISO 15066.

        S = v * (T_r + T_s) + v² / (2 * a_max) + C + base

        Args:
            velocity: Current robot velocity (m/s)

        Returns:
            Required protective separation distance (meters)
        """
        # Reaction and stopping distance
        reaction_dist = velocity * (self.reaction_time + self.stopping_time)

        # Braking distance
        braking_dist = (velocity ** 2) / (2 * self.max_deceleration) if self.max_deceleration > 0 else 0

        # Total protective distance
        S = reaction_dist + braking_dist + self.intrusion_distance + self.base_margin

        return S

    def evaluate(self, point: Tuple[float, float],
                 zones: List['GeofenceZone'],
                 velocity: float = 0.5) -> SafetyResult:
        """
        Evaluate safety with velocity-dependent margin.

        Args:
            point: Goal point (x, y)
            zones: List of forbidden zones
            velocity: Current/expected robot velocity (m/s)

        Returns:
            SafetyResult
        """
        # Compute velocity-dependent margin
        margin = self.compute_protective_distance(velocity)

        min_dist = float('inf')
        closest_zone = ""

        for zone in zones:
            # Check if inside
            if self._point_in_polygon(point, zone.vertices):
                return SafetyResult(
                    decision=SafetyDecision.REJECT,
                    reason=f"SSM: point inside forbidden zone '{zone.name}'",
                    distance_to_boundary=0.0,
                    margin_used=margin,
                    extra_info={
                        "method": "ssm",
                        "velocity": velocity,
                        "protective_distance": margin,
                        "inside_zone": True
                    }
                )

            dist = self._distance_to_boundary(point, zone.vertices)
            if dist < min_dist:
                min_dist = dist
                closest_zone = zone.name

        # Check against velocity-dependent margin
        if min_dist < margin:
            return SafetyResult(
                decision=SafetyDecision.REJECT,
                reason=f"SSM: point within protective distance of '{closest_zone}' "
                       f"(dist={min_dist:.3f}m < S={margin:.3f}m at v={velocity:.2f}m/s)",
                distance_to_boundary=min_dist,
                margin_used=margin,
                extra_info={
                    "method": "ssm",
                    "velocity": velocity,
                    "protective_distance": margin,
                    "closest_zone": closest_zone,
                    "components": {
                        "reaction_dist": velocity * (self.reaction_time + self.stopping_time),
                        "braking_dist": (velocity ** 2) / (2 * self.max_deceleration),
                        "intrusion_dist": self.intrusion_distance,
                        "base_margin": self.base_margin
                    }
                }
            )

        return SafetyResult(
            decision=SafetyDecision.ALLOW,
            reason=f"SSM: point safe (dist={min_dist:.3f}m > S={margin:.3f}m at v={velocity:.2f}m/s)",
            distance_to_boundary=min_dist,
            margin_used=margin,
            extra_info={
                "method": "ssm",
                "velocity": velocity,
                "protective_distance": margin,
                "closest_zone": closest_zone
            }
        )

    def _point_in_polygon(self, point: Tuple[float, float],
                          vertices: List[Tuple[float, float]]) -> bool:
        """Ray casting."""
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

    def _distance_to_boundary(self, point: Tuple[float, float],
                               vertices: List[Tuple[float, float]]) -> float:
        """Distance to polygon boundary."""
        min_dist = float('inf')
        n = len(vertices)

        for i in range(n):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % n]

            px, py = point
            x1, y1 = p1
            x2, y2 = p2

            dx = x2 - x1
            dy = y2 - y1

            if dx == 0 and dy == 0:
                dist = math.sqrt((px - x1)**2 + (py - y1)**2)
            else:
                t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx*dx + dy*dy)))
                nearest_x = x1 + t * dx
                nearest_y = y1 + t * dy
                dist = math.sqrt((px - nearest_x)**2 + (py - nearest_y)**2)

            if dist < min_dist:
                min_dist = dist

        return min_dist


# =============================================================================
# Factory for creating safety checkers
# =============================================================================

class SafetyMethodFactory:
    """Factory for creating safety method instances."""

    @staticmethod
    def create(method_name: str, zones: List['GeofenceZone'], **kwargs):
        """
        Create a safety method instance.

        Args:
            method_name: One of "selp", "cbf", "static_margin", "ssm"
            zones: List of GeofenceZone objects
            **kwargs: Method-specific parameters

        Returns:
            Safety checker instance
        """
        zone_names = [z.name for z in zones]

        if method_name == "selp":
            return SELPSpatialSafety(zone_names)

        elif method_name == "cbf":
            gamma = kwargs.get("gamma", 1.0)
            margin = kwargs.get("margin", 0.3)
            return CBFSpatialSafety(gamma=gamma, margin=margin)

        elif method_name == "static_margin":
            margin = kwargs.get("margin", 0.5)
            return StaticMarginSafety(margin=margin)

        elif method_name == "ssm":
            return SSMSpatialSafety(
                reaction_time=kwargs.get("reaction_time", 0.1),
                stopping_time=kwargs.get("stopping_time", 0.2),
                max_deceleration=kwargs.get("max_deceleration", 1.0),
                intrusion_distance=kwargs.get("intrusion_distance", 0.1),
                base_margin=kwargs.get("base_margin", 0.2)
            )

        else:
            raise ValueError(f"Unknown safety method: {method_name}")
