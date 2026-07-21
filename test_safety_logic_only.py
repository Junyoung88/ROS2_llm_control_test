#!/usr/bin/env python3
"""
Test safety decision logic WITHOUT simulation
This directly tests the safety method implementations
"""

import sys
sys.path.insert(0, '/home/jim/ros2_motion_planning_tutorials/src/mobile_manipulator_tutorial/src/geofence_policy_enforcer/geofence_policy_enforcer')

from geofence_core import ZoneType, PolicyAction, GeofenceZone
from safety_baselines import (
    SELPSpatialSafety,
    CBFSpatialSafety,
    StaticMarginSafety,
    SSMSpatialSafety,
    SafetyDecision
)
from dataclasses import dataclass
from typing import List
import math

# Zone definitions from geofence.yaml (updated 2026-01-21)
# Zone A: x=[1.5, 2.5], y=[0.5, 1.5] -> center (2.0, 1.0)
# Zone B: x=[1.5, 2.5], y=[2.5, 3.5] -> center (2.0, 3.0)
# Zone C: x=[-4.5, -3.5], y=[0.5, 1.5] -> center (-4.0, 1.0)
# Zone D: x=[3.0, 4.0], y=[-0.5, 0.5] -> center (3.5, 0.0)
ZONE_A = GeofenceZone(
    name="zone_A",
    zone_type=ZoneType.FORBIDDEN,
    priority=10,
    vertices=[(1.5, 0.5), (2.5, 0.5), (2.5, 1.5), (1.5, 1.5)]
)

ZONE_B = GeofenceZone(
    name="zone_B",
    zone_type=ZoneType.FORBIDDEN,
    priority=10,
    vertices=[(1.5, 2.5), (2.5, 2.5), (2.5, 3.5), (1.5, 3.5)]
)

ZONE_C = GeofenceZone(
    name="zone_C",
    zone_type=ZoneType.FORBIDDEN,
    priority=10,
    vertices=[(-4.5, 0.5), (-3.5, 0.5), (-3.5, 1.5), (-4.5, 1.5)]
)

ZONE_D = GeofenceZone(
    name="zone_D",
    zone_type=ZoneType.FORBIDDEN,
    priority=10,
    vertices=[(3.0, -0.5), (4.0, -0.5), (4.0, 0.5), (3.0, 0.5)]
)

SAFE_ZONE = GeofenceZone(
    name="safe_zone",
    zone_type=ZoneType.SAFE,
    priority=1,
    vertices=[(-5.0, -5.0), (5.0, -5.0), (5.0, 5.0), (-5.0, 5.0)]
)

ALL_ZONES = [ZONE_A, ZONE_B, ZONE_C, ZONE_D, SAFE_ZONE]
FORBIDDEN_ZONES = [z for z in ALL_ZONES if z.zone_type == ZoneType.FORBIDDEN]

def distance_to_zone(x: float, y: float, zone: GeofenceZone) -> float:
    """Calculate signed distance to zone boundary (negative = inside)"""
    from shapely.geometry import Point, Polygon
    point = Point(x, y)
    poly = Polygon(zone.vertices)
    if point.within(poly):
        return -point.distance(poly.exterior)
    return point.distance(poly)

def min_distance_to_forbidden(x: float, y: float) -> float:
    """Minimum signed distance to any forbidden zone"""
    return min(distance_to_zone(x, y, z) for z in FORBIDDEN_ZONES)

@dataclass
class TestPoint:
    name: str
    x: float
    y: float
    description: str
    expected: dict  # method -> expected_action

# Test points
TEST_POINTS = [
    TestPoint(
        name="zone_A_center",
        x=2.0, y=1.0,
        description="Center of Zone A (inside forbidden)",
        expected={
            'no_guard': 'ALLOW',
            'selp_proper': 'REJECT',
            'cbf': 'REJECT',
            'ssm': 'REJECT',
            'geofence': 'REJECT',
        }
    ),
    TestPoint(
        name="zone_A_boundary_outside_0.1m",
        x=1.4, y=1.0,
        description="0.1m outside Zone A west edge",
        expected={
            'no_guard': 'ALLOW',
            'selp_proper': 'ALLOW',  # No margin
            'cbf': 'REJECT',         # 0.3m margin
            'ssm': 'REJECT',         # ~0.35m margin
            'geofence': 'REJECT',    # 0.55m margin
        }
    ),
    TestPoint(
        name="zone_A_boundary_outside_0.4m",
        x=1.1, y=1.0,
        description="0.4m outside Zone A west edge",
        expected={
            'no_guard': 'ALLOW',
            'selp_proper': 'ALLOW',  # No margin
            'cbf': 'ALLOW',          # 0.3m margin - just outside
            'ssm': 'REJECT',         # ~0.575m margin at v=0.5 (default)
            'geofence': 'REJECT',    # 0.55m margin - still inside
        }
    ),
    TestPoint(
        name="completely_safe",
        x=0.0, y=0.0,
        description="Origin - far from any forbidden zone",
        expected={
            'no_guard': 'ALLOW',
            'selp_proper': 'ALLOW',
            'cbf': 'ALLOW',
            'ssm': 'ALLOW',
            'geofence': 'ALLOW',
        }
    ),
]

class NoGuardSafety:
    """No safety check - always allow"""
    def evaluate(self, point, zones):
        from safety_baselines import SafetyResult, SafetyDecision
        return SafetyResult(decision=SafetyDecision.ALLOW, reason="no_guard: no check")

def test_all_methods():
    print("=" * 70)
    print("SAFETY METHOD LOGIC TEST (No Simulation)")
    print("=" * 70)
    print("\nMethods and their safety margins:")
    print("  no_guard:     0m (no check)")
    print("  selp_proper:  0m (LTL spec only)")
    print("  cbf:          0.3m")
    print("  ssm:          ~0.575m at v=0.5 (velocity-dependent)")
    print("  geofence:     0.55m")
    print("=" * 70)

    # Initialize safety methods
    zone_names = [z.name for z in FORBIDDEN_ZONES]
    methods = {
        'no_guard': NoGuardSafety(),
        'selp_proper': SELPSpatialSafety(zone_names=zone_names),  # No margin
        'cbf': CBFSpatialSafety(margin=0.3),
        'ssm': SSMSpatialSafety(base_margin=0.2),  # plus velocity-dependent
        'geofence': StaticMarginSafety(margin=0.55),
    }

    all_passed = True

    for tp in TEST_POINTS:
        print(f"\n{'='*70}")
        print(f"TEST: {tp.name}")
        print(f"{'='*70}")
        print(f"Point: ({tp.x}, {tp.y})")
        print(f"Description: {tp.description}")

        dist = min_distance_to_forbidden(tp.x, tp.y)
        print(f"Distance to nearest forbidden zone: {dist:.3f}m")
        print(f"  {'INSIDE' if dist < 0 else 'OUTSIDE'} forbidden zone")
        print()

        print(f"{'Method':<15} {'Expected':<10} {'Actual':<10} {'Result'}")
        print("-" * 50)

        for method_name, method in methods.items():
            expected = tp.expected[method_name]

            # Get actual decision - evaluate(point, zones)
            result = method.evaluate((tp.x, tp.y), FORBIDDEN_ZONES)

            # Extract decision name
            actual = result.decision.name if hasattr(result.decision, 'name') else str(result.decision)

            # PROJECT is acceptable as REJECT variant
            if expected == 'REJECT' and actual == 'PROJECT':
                match = True
            elif expected == 'PROJECT' and actual == 'REJECT':
                match = True
            else:
                match = (actual == expected)

            status = "✓ PASS" if match else "✗ FAIL"
            if not match:
                all_passed = False

            print(f"{method_name:<15} {expected:<10} {actual:<10} {status}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if all_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED")

    # Key insight table
    print("\n" + "=" * 70)
    print("KEY INSIGHT: Safety Margins at Boundary")
    print("=" * 70)
    print("""
Distance from forbidden zone boundary:

  0.0m (inside):    ALL methods reject (except no_guard)
  0.1m (outside):   selp_proper ALLOWS, others REJECT
  0.4m (outside):   cbf ALLOWS, ssm+geofence REJECT
  0.6m+ (outside):  ALL methods ALLOW

Margin ranking (largest to smallest):
  SSM(v=0.5): ~0.575m > geofence: 0.55m > cbf: 0.3m > selp: 0m

This demonstrates how larger margins provide more safety buffer
for execution uncertainty, but may be overly conservative.
""")

    return all_passed

if __name__ == "__main__":
    success = test_all_methods()
    sys.exit(0 if success else 1)
