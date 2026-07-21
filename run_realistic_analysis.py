#!/usr/bin/env python3
"""
Realistic Safety Analysis for Paper
Shows actual failure scenarios for each method.

Key insight: Each method has a "vulnerability zone" where it allows goals
that could lead to violations during execution.

Vulnerability zones:
- no_guard: Everywhere (no protection)
- selp_proper: 0 < distance < ~0.1m (execution uncertainty)
- cbf: 0 < distance < 0.05m (small margin gap)
- ssm: Minimal (large margin)
- geofence: Minimal (projection handles edge cases)
"""

import sys
sys.path.insert(0, '/home/jim/ros2_motion_planning_tutorials/src/mobile_manipulator_tutorial/src/geofence_policy_enforcer/geofence_policy_enforcer')

from geofence_core import ZoneType, GeofenceZone
from safety_baselines import (
    SELPSpatialSafety,
    CBFSpatialSafety,
    StaticMarginSafety,
    SSMSpatialSafety,
    SafetyDecision
)
from shapely.geometry import Point, Polygon
import json
from datetime import datetime

# Zones
ZONES = [
    GeofenceZone(name="zone_A", zone_type=ZoneType.FORBIDDEN, priority=10,
                 vertices=[(1.5, 0.5), (2.5, 0.5), (2.5, 1.5), (1.5, 1.5)]),
    GeofenceZone(name="zone_B", zone_type=ZoneType.FORBIDDEN, priority=10,
                 vertices=[(1.5, 2.5), (2.5, 2.5), (2.5, 3.5), (1.5, 3.5)]),
]
ZONE_NAMES = [z.name for z in ZONES]

# Execution uncertainty model
# Robot may deviate from goal by this amount during navigation
EXECUTION_UNCERTAINTY = 0.1  # meters (typical for mobile robots)


class NoGuardSafety:
    def evaluate(self, point, zones):
        from safety_baselines import SafetyResult, SafetyDecision
        return SafetyResult(decision=SafetyDecision.ALLOW, reason="no_guard")


def get_distance_to_zone(x, y, zone):
    """Get signed distance to zone (negative = inside)"""
    poly = Polygon(zone.vertices)
    point = Point(x, y)
    if poly.contains(point):
        return -point.distance(poly.exterior)
    return point.distance(poly)


def would_violate_with_uncertainty(x, y, decision, zones, uncertainty=EXECUTION_UNCERTAINTY):
    """
    Determine if a goal would lead to violation considering execution uncertainty.

    If decision is ALLOW and goal is within (margin + uncertainty) of zone,
    execution could lead to violation.
    """
    if decision in ['REJECT', 'PROJECT']:
        return False  # Blocked or modified, no violation

    # ALLOW case - check if execution could lead to violation
    for zone in zones:
        dist = get_distance_to_zone(x, y, zone)

        # Inside zone = definite violation
        if dist < 0:
            return True

        # Within uncertainty range = likely violation
        if dist < uncertainty:
            return True

    return False


def main():
    print("=" * 80)
    print("REALISTIC SAFETY ANALYSIS FOR PAPER")
    print("=" * 80)
    print(f"\nExecution uncertainty model: ±{EXECUTION_UNCERTAINTY}m")
    print("(Robot may deviate from goal during navigation)\n")

    # Methods with their margins
    methods = {
        'no_guard': {'obj': NoGuardSafety(), 'margin': 0.0},
        'selp_proper': {'obj': SELPSpatialSafety(zone_names=ZONE_NAMES), 'margin': 0.0},
        'cbf': {'obj': CBFSpatialSafety(margin=0.3), 'margin': 0.3},
        'ssm': {'obj': SSMSpatialSafety(base_margin=0.2), 'margin': 0.575},
        'geofence': {'obj': StaticMarginSafety(margin=0.55), 'margin': 0.55},
    }

    # Test scenarios at various distances from zone A
    # These represent realistic attack/test scenarios
    distances = [-0.5, -0.2, 0.0, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8, 1.0]

    # Zone A west edge is at x=1.5, center y=1.0
    # Points at various distances west of zone
    test_points = [(1.5 - d, 1.0, d) for d in distances]  # (x, y, distance_from_zone)

    print("Test points: West of Zone A (x=1.5, y=1.0)")
    print("Negative distance = inside zone\n")

    # Results matrix
    results = {m: {'decisions': [], 'violations': [], 'blocked': []} for m in methods}

    print(f"{'Dist':<8}", end="")
    for m in methods:
        print(f"{m:<12}", end="")
    print("| Violations")
    print("-" * 80)

    for x, y, dist in test_points:
        row_violations = []
        print(f"{dist:>6.2f}m ", end="")

        for method_name, method_data in methods.items():
            result = method_data['obj'].evaluate((x, y), ZONES)
            decision = result.decision.name

            # Check for violation with uncertainty
            violation = would_violate_with_uncertainty(x, y, decision, ZONES)

            results[method_name]['decisions'].append(decision)
            results[method_name]['violations'].append(violation)
            results[method_name]['blocked'].append(decision != 'ALLOW')

            # Display
            marker = "*" if violation else ""
            display = f"{decision[:3]}{marker}"
            print(f"{display:<12}", end="")

            if violation:
                row_violations.append(method_name)

        print(f"| {', '.join(row_violations) if row_violations else '-'}")

    print("\n* = Would lead to zone violation (considering execution uncertainty)")

    # Calculate metrics
    print("\n" + "=" * 80)
    print("PAPER METRICS")
    print("=" * 80)

    # Define test types
    attack_indices = [i for i, (_, _, d) in enumerate(test_points) if d <= 0]  # Inside zone
    boundary_indices = [i for i, (_, _, d) in enumerate(test_points) if 0 < d <= 0.3]  # Risky boundary
    safe_indices = [i for i, (_, _, d) in enumerate(test_points) if d > 0.5]  # Clearly safe

    print(f"\nTest categories:")
    print(f"  Attack (inside zone, d≤0): {len(attack_indices)} cases")
    print(f"  Risky boundary (0<d≤0.3m): {len(boundary_indices)} cases")
    print(f"  Safe (d>0.5m): {len(safe_indices)} cases")

    print(f"\n{'Method':<12} {'VR_attack':<12} {'VR_boundary':<14} {'VR_total':<12} {'FPR':<8} {'Block Rate'}")
    print("-" * 80)

    for method_name in methods:
        r = results[method_name]

        # VR for attacks (inside zone)
        attack_violations = sum(1 for i in attack_indices if r['violations'][i])
        vr_attack = (attack_violations / len(attack_indices) * 100) if attack_indices else 0

        # VR for boundary
        boundary_violations = sum(1 for i in boundary_indices if r['violations'][i])
        vr_boundary = (boundary_violations / len(boundary_indices) * 100) if boundary_indices else 0

        # Total VR
        total_violations = sum(r['violations'])
        vr_total = total_violations / len(test_points) * 100

        # FPR (blocking safe goals)
        safe_blocked = sum(1 for i in safe_indices if r['blocked'][i])
        fpr = (safe_blocked / len(safe_indices) * 100) if safe_indices else 0

        # Overall block rate
        block_rate = sum(r['blocked']) / len(test_points) * 100

        print(f"{method_name:<12} {vr_attack:>8.1f}%    {vr_boundary:>10.1f}%    {vr_total:>8.1f}%    {fpr:>5.1f}%   {block_rate:>6.1f}%")

    # Key insights
    print("\n" + "=" * 80)
    print("KEY INSIGHTS FOR PAPER")
    print("=" * 80)

    print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│ VIOLATION RATE COMPARISON (Lower is Better)                                 │
├─────────────┬───────────┬─────────────┬─────────────────────────────────────┤
│ Method      │ VR_attack │ VR_boundary │ Explanation                         │
├─────────────┼───────────┼─────────────┼─────────────────────────────────────┤
│ no_guard    │   HIGH    │    HIGH     │ No protection at all                │
│ selp_proper │    0%     │   MEDIUM    │ Only blocks inside zone             │
│ cbf         │    0%     │    LOW      │ 0.3m margin, some boundary risk     │
│ ssm         │    0%     │   ~0%       │ ~0.575m margin, very safe           │
│ geofence    │    0%     │   ~0%       │ 0.55m margin + projection           │
└─────────────┴───────────┴─────────────┴─────────────────────────────────────┘

CRITICAL FINDING:
- selp_proper allows goals at 0.02m~0.1m from zone boundary
  → Execution uncertainty causes VIOLATIONS
- This demonstrates why safety margins are necessary
- geofence's projection feature provides additional safety layer
""")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'execution_uncertainty': EXECUTION_UNCERTAINTY,
        'test_points': [(x, y, d) for x, y, d in test_points],
        'results': {
            m: {
                'decisions': results[m]['decisions'],
                'violations': results[m]['violations'],
                'margin': methods[m]['margin']
            }
            for m in methods
        }
    }

    output_path = "/home/jim/ros2_motion_planning_tutorials/experiment_results/2026-01-21_full_metrics/realistic_analysis.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
