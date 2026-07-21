#!/usr/bin/env python3
"""
Run 30 trials per method/test case combination
Output data suitable for statistical analysis
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
import csv
import json
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict
import random

# Zones from geofence.yaml (updated 2026-01-21)
# Zone A: x=[1.5, 2.5], y=[0.5, 1.5] -> center (2.0, 1.0)
# Zone B: x=[1.5, 2.5], y=[2.5, 3.5] -> center (2.0, 3.0)
# Zone C: x=[-4.5, -3.5], y=[0.5, 1.5] -> center (-4.0, 1.0)
# Zone D: x=[3.0, 4.0], y=[-0.5, 0.5] -> center (3.5, 0.0)
ZONES = [
    GeofenceZone(name="zone_A", zone_type=ZoneType.FORBIDDEN, priority=10,
                 vertices=[(1.5, 0.5), (2.5, 0.5), (2.5, 1.5), (1.5, 1.5)]),
    GeofenceZone(name="zone_B", zone_type=ZoneType.FORBIDDEN, priority=10,
                 vertices=[(1.5, 2.5), (2.5, 2.5), (2.5, 3.5), (1.5, 3.5)]),
    GeofenceZone(name="zone_C", zone_type=ZoneType.FORBIDDEN, priority=10,
                 vertices=[(-4.5, 0.5), (-3.5, 0.5), (-3.5, 1.5), (-4.5, 1.5)]),
    GeofenceZone(name="zone_D", zone_type=ZoneType.FORBIDDEN, priority=10,
                 vertices=[(3.0, -0.5), (4.0, -0.5), (4.0, 0.5), (3.0, 0.5)]),
]

# Test scenarios
TEST_CASES = [
    # Inside forbidden zones (attack scenarios)
    {"name": "A1_center", "x": 2.0, "y": 1.0, "type": "attack", "desc": "Zone A center"},
    {"name": "A2_edge", "x": 1.6, "y": 1.0, "type": "attack", "desc": "Zone A near edge"},
    {"name": "B1_center", "x": 2.0, "y": 3.0, "type": "attack", "desc": "Zone B center"},
    {"name": "C1_center", "x": -4.0, "y": 1.0, "type": "attack", "desc": "Zone C center"},
    {"name": "D1_center", "x": 3.5, "y": 0.0, "type": "attack", "desc": "Zone D center"},

    # Boundary cases (within margin)
    {"name": "A_boundary_0.1m", "x": 1.4, "y": 1.0, "type": "boundary", "desc": "0.1m outside Zone A"},
    {"name": "A_boundary_0.2m", "x": 1.3, "y": 1.0, "type": "boundary", "desc": "0.2m outside Zone A"},
    {"name": "A_boundary_0.4m", "x": 1.1, "y": 1.0, "type": "boundary", "desc": "0.4m outside Zone A"},
    {"name": "A_boundary_0.5m", "x": 1.0, "y": 1.0, "type": "boundary", "desc": "0.5m outside Zone A"},

    # Safe locations (benign scenarios)
    {"name": "safe_origin", "x": 0.0, "y": 0.0, "type": "benign", "desc": "Origin"},
    {"name": "safe_nav1", "x": -1.0, "y": 2.0, "type": "benign", "desc": "Navigation point 1"},
    {"name": "safe_nav2", "x": 3.0, "y": -3.0, "type": "benign", "desc": "Navigation point 2"},
    {"name": "safe_far", "x": -2.0, "y": -2.0, "type": "benign", "desc": "Far from zones"},
]

METHODS = ['no_guard', 'selp_proper', 'cbf', 'ssm', 'geofence']
NUM_TRIALS = 30

class NoGuardSafety:
    def evaluate(self, point, zones):
        from safety_baselines import SafetyResult, SafetyDecision
        return SafetyResult(decision=SafetyDecision.ALLOW, reason="no_guard")

def get_method_instance(method_name):
    zone_names = [z.name for z in ZONES]
    if method_name == 'no_guard':
        return NoGuardSafety()
    elif method_name == 'selp_proper':
        return SELPSpatialSafety(zone_names=zone_names)
    elif method_name == 'cbf':
        return CBFSpatialSafety(margin=0.3)
    elif method_name == 'ssm':
        return SSMSpatialSafety(base_margin=0.2)
    elif method_name == 'geofence':
        return StaticMarginSafety(margin=0.55)

def run_trial(method, point, zones):
    """Run single trial and return result"""
    result = method.evaluate(point, zones)
    return result.decision.name

def main():
    print("=" * 70)
    print("30-TRIAL EXPERIMENT")
    print("=" * 70)
    print(f"Methods: {METHODS}")
    print(f"Test cases: {len(TEST_CASES)}")
    print(f"Trials per combination: {NUM_TRIALS}")
    print(f"Total trials: {len(METHODS) * len(TEST_CASES) * NUM_TRIALS}")
    print("=" * 70)

    # Results storage
    all_results = []
    summary = {method: {"allow": 0, "reject": 0, "project": 0} for method in METHODS}

    # Run trials
    for method_name in METHODS:
        print(f"\n[{method_name}]")
        method = get_method_instance(method_name)

        for tc in TEST_CASES:
            point = (tc["x"], tc["y"])
            decisions = []

            for trial in range(NUM_TRIALS):
                decision = run_trial(method, point, ZONES)
                decisions.append(decision)

                # Record result
                all_results.append({
                    "method": method_name,
                    "test_case": tc["name"],
                    "test_type": tc["type"],
                    "x": tc["x"],
                    "y": tc["y"],
                    "trial": trial + 1,
                    "decision": decision,
                })

                # Update summary
                summary[method_name][decision.lower()] += 1

            # Count decisions for this test case
            allow_count = decisions.count("ALLOW")
            reject_count = decisions.count("REJECT")
            project_count = decisions.count("PROJECT")

            # Determine majority decision
            majority = max(set(decisions), key=decisions.count)
            consistency = decisions.count(majority) / len(decisions) * 100

            print(f"  {tc['name']:<20} {majority:<8} ({consistency:.0f}% consistent)")

    # Save raw data to CSV
    csv_path = "/tmp/safety_method_30trials.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["method", "test_case", "test_type", "x", "y", "trial", "decision"])
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nRaw data saved to: {csv_path}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    # Per-method summary
    print(f"\n{'Method':<15} {'ALLOW':<10} {'REJECT':<10} {'PROJECT':<10} {'Total'}")
    print("-" * 55)
    for method in METHODS:
        s = summary[method]
        total = s['allow'] + s['reject'] + s['project']
        print(f"{method:<15} {s['allow']:<10} {s['reject']:<10} {s['project']:<10} {total}")

    # By test type
    print("\n" + "=" * 70)
    print("RESULTS BY TEST TYPE")
    print("=" * 70)

    for test_type in ['attack', 'boundary', 'benign']:
        print(f"\n{test_type.upper()} scenarios:")
        type_results = [r for r in all_results if r['test_type'] == test_type]

        for method in METHODS:
            method_results = [r for r in type_results if r['method'] == method]
            allow = sum(1 for r in method_results if r['decision'] == 'ALLOW')
            reject = sum(1 for r in method_results if r['decision'] == 'REJECT')
            project = sum(1 for r in method_results if r['decision'] == 'PROJECT')
            total = len(method_results)

            if total > 0:
                vr_exec = (reject + project) / total * 100 if test_type == 'attack' else None
                allow_rate = allow / total * 100

                if test_type == 'attack':
                    print(f"  {method:<15} Block rate: {100-allow_rate:5.1f}%  (ALLOW:{allow}, REJECT:{reject}, PROJECT:{project})")
                else:
                    print(f"  {method:<15} Allow rate: {allow_rate:5.1f}%  (ALLOW:{allow}, REJECT:{reject}, PROJECT:{project})")

    # Key metrics for paper
    print("\n" + "=" * 70)
    print("KEY METRICS (for paper)")
    print("=" * 70)

    attack_cases = [r for r in all_results if r['test_type'] == 'attack']
    boundary_cases = [r for r in all_results if r['test_type'] == 'boundary']
    benign_cases = [r for r in all_results if r['test_type'] == 'benign']

    print(f"\n{'Method':<15} {'Attack Block%':<15} {'Boundary Block%':<17} {'Benign Allow%'}")
    print("-" * 65)

    for method in METHODS:
        # Attack block rate
        ma = [r for r in attack_cases if r['method'] == method]
        attack_block = sum(1 for r in ma if r['decision'] != 'ALLOW') / len(ma) * 100 if ma else 0

        # Boundary block rate
        mb = [r for r in boundary_cases if r['method'] == method]
        boundary_block = sum(1 for r in mb if r['decision'] != 'ALLOW') / len(mb) * 100 if mb else 0

        # Benign allow rate
        mc = [r for r in benign_cases if r['method'] == method]
        benign_allow = sum(1 for r in mc if r['decision'] == 'ALLOW') / len(mc) * 100 if mc else 0

        print(f"{method:<15} {attack_block:>10.1f}%     {boundary_block:>12.1f}%      {benign_allow:>10.1f}%")

    # Save summary JSON
    json_path = "/tmp/safety_method_summary.json"
    summary_data = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "methods": METHODS,
            "num_trials": NUM_TRIALS,
            "test_cases": TEST_CASES,
        },
        "summary": summary,
        "metrics": {}
    }

    for method in METHODS:
        ma = [r for r in attack_cases if r['method'] == method]
        mb = [r for r in boundary_cases if r['method'] == method]
        mc = [r for r in benign_cases if r['method'] == method]

        summary_data["metrics"][method] = {
            "attack_block_rate": sum(1 for r in ma if r['decision'] != 'ALLOW') / len(ma) if ma else 0,
            "boundary_block_rate": sum(1 for r in mb if r['decision'] != 'ALLOW') / len(mb) if mb else 0,
            "benign_allow_rate": sum(1 for r in mc if r['decision'] == 'ALLOW') / len(mc) if mc else 0,
        }

    with open(json_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    print(f"\nSummary saved to: {json_path}")

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
