#!/usr/bin/env python3
"""
VR (Violation Rate) Test - Verifies actual zone violations are detected

This test runs a small experiment with measure_actual_violations=True
to verify that:
1. Direct control attack node works
2. ViolationMonitor detects actual zone entries
3. VR is properly recorded

Only tests no_guard + S1 (direct hazard goal) to minimize runtime.
"""

import sys
import os
sys.path.insert(0, 'src/mobile_manipulator_tutorial/src/geofence_policy_enforcer/experiments')

from scie_framework.config import ExperimentConfig, get_default_config
from scie_framework.experiment_runner import FactorialExperimentRunner


def main():
    print("=" * 60)
    print("VR (Violation Rate) TEST")
    print("Tests that actual zone violations are detected")
    print("=" * 60)

    config = get_default_config()

    # Only test no_guard (which always allows) and S1 (direct hazard goal to Zone A at 1.0, 2.0)
    config.methods = [m for m in config.methods if m.id == 'no_guard']
    config.scenarios = [s for s in config.scenarios if s.id == 'S1']
    config.intensity_levels = ["medium"]
    config.num_repetitions = 2  # Just 2 reps
    config.headless = True

    # Print zone and goal info
    s1 = config.scenarios[0]
    print(f"S1 goal: ({s1.goal_x}, {s1.goal_y}) - should be inside Zone A [0.5-1.5, 1.5-2.5]")

    print(f"Methods: {[m.id for m in config.methods]}")
    print(f"Scenarios: {[s.id for s in config.scenarios]}")
    print(f"Total trials: {len(config.methods) * len(config.scenarios) * config.num_repetitions}")
    print()

    runner = FactorialExperimentRunner(
        config=config,
        output_dir="/tmp/vr_test",
        checkpoint_interval=100,
        persistent_dir=os.path.expanduser("~/.scie_experiments")
    )

    # Run with measure_actual_violations=True
    summary = runner.run(
        include_intensity=False,
        include_noise=False,
        randomize=False,
        measure_actual_violations=True  # Enable direct control for FN cases
    )

    print("\n" + "=" * 60)
    print("VR TEST RESULTS")
    print("=" * 60)

    # Check if VR > 0 for no_guard
    vr = summary.get('overall', {}).get('vr', 0) * 100
    actual_violations = summary.get('overall', {}).get('actual_violation_count', 0)

    print(f"Violation Rate (VR): {vr:.1f}%")
    print(f"Actual Violations: {actual_violations}")

    if vr > 0:
        print("\n✓ SUCCESS: VR > 0 - Actual violations detected!")
        print("  Direct control successfully drove robot into forbidden zone")
    else:
        print("\n⚠ WARNING: VR = 0 - No actual violations detected")
        print("  Check ViolationMonitor or direct control attack node")

    # Print detailed results
    if 'by_method' in summary:
        print("\nBy Method:")
        for method, stats in summary['by_method'].items():
            print(f"  {method}:")
            print(f"    Accuracy: {stats['accuracy']*100:.1f}%")
            print(f"    VR: {stats.get('vr', 0)*100:.1f}%")
            print(f"    Actual violations: {stats.get('actual_violation_count', 0)}")

    return 0 if vr > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
