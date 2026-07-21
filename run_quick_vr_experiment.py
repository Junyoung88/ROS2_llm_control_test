#!/usr/bin/env python3
"""
Quick VR Verification Experiment - 5 reps per condition

Tests all scenarios and methods with actual violation detection enabled.
Estimated time: ~2 hours
"""

import sys
import os
sys.path.insert(0, 'src/mobile_manipulator_tutorial/src/geofence_policy_enforcer/experiments')

from scie_framework.config import ExperimentConfig, get_default_config
from scie_framework.experiment_runner import FactorialExperimentRunner


def main():
    print("=" * 60)
    print("QUICK VR VERIFICATION EXPERIMENT")
    print("5 reps per condition, actual violation detection enabled")
    print("=" * 60)

    config = get_default_config()

    # Use all methods and scenarios but only 5 reps
    config.num_repetitions = 5
    config.intensity_levels = ["medium"]  # Only medium intensity
    config.headless = True

    total = len(config.methods) * len(config.scenarios) * config.num_repetitions
    print(f"Methods: {[m.id for m in config.methods]}")
    print(f"Scenarios: {[s.id for s in config.scenarios]}")
    print(f"Repetitions: {config.num_repetitions}")
    print(f"Total trials: {total}")
    print(f"Estimated time: ~{total * 2 / 60:.1f} hours")
    print()

    runner = FactorialExperimentRunner(
        config=config,
        output_dir="/tmp/scie_quick_vr",
        checkpoint_interval=10,
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
    print("EXPERIMENT RESULTS")
    print("=" * 60)

    # Print summary
    if 'overall' in summary:
        overall = summary['overall']
        print(f"\nOverall Metrics:")
        print(f"  Accuracy: {overall.get('accuracy', 0)*100:.1f}%")
        print(f"  VR (Violation Rate): {overall.get('vr', 0)*100:.1f}%")
        print(f"  FNR (False Negative Rate): {overall.get('fnr', 0)*100:.1f}%")
        print(f"  BR (Block Rate): {overall.get('block_rate', 0)*100:.1f}%")

    if 'by_method' in summary:
        print("\n\nBy Method:")
        print("-" * 50)
        print(f"{'Method':<20} {'ACC':<8} {'VR':<8} {'FNR':<8} {'BR':<8}")
        print("-" * 50)
        for method, stats in summary['by_method'].items():
            acc = stats.get('accuracy', 0) * 100
            vr = stats.get('vr', 0) * 100
            fnr = stats.get('fnr', 0) * 100
            br = stats.get('block_rate', 0) * 100
            print(f"{method:<20} {acc:<8.1f} {vr:<8.1f} {fnr:<8.1f} {br:<8.1f}")

    print("\n" + "=" * 60)
    print(f"Results saved to: {runner.experiment_dir}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
