#!/usr/bin/env python3
"""Quick CBF/SSM verification experiment"""

import sys
import os
sys.path.insert(0, 'src/mobile_manipulator_tutorial/src/geofence_policy_enforcer/experiments')

from scie_framework.config import ExperimentConfig, MethodConfig, get_default_config
from scie_framework.experiment_runner import FactorialExperimentRunner


def main():
    print("=" * 60)
    print("CBF/SSM Boundary-Near Verification Experiment")
    print("=" * 60)

    config = get_default_config()

    # Only CBF and SSM
    config.methods = [m for m in config.methods if m.id in ('cbf', 'ssm')]
    config.num_repetitions = 2  # Quick test
    config.intensity_levels = ["low", "medium", "high", "extreme"]
    config.headless = True

    total = len(config.methods) * len(config.scenarios) * len(config.intensity_levels) * config.num_repetitions
    print(f"Methods: {[m.id for m in config.methods]}")
    print(f"Intensities: {config.intensity_levels}")
    print(f"Repetitions: {config.num_repetitions}")
    print(f"Total trials: {total}")
    print()

    runner = FactorialExperimentRunner(
        config=config,
        output_dir="/tmp/scie_cbf_ssm_test",
        checkpoint_interval=10,
        persistent_dir=os.path.expanduser("~/.scie_experiments")
    )

    summary = runner.run(
        include_intensity=True,
        include_noise=False,
        randomize=False,
        measure_actual_violations=False  # Skip direct control for speed
    )

    print("\n" + "=" * 60)
    print("RESULTS BY METHOD AND INTENSITY")
    print("=" * 60)

    if 'by_method' in summary:
        for method, stats in summary['by_method'].items():
            acc = stats.get('accuracy', 0) * 100
            vr = stats.get('vr', 0) * 100
            allow_rate = (1 - stats.get('block_rate', 0)) * 100
            print(f"\n{method}:")
            print(f"  Accuracy: {acc:.1f}%")
            print(f"  Allow Rate: {allow_rate:.1f}%")
            print(f"  Violation Rate: {vr:.1f}%")

    print(f"\nResults saved to: {runner.experiment_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
