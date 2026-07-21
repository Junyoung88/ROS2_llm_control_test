#!/usr/bin/env python3
"""Quick test for S2 (Relative Movement Salami Attack)"""

import sys
sys.path.insert(0, 'src/mobile_manipulator_tutorial/src/geofence_policy_enforcer/experiments')

from scie_framework.config import ExperimentConfig, get_default_config
from scie_framework.experiment_runner import FactorialExperimentRunner

def main():
    print("=" * 60)
    print("S2 Relative Movement Salami Attack Test")
    print("=" * 60)

    config = get_default_config()

    # Only CBF and SSM
    config.methods = [m for m in config.methods if m.id in ('cbf', 'ssm')]
    # Only S2
    config.scenarios = [s for s in config.scenarios if s.id == 'S2']
    config.num_repetitions = 1  # Quick test
    config.intensity_levels = ["low", "medium", "high", "extreme"]
    config.headless = True

    print(f"Methods: {[m.id for m in config.methods]}")
    print(f"Scenarios: {[s.id for s in config.scenarios]}")
    print(f"Intensities: {config.intensity_levels}")

    # Show S2 config
    s2 = config.scenarios[0]
    print(f"\nS2 Config:")
    print(f"  Name: {s2.name}")
    print(f"  Attack Type: {s2.attack_type}")
    print(f"  Relative Moves: {s2.relative_moves}")
    print(f"  Intensity Variations:")
    for level, var in s2.intensity_variations.items():
        moves = var.get('relative_moves', [])
        goal = (var.get('goal_x', s2.goal_x), var.get('goal_y', s2.goal_y))
        print(f"    {level}: {len(moves)} moves -> goal {goal}")

    total = len(config.methods) * len(config.scenarios) * len(config.intensity_levels) * config.num_repetitions
    print(f"\nTotal trials: {total}")
    print()

    runner = FactorialExperimentRunner(
        config=config,
        output_dir="/tmp/scie_s2_test",
        checkpoint_interval=5,
    )

    summary = runner.run(
        include_intensity=True,
        include_noise=False,
        randomize=False,
        measure_actual_violations=False
    )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    if 'by_method' in summary:
        for method, stats in summary['by_method'].items():
            acc = stats.get('accuracy', 0) * 100
            print(f"\n{method}:")
            print(f"  Accuracy: {acc:.1f}%")
            print(f"  Trials: {stats.get('total', 0)}")

    print(f"\nResults saved to: {runner.experiment_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
