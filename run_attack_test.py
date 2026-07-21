#!/usr/bin/env python3
"""Quick test for S4, S5, S6 attack implementations"""

import sys
sys.path.insert(0, 'src/mobile_manipulator_tutorial/src/geofence_policy_enforcer/experiments')

from scie_framework.config import ExperimentConfig, get_default_config
from scie_framework.experiment_runner import FactorialExperimentRunner

def main():
    print("=" * 60)
    print("S4/S5/S6 Attack Implementation Test")
    print("=" * 60)

    config = get_default_config()

    # Only SSM (most affected by these attacks)
    config.methods = [m for m in config.methods if m.id == 'ssm']
    # Only S4, S5, S6
    config.scenarios = [s for s in config.scenarios if s.id in ('S4', 'S5', 'S6')]
    config.num_repetitions = 1
    config.intensity_levels = ["low", "high"]  # Test extremes
    config.headless = True

    print(f"Methods: {[m.id for m in config.methods]}")
    print(f"Scenarios: {[s.id for s in config.scenarios]}")
    print(f"Intensities: {config.intensity_levels}")

    # Show scenario configs
    for s in config.scenarios:
        print(f"\n{s.id}: {s.name}")
        print(f"  Attack Type: {s.attack_type}")
        if s.id == 'S4':
            print(f"  Approach Velocity: {s.approach_velocity}")
            for level, var in s.intensity_variations.items():
                if level in config.intensity_levels:
                    vel = var.get('approach_velocity', s.approach_velocity)
                    goal = (var.get('goal_x', s.goal_x), var.get('goal_y', s.goal_y))
                    print(f"    {level}: velocity={vel}m/s, goal={goal}")
        elif s.id == 'S5':
            print(f"  Pose Spoofing - intensity-based offset")
        elif s.id == 'S6':
            print(f"  Latency: {s.latency_ms}ms")
            for level, var in s.intensity_variations.items():
                if level in config.intensity_levels:
                    lat = var.get('latency_ms', s.latency_ms)
                    print(f"    {level}: latency={lat}ms")

    total = len(config.methods) * len(config.scenarios) * len(config.intensity_levels) * config.num_repetitions
    print(f"\nTotal trials: {total}")
    print()

    runner = FactorialExperimentRunner(
        config=config,
        output_dir="/tmp/scie_attack_test",
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
