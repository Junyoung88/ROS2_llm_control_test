#!/usr/bin/env python3
"""
Extreme Intensity Test - Tests geofence limits

Only runs S5, S6, S7 with extreme intensity to show geofence limitations.
"""

import sys
import os
sys.path.insert(0, 'src/mobile_manipulator_tutorial/src/geofence_policy_enforcer/experiments')

from scie_framework.config import ExperimentConfig, get_default_config
from scie_framework.experiment_runner import FactorialExperimentRunner

def main():
    config = get_default_config()

    # Only test extreme intensity
    config.intensity_levels = ["extreme"]

    # Filter scenarios to S5, S6, S7 only
    config.scenarios = [s for s in config.scenarios if s.id in ['S5', 'S6', 'S7']]

    # 10 repetitions for quick test
    config.num_repetitions = 10
    config.headless = True

    total = len(config.methods) * len(config.scenarios) * config.num_repetitions
    print(f"Running extreme intensity test: {total} trials")
    print(f"Methods: {[m.id for m in config.methods]}")
    print(f"Scenarios: {[s.id for s in config.scenarios]}")
    print(f"Intensity: extreme only")
    print()

    runner = FactorialExperimentRunner(
        config=config,
        output_dir="/tmp/scie_experiments_extreme",
        checkpoint_interval=10,
        persistent_dir=os.path.expanduser("~/.scie_experiments")
    )

    summary = runner.run(
        include_intensity=True,  # Use extreme intensity
        include_noise=False,
        randomize=False
    )

    print("\n" + "="*60)
    print("EXTREME INTENSITY RESULTS")
    print("="*60)

    if 'by_method' in summary:
        print("\nAccuracy by Method (extreme intensity):")
        for method, stats in summary['by_method'].items():
            print(f"  {method}: {stats['accuracy']*100:.1f}%")

    if 'by_method_scenario' in summary:
        print("\nDetailed Results:")
        for key, stats in summary['by_method_scenario'].items():
            method, scenario = key.rsplit('_', 1)
            decision_rate = stats.get('block_rate', 0) * 100
            print(f"  {method} + {scenario}: acc={stats['accuracy']*100:.0f}%, blocked={decision_rate:.0f}%")

    return 0

if __name__ == "__main__":
    sys.exit(main())
