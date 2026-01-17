#!/usr/bin/env python3
"""
SCIE-level Factorial Experiment Runner

Usage:
    # Quick test (S1 only, 1 rep)
    python run_scie_experiment.py --quick

    # Standard experiment (all scenarios, 30 reps)
    python run_scie_experiment.py

    # Custom configuration
    python run_scie_experiment.py --reps 10 --scenarios S1,S2,S3 --methods safetychip,geofence

    # Resume from checkpoint
    python run_scie_experiment.py --resume /path/to/checkpoint.json

Output:
    /tmp/scie_experiments/experiment_YYYYMMDD_HHMMSS/
    ├── config.yaml           # Experiment configuration
    ├── data/
    │   ├── trials_*.jsonl    # Real-time trial logs
    │   ├── results_*.json    # Complete results
    │   ├── results_*.csv     # Tabular results for analysis
    │   └── summary_*.json    # Summary statistics
    └── checkpoint.json       # For resumption
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime

# Add framework to path
sys.path.insert(0, str(Path(__file__).parent))

from scie_framework.config import ExperimentConfig, get_default_config, get_quick_test_config
from scie_framework.experiment_runner import FactorialExperimentRunner


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SCIE-level factorial experiment for geofence safety evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --quick                    Quick test (S1, all methods, 1 rep)
  %(prog)s --reps 30                  Full experiment with 30 repetitions
  %(prog)s --scenarios S1,S2,S3       Run only specific scenarios
  %(prog)s --methods safetychip,geofence  Compare specific methods
  %(prog)s --resume checkpoint.json   Resume interrupted experiment
        """
    )

    parser.add_argument('--quick', action='store_true',
                       help='Run quick test (S1 only, 1 rep)')

    parser.add_argument('--reps', type=int, default=30,
                       help='Number of repetitions per condition (default: 30)')

    parser.add_argument('--scenarios', type=str, default=None,
                       help='Comma-separated list of scenarios (e.g., S1,S2,S3)')

    parser.add_argument('--methods', type=str, default=None,
                       help='Comma-separated list of methods (e.g., safetychip,geofence)')

    parser.add_argument('--intensity', action='store_true',
                       help='Include intensity factor in experiment')

    parser.add_argument('--noise', action='store_true',
                       help='Include noise factor in experiment')

    parser.add_argument('--output', type=str, default='/tmp/scie_experiments',
                       help='Output directory (default: /tmp/scie_experiments)')

    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint file for resumption')

    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')

    parser.add_argument('--no-randomize', action='store_true',
                       help='Run trials in order (no randomization)')

    parser.add_argument('--workspace', type=str,
                       default='/home/jim/ros2_motion_planning_tutorials/src/mobile_manipulator_tutorial',
                       help='ROS2 workspace path')

    return parser.parse_args()


def create_config(args) -> ExperimentConfig:
    """Create experiment configuration from arguments"""
    if args.quick:
        config = get_quick_test_config()
    else:
        config = get_default_config()

    # Override with command line arguments
    config.num_repetitions = args.reps
    config.random_seed = args.seed
    config.workspace_path = args.workspace
    config.output_base_path = args.output

    # Filter scenarios if specified
    if args.scenarios:
        scenario_ids = [s.strip().upper() for s in args.scenarios.split(',')]
        config.scenarios = [s for s in config.scenarios if s.id in scenario_ids]
        if not config.scenarios:
            print(f"Error: No valid scenarios found in: {args.scenarios}")
            sys.exit(1)

    # Filter methods if specified
    if args.methods:
        method_ids = [m.strip().lower() for m in args.methods.split(',')]
        config.methods = [m for m in config.methods if m.id in method_ids]
        if not config.methods:
            print(f"Error: No valid methods found in: {args.methods}")
            sys.exit(1)

    return config


def print_experiment_info(config: ExperimentConfig, args):
    """Print experiment information"""
    total = config.get_total_trials(
        include_intensity=args.intensity,
        include_noise=args.noise
    )

    print("\n" + "=" * 70)
    print("SCIE-LEVEL FACTORIAL EXPERIMENT")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Workspace:     {config.workspace_path}")
    print(f"  Output:        {config.output_base_path}")
    print(f"  Random seed:   {config.random_seed}")
    print(f"\nFactors:")
    print(f"  Methods:       {[m.id for m in config.methods]}")
    print(f"  Scenarios:     {[s.id for s in config.scenarios]}")
    print(f"  Repetitions:   {config.num_repetitions}")
    print(f"  Intensity:     {args.intensity}")
    print(f"  Noise:         {args.noise}")
    print(f"\nTotal trials:    {total}")

    # Estimate time
    time_per_trial_min = 1.5  # ~1.5 minutes per trial (conservative)
    total_time_min = total * time_per_trial_min
    total_time_hours = total_time_min / 60

    print(f"Estimated time:  {total_time_min:.0f} min ({total_time_hours:.1f} hours)")
    print("=" * 70)


def main():
    args = parse_args()

    # Create configuration
    config = create_config(args)

    # Print experiment info
    print_experiment_info(config, args)

    # Confirm before running full experiment
    if not args.quick and config.num_repetitions >= 10:
        print("\nThis will run a large experiment.")
        response = input("Continue? [y/N]: ").strip().lower()
        if response != 'y':
            print("Aborted.")
            sys.exit(0)

    # Create runner
    runner = FactorialExperimentRunner(config, args.output)

    # Run experiment
    print("\nStarting experiment...")
    print("Press Ctrl+C to interrupt (progress will be saved)\n")

    try:
        summary = runner.run(
            include_intensity=args.intensity,
            include_noise=args.noise,
            randomize=not args.no_randomize,
            resume_from=args.resume
        )

        # Print summary
        print("\n" + "=" * 70)
        print("EXPERIMENT COMPLETE")
        print("=" * 70)
        print(f"\nResults saved to: {runner.experiment_dir}")
        print(f"\nSummary:")
        print(f"  Total trials:     {summary.get('metadata', {}).get('total_trials', 0)}")
        print(f"  Completed:        {summary.get('metadata', {}).get('completed_trials', 0)}")
        print(f"  Failed:           {summary.get('metadata', {}).get('failed_trials', 0)}")

        if 'overall' in summary:
            print(f"\nOverall accuracy:   {summary['overall'].get('accuracy', 0)*100:.1f}%")

        print("\nNext steps:")
        print(f"  1. Analyze: python analyze_scie_results.py {runner.experiment_dir}/data/results_*.csv")
        print(f"  2. Visualize: python generate_scie_figures.py {runner.experiment_dir}/data/results_*.csv")

    except KeyboardInterrupt:
        print("\n\nExperiment interrupted. Progress has been saved.")
        print(f"Resume with: python {sys.argv[0]} --resume {runner.experiment_dir}/checkpoint.json")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
