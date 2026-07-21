#!/usr/bin/env python3
"""
Full SCIE Experiment Runner

Features:
- Checkpoint every 5 trials (resume on crash/reboot)
- Process cleanup between method changes
- Anomaly monitoring and reporting
- Intensity-based expected results

Usage:
    # Fresh start
    python3 run_full_experiment.py

    # Resume from checkpoint
    python3 run_full_experiment.py --resume

    # Resume from specific checkpoint
    python3 run_full_experiment.py --resume ~/.scie_experiments/checkpoint_20240101_120000.json

    # Quick test (S1 only)
    python3 run_full_experiment.py --quick
"""

import sys
import os
import argparse
from pathlib import Path

# Add experiment framework to path
sys.path.insert(0, 'src/mobile_manipulator_tutorial/src/geofence_policy_enforcer/experiments')

from scie_framework.config import ExperimentConfig, get_default_config
from scie_framework.experiment_runner import FactorialExperimentRunner


def find_latest_checkpoint():
    """Find the latest checkpoint file"""
    persistent_dir = Path(os.path.expanduser("~/.scie_experiments"))
    latest = persistent_dir / "checkpoint_latest.json"

    if latest.exists():
        return str(latest)

    # Find most recent checkpoint
    checkpoints = list(persistent_dir.glob("checkpoint_*.json"))
    if checkpoints:
        return str(max(checkpoints, key=lambda p: p.stat().st_mtime))

    return None


def main():
    parser = argparse.ArgumentParser(description="Run SCIE Factorial Experiment")
    parser.add_argument("--resume", nargs="?", const="auto",
                        help="Resume from checkpoint (auto or path)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test with S1 only")
    parser.add_argument("--output", type=str, default="/tmp/scie_experiments",
                        help="Output directory")
    parser.add_argument("--intensity", action="store_true", default=True,
                        help="Include intensity variations (default: True)")
    parser.add_argument("--no-intensity", action="store_true",
                        help="Disable intensity variations")
    parser.add_argument("--measure-violations", action="store_true",
                        help="Use direct control to measure actual violations (slower)")
    parser.add_argument("--methods", type=str, nargs="+",
                        help="Specific methods to test (e.g., --methods ssm geofence)")
    parser.add_argument("--scenarios", type=str, nargs="+",
                        help="Specific scenarios to test (e.g., --scenarios S1 S4)")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Skip confirmation prompt")
    args = parser.parse_args()

    print("=" * 70)
    print("SCIE FACTORIAL EXPERIMENT")
    print("=" * 70)

    # Load config
    config = get_default_config()

    # Filter methods if specified
    if args.methods:
        config.methods = [m for m in config.methods if m.id in args.methods]
        print(f"Methods: {[m.id for m in config.methods]}")

    # Filter scenarios if specified
    if args.scenarios:
        config.scenarios = [s for s in config.scenarios if s.id in args.scenarios]
        print(f"Scenarios: {[s.id for s in config.scenarios]}")

    # Quick test mode
    if args.quick:
        config.scenarios = [s for s in config.scenarios if s.id == "S1"]
        config.num_repetitions = 1
        config.intensity_levels = ["low", "high"]
        print("Quick test mode: S1 only, 1 rep, low/high intensity")

    # Calculate total trials
    include_intensity = args.intensity and not args.no_intensity
    num_intensities = len(config.intensity_levels) if include_intensity else 1
    total = (len(config.methods) * len(config.scenarios) *
             num_intensities * config.num_repetitions)

    print(f"\nExperiment Configuration:")
    print(f"  Methods: {len(config.methods)} ({[m.id for m in config.methods]})")
    print(f"  Scenarios: {len(config.scenarios)} ({[s.id for s in config.scenarios]})")
    print(f"  Intensities: {config.intensity_levels if include_intensity else ['medium']}")
    print(f"  Repetitions: {config.num_repetitions}")
    print(f"  Total trials: {total}")
    print(f"  Checkpoint interval: 5 trials")
    print(f"  Output: {args.output}")

    # Handle resume
    resume_path = None
    if args.resume:
        if args.resume == "auto":
            resume_path = find_latest_checkpoint()
            if resume_path:
                print(f"\nResuming from: {resume_path}")
            else:
                print("\nNo checkpoint found, starting fresh")
        else:
            resume_path = args.resume
            print(f"\nResuming from: {resume_path}")

    print("\n" + "=" * 70)

    # Confirm before starting
    if not args.quick and not args.yes:
        try:
            input("Press Enter to start experiment (Ctrl+C to cancel)...")
        except (KeyboardInterrupt, EOFError):
            print("\nCancelled (use --yes to skip confirmation)")
            return 1

    # Create runner
    runner = FactorialExperimentRunner(
        config=config,
        output_dir=args.output,
        checkpoint_interval=5,
    )

    # Run experiment
    try:
        summary = runner.run(
            include_intensity=include_intensity,
            include_noise=False,
            randomize=True,
            resume_from=resume_path,
            measure_actual_violations=args.measure_violations,
        )

        # Print summary
        print("\n" + "=" * 70)
        print("EXPERIMENT COMPLETE")
        print("=" * 70)

        if 'by_method' in summary:
            print("\nResults by Method:")
            for method, stats in summary['by_method'].items():
                acc = stats.get('accuracy', 0) * 100
                total_trials = stats.get('total', 0)
                allow_rate = stats.get('allow_rate', 0) * 100
                vr = stats.get('violation_rate', 0) * 100
                print(f"\n  {method}:")
                print(f"    Accuracy: {acc:.1f}%")
                print(f"    Allow Rate: {allow_rate:.1f}%")
                print(f"    Violation Rate: {vr:.1f}%")
                print(f"    Trials: {total_trials}")

        if 'anomalies' in summary:
            num_anomalies = len(summary.get('anomalies', []))
            if num_anomalies > 0:
                print(f"\n⚠️  Anomalies detected: {num_anomalies}")
                print("   Check experiment logs for details")

        print(f"\nResults saved to: {runner.experiment_dir}")
        print(f"Checkpoint: ~/.scie_experiments/checkpoint_latest.json")

        return 0

    except KeyboardInterrupt:
        print("\n\nExperiment interrupted!")
        print(f"Resume with: python3 {sys.argv[0]} --resume")
        return 1

    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        print(f"\nResume with: python3 {sys.argv[0]} --resume")
        return 1


if __name__ == "__main__":
    sys.exit(main())
