#!/usr/bin/env python3
"""
SCIE-level Experiment Runner

Runs full factorial experiment with:
- 5 methods × 8 scenarios × 30 repetitions = 1,200 trials
- Checkpoint every 10 trials (persistent to ~/.scie_experiments)
- Anomaly detection and real-time monitoring
- Resume capability from interruption
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime

# Add experiment framework to path
sys.path.insert(0, 'src/mobile_manipulator_tutorial/src/geofence_policy_enforcer/experiments')

from scie_framework.config import ExperimentConfig, get_default_config
from scie_framework.experiment_runner import FactorialExperimentRunner


def main():
    parser = argparse.ArgumentParser(description='Run SCIE-level geofence experiment')
    parser.add_argument('--reps', type=int, default=30, help='Number of repetitions')
    parser.add_argument('--resume', type=str, help='Path to checkpoint file to resume from')
    parser.add_argument('--headless', action='store_true', default=True, help='Run without GUI')
    parser.add_argument('--rtf', type=float, default=1.0, help='Real-time factor (sim speed)')
    parser.add_argument('--checkpoint-interval', type=int, default=10, help='Save checkpoint every N trials')
    parser.add_argument('--output', type=str, default='/tmp/scie_experiments', help='Output directory')
    parser.add_argument('--quick-test', action='store_true', help='Run quick test (S1 only, 1 rep)')
    args = parser.parse_args()

    # Get default config and customize
    config = get_default_config()
    config.num_repetitions = args.reps
    config.headless = args.headless
    config.real_time_factor = args.rtf
    config.output_base_path = args.output

    # Print experiment info
    total_trials = len(config.methods) * len(config.scenarios) * config.num_repetitions

    print("=" * 70)
    print("SCIE-LEVEL GEOFENCE SAFETY EXPERIMENT")
    print("=" * 70)
    print(f"Methods:      {len(config.methods)} ({', '.join(m.id for m in config.methods)})")
    print(f"Scenarios:    {len(config.scenarios)} (S1-S8)")
    print(f"Repetitions:  {config.num_repetitions}")
    print(f"Total trials: {total_trials}")
    print(f"Headless:     {config.headless}")
    print(f"RTF:          {config.real_time_factor}x")
    print(f"Checkpoint:   Every {args.checkpoint_interval} trials")
    print(f"Output:       {config.output_base_path}")
    if args.resume:
        print(f"Resuming:     {args.resume}")
    print("=" * 70)

    # Estimate time
    avg_trial_time = 45  # seconds per trial estimate
    estimated_hours = (total_trials * avg_trial_time) / 3600
    print(f"\nEstimated time: {estimated_hours:.1f} hours")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Create runner
    runner = FactorialExperimentRunner(
        config=config,
        output_dir=config.output_base_path,
        checkpoint_interval=args.checkpoint_interval,
        persistent_dir=os.path.expanduser("~/.scie_experiments")
    )

    # Add callbacks for real-time monitoring
    def on_trial_complete(result):
        """Called after each trial - check for critical issues"""
        if result.safety.actual_violation and result.conditions.method == "geofence":
            print("\n" + "!" * 60)
            print("CRITICAL: Geofence method had actual violation!")
            print(f"Scenario: {result.conditions.scenario}")
            print(f"Goal: ({result.goal_x}, {result.goal_y})")
            print(f"Penetration: {result.safety.actual_max_penetration}m")
            print("!" * 60 + "\n")

    def on_method_change(method):
        """Called when switching methods"""
        print(f"\n{'#' * 60}")
        print(f"# Now testing: {method}")
        print(f"{'#' * 60}\n")

    runner.on_trial_complete = on_trial_complete
    runner.on_method_change = on_method_change

    # Run experiment
    try:
        if args.quick_test:
            print("[MODE] Quick test - S1 only, 1 repetition")
            summary = runner.run_quick_test()
        else:
            # Group by method to minimize restarts (but randomize within method)
            # This reduces method switches from potentially 1200 to just 5
            summary = runner.run(
                include_intensity=False,  # Not using intensity variations for now
                include_noise=False,      # Not using noise variations for now
                randomize=False,          # Don't randomize to keep methods grouped
                resume_from=args.resume
            )

        # Print final summary
        print("\n" + "=" * 70)
        print("EXPERIMENT COMPLETE")
        print("=" * 70)

        if 'overall' in summary:
            overall = summary['overall']
            print(f"\n=== USENIX Metrics ===")
            print(f"Total trials: {overall.get('total', 0)}")
            print(f"Accuracy:     {overall.get('accuracy', 0):.1%}")
            print(f"\n1. VR (Violation Rate):    {overall.get('vr', 0):.1%}")
            print(f"2. ASR (Attack Success):   {overall.get('asr', 0):.1%}")
            print(f"3. FNR (False Negative):   {overall.get('fnr', 0):.1%}")
            print(f"4. TCR (Task Completion):  {overall.get('tcr', 0):.1%}")
            print(f"5. MD (Mean Distance):     {overall.get('md_mean', 0):.2f}m")
            print(f"6. BR/SR:                  {overall.get('br', 0):.1%} / {overall.get('sr', 0):.1%}")
            print(f"7. RT (Latency):           {overall.get('rt_mean_ms', 0):.1f}ms")
            print(f"8. OBR (Overblocking):     {overall.get('obr', 0):.1%}")

            print(f"\n=== By Method ===")
            for method, stats in summary.get('by_method', {}).items():
                print(f"\n{method}:")
                print(f"  Accuracy: {stats.get('accuracy', 0):.1%}")
                print(f"  VR: {stats.get('vr', 0):.1%}, FNR: {stats.get('fnr', 0):.1%}")

        # Copy to persistent storage
        import shutil
        persistent_output = Path(os.path.expanduser("~/.scie_experiments")) / f"experiment_{runner.experiment_id}"
        if runner.experiment_dir.exists():
            shutil.copytree(runner.experiment_dir, persistent_output, dirs_exist_ok=True)
            print(f"\nResults saved to: {persistent_output}")

        return 0

    except KeyboardInterrupt:
        print("\n\n[INFO] Experiment interrupted by user")
        print("[INFO] Checkpoint saved - you can resume with --resume flag")
        return 1
    except Exception as e:
        print(f"\n[ERROR] Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
