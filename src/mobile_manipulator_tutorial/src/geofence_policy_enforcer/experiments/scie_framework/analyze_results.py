#!/usr/bin/env python3
"""
USENIX Paper Analysis Script

Generates publication-ready tables from experiment results:
- Table 1: Overall Summary (Safety-Utility-Overhead)
- Table 2: Scenario × Method Robustness Matrix
- Table 3: Ablation Study
- Table 4: Runtime & Real-time Feasibility

Usage:
    python analyze_results.py /path/to/experiment_dir
    python analyze_results.py /path/to/results.csv
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import statistics


def load_results(path: str) -> List[Dict]:
    """Load experiment results from JSON, JSONL, or CSV"""
    path = Path(path)

    if path.is_dir():
        # Look for data files in directory
        data_dir = path / "data"
        if not data_dir.exists():
            data_dir = path

        # Try JSONL first (real-time log)
        jsonl_files = list(data_dir.glob("trials_*.jsonl"))
        if jsonl_files:
            results = []
            with open(jsonl_files[0], 'r') as f:
                for line in f:
                    if line.strip():
                        results.append(json.loads(line))
            return results

        # Try JSON
        json_files = list(data_dir.glob("results_*.json"))
        if json_files:
            with open(json_files[0], 'r') as f:
                data = json.load(f)
                return data.get("trials", [])

        # Try CSV
        csv_files = list(data_dir.glob("results_*.csv"))
        if csv_files:
            import csv
            with open(csv_files[0], 'r') as f:
                reader = csv.DictReader(f)
                return list(reader)

        print(f"No result files found in {path}")
        return []

    elif path.suffix == '.jsonl':
        results = []
        with open(path, 'r') as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
        return results

    elif path.suffix == '.json':
        with open(path, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            return data.get("trials", [])

    elif path.suffix == '.csv':
        import csv
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            return list(reader)

    else:
        print(f"Unknown file type: {path}")
        return []


def get_field(trial: Dict, field_name: str, default=None):
    """Get a field from trial data, handling nested structures"""
    # Direct access
    if field_name in trial:
        return trial[field_name]

    # Check nested structures
    for section in ['conditions', 'safety', 'performance', 'method_specific', 'ablation']:
        if section in trial and isinstance(trial[section], dict):
            if field_name in trial[section]:
                return trial[section][field_name]

    return default


def to_bool(val) -> bool:
    """Convert various representations to bool"""
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() in ('true', '1', 'yes')
    return bool(val)


def to_float(val, default=0.0) -> float:
    """Convert to float with default"""
    try:
        if val is None or val == '' or val == 'inf':
            return default
        return float(val)
    except (ValueError, TypeError):
        return default


@dataclass
class MethodStats:
    """Statistics for a single method"""
    method: str

    # Counts
    total: int = 0
    executed: int = 0
    violations: int = 0
    violations_in_executed: int = 0
    fn_count: int = 0
    fp_count: int = 0
    goal_reached: int = 0

    # Latencies
    latencies: List[float] = None

    # Distances
    min_distances: List[float] = None

    def __post_init__(self):
        self.latencies = []
        self.min_distances = []

    @property
    def vr(self) -> float:
        """Violation Rate"""
        return self.violations / self.total if self.total > 0 else 0

    @property
    def vr_exec(self) -> float:
        """Violation Rate (executed trials only) - KEY METRIC"""
        return self.violations_in_executed / self.executed if self.executed > 0 else 0

    @property
    def fnr(self) -> float:
        """False Negative Rate"""
        # FNR = FN / (FN + TP) where TP = violations caught
        caught = self.violations - self.fn_count
        return self.fn_count / (self.fn_count + caught) if (self.fn_count + caught) > 0 else 0

    @property
    def tcr(self) -> float:
        """Task Completion Rate"""
        return self.goal_reached / self.total if self.total > 0 else 0

    @property
    def fpr(self) -> float:
        """False Positive Rate"""
        safe_trials = self.total - self.violations
        return self.fp_count / safe_trials if safe_trials > 0 else 0

    @property
    def br(self) -> float:
        """Block Rate"""
        blocked = self.total - self.goal_reached
        return blocked / self.total if self.total > 0 else 0

    @property
    def rt_median(self) -> float:
        """Runtime median (ms)"""
        return statistics.median(self.latencies) if self.latencies else 0

    @property
    def rt_p95(self) -> float:
        """Runtime 95th percentile (ms)"""
        if not self.latencies:
            return 0
        sorted_lat = sorted(self.latencies)
        idx = int(0.95 * len(sorted_lat))
        return sorted_lat[min(idx, len(sorted_lat)-1)]

    @property
    def md_mean(self) -> float:
        """Mean minimum distance"""
        return statistics.mean(self.min_distances) if self.min_distances else 0

    @property
    def md_p5(self) -> float:
        """5th percentile min distance (worst tail)"""
        if not self.min_distances:
            return 0
        sorted_dist = sorted(self.min_distances)
        idx = int(0.05 * len(sorted_dist))
        return sorted_dist[idx]


def compute_method_stats(trials: List[Dict], method: str) -> MethodStats:
    """Compute statistics for a single method"""
    stats = MethodStats(method=method)

    for t in trials:
        m = get_field(t, 'method')
        if m != method:
            continue

        stats.total += 1

        # Execution tracking
        did_execute = to_bool(get_field(t, 'did_execute', False))
        if did_execute:
            stats.executed += 1

        # Violation tracking
        actual_violation = to_bool(get_field(t, 'actual_violation', False))
        if actual_violation:
            stats.violations += 1
            if did_execute:
                stats.violations_in_executed += 1

        # FN/FP tracking
        if to_bool(get_field(t, 'is_false_negative', False)):
            stats.fn_count += 1
        if to_bool(get_field(t, 'is_false_positive', False)):
            stats.fp_count += 1

        # Goal reached
        if to_bool(get_field(t, 'goal_reached', False)):
            stats.goal_reached += 1

        # Latency
        latency = to_float(get_field(t, 'decision_latency_ms', 0))
        if latency > 0:
            stats.latencies.append(latency)

        # Min distance
        dist = to_float(get_field(t, 'min_distance_to_forbidden', float('inf')))
        if dist < float('inf') and dist > -10:  # filter valid distances
            stats.min_distances.append(dist)

    return stats


def generate_table1(trials: List[Dict], methods: List[str]) -> str:
    """
    Generate Table 1: Overall Summary (Safety-Utility-Overhead)

    | Method    | VR_exec↓ | FNR↓  | TCR↑  | FPR↓  | RT_med | RT_p95 |
    |-----------|----------|-------|-------|-------|--------|--------|
    """
    output = []
    output.append("=" * 80)
    output.append("TABLE 1: Overall Summary (Safety-Utility-Overhead)")
    output.append("=" * 80)
    output.append("")

    # Header
    header = f"{'Method':<15} {'VR_exec↓':>8} {'FNR↓':>8} {'TCR↑':>8} {'FPR↓':>8} {'RT_med':>8} {'RT_p95':>8}"
    output.append(header)
    output.append("-" * len(header))

    for method in methods:
        stats = compute_method_stats(trials, method)
        if stats.total == 0:
            continue

        row = f"{method:<15} {stats.vr_exec:>7.1%} {stats.fnr:>7.1%} {stats.tcr:>7.1%} {stats.fpr:>7.1%} {stats.rt_median:>7.1f} {stats.rt_p95:>7.1f}"
        output.append(row)

    output.append("")
    output.append("Note: VR_exec = violations / executed_trials (key safety metric)")
    output.append("      FNR = false_negatives / (FN + TP), TCR = goal_reached / total")
    output.append("      RT in milliseconds")

    return "\n".join(output)


def generate_table2(trials: List[Dict], methods: List[str], scenarios: List[str]) -> str:
    """
    Generate Table 2: Scenario × Method Robustness Matrix

    Shows VR_exec for each scenario-method combination
    """
    output = []
    output.append("=" * 80)
    output.append("TABLE 2: Scenario × Method Robustness Matrix (VR_exec)")
    output.append("=" * 80)
    output.append("")

    # Header
    header = f"{'Scenario':<10}"
    for m in methods:
        header += f" {m[:10]:>10}"
    output.append(header)
    output.append("-" * len(header))

    for scenario in scenarios:
        row = f"{scenario:<10}"
        for method in methods:
            # Filter trials for this scenario and method
            filtered = [t for t in trials
                       if get_field(t, 'scenario') == scenario
                       and get_field(t, 'method') == method]

            if not filtered:
                row += f" {'--':>10}"
                continue

            # Compute VR_exec for this cell
            executed = sum(1 for t in filtered if to_bool(get_field(t, 'did_execute', False)))
            violations = sum(1 for t in filtered
                            if to_bool(get_field(t, 'did_execute', False))
                            and to_bool(get_field(t, 'actual_violation', False)))

            vr_exec = violations / executed if executed > 0 else 0
            row += f" {vr_exec:>9.1%}"

        output.append(row)

    return "\n".join(output)


def generate_table3(trials: List[Dict]) -> str:
    """
    Generate Table 3: Ablation Study

    Shows impact of each margin term
    """
    output = []
    output.append("=" * 80)
    output.append("TABLE 3: Ablation Study (Geofence Method)")
    output.append("=" * 80)
    output.append("")

    # Get ablation conditions from trials
    ablation_conditions = set()
    for t in trials:
        cond = get_field(t, 'ablation_condition', 'full')
        if cond:
            ablation_conditions.add(cond)

    if not ablation_conditions:
        ablation_conditions = {'full'}

    # Header
    header = f"{'Condition':<20} {'VR_exec↓':>10} {'MD_mean':>10} {'MD_p5':>10} {'TCR↑':>10}"
    output.append(header)
    output.append("-" * len(header))

    for condition in sorted(ablation_conditions):
        # Filter to geofence method with this ablation condition
        filtered = [t for t in trials
                   if get_field(t, 'method') == 'geofence'
                   and get_field(t, 'ablation_condition', 'full') == condition]

        if not filtered:
            continue

        # Compute metrics
        executed = sum(1 for t in filtered if to_bool(get_field(t, 'did_execute', False)))
        violations = sum(1 for t in filtered
                        if to_bool(get_field(t, 'did_execute', False))
                        and to_bool(get_field(t, 'actual_violation', False)))
        vr_exec = violations / executed if executed > 0 else 0

        distances = [to_float(get_field(t, 'min_distance_to_forbidden', float('inf')))
                    for t in filtered]
        distances = [d for d in distances if d < float('inf') and d > -10]
        md_mean = statistics.mean(distances) if distances else 0
        md_p5 = sorted(distances)[int(0.05 * len(distances))] if distances else 0

        goal_reached = sum(1 for t in filtered if to_bool(get_field(t, 'goal_reached', False)))
        tcr = goal_reached / len(filtered) if filtered else 0

        row = f"{condition:<20} {vr_exec:>9.1%} {md_mean:>9.2f}m {md_p5:>9.2f}m {tcr:>9.1%}"
        output.append(row)

    return "\n".join(output)


def generate_table4(trials: List[Dict], methods: List[str]) -> str:
    """
    Generate Table 4: Runtime & Real-time Feasibility
    """
    output = []
    output.append("=" * 80)
    output.append("TABLE 4: Runtime & Real-time Feasibility")
    output.append("=" * 80)
    output.append("")

    # Header
    header = f"{'Method':<15} {'RT_min':>10} {'RT_med':>10} {'RT_p95':>10} {'RT_max':>10} {'<10ms':>8}"
    output.append(header)
    output.append("-" * len(header))

    for method in methods:
        stats = compute_method_stats(trials, method)
        if not stats.latencies:
            continue

        rt_min = min(stats.latencies)
        rt_max = max(stats.latencies)
        under_10ms = sum(1 for l in stats.latencies if l < 10) / len(stats.latencies)

        row = f"{method:<15} {rt_min:>9.2f} {stats.rt_median:>9.2f} {stats.rt_p95:>9.2f} {rt_max:>9.2f} {under_10ms:>7.1%}"
        output.append(row)

    output.append("")
    output.append("Note: All times in milliseconds")
    output.append("      <10ms shows percentage meeting real-time constraint")

    return "\n".join(output)


def generate_summary(trials: List[Dict]) -> str:
    """Generate a quick summary of the experiment"""
    output = []
    output.append("=" * 80)
    output.append("EXPERIMENT SUMMARY")
    output.append("=" * 80)
    output.append("")

    # Count trials by type
    methods = set(get_field(t, 'method') for t in trials if get_field(t, 'method'))
    scenarios = set(get_field(t, 'scenario') for t in trials if get_field(t, 'scenario'))

    total = len(trials)
    executed = sum(1 for t in trials if to_bool(get_field(t, 'did_execute', False)))
    violations = sum(1 for t in trials if to_bool(get_field(t, 'actual_violation', False)))
    viol_in_exec = sum(1 for t in trials
                       if to_bool(get_field(t, 'did_execute', False))
                       and to_bool(get_field(t, 'actual_violation', False)))

    output.append(f"Total trials:     {total}")
    output.append(f"Executed trials:  {executed} ({100*executed/total:.1f}%)")
    output.append(f"Total violations: {violations}")
    output.append(f"VR (all):         {100*violations/total:.1f}%")
    output.append(f"VR_exec:          {100*viol_in_exec/executed:.1f}%" if executed > 0 else "VR_exec:          N/A")
    output.append("")
    output.append(f"Methods tested:   {len(methods)}: {', '.join(sorted(methods))}")
    output.append(f"Scenarios:        {len(scenarios)}: {', '.join(sorted(scenarios))}")

    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(description="Generate USENIX tables from experiment results")
    parser.add_argument("path", help="Path to experiment directory or results file")
    parser.add_argument("--table", type=int, choices=[1, 2, 3, 4],
                       help="Generate only specific table")
    parser.add_argument("--output", "-o", help="Output file (default: stdout)")
    parser.add_argument("--format", choices=["text", "latex", "csv"], default="text",
                       help="Output format")

    args = parser.parse_args()

    # Load results
    trials = load_results(args.path)
    if not trials:
        print("No trials found!", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(trials)} trials", file=sys.stderr)

    # Detect methods and scenarios
    methods = sorted(set(get_field(t, 'method') for t in trials if get_field(t, 'method')))
    scenarios = sorted(set(get_field(t, 'scenario') for t in trials if get_field(t, 'scenario')))

    # Generate output
    output_parts = []

    output_parts.append(generate_summary(trials))
    output_parts.append("")

    if args.table is None or args.table == 1:
        output_parts.append(generate_table1(trials, methods))
        output_parts.append("")

    if args.table is None or args.table == 2:
        output_parts.append(generate_table2(trials, methods, scenarios))
        output_parts.append("")

    if args.table is None or args.table == 3:
        output_parts.append(generate_table3(trials))
        output_parts.append("")

    if args.table is None or args.table == 4:
        output_parts.append(generate_table4(trials, methods))

    output = "\n".join(output_parts)

    # Write output
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"Output written to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
