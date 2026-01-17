#!/usr/bin/env python3
"""
Statistical Analysis for SCIE Experiment Results

Usage:
    python analyze_scie_results.py results.csv
    python analyze_scie_results.py results.csv --output analysis_report.json
    python analyze_scie_results.py results.csv --latex  # Generate LaTeX tables

Output:
    - Comprehensive statistical analysis report
    - ANOVA results with effect sizes
    - Post-hoc comparisons
    - LaTeX tables for paper
"""

import argparse
import sys
import json
from pathlib import Path

# Add framework to path
sys.path.insert(0, str(Path(__file__).parent))

from scie_framework.statistical_analysis import StatisticalAnalyzer, quick_analysis


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze SCIE experiment results with comprehensive statistics"
    )

    parser.add_argument('input', type=str,
                       help='Path to results CSV file')

    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output path for analysis JSON (default: input_analysis.json)')

    parser.add_argument('--latex', action='store_true',
                       help='Generate LaTeX tables')

    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Print detailed output')

    return parser.parse_args()


def print_analysis_report(results: dict, verbose: bool = False):
    """Print formatted analysis report"""
    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS REPORT")
    print("=" * 70)

    # Metadata
    meta = results.get('metadata', {})
    print(f"\nDataset: {meta.get('n_trials', 'N/A')} trials")
    print(f"Methods: {meta.get('methods', [])}")
    print(f"Scenarios: {meta.get('scenarios', [])}")

    # Safety Analysis
    if 'safety_by_method' in results:
        print("\n" + "-" * 50)
        print("SAFETY ANALYSIS")
        print("-" * 50)

        safety = results['safety_by_method']

        print("\nAccuracy by Method:")
        for method, acc in safety.get('accuracy_by_method', {}).items():
            print(f"  {method:15s}: {acc*100:6.1f}%")

        if 'chi_square' in safety:
            chi = safety['chi_square']
            sig = "***" if chi['p_value'] < 0.001 else "**" if chi['p_value'] < 0.01 else "*" if chi['p_value'] < 0.05 else ""
            print(f"\nChi-square test: χ²={chi['statistic']:.2f}, p={chi['p_value']:.4f}{sig}")
            print(f"  Cramer's V = {chi['effect_size']:.3f}")

        if 'pairwise_fisher' in safety:
            print("\nPairwise comparisons (Fisher's exact):")
            for pair, result in safety['pairwise_fisher'].items():
                sig = "*" if result['significant'] else ""
                print(f"  {pair}: p={result['p_value']:.4f}{sig}")

    # Latency Analysis
    if 'latency_by_method' in results:
        print("\n" + "-" * 50)
        print("LATENCY ANALYSIS")
        print("-" * 50)

        latency = results['latency_by_method']

        print("\nDescriptive Statistics:")
        print(f"  {'Method':15s} {'Mean':>10s} {'SD':>10s} {'Median':>10s} {'95% CI':>20s}")
        for method, stats in latency.get('descriptive', {}).items():
            ci = stats.get('ci_95', [0, 0])
            print(f"  {method:15s} {stats['mean']:10.1f} {stats['std']:10.1f} "
                  f"{stats['median']:10.1f} [{ci[0]:.1f}, {ci[1]:.1f}]")

        # Normality
        if verbose and 'normality' in latency:
            print("\nNormality tests (Shapiro-Wilk):")
            for method, result in latency['normality'].items():
                sig = "*" if result['significant'] else ""
                print(f"  {method}: W={result['statistic']:.3f}, p={result['p_value']:.4f}{sig}")

        # Main test
        if 'main_test' in latency:
            test = latency['main_test']
            sig = "***" if test['p_value'] < 0.001 else "**" if test['p_value'] < 0.01 else "*" if test['p_value'] < 0.05 else ""
            print(f"\n{test['test_name']}:")
            print(f"  Statistic = {test['statistic']:.2f}, p = {test['p_value']:.4f}{sig}")
            if test.get('effect_size'):
                print(f"  Effect size ({test['effect_size_type']}) = {test['effect_size']:.3f} ({test.get('interpretation', '')})")

    # Scenario Analysis
    if 'by_scenario' in results:
        print("\n" + "-" * 50)
        print("SCENARIO ANALYSIS")
        print("-" * 50)

        scenario = results['by_scenario']

        print("\nAccuracy by Scenario:")
        for s, acc in scenario.get('accuracy_by_scenario', {}).items():
            print(f"  {s}: {acc*100:.1f}%")

        if verbose and 'two_way_anova' in scenario:
            print("\nTwo-way ANOVA (Method × Scenario):")
            for effect, result in scenario['two_way_anova'].items():
                sig = "***" if result['p_value'] < 0.001 else "**" if result['p_value'] < 0.01 else "*" if result['p_value'] < 0.05 else ""
                print(f"  {effect}: F={result['statistic']:.2f}, p={result['p_value']:.4f}{sig}, "
                      f"η²={result['effect_size']:.3f}")

    print("\n" + "=" * 70)
    print("Legend: * p<.05, ** p<.01, *** p<.001")
    print("=" * 70)


def generate_latex_tables(analyzer: StatisticalAnalyzer, output_dir: Path):
    """Generate LaTeX tables for paper"""
    print("\nGenerating LaTeX tables...")

    # Safety table
    safety_latex = analyzer.generate_latex_table('safety')
    safety_path = output_dir / "table_safety.tex"
    with open(safety_path, 'w') as f:
        f.write(safety_latex)
    print(f"  Saved: {safety_path}")

    # Latency table
    latency_latex = analyzer.generate_latex_table('latency')
    latency_path = output_dir / "table_latency.tex"
    with open(latency_path, 'w') as f:
        f.write(latency_latex)
    print(f"  Saved: {latency_path}")


def main():
    args = parse_args()

    # Check input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        sys.exit(1)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_suffix('.analysis.json')

    print(f"\nAnalyzing: {input_path}")

    # Run analysis
    analyzer = StatisticalAnalyzer()
    analyzer.load_data(str(input_path))
    results = analyzer.analyze_all()

    # Print report
    print_analysis_report(results, verbose=args.verbose)

    # Save results
    analyzer.save_results(str(output_path))
    print(f"\nAnalysis saved to: {output_path}")

    # Generate LaTeX tables if requested
    if args.latex:
        generate_latex_tables(analyzer, output_path.parent)

    # Export for R
    r_path = input_path.with_suffix('.r.csv')
    try:
        analyzer.export_for_r(str(r_path))
        print(f"R-compatible CSV: {r_path}")
    except Exception as e:
        print(f"Note: R export failed: {e}")


if __name__ == '__main__':
    main()
