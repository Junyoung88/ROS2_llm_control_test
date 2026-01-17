#!/usr/bin/env python3
"""
Generate Publication-Ready Figures for SCIE Paper

Usage:
    python generate_scie_figures.py results.csv
    python generate_scie_figures.py results.csv --output ./figures
    python generate_scie_figures.py results.csv --format pdf,png,svg

Output:
    - accuracy_by_method.pdf/png    Method comparison bar plot
    - latency_boxplot.pdf/png       Latency distribution
    - accuracy_heatmap.pdf/png      Method × Scenario heatmap
    - scenario_comparison.pdf/png   Grouped bar by scenario
    - latency_by_scenario.pdf/png   Line plot
    - confusion_matrix.pdf/png      Decision confusion matrix
    - summary_figure.pdf/png        Multi-panel summary

All figures are generated at 300 DPI with publication-ready formatting.
"""

import argparse
import sys
from pathlib import Path

# Add framework to path
sys.path.insert(0, str(Path(__file__).parent))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate publication-ready figures for SCIE paper"
    )

    parser.add_argument('input', type=str,
                       help='Path to results CSV file')

    parser.add_argument('--output', '-o', type=str, default='./figures',
                       help='Output directory for figures (default: ./figures)')

    parser.add_argument('--format', '-f', type=str, default='pdf,png',
                       help='Output formats, comma-separated (default: pdf,png)')

    parser.add_argument('--dpi', type=int, default=300,
                       help='Figure DPI (default: 300)')

    parser.add_argument('--width', type=float, default=7.0,
                       help='Figure width in inches (default: 7.0 for double column)')

    parser.add_argument('--single-column', action='store_true',
                       help='Use single-column width (3.5 inches)')

    parser.add_argument('--figures', type=str, default=None,
                       help='Specific figures to generate (comma-separated)')

    parser.add_argument('--summary-only', action='store_true',
                       help='Only generate summary figure')

    return parser.parse_args()


def main():
    args = parse_args()

    # Check input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        sys.exit(1)

    # Import visualization (requires matplotlib)
    try:
        from scie_framework.visualization import (
            PaperFigureGenerator, FigureStyle, generate_figures_from_csv
        )
    except ImportError as e:
        print(f"Error: Visualization requires matplotlib and seaborn: {e}")
        print("Install with: pip install matplotlib seaborn")
        sys.exit(1)

    # Configure style
    width = 3.5 if args.single_column else args.width
    style = FigureStyle(
        width=width,
        height=width * 0.7,
        dpi=args.dpi
    )

    # Create generator
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating figures from: {input_path}")
    print(f"Output directory: {output_dir}")
    print(f"Format: {args.format}")
    print(f"DPI: {args.dpi}")
    print(f"Width: {width} inches")

    generator = PaperFigureGenerator(
        output_dir=str(output_dir),
        style=style
    )
    generator.load_data(str(input_path))

    # Generate figures
    if args.summary_only:
        print("\nGenerating summary figure only...")
        generator.create_summary_figure(save=True)
    elif args.figures:
        figure_names = [f.strip() for f in args.figures.split(',')]
        print(f"\nGenerating specific figures: {figure_names}")

        figure_map = {
            'accuracy': generator.plot_accuracy_by_method,
            'latency': generator.plot_latency_boxplot,
            'heatmap': generator.plot_accuracy_heatmap,
            'scenario': generator.plot_scenario_comparison,
            'latency_scenario': generator.plot_latency_by_scenario,
            'confusion': generator.plot_confusion_matrix,
            'margin': generator.plot_safety_margin_effect,
            'summary': generator.create_summary_figure,
        }

        for name in figure_names:
            if name in figure_map:
                print(f"  Generating: {name}")
                figure_map[name](save=True)
            else:
                print(f"  Warning: Unknown figure '{name}'")
                print(f"  Available: {list(figure_map.keys())}")
    else:
        print("\nGenerating all figures...")
        generator.generate_all_figures()
        generator.create_summary_figure(save=True)

    print(f"\nFigures saved to: {output_dir}")
    print("\nGenerated files:")
    for f in sorted(output_dir.glob("*.pdf")) + sorted(output_dir.glob("*.png")):
        print(f"  {f.name}")


if __name__ == '__main__':
    main()
