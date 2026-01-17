"""
Visualization Module for SCIE Paper Figures

Generates publication-ready figures following academic standards:
- High DPI (300+) for print quality
- Consistent color schemes
- Proper axis labels and legends
- Statistical annotations (significance bars, error bars)
- Multiple export formats (PDF, PNG, SVG)

Figure types:
1. Bar plots with error bars (method comparison)
2. Box plots (distribution comparison)
3. Heatmaps (method × scenario interaction)
4. Line plots (latency over trials)
5. Confusion matrices
6. Scenario success rate plots
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import warnings

try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    np = None
    pd = None
    plt = None
    sns = None
    warnings.warn("matplotlib/seaborn not available - visualization disabled")


@dataclass
class FigureStyle:
    """Publication figure style configuration"""
    # Size (inches)
    width: float = 7.0          # Single column: 3.5, Double column: 7.0
    height: float = 5.0
    dpi: int = 300

    # Fonts
    font_family: str = "serif"
    font_size: int = 10
    title_size: int = 12
    label_size: int = 10
    tick_size: int = 9
    legend_size: int = 9

    # Colors (colorblind-friendly)
    colors: List[str] = None
    cmap: str = "viridis"

    # Grid and spines
    grid: bool = True
    grid_alpha: float = 0.3
    spine_width: float = 0.8

    def __post_init__(self):
        if self.colors is None:
            # Colorblind-friendly palette
            self.colors = [
                "#4477AA",  # Blue
                "#EE6677",  # Red
                "#228833",  # Green
                "#CCBB44",  # Yellow
                "#66CCEE",  # Cyan
                "#AA3377",  # Purple
                "#BBBBBB",  # Gray
            ]

    def apply(self):
        """Apply style to matplotlib"""
        if not MATPLOTLIB_AVAILABLE:
            return

        plt.rcParams.update({
            'font.family': self.font_family,
            'font.size': self.font_size,
            'axes.titlesize': self.title_size,
            'axes.labelsize': self.label_size,
            'xtick.labelsize': self.tick_size,
            'ytick.labelsize': self.tick_size,
            'legend.fontsize': self.legend_size,
            'figure.dpi': self.dpi,
            'savefig.dpi': self.dpi,
            'axes.linewidth': self.spine_width,
            'axes.grid': self.grid,
            'grid.alpha': self.grid_alpha,
        })

        sns.set_palette(self.colors)


# SCIE-standard style
SCIE_STYLE = FigureStyle(
    width=7.0,
    height=5.0,
    dpi=300,
    font_family="serif",
    font_size=10
)

# Method display names and colors
METHOD_NAMES = {
    "no_guard": "No Guard",
    "safetychip": "SafetyChip",
    "selp": "SELP",
    "geofence": "Geofence (Ours)"
}

METHOD_COLORS = {
    "no_guard": "#BBBBBB",
    "safetychip": "#4477AA",
    "selp": "#EE6677",
    "geofence": "#228833"
}

SCENARIO_NAMES = {
    "S1": "Direct Hazard",
    "S2": "Salami Attack",
    "S3": "Drift Attack",
    "S4": "Multi-Zone",
    "S5": "Velocity",
    "S6": "Spoofing",
    "S7": "Latency"
}


class PaperFigureGenerator:
    """
    Generates publication-ready figures for SCIE papers.
    """

    def __init__(self, data: 'pd.DataFrame' = None,
                 output_dir: str = "./figures",
                 style: FigureStyle = None):
        """
        Initialize figure generator.

        Args:
            data: Experiment results DataFrame
            output_dir: Directory to save figures
            style: Figure style configuration
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib and seaborn required for visualization")

        self.data = data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.style = style or SCIE_STYLE
        self.style.apply()

    def load_data(self, csv_path: str) -> 'PaperFigureGenerator':
        """Load data from CSV"""
        self.data = pd.read_csv(csv_path)
        return self

    def _get_method_order(self) -> List[str]:
        """Get consistent method ordering"""
        return ["no_guard", "safetychip", "selp", "geofence"]

    def _get_scenario_order(self) -> List[str]:
        """Get consistent scenario ordering"""
        return ["S1", "S2", "S3", "S4", "S5", "S6", "S7"]

    def _save_figure(self, fig, name: str, formats: List[str] = None):
        """Save figure in multiple formats"""
        if formats is None:
            formats = ["pdf", "png"]

        for fmt in formats:
            path = self.output_dir / f"{name}.{fmt}"
            fig.savefig(path, format=fmt, bbox_inches='tight',
                       dpi=self.style.dpi, facecolor='white')
            print(f"Saved: {path}")

    # =========================================================================
    # Main Comparison Figures
    # =========================================================================

    def plot_accuracy_by_method(self, save: bool = True) -> 'plt.Figure':
        """
        Bar plot: Safety accuracy by method.

        This is typically Figure 1 in safety papers.
        """
        fig, ax = plt.subplots(figsize=(self.style.width * 0.6, self.style.height))

        # Calculate accuracy and confidence intervals
        methods = self._get_method_order()
        methods = [m for m in methods if m in self.data['method'].unique()]

        accuracies = []
        ci_lower = []
        ci_upper = []

        for method in methods:
            subset = self.data[self.data['method'] == method]['is_correct']
            acc = subset.mean()
            n = len(subset)

            # Wilson score interval for proportions
            z = 1.96  # 95% CI
            p = acc
            denom = 1 + z**2 / n
            center = (p + z**2 / (2*n)) / denom
            margin = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom

            accuracies.append(acc * 100)
            ci_lower.append(max(0, (center - margin) * 100))
            ci_upper.append(min(100, (center + margin) * 100))

        # Plot bars
        x = np.arange(len(methods))
        colors = [METHOD_COLORS.get(m, '#888888') for m in methods]
        bars = ax.bar(x, accuracies, color=colors, edgecolor='black', linewidth=0.5)

        # Error bars
        yerr_lower = [a - l for a, l in zip(accuracies, ci_lower)]
        yerr_upper = [u - a for a, u in zip(accuracies, ci_upper)]
        ax.errorbar(x, accuracies, yerr=[yerr_lower, yerr_upper],
                   fmt='none', color='black', capsize=3, capthick=1)

        # Labels
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_NAMES.get(m, m) for m in methods], rotation=15, ha='right')
        ax.set_ylabel('Accuracy (%)')
        ax.set_ylim(0, 105)
        ax.set_title('Safety Decision Accuracy by Method')

        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            ax.annotate(f'{acc:.1f}%',
                       xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 3), textcoords='offset points',
                       ha='center', va='bottom', fontsize=8)

        plt.tight_layout()

        if save:
            self._save_figure(fig, "accuracy_by_method")

        return fig

    def plot_latency_boxplot(self, save: bool = True) -> 'plt.Figure':
        """
        Box plot: Decision latency by method.
        """
        fig, ax = plt.subplots(figsize=(self.style.width * 0.6, self.style.height))

        methods = self._get_method_order()
        methods = [m for m in methods if m in self.data['method'].unique()]

        # Prepare data for seaborn
        plot_data = self.data[self.data['method'].isin(methods)].copy()
        plot_data['method_display'] = plot_data['method'].map(METHOD_NAMES)

        # Create ordered categorical
        order = [METHOD_NAMES.get(m, m) for m in methods]
        palette = {METHOD_NAMES.get(m, m): METHOD_COLORS.get(m, '#888888') for m in methods}

        sns.boxplot(x='method_display', y='decision_latency_ms',
                   data=plot_data, order=order, palette=palette, ax=ax)

        ax.set_xlabel('')
        ax.set_ylabel('Decision Latency (ms)')
        ax.set_title('Decision Latency Distribution by Method')
        plt.xticks(rotation=15, ha='right')

        plt.tight_layout()

        if save:
            self._save_figure(fig, "latency_boxplot")

        return fig

    def plot_accuracy_heatmap(self, save: bool = True) -> 'plt.Figure':
        """
        Heatmap: Accuracy by method × scenario.

        Shows interaction effects clearly.
        """
        fig, ax = plt.subplots(figsize=(self.style.width, self.style.height * 0.8))

        methods = self._get_method_order()
        scenarios = self._get_scenario_order()

        # Filter to available
        methods = [m for m in methods if m in self.data['method'].unique()]
        scenarios = [s for s in scenarios if s in self.data['scenario'].unique()]

        # Create pivot table
        pivot = self.data.pivot_table(
            values='is_correct',
            index='method',
            columns='scenario',
            aggfunc='mean'
        ) * 100

        # Reorder
        pivot = pivot.reindex(index=methods, columns=scenarios)

        # Rename for display
        pivot.index = [METHOD_NAMES.get(m, m) for m in pivot.index]
        pivot.columns = [SCENARIO_NAMES.get(s, s) for s in pivot.columns]

        # Plot heatmap
        sns.heatmap(pivot, annot=True, fmt='.0f', cmap='RdYlGn',
                   vmin=0, vmax=100, ax=ax,
                   cbar_kws={'label': 'Accuracy (%)'})

        ax.set_xlabel('Scenario')
        ax.set_ylabel('Method')
        ax.set_title('Safety Accuracy: Method × Scenario')

        plt.tight_layout()

        if save:
            self._save_figure(fig, "accuracy_heatmap")

        return fig

    def plot_scenario_comparison(self, save: bool = True) -> 'plt.Figure':
        """
        Grouped bar plot: Accuracy by scenario for each method.
        """
        fig, ax = plt.subplots(figsize=(self.style.width, self.style.height))

        methods = self._get_method_order()
        scenarios = self._get_scenario_order()

        methods = [m for m in methods if m in self.data['method'].unique()]
        scenarios = [s for s in scenarios if s in self.data['scenario'].unique()]

        x = np.arange(len(scenarios))
        width = 0.2
        offsets = np.linspace(-width*(len(methods)-1)/2, width*(len(methods)-1)/2, len(methods))

        for i, method in enumerate(methods):
            subset = self.data[self.data['method'] == method]
            accuracies = [
                subset[subset['scenario'] == s]['is_correct'].mean() * 100
                if s in subset['scenario'].values else 0
                for s in scenarios
            ]

            bars = ax.bar(x + offsets[i], accuracies, width,
                         label=METHOD_NAMES.get(method, method),
                         color=METHOD_COLORS.get(method, '#888888'),
                         edgecolor='black', linewidth=0.3)

        ax.set_xticks(x)
        ax.set_xticklabels([SCENARIO_NAMES.get(s, s) for s in scenarios],
                          rotation=45, ha='right')
        ax.set_ylabel('Accuracy (%)')
        ax.set_ylim(0, 105)
        ax.set_title('Safety Accuracy by Scenario and Method')
        ax.legend(loc='lower right', ncol=2)

        plt.tight_layout()

        if save:
            self._save_figure(fig, "scenario_comparison")

        return fig

    def plot_latency_by_scenario(self, save: bool = True) -> 'plt.Figure':
        """
        Line plot: Mean latency by scenario for each method.
        """
        fig, ax = plt.subplots(figsize=(self.style.width, self.style.height * 0.8))

        methods = self._get_method_order()
        scenarios = self._get_scenario_order()

        methods = [m for m in methods if m in self.data['method'].unique()]
        scenarios = [s for s in scenarios if s in self.data['scenario'].unique()]

        for method in methods:
            subset = self.data[self.data['method'] == method]
            latencies = []
            errors = []

            for scenario in scenarios:
                s_data = subset[subset['scenario'] == scenario]['decision_latency_ms']
                if len(s_data) > 0:
                    latencies.append(s_data.mean())
                    errors.append(s_data.std() / np.sqrt(len(s_data)))
                else:
                    latencies.append(np.nan)
                    errors.append(np.nan)

            ax.errorbar(range(len(scenarios)), latencies, yerr=errors,
                       label=METHOD_NAMES.get(method, method),
                       color=METHOD_COLORS.get(method, '#888888'),
                       marker='o', capsize=3, capthick=1, linewidth=2)

        ax.set_xticks(range(len(scenarios)))
        ax.set_xticklabels([SCENARIO_NAMES.get(s, s) for s in scenarios],
                          rotation=45, ha='right')
        ax.set_ylabel('Decision Latency (ms)')
        ax.set_title('Decision Latency by Scenario')
        ax.legend(loc='upper right')

        plt.tight_layout()

        if save:
            self._save_figure(fig, "latency_by_scenario")

        return fig

    def plot_confusion_matrix(self, method: str = None, save: bool = True) -> 'plt.Figure':
        """
        Confusion matrix for safety decisions.
        """
        fig, ax = plt.subplots(figsize=(self.style.width * 0.5, self.style.height * 0.8))

        if method:
            data = self.data[self.data['method'] == method]
            title = f'Confusion Matrix: {METHOD_NAMES.get(method, method)}'
        else:
            data = self.data
            title = 'Overall Confusion Matrix'

        # Create confusion matrix
        # Rows: Expected, Columns: Actual
        expected = data['expected_decision'].fillna('unknown')
        actual = data['decision'].fillna('unknown')

        labels = ['allow', 'reject']
        cm = np.zeros((2, 2))

        for i, exp in enumerate(labels):
            for j, act in enumerate(labels):
                cm[i, j] = ((expected == exp) & (actual == act)).sum()

        # Normalize
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                   xticklabels=labels, yticklabels=labels, ax=ax,
                   vmin=0, vmax=1)

        # Add counts
        for i in range(2):
            for j in range(2):
                ax.text(j + 0.5, i + 0.7, f'(n={int(cm[i,j])})',
                       ha='center', va='center', fontsize=8, color='gray')

        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(title)

        plt.tight_layout()

        if save:
            name = f"confusion_matrix_{method}" if method else "confusion_matrix"
            self._save_figure(fig, name)

        return fig

    def plot_safety_margin_effect(self, save: bool = True) -> 'plt.Figure':
        """
        Plot showing effect of safety margin on decisions.
        Only for geofence method.
        """
        fig, ax = plt.subplots(figsize=(self.style.width * 0.7, self.style.height))

        geofence_data = self.data[self.data['method'] == 'geofence']

        if 'min_distance_to_forbidden' not in geofence_data.columns:
            ax.text(0.5, 0.5, 'No distance data available',
                   ha='center', va='center', transform=ax.transAxes)
        else:
            # Scatter plot: distance vs decision
            allow = geofence_data[geofence_data['decision'] == 'allow']['min_distance_to_forbidden']
            reject = geofence_data[geofence_data['decision'] == 'reject']['min_distance_to_forbidden']

            ax.hist([allow, reject], bins=20, label=['Allow', 'Reject'],
                   color=[METHOD_COLORS['geofence'], '#EE6677'], alpha=0.7)

            ax.axvline(x=0.45, color='red', linestyle='--', linewidth=2,
                      label='Safety Margin (0.45m)')

            ax.set_xlabel('Min Distance to Forbidden Zone (m)')
            ax.set_ylabel('Count')
            ax.set_title('Decision Distribution by Distance (Geofence)')
            ax.legend()

        plt.tight_layout()

        if save:
            self._save_figure(fig, "safety_margin_effect")

        return fig

    def generate_all_figures(self):
        """Generate all standard figures for the paper."""
        print("\nGenerating all paper figures...")
        print("=" * 50)

        figures = [
            ("1. Accuracy by Method", self.plot_accuracy_by_method),
            ("2. Latency Boxplot", self.plot_latency_boxplot),
            ("3. Accuracy Heatmap", self.plot_accuracy_heatmap),
            ("4. Scenario Comparison", self.plot_scenario_comparison),
            ("5. Latency by Scenario", self.plot_latency_by_scenario),
            ("6. Confusion Matrix", self.plot_confusion_matrix),
            ("7. Safety Margin Effect", self.plot_safety_margin_effect),
        ]

        for name, func in figures:
            print(f"\n{name}...")
            try:
                func(save=True)
                plt.close('all')
            except Exception as e:
                print(f"  Error: {e}")

        print("\n" + "=" * 50)
        print(f"Figures saved to: {self.output_dir}")

    def create_summary_figure(self, save: bool = True) -> 'plt.Figure':
        """
        Create a multi-panel summary figure for the paper.

        Combines: accuracy bar, latency box, and heatmap in one figure.
        """
        fig = plt.figure(figsize=(self.style.width * 1.5, self.style.height * 1.2))

        # Layout: 2x2 grid
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # Panel A: Accuracy bar plot
        ax1 = fig.add_subplot(gs[0, 0])
        methods = [m for m in self._get_method_order() if m in self.data['method'].unique()]
        accuracies = [self.data[self.data['method'] == m]['is_correct'].mean() * 100 for m in methods]
        colors = [METHOD_COLORS.get(m) for m in methods]
        ax1.bar(range(len(methods)), accuracies, color=colors, edgecolor='black', linewidth=0.5)
        ax1.set_xticks(range(len(methods)))
        ax1.set_xticklabels([METHOD_NAMES.get(m, m) for m in methods], rotation=15, ha='right', fontsize=8)
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_ylim(0, 105)
        ax1.set_title('(a) Safety Accuracy')

        # Panel B: Latency boxplot
        ax2 = fig.add_subplot(gs[0, 1])
        plot_data = self.data[self.data['method'].isin(methods)].copy()
        order = [METHOD_NAMES.get(m, m) for m in methods]
        palette = {METHOD_NAMES.get(m, m): METHOD_COLORS.get(m) for m in methods}
        plot_data['method_display'] = plot_data['method'].map(METHOD_NAMES)
        sns.boxplot(x='method_display', y='decision_latency_ms',
                   data=plot_data, order=order, palette=palette, ax=ax2)
        ax2.set_xlabel('')
        ax2.set_ylabel('Latency (ms)')
        ax2.set_title('(b) Decision Latency')
        plt.sca(ax2)
        plt.xticks(rotation=15, ha='right', fontsize=8)

        # Panel C: Heatmap
        ax3 = fig.add_subplot(gs[1, :])
        scenarios = [s for s in self._get_scenario_order() if s in self.data['scenario'].unique()]
        pivot = self.data.pivot_table(values='is_correct', index='method', columns='scenario', aggfunc='mean') * 100
        pivot = pivot.reindex(index=methods, columns=scenarios)
        pivot.index = [METHOD_NAMES.get(m, m) for m in pivot.index]
        pivot.columns = [SCENARIO_NAMES.get(s, s) for s in pivot.columns]
        sns.heatmap(pivot, annot=True, fmt='.0f', cmap='RdYlGn', vmin=0, vmax=100, ax=ax3,
                   cbar_kws={'label': 'Accuracy (%)'})
        ax3.set_xlabel('')
        ax3.set_ylabel('')
        ax3.set_title('(c) Accuracy by Method × Scenario')

        plt.tight_layout()

        if save:
            self._save_figure(fig, "summary_figure")

        return fig


def generate_figures_from_csv(csv_path: str, output_dir: str = "./figures"):
    """
    Convenience function to generate all figures from CSV.

    Args:
        csv_path: Path to results CSV
        output_dir: Output directory for figures
    """
    generator = PaperFigureGenerator(output_dir=output_dir)
    generator.load_data(csv_path)
    generator.generate_all_figures()
    generator.create_summary_figure()
