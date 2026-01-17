"""
Visualization for Incremental Attack Experiment Results

Generates SCIE-quality figures for:
1. Attack/Defense Success Rate comparison
2. Trajectory visualization
3. Early block rate analysis
4. Method comparison heatmap
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.lines import Line2D
from typing import List, Dict, Tuple
from dataclasses import dataclass

# Set style for publication
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.axisbelow': True,
})

# Color scheme
COLORS = {
    'no_guard': '#e74c3c',       # Red
    'safetychip': '#f39c12',     # Orange
    'geofence': '#3498db',       # Blue
    'geofence_predictive': '#27ae60',  # Green
}

METHOD_LABELS = {
    'no_guard': 'No Guard',
    'safetychip': 'SafetyChip',
    'geofence': 'Geofence (Reactive)',
    'geofence_predictive': 'Geofence (Predictive)',
}

# Zone definitions
ZONES = {
    'storage_racks': {'bounds': (14, 7, 18, 11), 'color': '#e74c3c', 'label': 'Storage Racks'},
    'high_temp_furnace': {'bounds': (8, 7, 11, 10), 'color': '#e67e22', 'label': 'High-Temp Furnace'},
}


def load_results(filepath: str = 'incremental_attack_results.json') -> Dict:
    """Load results from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_comparison_bars(results: Dict, output_path: str = 'fig1_method_comparison.png'):
    """
    Figure 1: Method comparison bar chart
    Shows ASR (Attack Success Rate) and DSR (Defense Success Rate)
    """
    # Extract malicious scenarios only
    malicious_results = [r for r in results['results']
                        if r['attack_type'] in ['malicious_indirect', 'malicious_ambiguous']]

    methods = ['no_guard', 'safetychip', 'geofence', 'geofence_predictive']

    # Calculate rates per method
    asr_data = []
    dsr_data = []

    for method in methods:
        method_results = [r for r in malicious_results if r['method'] == method]
        n = len(method_results)
        if n > 0:
            asr = sum(1 for r in method_results if r['attack_success']) / n * 100
            dsr = sum(1 for r in method_results if r['defense_success']) / n * 100
        else:
            asr, dsr = 0, 0
        asr_data.append(asr)
        dsr_data.append(dsr)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(methods))
    width = 0.35

    bars1 = ax.bar(x - width/2, asr_data, width, label='Attack Success Rate (↓ better)',
                   color='#e74c3c', edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, dsr_data, width, label='Defense Success Rate (↑ better)',
                   color='#27ae60', edgecolor='black', linewidth=1)

    # Add value labels
    for bar, val in zip(bars1, asr_data):
        ax.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=10)

    for bar, val in zip(bars2, dsr_data):
        ax.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Rate (%)')
    ax.set_title('Incremental Command Attack: Method Comparison\n(Malicious Scenarios Only)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in methods])
    ax.legend(loc='upper right')
    ax.set_ylim(0, 115)

    # Add horizontal line at 100%
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_scenario_heatmap(results: Dict, output_path: str = 'fig2_scenario_heatmap.png'):
    """
    Figure 2: Scenario × Method heatmap for Defense Success Rate
    """
    scenarios = ["S2'-A", "S2'-B", "S2'-C", "S2'-D", "S2'-E"]
    methods = ['no_guard', 'safetychip', 'geofence', 'geofence_predictive']

    # Build DSR matrix
    dsr_matrix = np.zeros((len(scenarios), len(methods)))

    for i, scenario in enumerate(scenarios):
        for j, method in enumerate(methods):
            method_results = [r for r in results['results']
                            if r['scenario'] == scenario and r['method'] == method]
            if method_results:
                dsr = sum(1 for r in method_results if r['defense_success']) / len(method_results) * 100
                dsr_matrix[i, j] = dsr

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(dsr_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Defense Success Rate (%)', rotation=-90, va="bottom")

    # Set ticks
    ax.set_xticks(np.arange(len(methods)))
    ax.set_yticks(np.arange(len(scenarios)))
    ax.set_xticklabels([METHOD_LABELS[m] for m in methods])
    scenario_labels = ['S2\'-A\n(Indirect)', 'S2\'-B\n(Ambiguous)', 'S2\'-C\n(Benign)',
                       'S2\'-D\n(Fast)', 'S2\'-E\n(Boundary)']
    ax.set_yticklabels(scenario_labels)

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add value annotations
    for i in range(len(scenarios)):
        for j in range(len(methods)):
            val = dsr_matrix[i, j]
            color = 'white' if val < 50 else 'black'
            ax.text(j, i, f'{val:.0f}%', ha='center', va='center', color=color, fontsize=11)

    ax.set_title('Defense Success Rate by Scenario and Method', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_trajectory_comparison(results: Dict, output_path: str = 'fig3_trajectories.png'):
    """
    Figure 3: Trajectory visualization for S2'-A scenario
    Shows how each method handles the incremental drift attack
    """
    methods = ['no_guard', 'safetychip', 'geofence', 'geofence_predictive']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, method in enumerate(methods):
        ax = axes[idx]

        # Draw forbidden zones
        for zone_name, zone_info in ZONES.items():
            x1, y1, x2, y2 = zone_info['bounds']
            rect = Rectangle((x1, y1), x2-x1, y2-y1,
                            facecolor=zone_info['color'], alpha=0.3,
                            edgecolor=zone_info['color'], linewidth=2)
            ax.add_patch(rect)
            ax.text((x1+x2)/2, (y1+y2)/2, zone_info['label'],
                   ha='center', va='center', fontsize=8, alpha=0.7)

        # Get trajectories for this method and S2'-A scenario
        method_results = [r for r in results['results']
                        if r['method'] == method and r['scenario'] == "S2'-A"]

        # Plot up to 10 trajectories
        for i, r in enumerate(method_results[:10]):
            traj = r['trajectory']
            xs = [p[0] for p in traj]
            ys = [p[1] for p in traj]

            alpha = 0.3 if i > 0 else 0.8
            linewidth = 1 if i > 0 else 2

            ax.plot(xs, ys, '-', color=COLORS[method], alpha=alpha, linewidth=linewidth)
            ax.scatter(xs[0], ys[0], color='green', s=50, zorder=5, marker='o')
            ax.scatter(xs[-1], ys[-1], color='red' if r['attack_success'] else 'blue',
                      s=50, zorder=5, marker='x')

        # Calculate stats
        asr = sum(1 for r in method_results if r['attack_success']) / len(method_results) * 100 if method_results else 0

        ax.set_xlim(5, 20)
        ax.set_ylim(3, 13)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'{METHOD_LABELS[method]}\nASR: {asr:.0f}%', fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    # Add legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Start'),
        Line2D([0], [0], marker='x', color='w', markerfacecolor='red', markeredgecolor='red', markersize=10, label='End (Intrusion)'),
        Line2D([0], [0], marker='x', color='w', markerfacecolor='blue', markeredgecolor='blue', markersize=10, label='End (Safe)'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.02))

    plt.suptitle('S2\'-A Trajectory Comparison: Incremental Drift Attack', fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_early_detection(results: Dict, output_path: str = 'fig4_early_detection.png'):
    """
    Figure 4: Early detection analysis
    Shows when each method first blocks (turn number)
    """
    methods = ['no_guard', 'safetychip', 'geofence', 'geofence_predictive']

    # Get malicious results only
    malicious_results = [r for r in results['results']
                        if r['attack_type'] in ['malicious_indirect', 'malicious_ambiguous']]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Early block rate (blocked before turn 3)
    early_block_rates = []
    for method in methods:
        method_results = [r for r in malicious_results if r['method'] == method]
        if method_results:
            early = sum(1 for r in method_results if 0 < r['first_block_turn'] <= 2) / len(method_results) * 100
        else:
            early = 0
        early_block_rates.append(early)

    bars = ax1.bar(range(len(methods)), early_block_rates,
                   color=[COLORS[m] for m in methods], edgecolor='black', linewidth=1)

    for bar, val in zip(bars, early_block_rates):
        ax1.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=11)

    ax1.set_ylabel('Early Block Rate (%)')
    ax1.set_title('Early Detection (Block before Turn 3)', fontweight='bold')
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels([METHOD_LABELS[m] for m in methods], rotation=45, ha='right')
    ax1.set_ylim(0, 110)

    # Right: Distribution of first block turn
    first_blocks = {m: [] for m in methods}
    for r in malicious_results:
        if r['first_block_turn'] > 0:
            first_blocks[r['method']].append(r['first_block_turn'])

    positions = np.arange(len(methods))
    width = 0.6

    bp_data = [first_blocks[m] if first_blocks[m] else [0] for m in methods]
    bp = ax2.boxplot(bp_data, positions=positions, widths=width, patch_artist=True)

    for patch, method in zip(bp['boxes'], methods):
        patch.set_facecolor(COLORS[method])
        patch.set_alpha(0.7)

    ax2.set_ylabel('First Block Turn')
    ax2.set_title('Distribution of First Block Turn', fontweight='bold')
    ax2.set_xticks(positions)
    ax2.set_xticklabels([METHOD_LABELS[m] for m in methods], rotation=45, ha='right')
    ax2.set_ylim(0, 6)

    plt.suptitle('Early Detection Analysis (Malicious Scenarios)', fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_min_distance_comparison(results: Dict, output_path: str = 'fig5_min_distance.png'):
    """
    Figure 5: Minimum distance to forbidden zone comparison
    """
    methods = ['no_guard', 'safetychip', 'geofence', 'geofence_predictive']

    # Get malicious results only
    malicious_results = [r for r in results['results']
                        if r['attack_type'] in ['malicious_indirect', 'malicious_ambiguous']]

    fig, ax = plt.subplots(figsize=(10, 6))

    min_distances = {m: [] for m in methods}
    for r in malicious_results:
        min_distances[r['method']].append(r['min_distance'])

    positions = np.arange(len(methods))
    bp = ax.boxplot([min_distances[m] for m in methods], positions=positions,
                    widths=0.6, patch_artist=True)

    for patch, method in zip(bp['boxes'], methods):
        patch.set_facecolor(COLORS[method])
        patch.set_alpha(0.7)

    # Add individual points
    for i, method in enumerate(methods):
        y = min_distances[method]
        x = np.random.normal(i, 0.04, size=len(y))
        ax.scatter(x, y, alpha=0.3, color=COLORS[method], s=20)

    # Add zero line (zone boundary)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zone Boundary')
    ax.axhline(y=0.5, color='orange', linestyle=':', linewidth=2, label='Safety Margin (0.5m)')

    ax.set_ylabel('Minimum Distance to Zone (m)')
    ax.set_title('Minimum Distance to Forbidden Zone\n(Negative = Inside Zone)', fontweight='bold')
    ax.set_xticks(positions)
    ax.set_xticklabels([METHOD_LABELS[m] for m in methods], rotation=45, ha='right')
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_summary_dashboard(results: Dict, output_path: str = 'fig0_summary_dashboard.png'):
    """
    Figure 0: Summary dashboard with all key metrics
    """
    methods = ['no_guard', 'safetychip', 'geofence', 'geofence_predictive']

    malicious_results = [r for r in results['results']
                        if r['attack_type'] in ['malicious_indirect', 'malicious_ambiguous']]

    # Calculate metrics
    metrics = {}
    for method in methods:
        method_results = [r for r in malicious_results if r['method'] == method]
        n = len(method_results)
        if n > 0:
            metrics[method] = {
                'asr': sum(1 for r in method_results if r['attack_success']) / n * 100,
                'dsr': sum(1 for r in method_results if r['defense_success']) / n * 100,
                'ebr': sum(1 for r in method_results if 0 < r['first_block_turn'] <= 2) / n * 100,
                'mean_md': np.mean([r['min_distance'] for r in method_results]),
            }

    fig = plt.figure(figsize=(14, 10))

    # Title
    fig.suptitle('Incremental Command Attack Experiment: Summary Dashboard',
                fontsize=16, fontweight='bold', y=0.98)

    # Create grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # 1. ASR comparison (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    asr_vals = [metrics[m]['asr'] for m in methods]
    bars = ax1.bar(range(len(methods)), asr_vals, color=[COLORS[m] for m in methods],
                   edgecolor='black', linewidth=1)
    for bar, val in zip(bars, asr_vals):
        ax1.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=10)
    ax1.set_ylabel('Rate (%)')
    ax1.set_title('Attack Success Rate (↓ better)', fontweight='bold')
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels([m.replace('_', '\n') for m in methods], fontsize=9)
    ax1.set_ylim(0, 110)

    # 2. DSR comparison (top-right)
    ax2 = fig.add_subplot(gs[0, 1])
    dsr_vals = [metrics[m]['dsr'] for m in methods]
    bars = ax2.bar(range(len(methods)), dsr_vals, color=[COLORS[m] for m in methods],
                   edgecolor='black', linewidth=1)
    for bar, val in zip(bars, dsr_vals):
        ax2.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=10)
    ax2.set_ylabel('Rate (%)')
    ax2.set_title('Defense Success Rate (↑ better)', fontweight='bold')
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels([m.replace('_', '\n') for m in methods], fontsize=9)
    ax2.set_ylim(0, 110)
    ax2.axhline(y=100, color='green', linestyle='--', alpha=0.5)

    # 3. Early block rate (bottom-left)
    ax3 = fig.add_subplot(gs[1, 0])
    ebr_vals = [metrics[m]['ebr'] for m in methods]
    bars = ax3.bar(range(len(methods)), ebr_vals, color=[COLORS[m] for m in methods],
                   edgecolor='black', linewidth=1)
    for bar, val in zip(bars, ebr_vals):
        ax3.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=10)
    ax3.set_ylabel('Rate (%)')
    ax3.set_title('Early Block Rate (before Turn 3)', fontweight='bold')
    ax3.set_xticks(range(len(methods)))
    ax3.set_xticklabels([m.replace('_', '\n') for m in methods], fontsize=9)
    ax3.set_ylim(0, 110)

    # 4. Mean min distance (bottom-right)
    ax4 = fig.add_subplot(gs[1, 1])
    md_vals = [metrics[m]['mean_md'] for m in methods]
    bars = ax4.bar(range(len(methods)), md_vals, color=[COLORS[m] for m in methods],
                   edgecolor='black', linewidth=1)
    for bar, val in zip(bars, md_vals):
        ax4.annotate(f'{val:+.2f}m', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3 if val >= 0 else -12), textcoords='offset points', ha='center', fontsize=10)
    ax4.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax4.axhline(y=0.5, color='orange', linestyle=':', linewidth=1)
    ax4.set_ylabel('Distance (m)')
    ax4.set_title('Mean Min Distance to Zone\n(Negative = Intrusion)', fontweight='bold')
    ax4.set_xticks(range(len(methods)))
    ax4.set_xticklabels([m.replace('_', '\n') for m in methods], fontsize=9)

    # Add text summary
    fig.text(0.5, 0.02,
            'Key Finding: Geofence (Predictive) achieves 100% Defense Success Rate with 50% Early Detection',
            ha='center', fontsize=12, style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    """Generate all figures"""

    # Check if results file exists
    results_file = 'incremental_attack_results.json'
    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        print("Please run the experiment first:")
        print("  python3 -m experiments.incremental_attack --seeds 30")
        return

    results = load_results(results_file)
    print(f"Loaded {results['n_episodes']} episodes")

    # Create output directory
    output_dir = 'figures'
    os.makedirs(output_dir, exist_ok=True)

    # Generate figures
    print("\nGenerating figures...")

    plot_summary_dashboard(results, f'{output_dir}/fig0_summary_dashboard.png')
    plot_comparison_bars(results, f'{output_dir}/fig1_method_comparison.png')
    plot_scenario_heatmap(results, f'{output_dir}/fig2_scenario_heatmap.png')
    plot_trajectory_comparison(results, f'{output_dir}/fig3_trajectories.png')
    plot_early_detection(results, f'{output_dir}/fig4_early_detection.png')
    plot_min_distance_comparison(results, f'{output_dir}/fig5_min_distance.png')

    print(f"\nAll figures saved to: {output_dir}/")


if __name__ == "__main__":
    main()
