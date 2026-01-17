"""
Visualization for Latency Simulation Experiment Results
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['figure.figsize'] = (12, 5)


def load_results():
    """Load latency results"""
    result_path = os.path.join(os.path.dirname(__file__), '..', 'latency_results.json')
    with open(result_path, 'r') as f:
        return json.load(f)


def plot_latency_vs_asr(results, save_path=None):
    """Plot ASR vs Latency for both methods"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    latencies = [0, 100, 300, 500, 1000, 2000]

    # Extract data
    dynamic_asr = []
    predictive_asr = []
    dynamic_dsr = []
    predictive_dsr = []

    by_latency = results['summary']['by_latency']

    for lat in latencies:
        lat_str = str(lat)
        dynamic_asr.append(by_latency['dynamic'].get(lat_str, {}).get('asr', 0))
        predictive_asr.append(by_latency['predictive'].get(lat_str, {}).get('asr', 0))
        dynamic_dsr.append(by_latency['dynamic'].get(lat_str, {}).get('dsr', 0))
        predictive_dsr.append(by_latency['predictive'].get(lat_str, {}).get('dsr', 0))

    # Plot 1: ASR vs Latency
    x = np.arange(len(latencies))
    width = 0.35

    bars1 = ax1.bar(x - width/2, dynamic_asr, width, label='geofence_dynamic',
                    color='#3498db', edgecolor='black', linewidth=1.2)
    bars2 = ax1.bar(x + width/2, predictive_asr, width, label='geofence_predictive',
                    color='#e74c3c', edgecolor='black', linewidth=1.2, alpha=0.8)

    ax1.set_xlabel('Latency (ms)')
    ax1.set_ylabel('Attack Success Rate (%)')
    ax1.set_title('ASR vs Communication Latency')
    ax1.set_xticks(x)
    ax1.set_xticklabels(latencies)
    ax1.legend(loc='upper left')
    ax1.set_ylim(0, 100)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax1.annotate(f'{height:.0f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

    # Add critical threshold line
    ax1.axvline(x=2.5, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax1.annotate('Critical\nThreshold', xy=(2.5, 80), xytext=(3.2, 85),
                fontsize=11, color='red',
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))

    # Highlight safe zone
    ax1.axvspan(-0.5, 2.5, alpha=0.15, color='green', label='Safe Zone (ASR=0%)')
    ax1.axvspan(2.5, 5.5, alpha=0.15, color='red', label='Danger Zone')

    # Plot 2: DSR vs Latency (line plot)
    ax2.plot(latencies, dynamic_dsr, 'o-', linewidth=2.5, markersize=10,
             label='geofence_dynamic', color='#3498db')
    ax2.plot(latencies, predictive_dsr, 's--', linewidth=2.5, markersize=10,
             label='geofence_predictive', color='#e74c3c', alpha=0.8)

    ax2.set_xlabel('Latency (ms)')
    ax2.set_ylabel('Defense Success Rate (%)')
    ax2.set_title('DSR vs Communication Latency')
    ax2.legend(loc='lower left')
    ax2.set_ylim(0, 105)
    ax2.set_xlim(-100, 2100)

    # Add annotations
    ax2.annotate('Perfect Defense\n(0-300ms)', xy=(150, 100), fontsize=11,
                ha='center', color='green', fontweight='bold')
    ax2.annotate('Defense\nDegrades', xy=(750, 50), fontsize=11,
                ha='center', color='red', fontweight='bold')

    # Add critical threshold
    ax2.axvline(x=400, color='orange', linestyle='--', linewidth=2, alpha=0.7)
    ax2.fill_betweenx([0, 105], 0, 400, alpha=0.1, color='green')
    ax2.fill_betweenx([0, 105], 400, 2100, alpha=0.1, color='red')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def plot_method_comparison(results, save_path=None):
    """Show that predictive doesn't help with latency"""
    fig, ax = plt.subplots(figsize=(10, 6))

    latencies = [0, 100, 300, 500, 1000, 2000]
    by_latency = results['summary']['by_latency']

    dynamic_asr = [by_latency['dynamic'].get(str(lat), {}).get('asr', 0) for lat in latencies]
    predictive_asr = [by_latency['predictive'].get(str(lat), {}).get('asr', 0) for lat in latencies]

    # Calculate difference
    diff = [p - d for p, d in zip(predictive_asr, dynamic_asr)]

    ax.plot(latencies, dynamic_asr, 'o-', linewidth=3, markersize=12,
            label='geofence_dynamic', color='#3498db')
    ax.plot(latencies, predictive_asr, 's-', linewidth=3, markersize=12,
            label='geofence_predictive', color='#e74c3c',
            markerfacecolor='white', markeredgewidth=2)

    ax.set_xlabel('Communication Latency (ms)', fontsize=14)
    ax.set_ylabel('Attack Success Rate (%)', fontsize=14)
    ax.set_title('Predictive Method Does NOT Improve Latency Tolerance', fontsize=16, fontweight='bold')
    ax.legend(loc='upper left', fontsize=12)
    ax.set_ylim(-5, 100)

    # Add explanation box
    textstr = '\n'.join([
        'Both methods have IDENTICAL ASR',
        'at every latency level.',
        '',
        'Reason: Predictive predicts',
        'future POSITION, but uses',
        'delayed ZONE information.',
        '',
        'It cannot predict when new',
        'zones will appear.'
    ])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.98, 0.55, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    # Highlight overlap
    ax.annotate('Lines overlap\n(identical values)',
                xy=(1000, 60), xytext=(1400, 80),
                fontsize=11, ha='center',
                arrowprops=dict(arrowstyle='->', color='black'))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def plot_safety_margin_recommendation(save_path=None):
    """Visualize recommended safety margin based on latency"""
    fig, ax = plt.subplots(figsize=(10, 6))

    latencies = np.array([0, 100, 200, 300, 400, 500, 750, 1000, 1500, 2000])
    robot_speed = 2.0  # m/s (typical fast robot)

    # Distance traveled during latency
    distance_during_latency = robot_speed * (latencies / 1000)

    # Recommended safety margin = base + latency compensation
    base_margin = 0.5  # meters
    recommended_margin = base_margin + distance_during_latency

    ax.fill_between(latencies, base_margin, recommended_margin,
                    alpha=0.3, color='orange', label='Additional margin needed')
    ax.fill_between(latencies, 0, base_margin,
                    alpha=0.3, color='green', label='Base safety margin (0.5m)')
    ax.plot(latencies, recommended_margin, 'r-', linewidth=3,
            label='Recommended total margin')
    ax.axhline(y=base_margin, color='green', linestyle='--', linewidth=2)

    ax.set_xlabel('Communication Latency (ms)', fontsize=14)
    ax.set_ylabel('Safety Margin (meters)', fontsize=14)
    ax.set_title(f'Recommended Safety Margin vs Latency\n(Robot Speed = {robot_speed} m/s)',
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    ax.set_xlim(0, 2000)
    ax.set_ylim(0, 5)

    # Add annotations
    ax.annotate(f'At 500ms latency:\n{recommended_margin[5]:.1f}m margin needed',
                xy=(500, recommended_margin[5]), xytext=(700, 2.5),
                fontsize=11, arrowprops=dict(arrowstyle='->', color='red'))

    ax.annotate(f'At 1000ms latency:\n{recommended_margin[7]:.1f}m margin needed',
                xy=(1000, recommended_margin[7]), xytext=(1200, 3.5),
                fontsize=11, arrowprops=dict(arrowstyle='->', color='red'))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def main():
    results = load_results()

    # Create figures directory
    fig_dir = os.path.join(os.path.dirname(__file__), '..', 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    print("Generating latency visualization plots...\n")

    # Plot 1: ASR/DSR vs Latency
    plot_latency_vs_asr(
        results,
        save_path=os.path.join(fig_dir, 'latency_asr_dsr.png')
    )

    # Plot 2: Method comparison
    plot_method_comparison(
        results,
        save_path=os.path.join(fig_dir, 'latency_method_comparison.png')
    )

    # Plot 3: Safety margin recommendation
    plot_safety_margin_recommendation(
        save_path=os.path.join(fig_dir, 'latency_safety_margin.png')
    )

    print(f"\nAll figures saved to: {fig_dir}")


if __name__ == "__main__":
    main()
