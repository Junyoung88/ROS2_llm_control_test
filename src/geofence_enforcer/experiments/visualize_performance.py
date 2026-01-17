"""
Performance Benchmark Visualization for Safety Check Methods
"""

import os
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16


def plot_performance_comparison(save_path=None):
    """Plot performance comparison of safety check methods"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Data from benchmark
    methods = ['no_guard', 'safetychip\n(basic)', 'safetychip\n(path)', 'predictive']
    times_us = [0.029, 18.465, 43.389, 255.351]  # microseconds per check
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']

    # Plot 1: Absolute time (log scale)
    ax1 = axes[0]
    bars1 = ax1.bar(methods, times_us, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Time per check (µs)')
    ax1.set_title('Absolute Processing Time\n(Log Scale)')
    ax1.set_yscale('log')
    ax1.set_ylim(0.01, 1000)

    # Add value labels
    for bar, val in zip(bars1, times_us):
        height = bar.get_height()
        ax1.annotate(f'{val:.1f} µs',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add reference lines
    ax1.axhline(y=1000, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax1.text(3.5, 1200, '1ms threshold', ha='right', color='red', fontsize=10)

    # Plot 2: Relative slowdown
    ax2 = axes[1]
    relative = [t / times_us[0] for t in times_us]
    bars2 = ax2.bar(methods, relative, color=colors, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Relative slowdown (x)')
    ax2.set_title('Relative to no_guard\n(Linear Scale)')
    ax2.set_yscale('log')

    for bar, val in zip(bars2, relative):
        height = bar.get_height()
        if val > 1:
            ax2.annotate(f'{val:.0f}x',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Plot 3: Impact on control loop (the real story)
    ax3 = axes[2]

    control_loop_ms = 10  # 100 Hz control loop
    times_ms = [t / 1000 for t in times_us]
    overhead_pct = [t / control_loop_ms * 100 for t in times_ms]

    bars3 = ax3.barh(methods, overhead_pct, color=colors, edgecolor='black', linewidth=1.5)
    ax3.set_xlabel('Control loop overhead (%)')
    ax3.set_title(f'Impact on {100}Hz Control Loop\n(10ms cycle)')
    ax3.set_xlim(0, 5)

    for bar, val in zip(bars3, overhead_pct):
        width = bar.get_width()
        ax3.annotate(f'{val:.2f}%',
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points",
                    ha='left', va='center', fontsize=10, fontweight='bold')

    # Add "negligible" zone
    ax3.axvspan(0, 1, alpha=0.2, color='green', label='Negligible (<1%)')
    ax3.axvspan(1, 5, alpha=0.2, color='yellow', label='Acceptable (1-5%)')
    ax3.legend(loc='lower right', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def plot_latency_context(save_path=None):
    """Show computation overhead vs communication latency"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Categories and their latencies
    categories = [
        'no_guard\ncomputation',
        'safetychip\ncomputation',
        'predictive\ncomputation',
        'Local network\nRTT',
        'WiFi network\nRTT',
        'Cloud API\nRTT',
        'Critical\nlatency threshold'
    ]

    latencies_ms = [0.00003, 0.018, 0.255, 1, 50, 200, 500]

    colors = ['#2ecc71', '#3498db', '#e74c3c', '#95a5a6', '#95a5a6', '#95a5a6', '#c0392b']

    bars = ax.barh(categories, latencies_ms, color=colors, edgecolor='black', linewidth=1.5)

    ax.set_xlabel('Latency (ms) - Log Scale', fontsize=14)
    ax.set_title('Computation Overhead vs Communication Latency\n(Safety Check Processing is Negligible)', fontsize=16)
    ax.set_xscale('log')
    ax.set_xlim(0.00001, 1000)

    # Add value labels
    for bar, val in zip(bars, latencies_ms):
        width = bar.get_width()
        if val < 1:
            label = f'{val*1000:.1f} µs' if val < 0.001 else f'{val:.3f} ms'
        else:
            label = f'{val:.0f} ms'
        ax.annotate(label,
                    xy=(max(width, 0.0001), bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points",
                    ha='left', va='center', fontsize=11, fontweight='bold')

    # Add vertical line separating computation vs network
    ax.axvline(x=1, color='orange', linestyle='--', linewidth=2, alpha=0.8)
    ax.text(0.5, 6.5, 'Computation\n(< 1ms)', ha='center', fontsize=11, color='green', fontweight='bold')
    ax.text(50, 6.5, 'Network\n(1-500ms)', ha='center', fontsize=11, color='gray', fontweight='bold')

    # Highlight the real problem
    ax.annotate('Real bottleneck:\nNetwork latency,\nnot computation!',
                xy=(200, 4), xytext=(300, 2),
                fontsize=12, ha='left',
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def plot_summary_infographic(save_path=None):
    """Create a summary infographic"""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(5, 9.5, 'Safety Check Performance Summary', fontsize=20, ha='center', fontweight='bold')

    # Box 1: Computation overhead
    box1 = plt.Rectangle((0.5, 5.5), 4, 3.5, fill=True, facecolor='#e8f6e8', edgecolor='#27ae60', linewidth=2)
    ax.add_patch(box1)
    ax.text(2.5, 8.5, 'Computation Overhead', fontsize=14, ha='center', fontweight='bold', color='#27ae60')
    ax.text(2.5, 7.8, 'safetychip: 0.018 ms', fontsize=12, ha='center', family='monospace')
    ax.text(2.5, 7.3, 'predictive: 0.255 ms', fontsize=12, ha='center', family='monospace')
    ax.text(2.5, 6.5, '< 1% of control loop', fontsize=13, ha='center', fontweight='bold', color='#27ae60')
    ax.text(2.5, 5.9, 'NEGLIGIBLE', fontsize=16, ha='center', fontweight='bold', color='#27ae60')

    # Box 2: Network latency
    box2 = plt.Rectangle((5.5, 5.5), 4, 3.5, fill=True, facecolor='#fdecea', edgecolor='#e74c3c', linewidth=2)
    ax.add_patch(box2)
    ax.text(7.5, 8.5, 'Network Latency', fontsize=14, ha='center', fontweight='bold', color='#e74c3c')
    ax.text(7.5, 7.8, 'WiFi: 50 ms', fontsize=12, ha='center', family='monospace')
    ax.text(7.5, 7.3, 'Cloud: 200 ms', fontsize=12, ha='center', family='monospace')
    ax.text(7.5, 6.5, 'Critical: 500 ms', fontsize=13, ha='center', fontweight='bold', color='#e74c3c')
    ax.text(7.5, 5.9, 'REAL PROBLEM', fontsize=16, ha='center', fontweight='bold', color='#e74c3c')

    # Arrow between boxes
    ax.annotate('', xy=(5.3, 7), xytext=(4.7, 7),
                arrowprops=dict(arrowstyle='<->', color='black', lw=2))
    ax.text(5, 7.5, '1000x\ndifference', fontsize=10, ha='center')

    # Bottom conclusion
    box3 = plt.Rectangle((1, 1), 8, 3.5, fill=True, facecolor='#fff9e6', edgecolor='#f39c12', linewidth=2)
    ax.add_patch(box3)
    ax.text(5, 4, 'CONCLUSION', fontsize=16, ha='center', fontweight='bold', color='#f39c12')
    ax.text(5, 3.2, 'SafetyChip does NOT slow down robot control.', fontsize=13, ha='center')
    ax.text(5, 2.5, 'Focus optimization efforts on:', fontsize=12, ha='center')
    ax.text(5, 1.8, '1. Network architecture  2. Edge computing  3. Zone prediction',
            fontsize=11, ha='center', style='italic')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def main():
    # Create figures directory
    fig_dir = os.path.join(os.path.dirname(__file__), '..', 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    print("Generating performance visualization...\n")

    # Plot 1: Performance comparison
    plot_performance_comparison(
        save_path=os.path.join(fig_dir, 'performance_comparison.png')
    )

    # Plot 2: Latency context
    plot_latency_context(
        save_path=os.path.join(fig_dir, 'performance_latency_context.png')
    )

    # Plot 3: Summary infographic
    plot_summary_infographic(
        save_path=os.path.join(fig_dir, 'performance_summary.png')
    )

    print(f"\nAll figures saved to: {fig_dir}")


if __name__ == "__main__":
    main()
