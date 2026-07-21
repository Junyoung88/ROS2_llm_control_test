#!/usr/bin/env python3
"""
Geofence Policy Enforcer System Overview - Paper Figure
Generates a publication-quality system architecture diagram
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Polygon
from matplotlib.lines import Line2D
import numpy as np

# Set publication-quality defaults
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

def draw_box(ax, x, y, width, height, text, color='lightblue',
             fontsize=9, text_color='black', rounded=True, linewidth=1.5):
    """Draw a rounded box with text"""
    if rounded:
        box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                            boxstyle="round,pad=0.02,rounding_size=0.1",
                            facecolor=color, edgecolor='black', linewidth=linewidth)
    else:
        box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                            facecolor=color, edgecolor='black', linewidth=linewidth)
    ax.add_patch(box)

    # Handle multi-line text
    lines = text.split('\n')
    line_height = 0.12
    start_y = y + (len(lines) - 1) * line_height / 2
    for i, line in enumerate(lines):
        ax.text(x, start_y - i * line_height, line, ha='center', va='center',
                fontsize=fontsize, color=text_color, weight='bold' if i == 0 else 'normal')

def draw_arrow(ax, start, end, color='black', style='->', linewidth=1.5,
               connectionstyle="arc3,rad=0"):
    """Draw an arrow between two points"""
    arrow = FancyArrowPatch(start, end, arrowstyle=style,
                           connectionstyle=connectionstyle,
                           color=color, linewidth=linewidth,
                           mutation_scale=12)
    ax.add_patch(arrow)

def create_overview_figure():
    """Create the main system overview figure"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal')
    ax.axis('off')

    # Color scheme
    colors = {
        'input': '#E8F4FD',      # Light blue for input
        'safety': '#FFE6E6',     # Light red for safety layers
        'robot': '#E6FFE6',      # Light green for robot
        'monitor': '#FFF3E6',    # Light orange for monitors
        'method': '#F0E6FF',     # Light purple for methods
        'zone': '#FFE6F0',       # Light pink for zones
    }

    # === Title ===
    ax.text(5, 7.7, 'Geofence Policy Enforcer: System Overview',
            ha='center', va='center', fontsize=14, weight='bold')

    # === Left side: Input Layer ===
    draw_box(ax, 1.5, 6.5, 2.2, 0.7, 'User / LLM\nAgent',
             color=colors['input'], fontsize=9)

    draw_box(ax, 1.5, 5.5, 2.2, 0.6, 'Goal Command\n(x, y, θ)',
             color='white', fontsize=8)

    # Arrow from user to goal
    draw_arrow(ax, (1.5, 6.15), (1.5, 5.85))

    # === Center: Three-Layer Safety Architecture ===

    # Layer 1: Goal-Level Enforcement
    draw_box(ax, 5, 6.5, 3.5, 0.9, 'Layer 1: Goal Gate\n(Goal-Level Enforcement)',
             color=colors['safety'], fontsize=9)

    # Layer 2: Path-Level Enforcement
    draw_box(ax, 5, 5.2, 3.5, 0.9, 'Layer 2: Path Watchdog\n(Path-Level Enforcement)',
             color=colors['safety'], fontsize=9)

    # Layer 3: Control-Level Enforcement
    draw_box(ax, 5, 3.9, 3.5, 0.9, 'Layer 3: Cmd Vel Guard\n(Control-Level Enforcement)',
             color=colors['safety'], fontsize=9)

    # Arrows between layers
    draw_arrow(ax, (2.6, 5.5), (3.25, 5.8))  # Goal to Layer 1
    draw_arrow(ax, (5, 6.05), (5, 5.7))      # Layer 1 to Layer 2
    draw_arrow(ax, (5, 4.75), (5, 4.4))      # Layer 2 to Layer 3

    # === Right side: Safety Methods ===
    ax.text(8.5, 6.8, 'Safety Methods', ha='center', fontsize=10, weight='bold')

    methods = [
        ('Geofence\n(Ours)', colors['method']),
        ('CBF', '#E6E6FF'),
        ('SSM', '#E6E6FF'),
        ('Static\nMargin', '#E6E6FF'),
        ('SELP', '#E6E6FF'),
    ]

    for i, (name, color) in enumerate(methods):
        y_pos = 6.3 - i * 0.55
        draw_box(ax, 8.5, y_pos, 1.4, 0.45, name, color=color, fontsize=7)

    # Bracket connecting methods to Layer 1
    ax.plot([7.7, 7.7], [6.5, 3.55], 'k-', linewidth=1)
    ax.plot([7.7, 6.75], [5.2, 5.2], 'k-', linewidth=1)

    # === Bottom: Robot System ===
    draw_box(ax, 5, 2.5, 3.5, 0.8, 'Nav2 Navigation Stack',
             color=colors['robot'], fontsize=9)

    draw_box(ax, 5, 1.5, 3.5, 0.8, 'Mobile Manipulator\n(Velocity Control)',
             color=colors['robot'], fontsize=9)

    # Arrows to robot
    draw_arrow(ax, (5, 3.45), (5, 2.95))
    draw_arrow(ax, (5, 2.1), (5, 1.95))

    # === Left side: Uncertainty Monitors ===
    ax.text(1.5, 4.2, 'Uncertainty\nMonitors', ha='center', fontsize=9, weight='bold')

    monitors = [
        'σ_loc: Localization',
        'e_track: Path Tracking',
        'τ: Latency',
        'v_max: Velocity',
    ]

    for i, mon in enumerate(monitors):
        y_pos = 3.7 - i * 0.4
        draw_box(ax, 1.5, y_pos, 2.0, 0.35, mon, color=colors['monitor'], fontsize=7)

    # Arrow from monitors to safety layers
    ax.annotate('', xy=(3.2, 5.2), xytext=(2.5, 3.7),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.2,
                               connectionstyle='arc3,rad=0.2'))

    # === Feedback loop from robot ===
    ax.annotate('', xy=(1.5, 2.3), xytext=(3.2, 1.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.2,
                               connectionstyle='arc3,rad=-0.3'))
    ax.text(1.8, 1.7, '/odom\n/cmd_vel', fontsize=7, color='gray')

    # === Zone Visualization (bottom right) ===
    ax.text(8.5, 2.5, 'Geofence Zones', ha='center', fontsize=9, weight='bold')

    # Draw example zones
    zone_ax_x, zone_ax_y = 8.5, 1.5
    zone_scale = 0.8

    # Forbidden zone (red)
    forbidden = plt.Polygon([(8.0, 1.8), (8.6, 1.8), (8.6, 2.2), (8.0, 2.2)],
                           facecolor='#FF6B6B', edgecolor='darkred', alpha=0.7, linewidth=1.5)
    ax.add_patch(forbidden)
    ax.text(8.3, 2.0, 'Forbidden', ha='center', va='center', fontsize=6, color='white', weight='bold')

    # Safety margin (orange dashed)
    margin = plt.Rectangle((7.85, 1.65), 0.9, 0.7, fill=False,
                           edgecolor='orange', linestyle='--', linewidth=1.5)
    ax.add_patch(margin)

    # Safe zone (green)
    safe = plt.Polygon([(7.5, 1.0), (9.5, 1.0), (9.5, 1.5), (7.5, 1.5)],
                      facecolor='#90EE90', edgecolor='darkgreen', alpha=0.5, linewidth=1)
    ax.add_patch(safe)
    ax.text(8.5, 1.25, 'Safe Zone', ha='center', va='center', fontsize=6, color='darkgreen')

    # Robot position marker
    robot = Circle((8.7, 1.25), 0.08, facecolor='blue', edgecolor='black')
    ax.add_patch(robot)
    ax.text(8.9, 1.25, 'Robot', fontsize=6, va='center')

    # Legend for margin formula
    ax.text(8.5, 0.7, r'$M = k_\sigma \cdot \sigma_{loc} + e_{track} + v_{max} \cdot \tau$',
            ha='center', fontsize=8, style='italic')

    # === Decision Flow Labels ===
    ax.text(4.3, 5.9, 'ALLOW/\nREJECT/\nPROJECT', fontsize=6, ha='right', color='darkred')

    # Add enforcement labels
    ax.text(3.4, 6.5, '[1]', fontsize=10, ha='center', va='center', weight='bold',
            bbox=dict(boxstyle='circle', facecolor='white', edgecolor='black'))
    ax.text(3.4, 5.2, '[2]', fontsize=10, ha='center', va='center', weight='bold',
            bbox=dict(boxstyle='circle', facecolor='white', edgecolor='black'))
    ax.text(3.4, 3.9, '[3]', fontsize=10, ha='center', va='center', weight='bold',
            bbox=dict(boxstyle='circle', facecolor='white', edgecolor='black'))

    # === Legend ===
    legend_elements = [
        mpatches.Patch(facecolor=colors['input'], edgecolor='black', label='Input'),
        mpatches.Patch(facecolor=colors['safety'], edgecolor='black', label='Safety Layer'),
        mpatches.Patch(facecolor=colors['robot'], edgecolor='black', label='Robot System'),
        mpatches.Patch(facecolor=colors['monitor'], edgecolor='black', label='Monitor'),
        Line2D([0], [0], color='orange', linestyle='--', label='Safety Margin'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=7,
              framealpha=0.9, ncol=5, bbox_to_anchor=(0.0, -0.02))

    return fig

def create_detailed_goal_gate_figure():
    """Create detailed Goal Gate Node figure"""
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 6)
    ax.set_aspect('equal')
    ax.axis('off')

    colors = {
        'input': '#E8F4FD',
        'process': '#FFE6E6',
        'decision': '#FFFACD',
        'output': '#E6FFE6',
        'reject': '#FFB6C1',
    }

    ax.text(4.5, 5.7, 'Goal Gate Node: Policy Enforcement Flow',
            ha='center', fontsize=12, weight='bold')

    # Input
    draw_box(ax, 1.5, 4.5, 2.0, 0.8, 'NavigateToPose\nGoal (x, y, θ)',
             color=colors['input'], fontsize=8)

    # Load config
    draw_box(ax, 1.5, 3.3, 2.0, 0.6, 'Load Geofence\nConfiguration',
             color=colors['process'], fontsize=8)

    # Get margins
    draw_box(ax, 4.5, 4.5, 2.2, 0.8, 'Compute Dynamic\nSafety Margin M',
             color=colors['process'], fontsize=8)

    # Safety check
    draw_box(ax, 4.5, 3.3, 2.2, 0.8, 'Distance Check:\nd(goal, zone) > M?',
             color=colors['decision'], fontsize=8)

    # Decision branches
    draw_box(ax, 7.5, 4.2, 1.5, 0.6, 'ALLOW\n→ Nav2',
             color=colors['output'], fontsize=8)

    draw_box(ax, 7.5, 3.3, 1.5, 0.6, 'PROJECT\n→ Safe Point',
             color='#FFE4B5', fontsize=8)

    draw_box(ax, 7.5, 2.4, 1.5, 0.6, 'REJECT\n→ Abort',
             color=colors['reject'], fontsize=8)

    # Margin formula box
    draw_box(ax, 4.5, 1.8, 4.5, 0.9,
             'Margin Formula:\nM = k_σ·σ_loc + e_track + v_max·τ\n(0.45 + 0.05 + 0.05 = 0.55m)',
             color='white', fontsize=7)

    # Arrows
    draw_arrow(ax, (1.5, 4.1), (1.5, 3.65))
    draw_arrow(ax, (2.5, 4.5), (3.4, 4.5))
    draw_arrow(ax, (2.5, 3.3), (3.4, 3.3))
    draw_arrow(ax, (4.5, 4.1), (4.5, 3.75))
    draw_arrow(ax, (5.6, 3.9), (6.75, 4.2))
    draw_arrow(ax, (5.6, 3.3), (6.75, 3.3))
    draw_arrow(ax, (5.6, 2.9), (6.75, 2.4))

    # Labels
    ax.text(6.2, 4.2, 'd > M', fontsize=7, color='green')
    ax.text(6.2, 3.5, 'd ≈ M', fontsize=7, color='orange')
    ax.text(6.2, 2.5, 'd < M', fontsize=7, color='red')

    return fig

def create_layered_defense_figure():
    """Create layered defense visualization"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.text(5, 4.7, 'Three-Layer Defense-in-Depth Architecture',
            ha='center', fontsize=12, weight='bold')

    # Layer boxes with timing info
    layers = [
        ('Goal-Level', 'Goal Gate Node', 'Pre-navigation check\nDecision: ALLOW/REJECT/PROJECT', '#FFB3B3', 4.0),
        ('Path-Level', 'Path Watchdog', 'Real-time path monitoring\nAction: Cancel navigation', '#FFD9B3', 2.8),
        ('Control-Level', 'Cmd Vel Guard', 'Velocity command filter\nAction: Override velocity', '#B3FFB3', 1.6),
    ]

    for name, node, desc, color, y in layers:
        # Main box
        box = FancyBboxPatch((0.5, y - 0.4), 9, 0.8,
                            boxstyle="round,pad=0.02,rounding_size=0.1",
                            facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(box)

        # Labels
        ax.text(1.2, y, name, fontsize=10, weight='bold', va='center')
        ax.text(3.5, y, node, fontsize=9, va='center', style='italic')
        ax.text(7, y, desc, fontsize=8, va='center', ha='center')

    # Arrows showing data flow
    ax.annotate('', xy=(5, 3.55), xytext=(5, 3.85),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.annotate('', xy=(5, 2.35), xytext=(5, 2.65),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.annotate('', xy=(5, 1.15), xytext=(5, 1.45),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # Input/Output labels
    ax.text(5, 4.4, 'Goal Command', ha='center', fontsize=9)
    ax.text(5, 0.9, 'Safe Velocity → Robot', ha='center', fontsize=9)

    return fig

if __name__ == '__main__':
    # Create all figures
    print("Generating figures...")

    # Main overview
    fig1 = create_overview_figure()
    fig1.savefig('/home/jim/ros2_motion_planning_tutorials/figures/geofence_overview.pdf',
                 format='pdf', bbox_inches='tight')
    fig1.savefig('/home/jim/ros2_motion_planning_tutorials/figures/geofence_overview.png',
                 format='png', bbox_inches='tight', dpi=300)
    print("Saved: geofence_overview.pdf/png")

    # Goal Gate detail
    fig2 = create_detailed_goal_gate_figure()
    fig2.savefig('/home/jim/ros2_motion_planning_tutorials/figures/goal_gate_detail.pdf',
                 format='pdf', bbox_inches='tight')
    fig2.savefig('/home/jim/ros2_motion_planning_tutorials/figures/goal_gate_detail.png',
                 format='png', bbox_inches='tight', dpi=300)
    print("Saved: goal_gate_detail.pdf/png")

    # Layered defense
    fig3 = create_layered_defense_figure()
    fig3.savefig('/home/jim/ros2_motion_planning_tutorials/figures/layered_defense.pdf',
                 format='pdf', bbox_inches='tight')
    fig3.savefig('/home/jim/ros2_motion_planning_tutorials/figures/layered_defense.png',
                 format='png', bbox_inches='tight', dpi=300)
    print("Saved: layered_defense.pdf/png")

    print("\nAll figures generated successfully!")
    # plt.show()  # Commented out for headless generation
