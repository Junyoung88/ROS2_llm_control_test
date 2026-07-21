#!/usr/bin/env python3
"""
LLM-based Robot Control System with Probabilistic Dynamic Geofence
Publication-quality figure for paper
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Polygon
from matplotlib.lines import Line2D
import numpy as np

# Publication settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'text.usetex': False,
})

# Color scheme (professional, colorblind-friendly)
COLORS = {
    'user': '#E3F2FD',           # Light blue
    'llm': '#E8EAF6',            # Light indigo
    'nav': '#ECEFF1',            # Light gray-blue
    'robot': '#E0F2F1',          # Light teal
    'geofence_bg': '#FFF3E0',    # Light orange
    'geofence_header': '#FFE0B2', # Orange
    'block': '#FFCDD2',          # Light red
    'project': '#FFF9C4',        # Light yellow
    'stop': '#C8E6C9',           # Light green
    'attack': '#FCE4EC',         # Light pink
    'state': '#F3E5F5',          # Light purple
    'arrow_main': '#1565C0',     # Blue
    'arrow_feedback': '#2E7D32', # Green
    'arrow_attack': '#C62828',   # Red
    'arrow_geofence': '#6D4C41', # Brown
}


def draw_rounded_box(ax, x, y, w, h, label, sublabel=None, color='white',
                     fontsize=9, fontweight='bold', edgecolor='#455A64',
                     linewidth=1.5, alpha=1.0, text_color='black'):
    """Draw a rounded rectangle with label"""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor=color, edgecolor=edgecolor,
        linewidth=linewidth, alpha=alpha
    )
    ax.add_patch(box)

    if sublabel:
        ax.text(x + w/2, y + h*0.65, label, ha='center', va='center',
                fontsize=fontsize, fontweight=fontweight, color=text_color)
        ax.text(x + w/2, y + h*0.35, sublabel, ha='center', va='center',
                fontsize=fontsize-1, color='#546E7A', style='italic')
    else:
        ax.text(x + w/2, y + h/2, label, ha='center', va='center',
                fontsize=fontsize, fontweight=fontweight, color=text_color)


def draw_arrow(ax, start, end, color='black', lw=1.5, style='-|>',
               connectionstyle='arc3,rad=0', mutation_scale=12):
    """Draw arrow between points"""
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle=style,
        connectionstyle=connectionstyle,
        color=color, linewidth=lw,
        mutation_scale=mutation_scale
    )
    ax.add_patch(arrow)


def draw_dashed_box(ax, x, y, w, h, color='#90A4AE', linewidth=1.5, linestyle='--'):
    """Draw dashed border box"""
    rect = Rectangle((x, y), w, h, fill=False,
                     edgecolor=color, linewidth=linewidth, linestyle=linestyle)
    ax.add_patch(rect)


def create_figure():
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 9)
    ax.set_aspect('equal')
    ax.axis('off')

    # =========================================
    # LEFT SIDE: Robot Control Stack
    # =========================================

    # Dashed box for ROS2/Nav2 system
    draw_dashed_box(ax, 0.3, 0.8, 4.9, 7.5, color='#78909C')
    ax.text(2.75, 8.1, 'ROS2 Navigation Stack', ha='center', fontsize=9,
            color='#546E7A', style='italic')

    # --- User Input ---
    draw_rounded_box(ax, 0.5, 7.3, 2.2, 0.8, 'Natural Language',
                     'Command (User)', COLORS['user'])
    # User icon
    circle = Circle((0.85, 7.7), 0.15, facecolor='#1976D2', edgecolor='white')
    ax.add_patch(circle)
    ax.plot([0.85, 0.85], [7.35, 7.5], color='#1976D2', lw=2)
    ax.plot([0.65, 1.05], [7.4, 7.4], color='#1976D2', lw=2)

    # --- LLM Planner ---
    draw_rounded_box(ax, 0.5, 6.3, 4.5, 0.8, 'LLM Planner',
                     'Task Planning', COLORS['llm'])

    # Arrow: User -> LLM
    draw_arrow(ax, (2.75, 7.3), (2.75, 7.15), COLORS['arrow_main'], lw=2)

    # --- Goal Gate Node (핵심 Geofence 검증) ---
    draw_rounded_box(ax, 0.5, 5.1, 4.5, 0.85, 'Goal Gate',
                     'Goal-Level Enforcement', '#FFCCBC', edgecolor='#BF360C', linewidth=2)

    # Arrow: LLM -> Goal Gate
    draw_arrow(ax, (2.75, 6.3), (2.75, 6.0), COLORS['arrow_main'], lw=2)

    # --- Plan Generation ---
    draw_rounded_box(ax, 0.5, 3.9, 2.2, 0.85, 'Plan Generation',
                     'Path Planning', COLORS['nav'])

    # Arrow: Goal Gate -> Motion Planning
    draw_arrow(ax, (1.6, 5.1), (1.6, 4.8), COLORS['arrow_main'], lw=2)

    # --- Execution ---
    draw_rounded_box(ax, 0.5, 2.7, 2.2, 0.85, 'Execution',
                     'Velocity Commands', COLORS['nav'])

    # Arrow: Planning -> Execution
    draw_arrow(ax, (1.6, 3.9), (1.6, 3.6), COLORS['arrow_main'], lw=2)

    # --- State Estimation ---
    draw_rounded_box(ax, 0.5, 1.4, 2.2, 0.95, 'State Estimation', None, COLORS['state'])
    ax.text(1.6, 2.05, 'State Estimation', ha='center', fontsize=9, fontweight='bold')
    bullets = ['• Localization (Σ)', '• Velocity (v)', '• Delay (τ)']
    for i, b in enumerate(bullets):
        ax.text(0.7, 1.85 - i*0.22, b, fontsize=7, va='center')

    # --- Robot ---
    draw_rounded_box(ax, 0.5, 0.2, 4.5, 0.9, 'Robot',
                     'Motors / Actuators', COLORS['robot'])

    # Robot icon (simple mobile robot)
    robot_x, robot_y = 4.2, 0.5
    ax.add_patch(Rectangle((robot_x, robot_y), 0.5, 0.3, facecolor='#37474F', edgecolor='black'))
    ax.add_patch(Circle((robot_x+0.1, robot_y), 0.08, facecolor='#212121'))
    ax.add_patch(Circle((robot_x+0.4, robot_y), 0.08, facecolor='#212121'))

    # Arrow: Execution -> Robot
    draw_arrow(ax, (1.6, 2.7), (1.6, 1.15), COLORS['arrow_main'], lw=2)

    # Feedback arrow: Robot -> State Estimation
    ax.annotate('', xy=(1.6, 1.4), xytext=(1.6, 1.15),
                arrowprops=dict(arrowstyle='-|>', color=COLORS['arrow_feedback'],
                               lw=1.5, connectionstyle='arc3,rad=0'))
    ax.text(1.9, 1.27, 'Sensors', fontsize=7, color=COLORS['arrow_feedback'])

    # =========================================
    # CENTER: Safe Velocity Command
    # =========================================

    draw_rounded_box(ax, 3.0, 2.7, 2.0, 0.85, 'Safe Velocity',
                     'Command', '#C8E6C9', edgecolor='#2E7D32', linewidth=2)

    # Arrow: Execution -> Safe Velocity
    draw_arrow(ax, (2.75, 3.1), (3.0, 3.1), COLORS['arrow_main'], lw=1.5)

    # Arrow: Safe Velocity -> Robot
    draw_arrow(ax, (4.0, 2.7), (4.0, 1.15), COLORS['arrow_main'], lw=2)

    # Arrow: State Estimation -> Safe Velocity
    ax.annotate('', xy=(3.0, 2.3), xytext=(2.75, 1.9),
                arrowprops=dict(arrowstyle='-|>', color=COLORS['arrow_feedback'],
                               lw=1.5, connectionstyle='arc3,rad=0.2'))

    # =========================================
    # RIGHT SIDE: Probabilistic Dynamic Geofence
    # =========================================

    # Main geofence box
    geofence_x, geofence_y = 5.8, 2.8
    geofence_w, geofence_h = 5.8, 5.5

    # Background
    ax.add_patch(FancyBboxPatch(
        (geofence_x, geofence_y), geofence_w, geofence_h,
        boxstyle="round,pad=0.02,rounding_size=0.2",
        facecolor=COLORS['geofence_bg'], edgecolor='#8D6E63',
        linewidth=2
    ))

    # Header
    ax.add_patch(FancyBboxPatch(
        (geofence_x+0.1, geofence_y+geofence_h-0.7), geofence_w-0.2, 0.6,
        boxstyle="round,pad=0.01,rounding_size=0.1",
        facecolor=COLORS['geofence_header'], edgecolor='#6D4C41',
        linewidth=1.5
    ))
    ax.text(geofence_x + geofence_w/2, geofence_y + geofence_h - 0.4,
            'Probabilistic Dynamic Geofence', ha='center', fontsize=11,
            fontweight='bold', color='#4E342E')

    # --- Decision Icons Row ---
    icon_y = 7.4
    icon_labels = [
        ('BLOCK', '#E53935', 6.3),
        ('PROJECT', '#FB8C00', 8.0),
        ('STOP', '#43A047', 9.7),
    ]

    # Block icon (X)
    ax.text(6.5, icon_y, 'X', fontsize=14, ha='center', va='center',
            color='white', fontweight='bold',
            bbox=dict(boxstyle='circle', facecolor='#E53935', edgecolor='#B71C1C', pad=0.3))

    # Project icon (box first, then arrow)
    ax.add_patch(FancyBboxPatch((7.95, icon_y-0.25), 0.5, 0.5,
                                boxstyle="round,pad=0.02,rounding_size=0.1",
                                facecolor='#FB8C00', edgecolor='#E65100', linewidth=1.5, zorder=1))
    ax.annotate('', xy=(8.35, icon_y+0.15), xytext=(8.05, icon_y-0.15),
                arrowprops=dict(arrowstyle='->', color='white', lw=3), zorder=2)

    # Stop icon
    ax.text(9.9, icon_y, 'STOP', fontsize=8, ha='center', va='center',
            color='white', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#43A047', edgecolor='#1B5E20', pad=0.2))

    # Arrows between icons
    draw_arrow(ax, (6.9, icon_y), (7.7, icon_y), '#5D4037', lw=1.2, style='->')
    draw_arrow(ax, (8.6, icon_y), (9.4, icon_y), '#5D4037', lw=1.2, style='->')

    # --- Uncertainty Components ---
    comp_x = 6.0
    comp_y = 5.8
    comp_w = 5.4
    comp_h = 1.3

    ax.add_patch(Rectangle((comp_x, comp_y), comp_w, comp_h,
                           facecolor='white', edgecolor='#8D6E63', linewidth=1))

    ax.text(comp_x + 0.2, comp_y + comp_h - 0.25, '• Localization Uncertainty (Σ)',
            fontsize=9, va='center')
    ax.text(comp_x + 0.2, comp_y + comp_h - 0.55, '• Tracking Error (e)',
            fontsize=9, va='center')
    ax.text(comp_x + 0.2, comp_y + comp_h - 0.85, '• Latency & Velocity (τ, v)',
            fontsize=9, va='center')

    # Margin formula
    ax.text(comp_x + comp_w/2, comp_y + 0.15,
            r'$M = k_\sigma \cdot \sigma_{loc} + e_{track} + v \cdot \tau$',
            ha='center', fontsize=9, style='italic', color='#5D4037')

    # --- Three Decision Boxes ---
    box_y = 3.1
    box_h = 2.4
    box_w = 1.7
    gap = 0.15

    # BLOCK box
    bx = 6.0
    ax.add_patch(FancyBboxPatch(
        (bx, box_y), box_w, box_h,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor=COLORS['block'], edgecolor='#C62828', linewidth=1.5
    ))
    ax.text(bx + box_w/2, box_y + box_h - 0.3, 'BLOCK', ha='center',
            fontsize=10, fontweight='bold', color='#B71C1C')
    ax.text(bx + 0.1, box_y + box_h - 0.7, '• Reject unsafe', fontsize=7)
    ax.text(bx + 0.1, box_y + box_h - 0.95, '  goal target', fontsize=7)
    ax.text(bx + 0.1, box_y + box_h - 1.25, '• Return error', fontsize=7)
    ax.text(bx + 0.1, box_y + box_h - 1.5, '  to LLM', fontsize=7)
    ax.text(bx + box_w/2, box_y + 0.3, 'd < M', ha='center',
            fontsize=9, color='#C62828', fontweight='bold')

    # PROJECT box
    bx = 6.0 + box_w + gap
    ax.add_patch(FancyBboxPatch(
        (bx, box_y), box_w, box_h,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor=COLORS['project'], edgecolor='#F57C00', linewidth=1.5
    ))
    ax.text(bx + box_w/2, box_y + box_h - 0.3, 'PROJECT', ha='center',
            fontsize=10, fontweight='bold', color='#E65100')
    ax.text(bx + 0.1, box_y + box_h - 0.7, '• Correct path', fontsize=7)
    ax.text(bx + 0.1, box_y + box_h - 0.95, '  to safe point', fontsize=7)
    ax.text(bx + 0.1, box_y + box_h - 1.25, '• Nearest valid', fontsize=7)
    ax.text(bx + 0.1, box_y + box_h - 1.5, '  boundary', fontsize=7)
    ax.text(bx + box_w/2, box_y + 0.3, 'd ≈ M', ha='center',
            fontsize=9, color='#E65100', fontweight='bold')

    # STOP box
    bx = 6.0 + 2*(box_w + gap)
    ax.add_patch(FancyBboxPatch(
        (bx, box_y), box_w, box_h,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor=COLORS['stop'], edgecolor='#2E7D32', linewidth=1.5
    ))
    ax.text(bx + box_w/2, box_y + box_h - 0.3, 'STOP', ha='center',
            fontsize=10, fontweight='bold', color='#1B5E20')
    ax.text(bx + 0.1, box_y + box_h - 0.7, '• Set velocity', fontsize=7)
    ax.text(bx + 0.1, box_y + box_h - 0.95, '  to zero', fontsize=7)
    ax.text(bx + 0.1, box_y + box_h - 1.25, '• Emergency', fontsize=7)
    ax.text(bx + 0.1, box_y + box_h - 1.5, '  halt', fontsize=7)
    ax.text(bx + box_w/2, box_y + 0.3, 'Runtime', ha='center',
            fontsize=9, color='#1B5E20', fontweight='bold')

    # =========================================
    # Arrows connecting Geofence to main system
    # =========================================

    # Goal Gate -> Geofence (goal checking) - 핵심 연결
    draw_arrow(ax, (5.0, 5.5), (5.8, 5.8), COLORS['arrow_main'], lw=2)
    ax.text(5.2, 5.85, 'Goal', fontsize=7, ha='center', color=COLORS['arrow_main'])

    # Geofence -> Goal Gate (BLOCK/PROJECT feedback)
    ax.annotate('', xy=(5.0, 5.3), xytext=(5.8, 5.2),
                arrowprops=dict(arrowstyle='-|>', color=COLORS['arrow_attack'],
                               lw=1.5, connectionstyle='arc3,rad=0.2'))
    ax.text(5.5, 5.1, 'BLOCK/\nPROJECT', fontsize=6, color=COLORS['arrow_attack'], ha='center')

    # Execution -> Geofence (velocity monitoring for STOP)
    draw_arrow(ax, (2.75, 3.3), (5.8, 4.5), COLORS['arrow_main'], lw=1.5,
               connectionstyle='arc3,rad=0.15')
    ax.text(3.8, 4.2, 'Cmd Vel', fontsize=7, color=COLORS['arrow_main'])

    # Geofence -> Safe Velocity (output)
    ax.annotate('', xy=(4.5, 3.3), xytext=(5.8, 3.8),
                arrowprops=dict(arrowstyle='-|>', color=COLORS['arrow_geofence'],
                               lw=2, connectionstyle='arc3,rad=0.2'))
    ax.text(4.8, 3.75, 'Safe\nCmd', fontsize=7, ha='center', color=COLORS['arrow_geofence'])

    # State Estimation -> Geofence (uncertainty info)
    ax.annotate('', xy=(5.8, 3.5), xytext=(2.75, 1.9),
                arrowprops=dict(arrowstyle='-|>', color=COLORS['arrow_feedback'],
                               lw=1.5, connectionstyle='arc3,rad=-0.2'))
    ax.text(3.8, 2.4, 'Σ, v, τ', fontsize=8, color=COLORS['arrow_feedback'])

    # =========================================
    # BOTTOM RIGHT: Cyber-Physical Attacks
    # =========================================

    attack_x, attack_y = 8.5, 0.3
    attack_w, attack_h = 3.0, 2.2

    ax.add_patch(FancyBboxPatch(
        (attack_x, attack_y), attack_w, attack_h,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor=COLORS['attack'], edgecolor='#C62828',
        linewidth=1.5, linestyle='--'
    ))

    ax.text(attack_x + attack_w/2, attack_y + attack_h - 0.3,
            'Cyber-Physical Attacks', ha='center', fontsize=9,
            fontweight='bold', color='#B71C1C')

    attacks = ['• Prompt Injection', '• Sensor Spoofing', '• Latency Injection', '• Direct Control']
    for i, atk in enumerate(attacks):
        ax.text(attack_x + 0.2, attack_y + attack_h - 0.65 - i*0.35, atk,
                fontsize=8, color='#C62828')

    # Attack arrow to Robot
    ax.annotate('', xy=(5.0, 0.65), xytext=(8.5, 1.0),
                arrowprops=dict(arrowstyle='-|>', color=COLORS['arrow_attack'],
                               lw=1.5, linestyle='--', connectionstyle='arc3,rad=0.15'))
    ax.text(6.5, 1.1, 'Attacks', fontsize=8, color=COLORS['arrow_attack'])

    # Attack arrow to Geofence (showing defense)
    ax.annotate('', xy=(9.0, 2.8), xytext=(9.5, 2.5),
                arrowprops=dict(arrowstyle='-|>', color=COLORS['arrow_attack'],
                               lw=1.5, linestyle='--'))

    # =========================================
    # Legend / Caption area
    # =========================================

    ax.text(0.3, -0.1,
            'Figure X. Overview of LLM-based robot control system with probabilistic dynamic geofence. '
            'The geofence integrates with the standard pipeline (Plan Generation → Execution → Robot) '
            'to enforce spatial safety through BLOCK/PROJECT/STOP decisions.',
            fontsize=8, style='italic', wrap=True)

    return fig


if __name__ == '__main__':
    print("Generating system overview figure...")
    fig = create_figure()

    # Save in multiple formats
    fig.savefig('/home/jim/ros2_motion_planning_tutorials/figures/system_overview_paper.pdf',
                format='pdf', bbox_inches='tight', pad_inches=0.1)
    fig.savefig('/home/jim/ros2_motion_planning_tutorials/figures/system_overview_paper.png',
                format='png', bbox_inches='tight', pad_inches=0.1, dpi=300)

    print("Saved: system_overview_paper.pdf/png")
    print("Done!")
