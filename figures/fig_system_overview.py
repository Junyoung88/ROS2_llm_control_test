#!/usr/bin/env python3
"""
System Overview Figure for Paper — 2-panel
(a) System architecture with geofence at execution layer
(b) Spatial view with dynamic safety margin
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from matplotlib.lines import Line2D
import numpy as np

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'text.usetex': False,
})

C = {
    'upper_bg':    '#F5F5F5',
    'upper_box':   '#EEEEEE',
    'upper_edge':  '#9E9E9E',
    'exec_bg':     '#FFF8E1',
    'gf_box':      '#FFE0B2',
    'gf_edge':     '#E65100',
    'state_box':   '#E3F2FD',
    'state_edge':  '#1565C0',
    'robot_box':   '#E8F5E9',
    'robot_edge':  '#2E7D32',
    'allow':       '#388E3C',
    'project':     '#F57C00',
    'block':       '#D32F2F',
    'zone_fill':   '#FFCDD2',
    'zone_edge':   '#C62828',
    'margin_fill': '#FFF3E0',
    'margin_edge': '#E65100',
    'robot_blue':  '#1565C0',
    'arrow':       '#37474F',
    'arrow_fb':    '#1565C0',
    'gray':        '#757575',
}


def rbox(ax, x, y, w, h, label, sub=None, fc='white', ec='#455A64',
         lw=1.2, fs=8, fw='bold', tc='black', zorder=2):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.06",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder
    )
    ax.add_patch(box)
    if sub:
        ax.text(x + w/2, y + h*0.63, label, ha='center', va='center',
                fontsize=fs, fontweight=fw, color=tc, zorder=zorder+1)
        ax.text(x + w/2, y + h*0.30, sub, ha='center', va='center',
                fontsize=max(fs-1.5, 5.5), color=C['gray'], style='italic',
                zorder=zorder+1)
    else:
        ax.text(x + w/2, y + h/2, label, ha='center', va='center',
                fontsize=fs, fontweight=fw, color=tc, zorder=zorder+1)


def arr(ax, x1, y1, x2, y2, color='black', lw=1.3, style='-|>',
        conn='arc3,rad=0', zorder=3, ms=10):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, connectionstyle=conn,
        color=color, linewidth=lw, mutation_scale=ms, zorder=zorder
    )
    ax.add_patch(a)


# ==========================================================
# (a) System Architecture
# ==========================================================
def panel_a(ax):
    ax.set_xlim(-0.3, 4.2)
    ax.set_ylim(-0.5, 11.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('(a) System Architecture', fontsize=9, fontweight='bold', pad=6)

    bw = 3.0
    bh = 0.58
    bx = 0.2
    cx = bx + bw / 2

    # Y positions (top → bottom)
    y_cmd  = 10.2
    y_llm  = 9.3
    y_nav  = 8.4
    y_ctrl = 7.5

    y_gf   = 5.0
    y_dec  = 3.6
    y_st   = 2.0
    y_rob  = 0.3

    # ── Upper Layers bg ──
    ubg = FancyBboxPatch(
        (bx - 0.1, y_ctrl - 0.12), bw + 0.2, y_cmd + bh - y_ctrl + 0.3,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor=C['upper_bg'], edgecolor='#BDBDBD',
        linewidth=0.8, linestyle='--', zorder=0
    )
    ax.add_patch(ubg)
    ax.text(cx, y_cmd + bh + 0.18,
            'Upper layers (output not trusted)',
            ha='center', fontsize=5.5, color='#9E9E9E', style='italic')

    # Upper boxes
    rbox(ax, bx, y_cmd, bw, bh, 'Natural Language Command',
         fc=C['upper_box'], ec=C['upper_edge'], fs=7)
    rbox(ax, bx, y_llm, bw, bh, 'LLM Planner',
         sub='Goal generation',
         fc=C['upper_box'], ec=C['upper_edge'], fs=7)
    rbox(ax, bx, y_nav, bw, bh, 'Nav2 Path Planner',
         sub='Route planning',
         fc=C['upper_box'], ec=C['upper_edge'], fs=7)
    rbox(ax, bx, y_ctrl, bw, bh, 'Nav2 Controller',
         sub='Velocity commands',
         fc=C['upper_box'], ec=C['upper_edge'], fs=7)

    for yt, yb in [(y_cmd, y_llm), (y_llm, y_nav), (y_nav, y_ctrl)]:
        arr(ax, cx, yt, cx, yb + bh, C['arrow'], lw=1)

    # ── Execution Layer bg ──
    exec_bg = FancyBboxPatch(
        (bx - 0.1, y_dec - 0.2), bw + 0.2, y_gf + 1.15 - y_dec + 0.45,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor=C['exec_bg'], edgecolor=C['gf_edge'],
        linewidth=1.5, zorder=0
    )
    ax.add_patch(exec_bg)
    ax.text(cx, y_gf + 1.2,
            'Execution-Layer Enforcement',
            ha='center', fontsize=6.5, fontweight='bold', color=C['gf_edge'])

    # Arrow: Controller → Exec layer
    arr(ax, cx, y_ctrl, cx, y_gf + 0.95, C['arrow'], lw=1.5)
    ax.text(cx + 0.2, (y_ctrl + y_gf + 0.95) / 2 + 0.1,
            '/cmd_vel', fontsize=5.5, color=C['gray'], va='center')

    # ── Geofence Safety Check ──
    rbox(ax, bx, y_gf, bw, 0.9, 'Geofence Safety Check',
         sub='dist($\\hat{p}_t$, $\\mathcal{F}$)  $\\geq$  $M(t)$ ?',
         fc=C['gf_box'], ec=C['gf_edge'], lw=2, fs=8)

    # ── Decision row (3 small boxes below geofence) ──
    n_dec = 3
    dec_gap = 0.12
    dec_w = (bw - dec_gap * (n_dec - 1)) / n_dec
    dec_h = 0.45

    labels = ['ALLOW', 'PROJECT', 'BLOCK']
    colors_fc = ['#C8E6C9', '#FFE0B2', '#FFCDD2']
    colors_ec = [C['allow'], C['project'], C['block']]
    subs_dec = ['pass', 'safe point', 'halt']

    for i in range(n_dec):
        dx = bx + i * (dec_w + dec_gap)
        rbox(ax, dx, y_dec, dec_w, dec_h, labels[i],
             sub=subs_dec[i],
             fc=colors_fc[i], ec=colors_ec[i], fs=6.5, lw=1)

    # Arrows from geofence to each decision
    arr(ax, bx + bw*0.2, y_gf, bx + dec_w/2, y_dec + dec_h,
        C['allow'], lw=0.9, ms=7)
    arr(ax, cx, y_gf, cx, y_dec + dec_h,
        C['project'], lw=0.9, ms=7)
    arr(ax, bx + bw*0.8, y_gf, bx + 2*(dec_w+dec_gap) + dec_w/2, y_dec + dec_h,
        C['block'], lw=0.9, ms=7)

    # ── State Estimation ──
    rbox(ax, bx, y_st, bw, 0.65, 'State Estimation',
         sub='$\\Sigma_t$, $v$, $\\tau$, $a_{max}$',
         fc=C['state_box'], ec=C['state_edge'], fs=7)

    # Arrow: State → Geofence (right side, upward)
    se_right = bx + bw + 0.15
    ax.plot([se_right, se_right], [y_st + 0.32, y_gf + 0.45],
            color=C['arrow_fb'], lw=1, zorder=2)
    arr(ax, se_right, y_gf + 0.45, bx + bw + 0.01, y_gf + 0.45,
        C['arrow_fb'], lw=1, ms=7)
    ax.plot(se_right, y_st + 0.32, 'o', color=C['arrow_fb'], ms=2.5, zorder=3)
    ax.text(se_right + 0.08, (y_st + 0.5 + y_gf) / 2,
            '$M(t)$', fontsize=6.5, color=C['arrow_fb'],
            fontweight='bold', rotation=90, va='center', ha='left')

    # ── Robot ──
    rbox(ax, bx, y_rob, bw, 0.55, 'Robot',
         sub='Mobile Manipulator',
         fc=C['robot_box'], ec=C['robot_edge'], fs=7)

    # Arrow: Decision → Robot (through State)
    arr(ax, cx, y_dec, cx, y_st + 0.65, C['arrow'], lw=1.2)
    arr(ax, cx, y_st, cx, y_rob + 0.55, C['arrow'], lw=1.2)

    # Feedback: Robot → State (left side)
    fb_x = bx - 0.15
    ax.plot([fb_x, fb_x], [y_rob + 0.28, y_st + 0.32],
            color=C['arrow_fb'], lw=0.8, zorder=2)
    arr(ax, fb_x, y_st + 0.32, bx - 0.01, y_st + 0.32,
        C['arrow_fb'], lw=0.8, ms=7)
    ax.plot(fb_x, y_rob + 0.28, 'o', color=C['arrow_fb'], ms=2, zorder=3)
    ax.text(fb_x - 0.08, (y_rob + y_st) / 2 + 0.2,
            'odom\n$\\Sigma_t$', fontsize=5, color=C['arrow_fb'],
            rotation=90, va='center', ha='right')


# ==========================================================
# (b) Spatial View
# ==========================================================
def panel_b(ax):
    ax.set_xlim(-2.5, 10)
    ax.set_ylim(-3.0, 7.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('(b) Dynamic Safety Margin', fontsize=9, fontweight='bold', pad=6)

    # Zone
    zx, zy = 2.5, -0.5
    zw, zh = 3.0, 3.0
    zcx, zcy = zx + zw/2, zy + zh/2
    M = 1.1

    # ── Expanded margin ──
    expanded = FancyBboxPatch(
        (zx - M, zy - M), zw + 2*M, zh + 2*M,
        boxstyle="round,pad=0,rounding_size=0.35",
        facecolor=C['margin_fill'], edgecolor=C['margin_edge'],
        linewidth=2, linestyle=(0, (5, 3)), zorder=1
    )
    ax.add_patch(expanded)

    # ── Forbidden zone ──
    zone = Rectangle(
        (zx, zy), zw, zh,
        facecolor=C['zone_fill'], edgecolor=C['zone_edge'],
        linewidth=2, zorder=2
    )
    ax.add_patch(zone)
    ax.text(zcx, zcy + 0.3, 'Forbidden Zone', ha='center', va='center',
            fontsize=10, fontweight='bold', color=C['zone_edge'], zorder=3)
    ax.text(zcx, zcy - 0.4, '$\\mathcal{F}$', ha='center', va='center',
            fontsize=14, fontweight='bold', color=C['zone_edge'], zorder=3)

    # ── M(t) dimension arrow (right side, vertical) ──
    dim_x = zx + zw + M + 0.4
    y_zone_top = zy + zh
    y_exp_top = zy + zh + M
    ax.annotate('', xy=(dim_x, y_zone_top), xytext=(dim_x, y_exp_top),
                arrowprops=dict(arrowstyle='<->', color=C['gf_edge'],
                               lw=1.8, shrinkA=0, shrinkB=0))
    ax.text(dim_x + 0.25, (y_zone_top + y_exp_top) / 2, '$M(t)$',
            fontsize=11, fontweight='bold', color=C['gf_edge'],
            ha='left', va='center')

    # ── Formula box (top center) ──
    ax.text(4.0, 6.3,
            '$M(t) = M_{est} + M_{track} + M_{lat} + M_{brake}$',
            fontsize=9, ha='center', va='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                     edgecolor=C['gf_edge'], linewidth=1.3),
            zorder=5)

    # Term breakdown (2×2 grid)
    terms_left = [
        ('$M_{est}$',   '$= \\sqrt{\\chi^2_2(\\alpha)\\,\\lambda_{max}(\\Sigma_t)}$',
         'Localization unc.'),
        ('$M_{lat}$',   '$= v_{max} \\cdot \\tau$',
         'System latency'),
    ]
    terms_right = [
        ('$M_{track}$', '$= e_{track}$',
         'Path tracking err.'),
        ('$M_{brake}$', '$= v^2_{max}\\,/\\,(2\\,a_{max})$',
         'Braking distance'),
    ]

    base_y = 5.15
    for i, (sym, eq, desc) in enumerate(terms_left):
        ty = base_y - i * 0.55
        ax.text(-1.0, ty, f'{sym} {eq}', fontsize=6.5, ha='left', va='center')
        ax.text(-1.0, ty - 0.22, desc, fontsize=5.5, ha='left', va='center',
                color=C['gray'], style='italic')

    for i, (sym, eq, desc) in enumerate(terms_right):
        ty = base_y - i * 0.55
        ax.text(4.5, ty, f'{sym} {eq}', fontsize=6.5, ha='left', va='center')
        ax.text(4.5, ty - 0.22, desc, fontsize=5.5, ha='left', va='center',
                color=C['gray'], style='italic')

    # ── Safe robot (far left) ──
    rx, ry = -1.5, 1.0
    r_safe = Circle((rx, ry), 0.32, facecolor=C['robot_blue'],
                    edgecolor='#0D47A1', lw=1.5, zorder=4)
    ax.add_patch(r_safe)
    ax.text(rx, ry, 'R', ha='center', va='center',
            fontsize=9, fontweight='bold', color='white', zorder=5)
    ax.text(rx, ry - 0.65, '$\\hat{p}_t$', ha='center',
            fontsize=9, color=C['robot_blue'])

    # Distance: safe robot ↔ expanded boundary
    exp_left = zx - M
    arr(ax, rx + 0.38, ry, exp_left - 0.08, ry,
        C['allow'], lw=1.5, style='<->', ms=9)
    mid = (rx + exp_left) / 2
    ax.text(mid, ry + 0.4, '$d(\\hat{p}_t, \\mathcal{F})$',
            fontsize=8, ha='center', color=C['allow'], fontweight='bold')
    ax.text(mid, ry - 0.4, 'Safe ($\\geq M$)',
            fontsize=7, ha='center', color=C['allow'], fontweight='bold')

    # ── Unsafe robot (inside margin, NW of zone) ──
    rx2, ry2 = 1.8, 4.0
    r_un = Circle((rx2, ry2), 0.32, facecolor='#E53935',
                  edgecolor='#B71C1C', lw=1.5, zorder=4)
    ax.add_patch(r_un)
    ax.text(rx2, ry2, "R'", ha='center', va='center',
            fontsize=9, fontweight='bold', color='white', zorder=5)

    # Arrow toward zone
    arr(ax, rx2 + 0.25, ry2 - 0.25, zcx - 0.6, zy + zh + M - 0.05,
        C['block'], lw=1.3, style='-|>', conn='arc3,rad=-0.15', ms=9)
    ax.text(rx2 - 1.1, ry2 - 0.3, 'Unsafe ($< M$)', fontsize=7,
            ha='center', color=C['block'], fontweight='bold')
    ax.text(rx2 - 1.1, ry2 - 0.7, 'BLOCK', fontsize=8,
            ha='center', color=C['block'], fontweight='bold')

    # ── Planned path around zone ──
    try:
        from scipy.interpolate import make_interp_spline
        wx = np.array([-1.5, -0.5, 0.8, 2.0, zcx, 6.0, 7.5, 8.5, 9.2])
        wy = np.array([1.0, 1.3, 2.3, 3.5, 4.5, 3.5, 2.3, 1.3, 1.0])
        spl = make_interp_spline(wx, wy, k=3)
        xs = np.linspace(wx[0], wx[-1], 200)
        ys = spl(xs)
        ax.plot(xs, ys, ':', color=C['robot_blue'], lw=1.3, alpha=0.4, zorder=1)
        ax.text(8.2, 2.6, '$\\pi_t$', fontsize=9, color=C['robot_blue'], alpha=0.5)
    except ImportError:
        pass

    # ── Goal ──
    ax.plot(9.2, 1.0, '*', ms=16, color=C['allow'],
            markeredgecolor='#1B5E20', markeredgewidth=1.2, zorder=4)
    ax.text(9.2, 0.15, 'Goal', fontsize=7.5, ha='center', color='#2E7D32')

    # ── Legend ──
    legend_elements = [
        mpatches.Patch(facecolor=C['zone_fill'], edgecolor=C['zone_edge'],
                      lw=1.5, label='Forbidden zone $\\mathcal{F}$'),
        mpatches.Patch(facecolor=C['margin_fill'], edgecolor=C['margin_edge'],
                      lw=1, linestyle='--', label='Expanded by $M(t)$'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=C['robot_blue'],
               ms=7, label='Safe ($d \\geq M$)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#E53935',
               ms=7, label='Unsafe ($d < M$)'),
    ]
    ax.legend(handles=legend_elements, loc='lower center', fontsize=5.5,
              ncol=4, framealpha=0.9, bbox_to_anchor=(0.4, -0.15),
              columnspacing=0.6, handletextpad=0.3)


# ==========================================================
def create_figure():
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(7.16, 4.8),
        gridspec_kw={'width_ratios': [0.7, 1.3]}
    )
    panel_a(ax_a)
    panel_b(ax_b)
    plt.tight_layout(w_pad=0.2)
    return fig


if __name__ == '__main__':
    print("Generating 2-panel system overview figure...")
    fig = create_figure()
    base = '/home/jim/ros2_motion_planning_tutorials/figures'
    fig.savefig(f'{base}/fig_system_overview.pdf',
                format='pdf', bbox_inches='tight', pad_inches=0.05)
    fig.savefig(f'{base}/fig_system_overview.png',
                format='png', bbox_inches='tight', pad_inches=0.05, dpi=300)
    print("Saved: fig_system_overview.pdf/png")
