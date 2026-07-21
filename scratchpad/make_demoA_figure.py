#!/usr/bin/env python3
"""Demonstration A (threat reality) — STRONG map-consistent LiDAR spoof.
Undefended (no_guard): the true robot is dragged DEEP into the forbidden zone.
PETSE-CUSUM: fail-stops at the boundary. 3 seeds each, money_2x2/demoA/.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

D = "/home/jim/ros2_motion_planning_tutorials/experiment_results/gazebo_s1_s6/money_2x2/demoA"
ZX = (1.5, 4.5); ZY = (-4.0, -1.0)      # forbidden zone (warehouse, -Y toward racks)
MARGIN = 0.55                            # guard expanded boundary

def traj(tag):
    rows = [json.loads(l) for l in open(os.path.join(D, f"posmon_{tag}.log")) if l.strip()]
    return [r['x'] for r in rows], [r['y'] for r in rows]

def inzone_pts(tag):
    rows = [json.loads(l) for l in open(os.path.join(D, f"posmon_{tag}.log")) if l.strip()]
    return [(r['x'], r['y']) for r in rows if r.get('zone')]

fig, ax = plt.subplots(figsize=(9.5, 7.0))

# forbidden zone + expanded (guard) boundary
ax.add_patch(Rectangle((ZX[0], ZY[0]), ZX[1]-ZX[0], ZY[1]-ZY[0],
                       facecolor='#e34a33', alpha=0.18, edgecolor='#b30000', lw=2, zorder=1))
ax.add_patch(Rectangle((ZX[0]-MARGIN, ZY[0]-MARGIN), ZX[1]-ZX[0]+2*MARGIN,
                       (ZY[1]+MARGIN)-(ZY[0]-MARGIN),
                       facecolor='none', edgecolor='#b30000', lw=1.1, ls=':', zorder=1))
ax.text((ZX[0]+ZX[1])/2, -2.6, 'FORBIDDEN\nZONE', ha='center', va='center',
        color='#7f0000', fontsize=13, fontweight='bold', zorder=2)
ax.text(ZX[1]+MARGIN+0.05, ZY[1]+MARGIN, 'guard expanded\nboundary (+0.55 m)',
        color='#b30000', fontsize=8, va='bottom')

# goal + start
ax.scatter([4.5], [0.0], marker='*', s=300, c='#1a9850', edgecolors='k', zorder=5,
           label='commanded goal (4.5, 0)')
ax.scatter([0], [0], marker='s', s=90, c='k', zorder=5)
ax.text(0.05, 0.14, 'start', fontsize=9)

# undefended (no_guard) — dragged deep into zone
for i, v in enumerate(['noguard_v0', 'noguard_v1', 'noguard_v2']):
    xs, ys = traj(v)
    ax.plot(xs, ys, color='#b30000', lw=2.0, ls='--', alpha=0.85, zorder=4,
            label='undefended (no guard) — lured INTO zone' if i == 0 else None)
    pz = inzone_pts(v)
    if pz:
        ax.scatter([p[0] for p in pz], [p[1] for p in pz], s=30, c='#7f0000',
                   marker='o', edgecolors='none', alpha=0.55, zorder=5,
                   label='true pose INSIDE zone' if i == 0 else None)
        ax.scatter([xs[-1]], [ys[-1]], marker='X', s=120, c='#b30000', edgecolors='k', zorder=6)
        ax.annotate(f"seed {i}: y={min(ys):.2f}\n({len(pz)} in-zone)", (xs[-1], min(ys)),
                    textcoords="offset points", xytext=(8, -2), fontsize=7.8,
                    color='#7f0000', fontweight='bold')

# defended (PETSE-CUSUM) — fail-stop before zone
for i, v in enumerate(['cusum_v0', 'cusum_v1', 'cusum_v2']):
    xs, ys = traj(v)
    ax.plot(xs, ys, color='#1b7837', lw=2.6, alpha=0.95, zorder=4,
            label='PETSE-CUSUM — fail-stop before zone' if i == 0 else None)
    ax.scatter([xs[-1]], [ys[-1]], marker='o', s=95, c='#1b7837', edgecolors='k', zorder=6)

ax.annotate('CUSUM halts here\n(0 in-zone, y≈−0.2 to −0.8)', (2.4, -0.55),
            textcoords="offset points", xytext=(0, 0), fontsize=8.4, color='#1b7837',
            ha='center', fontweight='bold')

ax.axhline(0, color='gray', lw=0.5, alpha=0.5)
ax.set_xlabel('x  (m, world/map frame)')
ax.set_ylabel('y  (m)')
ax.set_xlim(-0.6, 6.0)
ax.set_ylim(-4.6, 1.2)
ax.set_aspect('equal')
ax.legend(loc='lower left', fontsize=8.6, framealpha=0.95)
ax.set_title("Demonstration A — Threat reality (strong map-consistent LiDAR spoof, 3 seeds):\n"
             "undefended robot dragged DEEP into the forbidden zone; PETSE-CUSUM fail-stops it at the boundary",
             fontsize=10.3)
ax.grid(alpha=0.2)
out = os.path.join(D, "..", "demoA_threat_trajectory.png")
out = os.path.normpath(out)
fig.savefig(out, dpi=150, bbox_inches='tight')
print("saved:", out)
for t in ['noguard_v0','noguard_v1','noguard_v2','cusum_v0','cusum_v1','cusum_v2']:
    xs, ys = traj(t); pz = inzone_pts(t)
    print(f"  {t}: min_y={min(ys):.2f}  in_zone={len(pz)}")
