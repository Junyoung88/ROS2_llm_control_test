#!/usr/bin/env python3
"""Realism (gap ④): map-consistent LiDAR spoof restricted to a limited FoV window.
As the spoofer's angular coverage shrinks (toward physically realistic), the real
beams anchor AMCL → the attack's AMCL displacement (d_abs) collapses below the
detection threshold and the robot reaches its goal. 0 incursion at every FoV.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

S = json.load(open('/tmp/fov_summary.json'))
FOVS = [360, 180, 90, 45]
THRESH = 0.95

fig, ax = plt.subplots(figsize=(10, 6))
xs = list(range(len(FOVS)))

# per-trial d_abs points + mean
for i, fov in enumerate(FOVS):
    d = S[str(fov)]['dabs']
    ax.scatter([i]*len(d), d, s=70, color='#d95f02', alpha=0.7, edgecolors='k',
               linewidths=0.5, zorder=3)
means = [np.mean(S[str(f)]['dabs']) for f in FOVS]
ax.plot(xs, means, '-o', color='#7f2704', lw=2.4, ms=9, zorder=4, label='mean AMCL displacement d$_{abs}$')

# detection threshold
ax.axhline(THRESH, ls='--', color='#1b7837', lw=2)
ax.text(2.5, THRESH+0.05, 'CUSUM detection threshold 0.95', color='#1b7837', fontsize=10, ha='center')
ax.axhspan(THRESH, 1.3, color='#1b7837', alpha=0.06)
ax.text(0.02, 1.18, 'attack strong enough to displace AMCL\n→ PETSE detects & fail-stops',
        color='#1b7837', fontsize=8.5, va='top')
ax.axhspan(0, THRESH, color='#762a83', alpha=0.04)
ax.text(3.0, 0.12, 'AMCL anchored by real beams\n→ attack neutralized, robot reaches goal',
        color='#4a1486', fontsize=8.5, ha='center')

# annotate outcomes per FoV
for i, fov in enumerate(FOVS):
    s = S[str(fov)]; n = s['n']
    ax.annotate(f"reached goal {s['reached']}/{n}\ndetected {s['rej']}/{n}\nincursions {s['inc']}/{n}",
                (i, means[i]), textcoords='offset points', xytext=(0, -52 if fov==360 else 14),
                ha='center', fontsize=8, color='#333')

ax.set_xticks(xs)
ax.set_xticklabels([f'{f}°\n({"full 360°" if f==360 else f"{int(f/360*100)}% of scan"})' for f in FOVS])
ax.set_xlabel('spoofer angular coverage (FoV window overridden)  —  more realistic →', fontsize=10.5)
ax.set_ylabel('peak AMCL displacement  d$_{abs}$  (m)', fontsize=10.5)
ax.set_ylim(0, 1.3)
ax.set_title("Realism (gap ④): a physically-limited-FoV LiDAR spoof is defeated by AMCL's multi-beam fusion\n"
             "the idealized full-360° replacement was the strong case; 0 zone incursion at every FoV",
             fontsize=11)
ax.legend(loc='center right', fontsize=9.5)
ax.grid(alpha=0.2, axis='y')
out = "/home/jim/ros2_motion_planning_tutorials/experiment_results/gazebo_s1_s6/money_2x2/realism_fov_curve.png"
fig.savefig(out, dpi=150, bbox_inches='tight')
print("saved:", out)
