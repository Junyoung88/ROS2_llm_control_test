#!/usr/bin/env python3
"""Coordinated LiDAR+odom attack — PETSE's honest limit (graceful degradation).
The attacker holds the cross-channel residual near a target ε while luring the robot.
Zone incursions (PETSE evaded) fall from 3/3 at perfect coordination (ε=0) to 0 once the
OBSERVED residual crosses the consistency threshold τ_c=0.95m (which happens around
ε~0.8 because imperfect real-time coordination adds ~0.2-0.3m of tracking error).
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

S = json.load(open('/tmp/coord7_summary.json'))
EPS = [0.0, 0.3, 0.6, 0.8, 0.95, 1.1, 1.3]
TAU = 0.95

inc_rate = [S[str(e)]['inc']/S[str(e)]['n'] for e in EPS]
n = [S[str(e)]['n'] for e in EPS]
mean_dabs = [S[str(e)]['mean_dabs'] for e in EPS]

fig, ax = plt.subplots(figsize=(10, 6))
x = np.array(EPS)

# incursion rate bars (PETSE evaded)
bars = ax.bar(x, inc_rate, width=0.09, color='#d73027', edgecolor='k', alpha=0.85,
              label='zone-incursion rate (PETSE evaded)')
for xi, r, ni, s in zip(x, inc_rate, n, EPS):
    ax.text(xi, r+0.03, f'{S[str(s)]["inc"]}/{ni}', ha='center', fontsize=9,
            color='#a50026', fontweight='bold')

ax.set_xlabel('attacker coordination target  ε  =  intended cross-channel residual  (m)', fontsize=10.5)
ax.set_ylabel('zone-incursion rate  (fraction of trials)', color='#a50026', fontsize=10.5)
ax.tick_params(axis='y', labelcolor='#a50026')
ax.set_ylim(0, 1.15)
ax.set_xlim(-0.12, 1.42)

# secondary axis: observed residual d_abs with tau_c
ax2 = ax.twinx()
ax2.plot(x, mean_dabs, '-o', color='#1f78b4', lw=2, ms=7, label='observed residual d$_{abs}$ (peak)')
ax2.axhline(TAU, ls='--', color='#1b7837', lw=1.8)
ax2.text(1.42, TAU+0.03, r'$\tau_c=0.95$', color='#1b7837', fontsize=10, ha='right')
ax2.set_ylabel('observed cross-channel residual  d$_{abs}$  (m)', color='#1f78b4', fontsize=10.5)
ax2.tick_params(axis='y', labelcolor='#1f78b4')
ax2.set_ylim(0, 1.6)

# annotate the two regimes
ax.annotate('perfect coordination\n(ε→0): PETSE defeated', (0.0, 1.0),
            textcoords='offset points', xytext=(6, -6), fontsize=8.5, color='#a50026')
ax.annotate('ε ≳ 0.95: residual crosses τ_c\n→ fully protected (0 incursions)', (0.95, 0.0),
            textcoords='offset points', xytext=(-30, 34), fontsize=8.5, color='#1b7837',
            ha='center')

ax.set_title("Coordinated LiDAR+odom spoofing: PETSE degrades gracefully\n"
             "incursions vanish once the observed residual crosses $\\tau_c$; real-time "
             "coordination is imperfect\n(residual $\\approx \\varepsilon + 0.2$ to $0.3$ m), so "
             "reliable evasion needs a target $\\varepsilon \\lesssim 0.65$", fontsize=10)
lines1, lab1 = ax.get_legend_handles_labels()
lines2, lab2 = ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, lab1+lab2, loc='center right', fontsize=9, framealpha=0.95)
ax.grid(alpha=0.2, axis='y')
out = "/home/jim/ros2_motion_planning_tutorials/experiment_results/gazebo_s1_s6/money_2x2/coord_epsilon_curve.png"
fig.savefig(out, dpi=150, bbox_inches='tight')
print("saved:", out)
for e in EPS: print(f"  ε={e}: incursion {S[str(e)]['inc']}/{S[str(e)]['n']}, mean d_abs {S[str(e)]['mean_dabs']}")
