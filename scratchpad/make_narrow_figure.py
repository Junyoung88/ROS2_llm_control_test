#!/usr/bin/env python3
"""Narrow (two-sided) corridor sweep — multi-seed × multi-method.
Two symmetric virtual keep-out zones at ±h bound the y=0 travel corridor; SAFE width
= 2(h − M) with the uncertainty margin M≈0.55 m. Robot drives y=0 to the goal (6,0),
which lies OUTSIDE the zones. Shows PETSE (a) does not nuisance-block a clearly-safe
corridor and (b) refuses the sub-margin squeeze that the point-margin baselines drive
straight through — the discriminator is PATH/EXECUTION awareness, not margin size."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

T = json.load(open('/tmp/narrow_tally.json'))   # {'geo|meth': [n, reach, viol]}
GEO = ['nc2_xwide', 'nc2_wide', 'nc2_tight']
LABELS = ['x-wide\n~1.2 m safe', 'wide\n~0.6 m (borderline)', 'tight\nmargins overlap\n(no safe path)']
METH = ['no_guard', 'cbf_inflated', 'geofence']
MLAB = ['No Guard', 'CBF-Adaptive\n(margin 0.55, point)', 'PETSE\n(margin 0.55 + path re-verify)']
COL = ['#9e9e9e', '#f4a259', '#2a9d8f']

reach = {m: [T[f'{g}|{m}'][1] / T[f'{g}|{m}'][0] for g in GEO] for m in METH}

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(GEO)); w = 0.26
for i, m in enumerate(METH):
    b = ax.bar(x + (i - 1) * w, reach[m], w, color=COL[i], edgecolor='k', lw=0.8, label=MLAB[i])
    for xi, r, g in zip(x + (i - 1) * w, reach[m], GEO):
        n, rc, vv = T[f'{g}|{m}']
        ax.text(xi, r + 0.02, f'{rc}/{n}', ha='center', fontsize=9, fontweight='bold')

ax.set_xticks(x); ax.set_xticklabels(LABELS, fontsize=10)
ax.set_ylabel('goal-reach rate  (fraction of 5 seeds)', fontsize=11)
ax.set_ylim(0, 1.15)
ax.set_title("Two-sided narrow-corridor sweep (multi-seed × multi-method)\n"
             "PETSE traverses the safe corridor (no over-block) yet is the ONLY method that "
             "refuses the\nimpassable one — CBF-Adaptive shares the 0.55 m margin but, being "
             "point-goal, drives through", fontsize=10.5)
ax.legend(loc='lower left', fontsize=9, framealpha=0.95)
ax.grid(alpha=0.2, axis='y')

# annotate the key discriminator at the tight corridor
ax.annotate('same margin, opposite outcome:\nPETSE 0/5 (stops) vs CBF 5/5 (drives through)',
            xy=(2 + w, 0.02), xytext=(1.15, 0.55), fontsize=9, color='#8a2f00',
            arrowprops=dict(arrowstyle='->', color='#8a2f00', lw=1.3),
            bbox=dict(boxstyle='round', fc='#fff3e6', ec='#f4a259'))
ax.annotate('clearly-safe corridor:\nPETSE 5/5 (no nuisance block)',
            xy=(0 + w, reach['geofence'][0]), xytext=(-0.35, 0.30), fontsize=9, color='#1d6a5f',
            arrowprops=dict(arrowstyle='->', color='#1d6a5f', lw=1.3),
            bbox=dict(boxstyle='round', fc='#e6f4f1', ec='#2a9d8f'))

out = "/home/jim/ros2_motion_planning_tutorials/experiment_results/gazebo_s1_s6/narrow/narrow_corridor.png"
fig.savefig(out, dpi=150, bbox_inches='tight')
print("saved", out)
for g in GEO:
    print(f"  {g}: " + ", ".join(f"{m}={T[f'{g}|{m}'][1]}/{T[f'{g}|{m}'][0]}(viol {T[f'{g}|{m}'][2]})" for m in METH))
