#!/usr/bin/env python3
"""Fab-cell testbed: normal transport vs spoofing-hijack (multi-seed × multi-method).
Realistic small semiconductor fab bay (fab_cell.sdf + fab_cell_map): a green AMR
transport lane between process-tool bays, plus an open east "confidential bay" (red
keep-out) at the lane end. LEFT: normal delivery to the lane goal — every method
reaches, PETSE included (no nuisance over-block of the realistic aisle). RIGHT: a
compromised robot is lured by a map-consistent localization spoof past its lane goal
into the restricted bay (to bring its camera onto the equipment) — the planning-level
methods (no guard / SELP / CBF) all enter the bay; only PETSE's execution-time
cross-channel re-verification fail-stops it at the boundary."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TRAV = json.load(open('/tmp/fab_traverse_tally.json'))
HIJ = json.load(open('/tmp/fab_hijack_tally.json'))
METH = ['no_guard', 'selp_proper', 'cbf_inflated', 'geofence']
MLAB = ['No Guard', 'SELP', 'CBF-Adaptive', 'PETSE']
COL = ['#9e9e9e', '#b07aa1', '#f4a259', '#2a9d8f']
N = 5

reach = [TRAV[m]['reach'] / N for m in METH]
incur = [HIJ[m]['incursion'] / N for m in METH]

fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5.2))
x = np.arange(len(METH))

# LEFT — normal transport: goal-reach rate
bars = axL.bar(x, reach, 0.62, color=COL, edgecolor='k', lw=0.8)
for xi, r, m in zip(x, reach, METH):
    axL.text(xi, r + 0.02, f'{TRAV[m]["reach"]}/{N}', ha='center', fontsize=10, fontweight='bold')
axL.set_xticks(x); axL.set_xticklabels(MLAB, fontsize=10)
axL.set_ylabel('goal-reach rate', fontsize=11)
axL.set_ylim(0, 1.18)
axL.set_title('Normal transport (no attack)\nrobot delivers down the green AMR lane to its goal',
              fontsize=10.5)
axL.axhline(1.0, ls=':', color='#2a9d8f', lw=1, alpha=0.6)
axL.text(3, 1.11, 'PETSE 5/5 — no over-block\nof the realistic fab aisle', ha='center',
         fontsize=8.5, color='#1d6a5f')
axL.grid(alpha=0.2, axis='y')

# RIGHT — spoofing hijack: zone-incursion rate
bars = axR.bar(x, incur, 0.62, color=COL, edgecolor='k', lw=0.8)
for xi, v, m in zip(x, incur, METH):
    lab = f'{HIJ[m]["incursion"]}/{N}'
    axR.text(xi, v + 0.02 if v > 0 else 0.02, lab, ha='center', fontsize=10, fontweight='bold',
             color='#a50026' if v > 0 else '#1d6a5f')
axR.set_xticks(x); axR.set_xticklabels(MLAB, fontsize=10)
axR.set_ylabel('restricted-bay incursion rate', fontsize=11)
axR.set_ylim(0, 1.18)
axR.set_title('Spoofing hijack (compromised robot → camera espionage)\n'
              'localization spoof lures the robot into the confidential bay',
              fontsize=10.5)
axR.annotate('planning-level methods:\nno execution-time re-check\n→ robot enters the bay',
             xy=(1, 1.0), xytext=(0.55, 0.60), fontsize=8.5, color='#8a2f00',
             ha='center', arrowprops=dict(arrowstyle='->', color='#8a2f00', lw=1.2),
             bbox=dict(boxstyle='round', fc='#fff3e6', ec='#f4a259'))
axR.annotate('PETSE 0/5:\ncross-channel runtime\nre-verification fail-stops',
             xy=(3, 0.0), xytext=(3, 0.42), fontsize=8.5, color='#1d6a5f', ha='center',
             arrowprops=dict(arrowstyle='->', color='#1d6a5f', lw=1.2),
             bbox=dict(boxstyle='round', fc='#e6f4f1', ec='#2a9d8f'))
axR.grid(alpha=0.2, axis='y')

fig.suptitle('Small semiconductor fab-cell testbed (5 seeds × 4 methods): '
             'PETSE preserves normal AMR transport yet is the only method that blocks '
             'the spoofing camera-hijack', fontsize=11.5, y=1.02)
fig.tight_layout()
out = "/home/jim/ros2_motion_planning_tutorials/experiment_results/gazebo_s1_s6/fab/fab_cell_result.png"
fig.savefig(out, dpi=150, bbox_inches='tight')
print("saved", out)
for m, ml in zip(METH, MLAB):
    print(f"  {ml:14s}: traverse reach {TRAV[m]['reach']}/{N}, hijack incursion {HIJ[m]['incursion']}/{N}")
