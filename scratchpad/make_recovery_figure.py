#!/usr/bin/env python3
"""Recovery/safe-hold timeline: transient strong spoof → CUSUM fail-stop → robot
halts before the zone and HOLDS (0 incursion) for the whole episode. Shows the
availability cost is a fail-SAFE hold, not a breach.
"""
import json, re, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "/home/jim/ros2_motion_planning_tutorials/experiment_results/gazebo_s1_s6/money_2x2/recovery"
TAG = "latch_v2"          # representative: lured to y=-0.60, halted 0.40 m short of zone
ZONE_Y = -1.0
OFFSET_THRESH = 0.95

# --- true-y over time (posmon) ---
pm = [json.loads(l) for l in open(f"{D}/posmon_{TAG}.log") if l.strip()]
t0 = pm[0]['t']
pt = [(r['t']-t0) for r in pm]
py = [r['y'] for r in pm]
px = [r['x'] for r in pm]

# --- d_abs over time (guard spoofmon + trip) ---
gt, gd = [], []
trip_t = trip_d = None
for l in open(f"{D}/guard_{TAG}.log"):
    m = re.search(r'\[(\d+\.\d+)\].*d_abs=(\d+\.\d+) jump', l)
    if m and 'spoofmon' in l:
        gt.append(float(m.group(1))-t0); gd.append(float(m.group(2)))
    e = re.search(r'\[(\d+\.\d+)\].*SPOOF DETECTED.*d_abs=(\d+\.\d+)', l)
    if e:
        trip_t = float(e.group(1))-t0; trip_d = float(e.group(2))
if trip_t is not None:
    gt.append(trip_t); gd.append(trip_d)

# spoof window (fires at delay 4s after goal; goal sent ~ episode start). Use guard
# first-sample as proxy for spoof onset region; window = [4, 44] relative to goal send.
# Align to posmon t0 (goal send ≈ episode start).
SPOOF_FIRE, SPOOF_DUR = 4.0, 40.0

fig, ax1 = plt.subplots(figsize=(11, 5.6))

# shaded spoof-active window
ax1.axvspan(SPOOF_FIRE, SPOOF_FIRE+SPOOF_DUR, color='#d95f02', alpha=0.10, zorder=0)
ax1.text(SPOOF_FIRE+SPOOF_DUR/2, 3.6, 'spoof ACTIVE\n(4 s → 44 s)', ha='center',
         color='#a63603', fontsize=9, fontweight='bold')
ax1.text(SPOOF_FIRE+SPOOF_DUR+2, 3.6, 'threat cleared\n(honest scans)', ha='left',
         color='#1b7837', fontsize=9)

# d_abs (left axis)
ax1.plot(gt, gd, '-o', color='#7570b3', lw=2, ms=5, label='CUSUM offset d$_{abs}$(t)', zorder=3)
ax1.axhline(OFFSET_THRESH, ls='--', color='#1b7837', lw=1.6)
ax1.text(1, OFFSET_THRESH+0.08, 'CUSUM threshold 0.95', color='#1b7837', fontsize=8.5)
if trip_t is not None:
    ax1.scatter([trip_t], [trip_d], s=220, marker='*', color='#d62728', edgecolors='k', zorder=6)
    ax1.annotate('FAIL-STOP\n(robot halts)', (trip_t, trip_d), textcoords='offset points',
                 xytext=(10, -6), fontsize=9, color='#d62728', fontweight='bold')
ax1.annotate('AMCL frozen after halt\n(robot stationary → no pose updates)',
             (trip_t+3, trip_d-0.35), fontsize=7.8, color='#7570b3')
ax1.set_xlabel('time since goal sent (s)')
ax1.set_ylabel('CUSUM offset  d$_{abs}$  (m)', color='#7570b3')
ax1.tick_params(axis='y', labelcolor='#7570b3')
ax1.set_ylim(-1.4, 4.2)

# true-y (right axis)
ax2 = ax1.twinx()
ax2.plot(pt, py, '-', color='#2c7fb8', lw=2.2, label='true robot y(t)', zorder=2)
ax2.axhline(ZONE_Y, ls='-', color='#b30000', lw=1.8)
ax2.axhspan(-4.5, ZONE_Y, color='#e34a33', alpha=0.10)
ax2.text(pt[-1]*0.5, ZONE_Y-0.25, 'FORBIDDEN ZONE  (y < −1.0)', color='#7f0000',
         fontsize=9, fontweight='bold', ha='center')
held_y = py[-1]
ax2.annotate(f'held at y={held_y:.2f}\n({abs(ZONE_Y-held_y):.2f} m short of zone,\n0 incursion, whole episode)',
             (pt[-1], held_y), textcoords='offset points', xytext=(-150, 18),
             fontsize=8.5, color='#2c7fb8', fontweight='bold')
ax2.set_ylabel('true robot y  (m)', color='#2c7fb8')
ax2.tick_params(axis='y', labelcolor='#2c7fb8')
ax2.set_ylim(-1.5, 4.0)

ax1.set_title("Recovery / safe-hold: transient LiDAR spoof → CUSUM fail-stop → robot holds "
              "BEFORE the zone\n(integrity preserved: 0 incursion even as the attack ramps and then clears)",
              fontsize=10.5)
lines1, lab1 = ax1.get_legend_handles_labels()
lines2, lab2 = ax2.get_legend_handles_labels()
ax1.legend(lines1+lines2, lab1+lab2, loc='center left', fontsize=9, framealpha=0.95)
ax1.grid(alpha=0.2)
out = os.path.join(D, "..", "recovery_safehold_timeline.png")
out = os.path.normpath(out)
fig.savefig(out, dpi=150, bbox_inches='tight')
print("saved:", out)
print(f"{TAG}: trip d_abs={trip_d} at t={trip_t:.1f}s, held y={held_y:.2f}, episode {pt[-1]:.0f}s")
