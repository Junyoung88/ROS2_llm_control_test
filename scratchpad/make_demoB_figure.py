#!/usr/bin/env python3
"""Demonstration B: memoryless vs CUSUM on stealthy slow-ramp LiDAR spoof.
Signal-separation scatter + 5-seed confusion table.
Data: experiment_results/gazebo_s1_s6/money_2x2/{guard,results}_*.jsonl
"""
import re, os, json, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

D = "/home/jim/ros2_motion_planning_tutorials/experiment_results/gazebo_s1_s6/money_2x2"
OFFSET_THRESH = 0.95   # cusum decision boundary (d_abs)
JUMP_THRESH   = 0.45   # memoryless decision boundary (per-update jump)

def guard_peaks(cell, v):
    """Return (peak d_abs, peak jump over full trace, peak jump during APPROACH
    (samples with d_abs<=1.0, i.e. before the robot is dragged deep), tripped)."""
    f = os.path.join(D, f"guard_{cell}_v{v}.log")
    da = jp = jp_appr = 0.0
    tripped = False
    if os.path.exists(f):
        for l in open(f):
            m = re.search(r'd_abs=(\d+\.\d+) jump=(\d+\.\d+) max_jump=(\d+\.\d+)', l)
            if m:
                d = float(m.group(1)); j = float(m.group(2))
                da = max(da, d); jp = max(jp, float(m.group(3)))
                if d <= 1.0:                       # approach phase (pre-incursion)
                    jp_appr = max(jp_appr, j)
            if 'SPOOF DETECTED' in l:
                tripped = True
                e = re.search(r'd_abs=(\d+\.\d+)', l)
                if e: da = max(da, float(e.group(1)))
    return da, jp, jp_appr, tripped

def result(cell, v):
    f = os.path.join(D, f"results_{cell}_v{v}.jsonl")
    rows = [json.loads(x) for x in open(f) if x.strip()]
    return rows[-1] if rows else {}

cells = ["clean_memoryless", "clean_cusum", "stealthy_memoryless", "stealthy_cusum"]
data = {}
for c in cells:
    data[c] = []
    for v in range(5):
        da, jp, jp_appr, trip = guard_peaks(c, v)
        r = result(c, v)
        data[c].append(dict(v=v, d_abs=da, jump=jp, jump_appr=jp_appr, trip=trip,
                            cls=r.get('classification'), viol=r.get('violated'),
                            dec=r.get('decision'),
                            pathmin=r.get('path_min_distance')))

# ---- confusion counts ----
def tally(c):
    from collections import Counter
    return Counter(d['cls'] for d in data[c])
print("=== 5-seed confusion tally ===")
for c in cells:
    t = tally(c)
    print(f"  {c:22s} TP={t.get('TP',0)} FP={t.get('FP',0)} TN={t.get('TN',0)} FN={t.get('FN',0)}"
          f"  (violations={sum(1 for d in data[c] if d['viol'])})")

# ================= FIGURE =================
fig = plt.figure(figsize=(13, 5.4))
gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.28)

# ---- Panel A: signal-separation scatter ----
axA = fig.add_subplot(gs[0, 0])
styles = {
    'clean_memoryless':   dict(c='#2c7fb8', m='o', lbl='clean (benign)'),
    'clean_cusum':        dict(c='#2c7fb8', m='o', lbl=None),
    'stealthy_memoryless':dict(c='#d95f02', m='^', lbl='spoof (slow ramp)'),
    'stealthy_cusum':     dict(c='#d95f02', m='^', lbl=None),
}
for c in cells:
    st = styles[c]
    xs = [d['jump_appr'] for d in data[c]]     # jump seen DURING APPROACH (pre-incursion)
    ys = [d['d_abs'] for d in data[c]]
    axA.scatter(xs, ys, c=st['c'], marker=st['m'], s=95, alpha=0.85,
                edgecolors='k', linewidths=0.6, label=st['lbl'], zorder=3)
    # annotate the effective-attack points (v3,v4 of memoryless) that shot up
    for d in data[c]:
        if c == 'stealthy_memoryless' and d['viol']:
            axA.annotate(f"v{d['v']}: zone incursion\n(memoryless never trips\nduring approach)",
                         (d['jump_appr'], d['d_abs']),
                         textcoords="offset points", xytext=(10, -2), fontsize=7.8,
                         color='#b30000', fontweight='bold')

# threshold lines
axA.axhline(OFFSET_THRESH, ls='--', c='#1b7837', lw=1.8)
axA.axvline(JUMP_THRESH, ls='--', c='#762a83', lw=1.8)
axA.text(JUMP_THRESH+0.006, 5.4, 'memoryless\nthreshold\n(jump=0.45)',
         color='#762a83', fontsize=8.5, va='top')
axA.text(0.012, OFFSET_THRESH+0.06, 'CUSUM threshold (offset d$_{abs}$=0.95)',
         color='#1b7837', fontsize=8.5)
# shaded "memoryless-blind" region: jump<0.45 (left of purple line)
axA.axvspan(0, JUMP_THRESH, color='#762a83', alpha=0.05, zorder=0)
axA.set_xlabel('max per-update jump  |Δc|  during approach  (memoryless decision variable)', fontsize=10)
axA.set_ylabel('peak accumulated offset  d$_{abs}$  (CUSUM decision variable)', fontsize=10)
axA.set_yscale('symlog', linthresh=1.0)
axA.set_xlim(0, 0.58)
axA.set_ylim(0, 6)
axA.set_yticks([0, 0.35, 0.95, 2, 5])
axA.set_yticklabels(['0', '0.35', '0.95', '2', '5'])
axA.set_title('(a) Detector decision variables:  slow spoof is invisible to jumps,\nvisible to accumulated offset',
              fontsize=10.5, loc='left')
axA.legend(loc='center right', fontsize=9, framealpha=0.95)
axA.grid(alpha=0.25)

# ---- Panel B: 5-seed confusion table (paired-by-seed view) ----
axB = fig.add_subplot(gs[0, 1])
axB.axis('off')
axB.set_title('(b) Paired-by-seed outcome  (5 seeds)', fontsize=10.5, loc='left')

# Build a per-seed matrix: rows seeds, cols [clean-mem, clean-cus, spoof-mem, spoof-cus]
col_cells = ['clean_memoryless','clean_cusum','stealthy_memoryless','stealthy_cusum']
col_hdr = ['clean\nmemoryless','clean\nCUSUM','spoof\nmemoryless','spoof\nCUSUM']
def celltxt(d):
    if d['cls'] == 'TN': return ('TN', '#c7e9c0')
    if d['cls'] == 'FP': return ('FP\n(zone-prox\nblock)', '#fdd0a2')
    if d['cls'] == 'TP': return ('TP\n(blocked)', '#a1d99b')
    if d['cls'] == 'FN':
        if d['viol']: return ('FN\n**INCURSION**', '#fb6a4a')
        return ('FN\n(no lure)', '#fee0d2')
    return ('?', '#eee')

nrows, ncols = 5, 4
x0, y0, cw, ch = 0.14, 0.80, 0.205, 0.145
# header
for j, h in enumerate(col_hdr):
    axB.text(x0 + j*cw + cw/2, y0 + ch*0.75, h, ha='center', va='center',
             fontsize=8.2, fontweight='bold')
for i in range(nrows):
    axB.text(x0 - 0.03, y0 - i*ch + ch*0.28, f"seed {i}", ha='right', va='center', fontsize=8)
    for j, c in enumerate(col_cells):
        d = data[c][i]
        txt, col = celltxt(d)
        axB.add_patch(Rectangle((x0 + j*cw, y0 - i*ch - ch*0.15), cw*0.94, ch*0.86,
                                facecolor=col, edgecolor='k', lw=0.5, alpha=0.9))
        axB.text(x0 + j*cw + cw*0.47, y0 - i*ch + ch*0.28, txt.replace('**',''),
                 ha='center', va='center', fontsize=6.6)

note = ("Effective lure (seeds 3,4): memoryless → zone incursion;\n"
        "CUSUM → fail-stop before boundary.\n"
        "Weak lure (seeds 0-2): no incursion under either detector.\n"
        "1 clean FP = runtime zone-proximity block (offset d$_{abs}$=0.10,\n"
        "not a detector false alarm).")
axB.text(0.02, 0.05, note, fontsize=7.8, va='bottom', family='monospace')

fig.suptitle("Demonstration B — Stealthy slow-ramp LiDAR spoof: memoryless jump detector is blind, "
             "CUSUM offset detector catches it",
             fontsize=11.5, y=0.995, x=0.01, ha='left', fontweight='bold')
out = os.path.join(D, "demoB_detector_comparison.png")
fig.savefig(out, dpi=150, bbox_inches='tight')
print("saved:", out)
