#!/usr/bin/env python3
"""Final 20-seed unified 2x2 confusion figure: {clean, strong map-consistent spoof}
x {memoryless, CUSUM}. Data money_2x2/unified20/."""
import json, glob, os, math
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

D = "/home/jim/ros2_motion_planning_tutorials/experiment_results/gazebo_s1_s6/money_2x2/unified20"

def wilson(k, n, z=1.96):
    if n == 0: return (0, 0)
    p = k/n; d = 1+z*z/n
    c = (p+z*z/(2*n))/d; h = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return (max(0, c-h)*100, min(1, c+h)*100)

def inzone(f):
    pf = f.replace('results_', 'posmon_').replace('.jsonl', '.log')
    try: return sum(1 for x in open(pf) if x.strip() and json.loads(x).get('zone'))
    except: return 0

cells = {}
for c in ['clean_cusum', 'clean_memoryless', 'spoof_cusum', 'spoof_memoryless']:
    rows = []
    for f in sorted(glob.glob(f'{D}/results_{c}_v*.jsonl')):
        if os.path.getsize(f) == 0: continue
        try: rows.append((f, json.loads(open(f).readlines()[-1])))
        except: pass
    cells[c] = rows

def summ(c):
    rows = cells[c]; t = Counter(r.get('classification') for _, r in rows)
    dang = ben = 0
    for f, r in rows:
        if r.get('classification') == 'FN':
            if r.get('violated') or inzone(f) > 0: dang += 1
            else: ben += 1
    inc = sum(1 for f, r in rows if inzone(f) > 0)
    return dict(n=len(rows), TP=t.get('TP', 0), TN=t.get('TN', 0), FP=t.get('FP', 0),
                FN=t.get('FN', 0), dang=dang, ben=ben, inc=inc)

S = {c: summ(c) for c in cells}

fig, ax = plt.subplots(figsize=(11, 6.4))
ax.axis('off')
ax.set_title("Unified 2×2 — strong map-consistent LiDAR spoof vs benign, 20 seeds each\n"
             "memoryless jump detector vs CUSUM cross-channel offset detector",
             fontsize=12.5, fontweight='bold')

# grid: rows = attack {clean, spoof}, cols = detector {memoryless, cusum}
cols = [('memoryless', 'clean_memoryless', 'spoof_memoryless'),
        ('CUSUM (PETSE)', 'clean_cusum', 'spoof_cusum')]
x0, y0, cw, ch = 0.30, 0.66, 0.34, 0.30

ax.text(x0+cw/2, 0.95, 'DETECTOR', ha='center', fontsize=11, fontweight='bold')
for j, (name, _, _) in enumerate(cols):
    ax.text(x0+j*cw, 0.89, name, ha='center', fontsize=11, fontweight='bold')
rows_lbl = ['BENIGN\n(clean nav)', 'ATTACK\n(strong spoof)']
for i in range(2):
    ax.text(x0-cw*0.62, y0-i*ch, rows_lbl[i], ha='center', va='center', fontsize=10.5, fontweight='bold')

for j, (name, cclean, cspoof) in enumerate(cols):
    for i, cell in enumerate([cclean, cspoof]):
        s = S[cell]
        if i == 0:  # benign row
            lo, hi = wilson(s['FP'], s['n'])
            head = f"{s['TN']}/{s['n']} TN"
            sub = f"{s['FP']} FP = {100*s['FP']/s['n']:.0f}%\n[{lo:.0f},{hi:.0f}]% CI\n(zone-prox block,\nnot detector)"
            col = '#c7e9c0' if s['FP'] <= 1 else '#fdd0a2'
        else:  # attack row
            eff = s['TP'] + s['dang']
            lo, hi = wilson(s['TP'], eff) if eff else (0, 0)
            head = f"{s['TP']}/{eff} detected = {100*s['TP']/eff if eff else 0:.0f}%"
            sub = f"[{lo:.0f},{hi:.0f}]% CI\nzone incursions: {s['inc']}/{s['n']}\n({s['ben']} attacks fizzled)"
            col = '#a1d99b' if s['inc'] == 0 else '#fb6a4a'
        ax.add_patch(Rectangle((x0+j*cw-cw*0.46, y0-i*ch-ch*0.42), cw*0.92, ch*0.84,
                               facecolor=col, edgecolor='k', lw=1.1, alpha=0.92))
        ax.text(x0+j*cw, y0-i*ch+ch*0.22, head, ha='center', va='center', fontsize=10.5, fontweight='bold')
        ax.text(x0+j*cw, y0-i*ch-ch*0.16, sub, ha='center', va='center', fontsize=8.3)

note = ("Attack efficacy: 18/20 spoof runs were effective (2 fizzled).  "
        "Detection rate = detected / effective attacks (Wilson 95% CI).\n"
        "CUSUM: 16/16 effective attacks caught, robot NEVER entered the zone (0/20).  "
        "memoryless: 4/18 caught, robot entered the zone in 14/20.\n"
        "Clean false-block 1/20 in BOTH detectors (identical) → it is a runtime zone-proximity "
        "block from clean drift, not a detector false alarm.")
ax.text(0.5, 0.06, note, ha='center', va='bottom', fontsize=8.6, family='monospace',
        bbox=dict(boxstyle='round', fc='#f7f7f7', ec='#999'))

out = os.path.join(D, "..", "final_unified_2x2.png")
out = os.path.normpath(out)
fig.savefig(out, dpi=150, bbox_inches='tight')
print("saved:", out)
for c, s in S.items(): print(c, s)
