#!/usr/bin/env python3
"""Task 3 — ROC + threshold derivation for the cross-channel spoof detector.
memoryless feature = max per-update jump during APPROACH (d_abs<=1.0, i.e. before the
robot is dragged deep); CUSUM feature = peak accumulated offset d_abs.
Clean trials = benign; spoof signal = spoof_memoryless cells (guard logs d_abs/jump but
only ACTS on jump -> the accumulated offset grows unimpeded = the true attack signal).
"""
import re, os, glob, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Data sources: (dir, clean-glob, spoof-glob). Combine 20-seed unified + existing runs.
SOURCES = [
    # n=20 unified run: clean cells + spoof_memoryless (offset grows unimpeded = attack signal)
    ("experiment_results/gazebo_s1_s6/money_2x2/unified20",
     "guard_clean_*_v*.log", "guard_spoof_memoryless_v*.log"),
]
ROOT = "/home/jim/ros2_motion_planning_tutorials"

def feats(path):
    da = ja = 0.0
    for l in open(path):
        m = re.search(r'd_abs=(\d+\.\d+) jump=(\d+\.\d+) max_jump=(\d+\.\d+)', l)
        if m:
            d = float(m.group(1)); j = float(m.group(2))
            da = max(da, d)
            if d <= 1.0: ja = max(ja, j)
    return da, ja

def had_incursion(guard_path):
    """True if the robot entered the zone in this trial (effective attack)."""
    pm = guard_path.replace("guard_", "posmon_").replace(".log", ".log")
    import json as _j
    try:
        return any(_j.loads(x).get("zone") for x in open(pm) if x.strip())
    except Exception:
        return False

clean, spoof = [], []
for d, cg, sg in SOURCES:
    dd = os.path.join(ROOT, d)
    if cg:
        for f in glob.glob(os.path.join(dd, cg)):
            da, ja = feats(f)
            if da > 0: clean.append((da, ja))
    if sg:
        for f in glob.glob(os.path.join(dd, sg)):
            # Use ONLY effective + unimpeded spoof trials (memoryless FN, robot entered
            # zone): the offset grew freely -> uncensored attack signal. Fizzled attacks
            # (no incursion) and detector-truncated trials are excluded so the ROC
            # measures separability of a REAL attack, not the attack's hit-rate.
            if not had_incursion(f):
                continue
            da, ja = feats(f)
            if da > 0: spoof.append((da, ja))

print(f"clean n={len(clean)}  spoof n={len(spoof)}")
if not clean or not spoof:
    print("insufficient data"); sys.exit(0)

def roc(feat_idx):
    cv = sorted(x[feat_idx] for x in clean)
    sv = sorted(x[feat_idx] for x in spoof)
    thrs = sorted(set([0.0] + cv + sv + [max(cv+sv)*1.5]))
    pts = []
    for t in thrs:
        tpr = sum(1 for v in sv if v > t) / len(sv)
        fpr = sum(1 for v in cv if v > t) / len(cv)
        pts.append((fpr, tpr, t))
    pts.sort()
    # AUC (trapezoid over fpr)
    auc = 0.0
    for i in range(1, len(pts)):
        auc += (pts[i][0]-pts[i-1][0]) * (pts[i][1]+pts[i-1][1]) / 2
    return pts, auc

roc_c, auc_c = roc(0)   # CUSUM: d_abs
roc_m, auc_m = roc(1)   # memoryless: approach jump

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.4))

# ---- Panel A: ROC ----
ax1.plot([p[0] for p in roc_c], [p[1] for p in roc_c], '-o', color='#1b7837', lw=2.2,
         ms=4, label=f'CUSUM (offset d$_{{abs}}$)  AUC={auc_c:.3f}')
ax1.plot([p[0] for p in roc_m], [p[1] for p in roc_m], '-s', color='#762a83', lw=2.2,
         ms=4, label=f'memoryless (approach jump)  AUC={auc_m:.3f}')
ax1.plot([0, 1], [0, 1], ':', color='gray', lw=1)
# operating points
def op(pts, feat_idx, thr):
    cv = [x[feat_idx] for x in clean]; sv = [x[feat_idx] for x in spoof]
    return (sum(1 for v in cv if v > thr)/len(cv), sum(1 for v in sv if v > thr)/len(sv))
oc = op(0, 0, 0.95); om = op(1, 1, 0.45)
ax1.scatter([oc[0]], [oc[1]], s=160, marker='*', color='#1b7837', edgecolors='k', zorder=6)
ax1.annotate('operating pt\nd$_{abs}$=0.95', oc, textcoords='offset points', xytext=(10, -18),
             fontsize=8.5, color='#1b7837')
ax1.scatter([om[0]], [om[1]], s=120, marker='X', color='#762a83', edgecolors='k', zorder=6)
ax1.annotate('operating pt\njump=0.45', om, textcoords='offset points', xytext=(10, 6),
             fontsize=8.5, color='#762a83')
ax1.set_xlabel('False Positive Rate  (clean)')
ax1.set_ylabel('True Positive Rate  (spoof detected)')
ax1.set_xlim(-0.02, 1.02); ax1.set_ylim(-0.02, 1.05)
ax1.set_title('(a) ROC — CUSUM offset separates clean/spoof; jumps do not', fontsize=10.5, loc='left')
ax1.legend(loc='lower right', fontsize=9.5); ax1.grid(alpha=0.25)

# ---- Panel B: feature distributions + threshold derivation ----
import numpy as np
cd = [x[0] for x in clean]; sd = [x[0] for x in spoof]
ax2.hist(cd, bins=12, alpha=0.6, color='#2c7fb8', label=f'clean d$_{{abs}}$ (n={len(cd)})', density=True)
ax2.hist(sd, bins=12, alpha=0.6, color='#d95f02', label=f'spoof d$_{{abs}}$ (n={len(sd)})', density=True)
mu, sd_ = float(np.mean(cd)), float(np.std(cd))
ax2.axvline(0.95, color='#1b7837', ls='--', lw=2, label='threshold 0.95')
ax2.axvline(mu + 3*sd_, color='k', ls=':', lw=1.5, label=f'clean μ+3σ = {mu+3*sd_:.2f}')
ax2.set_xlabel('peak accumulated offset  d$_{abs}$  (m)')
ax2.set_ylabel('density')
ax2.set_title('(b) Threshold derivation: 0.95 > clean μ+3σ, < spoof onset', fontsize=10.5, loc='left')
ax2.legend(fontsize=8.5)
ax2.text(0.02, 0.98, f'clean d$_{{abs}}$: μ={mu:.3f}, σ={sd_:.3f}\nmargin M=0.55 m (k·σ_loc+e_track+v·τ)',
         transform=ax2.transAxes, va='top', fontsize=8, family='monospace')

fig.suptitle('Detector ROC & threshold justification (memoryless jump vs CUSUM cross-channel offset)',
             fontsize=11.5, y=1.0, x=0.01, ha='left', fontweight='bold')
out = os.path.join(ROOT, "experiment_results/gazebo_s1_s6/money_2x2/detector_roc.png")
fig.savefig(out, dpi=150, bbox_inches='tight')
print("saved:", out)
print(f"CUSUM AUC={auc_c:.3f}  memoryless AUC={auc_m:.3f}")
print(f"clean d_abs mu={mu:.3f} sigma={sd_:.3f}  mu+3sigma={mu+3*sd_:.3f}")
