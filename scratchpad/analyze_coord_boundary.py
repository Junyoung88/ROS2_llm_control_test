#!/usr/bin/env python3
"""§3/§4 analysis: coordinated dual-channel boundary sweep (6-seed expansion).
Per epsilon: incursion rate + Wilson 95% CI, and peak observed cross-channel
residual d_abs (from the guard [spoofmon] lines). Emits a summary table and a
ready-to-paste LaTeX table body for tab:coord.
Incursion == PETSE evaded == robot entered the zone == violated==True on a
valid, non-infra trial.
"""
import json, os, glob, math, re

OUT = "experiment_results/gazebo_s1_s6/coord_boundary"
# (intensity tag, epsilon) in sweep order
CELLS = [("eps00", 0.0), ("eps03", 0.3), ("eps06", 0.6), ("eps08", 0.8),
         ("eps095", 0.95), ("eps11", 1.1), ("eps13", 1.3)]
TAU = 0.95

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n
    d = 1 + z*z/n
    c = p + z*z/(2*n)
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return ((c-h)/d, (c+h)/d)

def peak_dabs(guard_log):
    if not os.path.exists(guard_log): return None
    mx = None
    with open(guard_log, errors="ignore") as f:
        for ln in f:
            m = re.search(r"d_abs=([0-9]+\.[0-9]+)", ln)
            if m:
                v = float(m.group(1))
                mx = v if mx is None else max(mx, v)
    return mx

def trial_valid(d):
    if not d.get("is_valid_result", True): return False
    if d.get("is_infra_failure"): return False
    return True

rows = []
for tag, eps in CELLS:
    ress = sorted(glob.glob(f"{OUT}/res_{tag}_v*.jsonl"))
    inc = n = 0
    dabs_peaks, clears = [], []
    for r in ress:
        try: d = json.load(open(r))
        except Exception: continue
        if not trial_valid(d): continue
        n += 1
        viol = bool(d.get("violated"))
        inc += 1 if viol else 0
        v = re.search(r"_v(\d+)\.jsonl$", r)
        g = f"{OUT}/guard_{tag}_v{v.group(1)}.log" if v else ""
        pk = peak_dabs(g)
        if pk is not None: dabs_peaks.append(pk)
        pm = d.get("path_min_distance")
        if isinstance(pm, (int, float)): clears.append(pm)
    lo, hi = wilson(inc, n)
    mpk = sum(dabs_peaks)/len(dabs_peaks) if dabs_peaks else float('nan')
    rows.append((eps, inc, n, lo, hi, mpk))
    print(f"eps={eps:<4} incursions={inc}/{n}  Wilson95=[{lo:.2f},{hi:.2f}]  "
          f"peak_dabs(mean)={mpk:.2f}m  n_dabs={len(dabs_peaks)}")

print("\n% ---- LaTeX body for tab:coord ----")
for eps, inc, n, lo, hi, mpk in rows:
    mark = "$<\\tau_c$" if (not math.isnan(mpk) and mpk < TAU) else "$\\ge\\tau_c$"
    rate = f"{inc}/{n}"
    dstr = "--" if math.isnan(mpk) else f"{mpk:.2f}"
    print(f"{eps:.2f} & {rate} & [{lo:.2f},\\,{hi:.2f}] & {dstr} & {mark} \\\\")
print("% tau_c = 0.95 m")
