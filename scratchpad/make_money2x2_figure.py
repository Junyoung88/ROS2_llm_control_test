#!/usr/bin/env python3
"""Money 2x2: confusion table + cross-channel d_abs / jump trajectory figure.

Cells: {clean, stealthy} x {memoryless, cusum}, 2 seeds each.
Story: the map-consistent LiDAR spoof drags AMCL toward the +Y racks -> TRUE robot
lured into the -Y forbidden zone. The per-update jump stays tiny (<0.45) so the
MEMORYLESS detector misses it -> incursion. The cross-channel OFFSET d_abs grows
monotonically past 0.6 so CUSUM catches it -> fail-stop before the zone. Clean stays
d_abs<0.35 / jump<0.06 so neither detector false-alarms.
"""
import json, glob, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "experiment_results/gazebo_s1_s6/money_2x2"
OFFSET_THRESH = 0.6
JUMP_THRESH = 0.45
CELLS = ["clean_memoryless", "clean_cusum", "stealthy_memoryless", "stealthy_cusum"]

def load_cell(tag):
    rows = []
    for f in sorted(glob.glob(f"{OUT}/results_{tag}_v*.jsonl")):
        try:
            for line in open(f):
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        except Exception:
            pass
    return rows

def confusion():
    print(f"\n{'cell':22} {'n':>2} {'violated':>9} {'guard_stop':>11} {'reached':>8}  class")
    table = {}
    for tag in CELLS:
        rows = load_cell(tag)
        viol = sum(1 for r in rows if r.get("violated"))
        stop = sum(1 for r in rows if r.get("decision") in ("runtime_reject", "reject")
                   or "guard blocked" in (r.get("reason","").lower()))
        goal = sum(1 for r in rows if r.get("task_completed") and not r.get("violated"))
        cls = {}
        for r in rows:
            c = r.get("classification","?"); cls[c] = cls.get(c,0)+1
        clsstr = " ".join(f"{k}:{v}" for k,v in sorted(cls.items()))
        table[tag] = dict(n=len(rows), viol=viol, stop=stop, goal=goal, cls=clsstr)
        print(f"{tag:22} {len(rows):>2} {viol:>9} {stop:>11} {goal:>8}  {clsstr}")
    return table

def dabs_series(guardlog):
    ts, d, jump = [], [], []
    if not os.path.exists(guardlog):
        return None
    for line in open(guardlog):
        m = re.search(r"\[(\d+\.\d+)\].*\[spoofmon\] d_abs=([0-9.]+) jump=([0-9.]+)", line)
        if m:
            ts.append(float(m.group(1))); d.append(float(m.group(2))); jump.append(float(m.group(3)))
    if not ts:
        return None
    ts = np.array(ts) - ts[0]
    return ts, np.array(d), np.array(jump)

def plot():
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    def draw(a, idx, thr, thr_lbl, title, ylab):
        clean_lbl = spoof_lbl = True
        for tag in CELLS:
            color = "tab:green" if tag.startswith("clean") else "tab:red"
            for gl in sorted(glob.glob(f"{OUT}/guard_{tag}_v*.log")):
                s = dabs_series(gl)
                if not s: continue
                lbl = None
                if color == "tab:green" and clean_lbl: lbl = "clean"; clean_lbl = False
                if color == "tab:red" and spoof_lbl: lbl = "spoof"; spoof_lbl = False
                a.plot(s[0], s[idx], color=color, alpha=0.8, lw=1.6, label=lbl)
        a.axhline(thr, ls="--", color="k", lw=1.3, label=thr_lbl)
        a.set_title(title, fontsize=11)
        a.set_xlabel("time since spoof injection [s]"); a.set_ylabel(ylab)
        a.legend(loc="upper left", fontsize=9); a.grid(alpha=0.25)
    draw(ax[0], 1, OFFSET_THRESH, f"CUSUM threshold = {OFFSET_THRESH} m",
         "Cross-channel offset drift  d_abs = ‖c(t)−c(t₀)‖\nspoof crosses → CUSUM fail-stops; clean stays flat",
         "d_abs  [m]")
    draw(ax[1], 2, JUMP_THRESH, f"memoryless threshold = {JUMP_THRESH} m",
         "Per-update jump  ‖Δc‖  (memoryless detector)\nboth stay below → memoryless MISSES the slow spoof",
         "jump  [m]")
    plt.tight_layout()
    out = "experiment_results/gazebo_s1_s6/money2x2_detector.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print("\nsaved", out)
    return out

if __name__ == "__main__":
    confusion()
    plot()
