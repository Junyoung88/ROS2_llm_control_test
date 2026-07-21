#!/usr/bin/env python3
"""English figure for the paper (S5 localization-spoofing hijack): per-method zone-violation rate
over 5 seeds. Planning-layer baselines (No Guard/SELP/CBF) are all defeated 5/5 by map-consistent
LiDAR spoofing after approval; only PETSE's cross-channel runtime re-verification holds (0/5).
Saved to paper/fig_hijack.png."""
import json, glob, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.chdir("/home/jim/ros2_motion_planning_tutorials")
RES = "experiment_results/gazebo_s1_s6/wh_hijack"
ORDER = ["no_guard", "selp_proper", "cbf_inflated", "geofence"]
LABELS = {"no_guard": "No Guard", "selp_proper": "SELP",
          "cbf_inflated": "CBF\n(planning)", "geofence": "PETSE\n(cross-channel\nruntime)"}
COLORS = {"no_guard": "#9e9e9e", "selp_proper": "#c62828",
          "cbf_inflated": "#1565c0", "geofence": "#2e7d32"}

data = {}
for m in ORDER:
    viol = n = 0; vcs = []
    for f in sorted(glob.glob(f"{RES}/res_wh_hijack_{m}_v*.jsonl")):
        if os.path.getsize(f) == 0:
            continue
        try:
            d = json.load(open(f))
        except Exception:
            continue
        n += 1
        vcs.append(d.get("violation_count") or 0)
        if d.get("violated"):
            viol += 1
    if n:
        data[m] = (viol, n, sum(vcs) / len(vcs))
ms = [m for m in ORDER if m in data]

fig, ax = plt.subplots(figsize=(5.4, 3.4))
rates = [100.0 * data[m][0] / data[m][1] for m in ms]
bars = ax.bar(range(len(ms)), rates, width=0.6,
              color=[COLORS[m] for m in ms], edgecolor="black", linewidth=0.8)
ax.set_ylim(0, 118)
ax.set_ylabel("Zone-violation rate (\\%)".replace("\\", ""), fontsize=10.5)
ax.set_xticks(range(len(ms)))
ax.set_xticklabels([LABELS[m] for m in ms], fontsize=9.5)
for i, m in enumerate(ms):
    v, nn, vc = data[m]
    ax.text(i, rates[i] + 2.5, f"{v}/{nn}", ha="center", va="bottom",
            fontsize=10.5, fontweight="bold",
            color="#2e7d32" if v == 0 else "#c62828")
    if vc > 0:
        ax.text(i, min(rates[i] / 2, 50), f"{vc:.0f}\nsamples", ha="center", va="center",
                fontsize=8, color="white")
ax.grid(axis="y", alpha=0.25)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
out = "paper/fig_hijack.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print("wrote", out, "::", "  ".join(f"{m}={data[m][0]}/{data[m][1]}" for m in ms))
