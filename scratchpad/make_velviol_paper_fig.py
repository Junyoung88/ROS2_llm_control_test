#!/usr/bin/env python3
"""English clearance figure for the paper (R1 velocity-bound violation): per-seed minimum
clearance to the forbidden zone for No Guard / reactive fixed-margin baseline / PETSE, with the
0.55 m safety-margin line. Saved to paper/fig_velocity_reactive.png."""
import json, glob, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.chdir("/home/jim/ros2_motion_planning_tutorials")
RES = "experiment_results/gazebo_s1_s6/assum_violation"
MARGIN = 0.55
ORDER = ["no_guard", "static_reactive", "geofence"]
LABELS = {"no_guard": "No Guard", "static_reactive": "Reactive\nfixed margin",
          "geofence": "PETSE\n(runtime\nre-verify)"}
COLORS = {"no_guard": "#9e9e9e", "static_reactive": "#ef6c00", "geofence": "#2e7d32"}

data = {}
for m in ORDER:
    clr = []
    for f in sorted(glob.glob(f"{RES}/res_velreact_{m}_v*.jsonl")):
        if os.path.getsize(f) == 0:
            continue
        try:
            d = json.load(open(f))
        except Exception:
            continue
        pmd = d.get("path_min_distance")
        if pmd is None or pmd == float("inf"):
            pmd = 0.0
        clr.append(max(0.0, pmd))
    if clr:
        data[m] = clr
ms = [m for m in ORDER if m in data]

fig, ax = plt.subplots(figsize=(5.4, 3.5))
ax.axhspan(-0.3, 0.0, color="#c62828", alpha=0.15)
ax.axhline(0.0, color="#c62828", lw=1.4)
ax.text(len(ms) - 0.5, -0.17, "forbidden zone (breach)", fontsize=8.5, color="#c62828",
        ha="right", va="center")
ax.axhline(MARGIN, color="#555", lw=1.3, ls="--")
ax.text(len(ORDER) - 1 + 0.3, MARGIN + 0.05, f"safety margin {MARGIN} m", fontsize=8.5,
        color="#555", ha="right")
for i, m in enumerate(ms):
    vals = data[m]
    mean = sum(vals) / len(vals)
    breach = sum(1 for v in vals if v < MARGIN)
    ax.bar(i, mean, width=0.5, color=COLORS[m], edgecolor="black", linewidth=0.7, alpha=0.5)
    for j, v in enumerate(vals):
        ax.plot(i + (j - len(vals)/2) * 0.05, v, "o", color=COLORS[m],
                markeredgecolor="black", markersize=5, markeredgewidth=0.5)
    ax.text(i, max(mean, max(vals)) + 0.13, f"breach {breach}/{len(vals)}", ha="center",
            fontsize=9.5, fontweight="bold",
            color="#2e7d32" if breach == 0 else "#c62828")
ax.set_ylim(-0.3, 2.7)
ax.set_xlim(-0.62, len(ms) - 0.38)
ax.set_xticks(range(len(ms)))
ax.set_xticklabels([LABELS[m] for m in ms], fontsize=9.5)
ax.set_ylabel("Min. clearance to zone (m)", fontsize=10.5)
ax.grid(axis="y", alpha=0.25)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
out = "paper/fig_velocity_reactive.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print("wrote", out, "::", "  ".join(
    f"{m}=breach{sum(1 for v in data[m] if v<MARGIN)}/{len(data[m])}(min{min(data[m]):.2f})" for m in ms))
