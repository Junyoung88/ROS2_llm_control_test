#!/usr/bin/env python3
"""
Stress-test figure — split into two 1×2 images.
  Top:  (a) σ  +  (b) e_track
  Bot:  (c) τ  +  (d) v²/2a
Same style as original fig5, fonts enlarged for paper.
"""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path("/home/jim/ros2_motion_planning_tutorials")
with open(ROOT / "experiment_results/gazebo_s1_s6/monte_carlo_validation.json") as f:
    st = json.load(f)["stress_test"]

V_MAX = 0.5

# ─── Style ───────────────────────────────────────────────────────────────
STYLE = {
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif"],
    "font.size":         12,
    "axes.labelsize":    13,
    "axes.titlesize":    13,
    "legend.fontsize":   12,
    "xtick.labelsize":   12,
    "ytick.labelsize":   12,
    "lines.linewidth":   1.8,
    "lines.markersize":  6,
    "axes.linewidth":    1.1,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.major.size":  4,
    "ytick.major.size":  4,
}

SUBPLOT_LABELS = {
    "sigma":   r"(a) Localization $\sigma$",
    "e_track": r"(b) Tracking $e_\mathrm{track}$",
    "tau":     r"(c) Latency $\tau$",
    "a_max":   r"(d) Braking distance",
}
XLABEL = {
    "sigma":   r"$\sigma$: position std. dev. (m)",
    "e_track": r"$e_\mathrm{track}$: max lateral deviation (m)",
    "tau":     r"$\tau$: worst-case response time (s)",
    "a_max":   r"$v^2\!/2a_\mathrm{max}$: stopping distance (m)",
}

# Shared Y range across all 4 panels
y_max = max(max(r["fixed_viol_pct"] for r in rows)
            for rows in st.values()) * 1.1


def draw_panel(ax, param, rows):
    fixed   = [r["fixed_viol_pct"]   for r in rows]
    adapted = [r["adapted_viol_pct"] for r in rows]
    base_v  = rows[0]["base_value"]

    if param == "a_max":
        x = [V_MAX**2 / (2 * r["param_value"]) for r in rows]
        base_x = V_MAX**2 / (2 * base_v)
        x, fixed, adapted = x[::-1], fixed[::-1], adapted[::-1]
    else:
        x = [r["param_value"] for r in rows]
        base_x = base_v

    ax.plot(x, fixed, "s--", color="black", lw=2.0, ms=7,
            mfc="none", mew=1.3, zorder=3,
            label="No update ($M$ fixed at base)")
    ax.plot(x, adapted, "o-", color="black", lw=2.0, ms=7,
            mfc="black", zorder=3,
            label="Formula adapts ($M$ recalculated)")

    ax.fill_between(x, adapted, fixed, alpha=0.12, color="grey", zorder=1)
    ax.axvline(base_x, color="grey", ls=":", lw=1.0, zorder=2)

    # Annotate adaptive worst-case
    ax.annotate(f'{adapted[-1]:.1f}%',
                xy=(x[-1], adapted[-1]),
                xytext=(-10, 12), textcoords="offset points",
                fontsize=11, fontweight="bold", ha="right",
                arrowprops=dict(arrowstyle="-", color="0.4", lw=0.8))

    ax.set_xlabel(XLABEL[param])
    ax.set_ylabel("Violation rate (%)")
    ax.set_title(SUBPLOT_LABELS[param], loc="left", fontweight="bold")
    ax.grid(True, which="major", alpha=0.25, lw=0.5)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_ylim(0, y_max)


def make_row(params_pair, suffix):
    """Generate one 1×2 figure for a pair of parameters."""
    plt.rcParams.update(STYLE)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.5))

    p1, p2 = params_pair
    draw_panel(ax1, p1, st[p1])
    draw_panel(ax2, p2, st[p2])

    # Shared legend at top
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=12,
               frameon=True, edgecolor="0.8", fancybox=False,
               bbox_to_anchor=(0.5, 1.02))

    fig.tight_layout(rect=[0, 0, 1, 0.90])

    out_pdf = ROOT / f"figures/formula_validation/stress_plot_{suffix}.pdf"
    out_png = out_pdf.with_suffix('.png')
    fig.savefig(str(out_pdf), format='pdf', bbox_inches='tight')
    fig.savefig(str(out_png), format='png', dpi=300, bbox_inches='tight')
    print(f"Saved → {out_pdf}")
    plt.close()
    plt.rcdefaults()


# ─── Generate two figures ────────────────────────────────────────────────
make_row(("sigma", "e_track"), "top")     # (a) + (b)
make_row(("tau",   "a_max"),   "bottom")  # (c) + (d)
