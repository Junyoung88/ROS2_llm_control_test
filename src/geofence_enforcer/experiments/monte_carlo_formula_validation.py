#!/usr/bin/env python3
"""
Monte Carlo Simulation for Geofence Safety Margin Formula Validation.

Validates:  M = z_{1-ε}·σ + (e₀ + c₁·v) + v·τ + v²/(2·a_max)

by injecting realistic noise into a kinematic robot-approach model and
measuring zone violation rate as a function of the geofence margin M.

Key claims validated
--------------------
1. Formula margin M=0.562m achieves <0.3% violation rate (design target).
2. Ablation: removing any term raises violation rate at the same M value.
3. Sensitivity: formula-predicted M keeps violation rate <1% across all
   physically reasonable parameter ranges.

Physical model (1-D)
--------------------
Robot moves along x-axis toward zone boundary at x = x_boundary.

  Guard triggers when:   x_est  ≥  x_boundary − M
  Estimated position:    x_est  =  x_true + ε_loc
  True pos. at trigger:  x_trig =  (x_boundary − M) − ε_loc

After trigger, robot still travels:
  d_after = v_act · τ_act  +  v_act² / (2 · a_max)
            ^^^^^^^^^^^^        ^^^^^^^^^^^^^^^^
            latency phase       braking phase

Final robot position with tracking deviation:
  x_final = x_trig + d_after + ε_track

Violation:  x_final > x_boundary

Usage
-----
  python3 monte_carlo_formula_validation.py            # 100k trials, save + plot
  python3 monte_carlo_formula_validation.py --n 1000000 --no-plot
"""

import argparse
import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _PLT = True
except ImportError:
    _PLT = False
    print("WARNING: matplotlib not available — skipping figure generation")


# ─────────────────────────────────────────────────────────────────────────────
# z-quantile (Abramowitz & Stegun approximation, no scipy needed)
# ─────────────────────────────────────────────────────────────────────────────

def _z_quantile(epsilon: float) -> float:
    """Compute z_{1-ε} via rational approximation of the probit function."""
    p = 1.0 - epsilon
    if p <= 0.0 or p >= 1.0:
        return 3.0
    t = math.sqrt(-2.0 * math.log(min(p, 1.0 - p)))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    z = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)
    if p < 0.5:
        z = -z
    return z


# ─────────────────────────────────────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SimParams:
    """
    Representative physical noise parameters for mobile robot operations.
    Values based on typical indoor mobile robot characteristics (RA-L formulation).
    """
    epsilon: float = 0.003  # Risk parameter ε (quantile confidence)
    sigma:   float = 0.15   # AMCL localization std σ (m)
    e_0:     float = 0.03   # Static tracking error bound (m)
    c_1:     float = 0.04   # Velocity-dependent tracking coefficient (s)
    tau:     float = 0.10   # System response latency upper bound (s)
    v_max:   float = 0.50   # Max approach velocity (m/s)
    a_max:   float = 2.50   # Max deceleration magnitude (m/s²)

    @property
    def z_value(self) -> float:
        return _z_quantile(self.epsilon)

    @property
    def e_track(self) -> float:
        """Combined tracking error at v_max."""
        return self.e_0 + self.c_1 * self.v_max

    @property
    def formula_margin(self) -> float:
        """M = z_{1-ε}·σ + (e₀ + c₁·v) + v·τ + v²/(2·a_max)"""
        return (self.z_value * self.sigma
                + self.e_track
                + self.v_max * self.tau
                + self.v_max ** 2 / (2.0 * self.a_max))

    @property
    def breakdown(self) -> Dict[str, float]:
        return {
            r"localization ($z\sigma$)": self.z_value * self.sigma,
            r"tracking ($e_0{+}c_1 v$)": self.e_track,
            r"latency ($v\tau$)":         self.v_max * self.tau,
            r"braking ($v^2/2a$)":        self.v_max ** 2 / (2.0 * self.a_max),
        }

    def formula_str(self) -> str:
        bd = self.breakdown
        vals = list(bd.values())
        return (f"M = z(ε={self.epsilon})×{self.sigma}"
                f" + ({self.e_0}+{self.c_1}×{self.v_max})"
                f" + {self.v_max}×{self.tau}"
                f" + {self.v_max}²/(2×{self.a_max})\n"
                f"  = {vals[0]:.3f}"
                f" + {vals[1]:.3f}"
                f" + {vals[2]:.3f}"
                f" + {vals[3]:.3f}"
                f" = {self.formula_margin:.3f} m"
                f"  [z={self.z_value:.3f}]")


# ─────────────────────────────────────────────────────────────────────────────
# Core Monte Carlo
# ─────────────────────────────────────────────────────────────────────────────

X_BOUNDARY = 4.0   # zone left wall (m)


def run_mc(params: SimParams, margin: float, n: int, seed: int = 42) -> Dict:
    """
    Run n Monte Carlo trials for robot approaching zone boundary.

    Returns
    -------
    dict with violation_rate, viol_pct, mean_clearance, worst_breach_m, etc.
    """
    rng = np.random.default_rng(seed)

    # Sample noise sources
    eps_loc   = rng.normal(0.0, params.sigma, n)                              # AMCL error
    v_act     = rng.uniform(0.80 * params.v_max, params.v_max, n)            # speed
    # Velocity-dependent tracking error
    e_track_bound = params.e_0 + params.c_1 * v_act                           # per-sample
    eps_track = rng.uniform(-1.0, 1.0, n) * e_track_bound                    # tracking
    tau_act   = rng.uniform(0.0, 2.0 * params.tau, n)                        # latency

    # Physics
    x_trig  = (X_BOUNDARY - margin) - eps_loc
    d_after = v_act * tau_act + v_act ** 2 / (2.0 * params.a_max)
    x_final = x_trig + d_after + eps_track

    viol = x_final > X_BOUNDARY
    n_viol = int(viol.sum())

    return {
        "margin":          round(float(margin), 4),
        "n":               n,
        "violations":      n_viol,
        "violation_rate":  float(viol.mean()),
        "viol_pct":        float(viol.mean() * 100.0),
        "mean_clearance_m": float((X_BOUNDARY - x_final).mean()),
        "worst_breach_m":  float((x_final[viol] - X_BOUNDARY).max()) if viol.any() else 0.0,
    }


def sweep_margins(params: SimParams, margins, n: int, seed0: int = 42) -> List[Dict]:
    return [run_mc(params, float(m), n, seed0 + i) for i, m in enumerate(margins)]


# ─────────────────────────────────────────────────────────────────────────────
# Study definitions
# ─────────────────────────────────────────────────────────────────────────────

MARGINS = np.round(np.arange(0.00, 1.02, 0.04), 3)

# Build-up: incrementally add each formula term
# To zero out the localization term, set epsilon=0.5 so z=0
BUILDUP_CONFIGS: Dict[str, Dict] = {
    "No formula (M=0)":                  {"epsilon": 0.5, "e_0": 0.0, "c_1": 0.0,
                                          "tau": 0.0,     "a_max": 1e12},
    r"+ Localization ($z\sigma$)":       {"e_0": 0.0, "c_1": 0.0,
                                          "tau": 0.0, "a_max": 1e12},
    r"+ Tracking ($e_0{+}c_1 v$)":      {"tau": 0.0, "a_max": 1e12},
    r"+ Latency ($v\tau$)":             {"a_max": 1e12},
    r"+ Braking ($v^2/2a$)  [Full]":    {},
}

# Sequential colormap
BUILDUP_COLORS = [
    "#c62828",   # No formula
    "#e65100",   # + Localization
    "#f9a825",   # + Tracking
    "#2e7d32",   # + Latency
    "#1a73e8",   # + Braking (Full)
]

# Leave-one-out: remove ONE term at a time
LEAVE_ONE_OUT_CONFIGS: Dict[str, Dict] = {
    "Full formula":          {},
    r"w/o Localization":     {"epsilon": 0.5},      # z=0
    r"w/o Tracking":         {"e_0": 0.0, "c_1": 0.0},
    r"w/o Latency":          {"tau":     0.0},
    r"w/o Braking":          {"a_max":   1e12},
}
LOO_COLORS = [
    "#1a73e8",   # Full formula (blue)
    "#c62828",   # w/o Localization (red)
    "#e65100",   # w/o Tracking (orange)
    "#f9a825",   # w/o Latency (yellow-orange)
    "#6a1b9a",   # w/o Braking (purple)
]

# Sensitivity: vary one physical parameter; always apply formula-predicted margin
SENSITIVITY_SWEEPS: Dict[str, List[float]] = {
    "sigma":  [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    "v_max":  [0.10, 0.20, 0.30, 0.50, 0.70, 1.00, 1.50],
    "tau":    [0.00, 0.05, 0.10, 0.20, 0.30, 0.50, 0.80],
    "a_max":  [0.50, 1.00, 1.50, 2.50, 4.00, 8.00, 20.0],
}

PARAM_AXIS_LABELS = {
    "sigma":  r"Localization $\sigma$ (m)",
    "v_max":  r"Max velocity $v$ (m/s)",
    "tau":    r"System latency $\tau$ (s)",
    "a_max":  r"Max deceleration $a_{\max}$ (m/s²)",
}

# Stress test: vary one parameter while keeping margin FIXED at base M
STRESS_SWEEPS: Dict[str, List[float]] = {
    "sigma":  [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 1.00, 1.50],
    "e_0":    [0.00, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50],
    "tau":    [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.80, 1.00],
    "a_max":  [0.30, 0.50, 0.75, 1.00, 1.50, 2.50, 4.00, 8.00],
}


def run_ablation(base: SimParams, n: int) -> Dict:
    """Build-up study: incrementally add each formula term."""
    physical_curve = sweep_margins(base, MARGINS, n)

    buildup: Dict[str, Dict] = {}
    for label, overrides in BUILDUP_CONFIGS.items():
        formula_p = SimParams(**{**asdict(base), **overrides})
        M_pred = formula_p.formula_margin
        r = run_mc(base, M_pred, n, seed=42)
        r["formula_margin"] = round(M_pred, 4)
        buildup[label] = r

    return {"physical_curve": physical_curve, "buildup": buildup}


def run_leave_one_out(base: SimParams, n: int) -> Dict[str, Dict]:
    """Leave-one-out ablation: remove ONE term from full formula at a time."""
    loo: Dict[str, Dict] = {}
    full_margin = base.formula_margin

    for label, overrides in LEAVE_ONE_OUT_CONFIGS.items():
        formula_p = SimParams(**{**asdict(base), **overrides})
        M_pred = formula_p.formula_margin
        r = run_mc(base, M_pred, n, seed=42)
        r["formula_margin"] = round(M_pred, 4)
        r["missing_term_m"] = round(full_margin - M_pred, 4)
        loo[label] = r

    return loo


def run_sensitivity(base: SimParams, n: int) -> Dict[str, List[Dict]]:
    """For each parameter, sweep its value and apply the formula-predicted margin."""
    out: Dict[str, List[Dict]] = {}
    for param, vals in SENSITIVITY_SWEEPS.items():
        rows: List[Dict] = []
        for v in vals:
            p = SimParams(**{**asdict(base), param: v})
            m = p.formula_margin
            r = run_mc(p, m, n, seed=7777)
            r["param_value"]    = float(v)
            r["formula_margin"] = float(m)
            rows.append(r)
        out[param] = rows
    return out


def run_stress_test(base: SimParams, n: int) -> Dict[str, List[Dict]]:
    """
    Stress test: for each parameter value, compute violation rate under TWO
    margin strategies: fixed at base M vs. formula-adapted M.
    """
    fixed_margin = base.formula_margin
    out: Dict[str, List[Dict]] = {}
    for param, vals in STRESS_SWEEPS.items():
        rows: List[Dict] = []
        for v in vals:
            p = SimParams(**{**asdict(base), param: v})
            adapted_margin = p.formula_margin

            r_fixed   = run_mc(p, fixed_margin,   n, seed=9999)
            r_adapted = run_mc(p, adapted_margin,  n, seed=8888)

            rows.append({
                "param_value":      float(v),
                "base_value":       float(getattr(base, param)),
                "fixed_margin":     float(fixed_margin),
                "adapted_margin":   float(adapted_margin),
                "fixed_viol_pct":   r_fixed["viol_pct"],
                "adapted_viol_pct": r_adapted["viol_pct"],
                "viol_pct":         r_fixed["viol_pct"],
                "violation_rate":   r_fixed["violation_rate"],
            })
        out[param] = rows
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def _savefig(fig, path: Path, verbose: bool = True, dpi: int = 300) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    if verbose:
        print(f"    → {path}")


def fig_violation_vs_margin(sweep: List[Dict], params: SimParams, path: Path) -> None:
    """Main result: violation rate vs. margin."""
    margins = [r["margin"]   for r in sweep]
    rates   = [max(r["viol_pct"], 2e-4) for r in sweep]
    M = params.formula_margin

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.axvspan(0,   M,   alpha=0.07, color="red",   zorder=0)
    ax.axvspan(M,   1.0, alpha=0.05, color="green", zorder=0)

    ax.semilogy(margins, rates, "o-", color="#1a73e8", lw=2, ms=6,
                label="Zone violation rate", zorder=3)
    ax.axvline(M,   color="#e53935", ls="--", lw=1.8,
               label=f"Formula M = {M:.3f} m")
    ax.axhline(0.3, color="grey",   ls=":",  lw=1.2,
               label="0.3 % (design safety target)")

    for r in sweep:
        if abs(r["margin"] - M) < 0.022:
            ax.annotate(f"{r['viol_pct']:.3f}%",
                        xy=(M, max(r["viol_pct"], 2e-4)),
                        xytext=(M + 0.06, max(r["viol_pct"], 2e-4) * 3),
                        fontsize=10, color="#e53935",
                        arrowprops=dict(arrowstyle="->", color="#e53935"))

    ax.text(M / 2,       50, "Under-protected\n(violations ↑)",
            ha="center", va="center", color="red",   fontsize=10, alpha=0.7)
    ax.text((M + 1.0) / 2, 50, "Safe region",
            ha="center", va="center", color="green", fontsize=10, alpha=0.7)

    ax.set_xlabel("Geofence Margin M (m)", fontsize=13)
    ax.set_ylabel("Zone Violation Rate (%)", fontsize=13)
    ax.set_title("Geofence Safety Margin vs. Zone Violation Rate\n"
                 r"($M = z_{1-\varepsilon}\sigma + (e_0{+}c_1 v) + v\tau + v^2/2a$)",
                 fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(2e-4, 110)

    fig.tight_layout()
    _savefig(fig, path)


def fig_ablation(ablation: Dict, loo: Dict[str, Dict],
                 params: SimParams, path: Path) -> None:
    """2-panel ablation: build-up (left) and leave-one-out (right)."""
    buildup = ablation["buildup"]
    M = params.formula_margin
    bd = list(params.breakdown.values())

    fig, (ax_bu, ax_loo) = plt.subplots(
        1, 2, figsize=(13, 6),
        gridspec_kw={"width_ratios": [1, 1]})

    bu_rates, bu_margins = [], []
    for r in buildup.values():
        bu_rates.append(max(r["viol_pct"], 2e-4))
        bu_margins.append(r["formula_margin"])

    loo_rates, loo_margins, loo_missing = [], [], []
    for r in loo.values():
        loo_rates.append(max(r["viol_pct"], 2e-4))
        loo_margins.append(r["formula_margin"])
        loo_missing.append(r.get("missing_term_m", 0.0))

    def _draw_bars(ax, rates, colors, step_labels, title, margins_info,
                   info_fmt="M={:.2f}"):
        x_pos = range(len(step_labels))
        ax.bar(x_pos, rates, color=colors, edgecolor="white", width=0.6, zorder=3)
        ax.set_yscale("log")
        ax.axhline(0.3, color="#555555", ls="--", lw=1.5, zorder=4)
        ax.text(len(step_labels) - 0.48, 0.38, "0.3%\ntarget",
                fontsize=8.5, color="#555555", va="bottom", ha="center")

        for i, (v, m_val) in enumerate(zip(rates, margins_info)):
            ax.text(i, v * 2.0, f"{v:.2f}%", ha="center", va="bottom",
                    fontsize=10, fontweight="bold", color=colors[i])
            sub = info_fmt.format(m_val)
            ax.text(i, v * 0.52, sub, ha="center", va="top",
                    fontsize=8, color="white", fontweight="bold")

        ax.set_xticks(list(x_pos))
        ax.set_xticklabels(step_labels, fontsize=10)
        ax.set_yticks([0.1, 1, 10, 100])
        ax.set_yticklabels(["0.1%", "1%", "10%", "100%"], fontsize=10)
        ax.yaxis.grid(False)
        ax.xaxis.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylabel("Zone Violation Rate", fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.set_xlim(-0.5, len(step_labels) - 0.5)
        ax.set_ylim(0.05, 200)

    # ── (a) Build-up ─────────────────────────────────────────────────────────
    bu_step_labels = [
        "No\nformula",
        r"$+z\sigma$" f"\n(+{bd[0]:.2f}m)",
        r"$+e_0{+}c_1 v$" f"\n(+{bd[1]:.2f}m)",
        r"$+v\tau$" f"\n(+{bd[2]:.2f}m)",
        r"$+v^2/2a$" f"\n(+{bd[3]:.2f}m)\n[Full]",
    ]
    _draw_bars(ax_bu, bu_rates, BUILDUP_COLORS, bu_step_labels,
               "(a) Build-up  (sequential addition)\n"
               rf"No formula $\to$ Full $M={M:.3f}\,$m",
               bu_margins, info_fmt="M={:.3f}m")

    # ── (b) Leave-one-out ─────────────────────────────────────────────────────
    loo_step_labels = [
        f"Full\nformula\n({M:.3f}m)",
        r"w/o $z\sigma$" f"\n(−{bd[0]:.2f}m\n→{M-bd[0]:.3f}m)",
        r"w/o $e_0{+}c_1 v$" f"\n(−{bd[1]:.2f}m\n→{M-bd[1]:.3f}m)",
        r"w/o $v\tau$" f"\n(−{bd[2]:.2f}m\n→{M-bd[2]:.3f}m)",
        r"w/o $v^2/2a$" f"\n(−{bd[3]:.2f}m\n→{M-bd[3]:.3f}m)",
    ]
    _draw_bars(ax_loo, loo_rates, LOO_COLORS, loo_step_labels,
               "(b) Leave-one-out  (marginal contribution)\n"
               r"Each term removal yields different violation rate",
               loo_missing, info_fmt="−{:.3f}m")
    # Fix info label for "Full formula" (missing=0 → show M value instead)
    ax_loo.texts[-len(loo_step_labels)].set_text(f"M={M:.3f}m")

    fig.suptitle(
        r"Formula Ablation: $M = z_{1-\varepsilon}\sigma + (e_0{+}c_1 v) + v\tau + v^2/2a$"
        f"  =  {bd[0]:.3f} + {bd[1]:.3f} + {bd[2]:.3f} + {bd[3]:.3f}"
        f"  =  {M:.3f} m",
        fontsize=13, y=1.01)
    fig.tight_layout()
    _savefig(fig, path)


def fig_sensitivity(sensitivity: Dict[str, List[Dict]], path: Path) -> None:
    """2×2 grid: violation rate when formula-predicted margin is used."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes_flat = axes.flatten()

    for ax, (param, rows) in zip(axes_flat, sensitivity.items()):
        x_vals  = [r["param_value"]    for r in rows]
        rates   = [max(r["viol_pct"], 2e-4) for r in rows]
        margins = [r["formula_margin"] for r in rows]

        ax2 = ax.twinx()
        ln1, = ax.semilogy(x_vals, rates,   "o-", color="#1a73e8",
                            lw=2, ms=7, label="Violation rate")
        ln2, = ax2.plot(x_vals, margins, "s--", color="#e53935",
                         lw=1.5, ms=6, label="Formula M (right axis)")

        ax.axhline(0.3, color="grey", ls=":", lw=1.0)
        ax.set_xlabel(PARAM_AXIS_LABELS[param], fontsize=11)
        ax.set_ylabel("Violation Rate (%) at formula M", fontsize=10, color="#1a73e8")
        ax2.set_ylabel("Formula Margin M (m)",           fontsize=10, color="#e53935")
        ax.set_title(f"Sensitivity: {param}", fontsize=12)
        ax.grid(True, which="both", alpha=0.3)
        ax.set_ylim(2e-4, 110)
        ax.legend(handles=[ln1, ln2], fontsize=9)

    fig.suptitle(
        "Parameter Sensitivity: Violation Rate When Formula-Predicted Margin is Used\n"
        "(Rate stays near design target across all physically reasonable values)",
        fontsize=12)
    fig.tight_layout()
    _savefig(fig, path)


def fig_breakdown(params: SimParams, path: Path) -> None:
    """Pie + bar showing the contribution of each formula term to total M."""
    bd     = params.breakdown
    labels = list(bd.keys())
    values = list(bd.values())
    colors = ["#1a73e8", "#34a853", "#fbbc04", "#ea4335"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    wedges, _, autotexts = ax1.pie(
        values, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=90,
        textprops={"fontsize": 11})
    for at in autotexts:
        at.set_fontsize(11)
    ax1.set_title(f"Safety Margin Breakdown\n(Total M = {params.formula_margin:.3f} m)",
                  fontsize=13)

    bars = ax2.barh(labels, values, color=colors, edgecolor="white", height=0.5)
    for bar, v in zip(bars, values):
        ax2.text(v + 0.004, bar.get_y() + bar.get_height() / 2,
                 f"{v:.3f} m", va="center", fontsize=11)
    ax2.set_xlabel("Margin Contribution (m)", fontsize=12)
    ax2.set_title("Safety Margin Components", fontsize=13)
    ax2.set_xlim(0, max(values) * 1.45)
    ax2.grid(axis="x", alpha=0.3)

    fig.suptitle(
        r"$M = z_{1-\varepsilon}\sigma + (e_0{+}c_1 v) + v\tau + v^2/(2a_{\max})$"
        f"\n= {params.z_value:.3f}×{params.sigma}"
        f" + ({params.e_0}+{params.c_1}×{params.v_max})"
        f" + {params.v_max}×{params.tau}"
        f" + {params.v_max}²/(2×{params.a_max})"
        f" = {params.formula_margin:.3f} m",
        fontsize=13)
    fig.tight_layout()
    _savefig(fig, path)


def fig_stress_test(stress: Dict[str, List[Dict]], params: SimParams,
                    path: Path) -> None:
    """
    Publication-quality 2×2 grid: fixed margin vs. adaptive formula margin.
    """
    plt.rcParams.update({
        "font.family":      "serif",
        "font.serif":       ["Times New Roman", "DejaVu Serif"],
        "font.size":        9,
        "axes.labelsize":   9,
        "axes.titlesize":   9,
        "legend.fontsize":  7.5,
        "xtick.labelsize":  8,
        "ytick.labelsize":  8,
        "lines.linewidth":  1.2,
        "lines.markersize": 4,
    })

    M = params.formula_margin
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2))
    axes_flat = axes.flatten()

    SUBPLOT_LABELS = {
        "sigma":  r"(a) Localization uncertainty $\sigma$",
        "e_0":    r"(b) Static tracking error $e_0$",
        "tau":    r"(c) Sensor-to-actuator delay $\tau$",
        "a_max":  r"(d) Braking distance $v^2\!/2a_{\max}$",
    }
    XLABEL = {
        "sigma":  r"$\sigma$: position std. dev. (m)",
        "e_0":    r"$e_0$: static tracking error (m)",
        "tau":    r"$\tau$: worst-case response time (s)",
        "a_max":  r"$v^2\!/2a_{\max}$: stopping distance (m)",
    }

    for ax, (param, rows) in zip(axes_flat, stress.items()):
        fixed_rates   = [r["fixed_viol_pct"]   for r in rows]
        adapted_rates = [r["adapted_viol_pct"] for r in rows]
        base_v        = rows[0]["base_value"]

        if param == "a_max":
            v = params.v_max
            x_vals = [v ** 2 / (2.0 * r["param_value"]) for r in rows]
            base_x = v ** 2 / (2.0 * base_v)
            x_vals        = x_vals[::-1]
            fixed_rates   = fixed_rates[::-1]
            adapted_rates = adapted_rates[::-1]
        else:
            x_vals = [r["param_value"] for r in rows]
            base_x = base_v

        ax.plot(x_vals, fixed_rates, "s--", color="black", lw=1.2,
                ms=4, mfc="none", mew=1.0, zorder=3,
                label="No update ($M$ fixed at base)")
        ax.plot(x_vals, adapted_rates, "o-", color="black", lw=1.2,
                ms=4, mfc="black", zorder=3,
                label="Formula adapts ($M$ recalculated)")

        ax.fill_between(x_vals, adapted_rates, fixed_rates,
                        alpha=0.12, color="grey", zorder=1)

        ax.axvline(base_x, color="grey", ls=":", lw=0.8, zorder=2)

        worst_idx = -1
        ax.annotate(f'{adapted_rates[worst_idx]:.1f}%',
                    xy=(x_vals[worst_idx], adapted_rates[worst_idx]),
                    xytext=(-8, 8), textcoords="offset points",
                    fontsize=7, fontweight="bold", ha="right",
                    arrowprops=dict(arrowstyle="-", color="0.4", lw=0.6))

        ax.set_xlabel(XLABEL[param])
        ax.set_ylabel("Violation rate (%)")
        ax.set_title(SUBPLOT_LABELS[param], loc="left", fontweight="bold")
        ax.grid(True, which="major", alpha=0.25, lw=0.5)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))

    y_max = max(max(r["fixed_viol_pct"] for r in rows)
                for rows in stress.values()) * 1.1
    for ax in axes_flat:
        ax.set_ylim(0, y_max)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2,
               frameon=True, edgecolor="0.8", fancybox=False,
               bbox_to_anchor=(0.5, 1.01))

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _savefig(fig, path)

    plt.rcdefaults()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Monte Carlo geofence margin validation")
    ap.add_argument("--n",       type=int, default=500_000,
                    help="MC trials per margin point (default: 500,000)")
    ap.add_argument("--output",  type=str,
                    default="experiment_results/gazebo_s1_s6/monte_carlo_validation.json")
    ap.add_argument("--fig-dir", type=str, default="figures/formula_validation")
    ap.add_argument("--no-plot", action="store_true",
                    help="Skip figure generation")
    args = ap.parse_args()

    params = SimParams()
    M = params.formula_margin

    print("=" * 65)
    print("  Monte Carlo Geofence Margin Formula Validation")
    print("=" * 65)
    print(f"  {params.formula_str()}")
    print(f"  Trials per point: {args.n:,}")
    print()

    t0 = time.time()

    # ── 1. Margin sweep ──────────────────────────────────────────────────────
    print("[1/5] Margin sweep ...")
    margin_sweep = sweep_margins(params, MARGINS, args.n)

    for r in margin_sweep:
        if abs(r["margin"] - M) < 0.022:
            print(f"       M = {M:.3f} m  →  violation rate = {r['viol_pct']:.4f} %"
                  f"  ({r['violations']:,}/{r['n']:,}  trials)")

    # ── 2a. Build-up ablation ────────────────────────────────────────────────
    print("[2/5] Build-up study  (5 configs) ...")
    ablation = run_ablation(params, args.n)

    print(f"\n       Violation rates as each term is added:")
    print(f"       {'Step':42s}  {'Margin':>8s}  {'Violation':>10s}")
    print(f"       {'-'*65}")
    prev_viol = None
    for label, r in ablation["buildup"].items():
        Mp   = r["formula_margin"]
        viol = r["viol_pct"]
        if prev_viol is not None:
            arrow = f"  ↓ {prev_viol/max(viol,1e-6):.0f}×" if viol < prev_viol else ""
        else:
            arrow = ""
        clean = label.replace("$", "").replace("\\", "").replace("{", "").replace("}", "")
        print(f"       {clean:42s}  M={Mp:.3f} m  {viol:8.3f} %{arrow}")
        prev_viol = viol

    # ── 2b. Leave-one-out ────────────────────────────────────────────────────
    print(f"\n[3/5] Leave-one-out study  (5 configs) ...")
    loo = run_leave_one_out(params, args.n)

    print(f"\n       Violation rates when one term is removed from full formula:")
    print(f"       {'Variant':32s}  {'Margin':>8s}  {'Missing':>8s}  {'Violation':>10s}")
    print(f"       {'-'*70}")
    full_viol = loo.get("Full formula", {}).get("viol_pct", 0.0)
    for label, r in loo.items():
        Mp      = r["formula_margin"]
        viol    = r["viol_pct"]
        missing = r.get("missing_term_m", 0.0)
        if label == "Full formula":
            arrow = "  ← baseline"
        else:
            arrow = f"  ↑ {viol/max(full_viol,1e-6):.0f}× vs. full"
        clean = label.replace("$", "").replace("\\", "").replace("{", "").replace("}", "")
        print(f"       {clean:32s}  M={Mp:.3f} m  -{missing:.3f} m  {viol:8.3f} %{arrow}")

    # ── 3. Parameter sensitivity ─────────────────────────────────────────────
    print(f"\n[4/5] Parameter sensitivity  (4 params × 7 values) ...")
    sensitivity = run_sensitivity(params, args.n)

    # ── 4. Stress test ─────────────────────────────────────────────────────
    print(f"\n[5/5] Stress test  (4 params × ~8 values, fixed M={M:.3f}m) ...")
    stress_test = run_stress_test(params, args.n)

    print(f"\n       Violation rate at base vs. worst-case parameter values:")
    print(f"       {'Parameter':12s}  {'Base':>8s}  {'Base viol':>10s}  {'Worst':>8s}  {'Worst viol':>11s}")
    print(f"       {'-'*60}")
    for param, rows in stress_test.items():
        base_v = rows[0]["base_value"]
        base_row = next(r for r in rows if abs(r["param_value"] - base_v) < 1e-6)
        worst_row = max(rows, key=lambda r: r["viol_pct"])
        print(f"       {param:12s}  {base_v:8.3f}  {base_row['viol_pct']:8.2f} %"
              f"  {worst_row['param_value']:8.3f}  {worst_row['viol_pct']:9.2f} %")

    elapsed = time.time() - t0
    print(f"\n  Simulation complete in {elapsed:.1f} s")

    # ── Save JSON ────────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "timestamp":        time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_per_point":      args.n,
        "params":           asdict(params),
        "formula_margin":   M,
        "margin_breakdown": params.breakdown,
        "margin_sweep":     margin_sweep,
        "ablation":         {
            "physical_curve": ablation["physical_curve"],
            "buildup":        ablation["buildup"],
        },
        "leave_one_out":    loo,
        "sensitivity":      sensitivity,
        "stress_test":      stress_test,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Results saved → {out_path}")

    # ── Figures ──────────────────────────────────────────────────────────────
    if args.no_plot or not _PLT:
        return

    fig_dir = Path(args.fig_dir)
    print(f"\n  Generating figures → {fig_dir}/")
    fig_violation_vs_margin(
        margin_sweep, params,
        fig_dir / "fig1_violation_vs_margin.png")
    fig_ablation(
        ablation, loo, params,
        fig_dir / "fig2_ablation.png")
    fig_sensitivity(
        sensitivity,
        fig_dir / "fig3_sensitivity.png")
    fig_breakdown(
        params,
        fig_dir / "fig4_margin_breakdown.png")
    fig_stress_test(
        stress_test, params,
        fig_dir / "fig5_stress_test.png")
    print("\n  Done.")


if __name__ == "__main__":
    main()
