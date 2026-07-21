#!/usr/bin/env python3
"""
Risk-parameter (epsilon) selection: safety-availability Pareto frontier + a
concrete operator selection rule (reviewer R3 / AE: "the paper does not discuss
how an operator should choose epsilon to balance conservatism vs. availability").

epsilon is PETSE's single risk knob: it sets the localization quantile via
M_est = z_{1-eps} * sigma, so the whole margin M(eps) and hence the
safety-availability trade-off is a function of eps alone.

We produce:
  1. A smooth Pareto frontier (Monte-Carlo violation rate vs. availability loss)
     parameterized by eps, reusing the paper's monte_carlo physics.
  2. The real Gazebo eps-sweep (epsilon_multi) overlaid as validation that the
     violation rate rises monotonically with eps.
  3. A cost-based selection rule eps* = argmin [C_viol*P_viol + C_block*P_block]
     evaluated for several cost ratios, giving operators a recipe.

Availability model: a straight aisle of reference width W_ref loses a passable
strip of 2*M(eps) to the inflated keep-outs (from the narrow-corridor result,
run_geometry_stress: free passage = W - 2M). So availability loss =
min(1, 2*M/W_ref). Reported for a typical 3 m industrial aisle.

Output: experiment_results/gazebo_s1_s6/epsilon_pareto_results.json
        figures/epsilon_pareto.png
"""
import json
import os
import sys
from dataclasses import replace

import numpy as np

EXP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP)
from monte_carlo_formula_validation import SimParams, run_mc  # noqa: E402
from statistical_analysis import load_processed  # noqa: E402

OUT = ('/home/jim/ros2_motion_planning_tutorials/experiment_results/'
       'gazebo_s1_s6/epsilon_pareto_results.json')
FIG = '/home/jim/ros2_motion_planning_tutorials/figures/epsilon_pareto.png'
RESULTS = ('/home/jim/ros2_motion_planning_tutorials/experiment_results/'
           'gazebo_s1_s6/results.jsonl')

W_REF = 3.0          # reference industrial aisle width (m)
N_MC = 3_000_000     # Monte-Carlo trials per epsilon
EPS_GRID = np.concatenate([
    np.geomspace(1e-4, 1e-1, 40),   # fine log grid for the smooth frontier
])
EPS_MARKS = [1e-4, 1e-3, 3e-3, 1e-2, 5e-2, 1e-1]   # labelled points


def curve():
    base = SimParams()
    rows = []
    for i, eps in enumerate(EPS_GRID):
        p = replace(base, epsilon=float(eps))
        M = p.formula_margin
        r = run_mc(p, M, N_MC, seed=1000 + i)
        p_viol = r['violation_rate']
        p_block = min(1.0, 2.0 * M / W_REF)
        rows.append({
            'epsilon': float(eps), 'margin': round(M, 4),
            'p_viol': p_viol, 'p_block': p_block,
            'viol_pct': p_viol * 100, 'block_pct': p_block * 100,
        })
    return rows


def gazebo_empirical():
    """Real Gazebo epsilon_multi sweep: violation rate per epsilon."""
    from collections import defaultdict
    p = load_processed(RESULTS)
    byv = defaultdict(list)
    for r in p:
        if r.get('sweep_type') == 'epsilon_multi':
            byv[r.get('sweep_value')].append(r)
    out = []
    for eps in sorted(byv):
        rs = byv[eps]
        viol = sum(1 for r in rs if r.get('violated'))
        out.append({'epsilon': eps, 'n': len(rs),
                    'viol': viol, 'viol_pct': 100.0 * viol / len(rs)})
    return out


def selection_rule(rows, cost_ratios):
    """eps* = argmin [C_viol*P_viol + C_block*P_block] for each C_viol:C_block."""
    out = {}
    for cv, cb in cost_ratios:
        best = min(rows, key=lambda r: cv * r['p_viol'] + cb * r['p_block'])
        out[f'{cv}:{cb}'] = {
            'C_viol': cv, 'C_block': cb,
            'eps_star': best['epsilon'], 'margin': best['margin'],
            'viol_pct': round(best['viol_pct'], 4),
            'block_pct': round(best['block_pct'], 2),
        }
    return out


def plot(rows, emp, sel):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    eps = [r['epsilon'] for r in rows]
    vp = [r['viol_pct'] for r in rows]
    bp = [r['block_pct'] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.3))

    # (a) both costs vs epsilon (log x)
    axb = ax1.twinx()
    l1, = ax1.plot(eps, vp, '-', color='#c62828', lw=2, label='violation rate')
    l2, = axb.plot(eps, bp, '-', color='#1565c0', lw=2,
                   label='availability loss (3 m aisle)')
    ax1.set_xscale('log')
    ax1.set_xlabel('risk parameter $\\varepsilon$ (log)')
    ax1.set_ylabel('violation rate (%)', color='#c62828')
    axb.set_ylabel('availability loss (%)', color='#1565c0')
    ax1.tick_params(axis='y', colors='#c62828')
    axb.tick_params(axis='y', colors='#1565c0')
    # overlay Gazebo empirical violation
    ax1.plot([e['epsilon'] for e in emp], [e['viol_pct'] for e in emp],
             'o', color='#6a1b9a', ms=6, label='Gazebo empirical viol.')
    # default eps=0.003
    ax1.axvline(0.003, color='#888', ls=':', lw=1)
    ax1.set_title('(a) Safety vs. availability across $\\varepsilon$')
    ax1.legend(handles=[l1, l2,
               plt.Line2D([], [], marker='o', ls='', color='#6a1b9a')],
               labels=['violation rate (MC)', 'availability loss',
                       'Gazebo empirical viol.'],
               fontsize=7, loc='center left')

    # (b) Pareto frontier: availability loss vs violation rate
    ax2.plot(bp, vp, '-', color='#333', lw=1.5, zorder=1)
    sc = ax2.scatter(bp, vp, c=np.log10(eps), cmap='viridis', s=18, zorder=2)
    for em in EPS_MARKS:
        r = min(rows, key=lambda r: abs(r['epsilon'] - em))
        ax2.annotate(f'$\\varepsilon$={em:g}', (r['block_pct'], r['viol_pct']),
                     fontsize=7, xytext=(4, 4), textcoords='offset points')
    # mark selection-rule picks
    for key, s in sel.items():
        ax2.plot(s['block_pct'], s['viol_pct'], '*', ms=13, color='#c62828')
        ax2.annotate(f"$\\varepsilon^*$={s['eps_star']:.1g}\n({key})",
                     (s['block_pct'], s['viol_pct']), fontsize=6.5,
                     xytext=(6, -10), textcoords='offset points', color='#c62828')
    ax2.set_xlabel('availability loss (%)')
    ax2.set_ylabel('violation rate (%)')
    ax2.set_title('(b) Pareto frontier + cost-based $\\varepsilon^*$')
    fig.colorbar(sc, ax=ax2, label='$\\log_{10}\\varepsilon$')
    ax2.grid(True, alpha=0.25)

    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, dpi=150)
    print(f'[eps] Figure -> {FIG}')


def main():
    rows = curve()
    emp = gazebo_empirical()
    # cost ratios: safety-critical (viol very costly), balanced, availability-first
    cost_ratios = [(1000, 1), (100, 1), (10, 1), (1, 1)]
    sel = selection_rule(rows, cost_ratios)

    data = {'frontier': rows, 'gazebo_empirical': emp,
            'selection_rule': sel, 'W_ref_m': W_REF, 'n_mc': N_MC}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(data, f, indent=2)
    print(f'[eps] Saved -> {OUT}')

    print(f"\n{'eps':>8} {'margin':>7} {'viol%':>8} {'block%(3m aisle)':>16}")
    for em in EPS_MARKS:
        r = min(rows, key=lambda r: abs(r['epsilon'] - em))
        print(f"{em:>8g} {r['margin']:>7.3f} {r['viol_pct']:>8.4f} "
              f"{r['block_pct']:>16.2f}")

    print("\nGazebo empirical (epsilon_multi) violation vs eps:")
    for e in emp:
        print(f"  eps={e['epsilon']:<6} viol={e['viol']}/{e['n']} "
              f"({e['viol_pct']:.1f}%)")

    print("\nCost-based selection eps* = argmin[C_viol*P_viol + C_block*P_block]:")
    for key, s in sel.items():
        print(f"  C_viol:C_block = {key:>8}  ->  eps*={s['eps_star']:.2g}  "
              f"(M={s['margin']:.3f}m, viol={s['viol_pct']:.3f}%, "
              f"block={s['block_pct']:.1f}%)")

    plot(rows, emp, sel)


if __name__ == '__main__':
    main()
