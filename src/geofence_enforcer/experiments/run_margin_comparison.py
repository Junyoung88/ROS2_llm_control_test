#!/usr/bin/env python3
"""
Additive vs. RSS vs. empirical margin comparison (reviewer R1/AE: "additive
margin may be overly conservative and double-count coupled effects").

The rebuttal has two halves, and this script quantifies both:

  (1) Under INDEPENDENT (nominal) errors, the additive margin IS conservative:
      an RSS (quadrature) margin achieves the same ~0% violation with a smaller
      buffer.  This is exactly Remark 2 in the paper (conservatism ratio ~1.33).

  (2) Under ADVERSARIALLY ALIGNED errors (all sources pushed to their bound in
      the same direction -- the threat model of Def. 1 / Prop. 1), the RSS
      margin is UNSAFE (violation > 0) while the additive margin remains tight
      (0 violation).  So additive is not "double counting"; it is the exact
      minimax bound for the adversarial setting, and the extra 33% is the price
      of dropping the independence assumption an attacker can violate.

We reuse SimParams and the physics of run_mc from monte_carlo_formula_validation
so the comparison is apples-to-apples with the paper's validation.

Margin schemes compared (all built from the same four component bounds):
  static     : localization term only (z*sigma)          -- naive geofence
  rss        : sqrt(sum M_i^2)                            -- independence bound
  additive   : sum M_i (PETSE)                            -- adversarial bound
  emp_p99    : empirical 99th percentile of realized displacement (nominal)
  emp_p999   : empirical 99.9th percentile (nominal)

Output: experiment_results/gazebo_s1_s6/margin_comparison_results.json
        figures/margin_comparison.png
"""
import json
import math
import os
import sys
from dataclasses import asdict

import numpy as np

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP_DIR)
from monte_carlo_formula_validation import SimParams, X_BOUNDARY  # noqa: E402

OUT = ('/home/jim/ros2_motion_planning_tutorials/experiment_results/'
       'gazebo_s1_s6/margin_comparison_results.json')
FIG = '/home/jim/ros2_motion_planning_tutorials/figures/margin_comparison.png'

N = 2_000_000    # trials per condition
SEED = 12345


def components(p: SimParams):
    """The four physically-grounded margin bounds (at v_max)."""
    return {
        'est':   p.z_value * p.sigma,
        'track': p.e_0 + p.c_1 * p.v_max,
        'lat':   p.v_max * p.tau,
        'brake': p.v_max ** 2 / (2.0 * p.a_max),
    }


def margins(p: SimParams, emp=None):
    c = components(p)
    m = {
        'static':   c['est'],
        'rss':      math.sqrt(sum(v * v for v in c.values())),
        'additive': sum(c.values()),
    }
    if emp:
        m.update(emp)
    return m


def sample_displacement(p: SimParams, n, rng, adversarial: bool):
    """Realized true displacement of the robot past the trigger point,
    expressed as signed error relative to the AMCL estimate.

    Returns array 'reach' = how far the TRUE position ends up beyond the
    estimate-based trigger, so violation is (reach > margin).

    nominal:      each source sampled independently (signed).
    adversarial:  each source at its bound magnitude, all aligned toward zone.
    """
    if adversarial:
        # worst case (Def. 1 / Prop. 1): every source at its bound, all aligned
        # toward the zone. reach then equals the additive margin exactly.
        eps_loc = np.full(n, p.z_value * p.sigma)         # localization at z-bound
        v_act = np.full(n, p.v_max)                        # max speed
        e_track = np.full(n, p.e_0 + p.c_1 * p.v_max)      # tracking at bound
        tau_act = np.full(n, p.tau)                        # latency at bound
        d_kin = v_act * tau_act + v_act ** 2 / (2.0 * p.a_max)
        return eps_loc + e_track + d_kin

    # nominal: independent SIGNED errors (matches run_mc physics). Localization
    # and tracking can point either way (and cancel); the kinematic overshoot is
    # always toward the zone (robot moves forward before stopping).
    eps_loc = rng.normal(0.0, p.sigma, n)                  # signed
    v_act = rng.uniform(0.80 * p.v_max, p.v_max, n)
    e_bound = p.e_0 + p.c_1 * v_act
    e_track = rng.uniform(-1.0, 1.0, n) * e_bound          # signed
    tau_act = rng.uniform(0.0, 2.0 * p.tau, n)
    d_kin = v_act * tau_act + v_act ** 2 / (2.0 * p.a_max)
    return eps_loc + e_track + d_kin


def run_condition(p: SimParams, n, seed, adversarial: bool):
    rng = np.random.default_rng(seed)
    reach = sample_displacement(p, n, rng, adversarial)

    # empirical margins derived from the NOMINAL distribution only
    result = {}
    for name, M in _all_margins(p, reach if not adversarial else None).items():
        viol = reach > M
        k = int(viol.sum())
        result[name] = {
            'margin': round(float(M), 4),
            'violations': k,
            'n': n,
            'viol_pct': 100.0 * k / n,
            'viol_upper95': 100.0 * wilson_upper(k, n),
            'mean_clearance_m': round(float((M - reach).mean()), 4),
        }
    return result


_EMP_CACHE = {}


def _all_margins(p: SimParams, reach_nominal):
    if reach_nominal is not None:
        emp = {
            'emp_p99':  float(np.percentile(reach_nominal, 99.0)),
            'emp_p999': float(np.percentile(reach_nominal, 99.9)),
        }
        _EMP_CACHE['emp'] = emp
    else:
        emp = _EMP_CACHE.get('emp', {})
    return margins(p, emp)


def wilson_upper(k, n, z=1.96):
    if n == 0:
        return 0.0
    ph = k / n
    denom = 1 + z * z / n
    center = ph + z * z / (2 * n)
    spread = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n))
    return min(1.0, (center + spread) / denom)


def plot(data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    order = ['static', 'rss', 'emp_p99', 'emp_p999', 'additive']
    labels = {'static': 'Static\n($z\\sigma$ only)', 'rss': 'RSS\n$\\sqrt{\\sum M_i^2}$',
              'emp_p99': 'Empirical\np99', 'emp_p999': 'Empirical\np99.9',
              'additive': 'Additive\n(PETSE)'}
    nom = data['nominal']
    adv = data['adversarial']

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    # panel 1: margin size
    ax = axes[0]
    ms = [nom[k]['margin'] for k in order]
    bars = ax.bar(range(len(order)), ms,
                  color=['#999', '#f9a825', '#66bb6a', '#2e7d32', '#1a73e8'])
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([labels[k] for k in order], fontsize=8)
    ax.set_ylabel('margin (m)')
    ax.set_title('(a) Margin size')
    for b, m in zip(bars, ms):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.005, f'{m:.3f}',
                ha='center', fontsize=8)
    ax.grid(True, axis='y', alpha=0.25)

    # panel 2: nominal violation (independent errors)
    ax = axes[1]
    vp = [nom[k]['viol_pct'] for k in order]
    vu = [nom[k]['viol_upper95'] for k in order]
    ax.bar(range(len(order)), vp,
           color=['#999', '#f9a825', '#66bb6a', '#2e7d32', '#1a73e8'])
    ax.plot(range(len(order)), vu, 'k_', ms=14, label='95% upper bound')
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([labels[k] for k in order], fontsize=8)
    ax.set_ylabel('zone violation rate (%)')
    ax.set_title('(b) Independent errors (nominal)')
    ax.grid(True, axis='y', alpha=0.25)
    ax.legend(fontsize=7)

    # panel 3: adversarial-aligned violation
    ax = axes[2]
    va = [adv[k]['viol_pct'] for k in order]
    bars = ax.bar(range(len(order)), va,
                  color=['#999', '#f9a825', '#66bb6a', '#2e7d32', '#1a73e8'])
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([labels[k] for k in order], fontsize=8)
    ax.set_ylabel('zone violation rate (%)')
    ax.set_title('(c) Adversarially aligned errors')
    for b, v in zip(bars, va):
        if v > 0.5:
            ax.text(b.get_x() + b.get_width() / 2, v + 1, f'{v:.0f}%',
                    ha='center', fontsize=8, color='#c62828')
    ax.grid(True, axis='y', alpha=0.25)

    fig.suptitle('Additive vs. RSS vs. empirical margins: conservatism under '
                 'independence, necessity under adversarial alignment',
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, dpi=150)
    print(f'[margin] Figure -> {FIG}')


def main():
    p = SimParams()
    # nominal first (defines empirical margins), then adversarial reuses them
    nominal = run_condition(p, N, SEED, adversarial=False)
    adversarial = run_condition(p, N, SEED + 1, adversarial=True)

    data = {
        'params': asdict(p),
        'components': {k: round(v, 4) for k, v in components(p).items()},
        'nominal': nominal,
        'adversarial': adversarial,
        'n_per_condition': N,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(data, f, indent=2)
    print(f'[margin] Saved -> {OUT}')

    c = components(p)
    print(f"\nComponents (m): est={c['est']:.3f} track={c['track']:.3f} "
          f"lat={c['lat']:.3f} brake={c['brake']:.3f}")
    print(f"Additive={sum(c.values()):.3f}  "
          f"RSS={math.sqrt(sum(v*v for v in c.values())):.3f}  "
          f"ratio={sum(c.values())/math.sqrt(sum(v*v for v in c.values())):.3f}")

    order = ['static', 'rss', 'emp_p99', 'emp_p999', 'additive']
    print(f"\n{'scheme':>10} {'margin':>7} | "
          f"{'NOMINAL viol%':>13} {'clear(m)':>9} | {'ADVERS viol%':>12}")
    print('-' * 62)
    for k in order:
        print(f"{k:>10} {nominal[k]['margin']:>7.3f} | "
              f"{nominal[k]['viol_pct']:>12.4f}% {nominal[k]['mean_clearance_m']:>9.3f} | "
              f"{adversarial[k]['viol_pct']:>11.2f}%")

    plot(data)


if __name__ == '__main__':
    main()
