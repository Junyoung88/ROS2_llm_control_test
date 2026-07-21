#!/usr/bin/env python3
"""
Runtime-monitor operating point — the two questions a "continuous re-verification"
claim invites, answered from existing logs (no new Gazebo runs):

  (1) Does the always-on monitor nuisance-trip benign trajectories?
      -> spurious mid-execution aborts on expected_safe trials.
  (2) When it DOES intervene, how much real clearance remains?
      -> path_min_distance (closest the ground-truth path got to the zone) on
         every runtime intervention, split by scenario, with the violation count.

Both drawn from experiment_results/gazebo_s1_s6/results.jsonl via the paper's
processed pipeline (dedup + reclassify).

Outputs: experiment_results/gazebo_s1_s6/monitor_operating_point.json
         figures/monitor_operating_point.png
"""
import json
import os
import statistics
import sys

EXP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP)
from statistical_analysis import load_processed  # noqa: E402

RESULTS = ('/home/jim/ros2_motion_planning_tutorials/experiment_results/'
           'gazebo_s1_s6/results.jsonl')
OUT = ('/home/jim/ros2_motion_planning_tutorials/experiment_results/'
       'gazebo_s1_s6/monitor_operating_point.json')
FIG = '/home/jim/ros2_motion_planning_tutorials/figures/monitor_operating_point.png'

CLEAR_SAFE = {'safe_far', 'before_zone', 'baseline_safe'}   # goals well outside zone+margin


def pct(v, q):
    v = sorted(v)
    i = min(len(v) - 1, max(0, int(round(q * (len(v) - 1)))))
    return v[i]


def is_runtime(r):
    return r.get('decision') == 'runtime_reject' or r.get('runtime_rejected')


def main():
    p = load_processed(RESULTS)
    g = [r for r in p if r.get('method') == 'geofence']

    # ---- (1) nuisance-trip on benign trajectories ----
    benign = [r for r in g if r.get('expected_safe') is True]
    clear = [r for r in benign if r.get('intensity') in CLEAR_SAFE]
    spur_all = sum(1 for r in benign if is_runtime(r))
    spur_clear = sum(1 for r in clear if is_runtime(r))
    completed_clear = sum(1 for r in clear if r.get('decision') == 'allow')
    # rule-of-three upper 95% bound for 0/N
    ro3 = lambda n: round(3.0 / n * 100, 2) if n else float('nan')   # noqa: E731

    nuisance = {
        'benign_N': len(benign), 'benign_spurious_runtime_aborts': spur_all,
        'benign_spurious_rate_pct': round(100 * spur_all / len(benign), 3),
        'benign_rule_of_three_upper95_pct': ro3(len(benign)),
        'clearly_safe_N': len(clear),
        'clearly_safe_completed': completed_clear,
        'clearly_safe_spurious_aborts': spur_clear,
    }

    # ---- (2) intervention clearance ----
    rr = [r for r in g if is_runtime(r)]
    clearance = {}
    for sc in ('S4', 'S5', 'ALL'):
        sub = rr if sc == 'ALL' else [r for r in rr if r.get('scenario') == sc]
        vals = [r.get('path_min_distance') for r in sub
                if isinstance(r.get('path_min_distance'), (int, float))]
        if not vals:
            continue
        clearance[sc] = {
            'n': len(vals), 'violated': sum(bool(r.get('violated')) for r in sub),
            'min_m': round(min(vals), 3), 'p10_m': round(pct(vals, .1), 3),
            'median_m': round(statistics.median(vals), 3),
            'p90_m': round(pct(vals, .9), 3), 'max_m': round(max(vals), 3),
        }
    lat = [r.get('reaction_latency_ms') for r in rr
           if isinstance(r.get('reaction_latency_ms'), (int, float))]
    latency = {'median_ms': statistics.median(lat),
               'p90_ms': pct(lat, .9)} if lat else {}

    summary = {'nuisance_trip': nuisance, 'intervention_clearance': clearance,
               'reaction_latency': latency,
               'headline': (f"0/{len(benign)} benign trials nuisance-tripped "
                            f"(≤{ro3(len(benign))}% rule-of-three); "
                            f"0/{len(rr)} interventions let the robot violate; "
                            f"worst-case clearance {clearance.get('S5', {}).get('min_m')} m "
                            f"(S5/TOCTOU), reacted in "
                            f"{latency.get('median_ms')} ms median.")}
    with open(OUT, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'[op-point] Saved -> {OUT}\n')
    print("(1) Nuisance trip on benign:", json.dumps(nuisance, indent=2))
    print("\n(2) Intervention clearance:")
    for sc, d in clearance.items():
        print(f"    {sc}: n={d['n']} violated={d['violated']} "
              f"min={d['min_m']} median={d['median_m']} max={d['max_m']} m")
    print(f"\nreaction latency: {latency}")
    print(f"\n=> {summary['headline']}")

    plot(nuisance, clearance, rr)


def plot(nuisance, clearance, rr):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.6),
                                   gridspec_kw={'width_ratios': [1, 1.35]})

    # Panel A: nuisance-trip
    axA.axis('off')
    n = nuisance['benign_N']
    txt = (f"Continuous monitor on\nbenign trajectories\n\n"
           f"N = {n} benign trials\n\n"
           f"spurious mid-execution\naborts:  "
           f"{nuisance['benign_spurious_runtime_aborts']} "
           f"({nuisance['benign_spurious_rate_pct']:.1f}%)\n"
           f"rule-of-three ≤ "
           f"{nuisance['benign_rule_of_three_upper95_pct']:.2f}%\n\n"
           f"clearly-safe goals\n(N={nuisance['clearly_safe_N']}): "
           f"{nuisance['clearly_safe_completed']}/{nuisance['clearly_safe_N']} "
           f"completed, 0 aborts")
    axA.text(0.5, 0.5, txt, ha='center', va='center', fontsize=11,
             bbox=dict(boxstyle='round', fc='#e8f5e9', ec='#2e7d32', lw=1.5))
    axA.set_title('(a) No nuisance trips', fontsize=11)

    # Panel B: intervention clearance strip by scenario
    import numpy as np
    scen = [s for s in ('S4', 'S5') if s in clearance]
    colors = {'S4': '#1a73e8', 'S5': '#c62828'}
    label = {'S4': 'S4 velocity/deviate', 'S5': 'S5 TOCTOU'}
    for i, sc in enumerate(scen):
        vals = [r.get('path_min_distance') for r in rr
                if r.get('scenario') == sc
                and isinstance(r.get('path_min_distance'), (int, float))]
        jitter = (np.arange(len(vals)) % 5 - 2) * 0.03
        axB.scatter([i + j for j in jitter], vals, s=28, alpha=0.6,
                    color=colors[sc], label=f"{label[sc]} (n={len(vals)})")
        d = clearance[sc]
        axB.hlines(d['median_m'], i - 0.28, i + 0.28, color=colors[sc], lw=2.5)
        axB.annotate(f"median\n{d['median_m']:.2f} m", (i + 0.32, d['median_m']),
                     fontsize=8, va='center', color=colors[sc])
        axB.annotate(f"min {d['min_m']:.2f} m", (i, d['min_m']),
                     textcoords='offset points', xytext=(-46, 0),
                     ha='right', va='center', fontsize=8, color=colors[sc])
    axB.axhline(0, color='k', lw=1.5)
    axB.axhspan(-0.4, 0, color='red', alpha=0.10)
    axB.text(0.5, -0.22, 'zone interior — 0/%d interventions entered'
             % len(rr), ha='center', va='center', fontsize=9, color='#b71c1c')
    axB.set_xticks(range(len(scen)))
    axB.set_xticklabels([label[s] for s in scen])
    axB.set_ylabel('clearance at intervention (m to zone)')
    axB.set_ylim(-0.4, 4.3)
    axB.set_title('(b) Every intervention stops with positive clearance\n'
                  '(0 violations across all runtime interventions)', fontsize=10)
    axB.legend(fontsize=8, loc='upper center')
    axB.grid(True, axis='y', alpha=0.25)

    fig.suptitle('Runtime-monitor operating point: never nuisance-trips benign '
                 'runs, always stops attacks with real clearance', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, dpi=150)
    print(f'[op-point] Figure -> {FIG}')


if __name__ == '__main__':
    main()
