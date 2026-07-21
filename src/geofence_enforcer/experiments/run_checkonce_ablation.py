#!/usr/bin/env python3
"""
Check-once vs. continuous re-verification ablation (the paper's real thesis).

Reviewers read PETSE as "just a wider margin". This ablation shows the load-
bearing contribution is NOT the margin but the CONTINUOUS RUNTIME MONITOR that
re-verifies an already-approved task during execution. We hold the margin FIXED
(PETSE's 0.562 m) and ask a counterfactual:

    If we KEEP the full margin but disable the runtime monitor -- i.e. check
    once at approval time (goal + path gate) and never re-check -- what happens?

Every PETSE detection (TP) is attributed to the gate that caught it:
  * pre-motion  : goal/path gate at approval time  (decision == 'reject')
  * runtime     : runtime monitor during execution (decision == 'runtime_reject'
                  or reason mentions the runtime guard / TOCTOU bypass)

The "check-once" variant keeps only pre-motion detections; every TP that was
caught at runtime flips to FN (the approved task was safe at check time and went
wrong afterwards). Because the margin is identical in both arms, any recall drop
is attributable to continuous re-verification alone, not to margin width.

Scenario mapping (code -> paper): S1->S1 goal, S3->S2 path, S4->S3 velocity/
deviate, S5->S4 spoof/TOCTOU. (Code S2 salami is excluded from the paper.)

Output: experiment_results/gazebo_s1_s6/checkonce_ablation_results.json
        figures/checkonce_ablation.png
"""
import json
import os
import sys
from collections import defaultdict

EXP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP)
from statistical_analysis import load_processed  # noqa: E402

OUT = ('/home/jim/ros2_motion_planning_tutorials/experiment_results/'
       'gazebo_s1_s6/checkonce_ablation_results.json')
FIG = '/home/jim/ros2_motion_planning_tutorials/figures/checkonce_ablation.png'
RESULTS = ('/home/jim/ros2_motion_planning_tutorials/experiment_results/'
           'gazebo_s1_s6/results.jsonl')

PAPER = {'S1': 'S1\n(goal)', 'S3': 'S2\n(path)', 'S4': 'S3\n(velocity/\ndeviate)',
         'S5': 'S4\n(spoof/\nTOCTOU)'}
SCEN_ORDER = ['S1', 'S3', 'S4', 'S5']   # paper scenarios (code labels)


def gate_of(r):
    """Which gate caught this trial: 'runtime' vs 'premotion'."""
    dec = r.get('decision', '')
    reason = r.get('reason', '') or ''
    if dec == 'runtime_reject' or r.get('runtime_rejected') is True \
            or 'Runtime guard' in reason or 'TOCTOU bypass' in reason:
        return 'runtime'
    return 'premotion'


def main():
    p = load_processed(RESULTS)
    g = [r for r in p if r.get('method') == 'geofence'
         and r.get('sweep_type', '') == '']

    rows = []
    tot_unsafe = tot_full = tot_premotion = 0
    for sc in SCEN_ORDER:
        sub = [r for r in g if r.get('scenario') == sc]
        unsafe = [r for r in sub if not r.get('expected_safe', True)
                  and r.get('classification') in ('TP', 'FN')]
        tp = [r for r in unsafe if r.get('classification') == 'TP']
        tp_runtime = [r for r in tp if gate_of(r) == 'runtime']
        tp_premotion = [r for r in tp if gate_of(r) == 'premotion']

        n = len(unsafe)
        full_recall = len(tp) / n if n else float('nan')
        # counterfactual: disable runtime monitor, keep margin
        checkonce_recall = len(tp_premotion) / n if n else float('nan')
        rows.append({
            'scenario_code': sc, 'scenario_paper': PAPER[sc].replace('\n', ' '),
            'n_unsafe': n,
            'tp_full': len(tp),
            'tp_premotion': len(tp_premotion),
            'tp_runtime': len(tp_runtime),
            'recall_full': round(full_recall, 4),
            'recall_checkonce': round(checkonce_recall, 4),
            'runtime_only_flips_to_FN': len(tp_runtime),
        })
        tot_unsafe += n
        tot_full += len(tp)
        tot_premotion += len(tp_premotion)

    overall = {
        'n_unsafe': tot_unsafe,
        'recall_full': round(tot_full / tot_unsafe, 4),
        'recall_checkonce': round(tot_premotion / tot_unsafe, 4),
        'runtime_only_flips_to_FN': tot_full - tot_premotion,
        'margin_m': 0.562,
    }

    data = {'per_scenario': rows, 'overall': overall}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(data, f, indent=2)
    print(f'[ablation] Saved -> {OUT}')

    print(f"\nMargin held FIXED at {overall['margin_m']} m in BOTH arms.\n")
    print(f"{'paper':>16} {'n':>4} {'full recall':>12} "
          f"{'check-once':>12} {'runtime-only':>13}")
    for r in rows:
        print(f"{r['scenario_paper']:>16} {r['n_unsafe']:>4} "
              f"{r['recall_full']:>11.1%} {r['recall_checkonce']:>12.1%} "
              f"{r['tp_runtime']:>13}")
    print(f"{'OVERALL':>16} {overall['n_unsafe']:>4} "
          f"{overall['recall_full']:>11.1%} {overall['recall_checkonce']:>12.1%} "
          f"{overall['runtime_only_flips_to_FN']:>13}")
    print(f"\n=> Removing ONLY the runtime monitor (margin unchanged) drops "
          f"recall {overall['recall_full']:.0%} -> {overall['recall_checkonce']:.0%}. "
          f"The {overall['runtime_only_flips_to_FN']} lost detections are all "
          f"post-approval failures no static check can catch.")

    plot(rows, overall)


def plot(rows, overall):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [PAPER[r['scenario_code']] for r in rows]
    co = [r['recall_checkonce'] * 100 for r in rows]
    rt = [(r['recall_full'] - r['recall_checkonce']) * 100 for r in rows]

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    x = np.arange(len(rows))
    b1 = ax.bar(x, co, 0.6, label='check-once (goal+path gate, margin=0.562 m)',
                color='#90a4ae')
    b2 = ax.bar(x, rt, 0.6, bottom=co, color='#1a73e8',
                label='added by continuous runtime monitor')
    ax.axhline(100, color='#2e7d32', ls=':', lw=1)

    for i, r in enumerate(rows):
        if r['tp_runtime'] > 0:
            ax.annotate(f"+{r['tp_runtime']}\nruntime-only",
                        (i, co[i] + rt[i]), ha='center', va='bottom',
                        fontsize=8, color='#1a73e8')
        ax.annotate(f"{r['recall_checkonce']:.0%}", (i, co[i] / 2),
                    ha='center', va='center', fontsize=8, color='white')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('detection recall (%)')
    ax.set_ylim(0, 112)
    ax.set_title('Check-once vs. continuous re-verification '
                 '(margin held fixed at 0.562 m)\n'
                 f"overall recall {overall['recall_checkonce']:.0%} → "
                 f"{overall['recall_full']:.0%} from the runtime monitor alone")
    ax.legend(fontsize=8, loc='lower left')
    ax.grid(True, axis='y', alpha=0.25)
    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, dpi=150)
    print(f'[ablation] Figure -> {FIG}')


if __name__ == '__main__':
    main()
