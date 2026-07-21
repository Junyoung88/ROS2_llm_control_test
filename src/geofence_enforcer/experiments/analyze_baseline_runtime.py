#!/usr/bin/env python3
"""
Bulletproofing (1): is a runtime monitor ENOUGH, or does the mechanism matter?

On the identical S5 TOCTOU bypass attack (bias_1.5, above the goal-gate margin
boundary), we compare every baseline's *own* runtime behavior against PETSE.
Several baselines run a per-cycle check yet still fail, because they do not
continuously re-verify the PATH against an uncertainty-scaled spatial
constraint:
  * CBF        -- per-cycle QP, but goal-only (no path check)      -> fails
  * SSM        -- per-cycle velocity scaling, no spatial zone test -> fails
  * RoboGuard  -- action/goal-level check only (no path, no margin)-> fails
  * CBF-Adaptive -- CBF QP + PETSE's margin + position monitoring  -> catches
  * PETSE      -- continuous path/position re-verification         -> catches

Drawn from the 20-seed baseline campaign (results.jsonl), so no new runs.

Output: experiment_results/gazebo_s1_s6/baseline_runtime_toctou.json
        figures/baseline_runtime_toctou.png
"""
import json
import os
import sys

EXP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP)
from statistical_analysis import load_processed, VALID_CLASSES, wilson_ci  # noqa

RESULTS = ('/home/jim/ros2_motion_planning_tutorials/experiment_results/'
           'gazebo_s1_s6/results.jsonl')
OUT = ('/home/jim/ros2_motion_planning_tutorials/experiment_results/'
       'gazebo_s1_s6/baseline_runtime_toctou.json')
FIG = '/home/jim/ros2_motion_planning_tutorials/figures/baseline_runtime_toctou.png'

# (key, label, mechanism-category, note)
METHODS = [
    ('no_guard', 'No Guard', 'none', 'no check'),
    ('selp_proper', 'SELP', 'planning', 'LTL at planning only'),
    ('cbf', 'CBF', 'runtime_partial', 'per-cycle QP, goal-only'),
    ('ssm', 'SSM', 'runtime_partial', 'per-cycle speed scaling'),
    ('roboguard', 'RoboGuard', 'runtime_partial', 'action/goal-level only'),
    ('cbf_inflated', 'CBF-Adaptive', 'continuous', 'CBF + PETSE margin + pos.'),
    ('geofence', 'PETSE', 'continuous', 'continuous path re-verification'),
]
CATLABEL = {
    'none': 'no runtime check',
    'planning': 'planning-layer only',
    'runtime_partial': 'has a per-cycle check,\nbut no continuous path re-verification',
    'continuous': 'continuous path/position\nre-verification',
}
CATCOLOR = {'none': '#9e9e9e', 'planning': '#9e9e9e',
            'runtime_partial': '#c62828', 'continuous': '#1a73e8'}


def main():
    p = load_processed(RESULTS)
    s5 = [r for r in p if r.get('scenario') == 'S5' and r.get('sweep_type', '') == '']
    rows = []
    for key, label, cat, note in METHODS:
        sub = [r for r in s5 if r.get('method') == key
               and r.get('intensity') == 'toctou_bias_1.5']
        valid = [r for r in sub if r.get('classification') in VALID_CLASSES]
        n = len(valid)
        viol = sum(bool(r.get('violated')) for r in valid)
        lo, hi = wilson_ci(viol, n)
        rows.append({'method': key, 'label': label, 'category': cat, 'note': note,
                     'n': n, 'violated': viol,
                     'violation_rate': round(viol / n, 3) if n else None,
                     'vr_ci': [round(lo, 3), round(hi, 3)]})

    data = {'scenario': 'S5 TOCTOU bias_1.5 (above goal-gate boundary)',
            'seeds': 20, 'rows': rows,
            'headline': 'A per-cycle runtime check is not sufficient: CBF, SSM '
                        'and RoboGuard all run one yet violate 90-100% of TOCTOU '
                        'trials, because none continuously re-verifies the path '
                        'against the spatial constraint. Only continuous path '
                        're-verification (PETSE; CBF-Adaptive uses PETSE\'s '
                        'margin) closes the attack.'}
    with open(OUT, 'w') as f:
        json.dump(data, f, indent=2)
    print(f'[baseline-rt] Saved -> {OUT}\n')
    print(f"{'method':>14} {'category':>16} {'n':>3} {'viol':>5} {'VR':>7}")
    for r in rows:
        print(f"{r['label']:>14} {r['category']:>16} {r['n']:>3} "
              f"{r['violated']:>5} {r['violation_rate']*100:>6.1f}%")
    plot(rows)


def plot(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    x = np.arange(len(rows))
    vr = [r['violation_rate'] * 100 for r in rows]
    colors = [CATCOLOR[r['category']] for r in rows]
    err = [[100*r['violation_rate'] - 100*r['vr_ci'][0] for r in rows],
           [100*r['vr_ci'][1] - 100*r['violation_rate'] for r in rows]]
    bars = ax.bar(x, vr, 0.62, color=colors, yerr=err, capsize=3,
                  error_kw={'elinewidth': 1, 'alpha': 0.5})
    for i, r in enumerate(rows):
        ax.annotate(f"{r['violated']}/{r['n']}", (i, vr[i]),
                    textcoords='offset points', xytext=(0, 4),
                    ha='center', fontsize=9, fontweight='bold')
        ax.annotate(r['note'], (i, 2), rotation=90, ha='center', va='bottom',
                    fontsize=7.5, color='white' if vr[i] > 15 else '#333')
    ax.set_xticks(x)
    ax.set_xticklabels([r['label'] for r in rows], fontsize=10)
    ax.set_ylabel('zone-violation rate on TOCTOU bypass (%)')
    ax.set_ylim(0, 112)
    ax.set_title('A per-cycle runtime check is not enough — the mechanism is\n'
                 'S5 TOCTOU (bias 1.5, above goal-gate boundary), 20 seeds',
                 fontsize=11)
    # legend by category
    from matplotlib.patches import Patch
    seen = []
    handles = []
    for r in rows:
        if r['category'] not in seen and r['category'] in ('runtime_partial', 'continuous'):
            seen.append(r['category'])
            handles.append(Patch(color=CATCOLOR[r['category']],
                                 label=CATLABEL[r['category']]))
    handles.insert(0, Patch(color='#9e9e9e', label='no runtime re-verification'))
    ax.legend(handles=handles, fontsize=8.5, loc='center right')
    ax.grid(True, axis='y', alpha=0.25)
    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, dpi=150)
    print(f'[baseline-rt] Figure -> {FIG}')


if __name__ == '__main__':
    main()
