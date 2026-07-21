#!/usr/bin/env python3
"""Statistical rigor analysis for PETSE paper revision (TII reviewer response).

Adds the statistics reviewers asked for on top of analyze_gazebo_results.py:
  1. Wilson 95% CI for Recall / Precision / Violation Rate per method
  2. Rule-of-three upper bound when 0 violations observed
  3. Exact McNemar test (paired by scenario/intensity/seed) PETSE vs each baseline
  4. Fisher's exact test (unpaired robustness check) on violation counts
  5. Holm-Bonferroni correction across method comparisons
  6. Per-seed variance analysis (mean ± SD of recall/VR across 20 seeds)

No scipy dependency — exact tests implemented with math.comb.

Usage:
    python3 statistical_analysis.py [results.jsonl] [--out-dir DIR]
"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

from analyze_gazebo_results import (
    METHOD_ORDER,
    PAPER_METHOD_NAMES,
    bootstrap_f1_ci,
    load_results,
    merge_s2_salami,
    reclassify_runtime_guard,
    reclassify_s3,
    reclassify_s4,
    reclassify_s5,
    wilson_ci,
)

DEFAULT_RESULTS = Path('/home/jim/ros2_motion_planning_tutorials/'
                       'experiment_results/gazebo_s1_s6/results.jsonl')
DEFAULT_OUT_DIR = DEFAULT_RESULTS.parent

VALID_CLASSES = ('TP', 'TN', 'FP', 'FN')

# Within-margin probes: goals deliberately placed outside the zone but inside
# PETSE's designed margin (near_boundary 0.15m, mid_boundary 0.45m < 0.562m).
# Rejecting them is designed conservatism, not an error — reported as a
# separate "margin-probe rejection" category instead of FP (reviewer metric fix).
MARGIN_PROBE_INTENSITIES = {'near_boundary', 'mid_boundary'}


def is_margin_probe(r: dict) -> bool:
    return r.get('intensity') in MARGIN_PROBE_INTENSITIES


# =============================================================================
# Exact tests (no scipy)
# =============================================================================

def binom_two_sided_p(k: int, n: int) -> float:
    """Two-sided exact binomial p-value for k successes in n trials, p0=0.5.

    Used for McNemar exact test on discordant pairs.
    """
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1))
    p = 2.0 * tail / (2 ** n)
    return min(1.0, p)


def mcnemar_exact(b: int, c: int) -> float:
    """Exact McNemar test. b = A-only-success pairs, c = B-only-success pairs."""
    return binom_two_sided_p(min(b, c), b + c)


def fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher's exact test for the 2x2 table [[a, b], [c, d]].

    Sums hypergeometric probabilities of all tables (with same margins)
    at most as probable as the observed one.
    """
    row1, row2 = a + b, c + d
    col1 = a + c
    n = row1 + row2
    if n == 0:
        return 1.0

    def table_prob(x: int) -> float:
        # P(a=x) under hypergeometric with fixed margins
        return (math.comb(row1, x) * math.comb(row2, col1 - x)
                / math.comb(n, col1))

    lo = max(0, col1 - row2)
    hi = min(col1, row1)
    p_obs = table_prob(a)
    total = sum(table_prob(x) for x in range(lo, hi + 1)
                if table_prob(x) <= p_obs * (1 + 1e-9))
    return min(1.0, total)


def holm_bonferroni(pvals: dict) -> dict:
    """Holm-Bonferroni adjusted p-values. pvals: {name: p} → {name: p_adj}."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted = {}
    running_max = 0.0
    for rank, (name, p) in enumerate(items):
        p_adj = min(1.0, (m - rank) * p)
        running_max = max(running_max, p_adj)  # enforce monotonicity
        adjusted[name] = running_max
    return adjusted


def rule_of_three_upper(n: int) -> float:
    """95% upper bound on event probability when 0 events observed in n trials."""
    return 3.0 / n if n > 0 else float('nan')


# =============================================================================
# Data preparation
# =============================================================================

def dedup_last_wins(results: list) -> list:
    """Keep the last occurrence of each trial_id (resume runs append)."""
    last = {}
    for r in results:
        last[r['trial_id']] = r
    return list(last.values())


def load_processed(path: str) -> list:
    raw = dedup_last_wins(load_results(path))
    p = reclassify_s3(raw)
    p = reclassify_s4(p)
    p = reclassify_s5(p)
    p = reclassify_runtime_guard(p)
    p = merge_s2_salami(p)
    return p


def baseline_trials(processed: list, include_s2: bool) -> list:
    return [r for r in processed
            if r.get('sweep_type', '') == ''
            and (include_s2 or r.get('scenario') != 'S2')]


def pair_key(r: dict) -> tuple:
    return (r.get('scenario'), r.get('intensity'), r.get('seed'))


# =============================================================================
# Per-method summary with CIs
# =============================================================================

def method_summary(trials: list, method: str) -> dict:
    mt_all = [r for r in trials if r.get('method') == method]

    # Split off within-margin probes: their rejection is designed conservatism,
    # so they are excluded from the FP/TN confusion matrix and reported apart.
    probes = [r for r in mt_all if is_margin_probe(r)
              and r.get('classification') in VALID_CLASSES]
    mt = [r for r in mt_all if not is_margin_probe(r)]

    cls = defaultdict(int)
    for r in mt:
        cls[r.get('classification')] += 1
    tp, tn, fp, fn = cls['TP'], cls['TN'], cls['FP'], cls['FN']
    infra = cls['INFRA']
    valid = tp + tn + fp + fn

    n_unsafe = tp + fn
    n_safe = tn + fp
    recall = tp / n_unsafe if n_unsafe else float('nan')
    precision = tp / (tp + fp) if (tp + fp) else float('nan')
    fpr = fp / n_safe if n_safe else float('nan')
    f1 = (2 * precision * recall / (precision + recall)
          if n_unsafe and (tp + fp) and (precision + recall) else 0.0)

    # VR stays a physical metric over ALL valid trials (probes included)
    all_valid = [r for r in mt_all
                 if r.get('classification') in VALID_CLASSES]
    n_viol = sum(1 for r in all_valid if r.get('violated', False))
    vr = n_viol / len(all_valid) if all_valid else float('nan')
    valid_vr = len(all_valid)

    classifications = [r.get('classification') for r in mt
                       if r.get('classification') in VALID_CLASSES]
    f1_lo, f1_hi = bootstrap_f1_ci(classifications)

    # Margin-probe rejection rate (rejection = FP or TP label on a probe)
    n_probe = len(probes)
    probe_rej = sum(1 for r in probes
                    if r.get('classification') in ('FP', 'TP'))

    out = {
        'method': method,
        'name': PAPER_METHOD_NAMES.get(method, method),
        'n': len(mt_all), 'valid': valid, 'infra': infra,
        'n_probe': n_probe, 'probe_rej': probe_rej,
        'probe_rej_rate': probe_rej / n_probe if n_probe else float('nan'),
        'probe_rej_ci': wilson_ci(probe_rej, n_probe),
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'n_unsafe': n_unsafe, 'n_safe': n_safe,
        'recall': recall, 'recall_ci': wilson_ci(tp, n_unsafe),
        'precision': precision,
        'precision_ci': wilson_ci(tp, tp + fp) if (tp + fp) else (0.0, 0.0),
        'fpr': fpr, 'fpr_ci': wilson_ci(fp, n_safe) if n_safe else (0.0, 0.0),
        'f1': f1, 'f1_ci': (f1_lo, f1_hi),
        'n_viol': n_viol, 'vr': vr, 'valid_vr': valid_vr,
        'vr_ci': wilson_ci(n_viol, valid_vr),
    }
    if n_viol == 0 and valid_vr > 0:
        out['vr_rule_of_three'] = rule_of_three_upper(valid_vr)
    return out


# =============================================================================
# Paired comparisons (McNemar) + unpaired robustness (Fisher)
# =============================================================================

def paired_outcomes(trials: list, method_a: str, method_b: str,
                    outcome: str) -> tuple:
    """Build paired success indicators for two methods.

    outcome='detection': unsafe trials only; success = classification TP.
    outcome='no_violation': all valid trials; success = not violated.

    Pairs matched on (scenario, intensity, seed); pairs where either side
    is INFRA (or missing) are dropped.

    Returns (b, c, n_pairs): b = A-success & B-fail, c = A-fail & B-success.
    """
    def eligible(r):
        if r.get('classification') not in VALID_CLASSES:
            return False
        if outcome == 'detection':
            return not r.get('expected_safe', True)
        return True

    def success(r):
        if outcome == 'detection':
            return r.get('classification') == 'TP'
        return not r.get('violated', False)

    a_map = {pair_key(r): success(r) for r in trials
             if r.get('method') == method_a and eligible(r)}
    b_map = {pair_key(r): success(r) for r in trials
             if r.get('method') == method_b and eligible(r)}

    common = sorted(set(a_map) & set(b_map))
    b_cnt = sum(1 for k in common if a_map[k] and not b_map[k])
    c_cnt = sum(1 for k in common if not a_map[k] and b_map[k])
    return b_cnt, c_cnt, len(common)


def compare_methods(trials: list, reference: str = 'geofence') -> dict:
    """McNemar (paired) + Fisher (unpaired) for reference vs every other method."""
    others = [m for m in METHOD_ORDER if m != reference]
    results = {}

    for outcome in ('detection', 'no_violation'):
        raw_p = {}
        rows = {}
        for m in others:
            b, c, n_pairs = paired_outcomes(trials, reference, m, outcome)
            p_mcnemar = mcnemar_exact(b, c)

            # Unpaired Fisher on the same eligible populations
            def counts(method):
                mt = [r for r in trials if r.get('method') == method
                      and r.get('classification') in VALID_CLASSES]
                if outcome == 'detection':
                    mt = [r for r in mt if not r.get('expected_safe', True)]
                    succ = sum(1 for r in mt
                               if r.get('classification') == 'TP')
                else:
                    succ = sum(1 for r in mt if not r.get('violated', False))
                return succ, len(mt) - succ
            sa, fa = counts(reference)
            sb, fb = counts(m)
            p_fisher = fisher_exact_two_sided(sa, fa, sb, fb)

            rows[m] = {
                'pairs': n_pairs, 'ref_only_success': b,
                'other_only_success': c,
                'p_mcnemar': p_mcnemar, 'p_fisher': p_fisher,
                'ref_success': sa, 'ref_fail': fa,
                'other_success': sb, 'other_fail': fb,
            }
            raw_p[m] = p_mcnemar

        adj = holm_bonferroni(raw_p)
        for m in others:
            rows[m]['p_mcnemar_holm'] = adj[m]
        results[outcome] = rows

    return results


# =============================================================================
# Per-seed variance
# =============================================================================

def per_seed_stats(trials: list, method: str) -> dict:
    """Recall and violation rate per seed → mean ± SD across seeds."""
    by_seed = defaultdict(list)
    for r in trials:
        if r.get('method') == method \
                and r.get('classification') in VALID_CLASSES:
            by_seed[r.get('seed')].append(r)

    recalls, vrs = [], []
    for seed, rows in sorted(by_seed.items()):
        unsafe = [r for r in rows if not r.get('expected_safe', True)]
        tp = sum(1 for r in unsafe if r.get('classification') == 'TP')
        if unsafe:
            recalls.append(tp / len(unsafe))
        n_viol = sum(1 for r in rows if r.get('violated', False))
        vrs.append(n_viol / len(rows))

    def mean_sd(xs):
        if not xs:
            return float('nan'), float('nan')
        mu = sum(xs) / len(xs)
        var = (sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)
               if len(xs) > 1 else 0.0)
        return mu, math.sqrt(var)

    r_mu, r_sd = mean_sd(recalls)
    v_mu, v_sd = mean_sd(vrs)
    return {
        'n_seeds': len(by_seed),
        'recall_mean': r_mu, 'recall_sd': r_sd,
        'recall_min': min(recalls) if recalls else float('nan'),
        'recall_max': max(recalls) if recalls else float('nan'),
        'vr_mean': v_mu, 'vr_sd': v_sd,
        'vr_min': min(vrs) if vrs else float('nan'),
        'vr_max': max(vrs) if vrs else float('nan'),
    }


# =============================================================================
# Report generation
# =============================================================================

def fmt_pct(x: float) -> str:
    return f'{x:.1%}' if x == x else '—'


def fmt_ci(ci: tuple) -> str:
    return f'[{ci[0]:.1%}, {ci[1]:.1%}]'


def fmt_p(p: float) -> str:
    if p != p:
        return '—'
    if p < 0.001:
        return '<0.001'
    return f'{p:.3f}'


def build_report(trials: list, scenarios_label: str) -> str:
    lines = []
    lines.append(f'## Baseline comparison with 95% CIs ({scenarios_label})\n')
    lines.append('| Method | N | Valid | TP | FP | TN | FN | INFRA | '
                 'Recall [95% CI] | FPR [95% CI] | F1 [95% CI] | '
                 'Violation Rate [95% CI] | Margin-probe rej. |')
    lines.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|')

    summaries = {}
    for m in METHOD_ORDER:
        s = method_summary(trials, m)
        if s['n'] == 0:
            continue
        summaries[m] = s
        vr_cell = f"{fmt_pct(s['vr'])} {fmt_ci(s['vr_ci'])}"
        if 'vr_rule_of_three' in s:
            vr_cell += (f" (0/{s['valid_vr']}; "
                        f"rule-of-3 ≤{s['vr_rule_of_three']:.2%})")
        probe_cell = (f"{s['probe_rej']}/{s['n_probe']}"
                      if s['n_probe'] else '—')
        lines.append(
            f"| {s['name']} | {s['n']} | {s['valid']} | {s['tp']} | {s['fp']} "
            f"| {s['tn']} | {s['fn']} | {s['infra']} "
            f"| {fmt_pct(s['recall'])} {fmt_ci(s['recall_ci'])} "
            f"| {fmt_pct(s['fpr'])} {fmt_ci(s['fpr_ci'])} "
            f"| {s['f1']:.3f} [{s['f1_ci'][0]:.3f}, {s['f1_ci'][1]:.3f}] "
            f"| {vr_cell} | {probe_cell} |")

    lines.append('')
    lines.append('Recall = TP/(TP+FN) on unsafe trials. '
                 'FPR = FP/(FP+TN) on safe trials, EXCLUDING within-margin '
                 'probes (near/mid_boundary: goals outside the zone but '
                 'inside the designed margin — rejection is intentional '
                 'conservatism, reported separately as margin-probe '
                 'rejections). Violation Rate = physically-entered-zone / '
                 'all valid trials (probes included). '
                 'CIs: Wilson score (proportions), bootstrap 2000 resamples (F1).')
    lines.append('')

    # Margin-probe staircase: rejection per probe distance vs method margin
    lines.append('## Margin-probe rejections by probe distance '
                 '(designed conservatism, not FP)\n')
    lines.append('Probes sit outside the zone but inside PETSE\'s 0.562 m '
                 'margin: near_boundary = 0.15 m, mid_boundary = 0.45 m '
                 'from the boundary. A method rejects a probe iff the probe '
                 'distance is inside its own margin — the staircase below '
                 'is the margin formula acting as designed.\n')
    lines.append('| Method | Margin (m) | near_boundary (0.15 m) | '
                 'mid_boundary (0.45 m) |')
    lines.append('|---|---|---|---|')
    method_margins = {'no_guard': '0', 'selp_proper': '0', 'cbf': '0.30',
                      'cbf_inflated': '0.562', 'ssm': '0.575',
                      'roboguard': '0*', 'geofence': '0.562'}
    for m in METHOD_ORDER:
        probes = [r for r in trials if r.get('method') == m
                  and is_margin_probe(r)
                  and r.get('classification') in VALID_CLASSES]
        if not probes:
            continue
        cells = []
        for intensity in ('near_boundary', 'mid_boundary'):
            sub = [r for r in probes if r.get('intensity') == intensity]
            rej = sum(1 for r in sub
                      if r.get('classification') in ('FP', 'TP'))
            cells.append(f'{rej}/{len(sub)}' if sub else '—')
        lines.append(f"| {PAPER_METHOD_NAMES.get(m, m)} "
                     f"| {method_margins.get(m, '?')} "
                     f"| {cells[0]} | {cells[1]} |")
    lines.append('')

    # Paired tests
    comp = compare_methods(trials, reference='geofence')
    for outcome, title, desc in (
        ('detection',
         'Exact McNemar tests — detection (PETSE vs baseline, unsafe trials)',
         'Success = unsafe trial correctly rejected (TP). Pairs matched on '
         '(scenario, intensity, seed); INFRA pairs dropped.'),
        ('no_violation',
         'Exact McNemar tests — physical safety (PETSE vs baseline, all valid trials)',
         'Success = no physical zone violation during the trial.'),
    ):
        lines.append(f'## {title}\n')
        lines.append(desc + '\n')
        lines.append('| Baseline | Pairs | PETSE only ✓ | Baseline only ✓ | '
                     'McNemar p | Holm-adj. p | Fisher p (unpaired) |')
        lines.append('|---|---|---|---|---|---|---|')
        for m in METHOD_ORDER:
            if m == 'geofence' or m not in comp[outcome]:
                continue
            r = comp[outcome][m]
            lines.append(
                f"| {PAPER_METHOD_NAMES.get(m, m)} | {r['pairs']} "
                f"| {r['ref_only_success']} | {r['other_only_success']} "
                f"| {fmt_p(r['p_mcnemar'])} | {fmt_p(r['p_mcnemar_holm'])} "
                f"| {fmt_p(r['p_fisher'])} |")
        lines.append('')

    # Per-seed variance
    lines.append('## Per-seed variance (across random seeds)\n')
    lines.append('| Method | Seeds | Recall mean ± SD | Recall [min, max] | '
                 'VR mean ± SD | VR [min, max] |')
    lines.append('|---|---|---|---|---|---|')
    for m in METHOD_ORDER:
        if m not in summaries:
            continue
        ps = per_seed_stats(trials, m)
        lines.append(
            f"| {PAPER_METHOD_NAMES.get(m, m)} | {ps['n_seeds']} "
            f"| {ps['recall_mean']:.3f} ± {ps['recall_sd']:.3f} "
            f"| [{ps['recall_min']:.3f}, {ps['recall_max']:.3f}] "
            f"| {ps['vr_mean']:.3f} ± {ps['vr_sd']:.3f} "
            f"| [{ps['vr_min']:.3f}, {ps['vr_max']:.3f}] |")
    lines.append('')

    return '\n'.join(lines), summaries, comp


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    path = args[0] if args else str(DEFAULT_RESULTS)
    out_dir = DEFAULT_OUT_DIR
    if '--out-dir' in sys.argv:
        out_dir = Path(sys.argv[sys.argv.index('--out-dir') + 1])

    print(f'Loading: {path}')
    processed = load_processed(path)

    report_parts = ['# PETSE Statistical Analysis (TII revision)\n']
    report_parts.append(f'Source: `{path}` — '
                        f'{len(processed)} trials after post-processing, '
                        f'baseline subset used below.\n')

    json_out = {}
    for include_s2, label in ((False, 'S1+S3+S4+S5, baseline only'),
                              (True, 'S1-S5 incl. S2 salami, baseline only')):
        trials = baseline_trials(processed, include_s2)
        section, summaries, comp = build_report(trials, label)
        report_parts.append(f'\n# {label}\n')
        report_parts.append(section)
        key = 's1_s3_s4_s5' if not include_s2 else 's1_to_s5'
        json_out[key] = {
            'summaries': summaries,
            'comparisons': comp,
            'n_trials': len(trials),
        }

    report = '\n'.join(report_parts)
    md_path = out_dir / 'statistical_analysis.md'
    json_path = out_dir / 'statistical_analysis.json'
    md_path.write_text(report)
    json_path.write_text(json.dumps(json_out, indent=2, default=str))
    print(report)
    print(f'\nSaved: {md_path}\nSaved: {json_path}')


if __name__ == '__main__':
    main()
