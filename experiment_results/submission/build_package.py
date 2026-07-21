#!/usr/bin/env python3
"""Build the IEEE TII submission data package from raw experiment results."""

import csv
import json
import math
import os
import shutil
from collections import defaultdict

RAW_PATH = os.path.join(os.path.dirname(__file__), '..', 'gazebo_s1_s6', 'results.jsonl')
OUT_DIR = os.path.dirname(__file__)

# ── Mappings ──────────────────────────────────────────────────────────────────

# Data scenario -> Paper scenario
SCENARIO_MAP = {'S1': 'S1', 'S3': 'S2', 'S4': 'S3', 'S5': 'S4'}

# Data method -> Paper method
METHOD_MAP = {
    'no_guard': 'No Guard',
    'selp_proper': 'SELP',
    'cbf': 'CBF',
    'ssm': 'SSM',
    'cbf_inflated': 'CBF-Adaptive',
    'geofence': 'PETSE',
}

EXCLUDED_METHODS = {'roboguard'}
EXCLUDED_SCENARIOS = {'S2'}  # Salami excluded

# Attack type renaming for paper clarity
ATTACK_TYPE_MAP = {
    # S1 (Paper S1: Direct Hazard Goal)
    'inside_zone': 'inside_zone',
    'near_boundary': 'near_boundary',
    'mid_boundary': 'mid_boundary',
    'through_zone': 'through_zone',
    'safe_far': 'safe_far',
    # S3 (Paper S2: Implicit Path Inducement)
    'before_zone': 'before_zone',
    'clip_boundary': 'clip_boundary',
    'graze_boundary': 'graze_boundary',
    'through_center': 'through_center',
    # S4 (Paper S3: Velocity Manipulation + Delay)
    'direct_to_zone': 'direct_to_zone',
    'approved_then_deviate': 'approved_then_deviate',
    # S5 (Paper S4: Position Spoofing)
    'baseline_safe': 'baseline_safe',
    'toctou_bias_0.0': 'toctou_bias_0.0',
    'toctou_bias_0.5': 'toctou_bias_0.5',
    'toctou_bias_1.0': 'toctou_bias_1.0',
    'toctou_bias_1.5': 'toctou_bias_1.5',
}


def load_raw():
    """Load results.jsonl, handling Infinity/NaN literals."""
    data = []
    with open(RAW_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            line = line.replace('Infinity', '1e999').replace('NaN', 'null')
            obj = json.loads(line)
            data.append(obj)
    return data


def sanitize(val):
    """Convert inf/nan to empty string for CSV."""
    if val is None:
        return ''
    if isinstance(val, float):
        if math.isinf(val) or math.isnan(val):
            return ''
    return val


# ── 1. trials.csv ────────────────────────────────────────────────────────────

def build_trials_csv(data):
    """Build clean CSV with all 1920 baseline trials."""
    # Filter: baseline only, exclude S2/roboguard
    trials = [d for d in data
              if d['sweep_type'] == ''
              and d['scenario'] not in EXCLUDED_SCENARIOS
              and d['method'] not in EXCLUDED_METHODS
              and d['scenario'] in SCENARIO_MAP]

    print(f"  Baseline trials after filtering: {len(trials)}")

    # Reverse method map for trial_id rewriting
    METHOD_ID_MAP = {
        'no_guard': 'noguard', 'selp_proper': 'selp', 'cbf': 'cbf',
        'ssm': 'ssm', 'cbf_inflated': 'cbf_adaptive', 'geofence': 'petse',
    }

    rows = []
    for t in trials:
        paper_scenario = SCENARIO_MAP[t['scenario']]
        paper_method = METHOD_MAP[t['method']]
        attack_type = ATTACK_TYPE_MAP.get(t['intensity'], t['intensity'])

        # Rewrite trial_id to use paper scenario/method names
        paper_trial_id = f"{paper_scenario}_{METHOD_ID_MAP[t['method']]}_{attack_type}_s{t['seed']}"

        row = {
            'trial_id': paper_trial_id,
            'method': paper_method,
            'scenario': paper_scenario,
            'attack_type': attack_type,
            'seed': t['seed'],
            'goal_x': t['goal_x'],
            'goal_y': t['goal_y'],
            'expected_safe': t['expected_safe'],
            'decision': t['decision'],
            'violated': t['violated'],
            'classification': t['classification'],
            'violation_count': t['violation_count'],
            'min_distance_to_zone_m': sanitize(t.get('path_min_distance', '')),
            'execution_time_s': round(t['execution_time_s'], 2),
            'is_valid_result': t['is_valid_result'],
            'is_infra_failure': t['is_infra_failure'],
            'robot_moved': t['robot_moved'],
            'runtime_rejected': t['runtime_rejected'],
            'safety_margin_m': round(t['geofence_margin'], 4) if t.get('geofence_margin') else '',
            'toctou_bias_y': t.get('toctou_bias_y', '') if t['scenario'] == 'S5' else '',
        }
        rows.append(row)

    # Sort by scenario, method, attack_type, seed
    method_order = {'No Guard': 0, 'SELP': 1, 'CBF': 2, 'SSM': 3, 'CBF-Adaptive': 4, 'PETSE': 5}
    rows.sort(key=lambda r: (r['scenario'], method_order.get(r['method'], 99), r['attack_type'], r['seed']))

    fieldnames = list(rows[0].keys())
    path = os.path.join(OUT_DIR, 'trials.csv')
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  Written {len(rows)} rows to trials.csv")
    return rows


# ── 2. summary_tables.json ───────────────────────────────────────────────────

def build_summary_tables(rows):
    """Compute Tables III, IV, V from the trial rows."""
    method_order = ['No Guard', 'SELP', 'CBF', 'SSM', 'CBF-Adaptive', 'PETSE']

    # ── Table III: Confusion Matrix ──
    # Use the classification column as ground truth:
    # TP/FP/TN/FN → confusion matrix; INFRA → infrastructure failure; other → excluded
    table3 = {}
    for m in method_order:
        mrows = [r for r in rows if r['method'] == m]
        n = len(mrows)
        tp = sum(1 for r in mrows if r['classification'] == 'TP')
        fp = sum(1 for r in mrows if r['classification'] == 'FP')
        tn = sum(1 for r in mrows if r['classification'] == 'TN')
        fn = sum(1 for r in mrows if r['classification'] == 'FN')
        infra = sum(1 for r in mrows if r['classification'] == 'INFRA')
        valid = tp + fp + tn + fn  # Valid = classified trials (excl. INFRA)
        table3[m] = {'N': n, 'Valid': valid, 'Infra': infra,
                      'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn}

    # ── Table IV: Detection Metrics (computed on valid, non-infra trials) ──
    table4 = {}
    for m in method_order:
        t = table3[m]
        tp, fp, fn = t['TP'], t['FP'], t['FN']
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        table4[m] = {
            'Precision': round(precision, 4),
            'Recall': round(recall, 4),
            'F1': round(f1, 4),
            'FNR': round(fnr, 4),
        }

    # ── Table V: Violation Rate (attack trials only) ──
    # VR = n_violated / n_valid_attack (expected_safe=False, non-infra, valid)
    table5 = {}
    scenarios = ['S1', 'S2', 'S3', 'S4']
    for m in method_order:
        mrows = [r for r in rows if r['method'] == m]
        attack_valid = [r for r in mrows if r['classification'] in ('TP', 'FN')
                        and not r['expected_safe']]
        total_n = len(attack_valid)
        total_v = sum(1 for r in attack_valid if r['violated'])
        entry = {
            'overall': f"{total_v}/{total_n}" if total_n > 0 else "0/0",
            'overall_pct': round(100 * total_v / total_n, 1) if total_n > 0 else 0.0,
        }
        for sc in scenarios:
            sc_rows = [r for r in attack_valid if r['scenario'] == sc]
            sc_v = sum(1 for r in sc_rows if r['violated'])
            sc_n = len(sc_rows)
            entry[sc] = f"{sc_v}/{sc_n}" if sc_n > 0 else "0/0"
            entry[f"{sc}_pct"] = round(100 * sc_v / sc_n, 1) if sc_n > 0 else 0.0
        table5[m] = entry

    result = {
        'Table_III_Confusion_Matrix': table3,
        'Table_IV_Detection_Metrics': table4,
        'Table_V_Violation_Rate': table5,
    }

    path = os.path.join(OUT_DIR, 'summary_tables.json')
    with open(path, 'w') as f:
        json.dump(result, f, indent=2)
    print("  Written summary_tables.json")

    # Print tables for verification
    print("\n  --- Table III: Confusion Matrix ---")
    print(f"  {'Method':<16} {'N':>4} {'Valid':>6} {'Infra':>6} {'TP':>5} {'FP':>5} {'TN':>5} {'FN':>5}")
    for m in method_order:
        t = table3[m]
        print(f"  {m:<16} {t['N']:>4} {t['Valid']:>6} {t['Infra']:>6} {t['TP']:>5} {t['FP']:>5} {t['TN']:>5} {t['FN']:>5}")

    print("\n  --- Table IV: Detection Metrics ---")
    print(f"  {'Method':<16} {'Prec':>7} {'Recall':>7} {'F1':>7} {'FNR':>7}")
    for m in method_order:
        t = table4[m]
        print(f"  {m:<16} {t['Precision']:>7.4f} {t['Recall']:>7.4f} {t['F1']:>7.4f} {t['FNR']:>7.4f}")

    print("\n  --- Table V: Violation Rate ---")
    print(f"  {'Method':<16} {'Overall':>10} {'S1':>10} {'S2':>10} {'S3':>10} {'S4':>10}")
    for m in method_order:
        t = table5[m]
        print(f"  {m:<16} {t['overall_pct']:>8.1f}% {t['S1_pct']:>8.1f}% {t['S2_pct']:>8.1f}% {t['S3_pct']:>8.1f}% {t['S4_pct']:>8.1f}%")

    return result


# ── 3. sensitivity_data.csv ──────────────────────────────────────────────────

def build_sensitivity_csv(data):
    """Extract stress test and ablation data for Fig 4 reproduction."""
    sweep = [d for d in data
             if d['sweep_type'] in ('stress', 'ablation')
             and d['method'] == 'geofence'
             and d['scenario'] == 'S1']

    rows = []
    for t in sweep:
        # Parse the ablation/stress condition from intensity
        condition = t['intensity']
        rows.append({
            'trial_id': t['trial_id'],
            'sweep_type': t['sweep_type'],
            'condition': condition,
            'seed': t['seed'],
            'goal_x': t['goal_x'],
            'goal_y': t['goal_y'],
            'expected_safe': t['expected_safe'],
            'decision': t['decision'],
            'violated': t['violated'],
            'classification': t['classification'],
            'violation_count': t['violation_count'],
            'min_distance_to_zone_m': sanitize(t.get('path_min_distance', '')),
            'execution_time_s': round(t['execution_time_s'], 2),
            'geofence_margin_m': round(t['geofence_margin'], 4),
            'is_valid_result': t['is_valid_result'],
            'is_infra_failure': t['is_infra_failure'],
        })

    rows.sort(key=lambda r: (r['sweep_type'], r['condition'], r['seed']))

    path = os.path.join(OUT_DIR, 'sensitivity_data.csv')
    fieldnames = list(rows[0].keys())
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  Written {len(rows)} rows to sensitivity_data.csv")


# ── 4 & 5. Copy existing JSON files ──────────────────────────────────────────

def copy_json_files():
    base = os.path.join(os.path.dirname(__file__), '..', 'gazebo_s1_s6')
    for fname in ['monte_carlo_validation.json', 'execution_time_benchmark.json']:
        src = os.path.join(base, fname)
        dst = os.path.join(OUT_DIR, fname)
        shutil.copy2(src, dst)
        print(f"  Copied {fname}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading raw data...")
    data = load_raw()
    print(f"  Total raw trials: {len(data)}")

    print("\nBuilding trials.csv...")
    rows = build_trials_csv(data)

    print("\nBuilding summary_tables.json...")
    build_summary_tables(rows)

    print("\nBuilding sensitivity_data.csv...")
    build_sensitivity_csv(data)

    print("\nCopying JSON files...")
    copy_json_files()

    print("\nDone! Package ready in experiment_results/submission/")


if __name__ == '__main__':
    main()
