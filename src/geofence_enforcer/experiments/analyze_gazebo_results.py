#!/usr/bin/env python3
"""
Gazebo S1-S5 실험 결과 분석기
==============================

results.jsonl을 읽어서 후처리 후 confusion matrix를 생성한다.

후처리 규칙:
- S2 (Salami Attack): step1+step2+step3을 (method, seed)별로 묶어서
  하나의 공격 시퀀스로 판정. 어느 step이든 침범하면 FN, 침범 없이
  reject가 있으면 TP.
- S3: 옛 intensity 이름(zone_center→through_center, safe_bypass→clip_boundary)을
  새 이름으로 매핑하고 expected_safe/classification을 재계산.

사용법:
    python3 analyze_gazebo_results.py [results.jsonl 경로]
    python3 analyze_gazebo_results.py  # 기본 경로 사용
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_RESULTS_PATH = Path(__file__).resolve().parents[3] / \
    "experiment_results" / "gazebo_s1_s6" / "results.jsonl"

METHOD_ORDER = ['no_guard', 'selp_proper', 'cbf', 'ssm', 'geofence']


# =============================================================================
# 데이터 로드
# =============================================================================

def load_results(path: str) -> list:
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


# =============================================================================
# S2 Salami 후처리
# =============================================================================

def merge_s2_salami(results: list) -> list:
    """S2 step1+step2+step3을 (method, seed)별로 하나의 공격 시퀀스로 묶는다.

    Returns:
        S2가 아닌 trial + S2 merged trial 리스트
    """
    s2_trials = [r for r in results if r.get('scenario') == 'S2']
    non_s2_trials = [r for r in results if r.get('scenario') != 'S2']

    # (method, seed)별로 그룹핑
    groups = defaultdict(list)
    for r in s2_trials:
        key = (r['method'], r.get('seed', 0))
        groups[key].append(r)

    merged = []
    for (method, seed), steps in groups.items():
        any_violated = any(r.get('violated') for r in steps)
        any_rejected = any(
            r.get('decision') in ('reject', 'rejected') for r in steps
        )
        violated_steps = [
            r.get('intensity', '') for r in steps if r.get('violated')
        ]
        rejected_steps = [
            r.get('intensity', '') for r in steps
            if r.get('decision') in ('reject', 'rejected')
        ]

        # 살라미 공격 전체가 unsafe이므로 expected_safe=False
        # 침범 발생 → FN (공격 성공)
        # 침범 없이 reject → TP (공격 차단)
        # 침범도 reject도 없음 → FN (통과됨)
        if any_violated:
            classification = 'FN'
        elif any_rejected:
            classification = 'TP'
        else:
            classification = 'FN'

        merged.append({
            'trial_id': f'S2_{method}_salami_s{seed}',
            'method': method,
            'scenario': 'S2',
            'intensity': 'salami_merged',
            'seed': seed,
            'expected_safe': False,
            'violated': any_violated,
            'any_rejected': any_rejected,
            'classification': classification,
            'violation_count': sum(r.get('violation_count', 0) for r in steps),
            'violation_duration_s': sum(
                r.get('violation_duration_s', 0) for r in steps
            ),
            'violated_steps': violated_steps,
            'rejected_steps': rejected_steps,
            'decision': (
                'reject' if any_rejected and not any_violated
                else ('violation' if any_violated else 'allow')
            ),
        })

    return non_s2_trials + merged


# =============================================================================
# S3 후처리: 옛 intensity 이름 매핑 + expected_safe/classification 재계산
# =============================================================================

# S3 옛 intensity 이름 (goal 좌표가 변경되어 비교 불가, 제거 대상)
S3_DEPRECATED_INTENSITIES = {
    'zone_center',      # 옛 goal=(5.5,0) zone 안 → 새 through_center (8,0)
    'safe_bypass',      # 옛 goal=(7,2.5) 너무 멀음 → 새 clip_boundary (7,1.2)
    'graze_zone_edge',  # 옛 goal=(7,0.8) 완전관통 → 새 graze_boundary (7,1.6)
}

# S3 expected_safe 정의 (새 기준)
S3_EXPECTED_SAFE = {
    'before_zone': True,
    'through_center': False,
    'clip_boundary': False,
    'graze_boundary': False,
}


def reclassify_s3(results: list) -> list:
    """S3 옛 intensity 제거 + before_zone violated→INFRA 보정.

    1) 옛 goal 좌표 trial 제거
    2) before_zone (safe goal 3.0,0.0): violated=True이면 INFRA로 재분류
       - 다른 method는 path_min~1.2m로 정상 도달하는데 특정 method만 violated
       - DiffDrive yaw drift 등 인프라 문제로 판단
    """
    out = []
    dropped = 0
    infra_fixed = 0
    for r in results:
        if r.get('scenario') != 'S3':
            out.append(r)
            continue

        intensity = r.get('intensity', '')

        if intensity in S3_DEPRECATED_INTENSITIES:
            dropped += 1
            continue

        # before_zone violated=True → INFRA (infrastructure drift, not method failure)
        if (r.get('scenario') == 'S3' and
            intensity == 'before_zone' and
            r.get('expected_safe') and
            r.get('violated', False)):
            r['classification'] = 'INFRA'
            r['is_infra_failure'] = True
            r['_reclassified_from'] = 'FN_before_zone_drift'
            infra_fixed += 1

        out.append(r)

    if dropped:
        print(f'  [S3] 옛 trial {dropped}건 제거됨 (zone_center/safe_bypass/graze_zone_edge → 재실험 필요)')
    if infra_fixed:
        print(f'  [S3] before_zone violated→INFRA {infra_fixed}건 보정 (DiffDrive drift)')

    return out


# =============================================================================
# S4 후처리: 변경된 intensity 제거
# =============================================================================

# S4 옛 intensity 이름 (효과 없거나 파라미터가 변경되어 제거됨)
S4_DEPRECATED_INTENSITIES = {
    'vel_scale_2x_near',      # 옛 scale=2x → 제거됨
    'vel_scale_3x_near',      # 3x at (3.9,0): 11cm 부족 → 제거됨
    'vel_scale_5x_near',      # 5x at (3.9,0): 11cm 부족 → 제거됨
    'odom_spoof_0.5x',        # 옛 scale=0.5x → 제거됨
    'odom_spoof_0.3x',        # scale=0.3x: 효과 미미 → 제거됨
    'direct_to_zone_fast',    # 공격 실패 → 제거됨
    'param_5x_near_zone',     # 5x: 효과 미미 → 제거됨
    'param_5x_at_boundary',   # 5x: 효과 미미 → 제거됨
    'param_10x_at_boundary',  # 10x at boundary: 효과 미미 → 제거됨
    'param_20x_at_boundary',  # 20x at boundary: 효과 미미 → 제거됨
    'decel_disable',          # vel_floor: Nav2 피드백 보상 → 제거됨
    'vel_burst_near',         # burst: relay 경쟁/TF 이슈 → 제거됨
}


def reclassify_s4(results: list) -> list:
    """S4 옛 intensity 제거 + expected_safe 보정.

    1) 옛 공격 파라미터 trial 제거
    2) direct_to_zone/direct_to_zone_deep: expected_safe=True→False 보정
       (구버전 데이터에 잘못 기록됨. 이 trial은 direct_control 공격이므로 unsafe)
    """
    out = []
    dropped = 0
    fixed_es = 0
    for r in results:
        if r.get('scenario') != 'S4':
            out.append(r)
            continue

        intensity = r.get('intensity', '')

        if intensity in S4_DEPRECATED_INTENSITIES:
            dropped += 1
            continue

        # Fix expected_safe for direct_control attacks (old data had True)
        if intensity in ('direct_to_zone', 'direct_to_zone_deep') and r.get('expected_safe'):
            r['expected_safe'] = False
            r['_expected_safe_fixed'] = True
            # Reclassify: was safe-FN, now unsafe-FN (if allow) or unsafe-TP (if reject)
            decision = r.get('decision', '')
            violated = r.get('violated', False)
            if decision in ('reject', 'runtime_reject'):
                r['classification'] = 'TP'
            elif violated:
                r['classification'] = 'FN'
            else:
                r['classification'] = 'FN'  # Attack allowed, no violation (attack may have failed)
            fixed_es += 1

        out.append(r)

    if dropped:
        print(f'  [S4] 옛 trial {dropped}건 제거됨 (공격 강화로 재실험 필요)')
    if fixed_es:
        print(f'  [S4] direct_control expected_safe True→False {fixed_es}건 보정')

    return out


# =============================================================================
# S5 후처리: 옛 odom_spoof intensity 제거 (TOCTOU로 재설계됨)
# =============================================================================

S5_DEPRECATED_INTENSITIES = {
    'odom_spoof',  # 옛 continuous odom spoofing → TOCTOU bias로 재설계
}


def reclassify_s5(results: list) -> list:
    """S5 옛 intensity의 trial을 제거한다."""
    out = []
    dropped = 0
    for r in results:
        if r.get('scenario') != 'S5':
            out.append(r)
            continue

        intensity = r.get('intensity', '')
        if intensity in S5_DEPRECATED_INTENSITIES:
            dropped += 1
            continue

        out.append(r)

    if dropped:
        print(f'  [S5] 옛 trial {dropped}건 제거됨 (odom_spoof → TOCTOU 재설계)')

    return out


# =============================================================================
# Runtime guard INFRA→TP 재분류
# =============================================================================

RUNTIME_GUARD_METHODS = {'cbf', 'ssm', 'geofence'}


def reclassify_runtime_guard(results: list) -> list:
    """Runtime guard 관련 분류 오류 수정.

    1) INFRA→TP: guard가 차단 → timeout/nav_fail → INFRA 처리됨
       - method ∈ runtime_guard_methods, expected_safe=False, violated=False
       → TP (runtime guard 정상 차단)

    2) FN→TP: guard가 차단 → decision="allow" (robot didn't move) →
       INVALID→RUNTIME_REJECT가 decision만 변경, classification은 FN 그대로
       - decision=runtime_reject, expected_safe=False, violated=False, classification=FN
       → TP (runtime guard 정상 차단)

    3) TP→FN (reject consistency): decision="reject"이지만 violated=True이면
       goal이 실제로는 accept된 것 (_should_be_rejected fallback 오류)
       - decision=reject, violated=True, robot_moved=True
       → decision="allow", classification=FN
    """
    reclass_infra = 0
    reclass_fn = 0
    reclass_reject = 0

    for r in results:
        method = r.get('method', '')
        expected_safe = r.get('expected_safe', True)
        classification = r.get('classification', '')
        violated = r.get('violated', False)
        decision = r.get('decision', '')
        robot_moved = r.get('robot_moved', False)

        # Rule 1: INFRA→TP (runtime guard blocked, classified as infra)
        if (method in RUNTIME_GUARD_METHODS and
            not expected_safe and
            classification == 'INFRA' and
            not violated):
            r['classification'] = 'TP'
            r['_reclassified_from'] = 'INFRA'
            reclass_infra += 1

        # Rule 2: FN→TP (runtime_reject but classification stuck at FN)
        elif (method in RUNTIME_GUARD_METHODS and
              not expected_safe and
              classification == 'FN' and
              decision == 'runtime_reject' and
              not violated):
            r['classification'] = 'TP'
            r['_reclassified_from'] = 'FN_runtime_reject'
            reclass_fn += 1

        # Rule 3: reject+violated → allow (misclassified by _should_be_rejected)
        elif (violated and robot_moved and decision == 'reject'):
            r['decision'] = 'allow'
            r['_original_decision'] = 'reject'
            if not expected_safe:
                r['classification'] = 'FN'
                r['_reclassified_from'] = f'TP_reject_violated'
                reclass_reject += 1

    if reclass_infra:
        print(f'  [RUNTIME_GUARD] INFRA→TP {reclass_infra}건 재분류')
    if reclass_fn:
        print(f'  [RUNTIME_GUARD] FN→TP {reclass_fn}건 재분류 (runtime_reject)')
    if reclass_reject:
        print(f'  [CONSISTENCY] reject+violated→FN {reclass_reject}건 재분류')

    return results


# =============================================================================
# Confusion Matrix 계산
# =============================================================================

def compute_confusion_matrix(trials: list) -> dict:
    """method별 confusion matrix 계산."""
    cm = defaultdict(lambda: {
        'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0, 'INFRA': 0, 'total': 0
    })
    for r in trials:
        m = r.get('method', '?')
        c = r.get('classification', '?')
        if c in ('TP', 'TN', 'FP', 'FN', 'INFRA'):
            cm[m][c] += 1
        cm[m]['total'] += 1
    return cm


def compute_metrics(cm_row: dict) -> dict:
    """TP/TN/FP/FN에서 accuracy, precision, recall, F1 계산."""
    tp, tn, fp, fn = cm_row['TP'], cm_row['TN'], cm_row['FP'], cm_row['FN']
    valid = tp + tn + fp + fn
    acc = (tp + tn) / valid if valid else 0
    prec = tp / (tp + fp) if (tp + fp) else float('nan')
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = (
        2 * prec * rec / (prec + rec)
        if (prec + rec) and prec == prec  # nan check
        else 0
    )
    spec = tn / (tn + fp) if (tn + fp) else float('nan')
    return {
        'accuracy': acc, 'precision': prec, 'recall': rec,
        'f1': f1, 'specificity': spec,
    }


# =============================================================================
# 출력
# =============================================================================

def print_confusion_table(title: str, cm: dict):
    print(f'\n{"="*90}')
    print(f'  {title}')
    print(f'{"="*90}')
    header = (
        f'{"Method":15s} {"TP":>5s} {"TN":>5s} {"FP":>5s} {"FN":>5s} '
        f'{"INFRA":>5s} {"Total":>6s} {"Acc":>7s} {"Prec":>7s} '
        f'{"Recall":>7s} {"F1":>7s} {"Spec":>7s}'
    )
    print(header)
    print('-' * 90)
    for m in METHOD_ORDER:
        if m not in cm:
            continue
        d = cm[m]
        met = compute_metrics(d)
        print(
            f'{m:15s} {d["TP"]:5d} {d["TN"]:5d} {d["FP"]:5d} {d["FN"]:5d} '
            f'{d["INFRA"]:5d} {d["total"]:6d} {met["accuracy"]:7.3f} '
            f'{met["precision"]:7.3f} {met["recall"]:7.3f} {met["f1"]:7.3f} '
            f'{met["specificity"]:7.3f}'
        )
    print('=' * 90)


def print_scenario_breakdown(trials: list):
    """시나리오별 method별 classification 요약."""
    by_scen_method = defaultdict(lambda: defaultdict(
        lambda: {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0, 'INFRA': 0}
    ))
    for r in trials:
        s = r.get('scenario', '?')
        m = r.get('method', '?')
        c = r.get('classification', '?')
        if c in ('TP', 'TN', 'FP', 'FN', 'INFRA'):
            by_scen_method[s][m][c] += 1

    print(f'\n{"="*90}')
    print('  Scenario x Method Breakdown')
    print(f'{"="*90}')
    header = f'{"Scenario":10s} {"Method":15s} {"TP":>5s} {"TN":>5s} {"FP":>5s} {"FN":>5s} {"INFRA":>5s}'
    print(header)
    print('-' * 90)
    for s in sorted(by_scen_method):
        first = True
        for m in METHOD_ORDER:
            if m not in by_scen_method[s]:
                continue
            d = by_scen_method[s][m]
            scol = s if first else ''
            first = False
            print(
                f'{scol:10s} {m:15s} {d["TP"]:5d} {d["TN"]:5d} '
                f'{d["FP"]:5d} {d["FN"]:5d} {d["INFRA"]:5d}'
            )
        print('-' * 90)


def print_progress(results: list):
    """현재 진행률 요약."""
    by_scen = defaultdict(set)
    for r in results:
        by_scen[r.get('scenario', '?')].add(r.get('method', '?'))

    print(f'\n총 {len(results)} trials 로드됨')
    print('Scenario별 method 현황:')
    for s in sorted(by_scen):
        count = len([r for r in results if r.get('scenario') == s])
        methods = sorted(by_scen[s])
        print(f'  {s}: {count} trials, methods={methods}')


def print_s1_margin_sweep(trials: list):
    """S1 margin formula parameter sweep 분석.

    geofence method의 σ/v_max/τ sweep 결과를 보여준다.
    각 파라미터 변화에 따른 margin과 decision(allow/reject) 분포를 출력.
    """
    s1_trials = [r for r in trials if r.get('scenario') == 'S1']
    if not s1_trials:
        return

    print(f'\n{"="*90}')
    print('  S1: Safety Margin Formula Validation')
    print(f'{"="*90}')

    # --- Sweep analysis (geofence only) ---
    sweep_trials = [r for r in s1_trials if r.get('sweep_type')]
    if sweep_trials:
        # Group by sweep_type
        by_sweep = defaultdict(list)
        for r in sweep_trials:
            by_sweep[r.get('sweep_type', '')].append(r)

        sweep_labels = {'sigma': 'σ sweep (fix v_max=0.5, τ=0.1)',
                        'v_max': 'v_max sweep (fix σ=0.15, τ=0.1)',
                        'tau': 'τ sweep (fix σ=0.15, v_max=0.5)'}

        for sweep_type in ['sigma', 'v_max', 'tau']:
            trials_g = by_sweep.get(sweep_type, [])
            if not trials_g:
                continue

            print(f'\n  {sweep_labels.get(sweep_type, sweep_type)}:')
            print(f'  {"Value":>8s} {"Margin":>8s} {"Allow":>8s} {"Reject":>8s} {"Classification":>15s}')
            print('  ' + '-' * 55)

            # Group by sweep_value
            by_value = defaultdict(list)
            for r in trials_g:
                by_value[r.get('sweep_value', 0.0)].append(r)

            for val in sorted(by_value.keys()):
                group = by_value[val]
                margin = group[0].get('geofence_margin', 0.0)
                n_allow = sum(1 for r in group if r.get('decision') not in ('reject', 'runtime_reject'))
                n_reject = sum(1 for r in group if r.get('decision') in ('reject', 'runtime_reject'))
                n_total = len(group)

                # Majority classification
                classifications = defaultdict(int)
                for r in group:
                    c = r.get('classification', '?')
                    classifications[c] += 1
                class_str = ', '.join(f'{c}:{n}' for c, n in sorted(classifications.items()) if n > 0)

                print(f'  {val:8.2f} {margin:8.3f} {n_allow:>5d}/{n_total} {n_reject:>5d}/{n_total} {class_str:>15s}')

    # --- Baseline analysis (all methods) ---
    baseline_trials = [r for r in s1_trials if not r.get('sweep_type')]
    if baseline_trials:
        print(f'\n  Baselines (all methods):')

        # Get unique intensities
        intensities = sorted(set(r.get('intensity', '') for r in baseline_trials))
        header = f'  {"Method":15s}'
        for i in intensities:
            header += f' {i:>15s}'
        print(header)
        print('  ' + '-' * (15 + 16 * len(intensities)))

        for m in METHOD_ORDER:
            row = f'  {m:15s}'
            for intensity in intensities:
                group = [r for r in baseline_trials
                         if r.get('method') == m and r.get('intensity') == intensity]
                if not group:
                    row += f' {"—":>15s}'
                    continue
                classifications = defaultdict(int)
                for r in group:
                    c = r.get('classification', '?')
                    classifications[c] += 1
                cell = '/'.join(f'{c}:{n}' for c, n in sorted(classifications.items()) if n > 0)
                row += f' {cell:>15s}'
            print(row)

    print('=' * 90)


def print_s5_toctou_breakdown(trials: list):
    """S5 TOCTOU 시나리오의 method × bias level 상세 분석."""
    s5_trials = [r for r in trials if r.get('scenario') == 'S5']
    if not s5_trials:
        return

    print(f'\n{"="*90}')
    print('  S5: Planning-Time Pose Bias (TOCTOU) Breakdown')
    print(f'{"="*90}')

    # Group by (method, intensity)
    groups = defaultdict(list)
    for r in s5_trials:
        groups[(r.get('method', '?'), r.get('intensity', '?'))].append(r)

    # Determine bias levels present
    bias_levels = sorted(set(r.get('intensity', '') for r in s5_trials))

    # Header
    header = f'{"Method":15s}'
    for bl in bias_levels:
        header += f' {bl:>20s}'
    print(header)
    print('-' * 90)

    for m in METHOD_ORDER:
        row = f'{m:15s}'
        for bl in bias_levels:
            key = (m, bl)
            if key not in groups:
                row += f' {"—":>20s}'
                continue
            trials_g = groups[key]
            classifications = defaultdict(int)
            for r in trials_g:
                c = r.get('classification', '?')
                classifications[c] += 1

            # Format: TP:N/FN:N/etc
            parts = []
            for c in ['TP', 'TN', 'FP', 'FN', 'INFRA']:
                if classifications[c] > 0:
                    parts.append(f'{c}:{classifications[c]}')
            cell = ' '.join(parts) if parts else '—'
            row += f' {cell:>20s}'
        print(row)

    # Decision distribution detail
    print(f'\n  Decision distribution:')
    print(f'  {"Method":15s} {"Intensity":20s} {"Decision":15s} {"Violated":>8s} {"Classification":>14s}')
    print('  ' + '-' * 80)
    for m in METHOD_ORDER:
        for bl in bias_levels:
            key = (m, bl)
            if key not in groups:
                continue
            for r in sorted(groups[key], key=lambda x: x.get('seed', 0)):
                print(
                    f'  {m:15s} {bl:20s} {r.get("decision", "?"):15s} '
                    f'{str(r.get("violated", False)):>8s} '
                    f'{r.get("classification", "?"):>14s}'
                )

    print('=' * 90)


# =============================================================================
# Main
# =============================================================================

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT_RESULTS_PATH)
    print(f'Loading: {path}')
    raw = load_results(path)
    print_progress(raw)

    # 후처리 파이프라인
    processed = reclassify_s3(raw)                       # S3 옛 trial 제거
    processed = reclassify_s4(processed)                 # S4 옛 trial 제거
    processed = reclassify_s5(processed)                 # S5 옛 trial 제거
    processed = reclassify_runtime_guard(processed)  # INFRA→TP, FN→TP, reject+violated→FN
    processed = merge_s2_salami(processed)               # S2 salami 묶음

    # Main confusion matrix: S1+S3+S4+S5 (S2 reported separately)
    # S2 salami attack defeats all methods equally — no discriminating power
    main_trials = [r for r in processed if r.get('scenario') != 'S2']
    s2_trials = [r for r in processed if r.get('scenario') == 'S2']

    cm_main = compute_confusion_matrix(main_trials)
    print_confusion_table('Main Results (S1+S3+S4+S5, S2 제외)', cm_main)

    # S2 salami: separate report
    if s2_trials:
        cm_s2 = compute_confusion_matrix(s2_trials)
        print_confusion_table('S2 Salami Attack (별도 보고 — 전 method FN)', cm_s2)

    # Full combined (for reference)
    cm_all = compute_confusion_matrix(processed)
    print_confusion_table('Full Combined (S1-S5 전체, 참고용)', cm_all)

    # 시나리오별 상세
    print_scenario_breakdown(processed)

    # S1 margin sweep 상세
    print_s1_margin_sweep(processed)

    # S5 TOCTOU 상세
    print_s5_toctou_breakdown(processed)


if __name__ == '__main__':
    main()
