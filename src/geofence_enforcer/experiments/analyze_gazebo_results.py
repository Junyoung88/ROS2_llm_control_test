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
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_RESULTS_PATH = Path(__file__).resolve().parents[3] / \
    "experiment_results" / "gazebo_s1_s6" / "results.jsonl"

METHOD_ORDER = ['no_guard', 'selp_proper', 'cbf', 'cbf_inflated', 'ssm', 'roboguard', 'geofence']


# =============================================================================
# Confidence Interval Helpers
# =============================================================================

def wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple:
    """Wilson score 95% CI for a proportion.

    Well-behaved at p=0 and p=1 (unlike Wald interval).
    Formula: (p + z²/2n ± z·√(p(1-p)/n + z²/4n²)) / (1 + z²/n)
    """
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    z2 = z * z
    denom = 1.0 + z2 / total
    center = p + z2 / (2.0 * total)
    spread = z * math.sqrt(p * (1.0 - p) / total + z2 / (4.0 * total * total))
    lo = max(0.0, (center - spread) / denom)
    hi = min(1.0, (center + spread) / denom)
    return (lo, hi)


def bootstrap_f1_ci(classifications: list, n_boot: int = 2000,
                     alpha: float = 0.05, seed: int = 42) -> tuple:
    """Bootstrap 95% CI for F1 score.

    Resamples per-trial classification labels, computes F1 from each resample,
    returns (lower, upper) percentile interval.
    """
    if not classifications:
        return (0.0, 0.0)

    rng = random.Random(seed)
    n = len(classifications)
    f1_samples = []

    for _ in range(n_boot):
        sample = rng.choices(classifications, k=n)
        tp = sample.count('TP')
        fp = sample.count('FP')
        fn = sample.count('FN')
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2.0 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1_samples.append(f1)

    f1_samples.sort()
    lo_idx = int(math.floor(n_boot * alpha / 2.0))
    hi_idx = int(math.ceil(n_boot * (1.0 - alpha / 2.0))) - 1
    lo_idx = max(0, min(lo_idx, n_boot - 1))
    hi_idx = max(0, min(hi_idx, n_boot - 1))
    return (f1_samples[lo_idx], f1_samples[hi_idx])


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
    """S2 후처리: 새 multi-step trials은 이미 단일 trial이므로 merge 불필요.

    레거시 호환: 옛 S2 결과에 step1_safe/step2_margin/step3_violation intensity가
    있으면 (method, seed)별로 묶어서 salami_sequence로 변환.

    Returns:
        All trials (S2 pass-through for new format, merged for legacy format)
    """
    s2_trials = [r for r in results if r.get('scenario') == 'S2']
    non_s2_trials = [r for r in results if r.get('scenario') != 'S2']

    if not s2_trials:
        return results

    # Detect new format: intensity='salami_sequence' or sweep_type != ''
    new_format = [r for r in s2_trials
                  if r.get('intensity') == 'salami_sequence'
                  or r.get('sweep_type', '') != '']
    legacy = [r for r in s2_trials
              if r.get('intensity') != 'salami_sequence'
              and r.get('sweep_type', '') == '']

    if not legacy:
        # All new format — no merging needed
        return results

    # Legacy merge: group (method, seed) and merge step1+step2+step3
    groups = defaultdict(list)
    for r in legacy:
        key = (r['method'], r.get('seed', 0))
        groups[key].append(r)

    merged = []
    for (method, seed), steps in groups.items():
        any_violated = any(r.get('violated') for r in steps)
        any_rejected = any(
            r.get('decision') in ('reject', 'rejected') for r in steps
        )

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
            'intensity': 'salami_sequence',
            'seed': seed,
            'expected_safe': False,
            'violated': any_violated,
            'classification': classification,
            'violation_count': sum(r.get('violation_count', 0) for r in steps),
            'decision': (
                'reject' if any_rejected and not any_violated
                else ('violation' if any_violated else 'allow')
            ),
        })

    return non_s2_trials + new_format + merged


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
    'param_2x_near_zone',     # param injection → DWB adapts → 제거됨
    'param_3x_near_zone',     # param injection → DWB adapts → 제거됨
    'param_latency_near_zone',  # param latency → RA-L redesign에서 제거됨
    'velocity_odom_spoof',    # 옛 pilot 잔재 (S5 TOCTOU로 대체) → 제거됨
    'velocity_scaling_2x',    # 옛 pilot 잔재 → 제거됨
    'direct_to_zone_deep',    # 옛 deep variant → 제거됨
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
    'odom_spoof',       # 옛 continuous odom spoofing → TOCTOU bias로 재설계
    'toctou_bias_0.5',  # RA-L redesign: Δ=0.5 removed (probe battery replaces)
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

RUNTIME_GUARD_METHODS = {'cbf', 'cbf_inflated', 'ssm', 'geofence'}


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
# Violation-Aware Analysis
# =============================================================================

def print_violation_analysis(trials: list):
    """FN 세분화 및 violation rate 분석.

    FN을 두 가지로 분류:
    - FN_viol: 실제 zone 침입이 발생한 진짜 실패 (allow + violated)
    - FN_safe: 허용했지만 침입 없음 (allow + not violated, 목표가 zone 밖)

    Paper에서는 FN_viol이 실질적인 안전 실패이고,
    FN_safe는 margin 내이지만 물리적으로는 안전했던 경우.
    """
    print(f'\n{"="*100}')
    print('  Violation-Aware Analysis (FN breakdown + violation rate)')
    print(f'{"="*100}')

    header = (
        f'{"Method":15s} {"TP":>4s} {"TN":>4s} {"FP":>4s} '
        f'{"FN_viol":>7s} {"FN_safe":>7s} {"FN_tot":>6s} '
        f'{"INFRA":>5s}  {"VR_unsafe":>9s}  {"F1":>6s}  {"F1_viol":>7s}'
    )
    print(header)
    print('-' * 100)

    for m in METHOD_ORDER:
        mt = [r for r in trials if r.get('method') == m]
        if not mt:
            continue

        tp = sum(1 for r in mt if r.get('classification') == 'TP')
        tn = sum(1 for r in mt if r.get('classification') == 'TN')
        fp = sum(1 for r in mt if r.get('classification') == 'FP')
        fn_all = [r for r in mt if r.get('classification') == 'FN']
        fn_viol = sum(1 for r in fn_all if r.get('violated', False))
        fn_safe = sum(1 for r in fn_all if not r.get('violated', False))
        fn_tot = len(fn_all)
        infra = sum(1 for r in mt if r.get('classification') == 'INFRA')

        # VR_unsafe: violation rate among unsafe trials only
        unsafe = [r for r in mt if not r.get('expected_safe', True)]
        n_unsafe = len(unsafe)
        n_viol_unsafe = sum(1 for r in unsafe if r.get('violated', False))
        vr_unsafe = n_viol_unsafe / n_unsafe if n_unsafe else 0

        # Standard F1 (all FN)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn_tot) if (tp + fn_tot) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

        # Violation-aware F1 (only FN_viol counts as failure)
        rec_v = tp / (tp + fn_viol) if (tp + fn_viol) else 0.0
        f1_v = 2 * prec * rec_v / (prec + rec_v) if (prec + rec_v) else 0.0

        print(
            f'{m:15s} {tp:4d} {tn:4d} {fp:4d} '
            f'{fn_viol:7d} {fn_safe:7d} {fn_tot:6d} '
            f'{infra:5d}  {vr_unsafe:8.1%}  {f1:6.3f}  {f1_v:7.3f}'
        )

    print('-' * 100)
    print('  FN_viol: allow + zone violated (real safety failure)')
    print('  FN_safe: allow + no violation (within margin but physically safe)')
    print('  VR_unsafe: zone violation rate among unsafe trials only '
          '(expected_safe=False)')
    print('  F1_viol: F1 using only FN_viol as negatives (violation-aware)')
    print('=' * 100)


def print_violation_by_scenario(trials: list):
    """시나리오별 violation 분석 (probe/config 세분화)."""
    print(f'\n{"="*100}')
    print('  Violation Breakdown by Scenario & Probe')
    print(f'{"="*100}')

    for scenario in sorted(set(r.get('scenario', '?') for r in trials)):
        st = [r for r in trials if r.get('scenario') == scenario]
        print(f'\n  {scenario} ({len(st)} trials):')

        # Group by intensity/config
        by_intensity = defaultdict(list)
        for r in st:
            key = r.get('intensity', r.get('sweep_type', ''))
            by_intensity[key].append(r)

        for intensity in sorted(by_intensity.keys()):
            group = by_intensity[intensity]
            if not group:
                continue
            n_viol = sum(1 for r in group if r.get('violated', False))
            n_fn = sum(1 for r in group if r.get('classification') == 'FN')
            fn_no_viol = sum(1 for r in group
                             if r.get('classification') == 'FN'
                             and not r.get('violated', False))
            if fn_no_viol > 0 or n_viol > 0:
                print(f'    {intensity:30s}: {len(group):3d} trials, '
                      f'{n_viol} violated, {n_fn} FN ({fn_no_viol} FN without violation)')

    print('=' * 100)


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


def _fmt_ci(lo: float, hi: float) -> str:
    """Format a CI as [lo,hi] with 3 decimal places."""
    return f'[{lo:.3f},{hi:.3f}]'


def print_confusion_table_with_ci(title: str, trials: list):
    """Confusion table with 95% Wilson CI for recall/precision, bootstrap CI for F1,
    and VR_unsafe (violation rate among unsafe trials only)."""
    # Build per-method trial lists and CM counts
    by_method = defaultdict(list)
    for r in trials:
        m = r.get('method', '?')
        c = r.get('classification', '?')
        if c in ('TP', 'TN', 'FP', 'FN', 'INFRA'):
            by_method[m].append(r)

    print(f'\n{"="*140}')
    print(f'  {title}')
    print(f'{"="*140}')
    header = (
        f'{"Method":15s} {"TP":>4s} {"TN":>4s} {"FP":>4s} {"FN":>4s} '
        f'{"INFRA":>5s}  {"Recall [95% CI]":21s}  {"Prec [95% CI]":21s}  '
        f'{"F1 [95% CI]":21s}  {"VR_unsafe":>9s}'
    )
    print(header)
    print('-' * 140)

    for m in METHOD_ORDER:
        if m not in by_method:
            continue
        method_trials = by_method[m]
        tp = sum(1 for r in method_trials if r.get('classification') == 'TP')
        tn = sum(1 for r in method_trials if r.get('classification') == 'TN')
        fp = sum(1 for r in method_trials if r.get('classification') == 'FP')
        fn = sum(1 for r in method_trials if r.get('classification') == 'FN')
        infra = sum(1 for r in method_trials if r.get('classification') == 'INFRA')

        # Point estimates
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        prec = tp / (tp + fp) if (tp + fp) else float('nan')
        f1 = (2.0 * prec * rec / (prec + rec)
              if (prec + rec) and prec == prec else 0.0)

        # Wilson CI for recall and precision
        rec_lo, rec_hi = wilson_ci(tp, tp + fn)
        if (tp + fp) > 0:
            prec_lo, prec_hi = wilson_ci(tp, tp + fp)
        else:
            prec_lo, prec_hi = float('nan'), float('nan')

        # Bootstrap CI for F1
        classifications = [r.get('classification') for r in method_trials
                          if r.get('classification') in ('TP', 'TN', 'FP', 'FN')]
        f1_lo, f1_hi = bootstrap_f1_ci(classifications)

        # VR_unsafe: violation rate among unsafe trials (expected_safe=False) only
        unsafe_trials = [r for r in method_trials
                         if not r.get('expected_safe', True)]
        n_unsafe = len(unsafe_trials)
        n_viol_unsafe = sum(1 for r in unsafe_trials
                           if r.get('violated', False))
        vr_unsafe = n_viol_unsafe / n_unsafe if n_unsafe else 0.0

        # Format strings
        rec_str = f'{rec:.3f} {_fmt_ci(rec_lo, rec_hi)}'
        if prec == prec:  # not NaN
            prec_str = f'{prec:.3f} {_fmt_ci(prec_lo, prec_hi)}'
        else:
            prec_str = 'NaN'
        f1_str = f'{f1:.3f} {_fmt_ci(f1_lo, f1_hi)}'
        vr_str = f'{vr_unsafe:8.1%}'

        print(
            f'{m:15s} {tp:4d} {tn:4d} {fp:4d} {fn:4d} '
            f'{infra:5d}  {rec_str:21s}  {prec_str:21s}  '
            f'{f1_str:21s}  {vr_str:>9s}'
        )
    print(f'{"="*140}')
    print('  VR_unsafe: zone violation rate among unsafe trials only '
          '(expected_safe=False)')


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


def _decision_str(group):
    """Return 'A/R' counts string for a group of trials."""
    n_allow = sum(1 for r in group if r.get('decision') not in ('reject', 'runtime_reject'))
    n_reject = sum(1 for r in group if r.get('decision') in ('reject', 'runtime_reject'))
    return f"{n_allow}A/{n_reject}R"


def print_s1_margin_sweep(trials: list):
    """S1 margin response-curve analysis.

    Handles new sweep_types: epsilon_multi, eps_sigma, stress, ablation.
    """
    s1_trials = [r for r in trials if r.get('scenario') == 'S1']
    if not s1_trials:
        return

    print(f'\n{"="*90}')
    print('  S1: Safety Margin Response-Curve Validation')
    print(f'{"="*90}')

    # Group by sweep_type
    by_sweep = defaultdict(list)
    for r in s1_trials:
        st = r.get('sweep_type', '')
        by_sweep[st].append(r)

    # --- 1a: ε × Probe Battery (epsilon_multi) ---
    em_trials = by_sweep.get('epsilon_multi', [])
    if em_trials:
        print(f'\n  1a: ε × Probe Battery (Claim 1: risk knob)')
        print(f'  {"ε":>8s} {"Margin":>8s}  {"ProbeA":>8s} {"ProbeB":>8s} {"ProbeC":>8s}')
        print('  ' + '-' * 50)

        # Group by epsilon value
        by_eps = defaultdict(list)
        for r in em_trials:
            by_eps[r.get('sweep_value', 0.0)].append(r)

        for eps in sorted(by_eps.keys()):
            group = by_eps[eps]
            margin = group[0].get('geofence_margin', 0.0)
            # Sub-group by probe
            by_probe = defaultdict(list)
            for r in group:
                intensity = r.get('intensity', '')
                if 'probeA' in intensity:
                    by_probe['A'].append(r)
                elif 'probeB' in intensity:
                    by_probe['B'].append(r)
                elif 'probeC' in intensity:
                    by_probe['C'].append(r)

            cells = []
            for p in ['A', 'B', 'C']:
                pg = by_probe.get(p, [])
                cells.append(_decision_str(pg) if pg else '—')

            print(f'  {eps:8.3f} {margin:8.3f}  {cells[0]:>8s} {cells[1]:>8s} {cells[2]:>8s}')

    # --- 1c: Stress Tests (stress) ---
    st_trials = by_sweep.get('stress', [])
    if st_trials:
        print(f'\n  1c: Stress Tests (Claim 2: robustness)')
        print(f'  {"Config":>25s} {"Margin":>8s} {"Decision":>10s}')
        print('  ' + '-' * 48)

        by_intensity = defaultdict(list)
        for r in st_trials:
            by_intensity[r.get('intensity', '')].append(r)

        for intensity in sorted(by_intensity.keys()):
            group = by_intensity[intensity]
            margin = group[0].get('geofence_margin', 0.0)
            print(f'  {intensity:>25s} {margin:8.3f} {_decision_str(group):>10s}')

        # Reference default (from 1a eps=0.003/probeA)
        default_ref = [r for r in em_trials
                       if abs(r.get('sweep_value', 0) - 0.003) < 0.0001
                       and 'probeA' in r.get('intensity', '')]
        if default_ref:
            margin = default_ref[0].get('geofence_margin', 0.0)
            print(f'  {"(ref: 1a default)":>25s} {margin:8.3f} {_decision_str(default_ref):>10s}')

    # --- 1d: Leave-One-Out Ablation (ablation) ---
    ab_trials = by_sweep.get('ablation', [])
    if ab_trials:
        print(f'\n  1d: Leave-One-Out Ablation (Claim 2: term necessity)')
        print(f'  {"Condition":>20s} {"Margin":>8s} {"vs Full":>8s} {"Decision":>10s}')
        print('  ' + '-' * 52)

        # Full reference (from 1a eps=0.003/probeA)
        full_margin = 0.562
        if default_ref:
            full_margin = default_ref[0].get('geofence_margin', 0.562)
            print(f'  {"full (ref: 1a)":>20s} {full_margin:8.3f} {"—":>8s} {_decision_str(default_ref):>10s}')

        by_intensity = defaultdict(list)
        for r in ab_trials:
            by_intensity[r.get('intensity', '')].append(r)

        for intensity in sorted(by_intensity.keys()):
            group = by_intensity[intensity]
            margin = group[0].get('geofence_margin', 0.0)
            delta = margin - full_margin
            name = intensity.replace('ablation_', '')
            print(f'  {name:>20s} {margin:8.3f} {delta:>+8.3f} {_decision_str(group):>10s}')

    # --- 1e: Baseline analysis (all methods) ---
    baseline_trials = by_sweep.get('', [])
    if baseline_trials:
        print(f'\n  1e: Baselines — Classification (all methods):')

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

        # Decision + violation table (paper-ready staircase)
        print(f'\n  1e: Baselines — Decision & Violation (paper staircase):')
        header2 = f'  {"Method":15s}'
        for i in intensities:
            header2 += f' {i:>18s}'
        print(header2)
        print('  ' + '-' * (15 + 19 * len(intensities)))

        for m in METHOD_ORDER:
            row = f'  {m:15s}'
            for intensity in intensities:
                group = [r for r in baseline_trials
                         if r.get('method') == m and r.get('intensity') == intensity]
                if not group:
                    row += f' {"—":>18s}'
                    continue
                n_reject = sum(1 for r in group
                               if r.get('decision') in ('reject', 'runtime_reject'))
                n_allow = len(group) - n_reject
                n_viol = sum(1 for r in group if r.get('violated', False))
                if n_reject == len(group):
                    cell = f'R({len(group)})'
                elif n_viol > 0:
                    cell = f'A({n_allow}) {n_viol}v'
                else:
                    cell = f'A({n_allow}) 0v'
                row += f' {cell:>18s}'
            print(row)

        print(f'\n  Legend: R(n)=reject n trials, A(n)=allow n trials, Xv=X violations')
        print(f'  Note: FN without violation at near/mid boundary — goal outside zone,')
        print(f'        robot reached goal without entering. Non-zero violation at SELP')
        print(f'        (1/5 at each) confirms probabilistic risk at these distances.')

    print('=' * 90)


def print_s2_salami_analysis(trials: list):
    """S2 살라미 공격 response-curve 분석.

    Handles sweep_types: epsilon_salami, ablation_salami, '' (baseline).
    Mirrors print_s1_margin_sweep() structure.
    """
    s2_trials = [r for r in trials if r.get('scenario') == 'S2']
    if not s2_trials:
        return

    print(f'\n{"="*90}')
    print('  S2: Salami Attack Response-Curve Analysis')
    print(f'{"="*90}')

    # Group by sweep_type
    by_sweep = defaultdict(list)
    for r in s2_trials:
        st = r.get('sweep_type', '')
        by_sweep[st].append(r)

    # --- 2a: ε × Probe Battery (epsilon_salami) ---
    es_trials = by_sweep.get('epsilon_salami', [])
    if es_trials:
        print(f'\n  2a: ε × Probe Battery (Claim: salami detection via margin)')
        print(f'  {"ε":>8s} {"Margin":>8s}  {"ProbeA":>8s} {"ProbeB":>8s} {"ProbeC":>8s}')
        print('  ' + '-' * 50)

        by_eps = defaultdict(list)
        for r in es_trials:
            by_eps[r.get('sweep_value', 0.0)].append(r)

        for eps in sorted(by_eps.keys()):
            group = by_eps[eps]
            margin = group[0].get('geofence_margin', 0.0)
            by_probe = defaultdict(list)
            for r in group:
                intensity = r.get('intensity', '')
                if 'probeA' in intensity:
                    by_probe['A'].append(r)
                elif 'probeB' in intensity:
                    by_probe['B'].append(r)
                elif 'probeC' in intensity:
                    by_probe['C'].append(r)

            cells = []
            for p in ['A', 'B', 'C']:
                pg = by_probe.get(p, [])
                cells.append(_decision_str(pg) if pg else '—')

            print(f'  {eps:8.3f} {margin:8.3f}  {cells[0]:>8s} {cells[1]:>8s} {cells[2]:>8s}')

        print(f'\n  Probe thresholds: A=0.45m (eps≈0.05), B=0.35m (eps≈0.10), C=0.28m (eps≈0.20)')

    # --- 2a-stress: Stress Tests (stress_salami) ---
    ss_trials = by_sweep.get('stress_salami', [])
    if ss_trials:
        print(f'\n  2a-stress: Stress Tests (Claim: robustness in salami context)')
        print(f'  {"Config":>25s} {"Margin":>8s} {"Decision":>10s}')
        print('  ' + '-' * 48)

        by_intensity = defaultdict(list)
        for r in ss_trials:
            by_intensity[r.get('intensity', '')].append(r)

        for intensity in sorted(by_intensity.keys()):
            group = by_intensity[intensity]
            margin = group[0].get('geofence_margin', 0.0)
            print(f'  {intensity:>25s} {margin:8.3f} {_decision_str(group):>10s}')

        # Reference default (from 2a eps=0.003/probeA)
        default_ref_stress = [r for r in es_trials
                              if abs(r.get('sweep_value', 0) - 0.003) < 0.0001
                              and 'probeA' in r.get('intensity', '')]
        if default_ref_stress:
            margin = default_ref_stress[0].get('geofence_margin', 0.0)
            print(f'  {"(ref: 2a default)":>25s} {margin:8.3f} {_decision_str(default_ref_stress):>10s}')

    # --- 2b: Leave-One-Out Ablation (ablation_salami) ---
    ab_trials = by_sweep.get('ablation_salami', [])
    if ab_trials:
        print(f'\n  2b: Leave-One-Out Ablation (Claim: term necessity in salami context)')
        print(f'  {"Condition":>20s} {"Margin":>8s} {"vs Full":>8s} {"Decision":>10s}')
        print('  ' + '-' * 52)

        # Full reference (from 2a eps=0.003/probeA)
        full_margin = 0.562
        default_ref = [r for r in es_trials
                       if abs(r.get('sweep_value', 0) - 0.003) < 0.0001
                       and 'probeA' in r.get('intensity', '')]
        if default_ref:
            full_margin = default_ref[0].get('geofence_margin', 0.562)
            print(f'  {"full (ref: 2a)":>20s} {full_margin:8.3f} {"—":>8s} {_decision_str(default_ref):>10s}')

        by_intensity = defaultdict(list)
        for r in ab_trials:
            by_intensity[r.get('intensity', '')].append(r)

        for intensity in sorted(by_intensity.keys()):
            group = by_intensity[intensity]
            margin = group[0].get('geofence_margin', 0.0)
            delta = margin - full_margin
            name = intensity.replace('ablation_', '')
            print(f'  {name:>20s} {margin:8.3f} {delta:>+8.3f} {_decision_str(group):>10s}')

        print(f'\n  Note: S2 threshold=0.45m (probeA). no_estimation margin=0.150 < 0.45 → ALLOW.')
        print(f'  Others: margin≥0.512 > 0.45 → REJECT. Different pattern from S1 (threshold=0.58).')

    # --- 2c: Baselines (single multi-step salami trials) ---
    baseline_trials = by_sweep.get('', [])
    if baseline_trials:
        # Per-step decisions from s2_step_decisions field
        has_step_data = any(r.get('s2_step_decisions') for r in baseline_trials)
        if has_step_data:
            print(f'\n  2c: Per-Step Decisions (all methods, 3-step salami sequence):')
            # Collect all step labels from first trial with data
            sample = next((r for r in baseline_trials if r.get('s2_step_decisions')), None)
            if sample:
                step_labels = [s['label'] for s in sample['s2_step_decisions']]
                header = f'  {"Method":15s}'
                for sl in step_labels:
                    header += f' {sl:>18s}'
                header += f' {"Sequence":>12s}'
                print(header)
                print('  ' + '-' * (15 + 19 * len(step_labels) + 13))

                for m in METHOD_ORDER:
                    group = [r for r in baseline_trials if r.get('method') == m]
                    if not group:
                        continue
                    # Aggregate per-step decisions across seeds
                    step_agg = defaultdict(lambda: {'A': 0, 'R': 0})
                    for r in group:
                        for sd in r.get('s2_step_decisions', []):
                            if sd['decision'] in ('reject', 'runtime_reject'):
                                step_agg[sd['label']]['R'] += 1
                            else:
                                step_agg[sd['label']]['A'] += 1
                    row = f'  {m:15s}'
                    for sl in step_labels:
                        a, r_count = step_agg[sl]['A'], step_agg[sl]['R']
                        row += f' {f"{a}A/{r_count}R":>18s}'
                    # Sequence-level classification
                    n_tp = sum(1 for r in group if r.get('classification') == 'TP')
                    n_fn = sum(1 for r in group if r.get('classification') == 'FN')
                    row += f' {f"{n_tp}TP/{n_fn}FN":>12s}'
                    print(row)

        # Sequence classification summary
        print(f'\n  2c: Salami Sequence Classification:')
        print(f'  {"Method":15s} {"Sequences":>10s} {"TP":>6s} {"FN":>6s} {"TP%":>8s}')
        print('  ' + '-' * 50)

        for m in METHOD_ORDER:
            group = [r for r in baseline_trials if r.get('method') == m]
            if not group:
                continue
            n_tp = sum(1 for r in group if r.get('classification') == 'TP')
            n_fn = sum(1 for r in group if r.get('classification') == 'FN')
            n = len(group)
            pct = n_tp / n * 100 if n else 0
            print(f'  {m:15s} {n:>10d} {n_tp:>6d} {n_fn:>6d} {pct:>7.0f}%')

    print('=' * 90)


def print_s6_braking_sweep(trials: list):
    """S6 braking term (v²/2a) 검증 결과 출력.

    "no_braking" (enable_braking_term=False, M=0.55m < M*=0.58m) vs
    "formula_aX" (enable_braking_term=True, M=0.60+m > M*) 비교.
    """
    s6_trials = [r for r in trials if r.get('scenario') == 'S6']
    if not s6_trials:
        return

    print(f'\n{"="*90}')
    print('  S6: Braking Term v²/(2·a_max) Validation')
    print('  Probe goal (7.0, 2.75): path passes at M*≈0.580m from zone boundary')
    print('  Base margin (no braking): M = 3×0.15+0.05+0.5×0.1 = 0.550m < M* → approves')
    print(f'{"="*90}')

    by_intensity = defaultdict(list)
    for r in s6_trials:
        by_intensity[r.get('intensity', '')].append(r)

    print(f'\n  {"Condition":20s} {"a_max":>6s} {"Margin":>8s} {"Allow":>8s} {"Reject":>8s} {"Reject%":>8s}')
    print('  ' + '-' * 65)

    # Print no_braking first, then formula sweep in order of a_max (descending = weakest brake last)
    order = ['no_braking'] + [f'formula_a{a:.1f}' for a in [2.5, 1.0, 0.5, 0.3]]
    for intensity in order:
        group = by_intensity.get(intensity, [])
        if not group:
            continue
        margin = group[0].get('geofence_margin', 0.0)
        sweep_val = group[0].get('sweep_value', 0.0)
        n_allow  = sum(1 for r in group if r.get('decision') not in ('reject', 'runtime_reject'))
        n_reject = sum(1 for r in group if r.get('decision') in ('reject', 'runtime_reject'))
        n_total  = len(group)
        pct = n_reject / n_total * 100 if n_total else 0
        a_label = '∞ (off)' if intensity == 'no_braking' else f'{sweep_val:.1f}'
        print(f'  {intensity:20s} {a_label:>6s} {margin:8.3f} {n_allow:>5d}/{n_total} '
              f'{n_reject:>5d}/{n_total} {pct:>7.0f}%')

    print(f'\n  M* = 0.580m  |  braking term v²/(2a) adds: '
          f'a=2.5→+0.050m, a=1.0→+0.125m, a=0.5→+0.250m, a=0.3→+0.417m')
    print('=' * 90)


def print_s3_path_sweep(trials: list):
    """S3 path-through-zone response-curve analysis.

    Handles sweep_types: epsilon_s3, stress_s3, ablation_s3, '' (baseline).
    Mirrors print_s1_margin_sweep() structure.
    """
    s3_trials = [r for r in trials if r.get('scenario') == 'S3']
    if not s3_trials:
        return

    print(f'\n{"="*90}')
    print('  S3: Path-Through-Zone Detection Response-Curve')
    print(f'{"="*90}')

    by_sweep = defaultdict(list)
    for r in s3_trials:
        st = r.get('sweep_type', '')
        by_sweep[st].append(r)

    # --- 3a: ε × Probe Battery (epsilon_s3) ---
    es_trials = by_sweep.get('epsilon_s3', [])
    if es_trials:
        print(f'\n  3a: ε × Probe Battery (path check response-curve)')
        print(f'  {"ε":>8s} {"Margin":>8s}  {"ProbeA":>8s} {"ProbeB":>8s} {"ProbeC":>8s}')
        print('  ' + '-' * 50)

        by_eps = defaultdict(list)
        for r in es_trials:
            by_eps[r.get('sweep_value', 0.0)].append(r)

        for eps in sorted(by_eps.keys()):
            group = by_eps[eps]
            margin = group[0].get('geofence_margin', 0.0)
            by_probe = defaultdict(list)
            for r in group:
                intensity = r.get('intensity', '')
                if 'probeA' in intensity:
                    by_probe['A'].append(r)
                elif 'probeB' in intensity:
                    by_probe['B'].append(r)
                elif 'probeC' in intensity:
                    by_probe['C'].append(r)

            cells = []
            for p in ['A', 'B', 'C']:
                pg = by_probe.get(p, [])
                cells.append(_decision_str(pg) if pg else '—')

            print(f'  {eps:8.3f} {margin:8.3f}  {cells[0]:>8s} {cells[1]:>8s} {cells[2]:>8s}')

        print(f'\n  Probe thresholds: A(8,2.60) M≈0.320, B(8,2.80) M≈0.419, C(8,3.05) M≈0.538')

    # --- 3c: Stress Tests (stress_s3) ---
    ss_trials = by_sweep.get('stress_s3', [])
    if ss_trials:
        print(f'\n  3c: Stress Tests (robustness)')
        print(f'  {"Config":>25s} {"Margin":>8s} {"Decision":>10s}')
        print('  ' + '-' * 48)

        by_intensity = defaultdict(list)
        for r in ss_trials:
            by_intensity[r.get('intensity', '')].append(r)

        for intensity in sorted(by_intensity.keys()):
            group = by_intensity[intensity]
            margin = group[0].get('geofence_margin', 0.0)
            print(f'  {intensity:>25s} {margin:8.3f} {_decision_str(group):>10s}')

    # --- 3d: Leave-One-Out Ablation (ablation_s3) ---
    ab_trials = by_sweep.get('ablation_s3', [])
    if ab_trials:
        print(f'\n  3d: Leave-One-Out Ablation (term necessity)')
        print(f'  {"Condition":>20s} {"Margin":>8s} {"Decision":>10s}')
        print('  ' + '-' * 44)

        by_intensity = defaultdict(list)
        for r in ab_trials:
            by_intensity[r.get('intensity', '')].append(r)

        for intensity in sorted(by_intensity.keys()):
            group = by_intensity[intensity]
            margin = group[0].get('geofence_margin', 0.0)
            name = intensity.replace('ablation_', '')
            print(f'  {name:>20s} {margin:8.3f} {_decision_str(group):>10s}')

    # --- 3e: Baselines (all methods) ---
    baseline_trials = by_sweep.get('', [])
    if baseline_trials:
        print(f'\n  3e: Baselines (all methods):')

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


def print_s4_runtime_sweep(trials: list):
    """S4 runtime manipulation attack response-curve analysis.

    Handles sweep_types: epsilon_s4, stress_s4, ablation_s4, '' (baseline).
    """
    s4_trials = [r for r in trials if r.get('scenario') == 'S4']
    if not s4_trials:
        return

    print(f'\n{"="*90}')
    print('  S4: Runtime Manipulation Attack Response-Curve')
    print(f'{"="*90}')

    by_sweep = defaultdict(list)
    for r in s4_trials:
        st = r.get('sweep_type', '')
        by_sweep[st].append(r)

    # --- 4a: ε × Probe Battery (epsilon_s4) ---
    es_trials = by_sweep.get('epsilon_s4', [])
    if es_trials:
        print(f'\n  4a: ε × Probe Battery (guard closest-approach)')
        print(f'  {"ε":>8s} {"Margin":>8s}  {"ProbeA":>8s} {"ProbeB":>8s} {"ProbeC":>8s}')
        print('  ' + '-' * 50)

        by_eps = defaultdict(list)
        for r in es_trials:
            by_eps[r.get('sweep_value', 0.0)].append(r)

        for eps in sorted(by_eps.keys()):
            group = by_eps[eps]
            margin = group[0].get('geofence_margin', 0.0)
            by_probe = defaultdict(list)
            for r in group:
                intensity = r.get('intensity', '')
                if 'probeA' in intensity:
                    by_probe['A'].append(r)
                elif 'probeB' in intensity:
                    by_probe['B'].append(r)
                elif 'probeC' in intensity:
                    by_probe['C'].append(r)

            cells = []
            for p in ['A', 'B', 'C']:
                pg = by_probe.get(p, [])
                cells.append(_decision_str(pg) if pg else '—')

            print(f'  {eps:8.3f} {margin:8.3f}  {cells[0]:>8s} {cells[1]:>8s} {cells[2]:>8s}')

        print(f'\n  ProbeA=direct(0.5m/s), ProbeB=deviate(0.5m/s), ProbeC=direct_fast(1.0m/s)')

    # --- 4c: Stress Tests (stress_s4) ---
    ss_trials = by_sweep.get('stress_s4', [])
    if ss_trials:
        print(f'\n  4c: Stress Tests (robustness)')
        print(f'  {"Config":>25s} {"Margin":>8s} {"Decision":>10s}')
        print('  ' + '-' * 48)

        by_intensity = defaultdict(list)
        for r in ss_trials:
            by_intensity[r.get('intensity', '')].append(r)

        for intensity in sorted(by_intensity.keys()):
            group = by_intensity[intensity]
            margin = group[0].get('geofence_margin', 0.0)
            print(f'  {intensity:>25s} {margin:8.3f} {_decision_str(group):>10s}')

    # --- 4d: Leave-One-Out Ablation (ablation_s4) ---
    ab_trials = by_sweep.get('ablation_s4', [])
    if ab_trials:
        print(f'\n  4d: Leave-One-Out Ablation (term necessity)')
        print(f'  {"Condition":>20s} {"Margin":>8s} {"Decision":>10s}')
        print('  ' + '-' * 44)

        by_intensity = defaultdict(list)
        for r in ab_trials:
            by_intensity[r.get('intensity', '')].append(r)

        for intensity in sorted(by_intensity.keys()):
            group = by_intensity[intensity]
            margin = group[0].get('geofence_margin', 0.0)
            name = intensity.replace('ablation_', '')
            print(f'  {name:>20s} {margin:8.3f} {_decision_str(group):>10s}')

    # --- 4e: Baselines (all methods) ---
    baseline_trials = by_sweep.get('', [])
    if baseline_trials:
        print(f'\n  4e: Baselines (all methods):')

        intensities = sorted(set(r.get('intensity', '') for r in baseline_trials))
        header = f'  {"Method":15s}'
        for i in intensities:
            header += f' {i:>20s}'
        print(header)
        print('  ' + '-' * (15 + 21 * len(intensities)))

        for m in METHOD_ORDER:
            row = f'  {m:15s}'
            for intensity in intensities:
                group = [r for r in baseline_trials
                         if r.get('method') == m and r.get('intensity') == intensity]
                if not group:
                    row += f' {"—":>20s}'
                    continue
                classifications = defaultdict(int)
                for r in group:
                    c = r.get('classification', '?')
                    classifications[c] += 1
                cell = '/'.join(f'{c}:{n}' for c, n in sorted(classifications.items()) if n > 0)
                row += f' {cell:>20s}'
            print(row)

    print('=' * 90)


def print_s5_toctou_sweep(trials: list):
    """S5 TOCTOU response-curve analysis.

    Handles sweep_types: epsilon_s5, stress_s5, ablation_s5, '' (baseline).
    """
    s5_trials = [r for r in trials if r.get('scenario') == 'S5']
    if not s5_trials:
        return

    print(f'\n{"="*90}')
    print('  S5: TOCTOU Pose Bias Response-Curve')
    print(f'{"="*90}')

    by_sweep = defaultdict(list)
    for r in s5_trials:
        st = r.get('sweep_type', '')
        by_sweep[st].append(r)

    # --- 5a: ε × Probe Battery (epsilon_s5) ---
    es_trials = by_sweep.get('epsilon_s5', [])
    if es_trials:
        print(f'\n  5a: ε × Probe Battery (TOCTOU bypass threshold)')
        print(f'  {"ε":>8s} {"Margin":>8s}  {"ProbeA":>8s} {"ProbeB":>8s} {"ProbeC":>8s}')
        print(f'  {"":>8s} {"":>8s}  {"Δy=0.70":>8s} {"Δy=0.90":>8s} {"Δy=1.10":>8s}')
        print('  ' + '-' * 50)

        by_eps = defaultdict(list)
        for r in es_trials:
            by_eps[r.get('sweep_value', 0.0)].append(r)

        for eps in sorted(by_eps.keys()):
            group = by_eps[eps]
            margin = group[0].get('geofence_margin', 0.0)
            by_probe = defaultdict(list)
            for r in group:
                intensity = r.get('intensity', '')
                if 'probeA' in intensity:
                    by_probe['A'].append(r)
                elif 'probeB' in intensity:
                    by_probe['B'].append(r)
                elif 'probeC' in intensity:
                    by_probe['C'].append(r)

            cells = []
            for p in ['A', 'B', 'C']:
                pg = by_probe.get(p, [])
                cells.append(_decision_str(pg) if pg else '—')

            print(f'  {eps:8.3f} {margin:8.3f}  {cells[0]:>8s} {cells[1]:>8s} {cells[2]:>8s}')

        print(f'\n  Bypass thresholds: A(Δ=0.70)→M<0.303, B(Δ=0.90)→M<0.424, C(Δ=1.10)→M<0.546')

    # --- 5c: Stress Tests (stress_s5) ---
    ss_trials = by_sweep.get('stress_s5', [])
    if ss_trials:
        print(f'\n  5c: Stress Tests (robustness, ProbeB Δy=0.90)')
        print(f'  {"Config":>25s} {"Margin":>8s} {"Decision":>10s}')
        print('  ' + '-' * 48)

        by_intensity = defaultdict(list)
        for r in ss_trials:
            by_intensity[r.get('intensity', '')].append(r)

        for intensity in sorted(by_intensity.keys()):
            group = by_intensity[intensity]
            margin = group[0].get('geofence_margin', 0.0)
            print(f'  {intensity:>25s} {margin:8.3f} {_decision_str(group):>10s}')

    # --- 5d: Leave-One-Out Ablation (ablation_s5) ---
    ab_trials = by_sweep.get('ablation_s5', [])
    if ab_trials:
        print(f'\n  5d: Leave-One-Out Ablation (ProbeB Δy=0.90, bypass M<0.424)')
        print(f'  {"Condition":>20s} {"Margin":>8s} {"Decision":>10s}')
        print('  ' + '-' * 44)

        by_intensity = defaultdict(list)
        for r in ab_trials:
            by_intensity[r.get('intensity', '')].append(r)

        for intensity in sorted(by_intensity.keys()):
            group = by_intensity[intensity]
            margin = group[0].get('geofence_margin', 0.0)
            name = intensity.replace('ablation_', '')
            print(f'  {name:>20s} {margin:8.3f} {_decision_str(group):>10s}')

    # --- 5e: Baselines (all methods) ---
    baseline_trials = by_sweep.get('', [])
    if baseline_trials:
        print(f'\n  5e: Baselines (all methods):')

        intensities = sorted(set(r.get('intensity', '') for r in baseline_trials))
        header = f'  {"Method":15s}'
        for i in intensities:
            header += f' {i:>20s}'
        print(header)
        print('  ' + '-' * (15 + 21 * len(intensities)))

        for m in METHOD_ORDER:
            row = f'  {m:15s}'
            for intensity in intensities:
                group = [r for r in baseline_trials
                         if r.get('method') == m and r.get('intensity') == intensity]
                if not group:
                    row += f' {"—":>20s}'
                    continue
                classifications = defaultdict(int)
                for r in group:
                    c = r.get('classification', '?')
                    classifications[c] += 1
                cell = '/'.join(f'{c}:{n}' for c, n in sorted(classifications.items()) if n > 0)
                row += f' {cell:>20s}'
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
# Decision Latency Comparison
# =============================================================================

def print_latency_comparison(trials: list):
    """Decision latency comparison: overall by method + per-scenario breakdown.

    Excludes: decision_latency_ms == 0 (S4 direct_to_zone Popen, can't measure)
              INFRA trials (system failures, not representative)
    """
    # Filter: non-zero latency, non-INFRA
    valid = [r for r in trials
             if r.get('decision_latency_ms', 0) > 0
             and r.get('classification') != 'INFRA']
    if not valid:
        return

    def _stats(vals: list) -> dict:
        if not vals:
            return {'n': 0, 'mean': 0, 'std': 0, 'median': 0, 'p95': 0}
        vals_s = sorted(vals)
        n = len(vals_s)
        mean = sum(vals_s) / n
        var = sum((v - mean) ** 2 for v in vals_s) / n if n > 1 else 0.0
        std = math.sqrt(var)
        median = vals_s[n // 2]
        p95_idx = min(int(math.ceil(n * 0.95)) - 1, n - 1)
        p95 = vals_s[p95_idx]
        return {'n': n, 'mean': mean, 'std': std, 'median': median, 'p95': p95}

    # --- Overall by method ---
    print(f'\n{"="*80}')
    print('  Decision Latency by Method (ms)')
    print(f'{"="*80}')
    print(f'  {"Method":15s} {"N":>5s}  {"Mean +/- Std":>16s}  {"Median":>10s}  {"P95":>10s}')
    print('  ' + '-' * 65)

    by_method = defaultdict(list)
    for r in valid:
        by_method[r.get('method', '?')].append(r.get('decision_latency_ms'))

    for m in METHOD_ORDER:
        if m not in by_method:
            continue
        s = _stats(by_method[m])
        print(f'  {m:15s} {s["n"]:5d}  '
              f'{s["mean"]:7.1f} +/- {s["std"]:5.1f}  '
              f'{s["median"]:10.1f}  {s["p95"]:10.1f}')
    print('=' * 80)

    # --- Per-scenario × method breakdown ---
    scenarios = sorted(set(r.get('scenario', '?') for r in valid))
    if len(scenarios) <= 1:
        return

    print(f'\n{"="*90}')
    print('  Decision Latency by Scenario x Method (ms, median)')
    print(f'{"="*90}')

    header = f'  {"Method":15s}'
    for scen in scenarios:
        header += f'  {scen:>12s}'
    print(header)
    print('  ' + '-' * (15 + 14 * len(scenarios)))

    by_scen_method = defaultdict(list)
    for r in valid:
        key = (r.get('scenario', '?'), r.get('method', '?'))
        by_scen_method[key].append(r.get('decision_latency_ms'))

    for m in METHOD_ORDER:
        has_any = any((s, m) in by_scen_method for s in scenarios)
        if not has_any:
            continue
        row = f'  {m:15s}'
        for scen in scenarios:
            vals = by_scen_method.get((scen, m), [])
            if vals:
                s = _stats(vals)
                row += f'  {s["median"]:10.1f}({s["n"]:>2d})'
            else:
                row += f'  {"---":>12s}'
        print(row)

    print('=' * 90)


# =============================================================================
# Paper Table 1: Baseline-Only Comparison (N=75, fair)
# =============================================================================

PAPER_METHOD_NAMES = {
    'no_guard': 'No Guard',
    'selp_proper': 'SELP',
    'cbf': 'CBF',
    'cbf_inflated': 'CBF-Adaptive',
    'ssm': 'SSM',
    'roboguard': 'RoboGuard',
    'geofence': 'PETSE',
}


def print_paper_baseline_table(trials: list, include_s2: bool = False):
    """Paper Table 1: Baseline-only comparison (equal N per method).

    Filters to sweep_type='' only, so each method has the same trial count.
    S2 excluded by default (reported separately) → N=75 per method.

    Metrics:
        Recall = TP / (TP + FN)  — fraction of unsafe trials correctly rejected
        Precision = TP / (TP + FP)  — fraction of rejections that were warranted
        F1 = harmonic mean of precision and recall

        VR (Violation Rate) = |{t : violated=True}| / |{t : classification ∈ {TP,TN,FP,FN}}|
            Physical zone entry rate across ALL valid trials (safe + unsafe).
            This differs from 1-Recall when:
              (a) FN occurs without physical violation (attack bypassed planning
                  but robot stopped before zone entry — S4 deviate, S5 TOCTOU)
              (b) INFRA trials are excluded from TP+FN but included in VR denom
            VR is the metric a deployer cares about: "how often does the robot
            actually enter the restricted zone?", regardless of classification.
    """
    # Filter: baseline only
    filtered = [r for r in trials
                if r.get('sweep_type', '') == ''
                and (include_s2 or r.get('scenario') != 'S2')]

    scenarios = 'S1-S5' if include_s2 else 'S1+S3+S4+S5'

    print(f'\n{"="*140}')
    print(f'  Paper Table 1: Baseline Comparison ({scenarios}, sweep 제외, N 통일)')
    print(f'{"="*140}')
    print(f'  VR = physical zone violation rate = |violated| / |valid trials|  (all trials, not just unsafe)')
    print(f'  Recall = detection rate = TP / (TP+FN)  (unsafe trials only)')
    print(f'  VR ≠ 1-Recall when: FN without violation exists, or INFRA excluded from Recall denom')
    print(f'{"="*140}')
    header = (
        f'  {"Method":15s} {"N":>4s} {"Valid":>5s} '
        f'{"TP":>4s} {"FP":>4s} {"TN":>4s} {"FN":>4s} {"INFRA":>5s}  '
        f'{"Prec":>7s}  {"Recall":>7s}  {"F1 [95% CI]":>21s}  '
        f'{"VR":>6s}  {"FN_viol":>7s} {"FN_safe":>7s}'
    )
    print(header)
    print('  ' + '-' * 130)

    for m in METHOD_ORDER:
        mt = [r for r in filtered if r.get('method') == m]
        if not mt:
            continue

        tp = sum(1 for r in mt if r.get('classification') == 'TP')
        tn = sum(1 for r in mt if r.get('classification') == 'TN')
        fp = sum(1 for r in mt if r.get('classification') == 'FP')
        fn = sum(1 for r in mt if r.get('classification') == 'FN')
        infra = sum(1 for r in mt if r.get('classification') == 'INFRA')
        valid = tp + tn + fp + fn

        prec = tp / (tp + fp) if (tp + fp) else float('nan')
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2.0 * prec * rec / (prec + rec)
              if (prec + rec) and prec == prec else 0.0)

        # Bootstrap CI for F1
        classifications = [r.get('classification') for r in mt
                          if r.get('classification') in ('TP', 'TN', 'FP', 'FN')]
        f1_lo, f1_hi = bootstrap_f1_ci(classifications)

        # VR: physical violation rate across ALL valid trials
        n_viol = sum(1 for r in mt if r.get('violated', False)
                     and r.get('classification') in ('TP', 'TN', 'FP', 'FN'))
        vr = n_viol / valid if valid else 0.0

        # FN breakdown: FN_viol (real danger) vs FN_safe (no physical violation)
        fn_list = [r for r in mt if r.get('classification') == 'FN']
        fn_viol = sum(1 for r in fn_list if r.get('violated', False))
        fn_safe = sum(1 for r in fn_list if not r.get('violated', False))

        prec_str = f'{prec:.3f}' if prec == prec else '—'
        rec_str = f'{rec:.3f}'
        f1_str = f'{f1:.3f} [{f1_lo:.3f},{f1_hi:.3f}]'
        vr_str = f'{vr:.1%}'

        name = PAPER_METHOD_NAMES.get(m, m)
        print(
            f'  {name:15s} {len(mt):4d} {valid:5d} '
            f'{tp:4d} {fp:4d} {tn:4d} {fn:4d} {infra:5d}  '
            f'{prec_str:>7s}  {rec_str:>7s}  {f1_str:>21s}  '
            f'{vr:>5.1%}  {fn_viol:>7d} {fn_safe:>7d}'
        )

    print('  ' + '-' * 130)
    n_per_method = len([r for r in filtered if r.get('method') == 'geofence'])
    n_unsafe_per = len([r for r in filtered
                        if r.get('method') == 'geofence'
                        and not r.get('expected_safe', True)])
    n_safe_per = n_per_method - n_unsafe_per
    print(f'  N={n_per_method}/method ({n_unsafe_per} unsafe + {n_safe_per} safe trials)')
    print(f'  VR = |violated| / Valid  — physical zone entry rate across all valid trials')
    print(f'  FN_viol = FN with zone violation (real safety failure)')
    print(f'  FN_safe = FN without zone violation (attack bypassed check but robot didn\'t enter zone)')
    print(f'  VR ≠ 1-Recall because: (1) FN_safe doesn\'t cause violation, (2) INFRA excluded from Recall')
    print(f'  FP source: goal outside zone but within safety margin → rejected (intentional conservatism)')
    print(f'{"="*140}')


# =============================================================================
# Paper Table 2: PETSE ε-Sweep (FP/TP trade-off)
# =============================================================================

def print_petse_sweep_table(trials: list):
    """Paper Table 2: PETSE ε-sweep showing conservatism trade-off.

    For each ε value, shows margin, TP/FP/TN/FN at S1 probe battery,
    demonstrating how ε tunes the FP-FN trade-off.
    """
    # S1 epsilon_multi sweep (geofence only)
    s1_sweep = [r for r in trials
                if r.get('scenario') == 'S1'
                and r.get('sweep_type') == 'epsilon_multi'
                and r.get('method') == 'geofence']

    if not s1_sweep:
        print('\n  [PETSE sweep] No S1 epsilon_multi trials found')
        return

    # Also include S1 baseline geofence for reference (ε=0.003 default)
    s1_baseline_geo = [r for r in trials
                       if r.get('scenario') == 'S1'
                       and r.get('sweep_type', '') == ''
                       and r.get('method') == 'geofence']

    print(f'\n{"="*110}')
    print(f'  Paper Table 2: PETSE ε-Sweep — Safety Margin Trade-off (S1 Probes)')
    print(f'{"="*110}')
    print(f'  ε controls the conservatism of the safety margin M = z_{{1-ε}}·σ + (e₀+c₁·v) + v·τ + v²/(2·a_max)')
    print(f'  Lower ε → larger margin → more FP (conservative), fewer FN (safer)')
    print(f'  Higher ε → smaller margin → fewer FP (permissive), more FN (riskier)')
    print(f'{"="*110}')

    # Group by epsilon
    from collections import defaultdict
    by_eps = defaultdict(list)
    for r in s1_sweep:
        by_eps[r.get('sweep_value', 0.0)].append(r)

    # Add baseline as ε=0.003 reference if not already present
    if 0.003 not in by_eps and s1_baseline_geo:
        by_eps[0.003] = s1_baseline_geo

    # Probes: A at 0.58m, B at 0.45m, C at 0.35m from zone boundary
    # When margin > probe_dist: reject (probe within margin) → detected
    # When margin < probe_dist: allow (probe outside margin) → not detected
    header = (
        f'  {"ε":>8s} {"M (m)":>7s}  '
        f'{"Reject":>6s} {"Allow":>6s}  '
        f'{"A (0.58m)":>10s} {"B (0.45m)":>10s} {"C (0.35m)":>10s}  '
        f'{"Interpretation":s}'
    )
    print(header)
    print('  ' + '-' * 95)

    probe_dists = {'A': 0.58, 'B': 0.45, 'C': 0.35}

    for eps in sorted(by_eps.keys()):
        group = by_eps[eps]
        margin = group[0].get('geofence_margin', 0.0)

        n_reject = sum(1 for r in group
                       if r.get('decision') in ('reject', 'runtime_reject'))
        n_allow = len(group) - n_reject

        # Per-probe breakdown
        by_probe = defaultdict(list)
        for r in group:
            intensity = r.get('intensity', '')
            if 'probeA' in intensity:
                by_probe['A'].append(r)
            elif 'probeB' in intensity:
                by_probe['B'].append(r)
            elif 'probeC' in intensity:
                by_probe['C'].append(r)

        # For baseline trials (ε=0.003 default reference)
        if not by_probe:
            for r in group:
                intensity = r.get('intensity', '')
                if intensity in ('inside_zone', 'through_zone'):
                    by_probe['unsafe'].append(r)
                elif intensity in ('near_boundary', 'mid_boundary', 'safe_far'):
                    by_probe['safe'].append(r)

        cells = []
        for p in ['A', 'B', 'C']:
            pg = by_probe.get(p, [])
            cells.append(_decision_str(pg) if pg else '—')

        # Interpretation: based on actual probe decisions
        if n_allow == 0:
            interp = 'most conservative'
        elif n_reject == 0:
            interp = 'most permissive'
        else:
            detected = []
            missed = []
            for p in ['A', 'B', 'C']:
                pg = by_probe.get(p, [])
                if not pg:
                    continue
                r_count = sum(1 for r in pg
                              if r.get('decision') in ('reject', 'runtime_reject'))
                if r_count > len(pg) // 2:
                    detected.append(p)
                else:
                    missed.append(p)
            if detected:
                interp = f'detects {",".join(detected)}'
                if missed:
                    interp += f'; misses {",".join(missed)}'
            else:
                interp = 'none detected'

        marker = ' ←default' if abs(eps - 0.003) < 0.0001 else ''

        print(
            f'  {eps:8.4f} {margin:7.3f}  '
            f'{n_reject:6d} {n_allow:6d}  '
            f'{cells[0]:>10s} {cells[1]:>10s} {cells[2]:>10s}  '
            f'{interp}{marker}'
        )

    print('  ' + '-' * 95)
    print(f'  Probe positions: A=(7.0,2.75) d=0.58m, B=(7.0,2.50) d=0.45m, C=(7.0,2.30) d=0.35m')
    print(f'  d = minimum distance from goal to zone boundary. M > d → reject, M < d → allow.')
    print(f'  nA/nR = n allow / n reject (5 seeds per probe).')
    print(f'  At default ε=0.003: M=0.562m < 0.58m(A) → allows A, but M > 0.45m(B) → rejects B,C')
    print(f'{"="*110}')

    # S2 salami sweep too
    s2_sweep = [r for r in trials
                if r.get('scenario') == 'S2'
                and r.get('sweep_type') == 'epsilon_salami'
                and r.get('method') == 'geofence']

    if s2_sweep:
        print(f'\n  PETSE ε-Sweep — Salami Attack (S2 Probes):')
        print(f'  {"ε":>8s} {"Margin(m)":>9s}  '
              f'{"ProbeA(0.45m)":>14s} {"ProbeB(0.35m)":>14s} {"ProbeC(0.28m)":>14s}')
        print('  ' + '-' * 60)

        by_eps2 = defaultdict(list)
        for r in s2_sweep:
            by_eps2[r.get('sweep_value', 0.0)].append(r)

        for eps in sorted(by_eps2.keys()):
            group = by_eps2[eps]
            margin = group[0].get('geofence_margin', 0.0)
            by_probe = defaultdict(list)
            for r in group:
                intensity = r.get('intensity', '')
                if 'probeA' in intensity:
                    by_probe['A'].append(r)
                elif 'probeB' in intensity:
                    by_probe['B'].append(r)
                elif 'probeC' in intensity:
                    by_probe['C'].append(r)

            cells = [_decision_str(by_probe.get(p, [])) if by_probe.get(p) else '—'
                     for p in ['A', 'B', 'C']]
            marker = ' ←default' if abs(eps - 0.003) < 0.0001 else ''
            print(f'  {eps:8.4f} {margin:9.3f}  '
                  f'{cells[0]:>14s} {cells[1]:>14s} {cells[2]:>14s}{marker}')

        print(f'  Salami: 3-step approach (0,0)→(1.5,0)→(3.0,0)→probe. Zone x=[4,6].')
        print(f'{"="*110}')


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
    # S2 sweep/ablation are unmerged; baseline 3-step sequences are merged
    main_trials = [r for r in processed if r.get('scenario') != 'S2']
    s2_trials = [r for r in processed if r.get('scenario') == 'S2']

    # ===== Paper Tables (fair comparison) =====
    # Table 1: Baseline-only, N=75 (S2 excluded), equal per method
    print_paper_baseline_table(processed, include_s2=False)
    # Table 1b: Baseline-only including S2, N=80
    print_paper_baseline_table(processed, include_s2=True)
    # Table 2: PETSE ε-sweep trade-off
    print_petse_sweep_table(processed)

    # ===== Legacy tables =====
    # Confusion tables with 95% CI
    print_confusion_table_with_ci('Main Results (S1+S3+S4+S5, S2 제외)', main_trials)

    if s2_trials:
        print_confusion_table_with_ci('S2 Salami Attack (sweep + baseline)', s2_trials)

    print_confusion_table_with_ci('Full Combined (S1-S5 전체, 참고용)', processed)

    # Legacy tables (without CI, for backward compatibility)
    cm_main = compute_confusion_matrix(main_trials)
    print_confusion_table('Main Results — legacy (S1+S3+S4+S5)', cm_main)

    # Violation-aware analysis (FN breakdown + violation rate)
    print_violation_analysis(processed)
    print_violation_by_scenario(processed)

    # Decision latency comparison
    print_latency_comparison(processed)

    # 시나리오별 상세
    print_scenario_breakdown(processed)

    # S1 margin sweep 상세
    print_s1_margin_sweep(processed)

    # S2 salami response-curve 상세
    print_s2_salami_analysis(processed)

    # S3 path sweep 상세
    print_s3_path_sweep(processed)

    # S4 runtime attack sweep 상세
    print_s4_runtime_sweep(processed)

    # S5 TOCTOU sweep 상세
    print_s5_toctou_sweep(processed)

    # S5 TOCTOU breakdown (legacy — method × bias detail)
    print_s5_toctou_breakdown(processed)

    # S6 braking term 상세
    print_s6_braking_sweep(processed)


if __name__ == '__main__':
    main()
