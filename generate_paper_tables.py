#!/usr/bin/env python3
"""Generate Table I, II, III for the paper from experiment results."""

import json
from collections import defaultdict

RESULTS_FILE = "/home/jim/ros2_motion_planning_tutorials/experiment_results/gazebo_s1_s6/results.jsonl"

# Method display names and internal keys, will be sorted by F1 ascending later
METHOD_DISPLAY = {
    "no_guard": "No Guard",
    "selp_proper": "SELP",
    "cbf": "CBF",
    "cbf_inflated": "CBF-Adaptive",
    "ssm": "SSM",
    "geofence": "Geofence (Ours)",
}

SCENARIOS = ["S1", "S2", "S3", "S4", "S5"]

def load_and_filter():
    """Load results.jsonl, filter to S1-S5, exclude S1 geofence sweep trials."""
    trials = []
    with open(RESULTS_FILE) as f:
        for line in f:
            t = json.loads(line)
            # Exclude S6
            if t["scenario"] not in SCENARIOS:
                continue
            # Exclude S1 sweep trials (sigma, v_max, tau)
            if t["scenario"] == "S1" and t.get("sweep_type", "") not in ("", None):
                continue
            trials.append(t)
    return trials


def compute_tables(trials):
    """Compute all table data."""
    # Group by method
    by_method = defaultdict(list)
    for t in trials:
        by_method[t["method"]].append(t)

    # Group by (method, scenario)
    by_method_scenario = defaultdict(list)
    for t in trials:
        by_method_scenario[(t["method"], t["scenario"])].append(t)

    # --- Table I: Confusion Matrix ---
    table1 = {}
    for method in METHOD_DISPLAY:
        mt = by_method[method]
        n = len(mt)
        infra = sum(1 for t in mt if t["classification"] == "INFRA")
        valid = n - infra
        tp = sum(1 for t in mt if t["classification"] == "TP")
        fp = sum(1 for t in mt if t["classification"] == "FP")
        tn = sum(1 for t in mt if t["classification"] == "TN")
        fn = sum(1 for t in mt if t["classification"] == "FN")
        table1[method] = {"N": n, "Valid": valid, "Infra": infra,
                          "TP": tp, "FP": fp, "TN": tn, "FN": fn}

    # --- Table II: Detection Performance ---
    table2 = {}
    for method in METHOD_DISPLAY:
        d = table1[method]
        tp, fp, fn = d["TP"], d["FP"], d["FN"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else None
        rec = tp / (tp + fn) if (tp + fn) > 0 else None
        if prec is not None and rec is not None and (prec + rec) > 0:
            f1 = 2 * prec * rec / (prec + rec)
        else:
            f1 = 0.0
        fnr = fn / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
        table2[method] = {"Precision": prec, "Recall": rec, "F1": f1, "FNR": fnr}

    # --- Table III: Per-Scenario VR% ---
    table3 = {}
    for method in METHOD_DISPLAY:
        vr_by_scenario = {}
        total_fn = 0
        total_valid = 0
        for sc in SCENARIOS:
            sc_trials = by_method_scenario[(method, sc)]
            valid_sc = sum(1 for t in sc_trials if t["classification"] != "INFRA")
            fn_sc = sum(1 for t in sc_trials if t["classification"] == "FN")
            vr_by_scenario[sc] = fn_sc / valid_sc * 100 if valid_sc > 0 else 0.0
            total_fn += fn_sc
            total_valid += valid_sc
        overall_vr = total_fn / total_valid * 100 if total_valid > 0 else 0.0
        table3[method] = {"Overall": overall_vr, **vr_by_scenario}

    # --- Raw counts per scenario per method ---
    raw_counts = {}
    for method in METHOD_DISPLAY:
        raw_counts[method] = {}
        for sc in SCENARIOS:
            raw_counts[method][sc] = len(by_method_scenario[(method, sc)])

    return table1, table2, table3, raw_counts


def sort_methods_by_f1(table2):
    """Return method keys sorted by F1 ascending."""
    return sorted(METHOD_DISPLAY.keys(), key=lambda m: table2[m]["F1"])


def fmt_pct(val):
    """Format a percentage or fraction as string."""
    if val is None:
        return "--"
    return f"{val:.3f}"


def fmt_pct_100(val):
    """Format a percentage value."""
    if val is None:
        return "--"
    return f"{val:.1f}\\%"


def print_tables(table1, table2, table3, raw_counts, method_order):
    # ===== RAW COUNTS =====
    print("=" * 80)
    print("RAW TRIAL COUNTS PER SCENARIO PER METHOD")
    print("=" * 80)
    header = f"{'Method':<20}" + "".join(f"{sc:>8}" for sc in SCENARIOS) + f"{'Total':>8}"
    print(header)
    print("-" * len(header))
    for m in method_order:
        row = f"{METHOD_DISPLAY[m]:<20}"
        total = 0
        for sc in SCENARIOS:
            c = raw_counts[m][sc]
            total += c
            row += f"{c:>8}"
        row += f"{total:>8}"
        print(row)
    print()

    # ===== TABLE I: Confusion Matrix =====
    print("=" * 80)
    print("TABLE I: CONFUSION MATRIX")
    print("=" * 80)
    header = f"{'Method':<20}{'N':>6}{'Valid':>7}{'Infra':>7}{'TP':>6}{'FP':>6}{'TN':>6}{'FN':>6}"
    print(header)
    print("-" * len(header))
    for m in method_order:
        d = table1[m]
        print(f"{METHOD_DISPLAY[m]:<20}{d['N']:>6}{d['Valid']:>7}{d['Infra']:>7}"
              f"{d['TP']:>6}{d['FP']:>6}{d['TN']:>6}{d['FN']:>6}")
    print()

    # LaTeX for Table I
    print("--- LaTeX Table I ---")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Confusion matrix across all scenarios (S1--S5).}")
    print(r"\label{tab:confusion}")
    print(r"\begin{tabular}{lrrrrrrr}")
    print(r"\toprule")
    print(r"Method & $N$ & Valid & Infra & TP$\uparrow$ & FP$\downarrow$ & TN$\uparrow$ & FN$\downarrow$ \\")
    print(r"\midrule")
    for m in method_order:
        d = table1[m]
        name = METHOD_DISPLAY[m]
        if m == "geofence":
            name = r"\textbf{" + name + "}"
        print(f"{name} & {d['N']} & {d['Valid']} & {d['Infra']} & "
              f"{d['TP']} & {d['FP']} & {d['TN']} & {d['FN']} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print()

    # ===== TABLE II: Detection Performance =====
    print("=" * 80)
    print("TABLE II: DETECTION PERFORMANCE")
    print("=" * 80)
    header = f"{'Method':<20}{'Precision':>12}{'Recall':>10}{'F1':>10}{'FNR%':>10}"
    print(header)
    print("-" * len(header))
    for m in method_order:
        d = table2[m]
        prec_s = fmt_pct(d["Precision"])
        rec_s = fmt_pct(d["Recall"])
        f1_s = f"{d['F1']:.3f}"
        fnr_s = f"{d['FNR']:.1f}%"
        print(f"{METHOD_DISPLAY[m]:<20}{prec_s:>12}{rec_s:>10}{f1_s:>10}{fnr_s:>10}")
    print()

    # LaTeX for Table II
    print("--- LaTeX Table II ---")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Detection performance across all scenarios (S1--S5).}")
    print(r"\label{tab:detection}")
    print(r"\begin{tabular}{lrrrr}")
    print(r"\toprule")
    print(r"Method & Precision$\uparrow$ & Recall$\uparrow$ & F1$\uparrow$ & FNR$\downarrow$ \\")
    print(r"\midrule")
    for m in method_order:
        d = table2[m]
        prec_s = "--" if d["Precision"] is None else f"{d['Precision']:.3f}"
        rec_s = "--" if d["Recall"] is None else f"{d['Recall']:.3f}"
        f1_s = f"{d['F1']:.3f}"
        fnr_s = f"{d['FNR']:.1f}\\%"
        name = METHOD_DISPLAY[m]
        if m == "geofence":
            name = r"\textbf{" + name + "}"
        print(f"{name} & {prec_s} & {rec_s} & {f1_s} & {fnr_s} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print()

    # ===== TABLE III: Per-Scenario VR% =====
    print("=" * 80)
    print("TABLE III: VIOLATION RATE (VR%) PER SCENARIO")
    print("=" * 80)
    header = f"{'Method':<20}{'Overall':>10}" + "".join(f"{sc:>8}" for sc in SCENARIOS)
    print(header)
    print("-" * len(header))
    for m in method_order:
        d = table3[m]
        row = f"{METHOD_DISPLAY[m]:<20}{d['Overall']:>9.1f}%"
        for sc in SCENARIOS:
            row += f"{d[sc]:>7.1f}%"
        print(row)
    print()

    # LaTeX for Table III
    print("--- LaTeX Table III ---")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Violation rate (VR\%) per scenario. Lower is better.}")
    print(r"\label{tab:vr}")
    print(r"\begin{tabular}{lrrrrrr}")
    print(r"\toprule")
    print(r"Method & Overall & S1 & S2 & S3 & S4 & S5 \\")
    print(r"\midrule")
    for m in method_order:
        d = table3[m]
        name = METHOD_DISPLAY[m]
        if m == "geofence":
            name = r"\textbf{" + name + "}"
        vals = f"{d['Overall']:.1f}\\%"
        for sc in SCENARIOS:
            vals += f" & {d[sc]:.1f}\\%"
        print(f"{name} & {vals} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print()


def main():
    trials = load_and_filter()
    print(f"Loaded {len(trials)} trials after filtering (S1-S5, no S1 sweeps)")
    print()

    table1, table2, table3, raw_counts = compute_tables(trials)
    method_order = sort_methods_by_f1(table2)

    print(f"Method order (by F1 ascending): {[METHOD_DISPLAY[m] for m in method_order]}")
    print()

    print_tables(table1, table2, table3, raw_counts, method_order)

    # Sanity checks
    print("=" * 80)
    print("SANITY CHECKS")
    print("=" * 80)
    for m in method_order:
        d = table1[m]
        cm_sum = d["TP"] + d["FP"] + d["TN"] + d["FN"] + d["Infra"]
        ok = "OK" if cm_sum == d["N"] else "MISMATCH"
        print(f"  {METHOD_DISPLAY[m]:<20}: TP+FP+TN+FN+Infra = {cm_sum}, N = {d['N']}  [{ok}]")


if __name__ == "__main__":
    main()
