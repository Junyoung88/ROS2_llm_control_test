#!/usr/bin/env python3
"""Aggregate the NDSS attack-side experiments and render Korean-labeled figures for the advisor deck.
  (1) warehouse hijack        (wh_hijack/)         -> fig_wh_hijack.png
  (2) assumption violation    (assum_violation/)   -> fig_velviol.png     (R1-3, velocity bound)
  (3) time-varying zone       (tvzone/)            -> fig_tvzone.png      (R3-4)
Metric per method: violation rate (# seeds with any in-zone violation / N) + mean violation-count.
"""
import json, glob, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fp = fm.FontProperties(fname=FONT)
plt.rcParams["axes.unicode_minus"] = False

RES = "experiment_results/gazebo_s1_s6"
# display order + Korean labels
METHOD_LABELS = {
    "no_guard":        "무방비\n(no guard)",
    "selp_proper":     "SELP\n(승인만)",
    "static_margin":   "고정 마진\n(static)",
    "static_reactive": "반응형 고정마진\n(v_max 신뢰)",
    "cbf_inflated":    "CBF\n(계획계층)",
    "geofence":        "PETSE\n(런타임 재검증)",
}
COLORS = {
    "no_guard": "#9e9e9e", "selp_proper": "#c62828", "static_margin": "#ef6c00",
    "static_reactive": "#ef6c00", "cbf_inflated": "#1565c0", "geofence": "#2e7d32",
}

def tally(pattern):
    """pattern has one {m}; returns {method: (viol_seeds, n, mean_vc)}"""
    out = {}
    for m in METHOD_LABELS:
        fs = sorted(glob.glob(pattern.format(m=m)))
        if not fs:
            continue
        viol = n = 0; vcs = []
        for f in fs:
            if os.path.getsize(f) == 0:
                continue
            try:
                d = json.load(open(f))
            except Exception:
                continue
            n += 1
            vc = d.get("violation_count") or 0
            vcs.append(vc)
            if d.get("violated"):
                viol += 1
        if n:
            out[m] = (viol, n, sum(vcs) / len(vcs) if vcs else 0.0)
    return out

def bar_fig(data, title, subtitle, fname):
    ms = [m for m in METHOD_LABELS if m in data]
    if not ms:
        print(f"  SKIP {fname}: no data"); return
    rates = [100.0 * data[m][0] / data[m][1] for m in ms]
    ns = [data[m][1] for m in ms]
    viols = [data[m][0] for m in ms]
    vcs = [data[m][2] for m in ms]
    labels = [METHOD_LABELS[m] for m in ms]
    cols = [COLORS[m] for m in ms]

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    bars = ax.bar(range(len(ms)), rates, color=cols, edgecolor="black", linewidth=0.8, width=0.62)
    ax.set_ylim(0, 118)
    ax.set_ylabel("금지구역 침입률 (%)", fontproperties=fp, fontsize=13)
    ax.set_xticks(range(len(ms)))
    ax.set_xticklabels(labels, fontproperties=fp, fontsize=11)
    ax.set_title(title, fontproperties=fp, fontsize=15, fontweight="bold", pad=26)
    ax.text(0.5, 1.045, subtitle, transform=ax.transAxes, ha="center",
            fontproperties=fp, fontsize=10.5, color="#444")
    for i, b in enumerate(bars):
        ax.text(b.get_x() + b.get_width()/2, rates[i] + 2.5,
                f"{viols[i]}/{ns[i]}\n({rates[i]:.0f}%)", ha="center", va="bottom",
                fontproperties=fp, fontsize=10.5, fontweight="bold")
        if vcs[i] > 0:
            ax.text(b.get_x() + b.get_width()/2, min(rates[i]/2, 50),
                    f"평균\n{vcs[i]:.0f}회", ha="center", va="center",
                    fontproperties=fp, fontsize=8.5, color="white")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"scratchpad/{fname}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote scratchpad/{fname}  ::  " +
          "  ".join(f"{m}={data[m][0]}/{data[m][1]}" for m in ms))

def clearance_fig(pattern, title, subtitle, fname, margin=0.55):
    """R1-③ figure: per-seed minimum clearance to the zone, with the safety-margin line.
    Shows that the reactive fixed-margin baseline breaches its own margin under over-speed
    while PETSE holds a large clearance."""
    order = ["no_guard", "static_reactive", "geofence"]
    data = {}
    for m in order:
        fs = sorted(glob.glob(pattern.format(m=m)))
        clr = []
        for f in fs:
            if os.path.getsize(f) == 0:
                continue
            try:
                d = json.load(open(f))
            except Exception:
                continue
            pmd = d.get("path_min_distance")
            if pmd is None or pmd == float("inf"):
                pmd = 0.0
            clr.append(max(0.0, pmd))
        if clr:
            data[m] = clr
    ms = [m for m in order if m in data]
    if not ms:
        print(f"  SKIP {fname}: no data"); return
    fig, ax = plt.subplots(figsize=(8.2, 4.9))
    # zone (below 0 shaded) + margin band
    ax.axhspan(-0.3, 0.0, color="#c62828", alpha=0.16)
    ax.axhline(0.0, color="#c62828", lw=1.6)
    ax.text(len(ms) - 0.5, -0.15, "금지구역 (침입)", fontproperties=fp, fontsize=9.5,
            color="#c62828", ha="right", va="center")
    ax.axhline(margin, color="#555", lw=1.4, ls="--")
    ax.text(-0.44, margin + 0.03, f"안전마진 {margin}m", fontproperties=fp, fontsize=9.5, color="#555")
    for i, m in enumerate(ms):
        vals = data[m]
        mean = sum(vals) / len(vals)
        breach = sum(1 for v in vals if v < margin)
        enters = sum(1 for v in vals if v <= 0.001)
        ax.bar(i, mean, width=0.5, color=COLORS[m], edgecolor="black", linewidth=0.8, alpha=0.55)
        # jittered per-seed points
        for j, v in enumerate(vals):
            ax.plot(i + (j - len(vals)/2) * 0.055, v, "o", color=COLORS[m],
                    markeredgecolor="black", markersize=6, markeredgewidth=0.6)
        tag = f"침해 {breach}/{len(vals)}"
        ax.text(i, max(mean, max(vals)) + 0.12, tag, ha="center", fontproperties=fp,
                fontsize=11, fontweight="bold",
                color="#2e7d32" if breach == 0 else "#c62828")
    ax.set_ylim(-0.3, 2.7)
    ax.set_xlim(-0.6, len(ms) - 0.4)
    ax.set_xticks(range(len(ms)))
    ax.set_xticklabels([METHOD_LABELS[m] for m in ms], fontproperties=fp, fontsize=11)
    ax.set_ylabel("금지구역까지 최소 이격거리 (m)", fontproperties=fp, fontsize=13)
    ax.set_title(title, fontproperties=fp, fontsize=15, fontweight="bold", pad=26)
    ax.text(0.5, 1.045, subtitle, transform=ax.transAxes, ha="center",
            fontproperties=fp, fontsize=10.3, color="#444")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"scratchpad/{fname}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote scratchpad/{fname}  ::  " +
          "  ".join(f"{m}=breach{sum(1 for v in data[m] if v<margin)}/{len(data[m])}(min{min(data[m]):.2f})" for m in ms))

def main():
    os.chdir("/home/jim/ros2_motion_planning_tutorials")
    # (1) warehouse hijack
    wh = tally(RES + "/wh_hijack/res_wh_hijack_{m}_v*.jsonl")
    bar_fig(wh, "창고 하이잭: 탈취된 로봇을 금지구역으로 유도",
            "목표 승인 우회 + LiDAR 스푸핑 정합 공격 · 각 5시드 · PETSE만 런타임 재검증으로 차단",
            "fig_wh_hijack.png")
    # (2) assumption violation (R1-3): velocity-bound violation (reactive fixed-margin baseline)
    clearance_fig(RES + "/assum_violation/res_velreact_{m}_v*.jsonl",
            "가정 위반(R1-③): 선언된 속도 상한 초과 공격",
            "과속(v_max 3× 상한) 주행 · 선언 v_max 신뢰 고정마진은 자기 마진을 5/5 침해 · PETSE는 2.2m 이격 유지",
            "fig_velviol.png")
    # (3) time-varying zone (R3-4)
    tv = tally(RES + "/tvzone/res_tvzone_{m}_v*.jsonl")
    bar_fig(tv, "시변 금지구역(R3-4): 주행 중 금지구역 활성화",
            "목표 승인 후 금지구역 활성화 · 승인 시점 방식은 그대로 진입 · PETSE는 연속 재검증으로 정지",
            "fig_tvzone.png")

if __name__ == "__main__":
    main()
