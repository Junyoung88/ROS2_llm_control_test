#!/usr/bin/env python3
"""
Latency Effect Study - 시스템 지연에 따른 Geofence 효과 분석

이 실험은 시스템 지연(τ)이 증가할수록 Geofence 마진의
효과가 어떻게 변화하는지를 분석합니다.

핵심 질문:
- 네트워크/시스템 지연이 클수록 Geofence가 더 효과적인가?
- 마진 없이는 지연이 클수록 침범률이 얼마나 증가하는가?

실험 설계:
1. 다양한 τ 값 (50ms ~ 1000ms)
2. 각 τ에서 "마진 있음" vs "마진 없음" 비교
3. 침범률 변화 분석
"""

import sys
sys.path.insert(0, '/home/jim/ros2_motion_planning_tutorials/src/geofence_enforcer')

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

from geofence_enforcer.geofence_geometry import GeofenceGeometry


@dataclass
class LatencyTestResult:
    """지연 테스트 결과"""
    tau_ms: float              # 지연 시간 (ms)
    margin_cm: float           # 적용된 마진 (cm)
    with_margin_violations: int
    with_margin_attempts: int
    with_margin_rate: float    # 마진 있을 때 침범률
    no_margin_violations: int
    no_margin_attempts: int
    no_margin_rate: float      # 마진 없을 때 침범률
    protection_effect: float   # 보호 효과 (침범률 감소량)


class LatencyEffectSimulator:
    """시스템 지연 효과 시뮬레이터"""

    def __init__(self, seed: int = 42):
        np.random.seed(seed)

        # Geofence 설정
        self.geofence = GeofenceGeometry()
        self.geofence.add_zone("forbidden_zone", [
            (4.0, 4.0), (6.0, 4.0), (6.0, 6.0), (4.0, 6.0)
        ])

        # 고정 파라미터
        self.v_max = 1.5          # 로봇 최대 속도 (m/s)
        self.sigma_pos = 0.03     # 위치 불확실성 3cm
        self.e_track = 0.12       # 추종 오차 12cm
        self.alpha = 0.99

    def compute_full_margin(self, tau: float) -> float:
        """전체 마진 계산 (τ 포함)"""
        chi2_99 = 9.21
        estimation_term = np.sqrt(chi2_99 * self.sigma_pos**2)
        latency_term = self.v_max * tau
        return estimation_term + self.e_track + latency_term

    def generate_goal_near_boundary(self) -> Tuple[float, float]:
        """금지구역 경계 근처에 목표점 생성"""
        zone_center = (5.0, 5.0)
        zone_half_size = 1.0

        # 경계에서 0~55cm 거리에 목표점 생성
        edge = np.random.randint(4)
        offset = np.random.uniform(0, 0.55)
        along = np.random.uniform(-0.8, 0.8)

        if edge == 0:    # 상단
            return (zone_center[0] + along, zone_center[1] + zone_half_size + offset)
        elif edge == 1:  # 하단
            return (zone_center[0] + along, zone_center[1] - zone_half_size - offset)
        elif edge == 2:  # 좌측
            return (zone_center[0] - zone_half_size - offset, zone_center[1] + along)
        else:            # 우측
            return (zone_center[0] + zone_half_size + offset, zone_center[1] + along)

    def simulate_movement_with_latency(
        self,
        goal: Tuple[float, float],
        tau: float
    ) -> Tuple[float, float]:
        """지연을 고려한 이동 시뮬레이션

        지연 동안 로봇이 목표 방향으로 계속 이동하여
        overshoot 발생 가능
        """
        # 위치 추정 오차
        pos_error = np.random.normal(0, self.sigma_pos, 2)

        # 추종 오차 (목표 방향으로)
        track_error_mag = np.random.uniform(0, self.e_track)
        track_angle = np.random.uniform(0, 2*np.pi)
        track_error = track_error_mag * np.array([np.cos(track_angle), np.sin(track_angle)])

        # 지연으로 인한 overshoot (목표 방향으로)
        # 로봇이 금지구역 방향으로 이동 중이라고 가정
        zone_center = np.array([5.0, 5.0])
        goal_arr = np.array(goal)

        # 금지구역 중심 방향으로의 overshoot
        to_zone = zone_center - goal_arr
        dist_to_zone = np.linalg.norm(to_zone)
        if dist_to_zone > 0:
            direction = to_zone / dist_to_zone
            # 지연 동안 이동 거리 (속도 * 지연시간 * 랜덤 비율)
            overshoot = direction * self.v_max * tau * np.random.uniform(0.3, 1.0)
        else:
            overshoot = np.array([0, 0])

        actual_pos = goal_arr + pos_error + track_error + overshoot
        return (actual_pos[0], actual_pos[1])

    def run_single_config(
        self,
        tau: float,
        use_margin: bool,
        n_trials: int
    ) -> Tuple[int, int, int]:
        """단일 설정으로 실험"""
        margin = self.compute_full_margin(tau) if use_margin else 0.0

        blocked = 0
        attempted = 0
        violations = 0

        for _ in range(n_trials):
            goal = self.generate_goal_near_boundary()

            # 마진으로 검사
            check = self.geofence.check_point(goal[0], goal[1], margin=margin)

            if check.is_violated:
                blocked += 1
                continue

            attempted += 1

            # 지연을 고려한 실제 위치
            actual = self.simulate_movement_with_latency(goal, tau)

            # 원본 구역 침범 검사
            actual_check = self.geofence.check_point(actual[0], actual[1], margin=0)
            if actual_check.is_violated:
                violations += 1

        return blocked, attempted, violations

    def run_latency_study(
        self,
        tau_values_ms: List[float],
        n_trials: int = 3000
    ) -> List[LatencyTestResult]:
        """다양한 지연값으로 실험"""
        results = []

        for tau_ms in tau_values_ms:
            tau = tau_ms / 1000.0  # ms -> seconds
            margin = self.compute_full_margin(tau)

            print(f"[τ = {tau_ms:4.0f}ms] 마진 = {margin*100:.1f}cm ...", end=" ")

            # 마진 있을 때
            b1, a1, v1 = self.run_single_config(tau, use_margin=True, n_trials=n_trials)
            rate1 = (v1 / a1 * 100) if a1 > 0 else 0

            # 마진 없을 때
            b2, a2, v2 = self.run_single_config(tau, use_margin=False, n_trials=n_trials)
            rate2 = (v2 / a2 * 100) if a2 > 0 else 0

            # 보호 효과 = 마진 없을 때 침범률 - 마진 있을 때 침범률
            protection = rate2 - rate1

            print(f"마진O: {rate1:.2f}%, 마진X: {rate2:.2f}%, 보호효과: {protection:.2f}%p")

            results.append(LatencyTestResult(
                tau_ms=tau_ms,
                margin_cm=margin * 100,
                with_margin_violations=v1,
                with_margin_attempts=a1,
                with_margin_rate=rate1,
                no_margin_violations=v2,
                no_margin_attempts=a2,
                no_margin_rate=rate2,
                protection_effect=protection
            ))

        return results


def visualize_latency_effect(results: List[LatencyTestResult], output_path: str = None):
    """지연 효과 시각화"""

    tau_values = [r.tau_ms for r in results]
    with_margin = [r.with_margin_rate for r in results]
    no_margin = [r.no_margin_rate for r in results]
    protection = [r.protection_effect for r in results]
    margins = [r.margin_cm for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 침범률 비교
    ax1 = axes[0, 0]
    x = np.arange(len(tau_values))
    width = 0.35
    bars1 = ax1.bar(x - width/2, no_margin, width, label='No Margin', color='#e74c3c', alpha=0.8)
    bars2 = ax1.bar(x + width/2, with_margin, width, label='With Margin', color='#27ae60', alpha=0.8)
    ax1.set_xlabel('System Latency (ms)', fontsize=11)
    ax1.set_ylabel('Violation Rate (%)', fontsize=11)
    ax1.set_title('Violation Rate: With vs Without Geofence Margin', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{t:.0f}' for t in tau_values])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # 값 표시
    for bar, val in zip(bars1, no_margin):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, with_margin):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

    # 2. 보호 효과 (침범률 감소량)
    ax2 = axes[0, 1]
    colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(protection)))
    bars = ax2.bar(tau_values, protection, width=40, color=colors, edgecolor='darkgreen')
    ax2.set_xlabel('System Latency (ms)', fontsize=11)
    ax2.set_ylabel('Protection Effect (%p)', fontsize=11)
    ax2.set_title('Geofence Protection Effect by Latency\n(Violation Rate Reduction)', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, protection):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{val:.1f}%p', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 3. 적용된 마진 크기
    ax3 = axes[1, 0]
    ax3.plot(tau_values, margins, 'o-', color='#3498db', linewidth=2, markersize=8)
    ax3.fill_between(tau_values, margins, alpha=0.3, color='#3498db')
    ax3.set_xlabel('System Latency (ms)', fontsize=11)
    ax3.set_ylabel('Applied Margin (cm)', fontsize=11)
    ax3.set_title('Dynamic Margin Size by Latency\n(margin = √(χ²·λ) + e_track + v·τ)', fontsize=12, fontweight='bold')
    ax3.grid(alpha=0.3)

    for t, m in zip(tau_values, margins):
        ax3.annotate(f'{m:.0f}cm', (t, m), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=9)

    # 4. 핵심 메시지
    ax4 = axes[1, 1]
    ax4.axis('off')

    summary_text = """
    ═══════════════════════════════════════════════════════
                    KEY FINDINGS
    ═══════════════════════════════════════════════════════

    1. Higher latency → Higher violation rate (without margin)
       • τ=50ms:  {:.1f}% violations
       • τ=500ms: {:.1f}% violations
       • τ=1000ms: {:.1f}% violations

    2. Geofence margin effectively compensates for latency
       • Protection effect increases with latency
       • At τ=1000ms: {:.1f}%p reduction in violations

    3. Dynamic margin adapts to system conditions
       • margin = √(χ²·λ) + e_track + v_max·τ
       • At τ=1000ms: {:.0f}cm margin applied

    ═══════════════════════════════════════════════════════
                    CONCLUSION
    ═══════════════════════════════════════════════════════

    The v·τ term in the safety margin formula is essential
    for maintaining safety under network/system delays.

    Without this term, violation rates increase significantly
    as system latency grows.
    """.format(
        results[0].no_margin_rate,
        results[-2].no_margin_rate if len(results) > 1 else results[-1].no_margin_rate,
        results[-1].no_margin_rate,
        results[-1].protection_effect,
        results[-1].margin_cm
    )

    ax4.text(0.5, 0.5, summary_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='center', horizontalalignment='center',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6'))

    plt.tight_layout()

    if output_path is None:
        output_path = '/home/jim/ros2_motion_planning_tutorials/src/geofence_enforcer/demo/latency_effect_results.png'

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n그래프 저장됨: {output_path}")
    plt.close()


def main():
    print("=" * 70)
    print("        Latency Effect Study - 시스템 지연에 따른 Geofence 효과")
    print("=" * 70)
    print()
    print("실험: 다양한 시스템 지연(τ)에서 마진 유무에 따른 침범률 비교")
    print("v_max = 1.5 m/s, σ = 3cm, e_track = 12cm")
    print()

    simulator = LatencyEffectSimulator(seed=42)

    # 다양한 지연 값 테스트 (50ms ~ 1000ms)
    tau_values = [50, 100, 200, 300, 500, 750, 1000]

    results = simulator.run_latency_study(tau_values, n_trials=3000)

    print()
    print("=" * 70)
    print("                         RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'τ (ms)':<10} {'Margin':<12} {'With Margin':<15} {'No Margin':<15} {'Protection':<12}")
    print("-" * 70)
    for r in results:
        print(f"{r.tau_ms:<10.0f} {r.margin_cm:<10.1f}cm {r.with_margin_rate:<13.2f}% {r.no_margin_rate:<13.2f}% {r.protection_effect:<10.2f}%p")
    print("-" * 70)

    visualize_latency_effect(results)

    return results


if __name__ == "__main__":
    main()
