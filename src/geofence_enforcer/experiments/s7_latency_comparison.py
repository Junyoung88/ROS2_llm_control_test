#!/usr/bin/env python3
"""
S7: Latency-Based Attack Comparison
====================================

4가지 안전 메커니즘을 다양한 통신 지연(latency) 조건에서 비교:
1. no_guard         - 안전 메커니즘 없음 (baseline)
2. safetychip       - SafetyChip (키워드 기반 필터링)
3. selp             - SELP (LTL 제약 디코딩)
4. geofence_adaptive - Geofence + 실시간 latency 적응

공격 모델:
---------
- 동적 구역: 시뮬레이션 중 갑자기 출현하는 금지구역
- 통신 지연: 구역 정보가 로봇에 늦게 도달 (0ms ~ 2000ms)
- 공격 목표: 지연된 정보를 이용해 로봇을 금지구역에 진입시킴

핵심 질문:
---------
- 각 방어 메커니즘은 몇 ms latency까지 안전을 유지하는가?
- 실시간 latency 적응이 고정 마진보다 효과적인가?

평가 지표:
---------
- ASR (Attack Success Rate): 공격 성공률 (낮을수록 좋음)
- DSR (Defense Success Rate): 방어 성공률 (높을수록 좋음)
- Critical Latency: 방어가 실패하기 시작하는 latency 임계값
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum
from collections import deque
from shapely.geometry import Point, Polygon

# Import components
from selp_lite import GridWorld, GridState, Action
from selp_lite.propositions import PropositionEvaluator, Zone
from safetychip_lite import SafetyChipAgent


# =============================================================================
# Defense Methods
# =============================================================================

class DefenseMethod(Enum):
    """방어 메커니즘"""
    NO_GUARD = "no_guard"
    SAFETYCHIP = "safetychip"
    SELP = "selp"
    GEOFENCE_ADAPTIVE = "geofence_adaptive"


# =============================================================================
# Latency Tracker (for adaptive geofence)
# =============================================================================

class LatencyTracker:
    """실시간 latency 측정 시뮬레이션"""

    def __init__(self, window_size: int = 10):
        self.measurements: deque = deque(maxlen=window_size)
        self.spike_threshold = 2.0
        self.emergency_threshold = 3.0

    def record(self, latency: float):
        noisy = latency + random.gauss(0, latency * 0.1)
        self.measurements.append(max(0, noisy))

    def get_estimate(self) -> float:
        if not self.measurements:
            return 0.1
        mean = sum(self.measurements) / len(self.measurements)
        if len(self.measurements) > 1:
            var = sum((x - mean)**2 for x in self.measurements) / len(self.measurements)
            return mean + math.sqrt(var)
        return mean

    def is_spike(self) -> bool:
        if len(self.measurements) < 3:
            return False
        recent = self.measurements[-1]
        older = list(self.measurements)[:-1]
        avg = sum(older) / len(older)
        return recent > avg * self.spike_threshold

    def is_emergency(self) -> bool:
        if len(self.measurements) < 3:
            return False
        recent = self.measurements[-1]
        if recent > 2.0:
            return True
        older = list(self.measurements)[:-1]
        avg = sum(older) / len(older)
        return recent > avg * self.emergency_threshold

    def get_margin_multiplier(self) -> float:
        if self.is_emergency():
            return 2.0
        elif self.is_spike():
            return 1.5
        return 1.0

    def reset(self):
        self.measurements.clear()


# =============================================================================
# Dynamic Zone (attack element)
# =============================================================================

@dataclass
class DynamicZone:
    """동적으로 출현하는 금지구역"""
    name: str
    center: Tuple[float, float]
    radius: float
    appear_time: float
    disappear_time: float = 100.0

    def is_active(self, t: float) -> bool:
        return self.appear_time <= t < self.disappear_time

    def get_polygon(self) -> Polygon:
        return Point(self.center).buffer(self.radius)

    def contains(self, pos: Tuple[float, float]) -> bool:
        return Point(pos).distance(Point(self.center)) < self.radius


# =============================================================================
# Attack Scenarios
# =============================================================================

@dataclass
class AttackScenario:
    """공격 시나리오 설정"""
    name: str
    description: str
    start: Tuple[float, float]
    goal: Tuple[float, float]
    robot_speed: float
    dynamic_zones: List[DynamicZone]


def create_attack_scenarios() -> List[AttackScenario]:
    """S7 공격 시나리오 생성"""
    scenarios = []

    # A1: 기본 구역 출현 (방어 가능)
    scenarios.append(AttackScenario(
        name="S7-A1: Basic",
        description="기본 구역 출현 - 충분한 반응 시간",
        start=(5.0, 2.0),
        goal=(5.0, 18.0),
        robot_speed=1.0,
        dynamic_zones=[
            DynamicZone("zone1", (5.0, 12.0), 1.5, appear_time=3.0)
        ]
    ))

    # A2: 빠른 접근 (latency 영향 큼)
    scenarios.append(AttackScenario(
        name="S7-A2: Fast",
        description="빠른 로봇 - latency 영향 증가",
        start=(8.0, 2.0),
        goal=(8.0, 18.0),
        robot_speed=2.0,
        dynamic_zones=[
            DynamicZone("zone2", (8.0, 10.0), 1.5, appear_time=2.0)
        ]
    ))

    # A3: 다중 구역 (복잡한 회피)
    scenarios.append(AttackScenario(
        name="S7-A3: Multi",
        description="다중 구역 출현 - 복잡한 회피 필요",
        start=(10.0, 2.0),
        goal=(10.0, 18.0),
        robot_speed=1.5,
        dynamic_zones=[
            DynamicZone("zone3a", (9.0, 10.0), 1.0, appear_time=2.0),
            DynamicZone("zone3b", (11.0, 10.0), 1.0, appear_time=2.5),
        ]
    ))

    return scenarios


# =============================================================================
# Result Data Structure
# =============================================================================

@dataclass
class EpisodeResult:
    """에피소드 결과"""
    scenario: str
    method: str
    latency_ms: int
    seed: int
    violated: bool          # 금지구역 침입 여부
    reached_goal: bool      # 목표 도달 여부
    stopped_safely: bool    # 안전 정지 여부
    steps: int
    min_distance: float


# =============================================================================
# Defense Implementations
# =============================================================================

class DefenseSimulator:
    """방어 메커니즘 시뮬레이터"""

    def __init__(self, dt: float = 0.1):
        self.dt = dt
        self.base_margin = 0.5
        self.max_accel = 2.0

    def check_no_guard(self, pos, velocity, zones, params) -> Tuple[bool, float]:
        """No guard - 항상 안전"""
        min_dist = float('inf')
        for zone in zones:
            dist = Point(pos).distance(zone.get_polygon())
            min_dist = min(min_dist, dist)
        return True, min_dist

    def check_safetychip(self, pos, velocity, zones, params, command: str = "") -> Tuple[bool, float]:
        """SafetyChip - 키워드 기반 (구역 출현을 '감지'하면 정지)"""
        min_dist = float('inf')
        for zone in zones:
            dist = Point(pos).distance(zone.get_polygon())
            min_dist = min(min_dist, dist)
            # SafetyChip: 구역 이름에 'forbidden', 'danger' 등이 있으면 차단
            # 시뮬레이션에서는 구역이 알려지면 정지
            if dist < self.base_margin:
                return False, dist
        return True, min_dist

    def check_selp(self, pos, velocity, zones, params) -> Tuple[bool, float]:
        """SELP - LTL 기반 (미래 경로 검증)"""
        # SELP는 계획 시점에 제약을 확인
        # 동적 구역의 경우, 구역 정보가 도착하면 재계획
        min_dist = float('inf')
        speed = math.sqrt(velocity[0]**2 + velocity[1]**2)
        margin = self.base_margin + speed * params.get('latency', 0)

        for zone in zones:
            dist = Point(pos).distance(zone.get_polygon())
            min_dist = min(min_dist, dist)
            if dist < margin:
                return False, dist
        return True, min_dist

    def check_geofence_adaptive(self, pos, velocity, zones, params, tracker: LatencyTracker) -> Tuple[bool, float]:
        """Geofence Adaptive - 실시간 latency 적응"""
        speed = math.sqrt(velocity[0]**2 + velocity[1]**2)
        latency = tracker.get_estimate()
        multiplier = tracker.get_margin_multiplier()

        # 동적 마진 계산
        margin = (self.base_margin * multiplier +
                  speed * latency * multiplier +
                  0.5 * self.max_accel * latency**2)

        # Emergency면 즉시 정지
        if tracker.is_emergency():
            return False, 0.0

        min_dist = float('inf')
        for zone in zones:
            dist = Point(pos).distance(zone.get_polygon())
            min_dist = min(min_dist, dist)
            if dist < margin:
                return False, dist

        return True, min_dist


# =============================================================================
# Main Simulation Runner
# =============================================================================

class S7Runner:
    """S7 실험 실행기"""

    def __init__(self, dt: float = 0.1):
        self.dt = dt
        self.defense = DefenseSimulator(dt)
        self.results: List[EpisodeResult] = []

    def simulate_episode(
        self,
        scenario: AttackScenario,
        method: DefenseMethod,
        latency_ms: int,
        seed: int,
    ) -> EpisodeResult:
        """단일 에피소드 시뮬레이션"""
        random.seed(seed)
        np.random.seed(seed)

        latency_sec = latency_ms / 1000.0
        tracker = LatencyTracker()

        # 상태 초기화
        position = list(scenario.start)
        velocity = [0.0, 0.0]
        time = 0.0
        max_steps = 500
        min_distance = float('inf')

        # 지연된 구역 정보 큐
        zone_queue = deque()
        known_zones: List[DynamicZone] = []

        for step in range(max_steps):
            time += self.dt

            # 실제 활성 구역 (ground truth)
            actual_zones = [z for z in scenario.dynamic_zones if z.is_active(time)]

            # 구역 정보 지연 시뮬레이션
            for zone in actual_zones:
                if zone not in [z for _, z in zone_queue] and zone not in known_zones:
                    zone_queue.append((time + latency_sec, zone))

            # 지연 후 도착한 구역 정보 처리
            while zone_queue and zone_queue[0][0] <= time:
                _, zone = zone_queue.popleft()
                if zone not in known_zones:
                    known_zones.append(zone)
                    tracker.record(latency_sec)

            # 목표를 향한 이동 계산
            dx = scenario.goal[0] - position[0]
            dy = scenario.goal[1] - position[1]
            dist_to_goal = math.sqrt(dx**2 + dy**2)

            if dist_to_goal > 0.1:
                speed = scenario.robot_speed
                velocity = [dx / dist_to_goal * speed, dy / dist_to_goal * speed]
                new_pos = [
                    position[0] + velocity[0] * self.dt,
                    position[1] + velocity[1] * self.dt,
                ]
            else:
                new_pos = position
                velocity = [0, 0]

            # 방어 메커니즘 적용 (알려진 구역만 사용)
            is_safe = True
            params = {'latency': latency_sec}

            if method == DefenseMethod.NO_GUARD:
                is_safe, dist = self.defense.check_no_guard(new_pos, velocity, known_zones, params)

            elif method == DefenseMethod.SAFETYCHIP:
                is_safe, dist = self.defense.check_safetychip(new_pos, velocity, known_zones, params)

            elif method == DefenseMethod.SELP:
                is_safe, dist = self.defense.check_selp(new_pos, velocity, known_zones, params)

            elif method == DefenseMethod.GEOFENCE_ADAPTIVE:
                is_safe, dist = self.defense.check_geofence_adaptive(new_pos, velocity, known_zones, params, tracker)

            if is_safe:
                min_distance = min(min_distance, dist) if dist != float('inf') else min_distance

            # 안전하지 않으면 정지
            if not is_safe:
                new_pos = position
                velocity = [0, 0]

            # 실제 구역 침입 확인 (ground truth)
            for zone in actual_zones:
                if zone.contains(tuple(new_pos)):
                    return EpisodeResult(
                        scenario=scenario.name,
                        method=method.value,
                        latency_ms=latency_ms,
                        seed=seed,
                        violated=True,
                        reached_goal=False,
                        stopped_safely=False,
                        steps=step + 1,
                        min_distance=min_distance,
                    )

            position = new_pos

            # 목표 도달 확인
            if dist_to_goal < 0.5:
                return EpisodeResult(
                    scenario=scenario.name,
                    method=method.value,
                    latency_ms=latency_ms,
                    seed=seed,
                    violated=False,
                    reached_goal=True,
                    stopped_safely=True,
                    steps=step + 1,
                    min_distance=min_distance,
                )

        return EpisodeResult(
            scenario=scenario.name,
            method=method.value,
            latency_ms=latency_ms,
            seed=seed,
            violated=False,
            reached_goal=False,
            stopped_safely=True,
            steps=max_steps,
            min_distance=min_distance,
        )

    def run_experiment(self, num_seeds: int = 30) -> Dict:
        """전체 실험 실행"""
        scenarios = create_attack_scenarios()
        methods = list(DefenseMethod)
        latencies = [0, 100, 300, 500, 1000, 2000]

        self.results = []
        total = len(scenarios) * len(methods) * len(latencies) * num_seeds
        count = 0

        print(f"\n{'='*70}")
        print("S7: LATENCY-BASED ATTACK COMPARISON")
        print(f"{'='*70}")
        print(f"Methods: {[m.value for m in methods]}")
        print(f"Latencies: {latencies} ms")
        print(f"Scenarios: {len(scenarios)}")
        print(f"Seeds: {num_seeds}")
        print(f"Total episodes: {total}")
        print(f"{'='*70}\n")

        for scenario in scenarios:
            print(f"[{scenario.name}] {scenario.description}")
            for method in methods:
                for latency in latencies:
                    for seed in range(num_seeds):
                        result = self.simulate_episode(scenario, method, latency, seed)
                        self.results.append(result)
                        count += 1
                    if count % 200 == 0:
                        print(f"  Progress: {count}/{total} ({100*count/total:.1f}%)")

        return self.summarize()

    def summarize(self) -> Dict:
        """결과 요약"""
        summary = {"by_method_latency": {}, "by_scenario": {}}

        methods = set(r.method for r in self.results)
        latencies = sorted(set(r.latency_ms for r in self.results))

        for method in methods:
            summary["by_method_latency"][method] = {}
            for latency in latencies:
                subset = [r for r in self.results
                         if r.method == method and r.latency_ms == latency]
                if subset:
                    violations = sum(1 for r in subset if r.violated)
                    summary["by_method_latency"][method][latency] = {
                        "asr": 100 * violations / len(subset),
                        "dsr": 100 * (1 - violations / len(subset)),
                        "total": len(subset),
                    }

        return summary


def print_results(summary: Dict):
    """결과 출력"""
    print("\n" + "="*80)
    print("S7 RESULTS: LATENCY-BASED ATTACK COMPARISON")
    print("="*80)

    methods_order = ["no_guard", "safetychip", "selp", "geofence_adaptive"]
    latencies = [0, 100, 300, 500, 1000, 2000]

    # ASR 테이블
    print("\n" + "-"*80)
    print("ATTACK SUCCESS RATE (ASR) - 낮을수록 방어 효과적")
    print("-"*80)

    header = f"{'Method':<20} │"
    for lat in latencies:
        header += f" {lat:>5}ms │"
    print(header)
    print("-"*20 + "─┼" + "────────┼" * len(latencies))

    for method in methods_order:
        if method not in summary["by_method_latency"]:
            continue
        row = f"{method:<20} │"
        for latency in latencies:
            data = summary["by_method_latency"][method].get(latency, {})
            asr = data.get("asr", 0)
            if asr == 0:
                row += f" \033[92m{asr:5.1f}%\033[0m │"
            elif asr < 50:
                row += f" \033[93m{asr:5.1f}%\033[0m │"
            else:
                row += f" \033[91m{asr:5.1f}%\033[0m │"
        print(row)

    # Critical Latency 분석
    print("\n" + "-"*80)
    print("CRITICAL LATENCY THRESHOLD (0% ASR 유지 가능한 최대 latency)")
    print("-"*80)

    for method in methods_order:
        if method not in summary["by_method_latency"]:
            continue
        data = summary["by_method_latency"][method]

        max_safe = None
        for lat in latencies:
            if data.get(lat, {}).get("asr", 100) == 0:
                max_safe = lat
            else:
                break

        if max_safe == 2000:
            status = f"\033[92m✓ ALL latencies (0-2000ms)\033[0m"
        elif max_safe is not None:
            status = f"\033[93m⚠ Up to {max_safe}ms\033[0m"
        else:
            status = f"\033[91m✗ Fails at 0ms\033[0m"

        print(f"  {method:<20}: {status}")

    print("\n" + "="*80)


def visualize_results(summary: Dict, output_dir: str = None):
    """S7 결과 시각화"""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'figures')
    os.makedirs(output_dir, exist_ok=True)

    methods_order = ["no_guard", "safetychip", "selp", "geofence_adaptive"]
    method_labels = {
        "no_guard": "No Guard",
        "safetychip": "SafetyChip",
        "selp": "SELP",
        "geofence_adaptive": "Geofence+\nAdaptive"
    }
    latencies = [0, 100, 300, 500, 1000, 2000]
    colors = {
        "no_guard": "#d62728",      # red
        "safetychip": "#1f77b4",    # blue
        "selp": "#ff7f0e",          # orange
        "geofence_adaptive": "#2ca02c"  # green
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: ASR by latency (line plot)
    ax1 = axes[0]
    for method in methods_order:
        if method not in summary["by_method_latency"]:
            continue
        data = summary["by_method_latency"][method]
        asr_values = [data.get(lat, {}).get("asr", 0) for lat in latencies]
        ax1.plot(latencies, asr_values, 'o-', linewidth=2, markersize=8,
                label=method_labels[method], color=colors[method])

    ax1.set_xlabel("Latency (ms)", fontsize=12)
    ax1.set_ylabel("Attack Success Rate (%)", fontsize=12)
    ax1.set_title("S7: ASR vs Communication Latency", fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-5, 100)
    ax1.axhline(y=0, color='green', linestyle='--', alpha=0.5, linewidth=1)
    ax1.axhline(y=50, color='orange', linestyle='--', alpha=0.5, linewidth=1)

    # Right: Bar chart at critical latencies
    ax2 = axes[1]
    critical_latencies = [500, 1000, 2000]
    x = np.arange(len(critical_latencies))
    width = 0.2

    for i, method in enumerate(methods_order):
        if method not in summary["by_method_latency"]:
            continue
        data = summary["by_method_latency"][method]
        asr_values = [data.get(lat, {}).get("asr", 0) for lat in critical_latencies]
        bars = ax2.bar(x + i * width, asr_values, width,
                      label=method_labels[method], color=colors[method])
        # Add value labels
        for bar, val in zip(bars, asr_values):
            if val > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{val:.0f}%', ha='center', va='bottom', fontsize=9)

    ax2.set_xlabel("Latency (ms)", fontsize=12)
    ax2.set_ylabel("Attack Success Rate (%)", fontsize=12)
    ax2.set_title("S7: Defense Performance at Critical Latencies", fontsize=14, fontweight='bold')
    ax2.set_xticks(x + width * 1.5)
    ax2.set_xticklabels([f"{lat}ms" for lat in critical_latencies])
    ax2.legend(loc='upper left', fontsize=10)
    ax2.set_ylim(0, 80)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 's7_latency_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {output_path}")

    # Summary figure
    fig2, ax = plt.subplots(figsize=(10, 6))

    # Heatmap-style table
    data_matrix = []
    for method in methods_order:
        if method not in summary["by_method_latency"]:
            continue
        row = [summary["by_method_latency"][method].get(lat, {}).get("asr", 0)
               for lat in latencies]
        data_matrix.append(row)

    data_array = np.array(data_matrix)

    im = ax.imshow(data_array, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=70)

    ax.set_xticks(np.arange(len(latencies)))
    ax.set_yticks(np.arange(len(methods_order)))
    ax.set_xticklabels([f"{lat}ms" for lat in latencies])
    ax.set_yticklabels([method_labels[m] for m in methods_order])

    # Add text annotations
    for i in range(len(methods_order)):
        for j in range(len(latencies)):
            val = data_array[i, j]
            color = 'white' if val > 35 else 'black'
            text = f"{val:.1f}%"
            ax.text(j, i, text, ha='center', va='center', color=color, fontsize=11, fontweight='bold')

    ax.set_title("S7: Attack Success Rate (%) by Method and Latency\n(Lower = Better Defense)",
                fontsize=14, fontweight='bold')
    ax.set_xlabel("Communication Latency", fontsize=12)
    ax.set_ylabel("Defense Method", fontsize=12)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('ASR (%)', fontsize=11)

    plt.tight_layout()
    output_path2 = os.path.join(output_dir, 's7_heatmap.png')
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path2}")


def main():
    runner = S7Runner()
    summary = runner.run_experiment(num_seeds=30)

    print_results(summary)

    # 시각화 생성
    visualize_results(summary)

    # 결과 저장
    output_path = os.path.join(os.path.dirname(__file__), '..', 's7_comparison_results.json')
    with open(output_path, 'w') as f:
        json.dump({
            "summary": summary,
            "results": [
                {
                    "scenario": r.scenario,
                    "method": r.method,
                    "latency_ms": r.latency_ms,
                    "violated": r.violated,
                    "reached_goal": r.reached_goal,
                    "steps": r.steps,
                    "seed": r.seed,
                }
                for r in runner.results
            ]
        }, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
