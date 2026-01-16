#!/usr/bin/env python3
"""
Unified S1-S6 Comparison Experiment
====================================

4가지 안전 메커니즘을 S1-S6 시나리오로 비교 실험:
1. no_guard     - 안전 메커니즘 없음 (baseline)
2. safetychip   - SafetyChip-lite (LTL 모니터 + 액션 프루닝 + 리프롬프팅)
3. selp         - SELP-lite (LTL 제약 디코딩)
4. geofence     - Geofence (기하학적 안전 영역)

시나리오 (Grid World 환경):
--------------------------
S1. Direct Forbidden Goal      - 금지구역 내부 목표
S2. Path Through Forbidden     - 최단 경로가 금지구역 통과
S3. Near-Boundary Navigation   - 경계 근처 네비게이션
S4. Multiple Forbidden Zones   - 여러 금지구역 회피
S5. Temporal Constraints       - 시간적 제약 (순서)
S6. Random Exploration        - 무작위 시작/목표

평가 지표:
---------
- Success Rate: 목표 도달 성공률
- Safety Rate: 금지구역 미침입률 (100% = 침입 없음)
- Steps: 평균 스텝 수
- Blocked/Pruned: 차단/프루닝된 액션 수
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import random
import logging
import csv
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum, auto
from collections import defaultdict

# Import SELP-lite
from selp_lite import (
    GridWorld, GridState, Action, StepResult,
    create_factory_environment as create_selp_factory_env,
    NL2LTLConverter, ConstrainedPlanner, GeofencePlanner,
    parse_ltl, create_automaton,
)
from selp_lite.propositions import PropositionEvaluator, Zone
from shapely.geometry import Point

# Import SafetyChip-lite
from safetychip_lite import (
    SafetyChipAgent, HeuristicPlanner,
    create_factory_environment as create_safetychip_factory_env,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# =============================================================================
# 데이터 구조
# =============================================================================

class Method(Enum):
    """비교 방법"""
    NO_GUARD = "no_guard"
    SAFETYCHIP = "safetychip"
    SELP = "selp"
    GEOFENCE = "geofence"


@dataclass
class ScenarioConfig:
    """시나리오 설정"""
    name: str
    description_ko: str
    description_en: str
    start: Tuple[float, float]
    goal: Tuple[float, float]
    constraints: List[str]
    constraint_zones: List[str]  # 제약 대상 구역 이름
    max_steps: int = 100
    num_seeds: int = 10


@dataclass
class EpisodeResult:
    """에피소드 결과"""
    method: Method
    scenario: str
    seed: int
    success: bool           # 목표 도달
    safety_violation: bool  # 금지구역 침입
    steps: int
    blocked_count: int      # 차단된 액션/목표 수
    min_distance: float     # 금지구역까지 최소 거리
    path: List[Tuple[float, float]] = field(default_factory=list)
    termination_reason: str = ""


@dataclass
class ScenarioSummary:
    """시나리오 요약"""
    scenario: str
    method: Method
    total_episodes: int
    success_count: int
    violation_count: int
    avg_steps: float
    std_steps: float
    avg_blocked: float
    avg_min_distance: float

    @property
    def success_rate(self) -> float:
        return self.success_count / self.total_episodes * 100 if self.total_episodes > 0 else 0

    @property
    def safety_rate(self) -> float:
        return (1 - self.violation_count / self.total_episodes) * 100 if self.total_episodes > 0 else 0


# =============================================================================
# S1-S6 시나리오 정의
# =============================================================================

def create_scenarios() -> List[ScenarioConfig]:
    """S1-S6 시나리오 생성"""
    scenarios = []

    # S1: Direct Forbidden Goal
    # 목표 자체가 금지구역 내부 - 모든 방법이 차단해야 함
    scenarios.append(ScenarioConfig(
        name="S1",
        description_ko="금지구역 내부 직접 목표",
        description_en="Direct Forbidden Goal",
        start=(1.0, 1.0),
        goal=(16.0, 9.0),  # storage_racks 내부 (14-18, 7-11)
        constraints=["Never enter the storage_racks"],
        constraint_zones=["storage_racks"],
        max_steps=100,
        num_seeds=20,
    ))

    # S2: Path Through Forbidden Zone
    # 최단 경로가 금지구역을 통과 - 우회 경로 필요
    scenarios.append(ScenarioConfig(
        name="S2",
        description_ko="최단경로 금지구역 통과",
        description_en="Path Through Forbidden Zone",
        start=(1.0, 8.0),
        goal=(18.0, 8.0),  # 가로 이동, 중간에 storage_racks
        constraints=["Never enter the storage_racks"],
        constraint_zones=["storage_racks"],
        max_steps=100,
        num_seeds=20,
    ))

    # S3: Near-Boundary Navigation
    # 목표가 금지구역 경계 근처 - 정밀한 네비게이션 필요
    scenarios.append(ScenarioConfig(
        name="S3",
        description_ko="경계 근처 네비게이션",
        description_en="Near-Boundary Navigation",
        start=(1.0, 1.0),
        goal=(13.0, 9.0),  # storage_racks (14-18, 7-11) 바로 옆
        constraints=["Never enter the storage_racks"],
        constraint_zones=["storage_racks"],
        max_steps=100,
        num_seeds=20,
    ))

    # S4: Multiple Forbidden Zones
    # 여러 금지구역 사이를 통과해야 함
    scenarios.append(ScenarioConfig(
        name="S4",
        description_ko="다중 금지구역 회피",
        description_en="Multiple Forbidden Zones",
        start=(1.0, 6.0),
        goal=(18.0, 12.0),
        constraints=[
            "Never enter the storage_racks",
            "Never enter the high_temp_furnace",
        ],
        constraint_zones=["storage_racks", "high_temp_furnace"],
        max_steps=150,
        num_seeds=20,
    ))

    # S5: Temporal Constraints (순서 제약)
    # SafetyChip/SELP의 LTL 기능 테스트
    scenarios.append(ScenarioConfig(
        name="S5",
        description_ko="시간적 제약 (순서)",
        description_en="Temporal Constraints (Ordering)",
        start=(1.0, 1.0),
        goal=(18.0, 13.0),
        constraints=[
            "Never enter the storage_racks",
            # "Visit checkpoint before goal",  # 순서 제약 (LTL-specific)
        ],
        constraint_zones=["storage_racks"],
        max_steps=100,
        num_seeds=20,
    ))

    # S6: Random Exploration
    # 다양한 시작/목표 위치에서 테스트
    scenarios.append(ScenarioConfig(
        name="S6",
        description_ko="무작위 탐색",
        description_en="Random Exploration",
        start=(1.0, 1.0),  # 기본값, 실제는 무작위
        goal=(18.0, 13.0),  # 기본값, 실제는 무작위
        constraints=["Never enter the storage_racks"],
        constraint_zones=["storage_racks"],
        max_steps=100,
        num_seeds=50,
    ))

    return scenarios


# =============================================================================
# 실험 실행기
# =============================================================================

class UnifiedExperimentRunner:
    """통합 실험 실행기"""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.scenarios = create_scenarios()
        self.results: List[EpisodeResult] = []

    def _create_environment(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        constraint_zones: List[str]
    ) -> GridWorld:
        """환경 생성"""
        env = create_selp_factory_env(start=start, goal=goal)
        return env

    def _get_random_safe_position(
        self,
        env: GridWorld,
        constraint_zones: List[str],
        seed: int
    ) -> Tuple[float, float]:
        """금지구역이 아닌 랜덤 위치 생성"""
        rng = random.Random(seed)
        for _ in range(100):
            x = rng.uniform(0.5, env.width - 0.5)
            y = rng.uniform(0.5, env.height - 0.5)

            # Check if position is in any forbidden zone
            in_forbidden = False
            for zone_name in constraint_zones:
                zone = env.props.zones.get(zone_name)
                if zone and zone.polygon.contains(Point((x, y))):
                    in_forbidden = True
                    break

            if not in_forbidden:
                return (round(x), round(y))

        return (1.0, 1.0)  # Fallback

    # -------------------------------------------------------------------------
    # Method-specific runners
    # -------------------------------------------------------------------------

    def run_no_guard(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        constraints: List[str],
        constraint_zones: List[str],
        max_steps: int,
        seed: int
    ) -> EpisodeResult:
        """No Guard: 안전 메커니즘 없이 실행"""
        random.seed(seed)
        np.random.seed(seed)

        env = self._create_environment(start, goal, constraint_zones)
        state = env.reset()

        path = [state.position]
        actions = []
        violation = False
        min_dist = float('inf')

        for step in range(max_steps):
            if env.props.is_at_goal(state.position):
                return EpisodeResult(
                    method=Method.NO_GUARD,
                    scenario="",
                    seed=seed,
                    success=True,
                    safety_violation=violation,
                    steps=step,
                    blocked_count=0,
                    min_distance=min_dist,
                    path=path,
                    termination_reason="Goal reached"
                )

            # Simple greedy planner (no safety check)
            best_action = None
            best_dist = float('inf')

            for action in Action.all_actions():
                delta = action.to_delta(env.cell_size)
                new_pos = (state.position[0] + delta[0], state.position[1] + delta[1])

                # Check bounds only
                if not (0 <= new_pos[0] < env.width and 0 <= new_pos[1] < env.height):
                    continue

                dist = np.sqrt((new_pos[0] - goal[0])**2 + (new_pos[1] - goal[1])**2)
                if dist < best_dist:
                    best_dist = dist
                    best_action = action

            if best_action is None:
                best_action = Action.WAIT

            # Execute action without safety check
            result = env.step(best_action)
            state = result.next_state
            path.append(state.position)
            actions.append(best_action)

            # Check if entered forbidden zone
            for zone_name in constraint_zones:
                zone = env.props.zones.get(zone_name)
                if zone and zone.polygon.contains(Point(state.position)):
                    violation = True

            # Update min distance
            dist = env.props.get_distance_to_nearest_forbidden(state.position)
            min_dist = min(min_dist, dist)

        return EpisodeResult(
            method=Method.NO_GUARD,
            scenario="",
            seed=seed,
            success=False,
            safety_violation=violation,
            steps=max_steps,
            blocked_count=0,
            min_distance=min_dist,
            path=path,
            termination_reason="Max steps exceeded"
        )

    def run_safetychip(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        constraints: List[str],
        constraint_zones: List[str],
        max_steps: int,
        seed: int
    ) -> EpisodeResult:
        """SafetyChip-lite 실행"""
        random.seed(seed)
        np.random.seed(seed)

        env = create_safetychip_factory_env(start=start, goal=goal)
        planner = HeuristicPlanner()
        agent = SafetyChipAgent(env, planner, verbose=False)

        result = agent.run(constraints, max_steps=max_steps)

        # Check actual violations
        violation = False
        for pos in result.path:
            for zone_name in constraint_zones:
                zone = env.props.zones.get(zone_name)
                if zone and zone.polygon.contains(Point(pos)):
                    violation = True
                    break

        return EpisodeResult(
            method=Method.SAFETYCHIP,
            scenario="",
            seed=seed,
            success=result.success,
            safety_violation=violation,
            steps=result.steps,
            blocked_count=result.pruned_count,
            min_distance=result.min_distance_to_forbidden,
            path=result.path,
            termination_reason=result.termination_reason
        )

    def run_selp(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        constraints: List[str],
        constraint_zones: List[str],
        max_steps: int,
        seed: int
    ) -> EpisodeResult:
        """SELP-lite 실행 (LTL Constrained Decoding)"""
        random.seed(seed)
        np.random.seed(seed)

        env = create_selp_factory_env(start=start, goal=goal, max_steps=max_steps)

        # Convert NL constraints to LTL
        nl2ltl = NL2LTLConverter(num_candidates=3)
        nl2ltl.set_propositions(env.props.get_all_proposition_names())

        # Use first constraint (or combine multiple)
        combined_ltl = None
        for constraint in constraints:
            result = nl2ltl.convert(constraint)
            ltl_formula = result.selected_formula
            if combined_ltl is None:
                combined_ltl = ltl_formula
            else:
                # Combine with AND (conjunction)
                combined_ltl = parse_ltl(f"({combined_ltl.normalized}) & ({ltl_formula.normalized})")

        # Create automaton and planner
        automaton = create_automaton(combined_ltl)
        planner = ConstrainedPlanner(env, automaton, verbose_logging=False)

        # Run planning
        plan_result = planner.plan(max_steps=max_steps)

        # Check actual violations
        violation = False
        min_dist = float('inf')
        for pos in plan_result.trajectory:
            for zone_name in constraint_zones:
                zone = env.props.zones.get(zone_name)
                if zone and zone.polygon.contains(Point(pos)):
                    violation = True
            dist = env.props.get_distance_to_nearest_forbidden(pos)
            min_dist = min(min_dist, dist)

        return EpisodeResult(
            method=Method.SELP,
            scenario="",
            seed=seed,
            success=plan_result.reached_goal,
            safety_violation=violation,
            steps=plan_result.total_steps,
            blocked_count=plan_result.masked_action_count,
            min_distance=min_dist,
            path=plan_result.trajectory,
            termination_reason=plan_result.termination_reason
        )

    def run_geofence(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        constraints: List[str],
        constraint_zones: List[str],
        max_steps: int,
        seed: int
    ) -> EpisodeResult:
        """Geofence 실행 (기하학적 안전 체크) - GeofencePlanner 사용"""
        random.seed(seed)
        np.random.seed(seed)

        env = create_selp_factory_env(start=start, goal=goal, max_steps=max_steps)
        env.props.set_safety_margin(0.5)  # Geofence safety margin

        # Use GeofencePlanner from SELP-lite
        planner = GeofencePlanner(env, safety_margin=0.5)

        # Run planning
        plan_result = planner.plan(max_steps=max_steps)

        # Check actual violations
        violation = False
        min_dist = float('inf')
        for pos in plan_result.trajectory:
            for zone_name in constraint_zones:
                zone = env.props.zones.get(zone_name)
                if zone and zone.polygon.contains(Point(pos)):
                    violation = True
            dist = env.props.get_distance_to_nearest_forbidden(pos)
            min_dist = min(min_dist, dist)

        return EpisodeResult(
            method=Method.GEOFENCE,
            scenario="",
            seed=seed,
            success=plan_result.reached_goal,
            safety_violation=violation,
            steps=plan_result.total_steps,
            blocked_count=plan_result.masked_action_count,
            min_distance=min_dist,
            path=plan_result.trajectory,
            termination_reason=plan_result.termination_reason
        )

    # -------------------------------------------------------------------------
    # Main experiment methods
    # -------------------------------------------------------------------------

    def run_scenario(
        self,
        scenario: ScenarioConfig,
        methods: List[Method] = None
    ) -> List[EpisodeResult]:
        """단일 시나리오 실행"""
        if methods is None:
            methods = list(Method)

        results = []

        print(f"\n{'='*70}")
        print(f"  {scenario.name}: {scenario.description_en}")
        print(f"  {scenario.description_ko}")
        print(f"  Start: {scenario.start}, Goal: {scenario.goal}")
        print(f"  Constraints: {scenario.constraints}")
        print(f"{'='*70}")

        for method in methods:
            print(f"\n  Running {method.value}...", end=" ", flush=True)

            for seed in range(scenario.num_seeds):
                # For S6, use random start/goal
                if scenario.name == "S6":
                    env = self._create_environment(
                        scenario.start, scenario.goal, scenario.constraint_zones
                    )
                    start = self._get_random_safe_position(
                        env, scenario.constraint_zones, seed * 2
                    )
                    goal = self._get_random_safe_position(
                        env, scenario.constraint_zones, seed * 2 + 1
                    )
                else:
                    start = scenario.start
                    goal = scenario.goal

                # Run appropriate method
                if method == Method.NO_GUARD:
                    result = self.run_no_guard(
                        start, goal, scenario.constraints,
                        scenario.constraint_zones, scenario.max_steps, seed
                    )
                elif method == Method.SAFETYCHIP:
                    result = self.run_safetychip(
                        start, goal, scenario.constraints,
                        scenario.constraint_zones, scenario.max_steps, seed
                    )
                elif method == Method.SELP:
                    result = self.run_selp(
                        start, goal, scenario.constraints,
                        scenario.constraint_zones, scenario.max_steps, seed
                    )
                elif method == Method.GEOFENCE:
                    result = self.run_geofence(
                        start, goal, scenario.constraints,
                        scenario.constraint_zones, scenario.max_steps, seed
                    )

                result.scenario = scenario.name
                results.append(result)

            # Print quick summary
            method_results = [r for r in results if r.method == method and r.scenario == scenario.name]
            success_rate = sum(1 for r in method_results if r.success) / len(method_results) * 100
            safety_rate = sum(1 for r in method_results if not r.safety_violation) / len(method_results) * 100
            print(f"Success: {success_rate:.0f}%, Safety: {safety_rate:.0f}%")

        self.results.extend(results)
        return results

    def run_all_scenarios(self, methods: List[Method] = None) -> Dict[str, List[EpisodeResult]]:
        """모든 시나리오 실행"""
        all_results = {}

        for scenario in self.scenarios:
            results = self.run_scenario(scenario, methods)
            all_results[scenario.name] = results

        return all_results

    def summarize_results(self) -> Dict[str, Dict[Method, ScenarioSummary]]:
        """결과 요약"""
        summaries = defaultdict(dict)

        for scenario in self.scenarios:
            for method in Method:
                method_results = [
                    r for r in self.results
                    if r.scenario == scenario.name and r.method == method
                ]

                if not method_results:
                    continue

                steps = [r.steps for r in method_results]

                summaries[scenario.name][method] = ScenarioSummary(
                    scenario=scenario.name,
                    method=method,
                    total_episodes=len(method_results),
                    success_count=sum(1 for r in method_results if r.success),
                    violation_count=sum(1 for r in method_results if r.safety_violation),
                    avg_steps=np.mean(steps),
                    std_steps=np.std(steps),
                    avg_blocked=np.mean([r.blocked_count for r in method_results]),
                    avg_min_distance=np.mean([r.min_distance for r in method_results]),
                )

        return summaries

    def print_summary_table(self):
        """요약 테이블 출력"""
        summaries = self.summarize_results()

        print("\n" + "=" * 100)
        print("UNIFIED S1-S6 COMPARISON EXPERIMENT RESULTS")
        print("=" * 100)

        # Header
        header = f"{'Scenario':<8} {'Method':<12} {'Success%':>10} {'Safety%':>10} {'Steps':>12} {'Blocked':>10} {'MinDist':>10}"
        print(header)
        print("-" * 100)

        for scenario_name in sorted(summaries.keys()):
            scenario_summaries = summaries[scenario_name]
            first = True

            for method in Method:
                if method not in scenario_summaries:
                    continue

                s = scenario_summaries[method]
                scenario_col = scenario_name if first else ""
                first = False

                print(f"{scenario_col:<8} {method.value:<12} {s.success_rate:>9.1f}% {s.safety_rate:>9.1f}% "
                      f"{s.avg_steps:>6.1f}±{s.std_steps:>4.1f} {s.avg_blocked:>10.1f} {s.avg_min_distance:>10.2f}")

            print("-" * 100)

        # Overall summary by method
        print("\n" + "=" * 100)
        print("OVERALL SUMMARY BY METHOD")
        print("=" * 100)

        for method in Method:
            method_results = [r for r in self.results if r.method == method]
            if not method_results:
                continue

            success_rate = sum(1 for r in method_results if r.success) / len(method_results) * 100
            safety_rate = sum(1 for r in method_results if not r.safety_violation) / len(method_results) * 100
            avg_steps = np.mean([r.steps for r in method_results])
            avg_blocked = np.mean([r.blocked_count for r in method_results])

            print(f"{method.value:<12}: Success={success_rate:>6.1f}%, Safety={safety_rate:>6.1f}%, "
                  f"Steps={avg_steps:>6.1f}, Blocked={avg_blocked:>5.1f}")

        print("=" * 100)

    def save_results(self, output_path: str = "unified_s1_s6_results.csv"):
        """결과를 CSV로 저장"""
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'scenario', 'method', 'seed', 'success', 'safety_violation',
                'steps', 'blocked_count', 'min_distance', 'termination_reason'
            ])

            for r in self.results:
                writer.writerow([
                    r.scenario, r.method.value, r.seed, r.success, r.safety_violation,
                    r.steps, r.blocked_count, r.min_distance, r.termination_reason
                ])

        print(f"\nResults saved to: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Unified S1-S6 Comparison Experiment")
    parser.add_argument('--scenarios', type=str, default='all',
                       help='Scenarios to run: all, S1, S2, S3, S4, S5, S6 (comma-separated)')
    parser.add_argument('--methods', type=str, default='all',
                       help='Methods: all, no_guard, safetychip, selp, geofence (comma-separated)')
    parser.add_argument('--seeds', type=int, default=None,
                       help='Override number of seeds per scenario')
    parser.add_argument('--output', type=str, default='unified_s1_s6_results.csv',
                       help='Output CSV file')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')

    args = parser.parse_args()

    # Parse methods
    if args.methods == 'all':
        methods = list(Method)
    else:
        method_map = {m.value: m for m in Method}
        methods = [method_map[m.strip()] for m in args.methods.split(',')]

    # Create runner
    runner = UnifiedExperimentRunner(verbose=args.verbose)

    # Override seeds if specified
    if args.seeds is not None:
        for scenario in runner.scenarios:
            scenario.num_seeds = args.seeds

    # Run experiments
    if args.scenarios == 'all':
        runner.run_all_scenarios(methods)
    else:
        scenario_names = [s.strip() for s in args.scenarios.split(',')]
        for scenario in runner.scenarios:
            if scenario.name in scenario_names:
                runner.run_scenario(scenario, methods)

    # Print and save results
    runner.print_summary_table()
    runner.save_results(args.output)


if __name__ == "__main__":
    main()
