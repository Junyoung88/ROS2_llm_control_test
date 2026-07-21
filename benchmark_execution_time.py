#!/usr/bin/env python3
"""
Benchmark execution time for all safety methods.

Measures:
1. Goal evaluation (evaluate_point / evaluate): one-shot decision
2. cmd_vel guard (forward simulation / CBF / SSM): per-frame real-time check

Usage: python3 benchmark_execution_time.py
"""

import sys
import os
import time
import math
import statistics
import json

# Add the package to path
pkg_dir = os.path.join(
    os.path.dirname(__file__),
    'src/mobile_manipulator_tutorial/src/geofence_policy_enforcer'
)
sys.path.insert(0, pkg_dir)

from geofence_policy_enforcer.geofence_core import (
    GeofencePolicy, PolicyAction, ZoneType, UncertaintyParams
)
from geofence_policy_enforcer.safety_baselines import (
    SELPSpatialSafety, CBFSpatialSafety, StaticMarginSafety, SSMSpatialSafety,
    SafetyDecision
)

CONFIG_PATH = os.path.join(
    os.path.dirname(__file__),
    'src/mobile_manipulator_tutorial/src/geofence_policy_enforcer/config/geofence.yaml'
)

N_WARMUP = 1000
N_ITERATIONS = 10000


def benchmark(func, n_warmup=N_WARMUP, n_iter=N_ITERATIONS):
    """Run func n_iter times, return timing stats in microseconds."""
    # Warmup
    for _ in range(n_warmup):
        func()

    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        func()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)  # to microseconds

    return {
        'mean_us': statistics.mean(times),
        'median_us': statistics.median(times),
        'std_us': statistics.stdev(times),
        'min_us': min(times),
        'max_us': max(times),
        'p95_us': sorted(times)[int(0.95 * len(times))],
        'p99_us': sorted(times)[int(0.99 * len(times))],
        'n': n_iter,
    }


def fmt(stats):
    """Format stats as a one-liner."""
    return (f"mean={stats['mean_us']:.1f}μs, "
            f"median={stats['median_us']:.1f}μs, "
            f"p95={stats['p95_us']:.1f}μs, "
            f"p99={stats['p99_us']:.1f}μs, "
            f"std={stats['std_us']:.1f}μs")


def main():
    # Load geofence policy
    policy = GeofencePolicy(CONFIG_PATH)
    forbidden_zones = [z for z in policy.zones if z.zone_type == ZoneType.FORBIDDEN]
    zone_names = [z.name.replace(' ', '_').lower() for z in forbidden_zones]
    geofence_margin = policy.get_safety_margin()

    print(f"Config: {CONFIG_PATH}")
    print(f"Zones: {len(forbidden_zones)} forbidden, margin={geofence_margin:.3f}m")
    print(f"Iterations: {N_ITERATIONS} (warmup: {N_WARMUP})")
    print()

    # Initialize all safety checkers
    selp = SELPSpatialSafety(zone_names)
    cbf = CBFSpatialSafety(gamma=1.0, margin=0.1)
    cbf_inflated = CBFSpatialSafety(gamma=1.0, margin=geofence_margin)
    static_margin = StaticMarginSafety(margin=geofence_margin)
    ssm = SSMSpatialSafety(
        reaction_time=0.1, stopping_time=0.2,
        max_deceleration=1.0, intrusion_distance=0.05,
        base_margin=0.05
    )

    # Test points: representative positions from experiments
    test_points = {
        'safe_far':        (2.0, 0.0),     # Far from zone
        'near_boundary':   (3.85, 0.0),    # 0.15m from zone
        'mid_boundary':    (3.55, 0.0),    # 0.45m from zone (within margin)
        'inside_zone':     (5.0, 0.0),     # Inside forbidden zone
        'through_zone':    (8.0, 0.0),     # Through zone (path check)
    }

    # ============================================================
    # 1. GOAL EVALUATION BENCHMARK
    # ============================================================
    print("=" * 70)
    print("1. GOAL EVALUATION (one-shot decision per goal)")
    print("=" * 70)

    all_goal_results = {}

    for point_name, (px, py) in test_points.items():
        print(f"\n  Point: {point_name} ({px}, {py})")

        # Geofence
        stats = benchmark(lambda x=px, y=py: policy.evaluate_point(x, y))
        decision = policy.evaluate_point(px, py)
        print(f"    geofence:      {fmt(stats)}  → {decision.action.name}")
        all_goal_results[f'geofence_{point_name}'] = stats

        # SELP
        stats = benchmark(lambda p=(px, py): selp.evaluate(p, forbidden_zones))
        result = selp.evaluate((px, py), forbidden_zones)
        print(f"    selp_proper:   {fmt(stats)}  → {result.decision.name}")
        all_goal_results[f'selp_{point_name}'] = stats

        # CBF (small margin)
        stats = benchmark(lambda p=(px, py): cbf.evaluate(p, forbidden_zones))
        result = cbf.evaluate((px, py), forbidden_zones)
        print(f"    cbf:           {fmt(stats)}  → {result.decision.name}")
        all_goal_results[f'cbf_{point_name}'] = stats

        # CBF inflated
        stats = benchmark(lambda p=(px, py): cbf_inflated.evaluate(p, forbidden_zones))
        result = cbf_inflated.evaluate((px, py), forbidden_zones)
        print(f"    cbf_inflated:  {fmt(stats)}  → {result.decision.name}")
        all_goal_results[f'cbf_inflated_{point_name}'] = stats

        # SSM
        stats = benchmark(lambda p=(px, py): ssm.evaluate(p, forbidden_zones, velocity=0.5))
        result = ssm.evaluate((px, py), forbidden_zones, velocity=0.5)
        print(f"    ssm:           {fmt(stats)}  → {result.decision.name}")
        all_goal_results[f'ssm_{point_name}'] = stats

    # ============================================================
    # 2. CMD_VEL GUARD BENCHMARK (per-frame real-time check)
    # ============================================================
    print()
    print("=" * 70)
    print("2. CMD_VEL GUARD (per-frame, simulates 20Hz control loop)")
    print("=" * 70)

    # Simulate forward simulation (geofence guard)
    sim_horizon = 2.0
    sim_steps = 20
    dt = sim_horizon / sim_steps

    # Robot state: approaching zone at 0.5 m/s
    robot_x, robot_y, robot_yaw = 3.0, 0.0, 0.0
    vx, vy, wz = 0.5, 0.0, 0.0

    all_guard_results = {}

    def geofence_forward_sim():
        """Full forward simulation (geofence guard per-frame check)."""
        x, y, yaw = robot_x, robot_y, robot_yaw
        margin = policy.uncertainty.compute_margin(velocity=0.5)
        for step in range(sim_steps):
            x += (vx * math.cos(yaw) - vy * math.sin(yaw)) * dt
            y += (vx * math.sin(yaw) + vy * math.cos(yaw)) * dt
            yaw += wz * dt
            dist = policy.get_min_distance_to_forbidden(x, y)
            if dist < margin:
                return True  # Would violate
        return False

    stats = benchmark(geofence_forward_sim)
    print(f"\n  geofence (forward sim, {sim_steps} steps):  {fmt(stats)}")
    all_guard_results['geofence_forward_sim'] = stats

    # CBF per-frame check
    def cbf_guard_check():
        """CBF barrier check (per-frame)."""
        h, closest = cbf.compute_barrier((robot_x, robot_y), forbidden_zones)
        # Gradient + condition check
        min_dist = float('inf')
        for zone in forbidden_zones:
            cx = sum(v[0] for v in zone.vertices) / len(zone.vertices)
            cy = sum(v[1] for v in zone.vertices) / len(zone.vertices)
            dx, dy = robot_x - cx, robot_y - cy
            dist = math.sqrt(dx*dx + dy*dy)
            if dist < min_dist:
                min_dist = dist
                grad_h = (dx / max(dist, 1e-6), dy / max(dist, 1e-6))
        vx_w = vx * math.cos(robot_yaw)
        vy_w = vx * math.sin(robot_yaw)
        h_dot = grad_h[0] * vx_w + grad_h[1] * vy_w
        return h_dot + 1.0 * h >= 0

    stats = benchmark(cbf_guard_check)
    print(f"  cbf (barrier check):                     {fmt(stats)}")
    all_guard_results['cbf_guard'] = stats

    # CBF inflated per-frame
    def cbf_inflated_guard_check():
        """CBF-inflated barrier check (per-frame, with adaptive margin)."""
        margin = policy.uncertainty.compute_margin(velocity=0.5)
        cbf_inflated.margin = margin
        h, closest = cbf_inflated.compute_barrier((robot_x, robot_y), forbidden_zones)
        min_dist = float('inf')
        for zone in forbidden_zones:
            cx = sum(v[0] for v in zone.vertices) / len(zone.vertices)
            cy = sum(v[1] for v in zone.vertices) / len(zone.vertices)
            dx, dy = robot_x - cx, robot_y - cy
            dist = math.sqrt(dx*dx + dy*dy)
            if dist < min_dist:
                min_dist = dist
                grad_h = (dx / max(dist, 1e-6), dy / max(dist, 1e-6))
        vx_w = vx * math.cos(robot_yaw)
        vy_w = vx * math.sin(robot_yaw)
        h_dot = grad_h[0] * vx_w + grad_h[1] * vy_w
        return h_dot + 1.0 * h >= 0

    stats = benchmark(cbf_inflated_guard_check)
    print(f"  cbf_inflated (barrier + adaptive margin): {fmt(stats)}")
    all_guard_results['cbf_inflated_guard'] = stats

    # SSM per-frame
    def ssm_guard_check():
        """SSM check (per-frame): compare distance vs protective distance."""
        min_dist = policy.get_min_distance_to_forbidden(robot_x, robot_y)
        S = ssm.compute_protective_distance(0.5)
        return min_dist >= S

    stats = benchmark(ssm_guard_check)
    print(f"  ssm (distance vs protective margin):      {fmt(stats)}")
    all_guard_results['ssm_guard'] = stats

    # ============================================================
    # 3. MARGIN COMPUTATION ONLY
    # ============================================================
    print()
    print("=" * 70)
    print("3. MARGIN COMPUTATION (compute_margin only)")
    print("=" * 70)

    stats = benchmark(lambda: policy.uncertainty.compute_margin(velocity=0.5))
    print(f"\n  compute_margin(v=0.5):  {fmt(stats)}")
    all_guard_results['compute_margin'] = stats

    # ============================================================
    # 4. SUMMARY TABLE (for paper)
    # ============================================================
    print()
    print("=" * 70)
    print("SUMMARY FOR PAPER")
    print("=" * 70)

    # Average goal evaluation time across all test points
    methods = ['geofence', 'selp', 'cbf', 'cbf_inflated', 'ssm']
    print("\n  Goal Evaluation (mean across test points):")
    for method in methods:
        method_times = [all_goal_results[f'{method}_{pt}']['mean_us']
                        for pt in test_points.keys()
                        if f'{method}_{pt}' in all_goal_results]
        avg = statistics.mean(method_times)
        print(f"    {method:15s}: {avg:7.1f} μs ({avg/1000:.3f} ms)")

    print("\n  Guard (per-frame, single invocation):")
    guard_methods = [
        ('geofence', 'geofence_forward_sim'),
        ('cbf', 'cbf_guard'),
        ('cbf_inflated', 'cbf_inflated_guard'),
        ('ssm', 'ssm_guard'),
    ]
    for name, key in guard_methods:
        s = all_guard_results[key]
        print(f"    {name:15s}: {s['mean_us']:7.1f} μs ({s['mean_us']/1000:.3f} ms)")

    # Max frequency the guard can sustain
    print("\n  Max sustainable guard frequency:")
    for name, key in guard_methods:
        s = all_guard_results[key]
        max_hz = 1e6 / s['mean_us']
        print(f"    {name:15s}: {max_hz:,.0f} Hz")

    # Save results to JSON
    output = {
        'goal_evaluation': all_goal_results,
        'guard_per_frame': all_guard_results,
        'config': {
            'n_iterations': N_ITERATIONS,
            'n_warmup': N_WARMUP,
            'sim_steps': sim_steps,
            'sim_horizon': sim_horizon,
            'robot_position': [robot_x, robot_y],
            'velocity': vx,
        }
    }
    output_path = os.path.join(
        os.path.dirname(__file__),
        'experiment_results/gazebo_s1_s6/execution_time_benchmark.json'
    )
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {output_path}")


if __name__ == '__main__':
    main()
