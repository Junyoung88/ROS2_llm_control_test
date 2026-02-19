#!/usr/bin/env python3
"""
cmd_vel guard detection latency benchmark.

Measures the time from publishing a malicious cmd_vel to /cmd_vel_nav
until the guard's blocked response appears on /cmd_vel.

Two paths measured:
  1. Unapproved motion path  (nav_approved=False, fast flag check)
  2. Forward simulation path (nav_approved=True,  20-step sim)

Usage (guard must already be running):
  python3 measure_guard_latency.py [--n 100] [--output results_latency.json]

Prerequisites:
  - Guard node running:
      ros2 run geofence_policy_enforcer cmd_vel_guard_node \
        --ros-args \
        -p safety_method:=geofence \
        -p input_topic:=/cmd_vel_nav \
        -p output_topic:=/cmd_vel \
        -p odom_topic:=/odom
  - OR: run ./run_latency_benchmark.sh (starts guard + runs this script)
"""

import argparse
import json
import math
import time
import statistics
import sys
import threading
from typing import List, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String


# ─────────────────────────────────────────────────────────────────────────────
# Robot/zone setup matching actual experiment
# Zone: x=[4,6], y=[-1,1], margin=0.55m (sigma=0.15, v_max=0.5, tau=0.1)
# ─────────────────────────────────────────────────────────────────────────────
ROBOT_X_UNAPPROVED   = 1.0   # Far from zone — unapproved path doesn't do forward sim
ROBOT_X_FORWARDSIM   = 2.5   # Close enough: 0.5 m/s * 2s horizon reaches x=3.5 > margin boundary (3.45)
ROBOT_Y              = 0.0
ATTACK_VX            = 0.5   # m/s toward zone (+x direction)
ZONE_MARGIN          = 0.55  # Must match guard config


class LatencyBenchmark(Node):
    """Measures cmd_vel guard detection latency."""

    def __init__(self, n_samples: int = 100):
        super().__init__('guard_latency_benchmark')
        self.n_samples = n_samples

        qos_rel = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        qos_best = QoSProfile(depth=1,  reliability=ReliabilityPolicy.BEST_EFFORT)

        # Publishers
        self.odom_pub     = self.create_publisher(Odometry, '/odom',           qos_rel)
        self.cmd_nav_pub  = self.create_publisher(Twist,    '/cmd_vel_nav',     qos_rel)
        self.nav_state_pub = self.create_publisher(String,  '/geofence/nav_state', 10)

        # Subscriber — listens for guard output
        self.cmd_out_sub = self.create_subscription(
            Twist, '/cmd_vel', self._cmd_vel_out_callback, qos_rel
        )

        # State
        self._t_send: Optional[float] = None
        self._latencies: List[float]  = []
        self._recv_event               = threading.Event()
        self._last_blocked             = False  # True when guard published zero

        self.get_logger().info('LatencyBenchmark node initialized')

    # ──────────────────────────────────────────────────────────────
    # /cmd_vel output handler
    # ──────────────────────────────────────────────────────────────
    def _cmd_vel_out_callback(self, msg: Twist):
        if self._t_send is None:
            return

        is_zero = (abs(msg.linear.x) < 1e-6 and
                   abs(msg.linear.y) < 1e-6 and
                   abs(msg.angular.z) < 1e-6)

        if is_zero:
            t_recv = time.perf_counter()
            latency_ms = (t_recv - self._t_send) * 1000.0
            self._latencies.append(latency_ms)
            self._t_send = None
            self._recv_event.set()

    # ──────────────────────────────────────────────────────────────
    # Helpers: publish mock odom + nav_state
    # ──────────────────────────────────────────────────────────────
    def _publish_odom(self, x: float, y: float, vx: float = 0.0):
        msg = Odometry()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.pose.pose.position.x    = x
        msg.pose.pose.position.y    = y
        msg.pose.pose.orientation.w = 1.0
        msg.twist.twist.linear.x    = vx
        self.odom_pub.publish(msg)

    def _set_nav_approved(self, approved: bool):
        msg = String()
        msg.data = json.dumps({'state': 'active' if approved else 'idle'})
        self.nav_state_pub.publish(msg)

    def _publish_attack(self, vx: float = ATTACK_VX):
        cmd = Twist()
        cmd.linear.x = vx
        self._t_send = time.perf_counter()
        self.cmd_nav_pub.publish(cmd)

    def _spin_until_recv(self, timeout_s: float = 0.5) -> bool:
        """Spin ROS callbacks until guard response received or timeout."""
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            rclpy.spin_once(self, timeout_sec=0.002)
            if self._recv_event.is_set():
                self._recv_event.clear()
                return True
        self._t_send = None
        return False

    # ──────────────────────────────────────────────────────────────
    # Warmup
    # ──────────────────────────────────────────────────────────────
    def warmup(self, n: int = 10):
        self.get_logger().info(f'Warming up ({n} samples, discarded)...')
        saved = self._latencies.copy()
        self._latencies.clear()

        for _ in range(n):
            self._publish_odom(ROBOT_X_UNAPPROVED, ROBOT_Y)
            self._set_nav_approved(False)
            time.sleep(0.05)
            self._publish_attack()
            self._spin_until_recv(timeout_s=0.3)
            time.sleep(0.05)
            # Reset guard: allow a zero-velocity pass-through to clear state
            cmd_zero = Twist()
            self.cmd_nav_pub.publish(cmd_zero)
            rclpy.spin_once(self, timeout_sec=0.01)

        self._latencies = saved
        self.get_logger().info('Warmup done')

    # ──────────────────────────────────────────────────────────────
    # Measurement: unapproved motion path
    # ──────────────────────────────────────────────────────────────
    def measure_unapproved(self) -> List[float]:
        """
        Path: nav_approved=False + non-zero cmd_vel → immediate block.
        Guard code: lines 301-318 in cmd_vel_guard_node.py
        """
        self.get_logger().info(
            f'[Unapproved] Measuring {self.n_samples} samples '
            f'(robot @({ROBOT_X_UNAPPROVED},{ROBOT_Y}), vx={ATTACK_VX})'
        )
        latencies = []
        failed = 0

        for i in range(self.n_samples):
            # Set state: odom + nav NOT approved
            self._publish_odom(ROBOT_X_UNAPPROVED, ROBOT_Y)
            self._set_nav_approved(False)
            rclpy.spin_once(self, timeout_sec=0.02)
            time.sleep(0.02)  # Let guard process odom + nav_state

            # Send attack
            self._publish_attack()
            ok = self._spin_until_recv(timeout_s=0.3)

            if ok and self._latencies:
                latencies.append(self._latencies[-1])
            else:
                failed += 1

            # Brief pause between samples
            time.sleep(0.03)

            if (i + 1) % 20 == 0:
                self.get_logger().info(f'  [{i+1}/{self.n_samples}] ...')

        if failed:
            self.get_logger().warning(f'  {failed} samples timed out (dropped)')
        return latencies

    # ──────────────────────────────────────────────────────────────
    # Measurement: forward simulation path
    # ──────────────────────────────────────────────────────────────
    def measure_forward_sim(self) -> List[float]:
        """
        Path: nav_approved=True + cmd_vel toward zone → forward sim (20 steps) → block.
        Guard code: _process_forward_simulation(), lines 555-659

        Robot at (2.5, 0.0), vx=0.5 m/s:
          - 2s horizon, 20 steps (dt=0.1s)
          - After 9 steps: x ≈ 2.5 + 9*0.5*0.1 = 2.95m → dist=1.05m > margin(0.55)
          - After ~19 steps: x ≈ 3.45m → hits margin → block
        """
        self.get_logger().info(
            f'[ForwardSim] Measuring {self.n_samples} samples '
            f'(robot @({ROBOT_X_FORWARDSIM},{ROBOT_Y}), vx={ATTACK_VX}, margin={ZONE_MARGIN}m)'
        )
        latencies = []
        failed = 0

        for i in range(self.n_samples):
            # Set state: nav APPROVED (goal_gate authorized navigation)
            self._publish_odom(ROBOT_X_FORWARDSIM, ROBOT_Y)
            self._set_nav_approved(True)
            rclpy.spin_once(self, timeout_sec=0.02)
            time.sleep(0.02)

            # Send attack
            self._publish_attack()
            ok = self._spin_until_recv(timeout_s=0.5)

            if ok and self._latencies:
                latencies.append(self._latencies[-1])
            else:
                failed += 1

            time.sleep(0.03)

            if (i + 1) % 20 == 0:
                self.get_logger().info(f'  [{i+1}/{self.n_samples}] ...')

        if failed:
            self.get_logger().warning(f'  {failed} samples timed out (dropped)')
        return latencies


# ─────────────────────────────────────────────────────────────────────────────
# Statistics helper
# ─────────────────────────────────────────────────────────────────────────────
def compute_stats(name: str, data: List[float]) -> dict:
    if not data:
        print(f'[{name}] No data collected!')
        return {}

    data_sorted = sorted(data)
    n = len(data_sorted)
    mean  = statistics.mean(data_sorted)
    stdev = statistics.stdev(data_sorted) if n > 1 else 0.0
    med   = statistics.median(data_sorted)
    mn    = data_sorted[0]
    mx    = data_sorted[-1]
    p95   = data_sorted[int(n * 0.95)]
    p99   = data_sorted[min(int(n * 0.99), n - 1)]

    print(f'\n{"="*55}')
    print(f'  {name}')
    print(f'{"="*55}')
    print(f'  n          : {n}')
    print(f'  mean ± std : {mean:.3f} ± {stdev:.3f} ms')
    print(f'  median     : {med:.3f} ms')
    print(f'  min / max  : {mn:.3f} / {mx:.3f} ms')
    print(f'  P95 / P99  : {p95:.3f} / {p99:.3f} ms')
    print(f'{"="*55}')

    return {
        'name': name,
        'n': n,
        'mean_ms': round(mean, 3),
        'std_ms':  round(stdev, 3),
        'median_ms': round(med, 3),
        'min_ms':  round(mn, 3),
        'max_ms':  round(mx, 3),
        'p95_ms':  round(p95, 3),
        'p99_ms':  round(p99, 3),
        'raw_ms':  [round(v, 3) for v in data_sorted],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='cmd_vel guard detection latency benchmark')
    parser.add_argument('--n',       type=int, default=100,
                        help='Number of samples per test (default: 100)')
    parser.add_argument('--warmup',  type=int, default=10,
                        help='Warmup samples (discarded, default: 10)')
    parser.add_argument('--output',  type=str,
                        default='experiment_results/gazebo_s1_s6/guard_latency.json',
                        help='Output JSON file')
    parser.add_argument('--skip-forwardsim', action='store_true',
                        help='Skip forward simulation measurement (faster)')
    args = parser.parse_args()

    rclpy.init()
    node = LatencyBenchmark(n_samples=args.n)

    print('\n' + '='*55)
    print('  cmd_vel Guard Detection Latency Benchmark')
    print('='*55)
    print(f'  samples per test : {args.n}')
    print(f'  warmup samples   : {args.warmup}')
    print(f'  output file      : {args.output}')
    print('='*55 + '\n')

    # Wait for guard to be ready (odom needed)
    print('Waiting for guard node to be ready (sending odom for 2s)...')
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 2.0:
        node._publish_odom(ROBOT_X_UNAPPROVED, ROBOT_Y)
        rclpy.spin_once(node, timeout_sec=0.05)

    # Warmup
    node.warmup(n=args.warmup)

    results = {}

    # ── Test 1: Unapproved motion path ──
    lats_unapp = node.measure_unapproved()
    results['unapproved_motion'] = compute_stats(
        'Unapproved Motion Detection (flag check path)', lats_unapp
    )

    # ── Test 2: Forward simulation path ──
    if not args.skip_forwardsim:
        print('\nResetting nav_state to False for 0.5s before forward sim test...')
        node._set_nav_approved(False)
        time.sleep(0.5)

        lats_fwd = node.measure_forward_sim()
        results['forward_simulation'] = compute_stats(
            'Forward Simulation Blocking (20-step sim path)', lats_fwd
        )

    # ── Summary comparison ──
    print('\n' + '='*55)
    print('  SUMMARY (for paper)')
    print('='*55)
    for key, stat in results.items():
        if stat:
            print(f'  {key}:')
            print(f'    {stat["mean_ms"]:.2f} ± {stat["std_ms"]:.2f} ms  '
                  f'(median={stat["median_ms"]:.2f}, P99={stat["p99_ms"]:.2f})')
    print('='*55)

    # ── Save JSON ──
    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump({
            'timestamp':   time.strftime('%Y-%m-%dT%H:%M:%S'),
            'n_samples':   args.n,
            'robot_pos_unapproved':  [ROBOT_X_UNAPPROVED, ROBOT_Y],
            'robot_pos_forwardsim':  [ROBOT_X_FORWARDSIM, ROBOT_Y],
            'attack_vx':    ATTACK_VX,
            'zone_margin':  ZONE_MARGIN,
            'results':      results,
        }, f, indent=2)
    print(f'\nResults saved → {args.output}')

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
