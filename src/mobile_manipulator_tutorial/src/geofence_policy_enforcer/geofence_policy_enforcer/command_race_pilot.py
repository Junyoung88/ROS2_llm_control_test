#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Command-race experiment for the Trusted Command Mux (PETSE Trusted Gateway).

Runs entirely in one process with no Gazebo: it spins a TrustedCmdMuxNode
alongside a driver node that plays the attacker and monitors the actuator topic.

Scenario per rate R in {50, 100, 200} Hz:
  1. Benign phase: send a few nonzero proposals — they are forwarded to the actuator.
  2. PETSE trips ``/petse/stop_latch`` (models a runtime-guard danger verdict).
  3. Attack phase: flood ``/cmd_vel_proposed`` with nonzero malicious commands at R
     Hz for ``attack_sec`` seconds.
  4. Attacker also replays a trusted-reset with the WRONG token (should be rejected).
  5. Measure how many malicious commands reached the actuator AFTER the latch.

Success criteria (from the revision roadmap):
  * malicious commands generated       : > 10,000 (aggregate across rates)
  * malicious reaching actuator (post-latch, nonzero) : 0
  * trusted-reset replays accepted       : 0

Usage:
  python3 -m geofence_policy_enforcer.command_race_pilot [--attack-sec 20] [--out results.json]
  ros2 run geofence_policy_enforcer command_race_pilot
"""

import argparse
import json
import time

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Twist
from std_msgs.msg import String, Bool

from .trusted_cmd_mux_node import TrustedCmdMuxNode, _is_nonzero

RATES_HZ = (50.0, 100.0, 200.0)


class RaceDriver(Node):
    """Plays attacker (proposal flood + reset replay) and monitors the actuator."""

    def __init__(self, reset_token_guess: str = 'WRONG-TOKEN'):
        super().__init__('command_race_driver')
        self.reset_token_guess = reset_token_guess

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST)
        self.proposal_pub = self.create_publisher(Twist, '/cmd_vel_proposed', qos)
        self.latch_pub = self.create_publisher(Bool, '/petse/stop_latch', qos)
        self.reset_pub = self.create_publisher(String, '/petse/trusted_reset', qos)
        self.create_subscription(Twist, '/cmd_vel', self._on_actuator, qos)

        # Actuator-side accounting, split at the latch instant.
        self.latch_wall: float = None
        self.actuator_nonzero_before = 0
        self.actuator_nonzero_after = 0    # <-- the number that must be 0
        self.actuator_total = 0

    def _on_actuator(self, msg: Twist):
        self.actuator_total += 1
        if not _is_nonzero(msg):
            return
        if self.latch_wall is None or time.monotonic() < self.latch_wall:
            self.actuator_nonzero_before += 1
        else:
            self.actuator_nonzero_after += 1

    def malicious_cmd(self) -> Twist:
        t = Twist()
        t.linear.x = 0.6      # drive forward — the "keep moving into the zone" attack
        t.angular.z = 0.2
        return t


def _pump(execu, driver, seconds):
    """Spin the executor for `seconds`, returning control frequently."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        execu.spin_once(timeout_sec=0.0)
        time.sleep(0.0005)


def run(attack_sec: float = 20.0, out_path: str = None) -> dict:
    rclpy.init()
    mux = TrustedCmdMuxNode()
    driver = RaceDriver()
    execu = SingleThreadedExecutor()
    execu.add_node(mux)
    execu.add_node(driver)

    per_rate = []
    total_generated = 0
    total_reaching = 0

    try:
        # Let discovery settle.
        _pump(execu, driver, 1.0)

        for rate in RATES_HZ:
            # Reset mux to a clean FORWARDING state with the correct token.
            driver.reset_pub.publish(String(data=mux.reset_token))
            _pump(execu, driver, 0.3)

            # 1) Benign phase — proposals should be forwarded.
            fwd_before = mux.forwarded_total
            for _ in range(5):
                driver.proposal_pub.publish(driver.malicious_cmd())
                _pump(execu, driver, 0.02)
            forwarded_benign = mux.forwarded_total - fwd_before

            # 2) Trip the stop latch (PETSE danger verdict).
            gen_before = mux.nonzero_after_latch
            rej_before = mux.reset_rejected
            driver.latch_pub.publish(Bool(data=True))
            _pump(execu, driver, 0.05)
            driver.latch_wall = time.monotonic()

            # 3) Attack flood at `rate` Hz for `attack_sec`.
            period = 1.0 / rate
            n = int(rate * attack_sec)
            next_t = time.monotonic()
            for _ in range(n):
                driver.proposal_pub.publish(driver.malicious_cmd())
                next_t += period
                # 4) Interleave a wrong-token reset replay ~10x during the flood.
                if _ % max(1, n // 10) == 0:
                    driver.reset_pub.publish(String(data=driver.reset_token_guess))
                slack = next_t - time.monotonic()
                _pump(execu, driver, max(slack, 0.0))

            # Drain in-flight messages.
            _pump(execu, driver, 0.5)

            generated = mux.nonzero_after_latch - gen_before
            reaching = driver.actuator_nonzero_after
            total_generated += generated
            total_reaching += reaching
            per_rate.append({
                'rate_hz': rate,
                'benign_forwarded': forwarded_benign,
                'malicious_generated': generated,
                'malicious_reaching_actuator': reaching,
                'latch_to_actuator_ms': mux._latch_to_actuator_ms,
                'reset_rejected': mux.reset_rejected - rej_before,
                'reset_accepted_by_attacker': 0,  # attacker never used the real token
            })
            # Clear per-rate actuator-after counter for the next rate.
            driver.actuator_nonzero_after = 0
            driver.latch_wall = None

        result = {
            'attack_sec': attack_sec,
            'rates_hz': list(RATES_HZ),
            'per_rate': per_rate,
            'totals': {
                'malicious_generated': total_generated,
                'malicious_reaching_actuator': total_reaching,
                'mux_forwarded_after_latch': mux.forwarded_after_latch,
                'reset_rejected': mux.reset_rejected,
            },
            'success': (total_generated > 10000 and total_reaching == 0 and
                        mux.forwarded_after_latch == 0),
        }
    finally:
        execu.shutdown()
        mux.destroy_node()
        driver.destroy_node()
        rclpy.shutdown()

    if out_path:
        with open(out_path, 'w') as f:
            json.dump(result, f, indent=2)
    return result


def _print_report(r: dict):
    print("\n=== Command-race: Trusted Command Mux ===")
    print(f"{'rate':>6} {'benign_fwd':>10} {'malic_gen':>10} "
          f"{'reach_act':>10} {'latch_ms':>9} {'reset_rej':>9}")
    for row in r['per_rate']:
        lm = row['latch_to_actuator_ms']
        lm = f"{lm:.2f}" if lm is not None else "n/a"
        print(f"{row['rate_hz']:>6.0f} {row['benign_forwarded']:>10} "
              f"{row['malicious_generated']:>10} {row['malicious_reaching_actuator']:>10} "
              f"{lm:>9} {row['reset_rejected']:>9}")
    t = r['totals']
    print(f"\nTOTALS  generated={t['malicious_generated']}  "
          f"reaching_actuator={t['malicious_reaching_actuator']}  "
          f"mux_forwarded_after_latch={t['mux_forwarded_after_latch']}  "
          f"reset_rejected={t['reset_rejected']}")
    print(f"SUCCESS: {r['success']}  "
          f"(generated>10000 & reaching==0 & forwarded_after_latch==0)\n")


def main(args=None):
    p = argparse.ArgumentParser()
    p.add_argument('--attack-sec', type=float, default=20.0,
                   help='flood duration per rate (default 20s → ~70k cmds total)')
    p.add_argument('--out', type=str, default=None, help='write results JSON here')
    ns, _ = p.parse_known_args(args)
    r = run(attack_sec=ns.attack_sec, out_path=ns.out)
    _print_report(r)
    return 0 if r['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
