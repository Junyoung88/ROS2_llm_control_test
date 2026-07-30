#!/usr/bin/env python3
"""R2: in-situ command race against the Trusted Command Mux, in Gazebo.

When PETSE trips /petse/stop_latch (spoof fail-stop), this node immediately
floods /cmd_vel_proposed with nonzero commands at 100 Hz -- an attacker trying to
override the stop through the legitimate proposal channel -- and counts how many
nonzero commands actually reach the actuator topic /cmd_vel afterwards. With the
mux latched, the answer should be zero, and the real base should hold its stop
(stopping clearance comes from the trial's position monitor).

Named to avoid the runner's cleanup patterns (no amcl/guard/monitor/gz/relay).
"""
import json, sys, argparse
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import json as _json


class CmdRace(Node):
    def __init__(self, out, flood_hz, dur):
        super().__init__('cmdrace_gz')
        self.out = out; self.dur = dur
        self.latched = False
        self.t_latch = None
        self.flooded = 0
        self.nonzero_actuator_after = 0
        self.actuator_total_after = 0
        q = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                       history=HistoryPolicy.KEEP_LAST)
        self.pub = self.create_publisher(Twist, '/cmd_vel_proposed', q)
        # Detect the latch via the mux's periodic metrics (state == 'latched'),
        # not the one-shot /petse/stop_latch edge which a subscriber can miss.
        self.create_subscription(String, '/petse/mux_metrics', self._metrics, q)
        self.create_subscription(Twist, '/cmd_vel', self._act, q)
        self.create_timer(1.0/flood_hz, self._flood)
        self.get_logger().info('cmdrace armed: floods /cmd_vel_proposed when mux latches')

    def _now(self):
        return self.get_clock().now().nanoseconds/1e9

    def _metrics(self, msg):
        if self.latched:
            return
        try:
            m = _json.loads(msg.data)
        except Exception:
            return
        if m.get('state') == 'latched':
            self.latched = True; self.t_latch = self._now()
            self.get_logger().warn('MUX LATCHED (via metrics) -> begin command flood')

    def _flood(self):
        if not self.latched:
            return
        if self._now() - self.t_latch > self.dur:
            return
        t = Twist(); t.linear.x = 0.6; t.angular.z = 0.2
        self.pub.publish(t); self.flooded += 1

    def _act(self, msg):
        if not self.latched:
            return
        self.actuator_total_after += 1
        if abs(msg.linear.x) > 1e-6 or abs(msg.angular.z) > 1e-6:
            self.nonzero_actuator_after += 1

    def summary(self):
        return {
            'latched': self.latched,
            'malicious_flooded_after_latch': self.flooded,
            'nonzero_reaching_actuator_after_latch': self.nonzero_actuator_after,
            'actuator_msgs_after_latch': self.actuator_total_after,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='cmdrace.json')
    ap.add_argument('--flood-hz', type=float, default=100.0)
    ap.add_argument('--dur', type=float, default=20.0)
    ns, _ = ap.parse_known_args()
    rclpy.init(args=sys.argv)
    node = CmdRace(ns.out, ns.flood_hz, ns.dur)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        s = node.summary()
        json.dump(s, open(ns.out, 'w'), indent=2)
        node.get_logger().info(f'FINAL {s}')
        node.destroy_node()
        try: rclpy.shutdown()
        except Exception: pass


if __name__ == '__main__':
    main()
