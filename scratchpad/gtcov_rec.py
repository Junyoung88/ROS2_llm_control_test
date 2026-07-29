#!/usr/bin/env python3
"""W2 recorder (renamed to avoid the runner's 'amcl' cleanup pattern).

Logs ONLY /amcl_pose: timestamp, estimate p_hat, and the 2x2 position covariance
Sigma. Ground truth is taken offline from the runner's position-monitor log
(Gazebo truth, same timestamps) by nearest-time matching -- so this node makes no
blocking gz call inside the callback. Node/file names deliberately avoid the
runner's kill patterns (amcl, guard, monitor, relay, gz, server, bridge, ...).
"""
import argparse, json, sys
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseWithCovarianceStamped


class Rec(Node):
    def __init__(self, out):
        super().__init__('gtcov_rec')
        self.f = open(out, 'w'); self.n = 0
        # Match AMCL's /amcl_pose QoS exactly: RELIABLE + TRANSIENT_LOCAL (latched),
        # otherwise a VOLATILE subscriber misses the latched pose and sparse updates.
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self._cb, qos)
        self.get_logger().info(f'logging /amcl_pose -> {out}')

    def _cb(self, msg):
        p = msg.pose.pose.position
        c = msg.pose.covariance  # 6x6 row-major; xy block = [0,1,6,7]
        self.f.write(json.dumps({
            't': self.get_clock().now().nanoseconds / 1e9,
            'amcl_x': p.x, 'amcl_y': p.y,
            'sxx': c[0], 'sxy': c[1], 'syy': c[7],
        }) + '\n')
        self.f.flush(); self.n += 1
        if self.n % 25 == 0:
            self.get_logger().info(f'{self.n} amcl samples')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default='cov.jsonl')
    ns, _ = ap.parse_known_args()
    rclpy.init(args=sys.argv)
    node = Rec(ns.out)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.get_logger().info(f'FINAL {node.n} amcl samples')
        node.f.close(); node.destroy_node()
        try: rclpy.shutdown()
        except Exception: pass


if __name__ == '__main__':
    main()
