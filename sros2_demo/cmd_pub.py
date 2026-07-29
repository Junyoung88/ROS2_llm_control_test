#!/usr/bin/env python3
"""Minimal /cmd_vel publisher for the SROS2 exclusivity demo.

Publishes N nonzero Twists to /cmd_vel then exits. Run under the mux enclave
(allowed) or the attacker enclave (should be DENIED by DDS access control).
Node name is fixed so the SROS2 policy profile matches.
"""
import os
import sys
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class CmdPub(Node):
    def __init__(self, n):
        super().__init__('cmd_injector')
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.n = n
        self.i = 0
        self.create_timer(0.05, self._tick)

    def _tick(self):
        if self.i >= self.n:
            self.get_logger().info(f'published {self.i} cmd_vel messages, exiting')
            raise SystemExit
        t = Twist()
        t.linear.x = 0.6
        self.pub.publish(t)
        self.i += 1


def main():
    # N from env so ROS args (--enclave ...) can flow through argv to rclpy.
    n = int(os.environ.get('CMD_N', '40'))
    rclpy.init(args=sys.argv)
    node = CmdPub(n)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
