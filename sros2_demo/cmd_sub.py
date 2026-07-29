#!/usr/bin/env python3
"""Minimal /cmd_vel subscriber for the SROS2 exclusivity demo.

Counts nonzero /cmd_vel messages actually delivered. Under SROS2 Enforce, a
publisher whose enclave lacks publish permission on /cmd_vel is rejected at the
DDS layer, so its messages never reach this subscriber. Prints the count on exit
(SIGTERM/SIGINT) so the harness can diff allowed vs denied publishers.
"""
import sys
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class CmdSub(Node):
    def __init__(self):
        super().__init__('cmd_monitor')
        self.create_subscription(Twist, '/cmd_vel', self._cb, 10)
        self.count = 0

    def _cb(self, msg: Twist):
        if abs(msg.linear.x) > 1e-6 or abs(msg.angular.z) > 1e-6:
            self.count += 1
            if self.count % 10 == 0:
                self.get_logger().info(f'received {self.count} nonzero /cmd_vel')


def main():
    rclpy.init(args=sys.argv)
    node = CmdSub()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.get_logger().info(f'FINAL received nonzero /cmd_vel = {node.count}')
        print(f'MONITOR_FINAL_COUNT={node.count}', flush=True)
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
