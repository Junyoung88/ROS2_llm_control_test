#!/usr/bin/env python3
"""
Scan Relay Node
===============

Relay scan_real to scan for normal (non-attack) operation.
Used when gz_bridge publishes to scan_real instead of scan.

Usage:
    ros2 run geofence_policy_enforcer scan_relay
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


class ScanRelay(Node):
    """Simple relay from scan_real to scan"""

    def __init__(self):
        super().__init__('scan_relay')

        self.declare_parameter('input_topic', '/scan_real')
        self.declare_parameter('output_topic', '/scan')

        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value

        self.pub = self.create_publisher(LaserScan, output_topic, 10)
        self.sub = self.create_subscription(
            LaserScan, input_topic, self.callback, 10)

        self.get_logger().info(f'Scan relay: {input_topic} -> {output_topic}')

    def callback(self, msg: LaserScan):
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ScanRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
