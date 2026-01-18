#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attack Pose Spoofing - Simulates GPS/localization spoofing attack.

This attack node demonstrates how an attacker might try to bypass geofencing
by spoofing the robot's perceived position, making it think it's farther
from forbidden zones than it actually is.

Attack strategy:
- Intercept localization pose (e.g., /amcl_pose)
- Add an offset to make robot think it's in a different location
- Robot's safety system sees wrong position, allowing dangerous navigation
"""

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String


class PoseSpoofingAttackNode(Node):
    """
    Simulates GPS/localization spoofing attack.

    Strategy: Intercept pose messages and add offset to make robot
    think it's farther from forbidden zone than it actually is.
    """

    def __init__(self):
        super().__init__('attack_pose_spoofing')

        # Parameters
        self.declare_parameter('spoof_offset_x', 3.0)  # meters to add to perceived x
        self.declare_parameter('spoof_offset_y', 0.0)  # meters to add to perceived y
        self.declare_parameter('attack_duration', 60.0)  # seconds
        self.declare_parameter('target_zone_x', -4.7)  # forbidden zone center
        self.declare_parameter('target_zone_y', 5.6)

        self.spoof_offset_x = self.get_parameter('spoof_offset_x').get_parameter_value().double_value
        self.spoof_offset_y = self.get_parameter('spoof_offset_y').get_parameter_value().double_value
        self.attack_duration = self.get_parameter('attack_duration').get_parameter_value().double_value
        self.target_zone_x = self.get_parameter('target_zone_x').get_parameter_value().double_value
        self.target_zone_y = self.get_parameter('target_zone_y').get_parameter_value().double_value

        # QoS
        qos = QoSProfile(depth=10)

        # Subscribe to real pose sources
        self.amcl_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.amcl_callback,
            qos
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        # Publish spoofed poses
        # These would override the real pose in the safety system
        self.spoofed_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            '/spoofed_amcl_pose',
            qos
        )

        self.spoofed_odom_pub = self.create_publisher(
            Odometry,
            '/spoofed_odom',
            10
        )

        # Episode control
        self.episode_pub = self.create_publisher(String, '/geofence/episode_control', 10)

        # Attack state
        self.attack_active = False
        self.attack_start_time = None
        self.real_x = 0.0
        self.real_y = 0.0
        self.messages_spoofed = 0

        self.get_logger().info('Pose Spoofing Attack node initialized')
        self.get_logger().info(f'Spoof offset: ({self.spoof_offset_x}, {self.spoof_offset_y})')
        self.get_logger().info(f'Attack duration: {self.attack_duration}s')

        # Start attack after delay
        self.create_timer(2.0, self.start_attack)

    def start_attack(self):
        """Start the spoofing attack."""
        if self.attack_active:
            return

        self.attack_active = True
        self.attack_start_time = time.time()

        self.get_logger().info('=' * 50)
        self.get_logger().info('STARTING POSE SPOOFING ATTACK')
        self.get_logger().info('=' * 50)

        # Signal episode start
        msg = String()
        msg.data = 'start'
        self.episode_pub.publish(msg)

        # Create check timer
        self.create_timer(1.0, self.check_attack_duration)

    def check_attack_duration(self):
        """Check if attack duration has elapsed."""
        if not self.attack_active:
            return

        elapsed = time.time() - self.attack_start_time
        if elapsed > self.attack_duration:
            self.finish_attack()

    def amcl_callback(self, msg: PoseWithCovarianceStamped):
        """Intercept AMCL pose and publish spoofed version."""
        if not self.attack_active:
            return

        self.real_x = msg.pose.pose.position.x
        self.real_y = msg.pose.pose.position.y

        # Create spoofed message
        spoofed = PoseWithCovarianceStamped()
        spoofed.header = msg.header
        spoofed.pose = msg.pose

        # Add spoofing offset - make robot think it's farther from zone
        spoofed.pose.pose.position.x += self.spoof_offset_x
        spoofed.pose.pose.position.y += self.spoof_offset_y

        self.spoofed_pose_pub.publish(spoofed)
        self.messages_spoofed += 1

        # Log periodically
        if self.messages_spoofed % 10 == 0:
            real_dist = math.sqrt(
                (self.real_x - self.target_zone_x)**2 +
                (self.real_y - self.target_zone_y)**2
            )
            perceived_dist = math.sqrt(
                (spoofed.pose.pose.position.x - self.target_zone_x)**2 +
                (spoofed.pose.pose.position.y - self.target_zone_y)**2
            )
            self.get_logger().info(
                f'Spoofing: real_dist={real_dist:.2f}m, '
                f'perceived_dist={perceived_dist:.2f}m, '
                f'diff={perceived_dist - real_dist:.2f}m'
            )

    def odom_callback(self, msg: Odometry):
        """Intercept odometry and publish spoofed version."""
        if not self.attack_active:
            return

        # Create spoofed message
        spoofed = Odometry()
        spoofed.header = msg.header
        spoofed.child_frame_id = msg.child_frame_id
        spoofed.pose = msg.pose
        spoofed.twist = msg.twist

        # Add spoofing offset
        spoofed.pose.pose.position.x += self.spoof_offset_x
        spoofed.pose.pose.position.y += self.spoof_offset_y

        self.spoofed_odom_pub.publish(spoofed)

    def finish_attack(self):
        """Finish attack and report results."""
        self.attack_active = False
        elapsed = time.time() - self.attack_start_time

        self.get_logger().info('=' * 50)
        self.get_logger().info('POSE SPOOFING ATTACK COMPLETE')
        self.get_logger().info('=' * 50)
        self.get_logger().info(f'Duration: {elapsed:.1f}s')
        self.get_logger().info(f'Messages spoofed: {self.messages_spoofed}')
        self.get_logger().info(f'Final real position: ({self.real_x:.2f}, {self.real_y:.2f})')

        # Calculate final distances
        real_dist = math.sqrt(
            (self.real_x - self.target_zone_x)**2 +
            (self.real_y - self.target_zone_y)**2
        )
        perceived_dist = math.sqrt(
            (self.real_x + self.spoof_offset_x - self.target_zone_x)**2 +
            (self.real_y + self.spoof_offset_y - self.target_zone_y)**2
        )

        self.get_logger().info(f'Real distance to zone: {real_dist:.2f}m')
        self.get_logger().info(f'Perceived distance to zone: {perceived_dist:.2f}m')

        if real_dist < 0.5:
            self.get_logger().warn('DANGER: Robot very close to or inside forbidden zone!')
        elif real_dist < 1.0:
            self.get_logger().warn('WARNING: Robot approaching forbidden zone!')

        # Signal episode stop
        msg = String()
        msg.data = 'stop'
        self.episode_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PoseSpoofingAttackNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Attack interrupted')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
