#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attack Velocity Amplify - Simulates velocity manipulation during navigation.

This attack node demonstrates S5 scenario where an attacker intercepts
cmd_vel commands during navigation and amplifies them, causing the robot
to overshoot its intended stopping point and enter forbidden zones.

Attack strategy:
- Intercept cmd_vel from Nav2 (subscribe to /cmd_vel_nav2)
- Multiply linear velocity by amplification factor
- Republish to /cmd_vel (actual robot control)
- Robot stops late, overshooting into forbidden zone
"""

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Float32


class VelocityAmplifyAttackNode(Node):
    """
    Simulates velocity manipulation attack by amplifying cmd_vel.

    Strategy: Intercept velocity commands and multiply them, causing
    the robot to overshoot its intended destination.
    """

    def __init__(self):
        super().__init__('attack_velocity_amplify')

        # Parameters
        self.declare_parameter('amplification_factor', 2.0)  # 2x velocity
        self.declare_parameter('attack_duration', 60.0)  # seconds
        self.declare_parameter('target_zone_x', -4.7)  # forbidden zone center
        self.declare_parameter('target_zone_y', 5.6)
        self.declare_parameter('activation_distance', 3.0)  # Activate when within this distance

        self.amp_factor = self.get_parameter('amplification_factor').get_parameter_value().double_value
        self.attack_duration = self.get_parameter('attack_duration').get_parameter_value().double_value
        self.target_zone_x = self.get_parameter('target_zone_x').get_parameter_value().double_value
        self.target_zone_y = self.get_parameter('target_zone_y').get_parameter_value().double_value
        self.activation_distance = self.get_parameter('activation_distance').get_parameter_value().double_value

        # QoS for velocity commands
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)

        # Subscribe to original cmd_vel from Nav2
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            '/cmd_vel_nav2',  # Nav2's output (remapped)
            self.cmd_vel_callback,
            qos
        )

        # Publish amplified cmd_vel to robot
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel',  # Actual robot control
            qos
        )

        # Subscribe to odometry for position tracking
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        # Publish attack status
        self.status_pub = self.create_publisher(String, '/attack/velocity_status', 10)
        self.factor_pub = self.create_publisher(Float32, '/attack/applied_factor', 10)

        # Episode control
        self.episode_pub = self.create_publisher(String, '/geofence/episode_control', 10)

        # Attack state
        self.attack_active = False
        self.attack_start_time = None
        self.amplification_active = False
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.messages_amplified = 0
        self.messages_passed = 0

        self.get_logger().info('Velocity Amplify Attack node initialized')
        self.get_logger().info(f'Amplification factor: {self.amp_factor}x')
        self.get_logger().info(f'Activation distance: {self.activation_distance}m from zone')

        # Start attack after delay
        self.create_timer(2.0, self.start_attack)

    def odom_callback(self, msg: Odometry):
        """Update robot position from odometry."""
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y

        # Check if close enough to target zone to activate amplification
        dist_to_zone = math.sqrt(
            (self.robot_x - self.target_zone_x)**2 +
            (self.robot_y - self.target_zone_y)**2
        )

        # Activate amplification when approaching zone
        if self.attack_active and dist_to_zone < self.activation_distance:
            if not self.amplification_active:
                self.amplification_active = True
                self.get_logger().warn(
                    f'AMPLIFICATION ACTIVATED - Distance to zone: {dist_to_zone:.2f}m'
                )

    def start_attack(self):
        """Start the velocity amplification attack."""
        if self.attack_active:
            return

        self.attack_active = True
        self.attack_start_time = time.time()

        self.get_logger().info('=' * 50)
        self.get_logger().info('STARTING VELOCITY AMPLIFICATION ATTACK')
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

    def cmd_vel_callback(self, msg: Twist):
        """Intercept cmd_vel and optionally amplify."""
        if not self.attack_active:
            # Pass through unchanged
            self.cmd_vel_pub.publish(msg)
            return

        output = Twist()

        if self.amplification_active:
            # Amplify linear velocities
            output.linear.x = msg.linear.x * self.amp_factor
            output.linear.y = msg.linear.y * self.amp_factor
            output.linear.z = msg.linear.z

            # Keep angular velocity unchanged (amplifying would cause spinning)
            output.angular = msg.angular

            self.messages_amplified += 1

            # Publish applied factor for monitoring
            factor_msg = Float32()
            factor_msg.data = self.amp_factor
            self.factor_pub.publish(factor_msg)
        else:
            # Pass through unchanged until activation
            output = msg
            self.messages_passed += 1

            factor_msg = Float32()
            factor_msg.data = 1.0
            self.factor_pub.publish(factor_msg)

        self.cmd_vel_pub.publish(output)

        # Log periodically
        total = self.messages_amplified + self.messages_passed
        if total % 50 == 0 and total > 0:
            dist = math.sqrt(
                (self.robot_x - self.target_zone_x)**2 +
                (self.robot_y - self.target_zone_y)**2
            )
            self.get_logger().info(
                f'Attack: pos=({self.robot_x:.2f}, {self.robot_y:.2f}), '
                f'dist={dist:.2f}m, amp_active={self.amplification_active}, '
                f'amplified={self.messages_amplified}'
            )

    def finish_attack(self):
        """Finish attack and report results."""
        self.attack_active = False
        elapsed = time.time() - self.attack_start_time

        self.get_logger().info('=' * 50)
        self.get_logger().info('VELOCITY AMPLIFICATION ATTACK COMPLETE')
        self.get_logger().info('=' * 50)
        self.get_logger().info(f'Duration: {elapsed:.1f}s')
        self.get_logger().info(f'Messages amplified: {self.messages_amplified}')
        self.get_logger().info(f'Messages passed: {self.messages_passed}')
        self.get_logger().info(f'Final position: ({self.robot_x:.2f}, {self.robot_y:.2f})')

        dist_to_zone = math.sqrt(
            (self.robot_x - self.target_zone_x)**2 +
            (self.robot_y - self.target_zone_y)**2
        )
        self.get_logger().info(f'Final distance to zone: {dist_to_zone:.2f}m')

        if dist_to_zone < 0.5:
            self.get_logger().warn('DANGER: Robot very close to or inside forbidden zone!')

        # Signal episode stop
        msg = String()
        msg.data = 'stop'
        self.episode_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = VelocityAmplifyAttackNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Attack interrupted')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
