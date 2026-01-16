#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attack Cmd Vel - Simulates direct velocity command steering attack.

This attack node demonstrates how an attacker might try to bypass geofencing
by publishing small velocity commands that would gradually drift the robot
into a forbidden zone.

This is used to test and validate the Control PEP (cmd_vel_guard_node).
"""

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import Twist
from std_msgs.msg import String
from nav_msgs.msg import Odometry


class CmdVelAttackNode(Node):
    """
    Simulates velocity steering attack toward forbidden zone.

    Strategy: Publish small velocity commands directed toward the
    forbidden zone, simulating an attacker who has gained access
    to the velocity command topic.
    """

    def __init__(self):
        super().__init__('attack_cmd_vel')

        # Parameters
        self.declare_parameter('target_x', 3.0)  # Forbidden zone center
        self.declare_parameter('target_y', 3.0)
        self.declare_parameter('attack_velocity', 0.3)  # m/s
        self.declare_parameter('attack_duration', 30.0)  # seconds
        self.declare_parameter('publish_rate', 10.0)  # Hz
        self.declare_parameter('use_guarded_topic', True)  # Use cmd_vel_raw (guarded) or cmd_vel

        self.target_x = self.get_parameter('target_x').get_parameter_value().double_value
        self.target_y = self.get_parameter('target_y').get_parameter_value().double_value
        self.attack_velocity = self.get_parameter('attack_velocity').get_parameter_value().double_value
        self.attack_duration = self.get_parameter('attack_duration').get_parameter_value().double_value
        self.publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value
        self.use_guarded = self.get_parameter('use_guarded_topic').get_parameter_value().bool_value

        # Robot position
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.position_received = False

        # QoS for velocity commands
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)

        # Publisher - either to guarded topic or directly to cmd_vel
        topic = '/cmd_vel_raw' if self.use_guarded else '/cmd_vel'
        self.cmd_pub = self.create_publisher(Twist, topic, qos)

        # Odometry subscriber
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10
        )

        # Episode control publisher
        self.episode_pub = self.create_publisher(String, '/geofence/episode_control', 10)

        # Attack state
        self.attack_started = False
        self.attack_start_time = None
        self.start_x = 0.0
        self.start_y = 0.0

        # Statistics
        self.commands_sent = 0
        self.min_distance_to_target = float('inf')

        self.get_logger().info('Cmd Vel Attack node initialized')
        self.get_logger().info(f'Target: ({self.target_x}, {self.target_y})')
        self.get_logger().info(f'Attack velocity: {self.attack_velocity} m/s')
        self.get_logger().info(f'Duration: {self.attack_duration} s')
        self.get_logger().info(f'Publishing to: {topic}')

        # Start attack after short delay
        self.create_timer(2.0, self.start_attack)

    def odom_callback(self, msg: Odometry):
        """Update robot position from odometry."""
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = math.atan2(siny_cosp, cosy_cosp)

        self.position_received = True

        # Track minimum distance
        dist = math.sqrt((self.target_x - self.robot_x)**2 + (self.target_y - self.robot_y)**2)
        if dist < self.min_distance_to_target:
            self.min_distance_to_target = dist

    def start_attack(self):
        """Start the velocity command attack."""
        if self.attack_started:
            return
        self.attack_started = True

        self.get_logger().info('=' * 50)
        self.get_logger().info('STARTING CMD_VEL STEERING ATTACK')
        self.get_logger().info('=' * 50)

        # Signal episode start
        msg = String()
        msg.data = 'start'
        self.episode_pub.publish(msg)

        # Wait for position
        while not self.position_received and rclpy.ok():
            self.get_logger().info('Waiting for odometry...')
            rclpy.spin_once(self, timeout_sec=1.0)

        self.start_x = self.robot_x
        self.start_y = self.robot_y
        self.attack_start_time = time.time()

        self.get_logger().info(f'Starting position: ({self.robot_x:.2f}, {self.robot_y:.2f})')

        # Create attack timer
        period = 1.0 / self.publish_rate
        self.attack_timer = self.create_timer(period, self.attack_tick)

    def attack_tick(self):
        """Publish velocity command toward target."""
        elapsed = time.time() - self.attack_start_time

        # Check if attack duration exceeded
        if elapsed > self.attack_duration:
            self.finish_attack()
            return

        # Calculate direction to target
        dx = self.target_x - self.robot_x
        dy = self.target_y - self.robot_y
        dist = math.sqrt(dx*dx + dy*dy)

        if dist < 0.1:
            self.get_logger().warn('REACHED TARGET ZONE!')
            self.finish_attack()
            return

        # Normalize direction
        dx /= dist
        dy /= dist

        # Calculate velocity in robot frame
        # Robot heading is self.robot_yaw
        # We want to move toward (dx, dy) in world frame
        # Transform to robot frame
        cos_yaw = math.cos(self.robot_yaw)
        sin_yaw = math.sin(self.robot_yaw)

        # World to robot frame rotation
        robot_vx = dx * cos_yaw + dy * sin_yaw
        robot_vy = -dx * sin_yaw + dy * cos_yaw

        # Also add angular velocity to turn toward target
        target_yaw = math.atan2(dy, dx)
        yaw_error = target_yaw - self.robot_yaw

        # Normalize yaw error to [-pi, pi]
        while yaw_error > math.pi:
            yaw_error -= 2 * math.pi
        while yaw_error < -math.pi:
            yaw_error += 2 * math.pi

        # Create velocity command
        cmd = Twist()
        cmd.linear.x = self.attack_velocity * robot_vx
        cmd.linear.y = self.attack_velocity * robot_vy
        cmd.angular.z = 0.5 * yaw_error  # P-controller for heading

        # Publish command
        self.cmd_pub.publish(cmd)
        self.commands_sent += 1

        # Log progress periodically
        if self.commands_sent % int(self.publish_rate * 2) == 0:
            self.get_logger().info(
                f'Attack progress: {elapsed:.1f}s, '
                f'pos=({self.robot_x:.2f}, {self.robot_y:.2f}), '
                f'dist_to_target={dist:.2f}m'
            )

    def finish_attack(self):
        """Finish attack and report results."""
        # Stop publishing
        if hasattr(self, 'attack_timer'):
            self.attack_timer.cancel()

        # Send stop command
        stop_cmd = Twist()
        self.cmd_pub.publish(stop_cmd)

        elapsed = time.time() - self.attack_start_time

        self.get_logger().info('=' * 50)
        self.get_logger().info('ATTACK COMPLETE')
        self.get_logger().info('=' * 50)
        self.get_logger().info(f'Duration: {elapsed:.1f}s')
        self.get_logger().info(f'Commands sent: {self.commands_sent}')
        self.get_logger().info(f'Start position: ({self.start_x:.2f}, {self.start_y:.2f})')
        self.get_logger().info(f'Final position: ({self.robot_x:.2f}, {self.robot_y:.2f})')

        dist_traveled = math.sqrt(
            (self.robot_x - self.start_x)**2 +
            (self.robot_y - self.start_y)**2
        )
        self.get_logger().info(f'Distance traveled: {dist_traveled:.2f}m')

        dist_to_target = math.sqrt(
            (self.target_x - self.robot_x)**2 +
            (self.target_y - self.robot_y)**2
        )
        self.get_logger().info(f'Final distance to target: {dist_to_target:.2f}m')
        self.get_logger().info(f'Minimum distance to target: {self.min_distance_to_target:.2f}m')

        # Calculate expected distance without defense
        expected_dist = self.attack_velocity * elapsed
        actual_toward_target = math.sqrt(
            (self.start_x - self.target_x)**2 +
            (self.start_y - self.target_y)**2
        ) - dist_to_target

        if actual_toward_target < expected_dist * 0.5:
            self.get_logger().info('DEFENSE: Geofence significantly limited movement toward target!')
        else:
            self.get_logger().warn('WARNING: Robot moved significantly toward forbidden zone!')

        # Signal episode stop
        msg = String()
        msg.data = 'stop'
        self.episode_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = CmdVelAttackNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Attack interrupted')
        # Send stop command
        stop_cmd = Twist()
        node.cmd_pub.publish(stop_cmd)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
