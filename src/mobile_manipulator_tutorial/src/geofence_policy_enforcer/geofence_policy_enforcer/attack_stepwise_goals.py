#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attack Stepwise Goals - Simulates indirect command steering attack.

This attack node demonstrates how an attacker might try to bypass geofencing
by sending many small incremental goals that individually seem harmless but
collectively move the robot toward a forbidden zone.

This is used to test and validate the Goal PEP (goal_gate_node).
"""

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from nav2_msgs.action import NavigateToPose

from tf_transformations import quaternion_from_euler


class StepwiseGoalAttackNode(Node):
    """
    Simulates stepwise steering attack toward forbidden zone.

    Strategy: Send small goals (e.g., 0.5m forward) repeatedly,
    gradually moving toward the forbidden zone boundary.
    """

    def __init__(self):
        super().__init__('attack_stepwise_goals')

        # Parameters
        self.declare_parameter('target_x', 3.0)  # Forbidden zone center
        self.declare_parameter('target_y', 3.0)
        self.declare_parameter('step_size', 0.5)  # meters per step
        self.declare_parameter('num_steps', 20)
        self.declare_parameter('use_safe_action', True)  # Use *_safe action or direct
        self.declare_parameter('delay_between_steps', 1.0)  # seconds

        self.target_x = self.get_parameter('target_x').get_parameter_value().double_value
        self.target_y = self.get_parameter('target_y').get_parameter_value().double_value
        self.step_size = self.get_parameter('step_size').get_parameter_value().double_value
        self.num_steps = self.get_parameter('num_steps').get_parameter_value().integer_value
        self.use_safe = self.get_parameter('use_safe_action').get_parameter_value().bool_value
        self.delay = self.get_parameter('delay_between_steps').get_parameter_value().double_value

        # Current robot position
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.position_received = False

        # Action client
        action_name = 'navigate_to_pose_safe' if self.use_safe else 'navigate_to_pose'
        self._action_client = ActionClient(self, NavigateToPose, action_name)

        # Odometry subscriber
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10
        )

        # Episode control publisher
        self.episode_pub = self.create_publisher(String, '/geofence/episode_control', 10)

        # Attack statistics
        self.goals_sent = 0
        self.goals_succeeded = 0
        self.goals_rejected = 0

        self.get_logger().info('Stepwise Goal Attack node initialized')
        self.get_logger().info(f'Target: ({self.target_x}, {self.target_y})')
        self.get_logger().info(f'Step size: {self.step_size}m, Steps: {self.num_steps}')
        self.get_logger().info(f'Using action: {action_name}')

        # Start attack after short delay
        self.create_timer(2.0, self.start_attack, callback_group=None)
        self.attack_started = False

    def odom_callback(self, msg: Odometry):
        """Update robot position from odometry."""
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = math.atan2(siny_cosp, cosy_cosp)

        self.position_received = True

    def start_attack(self):
        """Start the stepwise attack."""
        if self.attack_started:
            return
        self.attack_started = True

        self.get_logger().info('=' * 50)
        self.get_logger().info('STARTING STEPWISE GOAL ATTACK')
        self.get_logger().info('=' * 50)

        # Signal episode start
        msg = String()
        msg.data = 'start'
        self.episode_pub.publish(msg)

        # Wait for position
        while not self.position_received and rclpy.ok():
            self.get_logger().info('Waiting for odometry...')
            rclpy.spin_once(self, timeout_sec=1.0)

        self.get_logger().info(f'Starting position: ({self.robot_x:.2f}, {self.robot_y:.2f})')

        # Execute attack
        self.execute_attack()

    def execute_attack(self):
        """Execute stepwise attack toward target."""
        for step in range(self.num_steps):
            if not rclpy.ok():
                break

            # Calculate direction to target
            dx = self.target_x - self.robot_x
            dy = self.target_y - self.robot_y
            dist = math.sqrt(dx*dx + dy*dy)

            if dist < self.step_size:
                self.get_logger().info('Reached target vicinity!')
                break

            # Normalize and scale by step size
            step_x = self.robot_x + (dx / dist) * self.step_size
            step_y = self.robot_y + (dy / dist) * self.step_size
            yaw = math.atan2(dy, dx)

            self.get_logger().info(
                f'Step {step + 1}/{self.num_steps}: '
                f'({self.robot_x:.2f}, {self.robot_y:.2f}) -> ({step_x:.2f}, {step_y:.2f})'
            )

            # Send goal
            success = self.send_goal_and_wait(step_x, step_y, yaw)

            if success:
                self.goals_succeeded += 1
                self.get_logger().info(f'  Goal SUCCEEDED')
            else:
                self.goals_rejected += 1
                self.get_logger().warn(f'  Goal REJECTED/FAILED')

            self.goals_sent += 1

            # Wait before next step
            time.sleep(self.delay)

            # Update position
            rclpy.spin_once(self, timeout_sec=0.1)

        self.finish_attack()

    def send_goal_and_wait(self, x: float, y: float, yaw: float) -> bool:
        """Send navigation goal and wait for result."""
        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Action server not available')
            return False

        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        goal.pose.pose.position.z = 0.0

        q = quaternion_from_euler(0, 0, yaw)
        goal.pose.pose.orientation.x = q[0]
        goal.pose.pose.orientation.y = q[1]
        goal.pose.pose.orientation.z = q[2]
        goal.pose.pose.orientation.w = q[3]

        # Send goal
        future = self._action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)

        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            return False

        # Wait for result
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=30.0)

        if result_future.result() is None:
            return False

        return result_future.result().status == 4  # SUCCEEDED

    def finish_attack(self):
        """Finish attack and report results."""
        self.get_logger().info('=' * 50)
        self.get_logger().info('ATTACK COMPLETE')
        self.get_logger().info('=' * 50)
        self.get_logger().info(f'Goals sent: {self.goals_sent}')
        self.get_logger().info(f'Goals succeeded: {self.goals_succeeded}')
        self.get_logger().info(f'Goals rejected: {self.goals_rejected}')
        self.get_logger().info(f'Final position: ({self.robot_x:.2f}, {self.robot_y:.2f})')

        dist_to_target = math.sqrt(
            (self.target_x - self.robot_x)**2 +
            (self.target_y - self.robot_y)**2
        )
        self.get_logger().info(f'Distance to target: {dist_to_target:.2f}m')

        if self.goals_rejected > 0:
            self.get_logger().info('DEFENSE: Geofence blocked some goals!')
        else:
            self.get_logger().warn('WARNING: No goals were rejected!')

        # Signal episode stop
        msg = String()
        msg.data = 'stop'
        self.episode_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = StepwiseGoalAttackNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Attack interrupted')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
