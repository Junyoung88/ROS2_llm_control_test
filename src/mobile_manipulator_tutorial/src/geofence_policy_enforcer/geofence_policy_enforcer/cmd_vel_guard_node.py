#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cmd Vel Guard Node - Control Policy Enforcement Point.

Intercepts cmd_vel commands and performs forward simulation to predict
if the robot would enter a forbidden zone. If so, modifies or blocks the command.

Strategies:
- STOP: Zero velocity if violation predicted
- SCALE: Scale velocity to stay safe
- REDIRECT: Modify velocity direction away from danger
"""

import math
import json
from enum import Enum
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import Twist, PoseStamped, TransformStamped
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from tf2_ros import TransformListener, Buffer

from .geofence_core import GeofencePolicy, PolicyAction


class OverrideStrategy(Enum):
    """Velocity override strategies when violation is predicted."""
    STOP = "stop"           # Zero all velocities
    SCALE = "scale"         # Scale down velocity
    REDIRECT = "redirect"   # Redirect away from danger


class CmdVelGuardNode(Node):
    """
    Velocity command filter with forward simulation.

    Subscribes to cmd_vel, simulates future positions, checks against
    geofence policy, and publishes safe velocity commands.
    """

    def __init__(self):
        super().__init__('cmd_vel_guard')

        # Parameters
        self.declare_parameter('geofence_config', '')
        self.declare_parameter('input_topic', '/cmd_vel_raw')
        self.declare_parameter('output_topic', '/cmd_vel')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('simulation_horizon', 2.0)  # seconds
        self.declare_parameter('simulation_steps', 20)
        self.declare_parameter('override_strategy', 'stop')
        self.declare_parameter('min_distance_threshold', 0.3)  # meters
        self.declare_parameter('scale_factor_decay', 0.8)

        # Load geofence policy
        config_path = self.get_parameter('geofence_config').get_parameter_value().string_value
        self.policy = GeofencePolicy(config_path if config_path else None)

        # Get parameters
        input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value

        self.sim_horizon = self.get_parameter('simulation_horizon').get_parameter_value().double_value
        self.sim_steps = self.get_parameter('simulation_steps').get_parameter_value().integer_value
        self.min_dist_threshold = self.get_parameter('min_distance_threshold').get_parameter_value().double_value
        self.scale_decay = self.get_parameter('scale_factor_decay').get_parameter_value().double_value

        strategy_str = self.get_parameter('override_strategy').get_parameter_value().string_value
        self.override_strategy = OverrideStrategy(strategy_str)

        # TF buffer for robot pose
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Current robot state
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.last_odom_time = None

        # QoS for real-time control
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)

        # Subscribers
        self.cmd_sub = self.create_subscription(
            Twist, input_topic, self.cmd_vel_callback, qos
        )
        self.odom_sub = self.create_subscription(
            Odometry, odom_topic, self.odom_callback, qos
        )

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, output_topic, qos)
        self.metrics_pub = self.create_publisher(String, '/geofence/cmd_vel_events', 10)
        self.status_pub = self.create_publisher(String, '/geofence/cmd_vel_status', 10)

        # Statistics
        self.stats = {
            'total_commands': 0,
            'passed': 0,
            'blocked': 0,
            'scaled': 0,
            'min_distance_ever': float('inf'),
        }

        # Status timer
        self.create_timer(1.0, self.publish_status)

        self.get_logger().info('Cmd Vel Guard node initialized')
        self.get_logger().info(f'Input: {input_topic} -> Output: {output_topic}')
        self.get_logger().info(f'Strategy: {self.override_strategy.value}')
        self.get_logger().info(f'Simulation horizon: {self.sim_horizon}s, steps: {self.sim_steps}')

    def odom_callback(self, msg: Odometry):
        """Update robot pose from odometry."""
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y

        # Extract yaw from quaternion
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = math.atan2(siny_cosp, cosy_cosp)

        self.last_odom_time = self.get_clock().now()

    def cmd_vel_callback(self, msg: Twist):
        """Process incoming velocity command."""
        self.stats['total_commands'] += 1

        # Check if we have recent pose data
        if self.last_odom_time is None:
            # No pose data yet, pass through with warning
            self.get_logger().warn_throttle(
                self.get_clock(), 5000,
                'No odometry data received yet, passing cmd_vel through'
            )
            self.cmd_pub.publish(msg)
            return

        # Check pose data freshness (within 0.5 seconds)
        age = (self.get_clock().now() - self.last_odom_time).nanoseconds / 1e9
        if age > 0.5:
            self.get_logger().warn_throttle(
                self.get_clock(), 5000,
                f'Odometry data is stale ({age:.2f}s old)'
            )

        # Perform forward simulation
        safe_cmd, min_dist, violation_time = self.simulate_trajectory(msg)

        # Update stats
        if min_dist < self.stats['min_distance_ever']:
            self.stats['min_distance_ever'] = min_dist

        # Publish the (possibly modified) command
        self.cmd_pub.publish(safe_cmd)

        # Log significant events
        if violation_time is not None:
            self.publish_event(msg, safe_cmd, min_dist, violation_time)

    def simulate_trajectory(self, cmd: Twist) -> Tuple[Twist, float, Optional[float]]:
        """
        Simulate trajectory and return safe command.

        Returns:
            (safe_command, min_distance_to_forbidden, violation_time_or_none)
        """
        dt = self.sim_horizon / self.sim_steps
        margin = self.policy.get_safety_margin()

        x, y, yaw = self.robot_x, self.robot_y, self.robot_yaw
        vx, vy, wz = cmd.linear.x, cmd.linear.y, cmd.angular.z

        min_dist = float('inf')
        violation_time = None
        violation_step = None

        # Simulate forward
        for step in range(self.sim_steps):
            t = step * dt

            # Update pose (differential drive model)
            x += (vx * math.cos(yaw) - vy * math.sin(yaw)) * dt
            y += (vx * math.sin(yaw) + vy * math.cos(yaw)) * dt
            yaw += wz * dt

            # Check distance to forbidden zones
            dist = self.policy.get_min_distance_to_forbidden(x, y)
            if dist < min_dist:
                min_dist = dist

            # Check for violation (inside zone or within margin)
            decision = self.policy.evaluate_point(x, y)
            if decision.action in (PolicyAction.REJECT, PolicyAction.PROJECT):
                if violation_time is None:
                    violation_time = t
                    violation_step = step
                break

            # Also check if distance is below threshold
            if dist < self.min_dist_threshold and violation_time is None:
                violation_time = t
                violation_step = step

        # If no violation, pass through
        if violation_time is None:
            self.stats['passed'] += 1
            return cmd, min_dist, None

        # Apply override strategy
        if self.override_strategy == OverrideStrategy.STOP:
            self.stats['blocked'] += 1
            safe_cmd = Twist()  # Zero velocity
            self.get_logger().warn(
                f'BLOCKED cmd_vel: predicted violation at t={violation_time:.2f}s, '
                f'min_dist={min_dist:.3f}m'
            )

        elif self.override_strategy == OverrideStrategy.SCALE:
            self.stats['scaled'] += 1
            # Scale based on how soon violation would occur
            scale = max(0.0, (violation_step / self.sim_steps) * self.scale_decay)
            safe_cmd = Twist()
            safe_cmd.linear.x = cmd.linear.x * scale
            safe_cmd.linear.y = cmd.linear.y * scale
            safe_cmd.angular.z = cmd.angular.z * scale
            self.get_logger().warn(
                f'SCALED cmd_vel by {scale:.2f}: predicted violation at t={violation_time:.2f}s'
            )

        elif self.override_strategy == OverrideStrategy.REDIRECT:
            self.stats['scaled'] += 1
            # Find direction away from nearest forbidden zone
            safe_cmd = self.compute_redirect(cmd, x, y)
            self.get_logger().warn(
                f'REDIRECTED cmd_vel: predicted violation at t={violation_time:.2f}s'
            )

        else:
            safe_cmd = Twist()

        return safe_cmd, min_dist, violation_time

    def compute_redirect(self, cmd: Twist, danger_x: float, danger_y: float) -> Twist:
        """Compute redirected velocity away from danger point."""
        # Direction from danger point back to robot
        dx = self.robot_x - danger_x
        dy = self.robot_y - danger_y
        dist = math.sqrt(dx*dx + dy*dy)

        if dist < 0.01:
            # Already at danger point, just stop
            return Twist()

        # Normalize
        dx /= dist
        dy /= dist

        # Project original velocity onto safe direction
        orig_speed = math.sqrt(cmd.linear.x**2 + cmd.linear.y**2)

        safe_cmd = Twist()
        # Move in retreat direction at reduced speed
        safe_cmd.linear.x = dx * orig_speed * 0.5
        safe_cmd.linear.y = dy * orig_speed * 0.5
        # Reduce angular velocity
        safe_cmd.angular.z = cmd.angular.z * 0.3

        return safe_cmd

    def publish_event(self, original: Twist, safe: Twist, min_dist: float, violation_time: float):
        """Publish cmd_vel guard event."""
        msg = String()
        msg.data = json.dumps({
            'event': 'velocity_modified',
            'original_vx': original.linear.x,
            'original_vy': original.linear.y,
            'original_wz': original.angular.z,
            'safe_vx': safe.linear.x,
            'safe_vy': safe.linear.y,
            'safe_wz': safe.angular.z,
            'min_distance': min_dist,
            'violation_time': violation_time,
            'strategy': self.override_strategy.value,
        })
        self.metrics_pub.publish(msg)

    def publish_status(self):
        """Publish periodic status."""
        msg = String()
        msg.data = json.dumps({
            'status': 'active',
            'strategy': self.override_strategy.value,
            'robot_pose': {
                'x': self.robot_x,
                'y': self.robot_y,
                'yaw': self.robot_yaw
            },
            'stats': self.stats,
            'current_min_dist': self.policy.get_min_distance_to_forbidden(
                self.robot_x, self.robot_y
            ) if self.policy.zones else float('inf'),
        })
        self.status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = CmdVelGuardNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Cmd Vel Guard node')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
