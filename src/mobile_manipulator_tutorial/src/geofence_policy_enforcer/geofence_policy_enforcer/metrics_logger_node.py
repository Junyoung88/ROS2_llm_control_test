#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Metrics Logger Node - Aggregates and logs geofence enforcement metrics.

Collects metrics from all PEP nodes, computes episode statistics,
publishes to /geofence/metrics, and writes CSV to disk.
"""

import os
import csv
import json
import time
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

from std_msgs.msg import String, Bool
from nav_msgs.msg import Odometry

from .geofence_core import GeofencePolicy


@dataclass
class EpisodeMetrics:
    """Metrics for a single episode/test run."""
    episode_id: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0

    # Distance metrics
    min_distance_to_forbidden: float = float('inf')
    avg_distance_to_forbidden: float = 0.0
    time_in_buffer_zone: float = 0.0
    time_in_safe_zone: float = 0.0

    # Goal metrics
    total_goals: int = 0
    goals_allowed: int = 0
    goals_rejected: int = 0
    goals_projected: int = 0

    # Velocity metrics
    total_velocity_commands: int = 0
    velocity_commands_passed: int = 0
    velocity_commands_blocked: int = 0
    velocity_commands_scaled: int = 0

    # Path metrics
    path_violations_detected: int = 0
    emergency_stops_triggered: int = 0

    # Violation flag
    any_violation_occurred: bool = False

    # Internal tracking
    _distance_samples: list = field(default_factory=list, repr=False)


class MetricsLoggerNode(Node):
    """
    Central metrics aggregation and logging node.
    """

    def __init__(self):
        super().__init__('metrics_logger')

        # Parameters
        self.declare_parameter('geofence_config', '')
        self.declare_parameter('csv_output_file', '/tmp/geofence_metrics.csv')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('sample_rate', 10.0)  # Hz
        self.declare_parameter('episode_timeout', 60.0)  # seconds

        # Load geofence policy for distance calculations
        config_path = self.get_parameter('geofence_config').get_parameter_value().string_value
        self.policy = GeofencePolicy(config_path if config_path else None)

        self.csv_file = self.get_parameter('csv_output_file').get_parameter_value().string_value
        self.episode_timeout = self.get_parameter('episode_timeout').get_parameter_value().double_value
        odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value

        # Current robot state
        self.robot_x = 0.0
        self.robot_y = 0.0

        # Episode tracking
        self.current_episode = EpisodeMetrics()
        self.episode_count = 0
        self.episode_active = False
        self.last_activity_time = time.time()

        # Zone tracking
        self.last_zone_check_time = time.time()
        self.in_buffer_zone = False

        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry, odom_topic, self.odom_callback, 10
        )

        # Subscribe to all PEP event topics
        self.goal_events_sub = self.create_subscription(
            String, '/geofence/goal_events', self.goal_events_callback, 10
        )
        self.cmd_vel_events_sub = self.create_subscription(
            String, '/geofence/cmd_vel_events', self.cmd_vel_events_callback, 10
        )
        self.path_violations_sub = self.create_subscription(
            String, '/geofence/path_violations', self.path_violations_callback, 10
        )
        self.emergency_stop_sub = self.create_subscription(
            Bool, '/geofence/emergency_stop', self.emergency_stop_callback, 10
        )

        # Subscribe to status topics for stats
        self.goal_status_sub = self.create_subscription(
            String, '/geofence/status', self.goal_status_callback, 10
        )
        self.cmd_vel_status_sub = self.create_subscription(
            String, '/geofence/cmd_vel_status', self.cmd_vel_status_callback, 10
        )
        self.path_status_sub = self.create_subscription(
            String, '/geofence/path_status', self.path_status_callback, 10
        )

        # Episode control subscriber
        self.episode_control_sub = self.create_subscription(
            String, '/geofence/episode_control', self.episode_control_callback, 10
        )

        # Publishers
        self.metrics_pub = self.create_publisher(String, '/geofence/metrics', 10)

        # Timers
        sample_rate = self.get_parameter('sample_rate').get_parameter_value().double_value
        self.create_timer(1.0 / sample_rate, self.sample_callback)
        self.create_timer(1.0, self.publish_metrics)
        self.create_timer(5.0, self.check_episode_timeout)

        # Initialize CSV file
        self.init_csv()

        self.get_logger().info('Metrics Logger node initialized')
        self.get_logger().info(f'CSV output: {self.csv_file}')
        self.get_logger().info(f'Send to /geofence/episode_control: "start", "stop", "reset"')

    def init_csv(self):
        """Initialize CSV file with headers."""
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'episode_id', 'timestamp', 'duration',
                    'min_distance_to_forbidden', 'avg_distance_to_forbidden',
                    'time_in_buffer_zone', 'time_in_safe_zone',
                    'total_goals', 'goals_allowed', 'goals_rejected', 'goals_projected',
                    'total_velocity_commands', 'velocity_passed', 'velocity_blocked', 'velocity_scaled',
                    'path_violations', 'emergency_stops',
                    'any_violation_occurred'
                ])

    def odom_callback(self, msg: Odometry):
        """Update robot position."""
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y

    def sample_callback(self):
        """Periodic sampling of distance metrics."""
        if not self.episode_active or not self.policy.zones:
            return

        dist = self.policy.get_min_distance_to_forbidden(self.robot_x, self.robot_y)

        # Update min distance
        if dist < self.current_episode.min_distance_to_forbidden:
            self.current_episode.min_distance_to_forbidden = dist

        # Track distance samples for averaging
        self.current_episode._distance_samples.append(dist)

        # Track zone time
        now = time.time()
        dt = now - self.last_zone_check_time
        self.last_zone_check_time = now

        margin = self.policy.get_safety_margin()
        if dist < margin * 2:  # In buffer zone
            self.current_episode.time_in_buffer_zone += dt
            self.in_buffer_zone = True
        else:
            self.current_episode.time_in_safe_zone += dt
            self.in_buffer_zone = False

        # Check for actual violation (inside forbidden zone)
        if dist <= 0:
            self.current_episode.any_violation_occurred = True
            self.get_logger().error(f'VIOLATION: Robot inside forbidden zone!')

    def goal_events_callback(self, msg: String):
        """Process goal gate events."""
        self.last_activity_time = time.time()
        if not self.episode_active:
            self.start_episode()

        try:
            data = json.loads(msg.data)
            action = data.get('action', '')

            self.current_episode.total_goals += 1

            if action == 'allow':
                self.current_episode.goals_allowed += 1
            elif action == 'reject':
                self.current_episode.goals_rejected += 1
            elif action == 'project':
                self.current_episode.goals_projected += 1

        except json.JSONDecodeError:
            pass

    def cmd_vel_events_callback(self, msg: String):
        """Process cmd_vel guard events."""
        self.last_activity_time = time.time()
        if not self.episode_active:
            self.start_episode()

        try:
            data = json.loads(msg.data)
            event = data.get('event', '')

            if event == 'velocity_modified':
                self.current_episode.total_velocity_commands += 1
                self.current_episode.velocity_commands_blocked += 1

        except json.JSONDecodeError:
            pass

    def path_violations_callback(self, msg: String):
        """Process path violation events."""
        self.last_activity_time = time.time()
        if not self.episode_active:
            self.start_episode()

        try:
            data = json.loads(msg.data)
            violations = data.get('violations', [])
            self.current_episode.path_violations_detected += len(violations)
        except json.JSONDecodeError:
            pass

    def emergency_stop_callback(self, msg: Bool):
        """Process emergency stop events."""
        if msg.data:
            self.current_episode.emergency_stops_triggered += 1
            self.get_logger().warn('Emergency stop recorded')

    def goal_status_callback(self, msg: String):
        """Update goal stats from status."""
        try:
            data = json.loads(msg.data)
            stats = data.get('stats', {})
            if stats:
                self.current_episode.goals_allowed = stats.get('allowed', 0)
                self.current_episode.goals_rejected = stats.get('rejected', 0)
                self.current_episode.goals_projected = stats.get('projected', 0)
                self.current_episode.total_goals = stats.get('total_goals', 0)
        except json.JSONDecodeError:
            pass

    def cmd_vel_status_callback(self, msg: String):
        """Update cmd_vel stats from status."""
        try:
            data = json.loads(msg.data)
            stats = data.get('stats', {})
            if stats:
                self.current_episode.velocity_commands_passed = stats.get('passed', 0)
                self.current_episode.velocity_commands_blocked = stats.get('blocked', 0)
                self.current_episode.velocity_commands_scaled = stats.get('scaled', 0)
                self.current_episode.total_velocity_commands = stats.get('total_commands', 0)
        except json.JSONDecodeError:
            pass

    def path_status_callback(self, msg: String):
        """Update path stats from status."""
        try:
            data = json.loads(msg.data)
            stats = data.get('stats', {})
            if stats:
                self.current_episode.path_violations_detected = stats.get('violations_detected', 0)
                self.current_episode.emergency_stops_triggered = stats.get('emergency_stops_triggered', 0)
        except json.JSONDecodeError:
            pass

    def episode_control_callback(self, msg: String):
        """Handle episode control commands."""
        cmd = msg.data.strip().lower()

        if cmd == 'start':
            self.start_episode()
        elif cmd == 'stop':
            self.stop_episode()
        elif cmd == 'reset':
            self.reset_episode()

    def start_episode(self):
        """Start a new episode."""
        if self.episode_active:
            return

        self.episode_count += 1
        self.current_episode = EpisodeMetrics()
        self.current_episode.episode_id = self.episode_count
        self.current_episode.start_time = time.time()
        self.episode_active = True
        self.last_activity_time = time.time()
        self.last_zone_check_time = time.time()

        self.get_logger().info(f'Episode {self.episode_count} started')

    def stop_episode(self):
        """Stop current episode and save metrics."""
        if not self.episode_active:
            return

        self.current_episode.end_time = time.time()
        self.current_episode.duration = self.current_episode.end_time - self.current_episode.start_time

        # Calculate average distance
        if self.current_episode._distance_samples:
            self.current_episode.avg_distance_to_forbidden = (
                sum(self.current_episode._distance_samples) /
                len(self.current_episode._distance_samples)
            )

        # Write to CSV
        self.write_csv()

        self.get_logger().info(f'Episode {self.current_episode.episode_id} stopped')
        self.get_logger().info(f'  Duration: {self.current_episode.duration:.2f}s')
        self.get_logger().info(f'  Min distance: {self.current_episode.min_distance_to_forbidden:.3f}m')
        self.get_logger().info(f'  Goals rejected: {self.current_episode.goals_rejected}')
        self.get_logger().info(f'  Violations: {self.current_episode.any_violation_occurred}')

        self.episode_active = False

    def reset_episode(self):
        """Reset without saving."""
        self.episode_active = False
        self.current_episode = EpisodeMetrics()
        self.get_logger().info('Episode reset')

    def check_episode_timeout(self):
        """Check for episode timeout due to inactivity."""
        if not self.episode_active:
            return

        if time.time() - self.last_activity_time > self.episode_timeout:
            self.get_logger().info('Episode timeout due to inactivity')
            self.stop_episode()

    def write_csv(self):
        """Write episode metrics to CSV."""
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.current_episode.episode_id,
                datetime.now().isoformat(),
                self.current_episode.duration,
                self.current_episode.min_distance_to_forbidden,
                self.current_episode.avg_distance_to_forbidden,
                self.current_episode.time_in_buffer_zone,
                self.current_episode.time_in_safe_zone,
                self.current_episode.total_goals,
                self.current_episode.goals_allowed,
                self.current_episode.goals_rejected,
                self.current_episode.goals_projected,
                self.current_episode.total_velocity_commands,
                self.current_episode.velocity_commands_passed,
                self.current_episode.velocity_commands_blocked,
                self.current_episode.velocity_commands_scaled,
                self.current_episode.path_violations_detected,
                self.current_episode.emergency_stops_triggered,
                self.current_episode.any_violation_occurred
            ])

    def publish_metrics(self):
        """Publish current metrics."""
        metrics = {
            'episode_active': self.episode_active,
            'episode_id': self.current_episode.episode_id,
            'duration': time.time() - self.current_episode.start_time if self.episode_active else 0,
            'robot_position': {'x': self.robot_x, 'y': self.robot_y},
            'current_distance_to_forbidden': self.policy.get_min_distance_to_forbidden(
                self.robot_x, self.robot_y
            ) if self.policy.zones else float('inf'),
            'min_distance_to_forbidden': self.current_episode.min_distance_to_forbidden,
            'goals': {
                'total': self.current_episode.total_goals,
                'allowed': self.current_episode.goals_allowed,
                'rejected': self.current_episode.goals_rejected,
                'projected': self.current_episode.goals_projected,
            },
            'velocity': {
                'total': self.current_episode.total_velocity_commands,
                'passed': self.current_episode.velocity_commands_passed,
                'blocked': self.current_episode.velocity_commands_blocked,
                'scaled': self.current_episode.velocity_commands_scaled,
            },
            'path_violations': self.current_episode.path_violations_detected,
            'emergency_stops': self.current_episode.emergency_stops_triggered,
            'any_violation': self.current_episode.any_violation_occurred,
            'in_buffer_zone': self.in_buffer_zone,
        }

        msg = String()
        msg.data = json.dumps(metrics)
        self.metrics_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = MetricsLoggerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Save final episode if active
        if node.episode_active:
            node.stop_episode()
        node.get_logger().info('Shutting down Metrics Logger node')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
