#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Path Watchdog Node - Path Policy Enforcement Point.

Monitors published paths and trajectories for forbidden zone intersections.
If a violation is detected, triggers navigation cancel and/or emergency stop.
"""

import math
import json
from typing import List, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy

from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import String, Bool
from nav_msgs.msg import Path
from action_msgs.msg import GoalStatus
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient

from .geofence_core import GeofencePolicy, PolicyAction


class PathWatchdogNode(Node):
    """
    Monitors navigation paths for geofence violations.

    Subscribes to planned paths, checks for forbidden zone intersections,
    and triggers stop/cancel if violations are detected.
    """

    def __init__(self):
        super().__init__('path_watchdog')

        # Parameters
        self.declare_parameter('geofence_config', '')
        self.declare_parameter('path_topic', '/plan')
        self.declare_parameter('local_path_topic', '/local_plan')
        self.declare_parameter('cancel_on_violation', True)
        self.declare_parameter('stop_on_violation', True)
        self.declare_parameter('check_interval', 0.5)  # seconds
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')

        # Load geofence policy
        config_path = self.get_parameter('geofence_config').get_parameter_value().string_value
        self.policy = GeofencePolicy(config_path if config_path else None)

        # Get parameters
        path_topic = self.get_parameter('path_topic').get_parameter_value().string_value
        local_path_topic = self.get_parameter('local_path_topic').get_parameter_value().string_value
        cmd_vel_topic = self.get_parameter('cmd_vel_topic').get_parameter_value().string_value

        self.cancel_on_violation = self.get_parameter('cancel_on_violation').get_parameter_value().bool_value
        self.stop_on_violation = self.get_parameter('stop_on_violation').get_parameter_value().bool_value

        # Subscribers
        qos_latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)

        self.global_path_sub = self.create_subscription(
            Path, path_topic, self.global_path_callback, qos_latched
        )
        self.local_path_sub = self.create_subscription(
            Path, local_path_topic, self.local_path_callback, 10
        )

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, cmd_vel_topic, 10)
        self.violation_pub = self.create_publisher(String, '/geofence/path_violations', 10)
        self.status_pub = self.create_publisher(String, '/geofence/path_status', 10)
        self.emergency_stop_pub = self.create_publisher(Bool, '/geofence/emergency_stop', 10)

        # Action client for canceling navigation
        self._nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # State
        self.last_global_path = None
        self.last_local_path = None
        self.violation_active = False
        self.current_goal_handle = None

        # Statistics
        self.stats = {
            'global_paths_checked': 0,
            'local_paths_checked': 0,
            'violations_detected': 0,
            'emergency_stops_triggered': 0,
        }

        # Status timer
        check_interval = self.get_parameter('check_interval').get_parameter_value().double_value
        self.create_timer(check_interval, self.periodic_check)
        self.create_timer(1.0, self.publish_status)

        self.get_logger().info('Path Watchdog node initialized')
        self.get_logger().info(f'Monitoring: {path_topic}, {local_path_topic}')
        self.get_logger().info(f'Cancel on violation: {self.cancel_on_violation}')
        self.get_logger().info(f'Stop on violation: {self.stop_on_violation}')

    def global_path_callback(self, msg: Path):
        """Check global path for violations."""
        self.stats['global_paths_checked'] += 1
        self.last_global_path = msg

        violations = self.check_path(msg, 'global')
        if violations:
            self.handle_violations(violations, 'global')

    def local_path_callback(self, msg: Path):
        """Check local path for violations."""
        self.stats['local_paths_checked'] += 1
        self.last_local_path = msg

        violations = self.check_path(msg, 'local')
        if violations:
            self.handle_violations(violations, 'local')

    def check_path(self, path: Path, path_type: str) -> List[dict]:
        """
        Check path for forbidden zone intersections.

        Returns list of violation info dicts.
        """
        if not path.poses:
            return []

        violations = []
        margin = self.policy.get_safety_margin()

        prev_point = None
        for i, pose in enumerate(path.poses):
            x = pose.pose.position.x
            y = pose.pose.position.y
            point = (x, y)

            # Check point itself
            decision = self.policy.evaluate_point(x, y)
            if decision.action in (PolicyAction.REJECT, PolicyAction.PROJECT):
                violations.append({
                    'type': 'point_violation',
                    'path_type': path_type,
                    'index': i,
                    'x': x,
                    'y': y,
                    'reason': decision.reason,
                    'zone': decision.violated_zone,
                    'distance': decision.min_distance_to_forbidden,
                })

            # Check segment from previous point
            if prev_point is not None:
                segment_decision = self.policy.evaluate_segment(prev_point, point)
                if segment_decision.action == PolicyAction.REJECT:
                    violations.append({
                        'type': 'segment_violation',
                        'path_type': path_type,
                        'index': i,
                        'from': prev_point,
                        'to': point,
                        'reason': segment_decision.reason,
                        'zone': segment_decision.violated_zone,
                    })

            prev_point = point

        return violations

    def handle_violations(self, violations: List[dict], path_type: str):
        """Handle detected violations."""
        self.stats['violations_detected'] += len(violations)
        self.violation_active = True

        # Log violations
        for v in violations:
            self.get_logger().error(
                f'PATH VIOLATION ({path_type}): {v["type"]} at index {v["index"]} - {v["reason"]}'
            )

        # Publish violation event
        msg = String()
        msg.data = json.dumps({
            'event': 'path_violation',
            'path_type': path_type,
            'violations': violations,
            'timestamp': self.get_clock().now().nanoseconds,
        })
        self.violation_pub.publish(msg)

        # Trigger emergency stop
        if self.stop_on_violation:
            self.trigger_emergency_stop()

        # Cancel navigation
        if self.cancel_on_violation:
            self.cancel_navigation()

    def trigger_emergency_stop(self):
        """Send emergency stop commands."""
        self.stats['emergency_stops_triggered'] += 1
        self.get_logger().warn('EMERGENCY STOP triggered due to path violation')

        # Publish zero velocity
        stop_cmd = Twist()
        self.cmd_vel_pub.publish(stop_cmd)

        # Publish emergency stop signal
        stop_msg = Bool()
        stop_msg.data = True
        self.emergency_stop_pub.publish(stop_msg)

    def cancel_navigation(self):
        """Cancel current navigation goal."""
        self.get_logger().warn('Attempting to cancel navigation')

        # We can't directly cancel without the goal handle
        # Instead, we rely on emergency stop and publish cancel request
        # In a full implementation, we'd track goal handles

        # Publish cancel event
        msg = String()
        msg.data = json.dumps({
            'event': 'navigation_cancel_requested',
            'reason': 'path_violation',
        })
        self.violation_pub.publish(msg)

    def periodic_check(self):
        """Periodically re-check paths."""
        # Re-check global path
        if self.last_global_path:
            violations = self.check_path(self.last_global_path, 'global_recheck')
            if violations and not self.violation_active:
                self.handle_violations(violations, 'global_recheck')

        # Clear violation flag if no issues found in recent check
        if self.last_global_path and self.last_local_path:
            global_violations = self.check_path(self.last_global_path, 'check')
            local_violations = self.check_path(self.last_local_path, 'check')
            if not global_violations and not local_violations:
                if self.violation_active:
                    self.get_logger().info('Path violation cleared')
                    # Publish clear signal
                    clear_msg = Bool()
                    clear_msg.data = False
                    self.emergency_stop_pub.publish(clear_msg)
                self.violation_active = False

    def publish_status(self):
        """Publish periodic status."""
        # Calculate current min distance along paths
        min_dist_global = float('inf')
        min_dist_local = float('inf')

        if self.last_global_path:
            for pose in self.last_global_path.poses:
                dist = self.policy.get_min_distance_to_forbidden(
                    pose.pose.position.x, pose.pose.position.y
                )
                if dist < min_dist_global:
                    min_dist_global = dist

        if self.last_local_path:
            for pose in self.last_local_path.poses:
                dist = self.policy.get_min_distance_to_forbidden(
                    pose.pose.position.x, pose.pose.position.y
                )
                if dist < min_dist_local:
                    min_dist_local = dist

        msg = String()
        msg.data = json.dumps({
            'status': 'violation_active' if self.violation_active else 'monitoring',
            'min_dist_global_path': min_dist_global,
            'min_dist_local_path': min_dist_local,
            'safety_margin': self.policy.get_safety_margin(),
            'stats': self.stats,
        })
        self.status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PathWatchdogNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Path Watchdog node')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
