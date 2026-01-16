#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Goal Gate Node - Policy Enforcement Point for navigation goals.

Acts as an action proxy for NavigateToPose and FollowWaypoints.
Evaluates goals against geofence policy and either:
- ALLOW: Forward to Nav2
- REJECT: Abort with reason
- PROJECT: Modify goal to nearest safe point and forward
"""

import math
import json
from datetime import datetime
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, ActionClient, GoalResponse, CancelResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.callback_groups import ReentrantCallbackGroup

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from nav2_msgs.action import NavigateToPose, FollowWaypoints
from std_srvs.srv import Trigger
from visualization_msgs.msg import MarkerArray

from tf_transformations import euler_from_quaternion, quaternion_from_euler

from .geofence_core import GeofencePolicy, PolicyAction, PolicyDecision


class GoalGateNode(Node):
    """
    Action proxy that enforces geofence policy on navigation goals.

    Intercepts NavigateToPose and FollowWaypoints actions, evaluates them
    against the geofence policy, and either forwards, rejects, or projects them.
    """

    def __init__(self):
        super().__init__('goal_gate')

        # Declare parameters
        self.declare_parameter('geofence_config', '')
        self.declare_parameter('enable_projection', True)
        self.declare_parameter('audit_log_file', '/tmp/geofence_audit.jsonl')
        self.declare_parameter('nav2_action_name', 'navigate_to_pose')
        self.declare_parameter('waypoints_action_name', 'follow_waypoints')

        # Load geofence policy
        config_path = self.get_parameter('geofence_config').get_parameter_value().string_value
        self.policy = GeofencePolicy(config_path if config_path else None)

        self.enable_projection = self.get_parameter('enable_projection').get_parameter_value().bool_value
        self.audit_log_file = self.get_parameter('audit_log_file').get_parameter_value().string_value

        # Callback group for concurrent action handling
        self.cb_group = ReentrantCallbackGroup()

        # Action clients to forward to Nav2
        nav2_action = self.get_parameter('nav2_action_name').get_parameter_value().string_value
        waypoints_action = self.get_parameter('waypoints_action_name').get_parameter_value().string_value

        self._nav_client = ActionClient(
            self, NavigateToPose, nav2_action,
            callback_group=self.cb_group
        )
        self._waypoints_client = ActionClient(
            self, FollowWaypoints, waypoints_action,
            callback_group=self.cb_group
        )

        # Action servers (proxy entry points)
        self._nav_server = ActionServer(
            self, NavigateToPose, 'navigate_to_pose_safe',
            execute_callback=self.execute_nav_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.cb_group
        )
        self._waypoints_server = ActionServer(
            self, FollowWaypoints, 'follow_waypoints_safe',
            execute_callback=self.execute_waypoints_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.cb_group
        )

        # Publishers for metrics and visualization
        self.metrics_pub = self.create_publisher(String, '/geofence/goal_events', 10)
        self.viz_pub = self.create_publisher(MarkerArray, '/geofence/zones_viz', 10)
        self.status_pub = self.create_publisher(String, '/geofence/status', 10)

        # Service for reloading geofence config
        self.reload_srv = self.create_service(
            Trigger, '/geofence/reload', self.reload_callback
        )

        # Visualization timer
        self.create_timer(2.0, self.publish_visualization)

        # Statistics
        self.stats = {
            'total_goals': 0,
            'allowed': 0,
            'rejected': 0,
            'projected': 0,
        }

        self.get_logger().info('Goal Gate node initialized')
        self.get_logger().info(f'Listening on: navigate_to_pose_safe, follow_waypoints_safe')
        self.get_logger().info(f'Forwarding to: {nav2_action}, {waypoints_action}')
        if config_path:
            self.get_logger().info(f'Geofence config: {config_path}')
            self.get_logger().info(f'Loaded {len(self.policy.zones)} zones')
            self.get_logger().info(f'Safety margin: {self.policy.get_safety_margin():.3f}m')

    def goal_callback(self, goal_request) -> GoalResponse:
        """Accept all goals initially; policy check happens in execute."""
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle) -> CancelResponse:
        """Allow cancellation."""
        return CancelResponse.ACCEPT

    async def execute_nav_callback(self, goal_handle: ServerGoalHandle):
        """Execute NavigateToPose with policy enforcement."""
        self.get_logger().info('Received NavigateToPose goal')

        request = goal_handle.request
        pose = request.pose

        # Evaluate against policy
        x = pose.pose.position.x
        y = pose.pose.position.y
        decision = self.policy.evaluate_point(x, y)

        self.stats['total_goals'] += 1

        # Log the decision
        self.audit_log(decision, 'NavigateToPose', pose)
        self.publish_metrics(decision)

        if decision.action == PolicyAction.REJECT:
            self.stats['rejected'] += 1
            self.get_logger().warn(f'REJECTED goal ({x:.2f}, {y:.2f}): {decision.reason}')
            goal_handle.abort()
            result = NavigateToPose.Result()
            return result

        elif decision.action == PolicyAction.PROJECT and self.enable_projection:
            self.stats['projected'] += 1
            if decision.projected_point:
                new_x, new_y = decision.projected_point
                self.get_logger().warn(
                    f'PROJECTED goal ({x:.2f}, {y:.2f}) -> ({new_x:.2f}, {new_y:.2f}): {decision.reason}'
                )
                pose.pose.position.x = new_x
                pose.pose.position.y = new_y
            else:
                self.get_logger().warn(f'REJECTED (projection failed): {decision.reason}')
                goal_handle.abort()
                result = NavigateToPose.Result()
                return result

        else:
            self.stats['allowed'] += 1
            self.get_logger().info(f'ALLOWED goal ({x:.2f}, {y:.2f})')

        # Forward to Nav2
        if not self._nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Nav2 action server not available')
            goal_handle.abort()
            return NavigateToPose.Result()

        # Send goal to Nav2
        nav_goal = NavigateToPose.Goal()
        nav_goal.pose = pose
        nav_goal.behavior_tree = request.behavior_tree

        send_goal_future = self._nav_client.send_goal_async(nav_goal)
        nav_goal_handle = await send_goal_future

        if not nav_goal_handle.accepted:
            self.get_logger().error('Nav2 rejected the goal')
            goal_handle.abort()
            return NavigateToPose.Result()

        # Wait for result
        result_future = nav_goal_handle.get_result_async()
        nav_result = await result_future

        # Forward result
        if nav_result.status == 4:  # SUCCEEDED
            goal_handle.succeed()
        else:
            goal_handle.abort()

        return nav_result.result

    async def execute_waypoints_callback(self, goal_handle: ServerGoalHandle):
        """Execute FollowWaypoints with policy enforcement."""
        self.get_logger().info('Received FollowWaypoints goal')

        request = goal_handle.request
        poses = request.poses

        # Evaluate all waypoints
        filtered_poses = []
        any_rejected = False

        for i, pose in enumerate(poses):
            x = pose.pose.position.x
            y = pose.pose.position.y
            decision = self.policy.evaluate_point(x, y)

            self.stats['total_goals'] += 1
            self.audit_log(decision, f'FollowWaypoints[{i}]', pose)

            if decision.action == PolicyAction.REJECT:
                self.stats['rejected'] += 1
                self.get_logger().warn(f'REJECTED waypoint {i} ({x:.2f}, {y:.2f}): {decision.reason}')
                any_rejected = True
                continue  # Skip this waypoint

            elif decision.action == PolicyAction.PROJECT and self.enable_projection:
                self.stats['projected'] += 1
                if decision.projected_point:
                    new_x, new_y = decision.projected_point
                    self.get_logger().warn(
                        f'PROJECTED waypoint {i} ({x:.2f}, {y:.2f}) -> ({new_x:.2f}, {new_y:.2f})'
                    )
                    pose.pose.position.x = new_x
                    pose.pose.position.y = new_y
                    filtered_poses.append(pose)
                else:
                    self.get_logger().warn(f'REJECTED waypoint {i} (projection failed)')
                    any_rejected = True
            else:
                self.stats['allowed'] += 1
                filtered_poses.append(pose)

        if not filtered_poses:
            self.get_logger().error('All waypoints rejected')
            goal_handle.abort()
            return FollowWaypoints.Result()

        # Forward to Nav2
        if not self._waypoints_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Nav2 waypoints action server not available')
            goal_handle.abort()
            return FollowWaypoints.Result()

        wp_goal = FollowWaypoints.Goal()
        wp_goal.poses = filtered_poses

        send_goal_future = self._waypoints_client.send_goal_async(wp_goal)
        wp_goal_handle = await send_goal_future

        if not wp_goal_handle.accepted:
            self.get_logger().error('Nav2 rejected waypoints goal')
            goal_handle.abort()
            return FollowWaypoints.Result()

        result_future = wp_goal_handle.get_result_async()
        wp_result = await result_future

        if wp_result.status == 4:  # SUCCEEDED
            goal_handle.succeed()
        else:
            goal_handle.abort()

        return wp_result.result

    def audit_log(self, decision: PolicyDecision, action_type: str, pose: PoseStamped):
        """Write audit log entry."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'action_type': action_type,
            'decision': decision.action.value,
            'reason': decision.reason,
            'original_x': decision.original_point[0],
            'original_y': decision.original_point[1],
            'projected_x': decision.projected_point[0] if decision.projected_point else None,
            'projected_y': decision.projected_point[1] if decision.projected_point else None,
            'min_distance_to_forbidden': decision.min_distance_to_forbidden,
            'violated_zone': decision.violated_zone,
        }

        try:
            with open(self.audit_log_file, 'a') as f:
                f.write(json.dumps(entry) + '\n')
        except Exception as e:
            self.get_logger().warn(f'Failed to write audit log: {e}')

    def publish_metrics(self, decision: PolicyDecision):
        """Publish metrics event."""
        msg = String()
        msg.data = json.dumps({
            'action': decision.action.value,
            'min_dist': decision.min_distance_to_forbidden,
            'stats': self.stats
        })
        self.metrics_pub.publish(msg)

    def publish_visualization(self):
        """Publish geofence visualization markers."""
        if self.policy.zones:
            markers = self.policy.get_forbidden_zones_as_marker_array()
            for m in markers.markers:
                m.header.stamp = self.get_clock().now().to_msg()
            self.viz_pub.publish(markers)

        # Publish status
        status_msg = String()
        status_msg.data = json.dumps({
            'status': 'active',
            'zones_count': len(self.policy.zones),
            'safety_margin': self.policy.get_safety_margin(),
            'stats': self.stats
        })
        self.status_pub.publish(status_msg)

    def reload_callback(self, request, response):
        """Handle geofence reload service request."""
        if self.policy.reload():
            self.get_logger().info('Geofence configuration reloaded')
            response.success = True
            response.message = f'Reloaded {len(self.policy.zones)} zones'
        else:
            self.get_logger().error('Failed to reload geofence configuration')
            response.success = False
            response.message = 'Reload failed'
        return response


def main(args=None):
    rclpy.init(args=args)
    node = GoalGateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Goal Gate node')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
