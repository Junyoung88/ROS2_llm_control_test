#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Navigator Node with Geofence Integration

Subscribes to /llm_command (std_msgs/String), calls Claude API to interpret
the command as a navigation goal, validates against geofence policy,
and sends it to Nav2's NavigateToPose action.

Geofence integration provides:
- Pre-validation of LLM-generated goals
- Claude prompt includes forbidden zone information
- Automatic goal projection to safe locations
- Audit logging of all navigation attempts
"""

import os
import json
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from tf_transformations import quaternion_from_euler
from typing import Optional

# Try to import the official Anthropic SDK; fall back to HTTP if unavailable
try:
    import anthropic
    ANTHROPIC_SDK_AVAILABLE = True
except ImportError:
    ANTHROPIC_SDK_AVAILABLE = False
    import urllib.request
    import urllib.error

# Try to import geofence policy; fall back to no geofencing
try:
    from geofence_policy_enforcer.geofence_core import GeofencePolicy, PolicyAction
    GEOFENCE_AVAILABLE = True
except ImportError:
    GEOFENCE_AVAILABLE = False


class LLMNavigatorNode(Node):
    """ROS 2 node that converts natural language commands to navigation goals via Claude."""

    # Base system prompt
    SYSTEM_PROMPT_BASE = """You are a robot navigation assistant. Your task is to interpret natural language commands and output a navigation goal as JSON.

You must respond with ONLY a valid JSON object, no other text. The JSON format is:
{
  "frame_id": "map",
  "x": <float>,
  "y": <float>,
  "yaw": <float in radians>
}

The environment is a small indoor warehouse/lab space. Coordinate ranges are approximately:
- x: -10 to 10 meters
- y: -10 to 10 meters
- yaw: -3.14159 to 3.14159 radians (where 0 is facing +x, pi/2 is facing +y)

Common reference points you can use:
- "origin" or "start" or "home": x=0, y=0, yaw=0
- "pick station" or "loading area": x=-4.7, y=5.6, yaw=3.14159
- "drop station" or "delivery area": x=-0.3, y=1.2, yaw=3.14159
- "center": x=0, y=0, yaw=0

Interpret directions like "go left", "move forward 2 meters", "turn around" relative to a robot starting at origin facing +x direction.

"""

    # Forbidden zone prompt addition
    FORBIDDEN_ZONE_PROMPT = """
IMPORTANT SAFETY CONSTRAINTS:
The following areas are FORBIDDEN ZONES. You must NEVER generate coordinates inside or near these areas:
{zones}

If the user asks to go to a forbidden zone, respond with coordinates to the nearest SAFE location instead.
Always maintain at least {margin:.2f} meters distance from forbidden zone boundaries.

"""

    def __init__(self):
        super().__init__('llm_navigator')

        # Declare parameters
        self.declare_parameter('geofence_config', '')
        self.declare_parameter('enable_geofence', True)
        self.declare_parameter('enable_projection', True)
        self.declare_parameter('use_safe_action', False)  # Use navigate_to_pose_safe proxy
        self.declare_parameter('action_name', 'navigate_to_pose')

        # Get API key from environment
        self.api_key = os.environ.get('ANTHROPIC_API_KEY')
        if not self.api_key:
            self.get_logger().error('ANTHROPIC_API_KEY environment variable not set!')
            self.get_logger().error('Set it with: export ANTHROPIC_API_KEY=your_key_here')

        # Initialize Anthropic client if SDK is available
        if ANTHROPIC_SDK_AVAILABLE and self.api_key:
            self.client = anthropic.Anthropic(api_key=self.api_key)
            self.get_logger().info('Using Anthropic Python SDK')
        else:
            self.client = None
            self.get_logger().info('Using HTTP fallback for Claude API')

        # Initialize geofence policy
        self.enable_geofence = self.get_parameter('enable_geofence').get_parameter_value().bool_value
        self.enable_projection = self.get_parameter('enable_projection').get_parameter_value().bool_value
        self.policy = None

        if GEOFENCE_AVAILABLE and self.enable_geofence:
            config_path = self.get_parameter('geofence_config').get_parameter_value().string_value
            if config_path:
                self.policy = GeofencePolicy(config_path)
                self.get_logger().info(f'Geofence enabled: {len(self.policy.zones)} zones loaded')
                self.get_logger().info(f'Safety margin: {self.policy.get_safety_margin():.3f}m')
            else:
                self.get_logger().warn('Geofence config not specified, geofencing disabled')
        elif not GEOFENCE_AVAILABLE:
            self.get_logger().warn('geofence_policy_enforcer not installed, geofencing disabled')

        # Build system prompt with forbidden zones
        self.system_prompt = self._build_system_prompt()

        # Determine action server name
        use_safe = self.get_parameter('use_safe_action').get_parameter_value().bool_value
        action_name = self.get_parameter('action_name').get_parameter_value().string_value
        if use_safe:
            action_name = 'navigate_to_pose_safe'

        # Nav2 action client
        self._nav_client = ActionClient(self, NavigateToPose, action_name)
        self.get_logger().info(f'Using action server: {action_name}')

        # Subscriber for LLM commands
        self.command_sub = self.create_subscription(
            String, '/llm_command', self.command_callback, 10
        )

        # Publishers
        self.status_pub = self.create_publisher(String, '/delivery_status', 10)
        self.geofence_event_pub = self.create_publisher(String, '/geofence/llm_events', 10)

        # State tracking
        self._current_goal_handle = None
        self._navigating = False

        self.publish_status('IDLE')
        self.get_logger().info('LLM Navigator node initialized')
        self.get_logger().info('Listening for commands on /llm_command')

    def _build_system_prompt(self) -> str:
        """Build system prompt including forbidden zone information."""
        prompt = self.SYSTEM_PROMPT_BASE

        if self.policy and self.policy.zones:
            # Build forbidden zones description
            zones_desc = []
            for zone in self.policy.zones:
                if zone.zone_type.value == 'forbidden':
                    # Calculate bounding box for simpler description
                    xs = [v[0] for v in zone.vertices]
                    ys = [v[1] for v in zone.vertices]
                    zones_desc.append(
                        f"- {zone.name}: rectangular area from ({min(xs):.1f}, {min(ys):.1f}) "
                        f"to ({max(xs):.1f}, {max(ys):.1f})"
                    )

            if zones_desc:
                prompt += self.FORBIDDEN_ZONE_PROMPT.format(
                    zones='\n'.join(zones_desc),
                    margin=self.policy.get_safety_margin()
                )

        prompt += "\nOutput ONLY the JSON object, nothing else."
        return prompt

    def publish_status(self, status: str):
        """Publish status message to /delivery_status."""
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)
        self.get_logger().info(f'Status: {status}')

    def publish_geofence_event(self, event_type: str, details: dict):
        """Publish geofence-related events."""
        msg = String()
        msg.data = json.dumps({
            'event': event_type,
            **details
        })
        self.geofence_event_pub.publish(msg)

    def command_callback(self, msg: String):
        """Handle incoming natural language command."""
        command = msg.data.strip()
        if not command:
            self.get_logger().warn('Received empty command, ignoring')
            return

        self.get_logger().info(f'Received command: "{command}"')

        if self._navigating:
            self.get_logger().warn('Navigation already in progress, ignoring new command')
            return

        # Call Claude API to interpret the command
        nav_goal = self.call_claude_api(command)
        if nav_goal is None:
            self.publish_status('FAILED')
            return

        # Validate against geofence policy
        validated_goal = self.validate_goal(nav_goal, command)
        if validated_goal is None:
            self.publish_status('REJECTED')
            return

        # Send navigation goal to Nav2
        self.send_nav_goal(validated_goal)

    def validate_goal(self, nav_goal: dict, original_command: str) -> Optional[dict]:
        """Validate goal against geofence policy."""
        if not self.policy:
            return nav_goal  # No geofence, pass through

        x, y = nav_goal['x'], nav_goal['y']
        decision = self.policy.evaluate_point(x, y)

        # Log the event
        self.publish_geofence_event('goal_evaluation', {
            'command': original_command,
            'original_x': x,
            'original_y': y,
            'decision': decision.action.value,
            'reason': decision.reason,
            'min_distance': decision.min_distance_to_forbidden,
        })

        if decision.action == PolicyAction.ALLOW:
            self.get_logger().info(
                f'Goal ({x:.2f}, {y:.2f}) ALLOWED - '
                f'distance to forbidden: {decision.min_distance_to_forbidden:.2f}m'
            )
            return nav_goal

        elif decision.action == PolicyAction.REJECT:
            if self.enable_projection and decision.projected_point:
                # Project to safe location
                new_x, new_y = decision.projected_point
                self.get_logger().warn(
                    f'Goal ({x:.2f}, {y:.2f}) PROJECTED to ({new_x:.2f}, {new_y:.2f}): '
                    f'{decision.reason}'
                )
                nav_goal['x'] = new_x
                nav_goal['y'] = new_y

                self.publish_geofence_event('goal_projected', {
                    'original_x': x,
                    'original_y': y,
                    'projected_x': new_x,
                    'projected_y': new_y,
                })
                return nav_goal
            else:
                self.get_logger().error(
                    f'Goal ({x:.2f}, {y:.2f}) REJECTED: {decision.reason}'
                )
                self.publish_geofence_event('goal_rejected', {
                    'x': x,
                    'y': y,
                    'reason': decision.reason,
                    'zone': decision.violated_zone,
                })
                return None

        elif decision.action == PolicyAction.PROJECT:
            if decision.projected_point:
                new_x, new_y = decision.projected_point
                self.get_logger().warn(
                    f'Goal ({x:.2f}, {y:.2f}) too close to forbidden zone, '
                    f'PROJECTED to ({new_x:.2f}, {new_y:.2f})'
                )
                nav_goal['x'] = new_x
                nav_goal['y'] = new_y
                return nav_goal
            else:
                return nav_goal

        elif decision.action == PolicyAction.WARN:
            self.get_logger().warn(
                f'Goal ({x:.2f}, {y:.2f}) is in BUFFER zone: {decision.reason}'
            )
            return nav_goal

        return nav_goal

    def call_claude_api(self, command: str) -> dict | None:
        """Call Claude API to convert natural language to navigation goal JSON."""
        if not self.api_key:
            self.get_logger().error('No API key available')
            return None

        try:
            if ANTHROPIC_SDK_AVAILABLE and self.client:
                response = self._call_with_sdk(command)
            else:
                response = self._call_with_http(command)

            # Parse JSON response
            nav_goal = json.loads(response)

            # Validate required fields
            required_fields = ['frame_id', 'x', 'y', 'yaw']
            for field in required_fields:
                if field not in nav_goal:
                    self.get_logger().error(f'Missing required field: {field}')
                    return None

            self.get_logger().info(
                f'Claude response: x={nav_goal["x"]:.2f}, y={nav_goal["y"]:.2f}, '
                f'yaw={nav_goal["yaw"]:.2f}'
            )
            return nav_goal

        except json.JSONDecodeError as e:
            self.get_logger().error(f'Failed to parse Claude response as JSON: {e}')
            self.get_logger().error(f'Response was: {response}')
            return None
        except Exception as e:
            self.get_logger().error(f'Claude API call failed: {e}')
            return None

    def _call_with_sdk(self, command: str) -> str:
        """Call Claude API using the official Anthropic SDK."""
        message = self.client.messages.create(
            model='claude-sonnet-4-20250514',
            max_tokens=256,
            system=self.system_prompt,
            messages=[
                {'role': 'user', 'content': command}
            ]
        )
        return message.content[0].text

    def _call_with_http(self, command: str) -> str:
        """Call Claude API using raw HTTP request (fallback)."""
        url = 'https://api.anthropic.com/v1/messages'
        headers = {
            'Content-Type': 'application/json',
            'x-api-key': self.api_key,
            'anthropic-version': '2023-06-01'
        }
        data = {
            'model': 'claude-sonnet-4-20250514',
            'max_tokens': 256,
            'system': self.system_prompt,
            'messages': [
                {'role': 'user', 'content': command}
            ]
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode('utf-8'),
            headers=headers,
            method='POST'
        )

        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode('utf-8'))
            return result['content'][0]['text']

    def send_nav_goal(self, nav_goal: dict):
        """Send navigation goal to Nav2's NavigateToPose action."""
        # Wait for action server
        if not self._nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('NavigateToPose action server not available')
            self.publish_status('FAILED')
            return

        # Create PoseStamped message
        pose = PoseStamped()
        pose.header.frame_id = nav_goal.get('frame_id', 'map')
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(nav_goal['x'])
        pose.pose.position.y = float(nav_goal['y'])
        pose.pose.position.z = 0.0

        # Convert yaw to quaternion
        q = quaternion_from_euler(0, 0, float(nav_goal['yaw']))
        pose.pose.orientation.x = q[0]
        pose.pose.orientation.y = q[1]
        pose.pose.orientation.z = q[2]
        pose.pose.orientation.w = q[3]

        # Create goal message
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose

        self.get_logger().info(
            f'Sending navigation goal: x={pose.pose.position.x:.2f}, '
            f'y={pose.pose.position.y:.2f}, yaw={nav_goal["yaw"]:.2f}'
        )

        self._navigating = True
        self.publish_status('MOVING')

        # Send goal asynchronously
        send_goal_future = self._nav_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """Handle goal acceptance/rejection."""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Navigation goal rejected by Nav2')
            self._navigating = False
            self.publish_status('FAILED')
            return

        self.get_logger().info('Navigation goal accepted')
        self._current_goal_handle = goal_handle

        # Request result
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def feedback_callback(self, feedback_msg):
        """Handle navigation feedback."""
        feedback = feedback_msg.feedback
        # Log progress periodically (every ~2 seconds based on feedback rate)
        remaining = feedback.distance_remaining
        if remaining > 0.5:  # Only log if significant distance remaining
            self.get_logger().debug(f'Distance remaining: {remaining:.2f}m')

    def result_callback(self, future):
        """Handle navigation result."""
        result = future.result()
        status = result.status

        self._navigating = False
        self._current_goal_handle = None

        # Status codes: 4 = SUCCEEDED, 5 = CANCELED, 6 = ABORTED
        if status == 4:  # SUCCEEDED
            self.get_logger().info('Navigation SUCCEEDED')
            self.publish_status('DONE')
        else:
            self.get_logger().error(f'Navigation FAILED with status: {status}')
            self.publish_status('FAILED')


def main(args=None):
    rclpy.init(args=args)
    node = LLMNavigatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down LLM Navigator node')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
