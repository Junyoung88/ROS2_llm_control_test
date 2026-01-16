#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Claude Vision-based Pick and Place Node

Uses Claude's vision API to detect objects (blocks) in camera images
and controls the robot arm to pick them up.

Subscribes to:
    /camera/image: Camera images
    /pick_command: Pick commands (std_msgs/String)

Actions used:
    /ur_arm_controller/follow_joint_trajectory: Arm control
    /gripper_controller/gripper_cmd: Gripper control
"""

import os
import json
import base64
import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import String
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import PoseArray
from control_msgs.action import FollowJointTrajectory, GripperCommand
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from cv_bridge import CvBridge
import cv2
from typing import Optional, Dict, List, Tuple

# Try to import Anthropic SDK
try:
    import anthropic
    ANTHROPIC_SDK_AVAILABLE = True
except ImportError:
    ANTHROPIC_SDK_AVAILABLE = False
    import urllib.request
    import urllib.error


class VisionPickPlaceNode(Node):
    """ROS2 node for vision-based pick and place using Claude API."""

    # UR5 arm joint names (order matters!)
    ARM_JOINTS = [
        'shoulder_pan_joint',
        'shoulder_lift_joint',
        'elbow_joint',
        'wrist_1_joint',
        'wrist_2_joint',
        'wrist_3_joint'
    ]

    # Predefined arm positions (in radians)
    # Joint order: shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3
    ARM_POSITIONS = {
        'home': [0.0, -1.57, 1.57, -1.57, -1.57, 0.0],
        'look_down': [0.0, -0.5, 0.5, -1.57, -1.57, 0.0],
        'look_forward': [3.14, -1.0, 0.5, -1.07, -1.57, 0.0],
        # Table height picking (front) - reaches ~0.5m forward at ~0.8m height
        'pre_pick_front': [3.14, -1.2, 1.0, -1.37, -1.57, 0.0],
        'pick_front': [3.14, -0.9, 1.3, -1.97, -1.57, 0.0],
        # Right side picking
        'pre_pick_right': [1.57, -1.2, 1.0, -1.37, -1.57, 0.0],
        'pick_right': [1.57, -0.9, 1.3, -1.97, -1.57, 0.0],
        # Left side picking
        'pre_pick_left': [-1.57, -1.2, 1.0, -1.37, -1.57, 0.0],
        'pick_left': [-1.57, -0.9, 1.3, -1.97, -1.57, 0.0],
        # Place position
        'place': [-1.57, -1.0, 1.5, -2.07, -1.57, 0.0],
        # Extended reach for table
        'reach_table': [3.14, -0.7, 1.5, -2.37, -1.57, 0.0],
    }

    VISION_PROMPT = """Analyze this image from a robot's camera.
Look for any small colored blocks, cubes, or box-shaped objects that could be picked up by a robot gripper.

If you find a block/cube:
1. Describe its color and approximate size
2. Estimate its position in the image:
   - horizontal_position: "left", "center", or "right" (relative to image center)
   - vertical_position: "top", "middle", or "bottom"
   - distance_estimate: "close" (< 0.5m), "medium" (0.5-1.5m), or "far" (> 1.5m)
3. Estimate the angle from camera center in degrees (-30 to +30, where negative is left)

Respond with ONLY a valid JSON object in this format:
{
  "block_detected": true/false,
  "color": "red/blue/green/etc or null",
  "horizontal_position": "left/center/right",
  "vertical_position": "top/middle/bottom",
  "distance_estimate": "close/medium/far",
  "angle_from_center": <float degrees>,
  "confidence": <float 0-1>,
  "description": "brief description"
}

If no block is detected, set block_detected to false and other fields to null."""

    def __init__(self):
        super().__init__('vision_pick_place')

        # Callback group for concurrent execution
        self.cb_group = ReentrantCallbackGroup()

        # API key
        self.api_key = os.environ.get('ANTHROPIC_API_KEY')
        if not self.api_key:
            self.get_logger().error('ANTHROPIC_API_KEY not set!')

        # Initialize Anthropic client
        if ANTHROPIC_SDK_AVAILABLE and self.api_key:
            self.client = anthropic.Anthropic(api_key=self.api_key)
            self.get_logger().info('Using Anthropic SDK')
        else:
            self.client = None
            self.get_logger().info('Using HTTP fallback')

        # CV Bridge
        self.bridge = CvBridge()

        # State
        self.latest_image: Optional[np.ndarray] = None
        self.latest_image_msg: Optional[Image] = None
        self.current_joint_positions: Dict[str, float] = {}
        self.is_busy = False
        self.object_world_position: Optional[Tuple[float, float, float]] = None

        # QoS for camera
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE
        )

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/image', self.image_callback, sensor_qos,
            callback_group=self.cb_group
        )
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10,
            callback_group=self.cb_group
        )
        self.command_sub = self.create_subscription(
            String, '/pick_command', self.command_callback, 10,
            callback_group=self.cb_group
        )
        # Subscribe to object pose from Gazebo
        self.object_pose_sub = self.create_subscription(
            PoseArray, '/model/object1/pose', self.object_pose_callback, 10,
            callback_group=self.cb_group
        )

        # Publishers
        self.status_pub = self.create_publisher(String, '/pick_status', 10)
        self.detection_pub = self.create_publisher(String, '/block_detection', 10)
        self.annotated_pub = self.create_publisher(Image, '/pick_annotated_image', 10)

        # Action clients
        self.arm_client = ActionClient(
            self, FollowJointTrajectory,
            '/ur_arm_controller/follow_joint_trajectory',
            callback_group=self.cb_group
        )
        self.gripper_client = ActionClient(
            self, GripperCommand,
            '/gripper_controller/gripper_cmd',
            callback_group=self.cb_group
        )

        self.publish_status('READY')
        self.get_logger().info('Vision Pick & Place node initialized')
        self.get_logger().info('Commands: "detect", "pick", "place", "home", "open", "close"')
        self.get_logger().info('Send commands to /pick_command topic')

    def publish_status(self, status: str):
        """Publish status message."""
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)
        self.get_logger().info(f'Status: {status}')

    def image_callback(self, msg: Image):
        """Store latest camera image."""
        try:
            self.latest_image_msg = msg
            if msg.encoding == 'rgb8':
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
                self.latest_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            else:
                self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Image conversion error: {e}')

    def joint_state_callback(self, msg: JointState):
        """Store current joint positions."""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.current_joint_positions[name] = msg.position[i]

    def object_pose_callback(self, msg: PoseArray):
        """Store object world position from Gazebo."""
        if msg.poses:
            pose = msg.poses[0]
            self.object_world_position = (
                pose.position.x,
                pose.position.y,
                pose.position.z
            )

    def command_callback(self, msg: String):
        """Handle pick commands."""
        command = msg.data.strip().lower()
        if not command:
            return

        self.get_logger().info(f'Received command: "{command}"')

        if self.is_busy:
            self.get_logger().warn('Robot is busy, ignoring command')
            return

        self.is_busy = True

        try:
            if command == 'detect':
                self.detect_block()
            elif command == 'pick':
                self.execute_pick_sequence()
            elif command == 'place':
                self.execute_place_sequence()
            elif command == 'home':
                self.move_to_position('home')
            elif command == 'look':
                self.move_to_position('look_down')
            elif command == 'open':
                self.control_gripper(open_gripper=True)
            elif command == 'close':
                self.control_gripper(open_gripper=False)
            elif command.startswith('move '):
                pos_name = command[5:].strip()
                if pos_name in self.ARM_POSITIONS:
                    self.move_to_position(pos_name)
                else:
                    self.get_logger().error(f'Unknown position: {pos_name}')
            else:
                self.get_logger().warn(f'Unknown command: {command}')
        except Exception as e:
            self.get_logger().error(f'Command execution error: {e}')
        finally:
            self.is_busy = False

    def detect_block(self) -> Optional[Dict]:
        """Detect block using Claude Vision API."""
        if self.latest_image is None:
            self.get_logger().error('No camera image available')
            return None

        self.publish_status('DETECTING')

        try:
            # Encode image to base64
            _, buffer = cv2.imencode('.jpg', self.latest_image,
                                      [cv2.IMWRITE_JPEG_QUALITY, 85])
            image_base64 = base64.b64encode(buffer).decode('utf-8')

            # Call Claude Vision API
            if ANTHROPIC_SDK_AVAILABLE and self.client:
                response = self.client.messages.create(
                    model='claude-sonnet-4-20250514',
                    max_tokens=512,
                    messages=[{
                        'role': 'user',
                        'content': [
                            {
                                'type': 'image',
                                'source': {
                                    'type': 'base64',
                                    'media_type': 'image/jpeg',
                                    'data': image_base64
                                }
                            },
                            {
                                'type': 'text',
                                'text': self.VISION_PROMPT
                            }
                        ]
                    }]
                )
                response_text = response.content[0].text
            else:
                response_text = self._call_vision_api_http(image_base64)

            # Parse response
            detection = json.loads(response_text)

            # Publish detection result
            det_msg = String()
            det_msg.data = json.dumps(detection)
            self.detection_pub.publish(det_msg)

            if detection.get('block_detected'):
                self.get_logger().info(
                    f"Block detected: {detection.get('color')} block at "
                    f"{detection.get('horizontal_position')}, "
                    f"distance: {detection.get('distance_estimate')}, "
                    f"confidence: {detection.get('confidence', 0):.0%}"
                )
                self.publish_status('BLOCK_DETECTED')

                # Draw annotation on image
                self._annotate_detection(detection)
            else:
                self.get_logger().info('No block detected')
                self.publish_status('NO_BLOCK')

            return detection

        except json.JSONDecodeError as e:
            self.get_logger().error(f'Failed to parse detection response: {e}')
            return None
        except Exception as e:
            self.get_logger().error(f'Detection error: {e}')
            return None

    def _call_vision_api_http(self, image_base64: str) -> str:
        """Call Claude Vision API via HTTP."""
        url = 'https://api.anthropic.com/v1/messages'
        headers = {
            'Content-Type': 'application/json',
            'x-api-key': self.api_key,
            'anthropic-version': '2023-06-01'
        }
        data = {
            'model': 'claude-sonnet-4-20250514',
            'max_tokens': 512,
            'messages': [{
                'role': 'user',
                'content': [
                    {
                        'type': 'image',
                        'source': {
                            'type': 'base64',
                            'media_type': 'image/jpeg',
                            'data': image_base64
                        }
                    },
                    {
                        'type': 'text',
                        'text': self.VISION_PROMPT
                    }
                ]
            }]
        }

        req = urllib.request.Request(
            url, data=json.dumps(data).encode('utf-8'),
            headers=headers, method='POST'
        )

        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode('utf-8'))
            return result['content'][0]['text']

    def _annotate_detection(self, detection: Dict):
        """Draw detection result on image."""
        if self.latest_image is None:
            return

        annotated = self.latest_image.copy()
        h, w = annotated.shape[:2]

        if detection.get('block_detected'):
            # Estimate bounding box from position info
            horiz = detection.get('horizontal_position', 'center')
            vert = detection.get('vertical_position', 'middle')

            # Map to image coordinates
            cx = {'left': w//4, 'center': w//2, 'right': 3*w//4}.get(horiz, w//2)
            cy = {'top': h//4, 'middle': h//2, 'bottom': 3*h//4}.get(vert, h//2)

            # Draw circle at estimated position
            cv2.circle(annotated, (cx, cy), 50, (0, 255, 0), 3)
            cv2.circle(annotated, (cx, cy), 5, (0, 255, 0), -1)

            # Draw label
            label = f"{detection.get('color', 'unknown')} block"
            cv2.putText(annotated, label, (cx - 60, cy - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            conf = detection.get('confidence', 0)
            cv2.putText(annotated, f"Conf: {conf:.0%}", (cx - 60, cy - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Publish annotated image
        try:
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            self.annotated_pub.publish(annotated_msg)
        except Exception as e:
            self.get_logger().error(f'Failed to publish annotated image: {e}')

    def execute_pick_sequence(self):
        """Execute full pick sequence."""
        self.publish_status('PICKING')

        # First detect the block
        detection = self.detect_block()

        if not detection or not detection.get('block_detected'):
            self.get_logger().error('No block detected, cannot pick')
            self.publish_status('PICK_FAILED')
            return

        # Determine pick position based on detection
        horiz = detection.get('horizontal_position', 'center')

        # Choose arm position based on block location
        if horiz == 'right':
            pre_pick_pos = 'pre_pick_right'
            pick_pos = 'pick_right'
        elif horiz == 'left':
            pre_pick_pos = 'pre_pick_left'
            pick_pos = 'pick_left'
        else:  # center or default
            pre_pick_pos = 'pre_pick_front'
            pick_pos = 'pick_front'

        self.get_logger().info(f'Executing pick sequence: {pre_pick_pos} -> {pick_pos}')

        # Open gripper
        self.get_logger().info('Opening gripper...')
        if not self.control_gripper(open_gripper=True):
            self.publish_status('PICK_FAILED')
            return

        # Move to pre-pick position
        self.get_logger().info(f'Moving to {pre_pick_pos}...')
        if not self.move_to_position(pre_pick_pos):
            self.publish_status('PICK_FAILED')
            return

        # Move to pick position
        self.get_logger().info(f'Moving to {pick_pos}...')
        if not self.move_to_position(pick_pos):
            self.publish_status('PICK_FAILED')
            return

        # Close gripper
        self.get_logger().info('Closing gripper...')
        if not self.control_gripper(open_gripper=False):
            self.publish_status('PICK_FAILED')
            return

        # Return to pre-pick position
        self.get_logger().info('Lifting...')
        if not self.move_to_position(pre_pick_pos):
            self.publish_status('PICK_FAILED')
            return

        self.publish_status('PICK_SUCCESS')
        self.get_logger().info('Pick sequence completed!')

    def execute_place_sequence(self):
        """Execute place sequence."""
        self.publish_status('PLACING')

        # Move to place position
        self.get_logger().info('Moving to place position...')
        if not self.move_to_position('place'):
            self.publish_status('PLACE_FAILED')
            return

        # Open gripper to release
        self.get_logger().info('Releasing object...')
        if not self.control_gripper(open_gripper=True):
            self.publish_status('PLACE_FAILED')
            return

        # Return to home
        self.get_logger().info('Returning home...')
        if not self.move_to_position('home'):
            self.publish_status('PLACE_FAILED')
            return

        self.publish_status('PLACE_SUCCESS')
        self.get_logger().info('Place sequence completed!')

    def move_to_position(self, position_name: str) -> bool:
        """Move arm to predefined position."""
        if position_name not in self.ARM_POSITIONS:
            self.get_logger().error(f'Unknown position: {position_name}')
            return False

        target_positions = self.ARM_POSITIONS[position_name]
        return self.move_arm(target_positions)

    def move_arm(self, joint_positions: List[float], duration: float = 3.0) -> bool:
        """Move arm to specified joint positions."""
        if not self.arm_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Arm action server not available')
            return False

        # Create trajectory
        trajectory = JointTrajectory()
        trajectory.joint_names = self.ARM_JOINTS

        point = JointTrajectoryPoint()
        point.positions = joint_positions
        point.velocities = [0.0] * len(joint_positions)
        point.time_from_start = Duration(sec=int(duration), nanosec=int((duration % 1) * 1e9))
        trajectory.points = [point]

        # Create goal
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory

        # Send goal
        self.get_logger().info(f'Sending arm trajectory...')
        future = self.arm_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Arm goal rejected')
            return False

        # Wait for result
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=duration + 5.0)

        result = result_future.result()
        if result.status == 4:  # SUCCEEDED
            self.get_logger().info('Arm movement completed')
            return True
        else:
            self.get_logger().error(f'Arm movement failed: status={result.status}')
            return False

    def control_gripper(self, open_gripper: bool) -> bool:
        """Control gripper open/close."""
        if not self.gripper_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Gripper action server not available')
            return False

        goal = GripperCommand.Goal()
        # Robotiq 85 gripper: 0.0 = open, 0.8 = closed
        goal.command.position = 0.0 if open_gripper else 0.8
        goal.command.max_effort = 50.0

        action_name = 'Opening' if open_gripper else 'Closing'
        self.get_logger().info(f'{action_name} gripper...')

        future = self.gripper_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Gripper goal rejected')
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=5.0)

        self.get_logger().info(f'Gripper {action_name.lower()} completed')
        return True


def main(args=None):
    rclpy.init(args=args)
    node = VisionPickPlaceNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
