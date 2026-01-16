#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Claude + MoveIt Integration Node (RT-2 Style)

A sophisticated manipulation system that combines:
- Claude Vision API for scene understanding
- Claude LLM for action planning
- MoveIt for motion planning and execution

Commands are given in natural language, e.g.:
- "Pick up the red block"
- "Move to the blue cube and grab it"
- "Place the object on the left side"

This mimics Google's RT-2 approach using Claude as the vision-language model.
"""

import os
import json
import base64
import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy, qos_profile_sensor_data
from rclpy.action import ActionClient
from std_msgs.msg import String
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion
from control_msgs.action import FollowJointTrajectory, GripperCommand
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from cv_bridge import CvBridge
import cv2
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum

# Anthropic SDK
try:
    import anthropic
    ANTHROPIC_SDK_AVAILABLE = True
except ImportError:
    ANTHROPIC_SDK_AVAILABLE = False
    import urllib.request


class ActionType(Enum):
    """Types of robot actions."""
    MOVE_ARM = "move_arm"
    OPEN_GRIPPER = "open_gripper"
    CLOSE_GRIPPER = "close_gripper"
    PICK = "pick"
    PLACE = "place"
    LOOK = "look"
    HOME = "home"


@dataclass
class DetectedObject:
    """Represents a detected object in the scene."""
    name: str
    color: str
    position_2d: Tuple[float, float]  # normalized (0-1) image coordinates
    estimated_3d: Optional[Tuple[float, float, float]] = None  # x, y, z in robot frame
    confidence: float = 0.0
    size_estimate: str = "medium"  # small, medium, large


@dataclass
class ActionPlan:
    """A planned action to execute."""
    action_type: ActionType
    target_position: Optional[Tuple[float, float, float]] = None
    target_object: Optional[str] = None
    description: str = ""


class ClaudeMoveItNode(Node):
    """ROS 2 node integrating Claude Vision with MoveIt for RT-2 style manipulation."""

    # Camera parameters (approximate, for depth estimation)
    CAMERA_FOV_H = 69.0  # degrees (typical RGB camera)
    CAMERA_FOV_V = 42.0  # degrees
    CAMERA_HEIGHT = 0.35  # meters from ground (camera_link z position)

    # Robot workspace bounds
    WORKSPACE_X_MIN, WORKSPACE_X_MAX = 0.2, 0.8  # meters from base
    WORKSPACE_Y_MIN, WORKSPACE_Y_MAX = -0.5, 0.5
    WORKSPACE_Z_MIN, WORKSPACE_Z_MAX = 0.0, 0.6

    # UR5 arm joint names
    ARM_JOINTS = [
        'shoulder_pan_joint',
        'shoulder_lift_joint',
        'elbow_joint',
        'wrist_1_joint',
        'wrist_2_joint',
        'wrist_3_joint'
    ]

    # Key positions for manipulation
    NAMED_POSITIONS = {
        'home': [0.0, -1.57, 1.57, -1.57, -1.57, 0.0],
        'ready': [0.0, -1.0, 1.0, -1.57, -1.57, 0.0],
        'look_table': [0.0, -0.8, 0.8, -1.57, -1.57, 0.0],
        'pre_grasp_front': [0.0, -1.2, 1.2, -1.57, -1.57, 0.0],
        'pre_grasp_left': [-1.57, -1.2, 1.2, -1.57, -1.57, 0.0],
        'pre_grasp_right': [1.57, -1.2, 1.2, -1.57, -1.57, 0.0],
    }

    # System prompt for scene understanding
    SCENE_UNDERSTANDING_PROMPT = """You are a robot vision system analyzing a scene for manipulation tasks.
Analyze the image and identify all graspable objects (blocks, cubes, boxes, bottles, etc.).

For each object found, provide:
1. name: descriptive name (e.g., "red_block", "blue_cube")
2. color: primary color
3. position: where in the image (use normalized coordinates 0-1)
   - x: 0=left edge, 0.5=center, 1=right edge
   - y: 0=top edge, 0.5=center, 1=bottom edge
4. size: "small" (<5cm), "medium" (5-15cm), or "large" (>15cm)
5. confidence: how confident you are (0-1)
6. graspable: true if a robot gripper could pick it up

Also describe:
- The overall scene layout
- Any obstacles or constraints
- The surface type (table, floor, shelf)

Respond ONLY with valid JSON:
{
  "scene_description": "brief description",
  "surface_type": "table/floor/shelf",
  "objects": [
    {
      "name": "object_name",
      "color": "color",
      "position": {"x": 0.5, "y": 0.5},
      "size": "medium",
      "confidence": 0.9,
      "graspable": true
    }
  ],
  "obstacles": ["list of obstacles"],
  "manipulation_notes": "any relevant notes for grasping"
}"""

    # System prompt for action planning
    ACTION_PLANNING_PROMPT = """You are a robot action planner. Given a natural language command and detected objects,
plan a sequence of robot actions to accomplish the task.

Available actions:
- LOOK: Move arm to observe the scene
- MOVE_ARM: Move end-effector to a position
- OPEN_GRIPPER: Open the gripper
- CLOSE_GRIPPER: Close the gripper to grasp
- PICK: Combined action (approach + close gripper + lift)
- PLACE: Combined action (move to target + open gripper + retract)
- HOME: Return arm to home position

For positions, use the robot's coordinate frame:
- X: forward (positive) / backward (negative), range 0.2-0.8m
- Y: left (positive) / right (negative), range -0.5 to 0.5m
- Z: up (positive), range 0.0-0.6m

Respond ONLY with valid JSON:
{
  "understood_task": "what you understood the user wants",
  "target_object": "name of target object or null",
  "actions": [
    {
      "action": "PICK/PLACE/MOVE_ARM/etc",
      "target_position": {"x": 0.5, "y": 0.0, "z": 0.3} or null,
      "description": "what this action does"
    }
  ],
  "success_probability": 0.8,
  "notes": "any concerns or assumptions"
}"""

    def __init__(self):
        super().__init__('claude_moveit')

        # Callback group
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
        self.current_joints: Dict[str, float] = {}
        self.detected_objects: List[DetectedObject] = []
        self.is_executing = False
        self.scene_cache: Optional[Dict] = None

        # QoS - match publisher settings
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE
        )

        # Subscribers - use default QoS (10) for better compatibility
        self.image_sub = self.create_subscription(
            Image, '/camera/image', self.image_callback, 10
        )
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_callback, 10,
            callback_group=self.cb_group
        )
        self.command_sub = self.create_subscription(
            String, '/manipulation_command', self.command_callback, 10,
            callback_group=self.cb_group
        )

        # Publishers
        self.status_pub = self.create_publisher(String, '/manipulation_status', 10)
        self.scene_pub = self.create_publisher(String, '/scene_analysis', 10)
        self.plan_pub = self.create_publisher(String, '/action_plan', 10)
        self.annotated_pub = self.create_publisher(Image, '/annotated_scene', 10)

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
        self.get_logger().info('='*50)
        self.get_logger().info('Claude + MoveIt Node Initialized')
        self.get_logger().info('='*50)
        self.get_logger().info('Send natural language commands to /manipulation_command')
        self.get_logger().info('Examples:')
        self.get_logger().info('  "Pick up the red block"')
        self.get_logger().info('  "Grab the cube and place it on the left"')
        self.get_logger().info('  "What objects do you see?"')
        self.get_logger().info('='*50)

    def publish_status(self, status: str, details: str = ""):
        """Publish status update."""
        msg = String()
        msg.data = json.dumps({"status": status, "details": details})
        self.status_pub.publish(msg)
        self.get_logger().info(f'Status: {status} {details}')

    def image_callback(self, msg: Image):
        """Store latest camera image."""
        try:
            if msg.encoding == 'rgb8':
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
                self.latest_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            else:
                self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # Log first few images received
            if not hasattr(self, '_img_count'):
                self._img_count = 0
            self._img_count += 1
            if self._img_count <= 3:
                self.get_logger().info(f'Received image {self._img_count}: {msg.width}x{msg.height}, encoding={msg.encoding}')
        except Exception as e:
            self.get_logger().error(f'Image conversion error: {e}')

    def joint_callback(self, msg: JointState):
        """Store current joint positions."""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.current_joints[name] = msg.position[i]

    def command_callback(self, msg: String):
        """Handle natural language manipulation commands."""
        command = msg.data.strip()
        if not command:
            return

        self.get_logger().info(f'Received command: "{command}"')

        if self.is_executing:
            self.get_logger().warn('Already executing, ignoring command')
            return

        self.is_executing = True
        self.publish_status('PROCESSING', command)

        try:
            # Step 1: Analyze scene
            scene = self.analyze_scene()
            if scene is None:
                self.publish_status('FAILED', 'Scene analysis failed')
                return

            # Step 2: Plan actions based on command and scene
            plan = self.plan_actions(command, scene)
            if plan is None:
                self.publish_status('FAILED', 'Action planning failed')
                return

            # Step 3: Execute the plan
            success = self.execute_plan(plan)

            if success:
                self.publish_status('SUCCESS', 'Task completed')
            else:
                self.publish_status('FAILED', 'Execution failed')

        except Exception as e:
            self.get_logger().error(f'Command execution error: {e}')
            self.publish_status('ERROR', str(e))
        finally:
            self.is_executing = False

    def analyze_scene(self) -> Optional[Dict]:
        """Analyze the current scene using Claude Vision."""
        if self.latest_image is None:
            self.get_logger().error('No camera image available')
            return None

        self.publish_status('ANALYZING', 'Understanding scene...')

        try:
            # Encode image
            _, buffer = cv2.imencode('.jpg', self.latest_image,
                                     [cv2.IMWRITE_JPEG_QUALITY, 85])
            image_base64 = base64.b64encode(buffer).decode('utf-8')

            # Call Claude Vision
            response_text = self._call_claude_vision(
                image_base64,
                self.SCENE_UNDERSTANDING_PROMPT
            )

            # Parse response - strip markdown code blocks if present
            clean_response = response_text.strip()
            if clean_response.startswith('```'):
                # Remove markdown code fences
                lines = clean_response.split('\n')
                # Remove first line (```json) and last line (```)
                if lines[0].startswith('```'):
                    lines = lines[1:]
                if lines and lines[-1].strip() == '```':
                    lines = lines[:-1]
                clean_response = '\n'.join(lines)

            self.get_logger().info(f'Claude response (first 300 chars): {clean_response[:300]}')
            scene = json.loads(clean_response)

            # Convert to DetectedObject list
            self.detected_objects = []
            for obj in scene.get('objects', []):
                detected = DetectedObject(
                    name=obj['name'],
                    color=obj.get('color', 'unknown'),
                    position_2d=(obj['position']['x'], obj['position']['y']),
                    confidence=obj.get('confidence', 0.5),
                    size_estimate=obj.get('size', 'medium')
                )
                # Estimate 3D position
                detected.estimated_3d = self._estimate_3d_position(
                    detected.position_2d,
                    detected.size_estimate
                )
                self.detected_objects.append(detected)

            # Cache and publish
            self.scene_cache = scene
            scene_msg = String()
            scene_msg.data = json.dumps(scene, indent=2)
            self.scene_pub.publish(scene_msg)

            # Annotate and publish image
            self._publish_annotated_image(scene)

            self.get_logger().info(f'Scene analysis: {len(self.detected_objects)} objects found')
            for obj in self.detected_objects:
                self.get_logger().info(f'  - {obj.name}: {obj.color}, 3D: {obj.estimated_3d}')

            return scene

        except json.JSONDecodeError as e:
            self.get_logger().error(f'Failed to parse scene analysis: {e}')
            return None
        except Exception as e:
            self.get_logger().error(f'Scene analysis error: {e}')
            return None

    def _estimate_3d_position(self, pos_2d: Tuple[float, float],
                              size: str) -> Tuple[float, float, float]:
        """
        Estimate 3D position from 2D image coordinates.
        Uses geometric reasoning based on camera parameters and assumed object sizes.
        """
        x_img, y_img = pos_2d

        # Estimate distance based on object size and position in image
        # Objects at bottom of image are closer
        size_to_distance = {'small': 0.6, 'medium': 0.5, 'large': 0.4}
        base_distance = size_to_distance.get(size, 0.5)

        # Adjust based on y position (lower in image = closer)
        distance_factor = 0.5 + y_img * 0.5  # 0.5-1.0 range
        estimated_distance = base_distance * distance_factor

        # Convert image coordinates to robot frame
        # X: forward from robot
        x_robot = max(0.25, min(0.7, estimated_distance))

        # Y: left/right (image x -> robot y)
        # Image x=0 is left, x=1 is right
        # Robot y+ is left, y- is right
        y_robot = (0.5 - x_img) * 0.8  # Scale to +-0.4m

        # Z: height (fixed for table-top objects)
        z_robot = 0.1 if y_img > 0.6 else 0.2  # Lower if at bottom of image

        return (round(x_robot, 3), round(y_robot, 3), round(z_robot, 3))

    def plan_actions(self, command: str, scene: Dict) -> Optional[List[ActionPlan]]:
        """Plan actions based on command and scene understanding."""
        self.publish_status('PLANNING', 'Creating action plan...')

        try:
            # Create context for action planning
            context = {
                "command": command,
                "scene": scene,
                "detected_objects": [
                    {
                        "name": obj.name,
                        "color": obj.color,
                        "estimated_position": obj.estimated_3d,
                        "confidence": obj.confidence
                    }
                    for obj in self.detected_objects
                ],
                "robot_state": {
                    "current_joints": {k: round(v, 3) for k, v in self.current_joints.items()
                                      if k in self.ARM_JOINTS}
                }
            }

            # Call Claude for action planning
            response = self._call_claude_text(
                self.ACTION_PLANNING_PROMPT,
                f"Context:\n{json.dumps(context, indent=2)}\n\nCommand: {command}"
            )

            # Strip markdown code blocks if present
            clean_response = response.strip()
            if clean_response.startswith('```'):
                lines = clean_response.split('\n')
                if lines[0].startswith('```'):
                    lines = lines[1:]
                if lines and lines[-1].strip() == '```':
                    lines = lines[:-1]
                clean_response = '\n'.join(lines)

            plan_data = json.loads(clean_response)

            # Publish plan
            plan_msg = String()
            plan_msg.data = json.dumps(plan_data, indent=2)
            self.plan_pub.publish(plan_msg)

            # Convert to ActionPlan objects
            actions = []
            for action_data in plan_data.get('actions', []):
                action_type = ActionType[action_data['action']]
                target_pos = None
                if action_data.get('target_position'):
                    tp = action_data['target_position']
                    target_pos = (tp['x'], tp['y'], tp['z'])

                actions.append(ActionPlan(
                    action_type=action_type,
                    target_position=target_pos,
                    target_object=plan_data.get('target_object'),
                    description=action_data.get('description', '')
                ))

            self.get_logger().info(f'Action plan created: {len(actions)} actions')
            for i, action in enumerate(actions):
                self.get_logger().info(f'  {i+1}. {action.action_type.value}: {action.description}')

            return actions

        except json.JSONDecodeError as e:
            self.get_logger().error(f'Failed to parse action plan: {e}')
            return None
        except Exception as e:
            self.get_logger().error(f'Action planning error: {e}')
            return None

    def execute_plan(self, actions: List[ActionPlan]) -> bool:
        """Execute a sequence of planned actions."""
        self.publish_status('EXECUTING', f'{len(actions)} actions to perform')

        for i, action in enumerate(actions):
            self.get_logger().info(f'Executing action {i+1}/{len(actions)}: {action.action_type.value}')
            self.publish_status('EXECUTING', action.description)

            success = False

            if action.action_type == ActionType.HOME:
                success = self._move_to_named_position('home')

            elif action.action_type == ActionType.LOOK:
                success = self._move_to_named_position('look_table')

            elif action.action_type == ActionType.OPEN_GRIPPER:
                success = self._control_gripper(open_gripper=True)

            elif action.action_type == ActionType.CLOSE_GRIPPER:
                success = self._control_gripper(open_gripper=False)

            elif action.action_type == ActionType.MOVE_ARM:
                if action.target_position:
                    success = self._move_to_position(action.target_position)
                else:
                    self.get_logger().error('MOVE_ARM requires target_position')

            elif action.action_type == ActionType.PICK:
                success = self._execute_pick(action)

            elif action.action_type == ActionType.PLACE:
                success = self._execute_place(action)

            if not success:
                self.get_logger().error(f'Action {action.action_type.value} failed')
                return False

        return True

    def _execute_pick(self, action: ActionPlan) -> bool:
        """Execute a pick action sequence."""
        # Find target object
        target_obj = None
        if action.target_object:
            for obj in self.detected_objects:
                if action.target_object.lower() in obj.name.lower():
                    target_obj = obj
                    break

        if target_obj and target_obj.estimated_3d:
            pick_pos = target_obj.estimated_3d
        elif action.target_position:
            pick_pos = action.target_position
        else:
            self.get_logger().error('No valid pick position')
            return False

        x, y, z = pick_pos

        # Pick sequence
        # 1. Open gripper
        if not self._control_gripper(open_gripper=True):
            return False

        # 2. Move above object
        approach_pos = (x, y, z + 0.15)
        if not self._move_to_cartesian_position(approach_pos):
            return False

        # 3. Move down to object
        if not self._move_to_cartesian_position(pick_pos):
            return False

        # 4. Close gripper
        if not self._control_gripper(open_gripper=False):
            return False

        # 5. Lift object
        lift_pos = (x, y, z + 0.2)
        if not self._move_to_cartesian_position(lift_pos):
            return False

        return True

    def _execute_place(self, action: ActionPlan) -> bool:
        """Execute a place action sequence."""
        if not action.target_position:
            # Default place position
            place_pos = (0.4, 0.3, 0.15)
        else:
            place_pos = action.target_position

        x, y, z = place_pos

        # Place sequence
        # 1. Move above target
        approach_pos = (x, y, z + 0.15)
        if not self._move_to_cartesian_position(approach_pos):
            return False

        # 2. Move down
        if not self._move_to_cartesian_position(place_pos):
            return False

        # 3. Open gripper
        if not self._control_gripper(open_gripper=True):
            return False

        # 4. Retract
        retract_pos = (x, y, z + 0.2)
        if not self._move_to_cartesian_position(retract_pos):
            return False

        return True

    def _move_to_named_position(self, position_name: str) -> bool:
        """Move arm to a named position."""
        if position_name not in self.NAMED_POSITIONS:
            self.get_logger().error(f'Unknown position: {position_name}')
            return False

        joint_positions = self.NAMED_POSITIONS[position_name]
        return self._move_arm_joints(joint_positions)

    def _move_to_cartesian_position(self, position: Tuple[float, float, float]) -> bool:
        """
        Move arm end-effector to a Cartesian position.
        Uses simple IK approximation since we don't have full MoveIt integration.
        """
        x, y, z = position

        # Simple IK approximation for UR5
        # This is a simplified version - full MoveIt would be more accurate

        # Shoulder pan based on y position
        shoulder_pan = math.atan2(y, x)

        # Distance to target in xy plane
        r = math.sqrt(x*x + y*y)

        # UR5 link lengths (approximate)
        l1 = 0.425  # upper arm
        l2 = 0.392  # forearm

        # Adjust z for robot base height
        z_adjusted = z - 0.55  # Robot base is at 0.55m

        # Simple 2-link IK
        d = math.sqrt(r*r + z_adjusted*z_adjusted)
        d = min(d, l1 + l2 - 0.01)  # Clamp to reachable distance

        # Elbow angle
        cos_elbow = (l1*l1 + l2*l2 - d*d) / (2*l1*l2)
        cos_elbow = max(-1, min(1, cos_elbow))
        elbow = math.acos(cos_elbow)

        # Shoulder lift
        alpha = math.atan2(z_adjusted, r)
        cos_beta = (l1*l1 + d*d - l2*l2) / (2*l1*d)
        cos_beta = max(-1, min(1, cos_beta))
        beta = math.acos(cos_beta)
        shoulder_lift = -(alpha + beta)

        # Wrist angles to keep gripper pointing down
        wrist_1 = -(math.pi/2 + shoulder_lift + elbow - math.pi)
        wrist_2 = -math.pi/2
        wrist_3 = 0.0

        joint_positions = [
            shoulder_pan,
            shoulder_lift,
            elbow,
            wrist_1,
            wrist_2,
            wrist_3
        ]

        self.get_logger().info(f'Moving to Cartesian ({x:.2f}, {y:.2f}, {z:.2f})')
        return self._move_arm_joints(joint_positions)

    def _move_to_position(self, position: Tuple[float, float, float]) -> bool:
        """Wrapper for Cartesian movement."""
        return self._move_to_cartesian_position(position)

    def _move_arm_joints(self, joint_positions: List[float], duration: float = 3.0) -> bool:
        """Move arm to specified joint positions."""
        if not self.arm_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Arm action server not available')
            return False

        trajectory = JointTrajectory()
        trajectory.joint_names = self.ARM_JOINTS

        point = JointTrajectoryPoint()
        point.positions = joint_positions
        point.velocities = [0.0] * 6
        point.time_from_start = Duration(sec=int(duration), nanosec=int((duration % 1) * 1e9))
        trajectory.points = [point]

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory

        future = self.arm_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Arm goal rejected')
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=duration + 5.0)

        result = result_future.result()
        return result.status == 4  # SUCCEEDED

    def _control_gripper(self, open_gripper: bool) -> bool:
        """Control gripper open/close."""
        if not self.gripper_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Gripper action server not available')
            return False

        goal = GripperCommand.Goal()
        goal.command.position = 0.0 if open_gripper else 0.8
        goal.command.max_effort = 50.0

        action_name = 'Opening' if open_gripper else 'Closing'
        self.get_logger().info(f'{action_name} gripper...')

        future = self.gripper_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)

        goal_handle = future.result()
        if not goal_handle.accepted:
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=5.0)

        return True

    def _call_claude_vision(self, image_base64: str, system_prompt: str) -> str:
        """Call Claude Vision API."""
        if ANTHROPIC_SDK_AVAILABLE and self.client:
            response = self.client.messages.create(
                model='claude-sonnet-4-20250514',
                max_tokens=2048,
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
                        {'type': 'text', 'text': system_prompt}
                    ]
                }]
            )
            return response.content[0].text
        else:
            return self._call_vision_http(image_base64, system_prompt)

    def _call_claude_text(self, system_prompt: str, user_message: str) -> str:
        """Call Claude text API."""
        if ANTHROPIC_SDK_AVAILABLE and self.client:
            response = self.client.messages.create(
                model='claude-sonnet-4-20250514',
                max_tokens=1024,
                system=system_prompt,
                messages=[{'role': 'user', 'content': user_message}]
            )
            return response.content[0].text
        else:
            return self._call_text_http(system_prompt, user_message)

    def _call_vision_http(self, image_base64: str, prompt: str) -> str:
        """HTTP fallback for vision API."""
        url = 'https://api.anthropic.com/v1/messages'
        headers = {
            'Content-Type': 'application/json',
            'x-api-key': self.api_key,
            'anthropic-version': '2023-06-01'
        }
        data = {
            'model': 'claude-sonnet-4-20250514',
            'max_tokens': 2048,
            'messages': [{
                'role': 'user',
                'content': [
                    {
                        'type': 'image',
                        'source': {'type': 'base64', 'media_type': 'image/jpeg', 'data': image_base64}
                    },
                    {'type': 'text', 'text': prompt}
                ]
            }]
        }

        req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'),
                                     headers=headers, method='POST')
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode('utf-8'))
            return result['content'][0]['text']

    def _call_text_http(self, system_prompt: str, user_message: str) -> str:
        """HTTP fallback for text API."""
        url = 'https://api.anthropic.com/v1/messages'
        headers = {
            'Content-Type': 'application/json',
            'x-api-key': self.api_key,
            'anthropic-version': '2023-06-01'
        }
        data = {
            'model': 'claude-sonnet-4-20250514',
            'max_tokens': 1024,
            'system': system_prompt,
            'messages': [{'role': 'user', 'content': user_message}]
        }

        req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'),
                                     headers=headers, method='POST')
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode('utf-8'))
            return result['content'][0]['text']

    def _publish_annotated_image(self, scene: Dict):
        """Publish image with detected objects annotated."""
        if self.latest_image is None:
            return

        annotated = self.latest_image.copy()
        h, w = annotated.shape[:2]

        for obj in scene.get('objects', []):
            # Get position
            pos = obj.get('position', {})
            cx = int(pos.get('x', 0.5) * w)
            cy = int(pos.get('y', 0.5) * h)

            # Draw marker
            color = (0, 255, 0)  # Green
            cv2.circle(annotated, (cx, cy), 30, color, 2)
            cv2.circle(annotated, (cx, cy), 3, color, -1)

            # Draw label
            label = f"{obj.get('name', 'object')}"
            cv2.putText(annotated, label, (cx - 50, cy - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            conf = obj.get('confidence', 0)
            cv2.putText(annotated, f"{conf:.0%}", (cx - 30, cy - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        try:
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            self.annotated_pub.publish(annotated_msg)
        except Exception as e:
            self.get_logger().error(f'Failed to publish annotated image: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = ClaudeMoveItNode()

    # Use MultiThreadedExecutor for concurrent callback handling
    from rclpy.executors import MultiThreadedExecutor
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Claude + MoveIt node')
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
