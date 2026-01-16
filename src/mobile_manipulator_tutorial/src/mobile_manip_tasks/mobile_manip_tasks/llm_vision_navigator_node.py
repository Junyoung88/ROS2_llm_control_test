#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Vision Navigator Node

Integrates object detection with LLM-based navigation.
Allows natural language commands like:
- "go to the person"
- "navigate to the chair"
- "move towards the detected box"

Subscribes to:
    /camera/image: Camera images for object detection
    /llm_command: Natural language navigation commands

Publishes to:
    /detected_objects: JSON detection data
    /detection_image: Annotated image
    /delivery_status: Navigation status
"""

import os
import json
import math
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from tf_transformations import quaternion_from_euler
from cv_bridge import CvBridge
import cv2
import numpy as np
from typing import List, Dict, Any, Optional

# Try to import YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# Try to import Anthropic SDK
try:
    import anthropic
    ANTHROPIC_SDK_AVAILABLE = True
except ImportError:
    ANTHROPIC_SDK_AVAILABLE = False
    import urllib.request
    import urllib.error

# COCO class names
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]


class LLMVisionNavigatorNode(Node):
    """ROS2 node integrating object detection with LLM navigation."""

    SYSTEM_PROMPT_BASE = """You are a robot navigation assistant with vision capabilities.
You can see objects through the robot's camera and navigate to them.

Your task is to interpret natural language commands and output a navigation goal as JSON.
You must respond with ONLY a valid JSON object, no other text.

JSON format:
{
  "frame_id": "map",
  "x": <float>,
  "y": <float>,
  "yaw": <float in radians>,
  "target_object": "<object name or null>"
}

The environment is a warehouse/lab space. Coordinate ranges:
- x: -10 to 10 meters
- y: -10 to 10 meters
- yaw: -3.14159 to 3.14159 radians

Reference points (warehouse layout):
- "origin"/"home"/"start": x=0, y=0, yaw=0
- "pick station"/"pickup": x=-4.7, y=5.6, yaw=3.14159
- "drop station"/"delivery area"/"delivery": x=-0.3, y=1.2, yaw=3.14159
- "stairs"/"staircase"/"계단": x=3.0, y=2.0, yaw=0  (right side of warehouse)
- "shelves"/"storage"/"선반": x=-5.0, y=3.0, yaw=1.57  (left side with blue racks)
- "center"/"middle"/"중앙": x=0, y=3.0, yaw=0
- "pillar"/"column"/"기둥": x=1.0, y=2.5, yaw=0

Direction commands:
- "forward"/"앞으로": move +2m in current facing direction
- "backward"/"뒤로": move -2m in current facing direction
- "left"/"왼쪽": turn left 90 degrees
- "right"/"오른쪽": turn right 90 degrees

"""

    VISION_PROMPT_ADDITION = """
DETECTED OBJECTS (from robot's camera):
{detected_objects}

When the user asks to go to a detected object:
1. Use the object's estimated position from the detection data
2. Set target_object to the object class name
3. Navigate to approximately 1 meter in front of the object

If no objects are detected or the requested object isn't visible,
respond with the nearest known location or inform that the object is not detected.
"""

    def __init__(self):
        super().__init__('llm_vision_navigator')

        # Declare parameters
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('detection_rate', 5.0)
        self.declare_parameter('image_topic', '/camera/image')
        self.declare_parameter('camera_hfov', 1.047)  # 60 degrees in radians
        self.declare_parameter('estimated_depth', 3.0)  # Default depth estimate in meters

        # Get parameters
        self.conf_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        detection_rate = self.get_parameter('detection_rate').get_parameter_value().double_value
        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.camera_hfov = self.get_parameter('camera_hfov').get_parameter_value().double_value
        self.estimated_depth = self.get_parameter('estimated_depth').get_parameter_value().double_value

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
            self.get_logger().info('Using HTTP fallback for Claude API')

        # CV Bridge
        self.bridge = CvBridge()

        # Initialize object detection backend
        self._init_detection_backend()

        # Detection state
        self.latest_detections: List[Dict[str, Any]] = []
        self.detection_count = 0
        self.min_detection_interval = 1.0 / detection_rate
        self.last_detection_time = 0.0

        # Robot state (estimated from odometry)
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0

        # Navigation state
        self._navigating = False
        self._current_goal_handle = None

        # QoS for camera
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE
        )

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, image_topic, self.image_callback, sensor_qos
        )
        self.command_sub = self.create_subscription(
            String, '/llm_command', self.command_callback, 10
        )

        # Publishers
        self.detection_pub = self.create_publisher(String, '/detected_objects', 10)
        self.annotated_pub = self.create_publisher(Image, '/detection_image', 10)
        self.status_pub = self.create_publisher(String, '/delivery_status', 10)

        # Nav2 action client
        self._nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self.publish_status('IDLE')
        self.get_logger().info('LLM Vision Navigator initialized')
        self.get_logger().info(f'Detection backend: {self.backend}')
        self.get_logger().info(f'Subscribing to: {image_topic}')
        self.get_logger().info('Listening for commands on /llm_command')

    def _init_detection_backend(self):
        """Initialize object detection backend."""
        self.backend = None
        self.model = None
        self.net = None

        if YOLO_AVAILABLE:
            try:
                self.get_logger().info('Loading YOLOv8 model...')
                self.model = YOLO('yolov8n.pt')
                self.backend = 'yolo'
                self.get_logger().info('YOLOv8 backend initialized')
                return
            except Exception as e:
                self.get_logger().warn(f'Failed to load YOLO: {e}')

        # Fallback to OpenCV DNN
        self._init_opencv_dnn()

    def _init_opencv_dnn(self):
        """Initialize OpenCV DNN backend."""
        import urllib.request

        model_dir = os.path.expanduser('~/.cache/object_detector')
        os.makedirs(model_dir, exist_ok=True)

        weights_path = os.path.join(model_dir, 'yolov4-tiny.weights')
        cfg_path = os.path.join(model_dir, 'yolov4-tiny.cfg')

        try:
            if not os.path.exists(weights_path):
                self.get_logger().info('Downloading YOLOv4-tiny weights...')
                urllib.request.urlretrieve(
                    'https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights',
                    weights_path
                )

            if not os.path.exists(cfg_path):
                self.get_logger().info('Downloading YOLOv4-tiny config...')
                urllib.request.urlretrieve(
                    'https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg',
                    cfg_path
                )

            self.net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

            self.layer_names = self.net.getLayerNames()
            self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

            self.backend = 'opencv_dnn'
            self.get_logger().info('OpenCV DNN backend initialized')

        except Exception as e:
            self.get_logger().error(f'Failed to initialize detection: {e}')
            self.backend = None

    def publish_status(self, status: str):
        """Publish navigation status."""
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)
        self.get_logger().info(f'Status: {status}')

    def image_callback(self, msg: Image):
        """Process camera images for object detection."""
        if self.backend is None:
            return

        current_time = self.get_clock().now().nanoseconds / 1e9
        if current_time - self.last_detection_time < self.min_detection_interval:
            return
        self.last_detection_time = current_time

        try:
            # Convert image
            if msg.encoding == 'rgb8':
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            else:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Run detection
            if self.backend == 'yolo':
                detections, annotated = self._detect_yolo(cv_image)
            else:
                detections, annotated = self._detect_opencv(cv_image)

            # Estimate world positions for detected objects
            self._estimate_object_positions(detections, cv_image.shape)

            self.latest_detections = detections
            self._publish_detections(detections)

            # Publish annotated image
            if annotated is not None:
                annotated_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
                annotated_msg.header = msg.header
                self.annotated_pub.publish(annotated_msg)

            self.detection_count += 1

        except Exception as e:
            self.get_logger().error(f'Detection error: {e}')

    def _detect_yolo(self, image: np.ndarray) -> tuple:
        """Run YOLO detection."""
        results = self.model(image, conf=self.conf_threshold, verbose=False)
        detections = []
        height, width = image.shape[:2]

        if results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                class_name = self.model.names[cls_id]

                detections.append(self._create_detection(
                    class_name, cls_id, conf,
                    int(x1), int(y1), int(x2), int(y2),
                    width, height
                ))

        return detections, results[0].plot()

    def _detect_opencv(self, image: np.ndarray) -> tuple:
        """Run OpenCV DNN detection."""
        height, width = image.shape[:2]

        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)

        boxes, confidences, class_ids = [], [], []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > self.conf_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, 0.4)

        detections = []
        annotated = image.copy()

        for i in indices:
            idx = i if isinstance(i, int) else i[0]
            x, y, w, h = boxes[idx]
            class_id = class_ids[idx]
            conf = confidences[idx]
            class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"class_{class_id}"

            detections.append(self._create_detection(
                class_name, class_id, conf,
                x, y, x + w, y + h,
                width, height
            ))

            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(annotated, f"{class_name}: {conf:.2f}", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return detections, annotated

    def _create_detection(self, class_name: str, class_id: int, confidence: float,
                          x1: int, y1: int, x2: int, y2: int,
                          img_width: int, img_height: int) -> Dict[str, Any]:
        """Create detection dictionary."""
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        return {
            'class': class_name,
            'class_id': class_id,
            'confidence': round(confidence, 3),
            'bbox': {
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'width': x2 - x1, 'height': y2 - y1,
                'center_x': int(center_x), 'center_y': int(center_y)
            },
            'normalized': {
                'center_x': round(center_x / img_width, 3),
                'center_y': round(center_y / img_height, 3)
            },
            'estimated_position': None  # Will be filled by _estimate_object_positions
        }

    def _estimate_object_positions(self, detections: List[Dict], image_shape: tuple):
        """Estimate world positions of detected objects based on image position."""
        height, width = image_shape[:2]

        for det in detections:
            # Get normalized x position (-0.5 to 0.5, where 0 is center)
            norm_x = det['normalized']['center_x'] - 0.5

            # Estimate angle from center of camera view
            angle_offset = norm_x * self.camera_hfov

            # Estimate object position relative to robot
            # Using simple projection (assumes flat ground, object at estimated depth)
            obj_angle = self.robot_yaw + angle_offset
            obj_x = self.robot_x + self.estimated_depth * math.cos(obj_angle)
            obj_y = self.robot_y + self.estimated_depth * math.sin(obj_angle)

            det['estimated_position'] = {
                'x': round(obj_x, 2),
                'y': round(obj_y, 2),
                'angle_offset': round(math.degrees(angle_offset), 1),
                'estimated_distance': self.estimated_depth
            }

    def _publish_detections(self, detections: List[Dict]):
        """Publish detection results."""
        msg = String()
        msg.data = json.dumps({
            'timestamp': self.get_clock().now().nanoseconds,
            'count': len(detections),
            'detections': detections
        })
        self.detection_pub.publish(msg)

        if self.detection_count % 30 == 0 and detections:
            items = [f"{d['class']}({d['confidence']:.0%})" for d in detections]
            self.get_logger().info(f"Detected: {', '.join(items)}")

    def command_callback(self, msg: String):
        """Handle natural language navigation commands."""
        command = msg.data.strip()
        if not command:
            return

        self.get_logger().info(f'Received command: "{command}"')

        if self._navigating:
            self.get_logger().warn('Navigation in progress, ignoring command')
            return

        # Build prompt with current detections
        nav_goal = self.call_claude_api(command)
        if nav_goal is None:
            self.publish_status('FAILED')
            return

        self.send_nav_goal(nav_goal)

    def _build_system_prompt(self) -> str:
        """Build system prompt including detected objects."""
        prompt = self.SYSTEM_PROMPT_BASE

        if self.latest_detections:
            objects_info = []
            for det in self.latest_detections:
                pos = det.get('estimated_position', {})
                objects_info.append(
                    f"- {det['class']} (confidence: {det['confidence']:.0%}, "
                    f"estimated position: x={pos.get('x', '?')}, y={pos.get('y', '?')})"
                )
            detected_str = '\n'.join(objects_info)
        else:
            detected_str = "No objects currently detected."

        prompt += self.VISION_PROMPT_ADDITION.format(detected_objects=detected_str)
        prompt += "\nOutput ONLY the JSON object, nothing else."

        return prompt

    def call_claude_api(self, command: str) -> Optional[Dict]:
        """Call Claude API to interpret command."""
        if not self.api_key:
            self.get_logger().error('No API key')
            return None

        system_prompt = self._build_system_prompt()

        try:
            if ANTHROPIC_SDK_AVAILABLE and self.client:
                response = self.client.messages.create(
                    model='claude-sonnet-4-20250514',
                    max_tokens=256,
                    system=system_prompt,
                    messages=[{'role': 'user', 'content': command}]
                )
                response_text = response.content[0].text
            else:
                response_text = self._call_http(command, system_prompt)

            # Parse JSON
            nav_goal = json.loads(response_text)

            required = ['frame_id', 'x', 'y', 'yaw']
            for field in required:
                if field not in nav_goal:
                    self.get_logger().error(f'Missing field: {field}')
                    return None

            target = nav_goal.get('target_object', 'none')
            self.get_logger().info(
                f'Claude response: x={nav_goal["x"]:.2f}, y={nav_goal["y"]:.2f}, '
                f'target={target}'
            )
            return nav_goal

        except json.JSONDecodeError as e:
            self.get_logger().error(f'JSON parse error: {e}')
            return None
        except Exception as e:
            self.get_logger().error(f'API error: {e}')
            return None

    def _call_http(self, command: str, system_prompt: str) -> str:
        """Call Claude API via HTTP."""
        url = 'https://api.anthropic.com/v1/messages'
        headers = {
            'Content-Type': 'application/json',
            'x-api-key': self.api_key,
            'anthropic-version': '2023-06-01'
        }
        data = {
            'model': 'claude-sonnet-4-20250514',
            'max_tokens': 256,
            'system': system_prompt,
            'messages': [{'role': 'user', 'content': command}]
        }

        req = urllib.request.Request(
            url, data=json.dumps(data).encode('utf-8'),
            headers=headers, method='POST'
        )

        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode('utf-8'))
            return result['content'][0]['text']

    def send_nav_goal(self, nav_goal: Dict):
        """Send navigation goal to Nav2."""
        if not self._nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Nav2 action server not available')
            self.publish_status('FAILED')
            return

        pose = PoseStamped()
        pose.header.frame_id = nav_goal.get('frame_id', 'map')
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(nav_goal['x'])
        pose.pose.position.y = float(nav_goal['y'])
        pose.pose.position.z = 0.0

        q = quaternion_from_euler(0, 0, float(nav_goal['yaw']))
        pose.pose.orientation.x = q[0]
        pose.pose.orientation.y = q[1]
        pose.pose.orientation.z = q[2]
        pose.pose.orientation.w = q[3]

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose

        target = nav_goal.get('target_object', 'position')
        self.get_logger().info(
            f'Navigating to {target}: x={pose.pose.position.x:.2f}, y={pose.pose.position.y:.2f}'
        )

        self._navigating = True
        self.publish_status('MOVING')

        send_goal_future = self._nav_client.send_goal_async(
            goal_msg, feedback_callback=self.feedback_callback
        )
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """Handle goal acceptance."""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Navigation goal rejected')
            self._navigating = False
            self.publish_status('FAILED')
            return

        self.get_logger().info('Navigation goal accepted')
        self._current_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def feedback_callback(self, feedback_msg):
        """Handle navigation feedback."""
        remaining = feedback_msg.feedback.distance_remaining
        if remaining > 0.5:
            self.get_logger().debug(f'Distance remaining: {remaining:.2f}m')

    def result_callback(self, future):
        """Handle navigation result."""
        result = future.result()
        self._navigating = False
        self._current_goal_handle = None

        if result.status == 4:  # SUCCEEDED
            self.get_logger().info('Navigation SUCCEEDED')
            self.publish_status('DONE')
        else:
            self.get_logger().error(f'Navigation FAILED (status: {result.status})')
            self.publish_status('FAILED')


def main(args=None):
    rclpy.init(args=args)
    node = LLMVisionNavigatorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
