#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Object Detector Node using YOLOv8 or OpenCV DNN

Subscribes to camera images, runs object detection,
and publishes detected objects with bounding boxes.

Published Topics:
    /detected_objects (std_msgs/String): JSON detection data
    /detection_image (sensor_msgs/Image): Annotated image with bounding boxes
    /detection_summary (std_msgs/String): Human-readable summary

Supports two backends:
    1. ultralytics YOLO (preferred, more accurate)
    2. OpenCV DNN with YOLOv4-tiny (fallback, no extra dependencies)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import json
import numpy as np
from typing import List, Dict, Any, Optional
import os
import urllib.request

# Try to import YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

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


class ObjectDetectorNode(Node):
    """ROS2 node for real-time object detection."""

    def __init__(self):
        super().__init__('object_detector')

        # Declare parameters
        self.declare_parameter('model', 'yolov8n.pt')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('image_topic', '/camera/image')
        self.declare_parameter('publish_annotated', True)
        self.declare_parameter('detection_rate', 5.0)
        self.declare_parameter('use_opencv_fallback', True)

        # Get parameters
        model_name = self.get_parameter('model').get_parameter_value().string_value
        self.conf_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.publish_annotated = self.get_parameter('publish_annotated').get_parameter_value().bool_value
        detection_rate = self.get_parameter('detection_rate').get_parameter_value().double_value
        use_opencv_fallback = self.get_parameter('use_opencv_fallback').get_parameter_value().bool_value

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Backend selection
        self.backend = None
        self.model = None
        self.net = None

        if YOLO_AVAILABLE:
            self._init_yolo(model_name)
        elif use_opencv_fallback:
            self._init_opencv_dnn()
        else:
            self.get_logger().error('No detection backend available!')
            self.get_logger().error('Install ultralytics: pip install ultralytics')

        # Rate limiting
        self.min_detection_interval = 1.0 / detection_rate
        self.last_detection_time = 0.0

        # QoS profile - use system default which auto-negotiates
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE
        )

        # Subscriber
        self.image_sub = self.create_subscription(
            Image, image_topic, self.image_callback, sensor_qos
        )

        # Publishers
        self.detection_pub = self.create_publisher(String, '/detected_objects', 10)
        self.annotated_pub = self.create_publisher(Image, '/detection_image', 10)
        self.summary_pub = self.create_publisher(String, '/detection_summary', 10)

        # State
        self.latest_detections: List[Dict[str, Any]] = []
        self.detection_count = 0

        self.get_logger().info(f'Object Detector initialized')
        self.get_logger().info(f'Backend: {self.backend}')
        self.get_logger().info(f'Subscribing to: {image_topic}')
        self.get_logger().info(f'Confidence threshold: {self.conf_threshold}')

    def _init_yolo(self, model_name: str):
        """Initialize YOLO backend."""
        try:
            self.get_logger().info(f'Loading YOLO model: {model_name}')
            self.model = YOLO(model_name)
            self.backend = 'yolo'
            self.get_logger().info('YOLO backend initialized successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to load YOLO: {e}')
            self._init_opencv_dnn()

    def _init_opencv_dnn(self):
        """Initialize OpenCV DNN backend with YOLOv4-tiny."""
        self.get_logger().info('Initializing OpenCV DNN backend...')

        # Model files
        model_dir = os.path.expanduser('~/.cache/object_detector')
        os.makedirs(model_dir, exist_ok=True)

        weights_path = os.path.join(model_dir, 'yolov4-tiny.weights')
        cfg_path = os.path.join(model_dir, 'yolov4-tiny.cfg')

        # Download if not exists
        weights_url = 'https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights'
        cfg_url = 'https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg'

        try:
            if not os.path.exists(weights_path):
                self.get_logger().info('Downloading YOLOv4-tiny weights...')
                urllib.request.urlretrieve(weights_url, weights_path)
                self.get_logger().info('Download complete')

            if not os.path.exists(cfg_path):
                self.get_logger().info('Downloading YOLOv4-tiny config...')
                urllib.request.urlretrieve(cfg_url, cfg_path)

            # Load network
            self.net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

            self.layer_names = self.net.getLayerNames()
            self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

            self.backend = 'opencv_dnn'
            self.get_logger().info('OpenCV DNN backend initialized successfully')

        except Exception as e:
            self.get_logger().error(f'Failed to initialize OpenCV DNN: {e}')
            self.backend = None

    def image_callback(self, msg: Image):
        """Process incoming camera images."""
        if self.backend is None:
            return

        # Rate limiting
        current_time = self.get_clock().now().nanoseconds / 1e9
        if current_time - self.last_detection_time < self.min_detection_interval:
            return
        self.last_detection_time = current_time

        # Debug: log first few image callbacks
        if self.detection_count < 3:
            self.get_logger().info(f'Received image: {msg.width}x{msg.height}, encoding={msg.encoding}')

        try:
            # Convert ROS Image to OpenCV - handle different encodings
            if msg.encoding == 'rgb8':
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            else:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Run detection based on backend
            if self.backend == 'yolo':
                detections, annotated = self._detect_yolo(cv_image)
            else:
                detections, annotated = self._detect_opencv(cv_image)

            # Publish results
            self._publish_detections(detections)

            if self.publish_annotated and annotated is not None:
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

        annotated = results[0].plot()
        self.latest_detections = detections
        return detections, annotated

    def _detect_opencv(self, image: np.ndarray) -> tuple:
        """Run OpenCV DNN detection."""
        height, width = image.shape[:2]

        # Create blob
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)

        # Forward pass
        outputs = self.net.forward(self.output_layers)

        # Process detections
        boxes = []
        confidences = []
        class_ids = []

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

        # Non-maximum suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, 0.4)

        detections = []
        annotated = image.copy()

        for i in indices:
            idx = i if isinstance(i, int) else i[0]
            x, y, w, h = boxes[idx]
            class_id = class_ids[idx]
            conf = confidences[idx]

            if class_id < len(COCO_CLASSES):
                class_name = COCO_CLASSES[class_id]
            else:
                class_name = f"class_{class_id}"

            detections.append(self._create_detection(
                class_name, class_id, conf,
                x, y, x + w, y + h,
                width, height
            ))

            # Draw bounding box
            color = (0, 255, 0)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(annotated, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        self.latest_detections = detections
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
                'width': x2 - x1,
                'height': y2 - y1,
                'center_x': int(center_x),
                'center_y': int(center_y)
            },
            'normalized': {
                'center_x': round(center_x / img_width, 3),
                'center_y': round(center_y / img_height, 3)
            }
        }

    def _publish_detections(self, detections: List[Dict[str, Any]]):
        """Publish detection results."""
        # JSON data
        detection_msg = String()
        detection_msg.data = json.dumps({
            'timestamp': self.get_clock().now().nanoseconds,
            'count': len(detections),
            'detections': detections
        })
        self.detection_pub.publish(detection_msg)

        # Summary
        if detections:
            items = [f"{d['class']}({d['confidence']:.0%})" for d in detections]
            summary = f"Detected {len(detections)}: {', '.join(items)}"
        else:
            summary = "No objects detected"

        summary_msg = String()
        summary_msg.data = summary
        self.summary_pub.publish(summary_msg)

        if self.detection_count % 30 == 0 and detections:
            self.get_logger().info(summary)


def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Object Detector')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
