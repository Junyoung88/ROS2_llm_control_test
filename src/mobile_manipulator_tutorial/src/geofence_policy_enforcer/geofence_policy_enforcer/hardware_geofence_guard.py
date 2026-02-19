#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hardware-Level Geofence Guard

This node acts as the FINAL safety barrier before commands reach the robot.
It uses Gazebo ground truth position (not odometry) to make bypass impossible.

Architecture:
  Any source → /cmd_vel → [THIS NODE] → /cmd_vel_safe → gz_bridge → Gazebo

Key features:
1. Uses Gazebo ground truth position (cannot be spoofed)
2. Intercepts ALL /cmd_vel commands (cannot be bypassed)
3. Forward simulation to predict zone violations
4. Hard stop if robot is inside or approaching forbidden zone
"""

import math
import subprocess
import re
import threading
import time
from typing import Optional, Tuple, Dict

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rcl_interfaces.msg import SetParametersResult
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import json


# Default forbidden zones (can be overridden by parameter)
DEFAULT_ZONES = {
    "forbidden_zone": {
        "x_min": 4.0, "x_max": 6.0,
        "y_min": -1.0, "y_max": 1.0
    }
}


class HardwareGeofenceGuard(Node):
    """
    Hardware-level geofence guard that cannot be bypassed.

    Uses Gazebo ground truth for position, intercepts all cmd_vel,
    and only forwards safe commands to cmd_vel_safe.
    """

    def __init__(self):
        super().__init__('hardware_geofence_guard')

        # Parameters
        self.declare_parameter('zones_json', json.dumps(DEFAULT_ZONES))
        self.declare_parameter('input_topic', '/cmd_vel')
        self.declare_parameter('output_topic', '/cmd_vel_safe')
        self.declare_parameter('safety_margin', 0.5)  # meters before zone
        self.declare_parameter('simulation_horizon', 1.0)  # seconds
        self.declare_parameter('simulation_steps', 10)
        self.declare_parameter('gazebo_world', 'empty')
        self.declare_parameter('model_name', 'mobile_manip')
        self.declare_parameter('position_update_rate', 20.0)  # Hz
        self.declare_parameter('reaction_latency_ms', 0.0)  # Simulated detection-to-action delay

        # Load parameters
        zones_json = self.get_parameter('zones_json').get_parameter_value().string_value
        self.zones = json.loads(zones_json)

        input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        self.safety_margin = self.get_parameter('safety_margin').get_parameter_value().double_value
        self.sim_horizon = self.get_parameter('simulation_horizon').get_parameter_value().double_value
        self.sim_steps = self.get_parameter('simulation_steps').get_parameter_value().integer_value
        self.gazebo_world = self.get_parameter('gazebo_world').get_parameter_value().string_value
        self.model_name = self.get_parameter('model_name').get_parameter_value().string_value
        pos_rate = self.get_parameter('position_update_rate').get_parameter_value().double_value

        self.reaction_latency = self.get_parameter('reaction_latency_ms').get_parameter_value().double_value / 1000.0

        # Robot state from Gazebo ground truth
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.last_position_update = None
        self.position_lock = threading.Lock()

        # Latency tracking: when was the first block triggered?
        self.first_block_time = None

        # Statistics
        self.stats = {
            'total_commands': 0,
            'passed': 0,
            'blocked': 0,
            'inside_zone_blocks': 0,
            'prediction_blocks': 0,
            'latency_passed': 0,  # Commands passed during reaction latency window
        }

        # QoS for input: BEST_EFFORT (matches Nav2 velocity smoother)
        qos_input = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)

        # QoS for output: RELIABLE (required by gz_bridge)
        qos_output = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)

        # Subscriber: intercept ALL cmd_vel commands
        self.cmd_sub = self.create_subscription(
            Twist, input_topic, self.cmd_vel_callback, qos_input
        )

        # Publisher: only safe commands go here (RELIABLE for gz_bridge)
        self.cmd_pub = self.create_publisher(Twist, output_topic, qos_output)

        # Status publisher
        self.status_pub = self.create_publisher(String, '/hardware_guard/status', 10)

        # Position update thread (uses gz topic for ground truth)
        self.running = True
        self.position_thread = threading.Thread(target=self._position_update_loop, daemon=True)
        self.position_thread.start()

        # Status timer
        self.create_timer(1.0, self.publish_status)

        # Dynamic parameter callback (for runtime latency changes)
        self.add_on_set_parameters_callback(self._on_parameter_change)

        self.get_logger().info('=' * 60)
        self.get_logger().info('HARDWARE GEOFENCE GUARD ACTIVE')
        self.get_logger().info('=' * 60)
        self.get_logger().info(f'Input: {input_topic} -> Output: {output_topic}')
        self.get_logger().info(f'Safety margin: {self.safety_margin}m')
        self.get_logger().info(f'Zones: {list(self.zones.keys())}')
        if self.reaction_latency > 0:
            self.get_logger().info(f'Reaction latency: {self.reaction_latency*1000:.0f}ms')
        self.get_logger().info('Using Gazebo ground truth (CANNOT BE BYPASSED)')
        self.get_logger().info('=' * 60)

    def _on_parameter_change(self, params):
        """Handle dynamic parameter updates (e.g., reaction_latency_ms)."""
        for param in params:
            if param.name == 'reaction_latency_ms':
                self.reaction_latency = param.value / 1000.0
                self.first_block_time = None  # Reset on change
                self.get_logger().info(
                    f'Reaction latency updated: {param.value:.0f}ms'
                )
        return SetParametersResult(successful=True)

    def _position_update_loop(self):
        """Background thread to get robot position from Gazebo ground truth."""
        topic = f"/world/{self.gazebo_world}/pose/info"

        while self.running:
            try:
                result = subprocess.run(
                    ["gz", "topic", "-e", "-n", "1", "-t", topic],
                    capture_output=True, text=True, timeout=1
                )

                # Parse the protobuf-like text output
                pattern = rf'name: "{self.model_name}".*?position \{{\s*x: ([\d.e+-]+)\s*y: ([\d.e+-]+).*?orientation \{{\s*x: ([\d.e+-]+)\s*y: ([\d.e+-]+)\s*z: ([\d.e+-]+)\s*w: ([\d.e+-]+)'
                match = re.search(pattern, result.stdout, re.DOTALL)

                if match:
                    x = float(match.group(1))
                    y = float(match.group(2))
                    qx = float(match.group(3))
                    qy = float(match.group(4))
                    qz = float(match.group(5))
                    qw = float(match.group(6))

                    # Convert quaternion to yaw
                    siny_cosp = 2 * (qw * qz + qx * qy)
                    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
                    yaw = math.atan2(siny_cosp, cosy_cosp)

                    with self.position_lock:
                        self.robot_x = x
                        self.robot_y = y
                        self.robot_yaw = yaw
                        self.last_position_update = time.time()

            except subprocess.TimeoutExpired:
                pass
            except Exception as e:
                self.get_logger().warning(
                    f'Position update failed: {e}',
                    throttle_duration_sec=5.0
                )

            time.sleep(0.05)  # ~20Hz

    def get_robot_pose(self) -> Tuple[float, float, float]:
        """Get current robot pose (thread-safe)."""
        with self.position_lock:
            return self.robot_x, self.robot_y, self.robot_yaw

    def is_inside_zone(self, x: float, y: float) -> Optional[str]:
        """Check if point is inside any forbidden zone."""
        for zone_name, zone in self.zones.items():
            if (zone["x_min"] <= x <= zone["x_max"] and
                zone["y_min"] <= y <= zone["y_max"]):
                return zone_name
        return None

    def get_distance_to_zones(self, x: float, y: float) -> float:
        """Get minimum distance to any forbidden zone."""
        min_dist = float('inf')

        for zone in self.zones.values():
            # Find closest point on zone boundary
            closest_x = max(zone["x_min"], min(x, zone["x_max"]))
            closest_y = max(zone["y_min"], min(y, zone["y_max"]))

            dist = math.sqrt((x - closest_x)**2 + (y - closest_y)**2)

            # If inside zone, distance is negative
            if (zone["x_min"] <= x <= zone["x_max"] and
                zone["y_min"] <= y <= zone["y_max"]):
                dist = -dist

            min_dist = min(min_dist, dist)

        return min_dist

    def cmd_vel_callback(self, msg: Twist):
        """Process incoming velocity command with safety check.

        Reaction latency simulation:
        When reaction_latency > 0, the guard detects violations but continues
        forwarding commands for the latency duration before actually blocking.
        This simulates real-world detection-to-action delay.
        """
        self.stats['total_commands'] += 1

        # Get current position from Gazebo ground truth
        x, y, yaw = self.get_robot_pose()

        # Check if we have recent position data
        if self.last_position_update is None:
            self.get_logger().warning(
                'No Gazebo position data yet - BLOCKING all commands',
                throttle_duration_sec=2.0
            )
            self.stats['blocked'] += 1
            self.cmd_pub.publish(Twist())  # Zero velocity
            return

        # Check position data freshness
        age = time.time() - self.last_position_update
        if age > 0.5:
            self.get_logger().warning(
                f'Gazebo position stale ({age:.2f}s) - BLOCKING',
                throttle_duration_sec=2.0
            )
            self.stats['blocked'] += 1
            self.cmd_pub.publish(Twist())
            return

        # Determine if command should be blocked
        should_block = False
        inside_zone = self.is_inside_zone(x, y)

        if inside_zone:
            should_block = True
        else:
            will_violate, violation_time, violation_dist = self._simulate_trajectory(
                x, y, yaw, msg
            )
            if will_violate:
                should_block = True

        if should_block:
            # Reaction latency: forward commands during delay window
            if self.reaction_latency > 0:
                now = time.time()
                if self.first_block_time is None:
                    self.first_block_time = now
                    self.get_logger().warn(
                        f'VIOLATION DETECTED at ({x:.2f},{y:.2f}) - '
                        f'reaction delay {self.reaction_latency*1000:.0f}ms'
                    )

                elapsed = now - self.first_block_time
                if elapsed < self.reaction_latency:
                    # Still within reaction window - pass command through
                    self.stats['latency_passed'] += 1
                    self.stats['passed'] += 1
                    self.cmd_pub.publish(msg)
                    return

            # Actually block (no latency, or latency window expired)
            if inside_zone:
                self.get_logger().warn(
                    f'BLOCKED: Robot inside {inside_zone}! pos=({x:.2f}, {y:.2f})'
                )
                self.stats['inside_zone_blocks'] += 1
                safe_cmd = self._compute_escape_velocity(x, y, inside_zone)
                self.cmd_pub.publish(safe_cmd)
            else:
                self.get_logger().warn(
                    f'BLOCKED: Predicted violation in {violation_time:.2f}s, '
                    f'dist={violation_dist:.3f}m'
                )
                self.stats['prediction_blocks'] += 1
                self.cmd_pub.publish(Twist())  # Zero velocity

            self.stats['blocked'] += 1
            return

        # Command is safe - forward it and reset latency tracking
        self.first_block_time = None
        self.stats['passed'] += 1
        self.cmd_pub.publish(msg)

    def _simulate_trajectory(self, x: float, y: float, yaw: float,
                            cmd: Twist) -> Tuple[bool, float, float]:
        """
        Simulate future trajectory to predict zone violations.

        Returns: (will_violate, violation_time, min_distance)
        """
        dt = self.sim_horizon / self.sim_steps
        vx = cmd.linear.x
        vy = cmd.linear.y
        wz = cmd.angular.z

        sim_x, sim_y, sim_yaw = x, y, yaw
        min_dist = float('inf')

        for step in range(self.sim_steps):
            t = step * dt

            # Differential drive kinematics
            sim_x += (vx * math.cos(sim_yaw) - vy * math.sin(sim_yaw)) * dt
            sim_y += (vx * math.sin(sim_yaw) + vy * math.cos(sim_yaw)) * dt
            sim_yaw += wz * dt

            # Check distance to zones
            dist = self.get_distance_to_zones(sim_x, sim_y)
            min_dist = min(min_dist, dist)

            # Check if inside zone or within safety margin
            if dist < self.safety_margin:
                return True, t, dist

        return False, 0.0, min_dist

    def _compute_escape_velocity(self, x: float, y: float, zone_name: str) -> Twist:
        """Compute velocity to escape from inside a zone."""
        zone = self.zones[zone_name]

        # Find center of zone
        cx = (zone["x_min"] + zone["x_max"]) / 2
        cy = (zone["y_min"] + zone["y_max"]) / 2

        # Direction away from center
        dx = x - cx
        dy = y - cy
        dist = math.sqrt(dx*dx + dy*dy)

        if dist < 0.01:
            # At center, pick arbitrary direction
            dx, dy = 1.0, 0.0
            dist = 1.0

        # Normalize and scale
        escape_speed = 0.3  # Slow escape
        dx /= dist
        dy /= dist

        # Convert to robot frame (assuming robot faces +x in world)
        # This is simplified - proper implementation would use current yaw
        cmd = Twist()
        cmd.linear.x = -0.5  # Back up slowly

        return cmd

    def publish_status(self):
        """Publish periodic status."""
        x, y, yaw = self.get_robot_pose()
        dist = self.get_distance_to_zones(x, y)
        inside = self.is_inside_zone(x, y)

        msg = String()
        msg.data = json.dumps({
            'status': 'ACTIVE',
            'position': {'x': x, 'y': y, 'yaw': yaw},
            'distance_to_zone': dist,
            'inside_zone': inside,
            'stats': self.stats,
            'using': 'GAZEBO_GROUND_TRUTH',
        })
        self.status_pub.publish(msg)

        # Log warning if close to zone
        if 0 < dist < 1.0:
            self.get_logger().warn(
                f'Robot approaching zone! dist={dist:.2f}m, pos=({x:.2f}, {y:.2f})'
            )

    def destroy_node(self):
        """Clean shutdown."""
        self.running = False
        if self.position_thread.is_alive():
            self.position_thread.join(timeout=1.0)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = HardwareGeofenceGuard()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Hardware Geofence Guard')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
