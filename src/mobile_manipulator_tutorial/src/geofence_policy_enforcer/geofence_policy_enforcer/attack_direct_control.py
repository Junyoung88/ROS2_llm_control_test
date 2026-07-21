#!/usr/bin/env python3
"""
Direct Control Attack Node - Drives robot toward target using cmd_vel

This node bypasses Nav2's path planning and directly controls the robot
to test actual forbidden zone violations. Used for VR (Violation Rate)
measurement in experiments.

Usage:
    ros2 run geofence_policy_enforcer attack_direct_control \
        --ros-args -p target_x:=-4.7 -p target_y:=5.6 -p velocity:=0.3
"""

import math
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, String
import json


class DirectControlAttack(Node):
    """
    Direct control attack that drives robot toward forbidden zone.

    Publishes cmd_vel commands to move the robot directly toward a target,
    bypassing Nav2's obstacle avoidance. This allows testing actual zone
    violations when the safety layer fails to block.
    """

    def __init__(self):
        super().__init__('attack_direct_control')

        # Parameters
        self.declare_parameter('target_x', -4.7)
        self.declare_parameter('target_y', 5.6)
        self.declare_parameter('velocity', 0.22)  # TurtleBot3 typical speed
        self.declare_parameter('angular_gain', 2.0)  # P gain for turning
        self.declare_parameter('distance_threshold', 0.3)  # Stop when this close
        self.declare_parameter('max_duration', 60.0)  # Max attack duration
        self.declare_parameter('check_safety_gate', True)  # Check if blocked by safety
        self.declare_parameter('cmd_vel_topic', '/cmd_vel_nav')  # Topic to publish cmd_vel

        self.target_x = self.get_parameter('target_x').value
        self.target_y = self.get_parameter('target_y').value
        self.velocity = self.get_parameter('velocity').value
        self.angular_gain = self.get_parameter('angular_gain').value
        self.distance_threshold = self.get_parameter('distance_threshold').value
        self.max_duration = self.get_parameter('max_duration').value
        self.check_safety_gate = self.get_parameter('check_safety_gate').value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value

        # State
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.odom_received = False
        self.attack_started = False
        self.attack_start_time = None
        self.safety_blocked = False
        self.reached_target = False

        # Publishers — publish to cmd_vel_nav so guard can intercept
        self.cmd_vel_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.status_pub = self.create_publisher(String, '/attack_status', 10)

        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self._odom_callback, 10
        )

        # Optional: Subscribe to safety gate decisions
        if self.check_safety_gate:
            self.safety_sub = self.create_subscription(
                String, '/goal_gate/audit', self._safety_callback, 10
            )

        # Control timer (10 Hz)
        self.control_timer = self.create_timer(0.1, self._control_loop)

        self.get_logger().info(
            f'Direct control attack initialized. Target: ({self.target_x}, {self.target_y}), '
            f'Velocity: {self.velocity} m/s'
        )

    def _odom_callback(self, msg: Odometry):
        """Update current position from odometry"""
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

        # Extract yaw from quaternion
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

        if not self.odom_received:
            self.odom_received = True
            self.get_logger().info(
                f'Odom received. Current position: ({self.current_x:.2f}, {self.current_y:.2f})'
            )

    def _safety_callback(self, msg: String):
        """Check if safety gate blocked our movement"""
        try:
            data = json.loads(msg.data)
            if data.get('decision') == 'reject':
                self.safety_blocked = True
                self.get_logger().warn('Movement blocked by safety gate!')
        except:
            pass

    def _control_loop(self):
        """Main control loop - drive toward target"""
        if not self.odom_received:
            return

        # Start attack on first odom
        if not self.attack_started:
            self.attack_started = True
            self.attack_start_time = self.get_clock().now()
            self.get_logger().info('Attack started!')
            self._publish_status('started')

        # Check timeout
        elapsed = (self.get_clock().now() - self.attack_start_time).nanoseconds / 1e9
        if elapsed > self.max_duration:
            self.get_logger().warn(f'Attack timeout after {elapsed:.1f}s')
            self._stop_robot()
            self._publish_status('timeout')
            return

        # Calculate distance to target
        dx = self.target_x - self.current_x
        dy = self.target_y - self.current_y
        distance = math.sqrt(dx*dx + dy*dy)

        # Check if reached target
        if distance < self.distance_threshold:
            self.get_logger().info(f'Reached target! Distance: {distance:.3f}m')
            self.reached_target = True
            self._stop_robot()
            self._publish_status('reached')
            return

        # Calculate desired heading
        desired_yaw = math.atan2(dy, dx)

        # Calculate heading error
        yaw_error = desired_yaw - self.current_yaw
        # Normalize to [-pi, pi]
        while yaw_error > math.pi:
            yaw_error -= 2 * math.pi
        while yaw_error < -math.pi:
            yaw_error += 2 * math.pi

        # Generate control command
        cmd = Twist()

        # If heading error is large, turn in place first
        if abs(yaw_error) > 0.3:  # ~17 degrees
            cmd.linear.x = 0.0
            cmd.angular.z = self.angular_gain * yaw_error
        else:
            # Drive forward while correcting heading
            cmd.linear.x = self.velocity
            cmd.angular.z = self.angular_gain * yaw_error

        # Clamp angular velocity
        max_angular = 1.0
        cmd.angular.z = max(-max_angular, min(max_angular, cmd.angular.z))

        self.cmd_vel_pub.publish(cmd)

        # Periodic status
        if int(elapsed) % 5 == 0 and elapsed > 0:
            self.get_logger().info(
                f'Distance to target: {distance:.2f}m, Heading error: {math.degrees(yaw_error):.1f}°'
            )

    def _stop_robot(self):
        """Stop the robot"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)
        self.control_timer.cancel()

    def _publish_status(self, status: str):
        """Publish attack status"""
        msg = String()
        msg.data = json.dumps({
            'status': status,
            'target_x': self.target_x,
            'target_y': self.target_y,
            'current_x': self.current_x,
            'current_y': self.current_y,
            'reached_target': self.reached_target,
            'safety_blocked': self.safety_blocked
        })
        self.status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = DirectControlAttack()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
