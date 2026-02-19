#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cmd Vel Guard Node - Control Policy Enforcement Point.

Intercepts cmd_vel commands and enforces safety using method-specific approaches:

- geofence: Forward simulation with uncertainty-aware margin (our method)
- cbf: Real-time Control Barrier Function with velocity modification (Ames et al., TAC 2017)
- ssm: Real-time Speed and Separation Monitoring with gradual deceleration (ISO 15066)
- selp: Forward simulation with LTL automaton checking
- static_margin: Forward simulation with fixed margin
"""

import math
import json
import time
from enum import Enum
from typing import Optional, Tuple, List

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import Twist, PoseStamped, TransformStamped
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from tf2_ros import TransformListener, Buffer

from .geofence_core import GeofencePolicy, PolicyAction, ZoneType
from .safety_baselines import (
    SELPSpatialSafety, CBFSpatialSafety, StaticMarginSafety, SSMSpatialSafety,
    SafetyDecision
)


class OverrideStrategy(Enum):
    """Velocity override strategies when violation is predicted."""
    STOP = "stop"           # Zero all velocities
    SCALE = "scale"         # Scale down velocity
    REDIRECT = "redirect"   # Redirect away from danger


class CmdVelGuardNode(Node):
    """
    Velocity command filter with multiple safety methods.

    Each method uses its native approach:
    - geofence: Forward simulation
    - cbf: Real-time barrier function with velocity projection
    - ssm: Real-time speed-distance monitoring with gradual slowdown
    """

    def __init__(self):
        super().__init__('cmd_vel_guard')

        # Parameters
        self.declare_parameter('geofence_config', '')
        self.declare_parameter('input_topic', '/cmd_vel_raw')
        self.declare_parameter('output_topic', '/cmd_vel')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('simulation_horizon', 2.0)  # seconds (for geofence)
        self.declare_parameter('simulation_steps', 20)
        self.declare_parameter('override_strategy', 'stop')
        self.declare_parameter('min_distance_threshold', 0.3)  # meters
        self.declare_parameter('scale_factor_decay', 0.8)
        self.declare_parameter('safety_method', 'geofence')
        self.declare_parameter('simulated_comm_latency_ms', 0.0)

        # CBF specific parameters
        self.declare_parameter('cbf_alpha', 1.0)  # Class-K function parameter

        # SSM specific parameters
        self.declare_parameter('ssm_min_velocity', 0.05)  # Minimum velocity before full stop

        # Get parameters first (needed by _init_safety_baselines)
        input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value

        self.sim_horizon = self.get_parameter('simulation_horizon').get_parameter_value().double_value
        self.sim_steps = self.get_parameter('simulation_steps').get_parameter_value().integer_value
        self.min_dist_threshold = self.get_parameter('min_distance_threshold').get_parameter_value().double_value
        self.scale_decay = self.get_parameter('scale_factor_decay').get_parameter_value().double_value
        self.cbf_alpha = self.get_parameter('cbf_alpha').get_parameter_value().double_value
        self.ssm_min_velocity = self.get_parameter('ssm_min_velocity').get_parameter_value().double_value
        self.comm_latency_sec = self.get_parameter('simulated_comm_latency_ms').get_parameter_value().double_value / 1000.0

        strategy_str = self.get_parameter('override_strategy').get_parameter_value().string_value
        self.override_strategy = OverrideStrategy(strategy_str)

        # Load geofence policy
        config_path = self.get_parameter('geofence_config').get_parameter_value().string_value
        self.policy = GeofencePolicy(config_path if config_path else None)

        # Safety method
        self.safety_method = self.get_parameter('safety_method').get_parameter_value().string_value

        # Initialize safety baselines based on method
        self._init_safety_baselines()

        # TF buffer for robot pose
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Current robot state
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.robot_vx = 0.0  # Current velocity from odom
        self.robot_vy = 0.0
        self.last_odom_time = None

        # Goal authorization state (geofence runtime feature)
        # Tracks whether goal_gate has approved active navigation
        self._nav_approved = False

        # QoS for real-time control — use RELIABLE to match Nav2 publishers
        cmd_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        odom_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        pub_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)

        # Subscribers
        self.cmd_sub = self.create_subscription(
            Twist, input_topic, self.cmd_vel_callback, cmd_qos
        )
        self.odom_sub = self.create_subscription(
            Odometry, odom_topic, self.odom_callback, odom_qos
        )

        # Navigation state subscriber (geofence unapproved motion detection)
        self._nav_state_sub = self.create_subscription(
            String, '/geofence/nav_state', self._nav_state_callback, 10
        )

        # DEBUG: Log actual resolved topic names
        self.get_logger().info(f'DEBUG subscriptions:')
        self.get_logger().info(f'  cmd_sub topic: {self.cmd_sub.topic_name}')
        self.get_logger().info(f'  odom_sub topic: {self.odom_sub.topic_name}')
        self.get_logger().info(f'  nav_state_sub topic: {self._nav_state_sub.topic_name}')

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, output_topic, pub_qos)
        self.metrics_pub = self.create_publisher(String, '/geofence/cmd_vel_events', 10)
        self.status_pub = self.create_publisher(String, '/geofence/cmd_vel_status', 10)

        # Statistics
        self.stats = {
            'total_commands': 0,
            'passed': 0,
            'blocked': 0,
            'scaled': 0,
            'cbf_interventions': 0,
            'ssm_slowdowns': 0,
            'min_distance_ever': float('inf'),
        }

        # Status timer
        self.create_timer(1.0, self.publish_status)

        # DEBUG: Heartbeat timer to verify spin loop is running
        self._heartbeat_count = 0
        self.create_timer(5.0, self._heartbeat)

        self.get_logger().info('=' * 60)
        self.get_logger().info('Cmd Vel Guard node initialized')
        self.get_logger().info(f'Safety method: {self.safety_method}')
        self.get_logger().info(f'Input: {input_topic} -> Output: {output_topic}')
        if self.comm_latency_sec > 0:
            self.get_logger().info(f'Simulated comm latency: {self.comm_latency_sec*1000:.0f}ms')
        self._log_method_info()
        self.get_logger().info('=' * 60)

    def _log_method_info(self):
        """Log method-specific information."""
        if self.safety_method == 'geofence':
            self.get_logger().info(f'  Mode: Forward Simulation')
            self.get_logger().info(f'  Horizon: {self.sim_horizon}s, Steps: {self.sim_steps}')
            self.get_logger().info(f'  Margin: {self.policy.get_safety_margin():.3f}m')
        elif self.safety_method == 'cbf':
            self.get_logger().info(f'  Mode: Real-time Control Barrier Function')
            self.get_logger().info(f'  Alpha (class-K): {self.cbf_alpha}')
            self.get_logger().info(f'  Margin: {self.cbf_checker.margin:.3f}m')
            self.get_logger().info(f'  Action: Velocity projection to safe set')
        elif self.safety_method == 'ssm':
            self.get_logger().info(f'  Mode: Real-time Speed and Separation Monitoring')
            self.get_logger().info(f'  Base margin: {self.ssm_checker.base_margin:.3f}m')
            self.get_logger().info(f'  Action: Gradual deceleration')
        else:
            self.get_logger().info(f'  Mode: Forward Simulation with {self.safety_method}')

    def _init_safety_baselines(self):
        """Initialize safety baseline checkers based on selected method."""
        zone_names = [z.name.replace(' ', '_').lower()
                      for z in self.policy.zones if z.zone_type == ZoneType.FORBIDDEN]

        geofence_margin = self.policy.get_safety_margin()

        # SELP: LTL automaton
        self.selp_checker = SELPSpatialSafety(zone_names)

        # CBF: Control Barrier Functions (Ames et al., TAC 2017)
        # Standard CBF uses small margin for real-time velocity control
        # The barrier h(x) = distance - margin should allow getting close to boundary
        cbf_margin = 0.1  # Small margin for real-time control
        self.cbf_checker = CBFSpatialSafety(gamma=self.cbf_alpha, margin=cbf_margin)

        # Static Margin (uses geofence margin for forward simulation)
        self.static_margin_checker = StaticMarginSafety(margin=geofence_margin)

        # SSM: Speed and Separation Monitoring (ISO 15066)
        # Standard SSM uses small base margin, safety comes from velocity-dependent term
        # S = v*(T_r + T_s) + v²/(2*a_max) + intrusion + base_margin
        ssm_base_margin = 0.05  # Minimal base margin
        self.ssm_checker = SSMSpatialSafety(
            reaction_time=0.1,      # System reaction time (s)
            stopping_time=0.2,      # Time to initiate stop (s)
            max_deceleration=1.0,   # Max deceleration (m/s²)
            intrusion_distance=0.05,  # Uncertainty margin (m)
            base_margin=ssm_base_margin
        )

        self.get_logger().info(f'Zones loaded: {len(self.policy.zones)}')
        self.get_logger().info(f'CBF margin: {cbf_margin}m (real-time control)')
        self.get_logger().info(f'SSM base margin: {ssm_base_margin}m (velocity-dependent)')

    def odom_callback(self, msg: Odometry):
        """Update robot pose and velocity from odometry."""
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y

        # Extract yaw from quaternion
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = math.atan2(siny_cosp, cosy_cosp)

        # Store current velocity
        self.robot_vx = msg.twist.twist.linear.x
        self.robot_vy = msg.twist.twist.linear.y

        self.last_odom_time = self.get_clock().now()

    def _nav_state_callback(self, msg: String):
        """Track goal_gate navigation authorization state."""
        try:
            data = json.loads(msg.data)
            new_state = (data.get('state') == 'active')
            if new_state != self._nav_approved:
                self._nav_approved = new_state
                self.get_logger().info(
                    f'Navigation {"authorized" if new_state else "completed"}'
                )
        except (json.JSONDecodeError, KeyError):
            pass

    def _compute_adaptive_margin(self, velocity: float) -> float:
        """Compute velocity-adaptive safety margin using geofence formula.

        Unlike the fixed margin (which uses configured v_max), this uses the
        actual robot velocity. When velocity exceeds v_max (e.g., param_10x
        attack), the margin grows accordingly.
        """
        u = self.policy.uncertainty
        return (u.k_sigma * u.localization_sigma
                + u.tracking_error
                + velocity * u.latency)

    def cmd_vel_callback(self, msg: Twist):
        """Process incoming velocity command using method-specific approach."""
        self.stats['total_commands'] += 1
        # Periodic debug: log every 50th command
        if self.stats['total_commands'] % 50 == 1:
            self.get_logger().info(
                f'Guard status: cmds={self.stats["total_commands"]}, '
                f'nav_approved={self._nav_approved}, odom={"yes" if self.last_odom_time else "no"}, '
                f'pos=({self.robot_x:.2f},{self.robot_y:.2f}), '
                f'passed={self.stats["passed"]}, blocked={self.stats["blocked"]}'
            )

        # Check if we have recent pose data
        if self.last_odom_time is None:
            self.get_logger().warning(
                'No odometry data received yet, passing cmd_vel through',
                throttle_duration_sec=5.0
            )
            self.cmd_pub.publish(msg)
            return

        # Check pose data freshness
        age = (self.get_clock().now() - self.last_odom_time).nanoseconds / 1e9
        if age > 0.5:
            self.get_logger().warning(
                f'Odometry data is stale ({age:.2f}s old)',
                throttle_duration_sec=5.0
            )

        # Geofence runtime: unapproved motion detection
        # If goal_gate hasn't authorized navigation, block cmd_vel toward zones.
        # This catches direct_control attacks that bypass Nav2 entirely.
        if self.safety_method == 'geofence' and not self._nav_approved:
            cmd_speed = math.sqrt(msg.linear.x**2 + msg.linear.y**2)
            if cmd_speed > 0.05:  # Non-trivial movement command
                self.stats['blocked'] += 1
                self.get_logger().warning(
                    f'BLOCKED (unapproved): cmd_vel without goal_gate authorization '
                    f'(speed={cmd_speed:.2f}m/s, pos=({self.robot_x:.2f}, {self.robot_y:.2f}))',
                    throttle_duration_sec=2.0
                )
                event_msg = String()
                event_msg.data = json.dumps({
                    'event': 'blocked_unapproved',
                    'speed': cmd_speed,
                    'position': [self.robot_x, self.robot_y],
                })
                self.metrics_pub.publish(event_msg)
                self.cmd_pub.publish(Twist())
                return

        # Route to method-specific handler
        if self.safety_method == 'cbf':
            safe_cmd, min_dist, modified = self._process_cbf(msg)
        elif self.safety_method == 'ssm':
            safe_cmd, min_dist, modified = self._process_ssm(msg)
        else:
            # geofence, selp, static_margin use forward simulation
            safe_cmd, min_dist, violation_time = self._process_forward_simulation(msg)
            modified = violation_time is not None

        # Update stats
        if min_dist < self.stats['min_distance_ever']:
            self.stats['min_distance_ever'] = min_dist

        # Simulated communication latency (for S4 latency variation experiment)
        if self.comm_latency_sec > 0:
            time.sleep(self.comm_latency_sec)

        # Publish the safe command
        self.cmd_pub.publish(safe_cmd)

    # =========================================================================
    # CBF: Real-time Control Barrier Function
    # =========================================================================

    def _process_cbf(self, cmd: Twist) -> Tuple[Twist, float, bool]:
        """
        Process command using Control Barrier Function (Ames et al., TAC 2017).

        CBF ensures forward invariance of safe set by maintaining h(x) >= 0.

        Barrier function: h(x) = d(x, Zone) - δ
        Safety condition: ḣ(x,u) + α*h(x) >= 0

        If the commanded velocity would violate the barrier condition,
        we project it onto the safe control set.

        Returns:
            (safe_cmd, min_distance, was_modified)
        """
        forbidden_zones = [z for z in self.policy.zones if z.zone_type == ZoneType.FORBIDDEN]

        # Current position
        x, y = self.robot_x, self.robot_y
        point = (x, y)

        # Compute barrier function value h(x)
        h, closest_zone = self.cbf_checker.compute_barrier(point, forbidden_zones)
        min_dist = h + self.cbf_checker.margin  # Actual distance to zone

        # Commanded velocity in world frame
        vx_cmd = cmd.linear.x * math.cos(self.robot_yaw) - cmd.linear.y * math.sin(self.robot_yaw)
        vy_cmd = cmd.linear.x * math.sin(self.robot_yaw) + cmd.linear.y * math.cos(self.robot_yaw)

        # Compute gradient of barrier function (points away from zone)
        grad_h = self._compute_barrier_gradient(point, forbidden_zones, closest_zone)

        # Compute ḣ = ∇h · v
        h_dot = grad_h[0] * vx_cmd + grad_h[1] * vy_cmd

        # CBF condition: ḣ + α*h >= 0
        cbf_condition = h_dot + self.cbf_alpha * h

        if cbf_condition >= 0:
            # Safe - pass through
            self.stats['passed'] += 1
            return cmd, min_dist, False

        # Unsafe - need to modify velocity
        # Project velocity onto safe set using QP solution:
        # min ||u - u_des||² s.t. ∇h·u + α*h >= 0
        #
        # Closed-form solution:
        # u_safe = u_des - (∇h·u_des + α*h) / ||∇h||² * ∇h  (if ∇h·u_des + α*h < 0)

        grad_norm_sq = grad_h[0]**2 + grad_h[1]**2

        if grad_norm_sq < 1e-6:
            # Gradient too small, just stop
            self.stats['blocked'] += 1
            self.stats['cbf_interventions'] += 1
            self.get_logger().warn(f'CBF: Gradient too small, stopping. h={h:.3f}')
            return Twist(), min_dist, True

        # Compute correction factor
        correction = -cbf_condition / grad_norm_sq

        # Project velocity
        vx_safe = vx_cmd + correction * grad_h[0]
        vy_safe = vy_cmd + correction * grad_h[1]

        # Convert back to robot frame
        cos_yaw = math.cos(self.robot_yaw)
        sin_yaw = math.sin(self.robot_yaw)

        safe_cmd = Twist()
        safe_cmd.linear.x = vx_safe * cos_yaw + vy_safe * sin_yaw
        safe_cmd.linear.y = -vx_safe * sin_yaw + vy_safe * cos_yaw
        safe_cmd.angular.z = cmd.angular.z  # Keep angular velocity

        self.stats['scaled'] += 1
        self.stats['cbf_interventions'] += 1

        # Log the intervention
        orig_speed = math.sqrt(cmd.linear.x**2 + cmd.linear.y**2)
        safe_speed = math.sqrt(safe_cmd.linear.x**2 + safe_cmd.linear.y**2)

        self.get_logger().warn(
            f'CBF intervention: h={h:.3f}, ḣ+αh={cbf_condition:.3f}, '
            f'speed {orig_speed:.2f}->{safe_speed:.2f} m/s'
        )

        return safe_cmd, min_dist, True

    def _compute_barrier_gradient(self, point: Tuple[float, float],
                                   zones: List, closest_zone: str) -> Tuple[float, float]:
        """
        Compute gradient of barrier function (unit vector pointing away from zone).
        """
        for zone in zones:
            if zone.name == closest_zone:
                # Approximate gradient using zone centroid
                cx = sum(v[0] for v in zone.vertices) / len(zone.vertices)
                cy = sum(v[1] for v in zone.vertices) / len(zone.vertices)

                dx = point[0] - cx
                dy = point[1] - cy
                dist = math.sqrt(dx*dx + dy*dy)

                if dist < 1e-6:
                    return (1.0, 0.0)

                return (dx / dist, dy / dist)

        return (1.0, 0.0)

    # =========================================================================
    # SSM: Real-time Speed and Separation Monitoring
    # =========================================================================

    def _process_ssm(self, cmd: Twist) -> Tuple[Twist, float, bool]:
        """
        Process command using Speed and Separation Monitoring (ISO 15066).

        SSM ensures protective separation distance S is maintained:
        S = v*(T_r + T_s) + v²/(2*a_max) + C + Z_d

        If current distance < S, we reduce velocity to satisfy the constraint.

        Returns:
            (safe_cmd, min_distance, was_modified)
        """
        forbidden_zones = [z for z in self.policy.zones if z.zone_type == ZoneType.FORBIDDEN]

        # Current position and commanded speed
        x, y = self.robot_x, self.robot_y
        cmd_speed = math.sqrt(cmd.linear.x**2 + cmd.linear.y**2)

        # Get minimum distance to any forbidden zone
        min_dist = self.policy.get_min_distance_to_forbidden(x, y)

        # Compute protective distance at commanded speed
        S_cmd = self.ssm_checker.compute_protective_distance(cmd_speed)

        if min_dist >= S_cmd:
            # Safe - pass through
            self.stats['passed'] += 1
            return cmd, min_dist, False

        # Need to reduce speed
        # Find maximum safe velocity: solve for v in S(v) = min_dist
        # S = v*(T_r + T_s) + v²/(2*a_max) + C + base
        # This is a quadratic in v

        T = self.ssm_checker.reaction_time + self.ssm_checker.stopping_time
        a_max = self.ssm_checker.max_deceleration
        C = self.ssm_checker.intrusion_distance + self.ssm_checker.base_margin

        # Rearrange: v²/(2*a_max) + T*v + (C - min_dist) = 0
        # v²/(2*a_max) + T*v + C - d = 0
        # Multiply by 2*a_max: v² + 2*a_max*T*v + 2*a_max*(C - d) = 0

        a_coef = 1.0
        b_coef = 2 * a_max * T
        c_coef = 2 * a_max * (C - min_dist)

        discriminant = b_coef**2 - 4 * a_coef * c_coef

        if discriminant < 0:
            # No solution - must stop
            self.stats['blocked'] += 1
            self.stats['ssm_slowdowns'] += 1
            self.get_logger().warn(
                f'SSM: Distance {min_dist:.3f}m < base margin, stopping'
            )
            return Twist(), min_dist, True

        # Take positive root (maximum safe velocity)
        v_safe = (-b_coef + math.sqrt(discriminant)) / (2 * a_coef)
        v_safe = max(0.0, v_safe)

        if v_safe < self.ssm_min_velocity:
            # Below minimum, just stop
            self.stats['blocked'] += 1
            self.stats['ssm_slowdowns'] += 1
            self.get_logger().warn(
                f'SSM: Safe velocity {v_safe:.3f} < min {self.ssm_min_velocity}, stopping'
            )
            return Twist(), min_dist, True

        # Scale velocity to safe level
        if cmd_speed > 1e-6:
            scale = v_safe / cmd_speed
        else:
            scale = 1.0

        safe_cmd = Twist()
        safe_cmd.linear.x = cmd.linear.x * scale
        safe_cmd.linear.y = cmd.linear.y * scale
        safe_cmd.angular.z = cmd.angular.z * scale  # Also scale rotation for consistency

        self.stats['scaled'] += 1
        self.stats['ssm_slowdowns'] += 1

        self.get_logger().warn(
            f'SSM slowdown: dist={min_dist:.3f}m, S={S_cmd:.3f}m, '
            f'speed {cmd_speed:.2f}->{v_safe:.2f} m/s (scale={scale:.2f})'
        )

        return safe_cmd, min_dist, True

    # =========================================================================
    # Forward Simulation (for geofence, selp, static_margin)
    # =========================================================================

    def _process_forward_simulation(self, cmd: Twist) -> Tuple[Twist, float, Optional[float]]:
        """
        Process command using forward simulation (geofence approach).

        Simulates trajectory for sim_horizon seconds and checks each point.

        Returns:
            (safe_cmd, min_distance, violation_time_or_none)
        """
        dt = self.sim_horizon / self.sim_steps

        x, y, yaw = self.robot_x, self.robot_y, self.robot_yaw
        vx, vy, wz = cmd.linear.x, cmd.linear.y, cmd.angular.z
        current_velocity = math.sqrt(vx*vx + vy*vy)

        min_dist = float('inf')
        violation_time = None
        violation_step = None

        forbidden_zones = [z for z in self.policy.zones if z.zone_type == ZoneType.FORBIDDEN]

        # Simulate forward
        for step in range(self.sim_steps):
            t = step * dt

            # Update pose (differential drive model)
            x += (vx * math.cos(yaw) - vy * math.sin(yaw)) * dt
            y += (vx * math.sin(yaw) + vy * math.cos(yaw)) * dt
            yaw += wz * dt

            point = (x, y)
            dist = self.policy.get_min_distance_to_forbidden(x, y)
            if dist < min_dist:
                min_dist = dist

            # Check for violation using selected safety method
            is_violation = False

            if self.safety_method == 'geofence':
                # Velocity-adaptive margin: use actual velocity for margin computation.
                # When robot moves faster than configured v_max (e.g., param_10x attack),
                # the margin grows → earlier blocking.
                v_actual = math.sqrt(self.robot_vx**2 + self.robot_vy**2)
                v_for_margin = max(v_actual, current_velocity)
                adaptive_margin = self._compute_adaptive_margin(v_for_margin)
                if dist < adaptive_margin:
                    is_violation = True

            elif self.safety_method == 'selp':
                result = self.selp_checker.evaluate(point, forbidden_zones)
                if result.decision == SafetyDecision.REJECT:
                    is_violation = True

            elif self.safety_method == 'static_margin':
                result = self.static_margin_checker.evaluate(point, forbidden_zones)
                if result.decision == SafetyDecision.REJECT:
                    is_violation = True

            else:
                # Fallback to geofence
                decision = self.policy.evaluate_point(x, y)
                if decision.action in (PolicyAction.REJECT, PolicyAction.PROJECT):
                    is_violation = True

            if is_violation and violation_time is None:
                violation_time = t
                violation_step = step
                break

        # If no violation, pass through
        if violation_time is None:
            self.stats['passed'] += 1
            return cmd, min_dist, None

        # Apply override strategy
        if self.override_strategy == OverrideStrategy.STOP:
            self.stats['blocked'] += 1
            safe_cmd = Twist()
            self.get_logger().warn(
                f'BLOCKED ({self.safety_method}): violation at t={violation_time:.2f}s, '
                f'min_dist={min_dist:.3f}m'
            )

        elif self.override_strategy == OverrideStrategy.SCALE:
            self.stats['scaled'] += 1
            scale = max(0.0, (violation_step / self.sim_steps) * self.scale_decay)
            safe_cmd = Twist()
            safe_cmd.linear.x = cmd.linear.x * scale
            safe_cmd.linear.y = cmd.linear.y * scale
            safe_cmd.angular.z = cmd.angular.z * scale
            self.get_logger().warn(
                f'SCALED ({self.safety_method}) by {scale:.2f}: violation at t={violation_time:.2f}s'
            )

        elif self.override_strategy == OverrideStrategy.REDIRECT:
            self.stats['scaled'] += 1
            safe_cmd = self._compute_redirect(cmd, x, y)
            self.get_logger().warn(
                f'REDIRECTED ({self.safety_method}): violation at t={violation_time:.2f}s'
            )

        else:
            safe_cmd = Twist()

        return safe_cmd, min_dist, violation_time

    def _compute_redirect(self, cmd: Twist, danger_x: float, danger_y: float) -> Twist:
        """Compute redirected velocity away from danger point."""
        dx = self.robot_x - danger_x
        dy = self.robot_y - danger_y
        dist = math.sqrt(dx*dx + dy*dy)

        if dist < 0.01:
            return Twist()

        dx /= dist
        dy /= dist

        orig_speed = math.sqrt(cmd.linear.x**2 + cmd.linear.y**2)

        safe_cmd = Twist()
        safe_cmd.linear.x = dx * orig_speed * 0.5
        safe_cmd.linear.y = dy * orig_speed * 0.5
        safe_cmd.angular.z = cmd.angular.z * 0.3

        return safe_cmd

    # =========================================================================
    # Status and Event Publishing
    # =========================================================================

    def _heartbeat(self):
        """DEBUG: Periodic heartbeat to verify spin loop is working."""
        self._heartbeat_count += 1
        self.get_logger().info(
            f'[HEARTBEAT #{self._heartbeat_count}] '
            f'cmds={self.stats["total_commands"]}, '
            f'odom={"yes" if self.last_odom_time else "no"}, '
            f'nav_approved={self._nav_approved}, '
            f'pos=({self.robot_x:.2f},{self.robot_y:.2f})'
        )

    def publish_status(self):
        """Publish periodic status."""
        min_dist_ever = self.stats['min_distance_ever']
        if min_dist_ever == float('inf'):
            min_dist_ever = -1.0

        current_min = self.policy.get_min_distance_to_forbidden(
            self.robot_x, self.robot_y
        ) if self.policy.zones else -1.0
        if current_min == float('inf'):
            current_min = -1.0

        safe_stats = dict(self.stats)
        safe_stats['min_distance_ever'] = min_dist_ever

        msg = String()
        msg.data = json.dumps({
            'status': 'active',
            'safety_method': self.safety_method,
            'robot_pose': {
                'x': self.robot_x,
                'y': self.robot_y,
                'yaw': self.robot_yaw
            },
            'stats': safe_stats,
            'current_min_dist': current_min,
        })
        self.status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = CmdVelGuardNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Cmd Vel Guard node')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
