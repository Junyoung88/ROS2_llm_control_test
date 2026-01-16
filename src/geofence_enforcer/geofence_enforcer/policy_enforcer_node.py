"""
Probabilistic Geofence Policy Enforcer Node

This ROS2 node enforces dynamic safety margins around forbidden zones,
accounting for:
1. State estimation uncertainty (from AMCL/SLAM covariance)
2. Execution/tracking error buffer
3. System latency compensation

The node dynamically computes the safety margin using:

    margin_α = √(χ²₂(α) · λ_max(Σ_xy)) + e_track + v_max · τ

And enforces this margin around defined forbidden polygons.

Subscriptions:
    /amcl_pose (PoseWithCovarianceStamped): Robot pose with covariance
    /odom (Odometry): Optional, for current velocity estimation
    /cmd_vel (Twist): Optional fallback for velocity estimation

Services:
    ~/validate_goal (ValidateGoal): Validate/project navigation goals
    ~/check_pose (CheckPose): Check if a pose violates geofence

Publications:
    ~/status (GeofenceStatus): Current geofence status and margin
    ~/inflated_zones (MarkerArray): Visualization of inflated zones

Parameters:
    alpha: Confidence level for uncertainty bound [0.95]
    e_track: Tracking error buffer in meters [0.05]
    v_max: Maximum velocity in m/s [0.5]
    tau: System latency in seconds [0.1]
    enable_geofence: Enable/disable enforcement [True]
    enable_projection: Project unsafe goals to safe points [True]
    geofence_config: Path to YAML config file [required]
    check_rate_hz: Rate for pose checking [10.0]
    safety_stop_topic: Topic to publish stop commands [/cmd_vel]

Usage:
    ros2 run geofence_enforcer policy_enforcer_node \\
        --ros-args \\
        -p geofence_config:=/path/to/geofence.yaml \\
        -p alpha:=0.95 \\
        -p e_track:=0.05 \\
        -p v_max:=0.5 \\
        -p tau:=0.1
"""

import numpy as np
from pathlib import Path
from typing import Optional
from threading import Lock

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

from geometry_msgs.msg import PoseWithCovarianceStamped, Twist, PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Float64
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Duration

from .margin_calculator import MarginCalculator, MarginComponents
from .geofence_geometry import GeofenceGeometry, GeofenceCheckResult


class PolicyEnforcerNode(Node):
    """
    ROS2 node for probabilistic geofence enforcement.

    This node continuously monitors the robot's pose and enforces
    safety margins around forbidden zones, dynamically adapting
    the margin based on localization uncertainty.
    """

    def __init__(self):
        super().__init__('policy_enforcer_node')

        # =====================================================================
        # PARAMETER DECLARATIONS
        # =====================================================================

        # Margin calculation parameters
        self.declare_parameter('alpha', 0.95)
        self.declare_parameter('e_track', 0.05)
        self.declare_parameter('v_max', 0.5)
        self.declare_parameter('tau', 0.1)

        # Geofence behavior parameters
        self.declare_parameter('enable_geofence', True)
        self.declare_parameter('enable_projection', True)
        self.declare_parameter('geofence_config', '')

        # Node behavior parameters
        self.declare_parameter('check_rate_hz', 10.0)
        self.declare_parameter('safety_stop_topic', '/cmd_vel')
        self.declare_parameter('publish_visualization', True)
        self.declare_parameter('visualization_rate_hz', 1.0)

        # Projection parameters
        self.declare_parameter('projection_safety_buffer', 0.01)

        # Load parameters
        self._alpha = self.get_parameter('alpha').value
        self._e_track = self.get_parameter('e_track').value
        self._v_max = self.get_parameter('v_max').value
        self._tau = self.get_parameter('tau').value
        self._enable_geofence = self.get_parameter('enable_geofence').value
        self._enable_projection = self.get_parameter('enable_projection').value
        self._config_path = self.get_parameter('geofence_config').value
        self._check_rate = self.get_parameter('check_rate_hz').value
        self._safety_stop_topic = self.get_parameter('safety_stop_topic').value
        self._publish_viz = self.get_parameter('publish_visualization').value
        self._viz_rate = self.get_parameter('visualization_rate_hz').value
        self._projection_buffer = self.get_parameter('projection_safety_buffer').value

        # =====================================================================
        # STATE VARIABLES
        # =====================================================================

        self._lock = Lock()

        # Latest pose and covariance
        self._current_pose: Optional[PoseWithCovarianceStamped] = None
        self._current_covariance: Optional[np.ndarray] = None

        # Current velocity estimate
        self._current_velocity: float = 0.0

        # Current margin (updated each cycle)
        self._current_margin: float = 0.0
        self._margin_components: Optional[MarginComponents] = None

        # Violation state
        self._is_violated: bool = False
        self._violation_count: int = 0

        # =====================================================================
        # INITIALIZE COMPONENTS
        # =====================================================================

        # Initialize margin calculator
        self._margin_calc = MarginCalculator(
            alpha=self._alpha,
            e_track=self._e_track,
            v_max=self._v_max,
            tau=self._tau
        )

        # Initialize geofence geometry
        self._geofence: Optional[GeofenceGeometry] = None
        if self._config_path:
            self._load_geofence_config(self._config_path)
        else:
            self.get_logger().warn(
                "No geofence_config parameter provided. "
                "Geofence enforcement disabled until config is loaded."
            )
            self._enable_geofence = False

        # =====================================================================
        # QOS PROFILES
        # =====================================================================

        # Reliable QoS for important safety messages
        reliable_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=10
        )

        # Best effort for high-frequency data
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            depth=10
        )

        # =====================================================================
        # CALLBACK GROUPS
        # =====================================================================

        self._pose_cb_group = MutuallyExclusiveCallbackGroup()
        self._timer_cb_group = ReentrantCallbackGroup()

        # =====================================================================
        # SUBSCRIBERS
        # =====================================================================

        # Primary: AMCL pose with covariance
        self._pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self._pose_callback,
            sensor_qos,
            callback_group=self._pose_cb_group
        )

        # Optional: Odometry for velocity
        self._odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self._odom_callback,
            sensor_qos,
            callback_group=self._pose_cb_group
        )

        # Optional: cmd_vel for velocity fallback
        self._cmd_vel_sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self._cmd_vel_callback,
            sensor_qos,
            callback_group=self._pose_cb_group
        )

        # Goal subscription for validation
        self._goal_sub = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self._goal_callback,
            reliable_qos,
            callback_group=self._pose_cb_group
        )

        # =====================================================================
        # PUBLISHERS
        # =====================================================================

        # Validated/projected goal output
        self._validated_goal_pub = self.create_publisher(
            PoseStamped,
            '~/validated_goal',
            reliable_qos
        )

        # Safety stop command
        self._safety_stop_pub = self.create_publisher(
            Twist,
            self._safety_stop_topic,
            reliable_qos
        )

        # Current margin value
        self._margin_pub = self.create_publisher(
            Float64,
            '~/current_margin',
            reliable_qos
        )

        # Violation status
        self._violation_pub = self.create_publisher(
            Bool,
            '~/is_violated',
            reliable_qos
        )

        # Visualization markers
        self._marker_pub = self.create_publisher(
            MarkerArray,
            '~/inflated_zones',
            reliable_qos
        )

        # =====================================================================
        # TIMERS
        # =====================================================================

        # Main check timer
        check_period = 1.0 / self._check_rate
        self._check_timer = self.create_timer(
            check_period,
            self._check_callback,
            callback_group=self._timer_cb_group
        )

        # Visualization timer
        if self._publish_viz:
            viz_period = 1.0 / self._viz_rate
            self._viz_timer = self.create_timer(
                viz_period,
                self._publish_visualization,
                callback_group=self._timer_cb_group
            )

        # =====================================================================
        # PARAMETER CALLBACK
        # =====================================================================

        self.add_on_set_parameters_callback(self._parameter_callback)

        # =====================================================================
        # STARTUP LOG
        # =====================================================================

        self._log_configuration()

    def _load_geofence_config(self, config_path: str) -> bool:
        """
        Load geofence configuration from YAML file.

        Args:
            config_path: Path to YAML configuration

        Returns:
            True if loaded successfully
        """
        try:
            path = Path(config_path)
            if not path.exists():
                self.get_logger().error(f"Geofence config not found: {path}")
                return False

            self._geofence = GeofenceGeometry.from_yaml(path)

            zone_names = self._geofence.get_zone_names()
            self.get_logger().info(
                f"Loaded {len(zone_names)} forbidden zones: {zone_names}"
            )

            return True

        except Exception as e:
            self.get_logger().error(f"Failed to load geofence config: {e}")
            return False

    def _log_configuration(self):
        """Log current configuration on startup."""
        self.get_logger().info("=" * 60)
        self.get_logger().info("Probabilistic Geofence Policy Enforcer")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"  α (confidence level):    {self._alpha}")
        self.get_logger().info(f"  e_track (tracking err):  {self._e_track} m")
        self.get_logger().info(f"  v_max (max velocity):    {self._v_max} m/s")
        self.get_logger().info(f"  τ (system latency):      {self._tau} s")
        self.get_logger().info(f"  χ²₂⁻¹(α):                {self._margin_calc.chi2_quantile:.4f}")
        self.get_logger().info(f"  Geofence enabled:        {self._enable_geofence}")
        self.get_logger().info(f"  Projection enabled:      {self._enable_projection}")
        self.get_logger().info(f"  Check rate:              {self._check_rate} Hz")

        if self._geofence:
            self.get_logger().info(f"  Zones loaded:            {len(self._geofence.get_zone_names())}")
        self.get_logger().info("=" * 60)

    def _parameter_callback(self, params):
        """Handle dynamic parameter updates."""
        from rcl_interfaces.msg import SetParametersResult

        for param in params:
            if param.name == 'alpha':
                if 0.0 < param.value < 1.0:
                    self._alpha = param.value
                    self._margin_calc.update_parameters(alpha=param.value)
                    self.get_logger().info(f"Updated alpha to {param.value}")
                else:
                    return SetParametersResult(
                        successful=False,
                        reason="alpha must be in (0, 1)"
                    )

            elif param.name == 'e_track':
                if param.value >= 0:
                    self._e_track = param.value
                    self._margin_calc.update_parameters(e_track=param.value)
                    self.get_logger().info(f"Updated e_track to {param.value}")
                else:
                    return SetParametersResult(
                        successful=False,
                        reason="e_track must be non-negative"
                    )

            elif param.name == 'v_max':
                if param.value >= 0:
                    self._v_max = param.value
                    self._margin_calc.update_parameters(v_max=param.value)
                    self.get_logger().info(f"Updated v_max to {param.value}")
                else:
                    return SetParametersResult(
                        successful=False,
                        reason="v_max must be non-negative"
                    )

            elif param.name == 'tau':
                if param.value >= 0:
                    self._tau = param.value
                    self._margin_calc.update_parameters(tau=param.value)
                    self.get_logger().info(f"Updated tau to {param.value}")
                else:
                    return SetParametersResult(
                        successful=False,
                        reason="tau must be non-negative"
                    )

            elif param.name == 'enable_geofence':
                self._enable_geofence = param.value
                self.get_logger().info(f"Geofence {'enabled' if param.value else 'disabled'}")

            elif param.name == 'enable_projection':
                self._enable_projection = param.value
                self.get_logger().info(f"Projection {'enabled' if param.value else 'disabled'}")

        return SetParametersResult(successful=True)

    # =========================================================================
    # SUBSCRIBER CALLBACKS
    # =========================================================================

    def _pose_callback(self, msg: PoseWithCovarianceStamped):
        """
        Handle incoming AMCL pose with covariance.

        The covariance is stored for margin computation.
        """
        with self._lock:
            self._current_pose = msg
            # Convert flat 36-element array to numpy
            self._current_covariance = np.array(msg.pose.covariance)

    def _odom_callback(self, msg: Odometry):
        """
        Handle odometry for velocity estimation.

        Extracts translational velocity magnitude from twist.
        """
        with self._lock:
            vx = msg.twist.twist.linear.x
            vy = msg.twist.twist.linear.y
            self._current_velocity = np.sqrt(vx * vx + vy * vy)

    def _cmd_vel_callback(self, msg: Twist):
        """
        Handle cmd_vel as fallback velocity estimate.

        Used when odometry is not available.
        """
        with self._lock:
            # Only update if we don't have odom data
            vx = msg.linear.x
            vy = msg.linear.y
            velocity = np.sqrt(vx * vx + vy * vy)
            # Use cmd_vel as fallback (odom is preferred)
            if self._current_velocity < 0.001:
                self._current_velocity = velocity

    def _goal_callback(self, msg: PoseStamped):
        """
        Validate incoming navigation goal against geofence.

        If the goal violates the geofence:
        - If projection enabled: project to nearest safe point
        - If projection disabled: block the goal

        Always publishes result to ~/validated_goal.
        """
        if not self._enable_geofence or self._geofence is None:
            # Pass through if geofence disabled
            self._validated_goal_pub.publish(msg)
            return

        goal_x = msg.pose.position.x
        goal_y = msg.pose.position.y

        # Get current margin
        margin = self._current_margin

        # Validate goal
        is_valid, safe_x, safe_y, message = self._geofence.validate_goal(
            goal_x,
            goal_y,
            margin=margin,
            enable_projection=self._enable_projection
        )

        self.get_logger().info(f"Goal validation: {message}")

        if is_valid:
            # Create validated goal message
            validated = PoseStamped()
            validated.header = msg.header
            validated.pose.position.x = safe_x
            validated.pose.position.y = safe_y
            validated.pose.position.z = msg.pose.position.z
            validated.pose.orientation = msg.pose.orientation

            self._validated_goal_pub.publish(validated)

            if safe_x != goal_x or safe_y != goal_y:
                self.get_logger().warn(
                    f"Goal PROJECTED: ({goal_x:.3f}, {goal_y:.3f}) -> "
                    f"({safe_x:.3f}, {safe_y:.3f})"
                )
        else:
            self.get_logger().error(
                f"Goal BLOCKED: ({goal_x:.3f}, {goal_y:.3f}) "
                f"violates geofence with margin {margin:.3f}m"
            )

    # =========================================================================
    # MAIN CHECK CALLBACK
    # =========================================================================

    def _check_callback(self):
        """
        Main periodic check: compute margin and verify current pose.

        This is the core enforcement loop that:
        1. Computes the dynamic safety margin from current covariance
        2. Checks if current position violates any forbidden zone
        3. Triggers safety stop if violated
        """
        if not self._enable_geofence:
            return

        with self._lock:
            pose = self._current_pose
            covariance = self._current_covariance
            velocity = self._current_velocity

        # Skip if no pose received yet
        if pose is None or covariance is None:
            return

        # ===== STEP 1: Compute dynamic safety margin =====
        self._margin_components = self._margin_calc.compute_margin(
            covariance,
            current_velocity=velocity
        )
        self._current_margin = self._margin_components.total_margin

        # Publish margin value
        margin_msg = Float64()
        margin_msg.data = self._current_margin
        self._margin_pub.publish(margin_msg)

        # ===== STEP 2: Check current pose against geofence =====
        if self._geofence is None:
            return

        robot_x = pose.pose.pose.position.x
        robot_y = pose.pose.pose.position.y

        check_result = self._geofence.check_point(
            robot_x,
            robot_y,
            margin=self._current_margin
        )

        # ===== STEP 3: Handle violation =====
        was_violated = self._is_violated
        self._is_violated = check_result.is_violated

        # Publish violation status
        violation_msg = Bool()
        violation_msg.data = self._is_violated
        self._violation_pub.publish(violation_msg)

        if self._is_violated:
            self._violation_count += 1

            # Log violation details
            self._log_violation(robot_x, robot_y, check_result)

            # Trigger safety stop
            self._trigger_safety_stop()

        elif was_violated and not self._is_violated:
            # Recovered from violation
            self.get_logger().info(
                f"RECOVERED from geofence violation. "
                f"Current position ({robot_x:.3f}, {robot_y:.3f}) is now safe."
            )

    def _log_violation(
        self,
        robot_x: float,
        robot_y: float,
        check_result: GeofenceCheckResult
    ):
        """
        Log detailed violation information.

        Provides comprehensive diagnostic output for debugging
        and analysis of safety margin behavior.
        """
        mc = self._margin_components

        self.get_logger().error("=" * 60)
        self.get_logger().error("GEOFENCE VIOLATION DETECTED")
        self.get_logger().error("=" * 60)
        self.get_logger().error(f"  Position:        ({robot_x:.4f}, {robot_y:.4f})")
        self.get_logger().error(f"  Violated zones:  {check_result.violated_zones}")
        self.get_logger().error(f"  Distance inside: {-check_result.distance_to_boundary:.4f} m")

        if check_result.nearest_safe_point:
            sx, sy = check_result.nearest_safe_point
            self.get_logger().error(f"  Nearest safe:    ({sx:.4f}, {sy:.4f})")

        self.get_logger().error("-" * 60)
        self.get_logger().error("MARGIN BREAKDOWN:")
        self.get_logger().error(f"  Total margin:      {mc.total_margin:.4f} m")
        self.get_logger().error(f"  ├─ Estimation:     {mc.estimation_term:.4f} m")
        self.get_logger().error(f"  │    χ²₂(α):       {mc.chi2_quantile:.4f}")
        self.get_logger().error(f"  │    λ_max:        {mc.lambda_max:.6f}")
        self.get_logger().error(f"  │    λ_min:        {mc.lambda_min:.6f}")
        self.get_logger().error(f"  ├─ Tracking err:   {mc.tracking_error:.4f} m")
        self.get_logger().error(f"  └─ Latency term:   {mc.latency_term:.4f} m")
        self.get_logger().error("=" * 60)
        self.get_logger().error(f"  Violation count:   {self._violation_count}")
        self.get_logger().error("=" * 60)

    def _trigger_safety_stop(self):
        """
        Publish zero velocity command to stop the robot.

        This is the primary safety response when a violation is detected.
        """
        stop_cmd = Twist()
        # All fields default to 0.0 (zero velocity)
        self._safety_stop_pub.publish(stop_cmd)

        self.get_logger().warn("SAFETY STOP triggered - publishing zero velocity")

    # =========================================================================
    # VISUALIZATION
    # =========================================================================

    def _publish_visualization(self):
        """
        Publish visualization markers for inflated forbidden zones.

        Publishes both original polygons and inflated versions
        for debugging and monitoring in RViz.
        """
        if self._geofence is None:
            return

        marker_array = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        # Get inflated zones
        margin = self._current_margin if self._current_margin > 0 else 0.1
        inflated_zones = self._geofence.get_inflated_zones(margin)

        marker_id = 0

        for name, zone in self._geofence._zones.items():
            # Original zone (red, semi-transparent)
            orig_marker = self._create_polygon_marker(
                zone.polygon,
                marker_id,
                stamp,
                name + "_original",
                r=0.8, g=0.0, b=0.0, a=0.3
            )
            marker_array.markers.append(orig_marker)
            marker_id += 1

            # Inflated zone (orange, more transparent)
            if name in inflated_zones:
                inflated_marker = self._create_polygon_marker(
                    inflated_zones[name],
                    marker_id,
                    stamp,
                    name + "_inflated",
                    r=1.0, g=0.5, b=0.0, a=0.2
                )
                marker_array.markers.append(inflated_marker)
                marker_id += 1

        # Current robot position marker (if available)
        if self._current_pose is not None:
            robot_marker = self._create_robot_marker(
                self._current_pose,
                marker_id,
                stamp
            )
            marker_array.markers.append(robot_marker)
            marker_id += 1

            # Margin circle around robot
            margin_marker = self._create_margin_circle_marker(
                self._current_pose,
                self._current_margin,
                marker_id,
                stamp
            )
            marker_array.markers.append(margin_marker)

        self._marker_pub.publish(marker_array)

    def _create_polygon_marker(
        self,
        polygon,
        marker_id: int,
        stamp,
        ns: str,
        r: float, g: float, b: float, a: float
    ) -> Marker:
        """Create a LINE_STRIP marker for polygon visualization."""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = stamp
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        marker.scale.x = 0.02  # Line width

        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.color.a = a

        # Extract polygon boundary points
        if hasattr(polygon, 'exterior'):
            coords = list(polygon.exterior.coords)
        else:
            coords = list(polygon.coords)

        from geometry_msgs.msg import Point as GeoPoint
        for x, y in coords:
            p = GeoPoint()
            p.x = float(x)
            p.y = float(y)
            p.z = 0.0
            marker.points.append(p)

        # Close the polygon
        if coords:
            p = GeoPoint()
            p.x = float(coords[0][0])
            p.y = float(coords[0][1])
            p.z = 0.0
            marker.points.append(p)

        marker.lifetime = Duration(sec=2, nanosec=0)

        return marker

    def _create_robot_marker(
        self,
        pose: PoseWithCovarianceStamped,
        marker_id: int,
        stamp
    ) -> Marker:
        """Create a sphere marker for robot position."""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = stamp
        marker.ns = "robot"
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = pose.pose.pose.position.x
        marker.pose.position.y = pose.pose.pose.position.y
        marker.pose.position.z = 0.1

        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1

        # Green if safe, red if violated
        if self._is_violated:
            marker.color.r = 1.0
            marker.color.g = 0.0
        else:
            marker.color.r = 0.0
            marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        marker.lifetime = Duration(sec=2, nanosec=0)

        return marker

    def _create_margin_circle_marker(
        self,
        pose: PoseWithCovarianceStamped,
        margin: float,
        marker_id: int,
        stamp
    ) -> Marker:
        """Create a circle marker showing current safety margin."""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = stamp
        marker.ns = "margin"
        marker.id = marker_id
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD

        marker.pose.position.x = pose.pose.pose.position.x
        marker.pose.position.y = pose.pose.pose.position.y
        marker.pose.position.z = 0.01

        marker.scale.x = margin * 2  # Diameter
        marker.scale.y = margin * 2
        marker.scale.z = 0.01  # Thin disk

        marker.color.r = 0.0
        marker.color.g = 0.5
        marker.color.b = 1.0
        marker.color.a = 0.3

        marker.lifetime = Duration(sec=2, nanosec=0)

        return marker

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def get_current_margin(self) -> float:
        """Get the current computed safety margin."""
        return self._current_margin

    def get_margin_breakdown(self) -> Optional[MarginComponents]:
        """Get detailed breakdown of margin components."""
        return self._margin_components

    def is_violated(self) -> bool:
        """Check if geofence is currently violated."""
        return self._is_violated

    def validate_point(
        self,
        x: float,
        y: float,
        margin: Optional[float] = None
    ) -> GeofenceCheckResult:
        """
        Validate a point against the geofence.

        Args:
            x: X coordinate
            y: Y coordinate
            margin: Optional margin override (uses current margin if None)

        Returns:
            GeofenceCheckResult with violation details
        """
        if self._geofence is None:
            return GeofenceCheckResult(
                is_violated=False,
                violated_zones=[],
                distance_to_boundary=float('inf')
            )

        m = margin if margin is not None else self._current_margin
        return self._geofence.check_point(x, y, margin=m)


def main(args=None):
    """Main entry point for the policy enforcer node."""
    rclpy.init(args=args)

    node = PolicyEnforcerNode()

    # Use multi-threaded executor for parallel callback processing
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        node.get_logger().info("Policy enforcer node spinning...")
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down policy enforcer node...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
