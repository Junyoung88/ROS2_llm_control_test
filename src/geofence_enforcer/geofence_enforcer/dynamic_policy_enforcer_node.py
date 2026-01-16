"""
Dynamic Probabilistic Geofence Policy Enforcer Node

This is the fully integrated ROS2 node that computes safety margins
using ALL real-time data sources:

1. Position Uncertainty (Σ_xy): From AMCL/SLAM covariance
2. Tracking Error (e_track): Measured from path following performance
3. System Latency (τ): Measured from sensor/command delays
4. Current Velocity (v): From odometry

Safety Margin Formula:
    margin_α = √(χ²₂(α) · λ_max(Σ_xy)) + e_track + v · τ

All parameters are dynamically updated in real-time!

Subscriptions:
    /amcl_pose - Position with covariance
    /odom - Odometry for velocity and latency measurement
    /plan - Planned path for tracking error measurement
    /scan - Laser scan for latency measurement

Publications:
    ~/dynamic_margin - Current computed margin with breakdown
    ~/safety_status - Real-time safety assessment
    ~/parameter_stats - Statistics of measured parameters

Services:
    ~/validate_goal - Validate/project navigation goals
    ~/get_margin - Get current margin value
"""

import numpy as np
from typing import Optional, Tuple
from threading import Lock
import json

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String, Float64
from std_srvs.srv import Trigger

from .margin_calculator import MarginCalculator, MarginComponents
from .geofence_geometry import GeofenceGeometry
from .latency_monitor import LatencyMonitor
from .tracking_error_monitor import TrackingErrorMonitor


class DynamicMarginData:
    """Container for all dynamic margin components."""
    def __init__(self):
        # From AMCL covariance
        self.sigma_xx: float = 0.01
        self.sigma_yy: float = 0.01
        self.sigma_xy: float = 0.0
        self.lambda_max: float = 0.01

        # Measured parameters
        self.e_track: float = 0.05
        self.tau: float = 0.1
        self.current_velocity: float = 0.0

        # Computed margin
        self.estimation_term: float = 0.0
        self.tracking_term: float = 0.0
        self.latency_term: float = 0.0
        self.total_margin: float = 0.0

        # Statistics
        self.e_track_samples: int = 0
        self.tau_samples: int = 0
        self.covariance_updates: int = 0

    def to_dict(self) -> dict:
        return {
            'covariance': {
                'sigma_xx': self.sigma_xx,
                'sigma_yy': self.sigma_yy,
                'sigma_xy': self.sigma_xy,
                'lambda_max': self.lambda_max
            },
            'parameters': {
                'e_track': self.e_track,
                'tau': self.tau,
                'velocity': self.current_velocity
            },
            'margin_breakdown': {
                'estimation': self.estimation_term,
                'tracking': self.tracking_term,
                'latency': self.latency_term,
                'total': self.total_margin
            },
            'sample_counts': {
                'e_track': self.e_track_samples,
                'tau': self.tau_samples,
                'covariance': self.covariance_updates
            }
        }


class DynamicPolicyEnforcerNode(Node):
    """
    Fully dynamic probabilistic geofence policy enforcer.

    All safety margin parameters are measured in real-time:
    - Covariance from AMCL
    - Tracking error from path following
    - Latency from sensor timestamps
    """

    def __init__(self):
        super().__init__('dynamic_policy_enforcer')

        # =====================================================================
        # PARAMETERS
        # =====================================================================

        self.declare_parameter('alpha', 0.95)
        self.declare_parameter('v_max', 0.5)
        self.declare_parameter('geofence_config', '')
        self.declare_parameter('enable_geofence', True)
        self.declare_parameter('enable_projection', True)
        self.declare_parameter('publish_rate_hz', 10.0)

        # Fallback values (used when no measurements available)
        self.declare_parameter('default_e_track', 0.05)
        self.declare_parameter('default_tau', 0.1)
        self.declare_parameter('default_sigma', 0.1)

        # Minimum samples before using measured values
        self.declare_parameter('min_samples_e_track', 20)
        self.declare_parameter('min_samples_tau', 10)

        self._alpha = self.get_parameter('alpha').value
        self._v_max = self.get_parameter('v_max').value
        self._config_path = self.get_parameter('geofence_config').value
        self._enable_geofence = self.get_parameter('enable_geofence').value
        self._enable_projection = self.get_parameter('enable_projection').value
        self._publish_rate = self.get_parameter('publish_rate_hz').value

        self._default_e_track = self.get_parameter('default_e_track').value
        self._default_tau = self.get_parameter('default_tau').value
        self._default_sigma = self.get_parameter('default_sigma').value
        self._min_samples_e_track = self.get_parameter('min_samples_e_track').value
        self._min_samples_tau = self.get_parameter('min_samples_tau').value

        # =====================================================================
        # STATE
        # =====================================================================

        self._lock = Lock()
        self._margin_data = DynamicMarginData()

        # Latest pose and covariance
        self._current_pose: Optional[PoseWithCovarianceStamped] = None
        self._current_covariance: Optional[np.ndarray] = None

        # =====================================================================
        # COMPONENTS
        # =====================================================================

        # Margin calculator
        self._margin_calc = MarginCalculator(
            alpha=self._alpha,
            e_track=self._default_e_track,
            v_max=self._v_max,
            tau=self._default_tau
        )

        # Latency monitor
        self._latency_monitor = LatencyMonitor(
            window_size=50,
            default_latency=self._default_tau
        )

        # Tracking error monitor
        self._tracking_monitor = TrackingErrorMonitor(
            window_size=100,
            default_error=self._default_e_track
        )

        # Geofence geometry
        self._geofence: Optional[GeofenceGeometry] = None
        if self._config_path:
            self._load_geofence(self._config_path)

        # =====================================================================
        # QOS
        # =====================================================================

        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            depth=10
        )

        reliable_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            depth=10
        )

        # =====================================================================
        # SUBSCRIBERS
        # =====================================================================

        cb_group = ReentrantCallbackGroup()

        # AMCL pose with covariance
        self._pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self._pose_callback,
            sensor_qos,
            callback_group=cb_group
        )

        # Odometry for velocity and latency
        self._odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self._odom_callback,
            sensor_qos,
            callback_group=cb_group
        )

        # Planned path for tracking error
        self._path_sub = self.create_subscription(
            Path,
            '/plan',
            self._path_callback,
            reliable_qos,
            callback_group=cb_group
        )

        # Laser scan for latency measurement
        self._scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self._scan_callback,
            sensor_qos,
            callback_group=cb_group
        )

        # Goal for tracking
        self._goal_sub = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self._goal_callback,
            reliable_qos,
            callback_group=cb_group
        )

        # =====================================================================
        # PUBLISHERS
        # =====================================================================

        # Dynamic margin value
        self._margin_pub = self.create_publisher(
            Float64,
            '~/dynamic_margin',
            reliable_qos
        )

        # Full margin data as JSON
        self._margin_data_pub = self.create_publisher(
            String,
            '~/margin_breakdown',
            reliable_qos
        )

        # Safety status
        self._status_pub = self.create_publisher(
            String,
            '~/safety_status',
            reliable_qos
        )

        # =====================================================================
        # SERVICES
        # =====================================================================

        self._validate_srv = self.create_service(
            Trigger,
            '~/get_margin_info',
            self._get_margin_info_callback
        )

        # =====================================================================
        # TIMER
        # =====================================================================

        self._update_timer = self.create_timer(
            1.0 / self._publish_rate,
            self._update_callback
        )

        # =====================================================================
        # STARTUP
        # =====================================================================

        self._log_startup()

    def _load_geofence(self, config_path: str) -> bool:
        """Load geofence configuration."""
        try:
            from pathlib import Path
            path = Path(config_path)
            if path.exists():
                self._geofence = GeofenceGeometry.from_yaml(path)
                self.get_logger().info(
                    f"Loaded geofence with {len(self._geofence.get_zone_names())} zones"
                )
                return True
        except Exception as e:
            self.get_logger().error(f"Failed to load geofence: {e}")
        return False

    def _log_startup(self):
        """Log configuration on startup."""
        self.get_logger().info("=" * 60)
        self.get_logger().info("Dynamic Probabilistic Geofence Policy Enforcer")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"  α (confidence):     {self._alpha}")
        self.get_logger().info(f"  v_max:              {self._v_max} m/s")
        self.get_logger().info(f"  Default e_track:    {self._default_e_track} m")
        self.get_logger().info(f"  Default τ:          {self._default_tau} s")
        self.get_logger().info("  Parameters: DYNAMIC (measured in real-time)")
        self.get_logger().info("=" * 60)

    # =========================================================================
    # CALLBACKS
    # =========================================================================

    def _pose_callback(self, msg: PoseWithCovarianceStamped):
        """Process AMCL pose with covariance."""
        with self._lock:
            self._current_pose = msg
            self._current_covariance = np.array(msg.pose.covariance)

            # Extract position covariance
            cov = self._current_covariance.reshape(6, 6)
            self._margin_data.sigma_xx = cov[0, 0]
            self._margin_data.sigma_yy = cov[1, 1]
            self._margin_data.sigma_xy = cov[0, 1]
            self._margin_data.covariance_updates += 1

            # Update tracking error with position
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            self._tracking_monitor.update_position(x, y)

    def _odom_callback(self, msg: Odometry):
        """Process odometry for velocity and latency."""
        # Record sensor latency
        self._latency_monitor.record_sensor_latency_from_header(msg.header)

        # Extract velocity
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        velocity = np.sqrt(vx*vx + vy*vy)

        with self._lock:
            self._margin_data.current_velocity = velocity

        # Check for command response (latency measurement)
        self._latency_monitor.complete_command_measurement(vx)

    def _path_callback(self, msg: Path):
        """Process planned path for tracking error."""
        self._tracking_monitor.set_path_from_nav_path(msg)

    def _scan_callback(self, msg: LaserScan):
        """Process laser scan for latency measurement."""
        self._latency_monitor.record_sensor_latency_from_header(msg.header)

    def _goal_callback(self, msg: PoseStamped):
        """Process navigation goal."""
        x = msg.pose.position.x
        y = msg.pose.position.y
        self._tracking_monitor.set_goal(x, y)

    def _get_margin_info_callback(self, request, response):
        """Service callback to get margin information."""
        with self._lock:
            data = self._margin_data.to_dict()
        response.success = True
        response.message = json.dumps(data, indent=2)
        return response

    # =========================================================================
    # MAIN UPDATE LOOP
    # =========================================================================

    def _update_callback(self):
        """Main update: compute dynamic margin and publish."""
        with self._lock:
            # Get measured values
            tau_stats = self._latency_monitor.get_stats()
            e_track_stats = self._tracking_monitor.get_stats()

            # Use measured tau if enough samples
            if tau_stats.sample_count >= self._min_samples_tau:
                measured_tau = self._latency_monitor.get_latency()
                self._margin_data.tau = measured_tau
                self._margin_data.tau_samples = tau_stats.sample_count
            else:
                self._margin_data.tau = self._default_tau

            # Use measured e_track if enough samples
            if e_track_stats.sample_count >= self._min_samples_e_track:
                measured_e_track = self._tracking_monitor.get_tracking_error()
                self._margin_data.e_track = measured_e_track
                self._margin_data.e_track_samples = e_track_stats.sample_count
            else:
                self._margin_data.e_track = self._default_e_track

            # Update margin calculator with dynamic values
            self._margin_calc.update_parameters(
                e_track=self._margin_data.e_track,
                tau=self._margin_data.tau
            )

            # Compute margin if we have covariance
            if self._current_covariance is not None:
                result = self._margin_calc.compute_margin(
                    self._current_covariance,
                    current_velocity=self._margin_data.current_velocity
                )

                self._margin_data.lambda_max = result.lambda_max
                self._margin_data.estimation_term = result.estimation_term
                self._margin_data.tracking_term = result.tracking_error
                self._margin_data.latency_term = result.latency_term
                self._margin_data.total_margin = result.total_margin
            else:
                # Use default covariance
                default_cov = np.zeros(36)
                default_cov[0] = self._default_sigma ** 2
                default_cov[7] = self._default_sigma ** 2
                result = self._margin_calc.compute_margin(default_cov)
                self._margin_data.total_margin = result.total_margin

        # Publish margin
        margin_msg = Float64()
        margin_msg.data = self._margin_data.total_margin
        self._margin_pub.publish(margin_msg)

        # Publish full breakdown
        data_msg = String()
        data_msg.data = json.dumps(self._margin_data.to_dict())
        self._margin_data_pub.publish(data_msg)

        # Publish status
        self._publish_status()

    def _publish_status(self):
        """Publish human-readable status."""
        with self._lock:
            md = self._margin_data

        status = (
            f"Margin: {md.total_margin*100:.1f}cm | "
            f"Est: {md.estimation_term*100:.1f}cm | "
            f"Track: {md.tracking_term*100:.1f}cm | "
            f"Lat: {md.latency_term*100:.1f}cm | "
            f"v: {md.current_velocity:.2f}m/s | "
            f"τ: {md.tau*1000:.0f}ms ({md.tau_samples} samples) | "
            f"e: {md.e_track*100:.1f}cm ({md.e_track_samples} samples)"
        )

        msg = String()
        msg.data = status
        self._status_pub.publish(msg)

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def get_current_margin(self) -> float:
        """Get current dynamic safety margin."""
        with self._lock:
            return self._margin_data.total_margin

    def get_margin_breakdown(self) -> DynamicMarginData:
        """Get full margin breakdown."""
        with self._lock:
            return self._margin_data

    def validate_goal(
        self,
        goal_x: float,
        goal_y: float
    ) -> Tuple[bool, float, float, str]:
        """
        Validate a goal against the geofence.

        Returns:
            (is_valid, safe_x, safe_y, message)
        """
        if self._geofence is None:
            return (True, goal_x, goal_y, "Geofence not configured")

        margin = self.get_current_margin()

        return self._geofence.validate_goal(
            goal_x, goal_y,
            margin=margin,
            enable_projection=self._enable_projection
        )


def main(args=None):
    rclpy.init(args=args)
    node = DynamicPolicyEnforcerNode()

    try:
        node.get_logger().info("Dynamic Policy Enforcer running...")
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
