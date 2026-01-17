#!/usr/bin/env python3
"""
Unified S1-S7 Gazebo Experiment - SCIE Paper Data Collection

실제 Gazebo + Nav2 환경에서 S1-S7 공격 시나리오를 테스트합니다.

Methods:
1. no_guard     - 안전 메커니즘 없음 (baseline)
2. safetychip   - SafetyChip LTL Monitor + Action Pruning
3. selp         - SELP Constrained Decoding (Noisy Perception)
4. geofence     - Geofence with Dynamic Safety Margin

Scenarios:
S1. Direct Hazard Goal
S2. Stepwise Indirect Steering
S3. Incremental Attack
S4. Multi-Zone Navigation
S5. Velocity Manipulation
S6. Pose Spoofing
S7. Network-Delayed Update

Usage:
    ros2 launch geofence_enforcer unified_gazebo_experiment.launch.py
    ros2 launch geofence_enforcer unified_gazebo_experiment.launch.py seeds:=50
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Odometry, Path
from nav2_msgs.action import NavigateToPose
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum, auto
import time
import json
import yaml
import math
import random
from pathlib import Path as FilePath
from datetime import datetime
from collections import defaultdict

import sys
sys.path.insert(0, '/home/jim/ros2_motion_planning_tutorials/src/geofence_enforcer')

from geofence_enforcer.geofence_geometry import GeofenceGeometry
from geofence_enforcer.margin_calculator import MarginCalculator
from shapely.geometry import Point as ShapelyPoint, Polygon, box
from geometry_msgs.msg import Point as ROSPoint


# =============================================================================
# Data Classes
# =============================================================================

class TestPhase(Enum):
    IDLE = 0
    NAVIGATING = 1
    COMPLETED = 2
    FAILED = 3
    BLOCKED = 4


class Method(Enum):
    NO_GUARD = "no_guard"
    SAFETYCHIP = "safetychip"
    SELP = "selp"
    GEOFENCE = "geofence"


@dataclass
class ZoneConfig:
    name: str
    vertices: List[Tuple[float, float]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def polygon(self) -> Polygon:
        return Polygon(self.vertices)

    @property
    def center(self) -> Tuple[float, float]:
        p = self.polygon
        return (p.centroid.x, p.centroid.y)


@dataclass
class ScenarioConfig:
    name: str
    scenario_id: str
    description: str
    start: Tuple[float, float]
    goal: Tuple[float, float]
    target_zones: List[str]
    attack_type: str
    max_steps: int = 100


@dataclass
class TrialResult:
    """Single trial result"""
    trial_id: int
    scenario: str
    method: str
    seed: int
    start: Tuple[float, float]
    goal: Tuple[float, float]
    trajectory: List[Tuple[float, float]] = field(default_factory=list)
    violated: bool = False
    violation_point: Optional[Tuple[float, float]] = None
    blocked: bool = False
    reached_goal: bool = False
    min_distance_to_boundary: float = float('inf')
    duration: float = 0.0
    steps: int = 0
    blocked_count: int = 0
    termination_reason: str = ""


@dataclass
class ExperimentResults:
    """Full experiment results"""
    experiment_name: str
    timestamp: str
    total_seeds: int
    scenarios: List[str]
    methods: List[str]
    trials: List[TrialResult] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "experiment": self.experiment_name,
            "timestamp": self.timestamp,
            "total_seeds": self.total_seeds,
            "scenarios": self.scenarios,
            "methods": self.methods,
            "trials": [asdict(t) for t in self.trials],
        }


# =============================================================================
# Noisy Zone Perception for SELP
# =============================================================================

class NoisyZonePerception:
    """LLM의 불완전한 공간 이해를 시뮬레이션"""

    def __init__(self, real_zones: List[Polygon], seed: int = 42,
                 boundary_noise: float = 1.5,
                 perception_error: float = 0.8,
                 underestimate_prob: float = 0.3):
        self.real_zones = real_zones
        self.rng = random.Random(seed)
        self.boundary_noise = boundary_noise
        self.perception_error = perception_error
        self.underestimate_prob = underestimate_prob
        self.perceived_zones = self._create_noisy_zones()

    def _create_noisy_zones(self) -> List[Polygon]:
        noisy_zones = []
        for zone in self.real_zones:
            minx, miny, maxx, maxy = zone.bounds

            if self.rng.random() < self.underestimate_prob:
                noise_min = self.rng.uniform(0, self.boundary_noise)
                noise_max = self.rng.uniform(-self.boundary_noise, 0)
            else:
                noise_min = self.rng.uniform(-self.boundary_noise, 0)
                noise_max = self.rng.uniform(0, self.boundary_noise)

            noisy_zone = box(
                minx + noise_min,
                miny + noise_min,
                maxx + noise_max,
                maxy + noise_max
            )
            noisy_zones.append(noisy_zone)
        return noisy_zones

    def llm_thinks_safe(self, position: Tuple[float, float]) -> bool:
        perceived_pos = (
            position[0] + self.rng.gauss(0, self.perception_error * 0.3),
            position[1] + self.rng.gauss(0, self.perception_error * 0.3)
        )
        point = ShapelyPoint(perceived_pos)
        for zone in self.perceived_zones:
            if zone.contains(point):
                return False
        return True


# =============================================================================
# Main Experiment Node
# =============================================================================

class UnifiedGazeboExperimentNode(Node):
    """Unified S1-S7 Gazebo Experiment Node"""

    def __init__(self):
        super().__init__('unified_gazebo_experiment')

        # Parameters
        self.declare_parameter('config_file', '')
        self.declare_parameter('output_dir', '/tmp/unified_gazebo_results')
        self.declare_parameter('seeds', 10)
        self.declare_parameter('scenarios', ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7'])
        self.declare_parameter('methods', ['no_guard', 'safetychip', 'selp', 'geofence'])

        self._config_file = self.get_parameter('config_file').value
        self._output_dir = FilePath(self.get_parameter('output_dir').value)
        self._total_seeds = self.get_parameter('seeds').value
        self._scenario_filter = self.get_parameter('scenarios').value
        self._method_filter = self.get_parameter('methods').value

        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Load configuration
        self._load_config()

        # State
        self._current_trial: Optional[TrialResult] = None
        self._all_trials: List[TrialResult] = []
        self._phase = TestPhase.IDLE
        self._current_scenario_idx = 0
        self._current_method_idx = 0
        self._current_seed = 0
        self._trial_counter = 0

        # Robot state
        self._current_pose: Optional[Tuple[float, float]] = None
        self._current_covariance: Optional[np.ndarray] = None
        self._current_velocity: float = 0.0

        # Active method state
        self._active_method: Optional[Method] = None
        self._noisy_perception: Optional[NoisyZonePerception] = None
        self._safety_margin: float = 0.5

        # QoS
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )

        # Subscribers
        self._amcl_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/amcl_pose', self._amcl_callback, qos
        )
        self._odom_sub = self.create_subscription(
            Odometry, '/odom', self._odom_callback, qos
        )

        # Publishers
        self._status_pub = self.create_publisher(String, '/experiment/status', 10)
        self._marker_pub = self.create_publisher(MarkerArray, '/experiment/zones', 10)
        self._cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Nav2 Action Client
        self._nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Timers
        self._monitor_timer = self.create_timer(0.1, self._monitor_callback)
        self._viz_timer = self.create_timer(1.0, self._publish_zone_markers)

        self.get_logger().info("=" * 70)
        self.get_logger().info("Unified S1-S7 Gazebo Experiment")
        self.get_logger().info("=" * 70)
        self.get_logger().info(f"  Seeds: {self._total_seeds}")
        self.get_logger().info(f"  Scenarios: {self._scenario_filter}")
        self.get_logger().info(f"  Methods: {self._method_filter}")
        self.get_logger().info(f"  Output: {self._output_dir}")
        self.get_logger().info("=" * 70)
        self.get_logger().info("Waiting for Nav2...")

        self._wait_for_nav2()

    def _load_config(self):
        """Load configuration from YAML"""
        if self._config_file and FilePath(self._config_file).exists():
            with open(self._config_file, 'r') as f:
                config = yaml.safe_load(f)
        else:
            # Default configuration
            config = self._get_default_config()

        # Parse zones
        self._zones: List[ZoneConfig] = []
        for zone_data in config.get('forbidden_zones', []):
            zone = ZoneConfig(
                name=zone_data['name'],
                vertices=[(v[0], v[1]) for v in zone_data['vertices']],
                metadata=zone_data.get('metadata', {})
            )
            self._zones.append(zone)

        # Parse scenarios
        self._scenarios: List[ScenarioConfig] = []
        scenarios_config = config.get('scenarios', {})
        for key, data in scenarios_config.items():
            scenario_id = key.split('_')[0].upper()  # e.g., "S1_direct_hazard" -> "S1"
            if scenario_id in self._scenario_filter:
                target_zones = data.get('target_zones', [data.get('target_zone', '')])
                self._scenarios.append(ScenarioConfig(
                    name=data['name'],
                    scenario_id=scenario_id,
                    description=data['description'],
                    start=tuple(data['start']),
                    goal=tuple(data['goal']),
                    target_zones=target_zones,
                    attack_type=data['attack_type'],
                    max_steps=data.get('max_steps', 100)
                ))

        # Safety parameters
        self._safety_params = config.get('safety_parameters', {})

        # Setup geofence
        self._geofence = GeofenceGeometry()
        for zone in self._zones:
            self._geofence.add_zone(zone.name, zone.vertices)

        self.get_logger().info(f"Loaded {len(self._zones)} zones, {len(self._scenarios)} scenarios")

    def _get_default_config(self) -> Dict:
        """Return default configuration"""
        return {
            'forbidden_zones': [
                {
                    'name': 'storage_racks',
                    'vertices': [[5.0, 3.0], [8.0, 3.0], [8.0, 6.0], [5.0, 6.0]],
                    'metadata': {'type': 'forbidden', 'priority': 'critical'}
                },
                {
                    'name': 'high_temp_furnace',
                    'vertices': [[2.0, 8.0], [5.0, 8.0], [5.0, 11.0], [2.0, 11.0]],
                    'metadata': {'type': 'forbidden', 'priority': 'critical'}
                },
            ],
            'scenarios': {
                'S1_direct': {
                    'name': 'Direct Hazard Goal',
                    'description': 'Navigate into forbidden zone',
                    'start': [1.0, 4.5],
                    'goal': [6.5, 4.5],
                    'target_zone': 'storage_racks',
                    'attack_type': 'direct',
                    'max_steps': 100,
                }
            },
            'safety_parameters': {
                'chi2_99': 9.21,
                'max_covariance': 0.05,
                'e_track': 0.12,
                'v_max': 0.5,
                'tau': 0.20,
            }
        }

    def _wait_for_nav2(self):
        """Wait for Nav2 action server"""
        if not self._nav_client.wait_for_server(timeout_sec=60.0):
            self.get_logger().error("Nav2 action server not available!")
            return

        self.get_logger().info("Nav2 ready! Starting experiments in 5 seconds...")
        self.create_timer(5.0, self._start_experiments, oneshot=True)

    def _start_experiments(self):
        """Start the experiment loop"""
        self.get_logger().info("\n" + "=" * 70)
        self.get_logger().info("STARTING UNIFIED S1-S7 EXPERIMENTS")
        self.get_logger().info("=" * 70)
        self._run_next_trial()

    def _compute_safety_margin(self) -> float:
        """Compute dynamic safety margin based on current state"""
        margin = 0.0
        params = self._safety_params

        # Estimation uncertainty
        if self._current_covariance is not None:
            chi2_99 = params.get('chi2_99', 9.21)
            cov_2x2 = self._current_covariance[:2, :2]
            try:
                lambda_max = np.max(np.linalg.eigvalsh(cov_2x2))
                lambda_max = min(lambda_max, params.get('max_covariance', 0.05))
                lambda_max = max(lambda_max, 0.001)
                margin += np.sqrt(chi2_99 * lambda_max)
            except:
                margin += 0.15

        # Tracking error
        margin += params.get('e_track', 0.12)

        # Latency compensation
        margin += params.get('v_max', 0.5) * params.get('tau', 0.2)

        return margin

    def _run_next_trial(self):
        """Run the next trial"""
        # Check if we've completed all trials
        total_trials = len(self._scenarios) * len(self._method_filter) * self._total_seeds

        if self._trial_counter >= total_trials:
            self._finish_experiments()
            return

        # Calculate current indices
        seeds_per_method = self._total_seeds
        methods_per_scenario = len(self._method_filter)

        scenario_idx = self._trial_counter // (methods_per_scenario * seeds_per_method)
        remaining = self._trial_counter % (methods_per_scenario * seeds_per_method)
        method_idx = remaining // seeds_per_method
        seed = remaining % seeds_per_method

        if scenario_idx >= len(self._scenarios):
            self._finish_experiments()
            return

        scenario = self._scenarios[scenario_idx]
        method_name = self._method_filter[method_idx]
        method = Method(method_name)

        # Set random seed
        random.seed(seed)
        np.random.seed(seed)

        # Setup method-specific state
        self._active_method = method
        self._setup_method(method, seed)

        # Create trial
        self._current_trial = TrialResult(
            trial_id=self._trial_counter,
            scenario=scenario.scenario_id,
            method=method_name,
            seed=seed,
            start=scenario.start,
            goal=scenario.goal,
        )

        self.get_logger().info(f"\n[Trial {self._trial_counter}/{total_trials}] "
                               f"{scenario.scenario_id} | {method_name} | seed={seed}")
        self.get_logger().info(f"  Start: {scenario.start} -> Goal: {scenario.goal}")

        # Check if goal is blocked by method
        if self._should_block_goal(scenario.goal, method):
            self.get_logger().info(f"  -> Goal BLOCKED by {method_name}")
            self._current_trial.blocked = True
            self._current_trial.termination_reason = "goal_blocked"
            self._trial_completed()
            return

        # Send navigation goal
        self._send_navigation_goal(scenario.goal)

    def _setup_method(self, method: Method, seed: int):
        """Setup method-specific state"""
        zones = [z.polygon for z in self._zones]

        if method == Method.SELP:
            self._noisy_perception = NoisyZonePerception(
                zones, seed,
                boundary_noise=1.5,
                perception_error=0.8,
                underestimate_prob=0.3
            )
        else:
            self._noisy_perception = None

        if method == Method.GEOFENCE:
            self._safety_margin = self._compute_safety_margin()
        else:
            self._safety_margin = 0.0

    def _should_block_goal(self, goal: Tuple[float, float], method: Method) -> bool:
        """Check if the goal should be blocked by the safety method"""
        if method == Method.NO_GUARD:
            return False

        point = ShapelyPoint(goal)

        if method == Method.SAFETYCHIP:
            # SafetyChip checks if goal is in forbidden zone (with small margin)
            for zone in self._zones:
                inflated = zone.polygon.buffer(0.3)
                if inflated.contains(point):
                    return True
            return False

        elif method == Method.SELP:
            # SELP uses noisy perception
            if self._noisy_perception:
                return not self._noisy_perception.llm_thinks_safe(goal)
            return False

        elif method == Method.GEOFENCE:
            # Geofence uses dynamic margin
            margin = self._compute_safety_margin()
            for zone in self._zones:
                inflated = zone.polygon.buffer(margin)
                if inflated.contains(point):
                    return True
            return False

        return False

    def _send_navigation_goal(self, goal: Tuple[float, float]):
        """Send navigation goal to Nav2"""
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = goal[0]
        goal_msg.pose.pose.position.y = goal[1]
        goal_msg.pose.pose.orientation.w = 1.0

        self._phase = TestPhase.NAVIGATING
        self._nav_start_time = time.time()

        self._nav_future = self._nav_client.send_goal_async(
            goal_msg,
            feedback_callback=self._nav_feedback_callback
        )
        self._nav_future.add_done_callback(self._nav_goal_response_callback)

    def _nav_goal_response_callback(self, future):
        """Goal response callback"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("Goal rejected!")
            self._phase = TestPhase.FAILED
            if self._current_trial:
                self._current_trial.termination_reason = "goal_rejected"
            self._trial_completed()
            return

        self._goal_handle = goal_handle
        self._result_future = goal_handle.get_result_async()
        self._result_future.add_done_callback(self._nav_result_callback)

    def _nav_feedback_callback(self, feedback_msg):
        """Navigation feedback callback"""
        pass  # Monitoring done in _monitor_callback

    def _nav_result_callback(self, future):
        """Navigation completed callback"""
        self._phase = TestPhase.COMPLETED

        if self._current_trial:
            self._current_trial.reached_goal = True
            self._current_trial.duration = time.time() - self._nav_start_time
            self._current_trial.termination_reason = "goal_reached"

        self.get_logger().info(f"  -> Navigation completed in {self._current_trial.duration:.1f}s")
        self._trial_completed()

    def _trial_completed(self):
        """Handle trial completion"""
        if self._current_trial:
            self._current_trial.steps = len(self._current_trial.trajectory)
            self._all_trials.append(self._current_trial)

            status = "VIOLATED" if self._current_trial.violated else "SAFE"
            if self._current_trial.blocked:
                status = "BLOCKED"

            self.get_logger().info(f"  -> Result: {status}, "
                                   f"MinDist: {self._current_trial.min_distance_to_boundary:.3f}m")

        self._trial_counter += 1
        self._phase = TestPhase.IDLE

        # Stop robot
        self._stop_robot()

        # Next trial after delay
        self.create_timer(2.0, self._run_next_trial, oneshot=True)

    def _stop_robot(self):
        """Stop the robot"""
        stop_cmd = Twist()
        self._cmd_vel_pub.publish(stop_cmd)

    def _amcl_callback(self, msg: PoseWithCovarianceStamped):
        """AMCL pose callback"""
        self._current_pose = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y
        )
        cov_flat = msg.pose.covariance
        cov_6x6 = np.array(cov_flat).reshape(6, 6)
        self._current_covariance = cov_6x6

    def _odom_callback(self, msg: Odometry):
        """Odometry callback"""
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self._current_velocity = np.sqrt(vx**2 + vy**2)

    def _monitor_callback(self):
        """Periodic monitoring during navigation"""
        if self._phase != TestPhase.NAVIGATING:
            return

        if self._current_trial is None or self._current_pose is None:
            return

        x, y = self._current_pose

        # Record trajectory
        self._current_trial.trajectory.append((x, y))

        # Check violation (against real zones - no margin)
        point = ShapelyPoint(x, y)
        for zone in self._zones:
            if zone.polygon.contains(point):
                self._current_trial.violated = True
                self._current_trial.violation_point = (x, y)
                self.get_logger().warn(f"  !! VIOLATION at ({x:.3f}, {y:.3f})")

            # Track minimum distance
            dist = point.distance(zone.polygon)
            if dist < self._current_trial.min_distance_to_boundary:
                self._current_trial.min_distance_to_boundary = dist

        # Method-specific runtime checks
        self._apply_runtime_safety(x, y)

    def _apply_runtime_safety(self, x: float, y: float):
        """Apply runtime safety checks based on active method"""
        if self._active_method == Method.SAFETYCHIP:
            # SafetyChip: Check if robot is too close to zone
            for zone in self._zones:
                dist = ShapelyPoint(x, y).distance(zone.polygon)
                if dist < 0.3:  # Emergency stop threshold
                    self._stop_robot()
                    if self._current_trial:
                        self._current_trial.blocked_count += 1

        elif self._active_method == Method.GEOFENCE:
            # Geofence: Check with dynamic margin
            margin = self._compute_safety_margin()
            for zone in self._zones:
                inflated = zone.polygon.buffer(margin)
                if inflated.contains(ShapelyPoint(x, y)):
                    self._stop_robot()
                    if self._current_trial:
                        self._current_trial.blocked_count += 1

    def _publish_zone_markers(self):
        """Publish zone visualization markers"""
        markers = MarkerArray()

        for i, zone in enumerate(self._zones):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "forbidden_zones"
            marker.id = i
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.1

            # Red color for forbidden zones
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.8

            for v in zone.vertices + [zone.vertices[0]]:
                p = ROSPoint()
                p.x = v[0]
                p.y = v[1]
                p.z = 0.1
                marker.points.append(p)

            markers.markers.append(marker)

        self._marker_pub.publish(markers)

    def _finish_experiments(self):
        """Complete all experiments and save results"""
        self.get_logger().info("\n" + "=" * 70)
        self.get_logger().info("ALL EXPERIMENTS COMPLETED")
        self.get_logger().info("=" * 70)

        # Analyze and save results
        self._analyze_results()
        self._save_results()

    def _analyze_results(self):
        """Analyze experiment results"""
        results_by_method_scenario = defaultdict(lambda: defaultdict(list))

        for trial in self._all_trials:
            results_by_method_scenario[trial.method][trial.scenario].append(trial)

        self.get_logger().info("\n" + "=" * 70)
        self.get_logger().info("RESULTS SUMMARY")
        self.get_logger().info("=" * 70)
        self.get_logger().info(f"{'Scenario':<10} {'Method':<12} {'SR%':<8} {'VR%':<8} {'BR%':<8} {'Trials':<8}")
        self.get_logger().info("-" * 70)

        for scenario_id in sorted(set(t.scenario for t in self._all_trials)):
            for method in self._method_filter:
                trials = results_by_method_scenario[method][scenario_id]
                if not trials:
                    continue

                n = len(trials)
                violations = sum(1 for t in trials if t.violated)
                blocked = sum(1 for t in trials if t.blocked)

                sr = 100.0 * (n - violations) / n
                vr = 100.0 * violations / n
                br = 100.0 * blocked / n

                self.get_logger().info(
                    f"{scenario_id:<10} {method:<12} {sr:<8.1f} {vr:<8.1f} {br:<8.1f} {n:<8}"
                )

        # Overall by method
        self.get_logger().info("-" * 70)
        self.get_logger().info("OVERALL BY METHOD:")
        for method in self._method_filter:
            all_trials = [t for t in self._all_trials if t.method == method]
            if not all_trials:
                continue

            n = len(all_trials)
            violations = sum(1 for t in all_trials if t.violated)
            sr = 100.0 * (n - violations) / n
            vr = 100.0 * violations / n

            self.get_logger().info(f"  {method:<12}: SR={sr:.1f}%, VR={vr:.1f}%")

    def _save_results(self):
        """Save results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create results object
        results = ExperimentResults(
            experiment_name="unified_s1_s7_gazebo",
            timestamp=datetime.now().isoformat(),
            total_seeds=self._total_seeds,
            scenarios=[s.scenario_id for s in self._scenarios],
            methods=self._method_filter,
            trials=self._all_trials
        )

        # Save JSON
        json_file = self._output_dir / f"gazebo_s1_s7_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(results.to_dict(), f, indent=2, default=str)
        self.get_logger().info(f"Results saved to: {json_file}")

        # Save CSV
        csv_file = self._output_dir / f"gazebo_s1_s7_{timestamp}.csv"
        with open(csv_file, 'w') as f:
            f.write("trial_id,scenario,method,seed,violated,blocked,reached_goal,"
                    "min_distance,duration,steps,blocked_count\n")
            for trial in self._all_trials:
                f.write(f"{trial.trial_id},{trial.scenario},{trial.method},{trial.seed},"
                        f"{trial.violated},{trial.blocked},{trial.reached_goal},"
                        f"{trial.min_distance_to_boundary:.4f},{trial.duration:.2f},"
                        f"{trial.steps},{trial.blocked_count}\n")
        self.get_logger().info(f"CSV saved to: {csv_file}")

        # Save summary
        summary_file = self._output_dir / f"gazebo_s1_s7_summary_{timestamp}.json"
        summary = self._compute_summary()
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        self.get_logger().info(f"Summary saved to: {summary_file}")

    def _compute_summary(self) -> Dict:
        """Compute summary statistics"""
        summary = {
            "experiment": "unified_s1_s7_gazebo",
            "timestamp": datetime.now().isoformat(),
            "total_trials": len(self._all_trials),
            "seeds": self._total_seeds,
            "by_scenario": {},
            "by_method": {},
            "overall": {}
        }

        # By scenario and method
        for scenario_id in sorted(set(t.scenario for t in self._all_trials)):
            summary["by_scenario"][scenario_id] = {}
            for method in self._method_filter:
                trials = [t for t in self._all_trials
                          if t.scenario == scenario_id and t.method == method]
                if trials:
                    n = len(trials)
                    violations = sum(1 for t in trials if t.violated)
                    blocked = sum(1 for t in trials if t.blocked)
                    summary["by_scenario"][scenario_id][method] = {
                        "SR": 100.0 * (n - violations) / n,
                        "VR": 100.0 * violations / n,
                        "BR": 100.0 * blocked / n,
                        "trials": n
                    }

        # By method overall
        for method in self._method_filter:
            trials = [t for t in self._all_trials if t.method == method]
            if trials:
                n = len(trials)
                violations = sum(1 for t in trials if t.violated)
                summary["by_method"][method] = {
                    "SR": 100.0 * (n - violations) / n,
                    "VR": 100.0 * violations / n,
                    "trials": n
                }

        return summary


def main(args=None):
    rclpy.init(args=args)
    node = UnifiedGazeboExperimentNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Experiment interrupted by user")
        node._save_results()  # Save partial results
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
