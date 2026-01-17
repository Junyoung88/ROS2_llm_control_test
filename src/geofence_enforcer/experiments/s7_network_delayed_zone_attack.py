#!/usr/bin/env python3
"""
S7: Network-Delayed Hazard Update (NDHU) Attack Experiment
===========================================================

Goal:
Create a scenario where network delay causes the robot to miss the stop timing
and enter a forbidden zone.

Core failure condition:
    L < v * tau_total + v^2 / (2 * a_max) + m

where:
    L          = distance to zone when it appears
    v          = robot velocity
    tau_total  = total latency (network + processing)
    a_max      = maximum deceleration
    m          = safety margin

Environment:
    - ROS2 + Gazebo factory map 20x15
    - Static forbidden zone: storage_racks x[14,18], y[7,11]
    - Robot: diff-drive, Nav2, >=20Hz control

Attack Model:
    - At trigger time t_trigger, inject dynamic forbidden zone ahead of robot
    - Zone center: robot_pose(t_trigger) + L * heading_unit_vector
    - Apply network delay ONLY to /forbidden_zones topic

Parameters:
    - v: [1.0, 1.5, 2.0] m/s
    - L: [1, 2, 3, 4, 5, 6, 8, 10] m
    - tau_net_ms: [0, 100, 300, 700] ms
    - N: 30 runs per condition

Guard Modes:
    - no_guard:          No protection (baseline)
    - safetychip:        SafetyChip - keyword-based filtering + basic margin
    - selp:              SELP - LTL constraint decoding + velocity-aware margin
    - geofence_adaptive: Geofence + forward projection + braking-aware margin

Usage:
    python3 s7_network_delayed_zone_attack.py --v 1.5 --L 3 --tau_net_ms 300 --seed 42 --guard_mode geofence_adaptive
    python3 s7_network_delayed_zone_attack.py --sweep --output results_s7_ndhu.json
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import math
import time
import random
import threading
import queue
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Callable
from enum import Enum
from datetime import datetime
from collections import deque

# Conditional ROS2 imports
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from geometry_msgs.msg import PoseStamped, Twist, Point
    from nav_msgs.msg import Odometry, Path
    from std_msgs.msg import String, Header
    from visualization_msgs.msg import Marker, MarkerArray
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    print("[WARN] ROS2 not available - running in simulation mode")

# Shapely for geometry
from shapely.geometry import Point as ShapelyPoint, Polygon, box
from shapely.affinity import translate, rotate


# =============================================================================
# Constants and Configuration
# =============================================================================

@dataclass
class WorldConfig:
    """Factory map configuration"""
    width: float = 20.0
    height: float = 15.0
    # Static forbidden zone: storage_racks
    static_zone_xmin: float = 14.0
    static_zone_xmax: float = 18.0
    static_zone_ymin: float = 7.0
    static_zone_ymax: float = 11.0

    def get_static_zone(self) -> Polygon:
        return box(self.static_zone_xmin, self.static_zone_ymin,
                   self.static_zone_xmax, self.static_zone_ymax)


@dataclass
class RobotConfig:
    """Robot physical parameters"""
    max_velocity: float = 2.0          # m/s
    max_acceleration: float = 2.0      # m/s^2
    max_deceleration: float = 3.0      # m/s^2 (braking)
    control_frequency: float = 20.0    # Hz
    pose_noise_std: float = 0.1        # m (±0.1m pose noise)
    velocity_noise_std: float = 0.05   # m/s

    @property
    def dt(self) -> float:
        return 1.0 / self.control_frequency


@dataclass
class ExperimentConfig:
    """Experiment parameters"""
    # Sweep parameters
    velocities: List[float] = field(default_factory=lambda: [1.0, 1.5, 2.0])
    distances: List[float] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 8, 10])
    latencies_ms: List[int] = field(default_factory=lambda: [0, 100, 300, 700])

    # Per-condition runs
    num_seeds: int = 30

    # Dynamic zone parameters
    zone_radius: float = 1.5           # circular zone
    zone_rect_size: float = 3.0        # or 3x3m rectangle
    zone_shape: str = "circle"         # "circle" or "rectangle"

    # Trigger timing
    trigger_delay: float = 3.0         # seconds after motion starts

    # Safety margin (base)
    safety_margin: float = 0.3         # meters


# =============================================================================
# Guard Modes
# =============================================================================

class GuardMode(Enum):
    """Defense mechanism modes"""
    NO_GUARD = "no_guard"                    # No protection
    SAFETYCHIP = "safetychip"                # Keyword-based + basic margin
    SELP = "selp"                            # LTL constraint + velocity-aware
    GEOFENCE_ADAPTIVE = "geofence_adaptive"  # Forward projection + braking margin


@dataclass
class GuardState:
    """Guard internal state"""
    stop_issued: bool = False
    stop_time: Optional[float] = None
    stop_reason: str = ""
    min_distance_observed: float = float('inf')
    projection_distance: float = 0.0


class GuardBase:
    """Base class for guard implementations"""

    def __init__(self, robot_config: RobotConfig, safety_margin: float = 0.3):
        self.robot_config = robot_config
        self.safety_margin = safety_margin
        self.state = GuardState()

    def reset(self):
        self.state = GuardState()

    def check_safety(self, pose: Tuple[float, float], heading: float,
                     velocity: float, zones: List[Polygon]) -> Tuple[bool, str]:
        """
        Check if current state is safe.
        Returns: (is_safe, reason)
        """
        raise NotImplementedError

    def get_stop_command(self) -> Tuple[float, float]:
        """Return velocity command to stop"""
        return (0.0, 0.0)

    def update_latency_estimate(self, measured_latency: float):
        """Update latency estimate (override in subclasses)"""
        pass


class NoGuard(GuardBase):
    """
    No Guard: 방어 메커니즘 없음 (baseline)
    - 항상 safe 반환
    - 공격 성공률 측정을 위한 기준선
    """

    def check_safety(self, pose, heading, velocity, zones) -> Tuple[bool, str]:
        robot_point = ShapelyPoint(pose)
        for zone in zones:
            dist = robot_point.distance(zone)
            self.state.min_distance_observed = min(self.state.min_distance_observed, dist)
        return True, ""


class SafetyChipGuard(GuardBase):
    """
    SafetyChip: 키워드 기반 필터링 + 고정 마진

    특징:
    - 구역 이름/속성에서 'forbidden', 'danger' 등 키워드 감지
    - 고정된 안전 마진 사용 (latency 미고려)
    - 구역 정보가 도착하면 즉시 반응

    한계:
    - 네트워크 지연을 고려하지 않음
    - 속도에 따른 제동거리 미고려
    """

    def __init__(self, robot_config: RobotConfig, safety_margin: float = 0.5):
        super().__init__(robot_config, safety_margin)
        self.fixed_margin = safety_margin  # 고정 마진

    def check_safety(self, pose, heading, velocity, zones) -> Tuple[bool, str]:
        robot_point = ShapelyPoint(pose)

        for zone in zones:
            dist = robot_point.distance(zone)
            self.state.min_distance_observed = min(self.state.min_distance_observed, dist)

            # SafetyChip: 고정 마진 내에 있으면 정지
            if dist < self.fixed_margin:
                if not self.state.stop_issued:
                    self.state.stop_issued = True
                    self.state.stop_time = time.time()
                    self.state.stop_reason = f"safetychip_margin (d={dist:.3f}m < margin={self.fixed_margin:.2f}m)"
                return False, self.state.stop_reason

        return True, ""


class SELPGuard(GuardBase):
    """
    SELP: LTL 제약 기반 디코딩 + 속도 비례 마진

    특징:
    - LTL 형식의 안전 제약 사용 (G(¬forbidden))
    - 속도에 비례하는 마진: margin = base + v * tau
    - 계획 시점에 제약 검증

    한계:
    - latency를 알려진 상수로 가정
    - 실시간 latency 변화에 적응 못함
    """

    def __init__(self, robot_config: RobotConfig, safety_margin: float = 0.3,
                 assumed_latency: float = 0.1):
        super().__init__(robot_config, safety_margin)
        self.assumed_latency = assumed_latency  # 가정된 고정 latency

    def compute_margin(self, velocity: float) -> float:
        """속도 비례 마진 계산"""
        # margin = base_margin + v * tau_assumed
        return self.safety_margin + velocity * self.assumed_latency

    def check_safety(self, pose, heading, velocity, zones) -> Tuple[bool, str]:
        robot_point = ShapelyPoint(pose)
        margin = self.compute_margin(velocity)

        for zone in zones:
            dist = robot_point.distance(zone)
            self.state.min_distance_observed = min(self.state.min_distance_observed, dist)

            # SELP: 속도 비례 마진 적용
            if dist < margin:
                if not self.state.stop_issued:
                    self.state.stop_issued = True
                    self.state.stop_time = time.time()
                    self.state.stop_reason = f"selp_constraint (d={dist:.3f}m < margin={margin:.3f}m)"
                return False, self.state.stop_reason

        return True, ""


class GeofenceAdaptiveGuard(GuardBase):
    """
    Geofence Adaptive: 전방 예측 + 제동거리 기반 마진 + 실시간 latency 적응

    특징:
    - 실시간 latency 측정 및 적응
    - 물리 기반 제동거리 계산: d_brake = v² / (2 * a_max)
    - 전방 경로 투영으로 교차 검사
    - 안전 조건: d > v * τ + v² / (2 * a_max) + m

    장점:
    - 네트워크 지연 변화에 적응
    - 물리적으로 정확한 제동거리 고려
    - 전방 예측으로 조기 감지
    """

    def __init__(self, robot_config: RobotConfig, safety_margin: float = 0.3,
                 estimated_latency: float = 0.1):
        super().__init__(robot_config, safety_margin)
        self.estimated_latency = estimated_latency
        self.latency_history = deque(maxlen=20)

    def update_latency_estimate(self, measured_latency: float):
        """측정된 latency로 추정값 업데이트"""
        self.latency_history.append(measured_latency)
        if len(self.latency_history) >= 3:
            # 90th percentile 사용 (보수적 추정)
            sorted_hist = sorted(self.latency_history)
            idx = int(len(sorted_hist) * 0.9)
            self.estimated_latency = sorted_hist[idx]

    def compute_braking_distance(self, velocity: float) -> float:
        """제동거리 계산: d = v² / (2 * a_max)"""
        return velocity ** 2 / (2 * self.robot_config.max_deceleration)

    def compute_reaction_distance(self, velocity: float) -> float:
        """반응 시간 동안 이동 거리: d = v * τ"""
        return velocity * self.estimated_latency

    def compute_required_distance(self, velocity: float) -> float:
        """안전 정지에 필요한 총 거리"""
        d_react = self.compute_reaction_distance(velocity)
        d_brake = self.compute_braking_distance(velocity)
        return d_react + d_brake + self.safety_margin

    def check_safety(self, pose, heading, velocity, zones) -> Tuple[bool, str]:
        robot_point = ShapelyPoint(pose)

        # 필요 정지거리 계산
        required_dist = self.compute_required_distance(velocity)
        self.state.projection_distance = required_dist

        # 전방 투영점
        proj_x = pose[0] + required_dist * math.cos(heading)
        proj_y = pose[1] + required_dist * math.sin(heading)

        # 투영 경로 생성
        from shapely.geometry import LineString
        projection_line = LineString([pose, (proj_x, proj_y)])

        for zone in zones:
            dist = robot_point.distance(zone)
            self.state.min_distance_observed = min(self.state.min_distance_observed, dist)

            # 투영 경로가 구역과 교차하는지 확인
            if projection_line.intersects(zone):
                if not self.state.stop_issued:
                    self.state.stop_issued = True
                    self.state.stop_time = time.time()
                    self.state.stop_reason = f"projection_intersects (d={dist:.3f}m, req={required_dist:.3f}m)"
                return False, self.state.stop_reason

            # 현재 거리가 필요 거리보다 작은지 확인
            if dist < required_dist:
                if not self.state.stop_issued:
                    self.state.stop_issued = True
                    self.state.stop_time = time.time()
                    self.state.stop_reason = f"insufficient_distance (d={dist:.3f}m < req={required_dist:.3f}m)"
                return False, self.state.stop_reason

        return True, ""


def create_guard(mode: GuardMode, robot_config: RobotConfig,
                 safety_margin: float, estimated_latency: float = 0.1) -> GuardBase:
    """Factory function to create guard instance"""
    if mode == GuardMode.NO_GUARD:
        return NoGuard(robot_config, safety_margin)
    elif mode == GuardMode.SAFETYCHIP:
        return SafetyChipGuard(robot_config, safety_margin=0.5)
    elif mode == GuardMode.SELP:
        return SELPGuard(robot_config, safety_margin, assumed_latency=estimated_latency)
    elif mode == GuardMode.GEOFENCE_ADAPTIVE:
        return GeofenceAdaptiveGuard(robot_config, safety_margin, estimated_latency)
    else:
        raise ValueError(f"Unknown guard mode: {mode}")


# =============================================================================
# Network Delay Simulator
# =============================================================================

class NetworkDelaySimulator:
    """
    Simulates network delay on zone updates.

    Can operate in two modes:
    1. Software simulation (queue-based delay)
    2. Real netem (requires sudo, Linux only)
    """

    def __init__(self, delay_ms: int, jitter_ms: int = 0, loss_percent: float = 0.0,
                 use_netem: bool = False, interface: str = "lo"):
        self.delay_ms = delay_ms
        self.jitter_ms = jitter_ms
        self.loss_percent = loss_percent
        self.use_netem = use_netem
        self.interface = interface

        # Software delay queue
        self.delay_queue: queue.Queue = queue.Queue()
        self.output_queue: queue.Queue = queue.Queue()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Start delay simulation"""
        if self.use_netem:
            self._setup_netem()
        else:
            self._running = True
            self._thread = threading.Thread(target=self._delay_worker, daemon=True)
            self._thread.start()

    def stop(self):
        """Stop delay simulation"""
        if self.use_netem:
            self._teardown_netem()
        else:
            self._running = False
            if self._thread:
                self._thread.join(timeout=1.0)

    def send(self, data: any, timestamp: float) -> float:
        """
        Send data through delay simulator.
        Returns: scheduled delivery time
        """
        # Calculate actual delay with jitter
        actual_delay_ms = self.delay_ms
        if self.jitter_ms > 0:
            actual_delay_ms += random.gauss(0, self.jitter_ms)
            actual_delay_ms = max(0, actual_delay_ms)

        # Simulate packet loss
        if self.loss_percent > 0 and random.random() < self.loss_percent / 100:
            return -1  # Lost packet

        delivery_time = timestamp + actual_delay_ms / 1000.0
        self.delay_queue.put((delivery_time, data, timestamp))
        return delivery_time

    def receive(self, current_time: float) -> List[Tuple[any, float, float]]:
        """
        Receive data that has completed delay.
        Returns: list of (data, publish_time, receive_time)
        """
        received = []
        while not self.output_queue.empty():
            try:
                item = self.output_queue.get_nowait()
                received.append(item)
            except queue.Empty:
                break
        return received

    def _delay_worker(self):
        """Background worker for software delay simulation"""
        pending = []
        while self._running:
            # Check for new items
            try:
                while True:
                    item = self.delay_queue.get_nowait()
                    pending.append(item)
            except queue.Empty:
                pass

            # Check for items ready to deliver
            current_time = time.time()
            still_pending = []
            for delivery_time, data, publish_time in pending:
                if current_time >= delivery_time:
                    self.output_queue.put((data, publish_time, current_time))
                else:
                    still_pending.append((delivery_time, data, publish_time))
            pending = still_pending

            time.sleep(0.001)  # 1ms resolution

    def _setup_netem(self):
        """Setup Linux netem for real network delay"""
        cmd = f"sudo tc qdisc add dev {self.interface} root netem delay {self.delay_ms}ms"
        if self.jitter_ms > 0:
            cmd += f" {self.jitter_ms}ms"
        if self.loss_percent > 0:
            cmd += f" loss {self.loss_percent}%"
        os.system(cmd)

    def _teardown_netem(self):
        """Remove netem rules"""
        os.system(f"sudo tc qdisc del dev {self.interface} root netem 2>/dev/null")


# =============================================================================
# Dynamic Zone Injector
# =============================================================================

@dataclass
class DynamicZoneEvent:
    """Event representing dynamic zone injection"""
    zone_id: str
    center: Tuple[float, float]
    shape: str  # "circle" or "rectangle"
    radius: float  # for circle
    width: float   # for rectangle
    height: float  # for rectangle
    publish_time: float
    receive_time: Optional[float] = None

    def get_polygon(self) -> Polygon:
        if self.shape == "circle":
            return ShapelyPoint(self.center).buffer(self.radius)
        else:
            half_w, half_h = self.width / 2, self.height / 2
            return box(self.center[0] - half_w, self.center[1] - half_h,
                      self.center[0] + half_w, self.center[1] + half_h)


class DynamicZoneInjector:
    """
    Injects dynamic forbidden zones ahead of the robot.

    At trigger time t_trigger:
        zone_center = robot_pose(t_trigger) + L * heading_unit_vector(t_trigger)
    """

    def __init__(self, config: ExperimentConfig, network_sim: NetworkDelaySimulator):
        self.config = config
        self.network_sim = network_sim
        self.pending_zones: List[DynamicZoneEvent] = []
        self.active_zones: List[DynamicZoneEvent] = []
        self._zone_counter = 0

    def inject_zone(self, robot_pose: Tuple[float, float], robot_heading: float,
                    distance_L: float, current_time: float) -> DynamicZoneEvent:
        """
        Inject a dynamic zone ahead of the robot.

        Args:
            robot_pose: Current (x, y) position
            robot_heading: Current heading in radians
            distance_L: Distance ahead to place zone center
            current_time: Current simulation time

        Returns:
            DynamicZoneEvent describing the injected zone
        """
        # Calculate zone center
        center_x = robot_pose[0] + distance_L * math.cos(robot_heading)
        center_y = robot_pose[1] + distance_L * math.sin(robot_heading)

        self._zone_counter += 1
        zone_id = f"dynamic_zone_{self._zone_counter}"

        event = DynamicZoneEvent(
            zone_id=zone_id,
            center=(center_x, center_y),
            shape=self.config.zone_shape,
            radius=self.config.zone_radius,
            width=self.config.zone_rect_size,
            height=self.config.zone_rect_size,
            publish_time=current_time,
        )

        # Send through network delay simulator
        delivery_time = self.network_sim.send(event, current_time)
        if delivery_time > 0:
            self.pending_zones.append(event)

        return event

    def update(self, current_time: float) -> List[DynamicZoneEvent]:
        """
        Update zone states, return newly received zones.
        """
        received = self.network_sim.receive(current_time)
        newly_active = []

        for data, publish_time, receive_time in received:
            if isinstance(data, DynamicZoneEvent):
                data.receive_time = receive_time
                self.active_zones.append(data)
                newly_active.append(data)
                # Remove from pending
                self.pending_zones = [z for z in self.pending_zones
                                     if z.zone_id != data.zone_id]

        return newly_active

    def get_active_zone_polygons(self) -> List[Polygon]:
        """Get list of active zone polygons"""
        return [z.get_polygon() for z in self.active_zones]

    def get_ground_truth_zones(self) -> List[Polygon]:
        """Get all zones including pending (ground truth for violation check)"""
        all_zones = self.active_zones + self.pending_zones
        return [z.get_polygon() for z in all_zones]

    def reset(self):
        self.pending_zones = []
        self.active_zones = []


# =============================================================================
# Episode Result and Logging
# =============================================================================

@dataclass
class EpisodeTimestamps:
    """Detailed timestamps for analysis"""
    t_motion_start: float = 0.0
    t_zone_published: float = 0.0
    t_zone_received: float = 0.0
    t_stop_issued: float = 0.0
    t_robot_stopped: float = 0.0

    @property
    def tau_network(self) -> float:
        """Measured network latency"""
        if self.t_zone_received > 0 and self.t_zone_published > 0:
            return self.t_zone_received - self.t_zone_published
        return 0.0

    @property
    def tau_total(self) -> float:
        """Total reaction time from zone publish to stop issued"""
        if self.t_stop_issued > 0 and self.t_zone_published > 0:
            return self.t_stop_issued - self.t_zone_published
        return 0.0


@dataclass
class EpisodeResult:
    """Complete episode result"""
    # Configuration
    velocity: float
    distance_L: float
    latency_ms: int
    guard_mode: str
    seed: int

    # Outcomes
    violation_flag: bool
    reached_goal: bool
    stopped_safely: bool

    # Metrics
    time_to_violation: float  # -1 if no violation
    min_distance_to_zone: float
    stop_reason: str

    # Timestamps
    timestamps: EpisodeTimestamps = field(default_factory=EpisodeTimestamps)

    # Computed
    tau_total_measured: float = 0.0
    theoretical_safe_distance: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d['timestamps'] = asdict(self.timestamps)
        return d


# =============================================================================
# Simulation Environment
# =============================================================================

class SimulationEnvironment:
    """
    Simulation environment for NDHU experiment.
    Can run in pure Python or with ROS2/Gazebo.
    """

    def __init__(self, world_config: WorldConfig, robot_config: RobotConfig,
                 use_ros2: bool = False):
        self.world_config = world_config
        self.robot_config = robot_config
        self.use_ros2 = use_ros2 and ROS2_AVAILABLE

        # Robot state
        self.pose: Tuple[float, float] = (2.0, 7.5)  # Start position
        self.heading: float = 0.0  # Facing +x
        self.velocity: float = 0.0
        self.angular_velocity: float = 0.0

        # Static zones
        self.static_zones: List[Polygon] = [world_config.get_static_zone()]

        # Simulation time
        self.sim_time: float = 0.0

    def reset(self, start_pose: Tuple[float, float] = (2.0, 7.5),
              start_heading: float = 0.0):
        """Reset environment to initial state"""
        self.pose = start_pose
        self.heading = start_heading
        self.velocity = 0.0
        self.angular_velocity = 0.0
        self.sim_time = 0.0

    def get_noisy_pose(self) -> Tuple[float, float]:
        """Get pose with noise (simulating localization uncertainty)"""
        noise_x = random.gauss(0, self.robot_config.pose_noise_std)
        noise_y = random.gauss(0, self.robot_config.pose_noise_std)
        return (self.pose[0] + noise_x, self.pose[1] + noise_y)

    def get_noisy_velocity(self) -> float:
        """Get velocity with noise"""
        noise = random.gauss(0, self.robot_config.velocity_noise_std)
        return max(0, self.velocity + noise)

    def step(self, cmd_vel: float, cmd_angular: float = 0.0) -> bool:
        """
        Execute one simulation step.

        Returns: True if step successful, False if collision
        """
        dt = self.robot_config.dt

        # Apply acceleration limits
        vel_diff = cmd_vel - self.velocity
        max_accel = self.robot_config.max_acceleration * dt
        max_decel = self.robot_config.max_deceleration * dt

        if vel_diff > 0:
            vel_diff = min(vel_diff, max_accel)
        else:
            vel_diff = max(vel_diff, -max_decel)

        self.velocity = max(0, self.velocity + vel_diff)
        self.angular_velocity = cmd_angular

        # Update pose
        self.heading += self.angular_velocity * dt
        self.pose = (
            self.pose[0] + self.velocity * math.cos(self.heading) * dt,
            self.pose[1] + self.velocity * math.sin(self.heading) * dt
        )

        self.sim_time += dt

        # Check bounds
        if (self.pose[0] < 0 or self.pose[0] > self.world_config.width or
            self.pose[1] < 0 or self.pose[1] > self.world_config.height):
            return False

        return True

    def check_collision(self, zones: List[Polygon]) -> bool:
        """Check if robot is inside any zone"""
        robot_point = ShapelyPoint(self.pose)
        for zone in zones:
            if zone.contains(robot_point):
                return True
        return False


# =============================================================================
# Episode Runner
# =============================================================================

class EpisodeRunner:
    """Runs a single episode of the NDHU experiment"""

    def __init__(self, world_config: WorldConfig, robot_config: RobotConfig,
                 experiment_config: ExperimentConfig):
        self.world_config = world_config
        self.robot_config = robot_config
        self.experiment_config = experiment_config

    def run_episode(self, velocity: float, distance_L: float, latency_ms: int,
                    guard_mode: GuardMode, seed: int) -> EpisodeResult:
        """
        Run a single episode.

        Args:
            velocity: Target robot velocity (m/s)
            distance_L: Distance to place dynamic zone (m)
            latency_ms: Network latency (ms)
            guard_mode: Defense mechanism to use
            seed: Random seed

        Returns:
            EpisodeResult with all metrics
        """
        # Set random seed
        random.seed(seed)
        np.random.seed(seed)

        # Initialize components
        env = SimulationEnvironment(self.world_config, self.robot_config)
        network_sim = NetworkDelaySimulator(latency_ms, jitter_ms=latency_ms // 10)
        zone_injector = DynamicZoneInjector(self.experiment_config, network_sim)
        guard = create_guard(guard_mode, self.robot_config,
                           self.experiment_config.safety_margin,
                           estimated_latency=latency_ms / 1000.0)

        # Start network simulation
        network_sim.start()

        # Initialize result
        timestamps = EpisodeTimestamps()
        result = EpisodeResult(
            velocity=velocity,
            distance_L=distance_L,
            latency_ms=latency_ms,
            guard_mode=guard_mode.value,
            seed=seed,
            violation_flag=False,
            reached_goal=False,
            stopped_safely=False,
            time_to_violation=-1,
            min_distance_to_zone=float('inf'),
            stop_reason="",
            timestamps=timestamps,
        )

        # Compute theoretical safe distance
        tau_sec = latency_ms / 1000.0
        result.theoretical_safe_distance = (
            velocity * tau_sec +
            velocity ** 2 / (2 * self.robot_config.max_deceleration) +
            self.experiment_config.safety_margin
        )

        # Reset environment
        env.reset()
        timestamps.t_motion_start = env.sim_time

        # State flags
        zone_injected = False
        injected_zone_event: Optional[DynamicZoneEvent] = None
        motion_started = False
        robot_stopped = False
        stop_issued_time = None

        # Goal position (straight ahead)
        goal_x = self.world_config.width - 2.0

        # Main simulation loop
        max_steps = int(30.0 / self.robot_config.dt)  # 30 second max

        for step in range(max_steps):
            current_time = env.sim_time

            # Update zone injector (receive delayed zones)
            newly_received = zone_injector.update(current_time)
            for zone_event in newly_received:
                timestamps.t_zone_received = current_time
                # Update guard's latency estimate
                if isinstance(guard, GeofenceAdaptiveGuard):
                    measured_latency = zone_event.receive_time - zone_event.publish_time
                    guard.update_latency_estimate(measured_latency)

            # Get sensor readings (noisy)
            noisy_pose = env.get_noisy_pose()
            noisy_velocity = env.get_noisy_velocity()

            # Inject zone after trigger delay
            if (not zone_injected and
                current_time >= self.experiment_config.trigger_delay and
                motion_started):

                injected_zone_event = zone_injector.inject_zone(
                    env.pose, env.heading, distance_L, current_time
                )
                timestamps.t_zone_published = current_time
                zone_injected = True

            # Get known zones (static + received dynamic)
            known_zones = self.world_config.get_static_zone()
            all_known_zones = [known_zones] + zone_injector.get_active_zone_polygons()

            # Check guard
            is_safe, stop_reason = guard.check_safety(
                noisy_pose, env.heading, noisy_velocity, all_known_zones
            )

            # Determine velocity command
            if not is_safe and not robot_stopped:
                cmd_vel = 0.0  # Stop command
                if stop_issued_time is None:
                    stop_issued_time = current_time
                    timestamps.t_stop_issued = current_time
                    result.stop_reason = stop_reason
            elif robot_stopped:
                cmd_vel = 0.0
            else:
                cmd_vel = velocity
                motion_started = True

            # Execute step
            env.step(cmd_vel)

            # Check if robot has stopped
            if env.velocity < 0.01 and stop_issued_time is not None:
                if not robot_stopped:
                    robot_stopped = True
                    timestamps.t_robot_stopped = current_time

            # Check violation (ground truth)
            ground_truth_zones = zone_injector.get_ground_truth_zones()
            all_ground_truth = [self.world_config.get_static_zone()] + ground_truth_zones

            if env.check_collision(all_ground_truth):
                result.violation_flag = True
                result.time_to_violation = current_time - timestamps.t_zone_published
                break

            # Update min distance
            robot_point = ShapelyPoint(env.pose)
            for zone in all_ground_truth:
                dist = robot_point.distance(zone)
                result.min_distance_to_zone = min(result.min_distance_to_zone, dist)

            # Check goal reached
            if env.pose[0] >= goal_x:
                result.reached_goal = True
                break

            # Check if stopped safely (velocity near zero, no violation)
            if robot_stopped and env.velocity < 0.01:
                # Wait a bit to confirm stop
                if current_time > timestamps.t_robot_stopped + 0.5:
                    result.stopped_safely = True
                    break

        # Cleanup
        network_sim.stop()

        # Final metrics
        result.tau_total_measured = timestamps.tau_total
        result.min_distance_to_zone = guard.state.min_distance_observed

        if not result.stop_reason and guard.state.stop_reason:
            result.stop_reason = guard.state.stop_reason

        return result


# =============================================================================
# Experiment Orchestrator
# =============================================================================

class ExperimentOrchestrator:
    """Orchestrates full parameter sweep experiment"""

    def __init__(self, world_config: WorldConfig = None,
                 robot_config: RobotConfig = None,
                 experiment_config: ExperimentConfig = None):
        self.world_config = world_config or WorldConfig()
        self.robot_config = robot_config or RobotConfig()
        self.experiment_config = experiment_config or ExperimentConfig()
        self.runner = EpisodeRunner(self.world_config, self.robot_config,
                                   self.experiment_config)
        self.results: List[EpisodeResult] = []

    def run_single(self, v: float, L: float, tau_ms: int,
                   guard_mode: str, seed: int) -> EpisodeResult:
        """Run single episode with specified parameters"""
        mode = GuardMode(guard_mode)
        return self.runner.run_episode(v, L, tau_ms, mode, seed)

    def run_sweep(self, guard_modes: List[str] = None,
                  progress_callback: Callable = None) -> List[EpisodeResult]:
        """
        Run full parameter sweep.

        Args:
            guard_modes: List of guard modes to test (default: all)
            progress_callback: Function called with (current, total) for progress

        Returns:
            List of all episode results
        """
        if guard_modes is None:
            guard_modes = [m.value for m in GuardMode]

        config = self.experiment_config
        total = (len(config.velocities) * len(config.distances) *
                len(config.latencies_ms) * len(guard_modes) * config.num_seeds)

        self.results = []
        count = 0

        print(f"\n{'='*70}")
        print("S7: NETWORK-DELAYED HAZARD UPDATE (NDHU) EXPERIMENT")
        print(f"{'='*70}")
        print(f"Velocities: {config.velocities} m/s")
        print(f"Distances L: {config.distances} m")
        print(f"Latencies: {config.latencies_ms} ms")
        print(f"Guard modes: {guard_modes}")
        print(f"Seeds per condition: {config.num_seeds}")
        print(f"Total episodes: {total}")
        print(f"{'='*70}\n")

        for v in config.velocities:
            for L in config.distances:
                for tau_ms in config.latencies_ms:
                    for mode in guard_modes:
                        for seed in range(config.num_seeds):
                            result = self.run_single(v, L, tau_ms, mode, seed)
                            self.results.append(result)
                            count += 1

                            if progress_callback:
                                progress_callback(count, total)
                            elif count % 100 == 0:
                                print(f"  Progress: {count}/{total} ({100*count/total:.1f}%)")

        return self.results

    def analyze_results(self) -> Dict:
        """Analyze results and compute summary statistics"""
        if not self.results:
            return {}

        summary = {
            "by_guard_mode": {},
            "by_velocity": {},
            "by_distance": {},
            "by_latency": {},
            "transition_curves": {},
            "metadata": {
                "total_episodes": len(self.results),
                "timestamp": datetime.now().isoformat(),
            }
        }

        # Group by guard mode
        for mode in set(r.guard_mode for r in self.results):
            subset = [r for r in self.results if r.guard_mode == mode]
            violations = sum(1 for r in subset if r.violation_flag)
            summary["by_guard_mode"][mode] = {
                "total": len(subset),
                "violations": violations,
                "violation_rate": 100 * violations / len(subset),
                "avg_min_distance": np.mean([r.min_distance_to_zone for r in subset]),
            }

        # Transition curves: violation rate vs L for each (guard_mode, latency)
        for mode in set(r.guard_mode for r in self.results):
            summary["transition_curves"][mode] = {}
            for tau in self.experiment_config.latencies_ms:
                curve = {}
                for L in self.experiment_config.distances:
                    subset = [r for r in self.results
                             if r.guard_mode == mode and r.latency_ms == tau
                             and r.distance_L == L]
                    if subset:
                        violations = sum(1 for r in subset if r.violation_flag)
                        curve[L] = 100 * violations / len(subset)
                summary["transition_curves"][mode][tau] = curve

        return summary

    def save_results(self, filepath: str):
        """Save results to JSON file"""
        output = {
            "summary": self.analyze_results(),
            "episodes": [r.to_dict() for r in self.results],
            "config": {
                "world": asdict(self.world_config),
                "robot": asdict(self.robot_config),
                "experiment": asdict(self.experiment_config),
            }
        }

        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {filepath}")

    def print_summary(self):
        """Print summary to console"""
        summary = self.analyze_results()

        print(f"\n{'='*70}")
        print("RESULTS SUMMARY")
        print(f"{'='*70}")

        print("\nViolation Rate by Guard Mode:")
        print("-" * 50)
        for mode, stats in summary["by_guard_mode"].items():
            rate = stats["violation_rate"]
            color = "\033[92m" if rate == 0 else "\033[93m" if rate < 30 else "\033[91m"
            print(f"  {mode}: {color}{rate:5.1f}%\033[0m "
                  f"({stats['violations']}/{stats['total']} violations)")

        print("\nTransition Curves (Violation Rate vs Distance L):")
        print("-" * 70)

        for mode in summary["transition_curves"]:
            print(f"\n  {mode}:")
            header = "    L (m): " + " ".join(f"{L:>5}" for L in self.experiment_config.distances)
            print(header)

            for tau in self.experiment_config.latencies_ms:
                curve = summary["transition_curves"][mode].get(tau, {})
                values = [curve.get(L, 0) for L in self.experiment_config.distances]
                row = f"  {tau:>3}ms:  " + " ".join(f"{v:5.1f}" for v in values)
                print(row)


# =============================================================================
# Visualization
# =============================================================================

def plot_transition_curves(results_file: str, output_dir: str = None):
    """Plot violation rate vs distance L (transition curves)"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for plotting")
        return

    with open(results_file) as f:
        data = json.load(f)

    if output_dir is None:
        output_dir = os.path.dirname(results_file)

    summary = data["summary"]
    config = data["config"]["experiment"]
    distances = config["distances"]
    latencies = config["latencies_ms"]

    modes_order = ["no_guard", "safetychip", "selp", "geofence_adaptive"]
    colors = {
        "no_guard": "#d62728",         # red
        "safetychip": "#1f77b4",       # blue
        "selp": "#ff7f0e",             # orange
        "geofence_adaptive": "#2ca02c", # green
    }
    labels = {
        "no_guard": "No Guard",
        "safetychip": "SafetyChip",
        "selp": "SELP",
        "geofence_adaptive": "Geofence\nAdaptive",
    }

    # Plot 1: Transition curves for each guard mode
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, mode in enumerate(modes_order):
        ax = axes[idx]
        curves = summary["transition_curves"].get(mode, {})

        for tau in latencies:
            curve = curves.get(str(tau), {})
            if curve:
                L_vals = [float(k) for k in curve.keys()]
                vr_vals = [curve[str(int(L))] for L in sorted(L_vals)]
                L_sorted = sorted(L_vals)

                label = f"τ={tau}ms"
                alpha = 0.4 + 0.6 * (latencies.index(tau) / len(latencies))
                ax.plot(L_sorted, vr_vals, 'o-', label=label,
                       color=colors[mode], alpha=alpha, linewidth=2)

        ax.set_xlabel("Distance L (m)", fontsize=11)
        ax.set_ylabel("Violation Rate (%)", fontsize=11)
        ax.set_title(f"{labels[mode].replace(chr(10), ' ')}", fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-5, 105)
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 's7_ndhu_transition_curves.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

    # Plot 2: Comparison at specific latency
    fig2, ax = plt.subplots(figsize=(12, 6))

    target_latency = 300  # Focus on 300ms
    for mode in modes_order:
        curves = summary["transition_curves"].get(mode, {})
        curve = curves.get(str(target_latency), {})
        if curve:
            L_vals = sorted([float(k) for k in curve.keys()])
            vr_vals = [curve[str(int(L))] for L in L_vals]

            ax.plot(L_vals, vr_vals, 'o-', label=labels[mode].replace('\n', ' '),
                   color=colors[mode], linewidth=2, markersize=8)

    ax.set_xlabel("Distance L (m)", fontsize=12)
    ax.set_ylabel("Violation Rate (%)", fontsize=12)
    ax.set_title(f"S7 NDHU: Guard Comparison at τ={target_latency}ms",
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105)

    # Add theoretical transition point annotation
    # L_critical = v * tau + v^2/(2*a) + m
    v_typical = 1.5
    a_max = 3.0
    m = 0.3
    L_critical = v_typical * (target_latency/1000) + v_typical**2/(2*a_max) + m
    ax.axvline(x=L_critical, color='purple', linestyle=':', alpha=0.7, linewidth=2)
    ax.annotate(f'L_critical ≈ {L_critical:.1f}m\n(v={v_typical}, τ={target_latency}ms)',
               xy=(L_critical, 50), xytext=(L_critical + 1.5, 60),
               fontsize=10, color='purple',
               arrowprops=dict(arrowstyle='->', color='purple', alpha=0.7))

    plt.tight_layout()
    output_path2 = os.path.join(output_dir, 's7_ndhu_guard_comparison.png')
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path2}")

    # Plot 3: Heatmap of violation rates
    fig3, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, mode in enumerate(modes_order):
        ax = axes[idx]
        curves = summary["transition_curves"].get(mode, {})

        # Build matrix
        matrix = []
        for tau in latencies:
            curve = curves.get(str(tau), {})
            row = [curve.get(str(int(L)), 0) for L in distances]
            matrix.append(row)

        matrix = np.array(matrix)

        im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=100)

        ax.set_xticks(np.arange(len(distances)))
        ax.set_yticks(np.arange(len(latencies)))
        ax.set_xticklabels([str(d) for d in distances])
        ax.set_yticklabels([f"{tau}ms" for tau in latencies])

        # Add text
        for i in range(len(latencies)):
            for j in range(len(distances)):
                val = matrix[i, j]
                color = 'white' if val > 50 else 'black'
                ax.text(j, i, f"{val:.0f}", ha='center', va='center',
                       color=color, fontsize=9, fontweight='bold')

        ax.set_xlabel("Distance L (m)", fontsize=11)
        ax.set_ylabel("Latency τ", fontsize=11)
        ax.set_title(labels[mode].replace('\n', ' '), fontsize=12, fontweight='bold')

        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle("S7 NDHU: Violation Rate Heatmaps (Lower = Better)", fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path3 = os.path.join(output_dir, 's7_ndhu_heatmaps.png')
    plt.savefig(output_path3, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path3}")


# =============================================================================
# CLI Interface
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="S7: Network-Delayed Hazard Update (NDHU) Attack Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single run
  python3 s7_network_delayed_zone_attack.py --v 1.5 --L 3 --tau_net_ms 300 --seed 42 --guard_mode geofence_adaptive

  # Full sweep
  python3 s7_network_delayed_zone_attack.py --sweep --output results_s7_ndhu.json

  # Plot existing results
  python3 s7_network_delayed_zone_attack.py --plot results_s7_ndhu.json
        """
    )

    # Single run parameters
    parser.add_argument('--v', type=float, help='Robot velocity (m/s)')
    parser.add_argument('--L', type=float, help='Distance to zone (m)')
    parser.add_argument('--tau_net_ms', type=int, help='Network latency (ms)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--guard_mode', type=str,
                       choices=['no_guard', 'safetychip', 'selp', 'geofence_adaptive'],
                       default='geofence_adaptive', help='Guard mode')

    # Sweep parameters
    parser.add_argument('--sweep', action='store_true', help='Run full parameter sweep')
    parser.add_argument('--num_seeds', type=int, default=30,
                       help='Number of seeds per condition')
    parser.add_argument('--output', type=str, default='s7_ndhu_results.json',
                       help='Output file for results')

    # Analysis
    parser.add_argument('--plot', type=str, metavar='RESULTS_FILE',
                       help='Plot results from JSON file')

    # Experiment config overrides
    parser.add_argument('--velocities', type=float, nargs='+',
                       help='Override velocity values')
    parser.add_argument('--distances', type=float, nargs='+',
                       help='Override distance values')
    parser.add_argument('--latencies', type=int, nargs='+',
                       help='Override latency values')

    return parser.parse_args()


def main():
    args = parse_args()

    # Plot mode
    if args.plot:
        plot_transition_curves(args.plot)
        return

    # Create configs
    experiment_config = ExperimentConfig(num_seeds=args.num_seeds)

    if args.velocities:
        experiment_config.velocities = args.velocities
    if args.distances:
        experiment_config.distances = args.distances
    if args.latencies:
        experiment_config.latencies_ms = args.latencies

    orchestrator = ExperimentOrchestrator(experiment_config=experiment_config)

    if args.sweep:
        # Full sweep
        orchestrator.run_sweep()
        orchestrator.print_summary()

        output_path = os.path.join(os.path.dirname(__file__), '..', args.output)
        orchestrator.save_results(output_path)

        # Generate plots
        plot_transition_curves(output_path,
                              os.path.join(os.path.dirname(__file__), '..', 'figures'))

    elif args.v is not None and args.L is not None and args.tau_net_ms is not None:
        # Single run
        result = orchestrator.run_single(
            args.v, args.L, args.tau_net_ms, args.guard_mode, args.seed
        )

        print(f"\n{'='*60}")
        print("SINGLE EPISODE RESULT")
        print(f"{'='*60}")
        print(f"Configuration:")
        print(f"  Velocity: {args.v} m/s")
        print(f"  Distance L: {args.L} m")
        print(f"  Latency: {args.tau_net_ms} ms")
        print(f"  Guard mode: {args.guard_mode}")
        print(f"  Seed: {args.seed}")
        print()
        print(f"Results:")
        print(f"  Violation: {'YES' if result.violation_flag else 'NO'}")
        print(f"  Min distance to zone: {result.min_distance_to_zone:.3f} m")
        print(f"  Stop reason: {result.stop_reason or 'N/A'}")
        print(f"  Theoretical safe distance: {result.theoretical_safe_distance:.3f} m")
        print()
        print(f"Timestamps:")
        print(f"  τ_network (measured): {result.timestamps.tau_network*1000:.1f} ms")
        print(f"  τ_total (measured): {result.timestamps.tau_total*1000:.1f} ms")
        print(f"{'='*60}")

    else:
        print("Error: Specify either --sweep or (--v, --L, --tau_net_ms)")
        print("Use --help for usage information")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
