#!/usr/bin/env python3
"""
Gazebo Integration Test - 실제 Nav2 환경에서 Geofence Ablation Study

이 테스트는 Gazebo + Nav2 + AMCL 환경에서 실제로
안전 마진의 효과를 검증합니다.

실험 방법:
1. TurtleBot3를 금지구역 근처로 navigation goal 전송
2. 실제 로봇 궤적이 금지구역을 침범하는지 모니터링
3. 마진 설정별로 반복하여 비교

토픽:
- /amcl_pose: AMCL 위치 추정 (공분산 포함)
- /odom: Odometry (ground truth에 가까움)
- /goal_pose: Navigation goal
- /cmd_vel: 로봇 속도 명령
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Odometry
from nav2_msgs.action import NavigateToPose
from std_msgs.msg import String

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum
import time
import json
from pathlib import Path

import sys
sys.path.insert(0, '/home/jim/ros2_motion_planning_tutorials/src/geofence_enforcer')

from geofence_enforcer.geofence_geometry import GeofenceGeometry
from geofence_enforcer.margin_calculator import MarginCalculator


class TestPhase(Enum):
    IDLE = 0
    NAVIGATING = 1
    COMPLETED = 2
    FAILED = 3


@dataclass
class NavigationTrial:
    """단일 네비게이션 시도 결과"""
    trial_id: int
    margin_config: str
    margin_value: float
    goal_x: float
    goal_y: float
    start_x: float
    start_y: float
    trajectory: List[Tuple[float, float]] = field(default_factory=list)
    covariances: List[float] = field(default_factory=list)  # max eigenvalue
    violated: bool = False
    min_distance_to_boundary: float = float('inf')
    navigation_success: bool = False
    duration: float = 0.0


@dataclass
class AblationConfig:
    """Ablation 설정"""
    name: str
    use_estimation_term: bool
    use_tracking_error: bool
    use_latency_term: bool
    e_track: float = 0.12
    v_max: float = 1.5
    tau: float = 0.20


class GazeboIntegrationTestNode(Node):
    """Gazebo 통합 테스트 노드"""

    def __init__(self):
        super().__init__('gazebo_integration_test')

        # 파라미터
        self.declare_parameter('geofence_config', '')
        self.declare_parameter('output_dir', '/tmp/geofence_test_results')
        self.declare_parameter('trials_per_config', 10)
        self.declare_parameter('alpha', 0.99)

        self._geofence_config = self.get_parameter('geofence_config').value
        self._output_dir = Path(self.get_parameter('output_dir').value)
        self._trials_per_config = self.get_parameter('trials_per_config').value
        self._alpha = self.get_parameter('alpha').value

        # 출력 디렉토리 생성
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Geofence 설정
        self._setup_geofence()

        # Ablation 설정들
        self._ablation_configs = [
            AblationConfig("Full", True, True, True),
            AblationConfig("No_e_track", True, False, True),
            AblationConfig("No_v_tau", True, True, False),
            AblationConfig("No_both", True, False, False),
            AblationConfig("No_margin", False, False, False),
        ]

        # 상태
        self._current_trial: Optional[NavigationTrial] = None
        self._all_trials: List[NavigationTrial] = []
        self._phase = TestPhase.IDLE
        self._current_config_idx = 0
        self._current_trial_idx = 0

        # 현재 로봇 상태
        self._current_pose: Optional[Tuple[float, float]] = None
        self._current_covariance: Optional[np.ndarray] = None
        self._current_velocity: float = 0.0

        # QoS 설정
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )

        # Subscribers
        self._amcl_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self._amcl_callback,
            qos
        )

        self._odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self._odom_callback,
            qos
        )

        # Nav2 Action Client
        self._nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Publishers
        self._status_pub = self.create_publisher(String, '/geofence_test/status', 10)

        # 타이머
        self._monitor_timer = self.create_timer(0.1, self._monitor_callback)  # 10Hz 모니터링
        self._test_timer = None  # 테스트 시작 후 활성화
        self._delay_timer = None  # 시도 간 지연용

        self.get_logger().info("=" * 60)
        self.get_logger().info("Gazebo Integration Test Node Started")
        self.get_logger().info(f"  Trials per config: {self._trials_per_config}")
        self.get_logger().info(f"  Configs to test: {len(self._ablation_configs)}")
        self.get_logger().info(f"  Output: {self._output_dir}")
        self.get_logger().info("=" * 60)
        self.get_logger().info("Waiting for Nav2 action server...")

        # Nav2 준비 대기
        self._wait_for_nav2()

    def _setup_geofence(self):
        """Geofence 설정"""
        if self._geofence_config:
            self._geofence = GeofenceGeometry.from_yaml(self._geofence_config)
        else:
            # 기본 금지구역 설정 (TurtleBot3 world 기준)
            self._geofence = GeofenceGeometry()
            # 예시 금지구역 (실제 환경에 맞게 조정 필요)
            self._geofence.add_zone("forbidden_zone_1", [
                (1.0, 1.0), (2.0, 1.0), (2.0, 2.0), (1.0, 2.0)
            ])

        self.get_logger().info(f"Geofence zones: {self._geofence.get_zone_names()}")

    def _wait_for_nav2(self):
        """Nav2 Action Server 대기"""
        if not self._nav_client.wait_for_server(timeout_sec=30.0):
            self.get_logger().error("Nav2 action server not available!")
            return

        self.get_logger().info("Nav2 action server ready!")
        self.get_logger().info("Starting test in 3 seconds...")

        # 3초 후 테스트 시작
        self._test_timer = self.create_timer(3.0, self._start_tests)

    def _start_tests(self):
        """테스트 시작"""
        if self._test_timer:
            self._test_timer.cancel()
            self._test_timer = None

        self.get_logger().info("=" * 60)
        self.get_logger().info("STARTING ABLATION STUDY")
        self.get_logger().info("=" * 60)

        self._run_next_trial()

    def _compute_margin(self, config: AblationConfig) -> float:
        """설정에 따른 마진 계산

        공분산이 비정상적으로 큰 경우 (AMCL 미수렴) 최대값 제한 적용
        - 최대 σ² = 0.05 (σ = 22cm, 일반적인 AMCL 수렴 후 값)
        """
        margin = 0.0

        if config.use_estimation_term and self._current_covariance is not None:
            chi2_99 = 9.21
            cov_2x2 = self._current_covariance[:2, :2]
            try:
                lambda_max = np.max(np.linalg.eigvalsh(cov_2x2))
                # AMCL이 수렴하지 않은 경우 너무 큰 공분산 제한
                # 일반적으로 수렴 후 σ ≈ 5-20cm (σ² = 0.0025 ~ 0.04)
                MAX_COVARIANCE = 0.05  # σ² max = 0.05 (σ ≈ 22cm)
                lambda_max = min(lambda_max, MAX_COVARIANCE)
                lambda_max = max(lambda_max, 0.001)  # 최소값
                margin += np.sqrt(chi2_99 * lambda_max)
                self.get_logger().debug(f"Estimation margin: {np.sqrt(chi2_99 * lambda_max)*100:.1f}cm (λ={lambda_max:.4f})")
            except:
                margin += 0.15  # 기본값

        if config.use_tracking_error:
            margin += config.e_track

        if config.use_latency_term:
            margin += config.v_max * config.tau

        return margin

    def _generate_goal_near_boundary(self) -> Tuple[float, float]:
        """금지구역 경계 근처에 목표점 생성

        목표점을 경계에서 10~50cm 거리에 배치
        사각형 존의 각 변 중앙 근처에 목표점 생성
        """
        # 금지구역 정보 가져오기
        zones = self._geofence.get_zones_by_type('forbidden')
        if not zones:
            zones = {name: self._geofence._zones[name]
                    for name in self._geofence.get_zone_names()}

        if not zones:
            # 기본 목표점
            return (2.5, 0.5)

        # 무작위 zone 선택
        zone = list(zones.values())[np.random.randint(len(zones))]
        vertices = zone.vertices

        # 사각형의 경계(변)를 따라 목표점 생성
        # 각 변의 중점 계산
        n = len(vertices)
        edge_midpoints = []
        edge_normals = []

        for i in range(n):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % n]
            # 변의 중점
            mx = (v1[0] + v2[0]) / 2
            my = (v1[1] + v2[1]) / 2
            edge_midpoints.append((mx, my))

            # 변의 바깥쪽 법선 벡터
            # 반시계방향 polygon의 경우 (dy, -dx)가 바깥쪽
            # 시계방향 polygon의 경우 (-dy, dx)가 바깥쪽
            # Shapely는 보통 반시계방향, 직접 정의는 보통 시계방향
            dx = v2[0] - v1[0]
            dy = v2[1] - v1[1]
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                # 90도 시계방향 회전 (dy, -dx) = 바깥쪽 for CCW polygon
                nx = dy / length
                ny = -dx / length
                edge_normals.append((nx, ny))
            else:
                edge_normals.append((1.0, 0.0))

        # 무작위 변 선택
        edge_idx = np.random.randint(n)
        mx, my = edge_midpoints[edge_idx]
        nx, ny = edge_normals[edge_idx]

        # 경계에서 50~120cm 바깥으로
        # 이렇게 하면:
        #   - Full (110cm): 50-110cm 목표 차단, 110-120cm 통과
        #   - No_e_track (98cm): 50-98cm 차단, 98-120cm 통과
        #   - No_v_tau (80cm): 50-80cm 차단, 80-120cm 통과
        #   - No_both (68cm): 50-68cm 차단, 68-120cm 통과
        #   - No_margin (0cm): 전부 통과
        offset = np.random.uniform(0.50, 1.20)

        # 중점을 따라 약간 랜덤하게 이동 (변의 30% 범위 내)
        v1 = vertices[edge_idx]
        v2 = vertices[(edge_idx + 1) % n]
        edge_len = np.sqrt((v2[0]-v1[0])**2 + (v2[1]-v1[1])**2)
        along_edge = np.random.uniform(-0.3, 0.3) * edge_len
        dx_edge = (v2[0] - v1[0]) / edge_len if edge_len > 0 else 0
        dy_edge = (v2[1] - v1[1]) / edge_len if edge_len > 0 else 0

        goal_x = mx + along_edge * dx_edge + offset * nx
        goal_y = my + along_edge * dy_edge + offset * ny

        return (goal_x, goal_y)

    def _run_next_trial(self):
        """다음 시도 실행"""
        if self._current_config_idx >= len(self._ablation_configs):
            self._finish_tests()
            return

        config = self._ablation_configs[self._current_config_idx]

        if self._current_trial_idx >= self._trials_per_config:
            # 다음 설정으로
            self._current_config_idx += 1
            self._current_trial_idx = 0
            self._run_next_trial()
            return

        # 마진 계산
        margin = self._compute_margin(config)

        # 목표점 생성
        goal_x, goal_y = self._generate_goal_near_boundary()

        # 시작 위치
        start_x, start_y = self._current_pose if self._current_pose else (0.0, 0.0)

        # Trial 생성
        trial_id = self._current_config_idx * self._trials_per_config + self._current_trial_idx
        self._current_trial = NavigationTrial(
            trial_id=trial_id,
            margin_config=config.name,
            margin_value=margin,
            goal_x=goal_x,
            goal_y=goal_y,
            start_x=start_x,
            start_y=start_y
        )

        self.get_logger().info(f"\n[Trial {trial_id}] Config: {config.name}, "
                              f"Margin: {margin*100:.1f}cm, "
                              f"Goal: ({goal_x:.2f}, {goal_y:.2f})")

        # 목표점이 팽창된 금지구역 안인지 체크
        check = self._geofence.check_point(goal_x, goal_y, margin=margin)
        if check.is_violated:
            self.get_logger().info(f"  -> Goal BLOCKED by margin (inside inflated zone)")
            self._current_trial.navigation_success = False
            self._current_trial.violated = False  # 차단되어서 침범 아님
            self._trial_completed()
            return

        # Navigation 시작
        self._send_navigation_goal(goal_x, goal_y)

    def _send_navigation_goal(self, x: float, y: float):
        """Navigation goal 전송"""
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.orientation.w = 1.0

        self._phase = TestPhase.NAVIGATING
        self._nav_start_time = time.time()

        self._nav_future = self._nav_client.send_goal_async(
            goal_msg,
            feedback_callback=self._nav_feedback_callback
        )
        self._nav_future.add_done_callback(self._nav_goal_response_callback)

    def _nav_goal_response_callback(self, future):
        """Goal 응답 콜백"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("Goal rejected!")
            self._phase = TestPhase.FAILED
            self._trial_completed()
            return

        self._goal_handle = goal_handle
        self._result_future = goal_handle.get_result_async()
        self._result_future.add_done_callback(self._nav_result_callback)

    def _nav_feedback_callback(self, feedback_msg):
        """Navigation feedback"""
        pass  # 모니터링은 _monitor_callback에서

    def _nav_result_callback(self, future):
        """Navigation 완료"""
        result = future.result()
        self._phase = TestPhase.COMPLETED

        if self._current_trial:
            self._current_trial.navigation_success = True
            self._current_trial.duration = time.time() - self._nav_start_time

        self.get_logger().info(f"  -> Navigation completed in {self._current_trial.duration:.1f}s")
        self._trial_completed()

    def _trial_completed(self):
        """Trial 완료 처리"""
        if self._current_trial:
            self._all_trials.append(self._current_trial)

            status = "VIOLATED" if self._current_trial.violated else "SAFE"
            self.get_logger().info(f"  -> Result: {status}, "
                                  f"Min distance: {self._current_trial.min_distance_to_boundary:.3f}m")

        self._current_trial_idx += 1
        self._phase = TestPhase.IDLE

        # 잠시 대기 후 다음 시도
        self._delay_timer = self.create_timer(1.0, self._delayed_next_trial)

    def _delayed_next_trial(self):
        """지연 후 다음 시도"""
        if self._delay_timer:
            self._delay_timer.cancel()
            self._delay_timer = None
        self._run_next_trial()

    def _amcl_callback(self, msg: PoseWithCovarianceStamped):
        """AMCL pose 콜백"""
        self._current_pose = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y
        )

        # 공분산 추출 (6x6 -> 2x2)
        cov_flat = msg.pose.covariance
        cov_6x6 = np.array(cov_flat).reshape(6, 6)
        self._current_covariance = cov_6x6

    def _odom_callback(self, msg: Odometry):
        """Odometry 콜백 (ground truth에 가까움)"""
        # 속도 계산
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self._current_velocity = np.sqrt(vx**2 + vy**2)

    def _monitor_callback(self):
        """주기적 모니터링"""
        if self._phase != TestPhase.NAVIGATING:
            return

        if self._current_trial is None or self._current_pose is None:
            return

        x, y = self._current_pose

        # 궤적 기록
        self._current_trial.trajectory.append((x, y))

        # 공분산 기록
        if self._current_covariance is not None:
            try:
                lambda_max = np.max(np.linalg.eigvalsh(self._current_covariance[:2, :2]))
                self._current_trial.covariances.append(lambda_max)
            except:
                pass

        # 금지구역 침범 검사 (마진 없이 - 실제 침범)
        check = self._geofence.check_point(x, y, margin=0)

        if check.is_violated:
            self._current_trial.violated = True
            self.get_logger().warn(f"  !! VIOLATION at ({x:.3f}, {y:.3f})")

        # 경계까지 거리 기록
        if check.distance_to_boundary < self._current_trial.min_distance_to_boundary:
            self._current_trial.min_distance_to_boundary = check.distance_to_boundary

    def _finish_tests(self):
        """모든 테스트 완료"""
        self.get_logger().info("\n" + "=" * 60)
        self.get_logger().info("ALL TESTS COMPLETED")
        self.get_logger().info("=" * 60)

        # 결과 분석
        self._analyze_results()

        # 결과 저장
        self._save_results()

    def _analyze_results(self):
        """결과 분석"""
        results_by_config = {}

        for trial in self._all_trials:
            config = trial.margin_config
            if config not in results_by_config:
                results_by_config[config] = {
                    'trials': 0,
                    'violations': 0,
                    'blocked': 0,
                    'margins': []
                }

            results_by_config[config]['trials'] += 1
            results_by_config[config]['margins'].append(trial.margin_value)

            if trial.violated:
                results_by_config[config]['violations'] += 1
            if not trial.navigation_success and not trial.violated:
                results_by_config[config]['blocked'] += 1

        self.get_logger().info("\nRESULTS SUMMARY:")
        self.get_logger().info("-" * 60)
        self.get_logger().info(f"{'Config':<20} {'Margin(cm)':<12} {'Trials':<8} {'Blocked':<8} {'Violations':<10} {'Rate':<8}")
        self.get_logger().info("-" * 60)

        for config, data in results_by_config.items():
            avg_margin = np.mean(data['margins']) * 100 if data['margins'] else 0
            rate = data['violations'] / data['trials'] * 100 if data['trials'] > 0 else 0

            self.get_logger().info(
                f"{config:<20} {avg_margin:<12.1f} {data['trials']:<8} "
                f"{data['blocked']:<8} {data['violations']:<10} {rate:<8.1f}%"
            )

        self.get_logger().info("-" * 60)

    def _save_results(self):
        """결과 저장"""
        # JSON으로 저장
        results_data = []
        for trial in self._all_trials:
            results_data.append({
                'trial_id': trial.trial_id,
                'config': trial.margin_config,
                'margin_cm': trial.margin_value * 100,
                'goal': [trial.goal_x, trial.goal_y],
                'violated': trial.violated,
                'min_distance': trial.min_distance_to_boundary,
                'navigation_success': trial.navigation_success,
                'duration': trial.duration,
                'trajectory_length': len(trial.trajectory)
            })

        output_file = self._output_dir / 'gazebo_ablation_results.json'
        with open(output_file, 'w') as f:
            json.dump(results_data, f, indent=2)

        self.get_logger().info(f"\nResults saved to: {output_file}")

        # CSV로도 저장
        csv_file = self._output_dir / 'gazebo_ablation_results.csv'
        with open(csv_file, 'w') as f:
            f.write("trial_id,config,margin_cm,goal_x,goal_y,violated,min_distance,nav_success,duration\n")
            for trial in self._all_trials:
                f.write(f"{trial.trial_id},{trial.margin_config},{trial.margin_value*100:.1f},"
                       f"{trial.goal_x:.3f},{trial.goal_y:.3f},{trial.violated},"
                       f"{trial.min_distance_to_boundary:.4f},{trial.navigation_success},{trial.duration:.2f}\n")

        self.get_logger().info(f"CSV saved to: {csv_file}")


def main(args=None):
    rclpy.init(args=args)

    node = GazeboIntegrationTestNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Test interrupted by user")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
