#!/usr/bin/env python3
"""
Runtime Geofence Monitor - 실시간 금지구역 감시

이 모듈은 로봇의 실시간 위치를 모니터링하여
금지구역 진입 시 즉시 정지 명령을 내립니다.

이것은 Safety Layer의 두 번째 방어선입니다:
- Layer 3 (Safety Layer): 목표 좌표 검증
- Layer 4 (Runtime Monitor): 실시간 위치 감시 + 긴급 정지

문제 상황:
    목표 좌표는 안전해도, 경로(Path)가 금지구역을 통과할 수 있음

해결책:
    실시간으로 로봇 위치를 감시하고, 금지구역 진입 시 즉시 정지
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from std_msgs.msg import String, Bool
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from action_msgs.srv import CancelGoal

from typing import Optional, Tuple, List
from dataclasses import dataclass
import time

from .geofence_geometry import GeofenceGeometry


@dataclass
class ViolationEvent:
    """금지구역 위반 이벤트"""
    timestamp: float
    position: Tuple[float, float]
    zone_name: str
    action_taken: str


class RuntimeGeofenceMonitor(Node):
    """
    실시간 금지구역 감시 노드

    기능:
    1. 로봇 위치를 실시간 모니터링
    2. 금지구역 진입 감지
    3. 즉시 정지 명령 발행
    4. Navigation 취소

    이것은 Defense-in-Depth의 마지막 방어선입니다.
    """

    def __init__(self, geofence: GeofenceGeometry, safety_margin: float = 0.1):
        super().__init__('runtime_geofence_monitor')

        self._geofence = geofence
        self._safety_margin = safety_margin

        # QoS 설정
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            depth=10
        )

        # 현재 위치
        self._current_position: Optional[Tuple[float, float]] = None

        # 위반 기록
        self._violations: List[ViolationEvent] = []
        self._in_violation = False

        # 위치 구독
        self._pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self._pose_callback,
            sensor_qos
        )

        # 정지 명령 발행
        self._cmd_vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        # 긴급 정지 상태 발행
        self._emergency_stop_pub = self.create_publisher(
            Bool,
            '/geofence/emergency_stop',
            10
        )

        # 이벤트 로그 발행
        self._event_pub = self.create_publisher(
            String,
            '/geofence/violations',
            10
        )

        # 모니터링 타이머 (10Hz)
        self._monitor_timer = self.create_timer(0.1, self._monitor_callback)

        # 통계
        self._total_checks = 0
        self._total_violations = 0

        self.get_logger().info("Runtime Geofence Monitor started")
        self.get_logger().info(f"Monitoring {len(geofence.get_zone_names())} zones")
        self.get_logger().info(f"Safety margin: {safety_margin}m")

    def _pose_callback(self, msg: PoseWithCovarianceStamped):
        """위치 업데이트"""
        self._current_position = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y
        )

    def _monitor_callback(self):
        """주기적 모니터링"""
        if self._current_position is None:
            return

        self._total_checks += 1
        x, y = self._current_position

        # 금지구역 체크
        result = self._geofence.check_point(x, y, margin=self._safety_margin)

        if result.is_violated:
            self._handle_violation(x, y, result.in_forbidden_zones)
        else:
            if self._in_violation:
                self.get_logger().info(f"Robot exited forbidden zone at ({x:.2f}, {y:.2f})")
                self._in_violation = False

            # 버퍼 구역 경고
            if result.in_buffer_zones:
                self.get_logger().warn(
                    f"Robot in buffer zone: {result.in_buffer_zones} at ({x:.2f}, {y:.2f})"
                )

    def _handle_violation(self, x: float, y: float, zones: List[str]):
        """금지구역 위반 처리"""
        self._total_violations += 1

        # 즉시 정지!
        self._emergency_stop()

        # 이벤트 기록
        event = ViolationEvent(
            timestamp=time.time(),
            position=(x, y),
            zone_name=zones[0] if zones else "unknown",
            action_taken="EMERGENCY_STOP"
        )
        self._violations.append(event)

        # 로그
        if not self._in_violation:
            self.get_logger().error(
                f"🚨 FORBIDDEN ZONE VIOLATION! Position: ({x:.2f}, {y:.2f}), Zone: {zones}"
            )
            self._in_violation = True

        # 이벤트 발행
        msg = String()
        msg.data = f"VIOLATION|{x:.3f},{y:.3f}|{zones}|EMERGENCY_STOP"
        self._event_pub.publish(msg)

    def _emergency_stop(self):
        """긴급 정지"""
        # 속도 0 명령
        stop_cmd = Twist()
        stop_cmd.linear.x = 0.0
        stop_cmd.linear.y = 0.0
        stop_cmd.angular.z = 0.0

        # 여러 번 발행 (확실하게)
        for _ in range(5):
            self._cmd_vel_pub.publish(stop_cmd)

        # 긴급 정지 상태 발행
        emergency_msg = Bool()
        emergency_msg.data = True
        self._emergency_stop_pub.publish(emergency_msg)

    def get_statistics(self) -> dict:
        """통계 반환"""
        return {
            'total_checks': self._total_checks,
            'total_violations': self._total_violations,
            'violation_rate': self._total_violations / max(1, self._total_checks),
            'recent_violations': [
                {
                    'time': v.timestamp,
                    'position': v.position,
                    'zone': v.zone_name
                }
                for v in self._violations[-10:]
            ]
        }


def main(args=None):
    rclpy.init(args=args)

    # Geofence 설정
    geofence = GeofenceGeometry()
    geofence.add_zone(
        "workcell",
        [(4.5, 4.5), (6.5, 4.5), (6.5, 6.5), (4.5, 6.5)],
        metadata={'type': 'forbidden'}
    )

    node = RuntimeGeofenceMonitor(geofence, safety_margin=0.1)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stats = node.get_statistics()
        print(f"\n=== Runtime Monitor Statistics ===")
        print(f"Total checks: {stats['total_checks']}")
        print(f"Total violations: {stats['total_violations']}")
        print(f"Violation rate: {stats['violation_rate']*100:.2f}%")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
