#!/usr/bin/env python3
"""
Costmap Integration Test - Nav2 KeepoutFilter 연동 테스트

이 테스트는:
1. costmap_mask_publisher 노드 실행
2. /keepout_filter_mask 토픽 발행 확인
3. 금지구역이 마스크에 올바르게 표시되는지 검증

테스트 방법:
    # 1. 이 스크립트 실행
    python3 demo/costmap_integration_test.py

    # 2. Nav2와 함께 테스트 (별도 터미널)
    ros2 launch geofence_enforcer costmap_mask.launch.py

    # 3. 토픽 확인
    ros2 topic echo /keepout_filter_mask --once
    ros2 topic echo /costmap_filter_info --once
"""

import sys
import os
import time
from typing import Optional

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
    from nav_msgs.msg import OccupancyGrid
    from nav2_msgs.msg import CostmapFilterInfo
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    print("⚠️  ROS2가 설치되지 않았습니다.")


class CostmapIntegrationTestNode(Node):
    """Costmap mask 수신 테스트 노드"""

    def __init__(self):
        super().__init__('costmap_integration_test')

        # QoS (transient local to match publisher)
        latched_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=1
        )

        # 수신된 메시지
        self.mask_msg: Optional[OccupancyGrid] = None
        self.filter_info_msg: Optional[CostmapFilterInfo] = None

        # 구독자
        self.mask_sub = self.create_subscription(
            OccupancyGrid,
            '/keepout_filter_mask',
            self._mask_callback,
            latched_qos
        )

        self.filter_info_sub = self.create_subscription(
            CostmapFilterInfo,
            '/costmap_filter_info',
            self._filter_info_callback,
            latched_qos
        )

        self.get_logger().info("Costmap Integration Test Node started")
        self.get_logger().info("Waiting for /keepout_filter_mask and /costmap_filter_info...")

    def _mask_callback(self, msg: OccupancyGrid):
        """마스크 수신"""
        self.mask_msg = msg
        self.get_logger().info(
            f"Received mask: {msg.info.width}x{msg.info.height} cells, "
            f"resolution: {msg.info.resolution}m"
        )

    def _filter_info_callback(self, msg: CostmapFilterInfo):
        """필터 정보 수신"""
        self.filter_info_msg = msg
        self.get_logger().info(
            f"Received filter info: type={msg.type}, "
            f"mask_topic={msg.filter_mask_topic}"
        )


def run_integration_test():
    """통합 테스트 실행"""
    print("=" * 60)
    print(" Costmap Integration Test ".center(60))
    print("=" * 60)

    if not ROS_AVAILABLE:
        print("\n❌ ROS2가 설치되지 않아 테스트를 실행할 수 없습니다.")
        return False

    rclpy.init()
    test_node = CostmapIntegrationTestNode()

    print("\n[1] costmap_mask_publisher 토픽 대기 중...")
    print("    (별도 터미널에서 다음 명령 실행:)")
    print("    ros2 launch geofence_enforcer costmap_mask.launch.py")
    print()

    # 토픽 대기 (최대 30초)
    start_time = time.time()
    timeout = 30.0

    while time.time() - start_time < timeout:
        rclpy.spin_once(test_node, timeout_sec=1.0)

        if test_node.mask_msg is not None and test_node.filter_info_msg is not None:
            break

        elapsed = time.time() - start_time
        print(f"\r    대기 중... ({elapsed:.1f}초)", end="", flush=True)

    print()

    # 결과 확인
    print("\n[2] 결과 확인")
    print("-" * 40)

    results = []

    # 마스크 확인
    if test_node.mask_msg is not None:
        mask = test_node.mask_msg
        print(f"✅ /keepout_filter_mask 수신 완료")
        print(f"   - 크기: {mask.info.width}x{mask.info.height} cells")
        print(f"   - 해상도: {mask.info.resolution}m")
        print(f"   - 프레임: {mask.header.frame_id}")
        print(f"   - 원점: ({mask.info.origin.position.x}, {mask.info.origin.position.y})")

        # 금지구역 셀 수 확인
        lethal_count = sum(1 for cell in mask.data if cell == 100)
        free_count = sum(1 for cell in mask.data if cell == 0)
        print(f"   - 금지구역 셀: {lethal_count}")
        print(f"   - 자유 영역 셀: {free_count}")

        if lethal_count > 0:
            results.append(("Mask 발행", True))
            results.append(("금지구역 마킹", True))
        else:
            results.append(("Mask 발행", True))
            results.append(("금지구역 마킹", False))
    else:
        print("❌ /keepout_filter_mask 수신 실패")
        results.append(("Mask 발행", False))

    # 필터 정보 확인
    if test_node.filter_info_msg is not None:
        info = test_node.filter_info_msg
        print(f"\n✅ /costmap_filter_info 수신 완료")
        print(f"   - 타입: {info.type} (0=KEEPOUT)")
        print(f"   - 마스크 토픽: {info.filter_mask_topic}")
        print(f"   - base: {info.base}, multiplier: {info.multiplier}")

        if info.type == 0 and info.filter_mask_topic == '/keepout_filter_mask':
            results.append(("Filter 정보 발행", True))
        else:
            results.append(("Filter 정보 발행", False))
    else:
        print("\n❌ /costmap_filter_info 수신 실패")
        results.append(("Filter 정보 발행", False))

    # 금지구역 좌표 검증
    if test_node.mask_msg is not None:
        print("\n[3] 금지구역 좌표 검증")
        print("-" * 40)

        mask = test_node.mask_msg
        resolution = mask.info.resolution
        origin_x = mask.info.origin.position.x
        origin_y = mask.info.origin.position.y

        # 금지구역 중심 (5.5, 5.5)
        test_points = [
            ("금지구역 중심 (5.5, 5.5)", 5.5, 5.5, 100),
            ("안전구역 (2.0, 2.0)", 2.0, 2.0, 0),
            ("안전구역 (8.0, 8.0)", 8.0, 8.0, 0),
        ]

        for name, x, y, expected in test_points:
            col = int((x - origin_x) / resolution)
            row = int((y - origin_y) / resolution)
            idx = row * mask.info.width + col

            if 0 <= idx < len(mask.data):
                actual = mask.data[idx]
                status = "✅" if actual == expected else "❌"
                print(f"   {status} {name}: expected={expected}, actual={actual}")
                results.append((name, actual == expected))
            else:
                print(f"   ⚠️  {name}: 인덱스 범위 초과")
                results.append((name, False))

    # 최종 결과
    print("\n" + "=" * 60)
    print(" 테스트 결과 ".center(60))
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status}: {name}")

    print(f"\n   총 {total}개 중 {passed}개 통과")

    test_node.destroy_node()
    rclpy.shutdown()

    return passed == total


def main():
    """메인 함수"""
    success = run_integration_test()

    if success:
        print("\n✅ 통합 테스트 완료!")
        print("   Nav2 KeepoutFilter와 함께 사용할 준비가 되었습니다.")
        print("\n다음 단계:")
        print("   1. Nav2 실행 시 nav2_keepout.yaml 설정 사용")
        print("   2. ros2 launch geofence_enforcer costmap_mask.launch.py 실행")
        print("   3. 경로 계획이 금지구역을 자동으로 피하는지 확인")
    else:
        print("\n❌ 일부 테스트 실패")
        print("   costmap_mask_publisher가 실행 중인지 확인하세요.")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
