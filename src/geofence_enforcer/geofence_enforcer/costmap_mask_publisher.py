#!/usr/bin/env python3
"""
Costmap Mask Publisher - Geofence를 Nav2 Costmap에 통합

이 노드는 Geofence YAML 설정을 읽어 Nav2 Keepout Filter용
Costmap mask를 생성하고 발행합니다.

이렇게 하면:
1. 경로 계획 단계에서 금지구역을 자동으로 피함
2. Runtime 정지가 필요 없음
3. 로봇이 금지구역을 통과하는 경로를 아예 생성하지 않음

토픽:
    /keepout_filter_mask (nav_msgs/OccupancyGrid) - Costmap mask
    /costmap_filter_info (nav2_msgs/CostmapFilterInfo) - Filter info

사용:
    ros2 run geofence_enforcer costmap_mask_publisher --ros-args \\
        -p geofence_config:=/path/to/geofence.yaml \\
        -p resolution:=0.05
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

from nav_msgs.msg import OccupancyGrid, MapMetaData
from geometry_msgs.msg import Pose
from std_msgs.msg import Header
from nav2_msgs.msg import CostmapFilterInfo

import numpy as np
from typing import List, Tuple, Optional
import yaml
from pathlib import Path

from .geofence_geometry import GeofenceGeometry


class CostmapMaskPublisher(Node):
    """
    Geofence를 Nav2 Costmap mask로 변환하여 발행

    Nav2의 KeepoutFilter는 이 mask를 사용하여
    경로 계획 시 금지구역을 피합니다.
    """

    def __init__(self):
        super().__init__('costmap_mask_publisher')

        # 파라미터
        self.declare_parameter('geofence_config', '')
        self.declare_parameter('resolution', 0.05)
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('map_width', 20.0)  # meters
        self.declare_parameter('map_height', 20.0)  # meters
        self.declare_parameter('map_origin_x', -10.0)
        self.declare_parameter('map_origin_y', -10.0)
        self.declare_parameter('publish_rate', 1.0)  # Hz
        self.declare_parameter('inflation_radius', 0.1)  # meters

        self._config_path = self.get_parameter('geofence_config').value
        self._resolution = self.get_parameter('resolution').value
        self._map_frame = self.get_parameter('map_frame').value
        self._map_width = self.get_parameter('map_width').value
        self._map_height = self.get_parameter('map_height').value
        self._map_origin_x = self.get_parameter('map_origin_x').value
        self._map_origin_y = self.get_parameter('map_origin_y').value
        self._publish_rate = self.get_parameter('publish_rate').value
        self._inflation_radius = self.get_parameter('inflation_radius').value

        # QoS (latched like tf_static)
        latched_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=1
        )

        # Publishers
        self._mask_pub = self.create_publisher(
            OccupancyGrid,
            '/keepout_filter_mask',
            latched_qos
        )

        self._filter_info_pub = self.create_publisher(
            CostmapFilterInfo,
            '/costmap_filter_info',
            latched_qos
        )

        # Geofence 로드
        self._geofence: Optional[GeofenceGeometry] = None
        self._mask_msg: Optional[OccupancyGrid] = None

        if self._config_path:
            self._load_geofence()
            self._generate_mask()

        # 타이머
        self._timer = self.create_timer(
            1.0 / self._publish_rate,
            self._publish_callback
        )

        self.get_logger().info("Costmap Mask Publisher started")
        self.get_logger().info(f"  Config: {self._config_path}")
        self.get_logger().info(f"  Resolution: {self._resolution}m")
        self.get_logger().info(f"  Map size: {self._map_width}x{self._map_height}m")

    def _load_geofence(self):
        """Geofence YAML 로드"""
        try:
            self._geofence = GeofenceGeometry.from_yaml(self._config_path)
            zones = self._geofence.get_zone_names()
            forbidden = self._geofence.get_forbidden_zone_names()
            self.get_logger().info(f"Loaded {len(zones)} zones ({len(forbidden)} forbidden)")
        except Exception as e:
            self.get_logger().error(f"Failed to load geofence: {e}")

    def _generate_mask(self):
        """Costmap mask 생성"""
        if self._geofence is None:
            return

        # 그리드 크기 계산
        width_cells = int(self._map_width / self._resolution)
        height_cells = int(self._map_height / self._resolution)

        # 빈 그리드 (0 = free)
        grid = np.zeros((height_cells, width_cells), dtype=np.int8)

        # 금지구역을 마킹 (100 = lethal obstacle)
        # 'forbidden' 타입 또는 모든 zone을 금지구역으로 처리
        forbidden_zones = self._geofence.get_zones_by_type('forbidden')

        # 'forbidden' 타입이 없으면 모든 zone을 금지구역으로 처리
        if not forbidden_zones:
            forbidden_zones = {name: self._geofence._zones[name]
                             for name in self._geofence.get_zone_names()}

        for zone_name, zone in forbidden_zones.items():
            self._mark_polygon_on_grid(
                grid, zone.vertices, 100, self._inflation_radius
            )
            self.get_logger().info(f"Marked forbidden zone: {zone_name}")

        # OccupancyGrid 메시지 생성
        msg = OccupancyGrid()
        msg.header = Header()
        msg.header.frame_id = self._map_frame
        msg.header.stamp = self.get_clock().now().to_msg()

        msg.info = MapMetaData()
        msg.info.resolution = self._resolution
        msg.info.width = width_cells
        msg.info.height = height_cells
        msg.info.origin = Pose()
        msg.info.origin.position.x = self._map_origin_x
        msg.info.origin.position.y = self._map_origin_y
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0

        # 그리드 데이터 (row-major order)
        msg.data = grid.flatten().tolist()

        self._mask_msg = msg
        self.get_logger().info(f"Generated mask: {width_cells}x{height_cells} cells")

    def _mark_polygon_on_grid(
        self,
        grid: np.ndarray,
        vertices: List[Tuple[float, float]],
        value: int,
        inflation: float = 0.0
    ):
        """폴리곤을 그리드에 마킹"""
        from shapely.geometry import Point, Polygon

        # 폴리곤 생성 및 팽창
        polygon = Polygon(vertices)
        if inflation > 0:
            polygon = polygon.buffer(inflation)

        height, width = grid.shape

        # 각 셀 체크
        for row in range(height):
            for col in range(width):
                # 셀 중심 좌표
                x = self._map_origin_x + (col + 0.5) * self._resolution
                y = self._map_origin_y + (row + 0.5) * self._resolution

                point = Point(x, y)
                if polygon.contains(point) or polygon.touches(point):
                    grid[row, col] = value

    def _publish_callback(self):
        """주기적으로 mask 발행"""
        if self._mask_msg is None:
            return

        # 타임스탬프 업데이트
        self._mask_msg.header.stamp = self.get_clock().now().to_msg()

        # Mask 발행
        self._mask_pub.publish(self._mask_msg)

        # Filter info 발행
        info_msg = CostmapFilterInfo()
        info_msg.header.frame_id = self._map_frame
        info_msg.header.stamp = self.get_clock().now().to_msg()
        info_msg.type = 0  # KEEPOUT
        info_msg.filter_mask_topic = '/keepout_filter_mask'
        info_msg.base = 0.0
        info_msg.multiplier = 1.0

        self._filter_info_pub.publish(info_msg)

    def update_geofence(self, config_path: str):
        """Geofence 동적 업데이트"""
        self._config_path = config_path
        self._load_geofence()
        self._generate_mask()
        self.get_logger().info("Geofence updated, new mask generated")


def main(args=None):
    rclpy.init(args=args)

    node = CostmapMaskPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
