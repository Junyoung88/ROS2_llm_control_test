#!/usr/bin/env python3
"""
Build an ACCURATE occupancy-grid map of warehouse.sdf via ground-truth-pose
mapping (mapping with known poses), NOT SLAM.

Motivation: the slam_toolbox map (warehouse_map.*) is noisy/distorted enough that
AMCL localizes with ~0.5-1m error (sometimes wrong heading) — which dominates the
LiDAR-spoof detector's cross-channel signal and makes clean runs erratic. Because
/odom here is a relay of Gazebo's near-ground-truth odometry (no unbounded drift),
we can register each real /scan at the true robot pose and accumulate a clean,
sharp log-odds occupancy grid, perfectly aligned to the Gazebo/world frame (so map
== Gazebo == the frame goals & zones are defined in). AMCL against this map is
accurate → the detector cleanly separates spoof (drift to meters) from clean.

Pipeline:
  1. Boot warehouse.sdf via SimulationManager (Gazebo, relays, cmd_vel bridge).
  2. One node: subscribe /scan + /odom, drive a reactive wander to sweep the
     building, and ray-cast every Nth scan into a log-odds grid at the odom pose.
  3. Threshold -> nav2 PGM/YAML (free=254, occupied=0, unknown=205).

Usage: python3 create_gt_map.py --drive-seconds 360 --out warehouse_map_gt
"""
import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                "src/geofence_enforcer/experiments"))
from run_gazebo_s1_s6 import SimulationManager  # noqa: E402

import numpy as np  # noqa: E402

MAPS_DIR = os.path.join(
    os.path.dirname(__file__),
    "src/mobile_manipulator_tutorial/src/mobile_manip_moveit_config/maps")

# Grid config (world/Gazebo frame). Warehouse fits comfortably in +/-20 m.
RES = 0.05
X_MIN, Y_MIN = -20.0, -20.0
W = int(40.0 / RES)   # 800
H = int(40.0 / RES)   # 800
L_OCC = 0.85          # log-odds increment for a hit
L_FREE = 0.40         # log-odds decrement for a pass-through
L_CLAMP = 12.0        # clamp magnitude
BEAM_STRIDE = 2       # use every 2nd beam (denser near-field sampling)
SCAN_STRIDE = 2       # process every 2nd scan
MAX_MAP_RANGE = 6.0   # only map returns within this range (sharp near-field)
MODEL_INDEX = 11      # index of the 'mobile_manip' root pose in /world/empty/pose/info
                      # (95 entries; name is dropped by the Pose_V->TF bridge)
MIN_MAP_RANGE = 0.20  # scan is clean full-360 (verified: no arm self-occlusion),
                      # so only drop the very-near range_min band.


def world_to_cell(x, y):
    cx = int((x - X_MIN) / RES)
    cy = int((y - Y_MIN) / RES)
    return cx, cy


def build_map(duration_s):
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import LaserScan
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import Twist
    from tf2_msgs.msg import TFMessage

    # Hit/pass counters (robust to oblique walls: a cell hit >=N times is a wall
    # regardless of how many rays pass through it — avoids free-space erasure).
    hits = np.zeros((H, W), dtype=np.uint16)
    passes = np.zeros((H, W), dtype=np.uint16)

    class Mapper(Node):
        def __init__(self):
            super().__init__("gt_mapper")
            self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
            self.create_subscription(LaserScan, "/scan", self.on_scan, 10)
            self.create_subscription(Odometry, "/odom", self.on_odom, 20)
            # TRUE Gazebo model pose (bridged from /world/empty/pose/info as TF).
            # Registering scans against this instead of drifting wheel /odom
            # removes the accumulated-drift smear in the central operating region.
            self.create_subscription(TFMessage, "/world/empty/pose/info",
                                     self.on_gtpose, 20)
            self.pose = None            # (x, y, yaw) TRUE pose from bridged gz poses (REGISTRATION only)
            self.odom_pose = None       # (x, y, yaw) wheel-odom pose (DRIVING control)
            self.wz = 0.0               # latest angular velocity (rad/s)
            self._odom_xy = None        # rough odom position (index sanity/fallback)
            self._last_true = None      # last accepted TRUE (x,y) for continuity gate
            self.started = False        # set once robot first leaves the origin
            from collections import deque
            self.obuf = deque(maxlen=200)   # (t,x,y,yaw) TRUE-pose history for time-match
            self._gt_warned = False
            self.front = None
            self.best_bearing = 0.0
            self.open_left = self.open_right = 1.0
            self.t0 = time.time()
            self.phase_t0 = time.time()
            self.mode = "spin"        # seed with one spin, then follow waypoints
            self.scan_count = 0
            self.integrated = 0
            self.last_log = time.time()
            # Boustrophedon coverage over the operating region + flanking shelves so
            # the whole navigable area (and the walls/shelves AMCL keys off) is
            # mapped at true poses. Robot starts (0,0); nav region ~x[-1,5] y[-1,7].
            wps = []
            ys = [0, 2, 4, 6, 7, 5, 3, 1, -1, -3]
            for j, y in enumerate(ys):
                xr = [-5, -3, -1, 1, 3, 5]
                if j % 2 == 1:
                    xr = xr[::-1]
                for x in xr:
                    wps.append((float(x), float(y)))
            wps.append((0.0, 0.0))    # return home
            self.waypoints = wps
            self.wp_idx = 0
            self.wp_t0 = time.time()
            self.create_timer(0.1, self.step)

        def on_odom(self, msg):
            # /odom (wheel odometry): DRIVES the waypoint controller (proven to
            # traverse the whole building; true-pose control diverged into an
            # in-place spin when a fragile index pick flickered onto a static
            # nearby object). Also gives the rotation gate + a position estimate
            # used only to sanity-check the true-pose index selection.
            self.wz = msg.twist.twist.angular.z
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            yw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                            1 - 2 * (q.y * q.y + q.z * q.z))
            self._odom_xy = (p.x, p.y)
            self.odom_pose = (p.x, p.y, yw)

        def on_gtpose(self, msg):
            # The ros_gz Pose_V->TF bridge drops names, but preserves ORDER: the
            # mobile_manip model-root pose is at MODEL_INDEX in /world/empty/pose/info
            # (95 entries). Use ONLY that index (the old nearest-to-odom fallback
            # could latch onto a static shelf/wall segment near the robot, giving a
            # bogus static yaw that corrupted both control and the map). A temporal
            # continuity gate rejects any single-update teleport (>1.0 m) so a rare
            # index glitch can't poison the pose buffer.
            trs = msg.transforms
            n = len(trs)
            if n <= MODEL_INDEX:
                return
            tr = trs[MODEL_INDEX]
            t3 = tr.transform.translation
            q = tr.transform.rotation
            yw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                            1 - 2 * (q.y * q.y + q.z * q.z))
            x, y = t3.x, t3.y
            # sanity vs odom (index should sit within ~2 m of the wheel-odom guess)
            if self._odom_xy is not None and math.hypot(
                    x - self._odom_xy[0], y - self._odom_xy[1]) > 2.5:
                return
            # continuity: reject a teleport from the last accepted true pose
            if self._last_true is not None and math.hypot(
                    x - self._last_true[0], y - self._last_true[1]) > 1.0:
                return
            self._last_true = (x, y)
            self.pose = (x, y, yw)
            ts = msg.transforms[0].header.stamp.sec + msg.transforms[0].header.stamp.nanosec * 1e-9
            self.obuf.append((ts, x, y, yw))

        def pose_at(self, ts):
            """Odom pose interpolated to scan time ts (linear, yaw-unwrapped)."""
            buf = self.obuf
            if not buf:
                return None
            # find bracketing samples
            lo = None
            for k in range(len(buf) - 1, -1, -1):
                if buf[k][0] <= ts:
                    lo = k
                    break
            if lo is None:
                return (buf[0][1], buf[0][2], buf[0][3])
            if lo == len(buf) - 1:
                return (buf[lo][1], buf[lo][2], buf[lo][3])
            t0, x0, y0, a0 = buf[lo]
            t1, x1, y1, a1 = buf[lo + 1]
            if t1 <= t0:
                return (x0, y0, a0)
            f = max(0.0, min(1.0, (ts - t0) / (t1 - t0)))
            da = math.atan2(math.sin(a1 - a0), math.cos(a1 - a0))
            return (x0 + f * (x1 - x0), y0 + f * (y1 - y0), a0 + f * da)

        def on_scan(self, msg):
            self.scan_count += 1
            n = len(msg.ranges)
            if n == 0:
                return
            amin, ainc = msg.angle_min, msg.angle_increment
            rmax = msg.range_max
            # --- wander sensing (front sector + most-open bearing) ---
            ranges = np.asarray(msg.ranges, dtype=np.float32)
            angles = amin + np.arange(n) * ainc
            finite = np.isfinite(ranges) & (ranges > msg.range_min) & (ranges < rmax)
            r_eff = np.where(finite, ranges, rmax)
            front_mask = (angles >= -0.44) & (angles <= 0.44)
            self.front = float(r_eff[front_mask].min()) if front_mask.any() else rmax
            hemi = (angles >= -1.75) & (angles <= 1.75)
            if hemi.any():
                idx = np.argmax(np.where(hemi, r_eff, -1))
                self.best_bearing = float(angles[idx])
            self.open_left = float(r_eff[(angles > 0.1) & (angles <= 1.75)].max(initial=0))
            self.open_right = float(r_eff[(angles >= -1.75) & (angles < -0.1)].max(initial=0))

            # --- integrate scan into grid at known pose (every SCAN_STRIDE) ---
            # Skip while rotating: at ~0.9 rad/s the odom-yaw/scan timing mismatch
            # smears returns into arcs. Also skip the INITIAL in-place spin at the
            # origin (its smeared scans created the phantom-obstacle "spider" that
            # boxed the robot in and aborted Nav2) — only start mapping once the
            # robot has first driven >0.5 m away from the start.
            if self.odom_pose is not None and not self.started:
                if math.hypot(self.odom_pose[0], self.odom_pose[1]) > 0.5:
                    self.started = True
            if (self.pose is None or not self.started
                    or (self.scan_count % SCAN_STRIDE) != 0
                    or abs(self.wz) > 0.10):
                return
            # use the odom pose at THIS SCAN's timestamp (kills async smear)
            sts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            pm = self.pose_at(sts)
            if pm is None:
                return
            rx, ry, ryaw = pm
            ocx, ocy = world_to_cell(rx, ry)
            if not (0 <= ocx < W and 0 <= ocy < H):
                return
            b = slice(None, None, BEAM_STRIDE)
            ba = angles[b]
            br = ranges[b]
            # Only map NEAR-FIELD returns: beyond ~MAX_RANGE oblique hits are
            # sparse and smear (the radial-streak noise). The operating region's
            # walls/shelves are all within a few metres, which is what AMCL needs.
            # Classify each beam:
            #  hit  = a real wall/shelf return in [MIN, MAX] -> mark occupied end
            #  miss = nothing within MAX (inf / >=MAX)        -> carve free to MAX
            #  self = closer than MIN (arm/body self-return)  -> SKIP (occluded)
            hit = (np.isfinite(br) & (br > MIN_MAP_RANGE) & (br < rmax)
                   & (br < MAX_MAP_RANGE))
            selfret = np.isfinite(br) & (br <= MIN_MAP_RANGE) & (br > msg.range_min)
            use = ~selfret                       # process hits and clean misses only
            er = np.where(hit, br, MAX_MAP_RANGE)
            wa = ryaw + ba
            ex = rx + er * np.cos(wa)
            ey = ry + er * np.sin(wa)
            for k in range(len(ba)):
                if use[k]:
                    self._raycast(ocx, ocy, ex[k], ey[k], bool(hit[k]))
            self.integrated += 1

        def _raycast(self, ocx, ocy, ex, ey, hit):
            ecx, ecy = world_to_cell(ex, ey)
            ecx = max(0, min(W - 1, ecx))
            ecy = max(0, min(H - 1, ecy))
            steps = max(abs(ecx - ocx), abs(ecy - ocy))
            if steps == 0:
                return
            xs = np.linspace(ocx, ecx, steps + 1).astype(np.int32)
            ys = np.linspace(ocy, ecy, steps + 1).astype(np.int32)
            # cells traversed before the endpoint are free evidence
            passes[ys[:-1], xs[:-1]] += 1
            if hit:
                hits[ecy, ecx] += 1

        def step(self):
            now = time.time()
            if now - self.t0 > duration_s:
                self.pub.publish(Twist())
                rclpy.shutdown()
                return
            if self.front is None:
                return
            tw = Twist()
            ep = now - self.phase_t0
            if self.mode == "spin":
                # one initial spin to seed the map, then switch to waypoints
                tw.angular.z = 0.8
                if ep > (2 * math.pi * 1.05) / 0.8:
                    self.mode = "goto"; self.phase_t0 = now; self.wp_t0 = now
            elif self.mode == "goto":
                if self.odom_pose is None:
                    return
                rx, ry, ryaw = self.odom_pose
                if self.wp_idx >= len(self.waypoints):
                    self.pub.publish(Twist()); return
                tx, ty = self.waypoints[self.wp_idx]
                dx, dy = tx - rx, ty - ry
                dist = math.hypot(dx, dy)
                # advance on arrival or if stuck too long on one waypoint
                if dist < 0.5 or (now - self.wp_t0) > 18.0:
                    self.wp_idx += 1; self.wp_t0 = now
                    # brief spin every 4th waypoint to capture surroundings
                    if self.wp_idx % 4 == 0:
                        self.mode = "wpspin"; self.phase_t0 = now
                    return
                if self.front < 0.55:
                    # obstacle ahead -> rotate toward the more open hemisphere
                    tw.angular.z = 0.8 if self.open_left >= self.open_right else -0.8
                else:
                    hd = math.atan2(dy, dx) - ryaw
                    hd = math.atan2(math.sin(hd), math.cos(hd))   # wrap to [-pi,pi]
                    tw.angular.z = max(-0.8, min(0.8, 1.3 * hd))
                    tw.linear.x = 0.22 if abs(hd) < 0.6 else 0.05
            elif self.mode == "wpspin":
                tw.angular.z = 0.9
                if ep > (2 * math.pi * 0.6) / 0.9:
                    self.mode = "goto"; self.phase_t0 = now; self.wp_t0 = now
            self.pub.publish(tw)
            if now - self.last_log > 15:
                self.last_log = now
                op = None if self.odom_pose is None else tuple(round(v, 2) for v in self.odom_pose)
                tp = None if self.pose is None else tuple(round(v, 2) for v in self.pose)
                print(f"[GTMAP] t={int(now-self.t0)}s mode={self.mode} wp={self.wp_idx}/{len(self.waypoints)} "
                      f"scans={self.scan_count} integrated={self.integrated} "
                      f"odom={op} true={tp}",
                      flush=True)

    rclpy.init()
    node = Mapper()
    try:
        rclpy.spin(node)
    except Exception:
        pass
    finally:
        try:
            node.pub.publish(Twist())
        except Exception:
            pass
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()
    return hits, passes


def save_map(hits, passes, stem):
    os.makedirs(MAPS_DIR, exist_ok=True)
    # occupancy image (nav2: 254 free/white, 0 occ/black, 205 unknown).
    # occupied = hit at least twice (robust to a stray beam); free = seen through
    # and never (or barely) hit; else unknown.
    img = np.full((H, W), 205, dtype=np.uint8)
    free_mask = (passes >= 2) & (hits < 3)
    occ_mask = hits >= 3
    img[free_mask] = 254
    img[occ_mask] = 0
    # The robot start (world origin) is known-free — force-clear a 0.5 m radius so
    # a stray phantom cell there can never box the planner in.
    ocx, ocy = world_to_cell(0.0, 0.0)
    rr = int(0.5 / RES)
    yy, xx = np.ogrid[-rr:rr + 1, -rr:rr + 1]
    disk = (xx * xx + yy * yy) <= rr * rr
    r0c, r1c = max(0, ocy - rr), min(H, ocy + rr + 1)
    c0c, c1c = max(0, ocx - rr), min(W, ocx + rr + 1)
    sub = img[r0c:r1c, c0c:c1c]
    dsub = disk[(r0c - (ocy - rr)):(r1c - (ocy - rr)), (c0c - (ocx - rr)):(c1c - (ocx - rr))]
    sub[dsub] = 254
    img[r0c:r1c, c0c:c1c] = sub
    # crop to observed region + small border for a tidy map
    observed = (hits > 0) | (passes > 0)
    if observed.any():
        rows = np.any(observed, axis=1)
        cols = np.any(observed, axis=0)
        r0, r1 = np.where(rows)[0][[0, -1]]
        c0, c1 = np.where(cols)[0][[0, -1]]
        pad = 10
        r0 = max(0, r0 - pad); r1 = min(H - 1, r1 + pad)
        c0 = max(0, c0 - pad); c1 = min(W - 1, c1 + pad)
    else:
        r0, r1, c0, c1 = 0, H - 1, 0, W - 1
    crop = img[r0:r1 + 1, c0:c1 + 1]
    # PGM rows go top-down; nav2 origin is bottom-left, so flip vertically on save
    ch, cw = crop.shape
    origin_x = X_MIN + c0 * RES
    origin_y = Y_MIN + r0 * RES
    pgm_path = os.path.join(MAPS_DIR, stem + ".pgm")
    yaml_path = os.path.join(MAPS_DIR, stem + ".yaml")
    flipped = np.flipud(crop)
    with open(pgm_path, "wb") as f:
        f.write(f"P5\n{cw} {ch}\n255\n".encode())
        f.write(flipped.tobytes())
    with open(yaml_path, "w") as f:
        f.write(f"image: {stem}.pgm\n")
        f.write(f"resolution: {RES}\n")
        f.write(f"origin: [{origin_x:.4f}, {origin_y:.4f}, 0.0]\n")
        f.write("negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\n")
    occ = int((crop == 0).sum()); free = int((crop == 254).sum())
    print(f"[GTMAP] saved {stem}: {cw}x{ch} occ={occ} free={free} "
          f"origin=({origin_x:.2f},{origin_y:.2f})")
    return os.path.exists(pgm_path) and os.path.exists(yaml_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive-seconds", type=int, default=360)
    ap.add_argument("--out", default="warehouse_map_gt")
    args = ap.parse_args()

    import subprocess
    runner = SimulationManager(headless=True)
    bridge = None
    try:
        print("[BOOT] Starting Gazebo warehouse + robot...")
        if not runner.start_gazebo(world="warehouse.sdf"):
            print("[BOOT][ERROR] Gazebo failed to boot")
            return 1
        # Bridge the TRUE model poses (gz /world/empty/pose/info) into ROS as TF so
        # the mapper can register scans against ground truth instead of drifting odom.
        print("[BOOT] Starting ground-truth pose bridge...")
        bridge = subprocess.Popen(
            ["ros2", "run", "ros_gz_bridge", "parameter_bridge",
             "/world/empty/pose/info@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("[BOOT] Gazebo up. TRUE-pose mapping "
              f"({args.drive_seconds}s)...")
        time.sleep(3)
        hits, passes = build_map(args.drive_seconds)
        ok = save_map(hits, passes, args.out)
        return 0 if ok else 3
    finally:
        print("[CLEANUP] Stopping sim...")
        if bridge is not None:
            try:
                bridge.terminate()
            except Exception:
                pass
        try:
            runner.stop_all(reset_daemon=True)
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
