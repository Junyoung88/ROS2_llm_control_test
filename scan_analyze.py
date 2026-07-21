#!/usr/bin/env python3
"""
Identify the LiDAR angular sector occluded by the robot's own manipulator arm.
Boots the warehouse, holds the robot at the origin (walls far away in the open
centre), aggregates ~15 s of /scan, and reports per-beam statistics so we can see
which bearings return consistently-close (self) hits. Those bearings get masked.
"""
import os
import sys
import time
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                "src/geofence_enforcer/experiments"))
from run_gazebo_s1_s6 import SimulationManager  # noqa: E402
import numpy as np  # noqa: E402


def analyze(seconds=15):
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import LaserScan

    class A(Node):
        def __init__(self):
            super().__init__("scan_analyze")
            self.create_subscription(LaserScan, "/scan", self.cb, 10)
            self.acc = None
            self.close = None
            self.n = 0
            self.amin = self.ainc = None
            self.rmax = None
            self.t0 = time.time()
            self.create_timer(0.2, self.tick)

        def cb(self, m):
            r = np.asarray(m.ranges, dtype=float)
            if self.acc is None:
                self.acc = np.zeros(len(r))
                self.close = np.zeros(len(r))
                self.amin, self.ainc, self.rmax = m.angle_min, m.angle_increment, m.range_max
                self.rmin = m.range_min
            fin = np.isfinite(r) & (r > m.range_min)
            self.acc[fin] += r[fin]
            self.close += ((r > m.range_min) & (r < 0.6)).astype(float)
            self.n += 1

        def tick(self):
            if time.time() - self.t0 > seconds:
                rclpy.shutdown()

    rclpy.init()
    node = A()
    try:
        rclpy.spin(node)
    except Exception:
        pass
    if node.n == 0:
        print("SCANAN: no scans received")
        return
    close_frac = node.close / node.n
    n = len(close_frac)
    angles = node.amin + np.arange(n) * node.ainc
    # report contiguous sectors where >50% of scans are close (self-return)
    mask = close_frac > 0.5
    print(f"SCANAN: {node.n} scans, {n} beams, angle[{math.degrees(node.amin):.0f}"
          f",{math.degrees(node.amin + (n-1)*node.ainc):.0f}]deg, "
          f"close-beams={int(mask.sum())}/{n}")
    # print sectors
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            a0 = math.degrees(angles[i]); a1 = math.degrees(angles[j - 1])
            med = np.median((node.acc / np.maximum(node.n, 1))[i:j])
            print(f"  SELF sector: beams[{i}:{j}] angle[{a0:.1f},{a1:.1f}]deg "
                  f"(~{med:.2f}m, {100*close_frac[i:j].mean():.0f}% close)")
            i = j
        else:
            i += 1
    # also print coarse 30-deg-bin close fractions
    print("  30deg-bin close-fraction:")
    for b0 in range(-180, 180, 30):
        m2 = (angles >= math.radians(b0)) & (angles < math.radians(b0 + 30))
        if m2.any():
            print(f"    [{b0:+4d},{b0+30:+4d}]deg close={100*close_frac[m2].mean():4.0f}%")


def main():
    runner = SimulationManager(headless=True)
    try:
        if not runner.start_gazebo(world="warehouse.sdf"):
            print("boot failed"); return 1
        print("[BOOT] up, analyzing scan at origin...")
        time.sleep(4)
        analyze(15)
        return 0
    finally:
        try:
            runner.stop_all(reset_daemon=True)
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
