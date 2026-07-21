#!/usr/bin/env python3
"""
Build an occupancy-grid map of warehouse.sdf via slam_toolbox + reactive wander.

Prior experiments localized AMCL against a *blank* empty_map.yaml, which is
meaningless inside the cluttered warehouse (LIDAR sees walls/bins but the map is
empty -> AMCL never converges -> /initialpose blocks -> Nav2 aborts). This script
produces a real warehouse occupancy grid so AMCL can actually localize, which is
the prerequisite for the warehouse+AMCL+LIDAR-spoofing experiment.

Pipeline:
  1. Boot warehouse.sdf + robot (reuse GazeboExperimentRunner: EKF odom TF,
     odom_real->odom relay, scan_real->/scan relay, cmd_vel bridge).
  2. Launch slam_toolbox online_async (mapping mode, /scan, map->odom).
  3. Reactive random-walk wander (subscribe /scan, publish /cmd_vel) so the LIDAR
     sweeps the whole warehouse while slam_toolbox accumulates + closes loops.
  4. Save map -> maps/warehouse_map.{yaml,pgm} via nav2_map_server map_saver_cli.

Run from repo root:
  python3 create_warehouse_map.py [--drive-seconds N] [--headless-slam]
"""
import os
import sys
import time
import math
import signal
import argparse
import subprocess

WS = "/home/jim/ros2_motion_planning_tutorials"
sys.path.insert(0, os.path.join(WS, "src/geofence_enforcer/experiments"))
from run_gazebo_s1_s6 import SimulationManager  # noqa: E402

MAPS_DIR = os.path.join(
    WS, "src/mobile_manipulator_tutorial/src/mobile_manip_moveit_config/maps")
MAP_STEM = os.path.join(MAPS_DIR, "warehouse_map")
SLAM_PARAMS = os.path.join(
    WS, "src/mobile_manipulator_tutorial/src/mobile_manip_moveit_config/"
        "config/slam_toolbox_mapping.yaml")
SRC = "source /opt/ros/jazzy/setup.bash && source {}/install/setup.bash".format(WS)


def launch_slam():
    """Launch slam_toolbox online_async in mapping mode."""
    cmd = (
        f"{SRC} && ros2 launch slam_toolbox online_async_launch.py "
        f"use_sim_time:=true slam_params_file:={SLAM_PARAMS}"
    )
    log = open("/tmp/slam_mapping.log", "w")
    proc = subprocess.Popen(cmd, shell=True, executable="/bin/bash",
                            stdout=log, stderr=log, preexec_fn=os.setsid)
    return proc, log


def wait_for_map_topic(timeout=60):
    """Wait until slam_toolbox publishes /map."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = subprocess.run(
            f"{SRC} && ros2 topic list 2>/dev/null | grep -qx '/map'",
            shell=True, executable="/bin/bash")
        if r.returncode == 0:
            # confirm data flows
            r2 = subprocess.run(
                f"{SRC} && timeout 8 ros2 topic echo /map --once 2>/dev/null | wc -l",
                shell=True, executable="/bin/bash", capture_output=True, text=True)
            if int((r2.stdout or "0").strip() or 0) > 3:
                return True
        time.sleep(3)
    return False


def wander(duration_s):
    """Reactive random-walk exploration to sweep the warehouse with the LIDAR."""
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import LaserScan
    from geometry_msgs.msg import Twist

    class Wanderer(Node):
        def __init__(self):
            super().__init__("warehouse_wanderer")
            self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
            self.sub = self.create_subscription(
                LaserScan, "/scan", self.on_scan, 10)
            self.front = None       # min range in +/-25deg front sector
            self.best_bearing = 0.0  # bearing (rad, robot frame) of most-open front dir
            self.open_left = 1.0     # max range over left hemisphere
            self.open_right = 1.0    # max range over right hemisphere
            self.t0 = time.time()
            self.phase_t0 = time.time()
            self.mode = "spin"       # start with a 360 spin to seed the map
            self.last_yaw_time = time.time()
            self.timer = self.create_timer(0.1, self.step)
            self.scan_count = 0

        def on_scan(self, msg):
            self.scan_count += 1
            n = len(msg.ranges)
            if n == 0:
                return
            amin, ainc = msg.angle_min, msg.angle_increment
            front_vals = []
            best_r, best_b = -1.0, 0.0
            left_max, right_max = 0.0, 0.0
            for i, r in enumerate(msg.ranges):
                if not (msg.range_min < r < msg.range_max) or math.isinf(r):
                    r_eff = msg.range_max
                else:
                    r_eff = r
                a = amin + i * ainc
                if -0.44 <= a <= 0.44:                 # +/-25deg front
                    front_vals.append(r_eff)
                # most-open direction within the front hemisphere (+/-100deg)
                if -1.75 <= a <= 1.75 and r_eff > best_r:
                    best_r, best_b = r_eff, a
                if 0.1 < a <= 1.75:
                    left_max = max(left_max, r_eff)
                if -1.75 <= a < -0.1:
                    right_max = max(right_max, r_eff)
            self.front = min(front_vals) if front_vals else msg.range_max
            self.best_bearing = best_b
            self.open_left, self.open_right = left_max, right_max

        def step(self):
            now = time.time()
            if now - self.t0 > duration_s:
                self.pub.publish(Twist())
                rclpy.shutdown()
                return
            if self.front is None:
                return  # no scan yet

            tw = Twist()
            elapsed_phase = now - self.phase_t0

            if self.mode == "spin":
                # rotate ~1.3 full turns to capture surroundings + close loops
                tw.angular.z = 0.7
                if elapsed_phase > (2 * math.pi * 1.3) / 0.7:
                    self.mode = "drive"
                    self.phase_t0 = now
            elif self.mode == "drive":
                if self.front > 0.75:
                    tw.linear.x = 0.22
                    # steer toward the most-open direction (frontier seeking)
                    tw.angular.z = max(-0.6, min(0.6, 1.0 * self.best_bearing))
                    # periodically re-spin to close loops / fill occlusions
                    if elapsed_phase > 6.0:
                        self.mode = "spin"
                        self.phase_t0 = now
                else:
                    # obstacle ahead -> turn toward the more open hemisphere
                    self.mode = "turn"
                    self.phase_t0 = now
                    self.turn_dir = 1.0 if self.open_left >= self.open_right else -1.0
            elif self.mode == "turn":
                tw.angular.z = 0.8 * getattr(self, "turn_dir", 1.0)
                if self.front > 1.3 or elapsed_phase > 5.0:
                    self.mode = "drive"
                    self.phase_t0 = now
            self.pub.publish(tw)

            if int(now - self.t0) % 20 == 0 and (now - self.last_yaw_time) > 1:
                self.last_yaw_time = now
                print(f"[WANDER] t={int(now-self.t0)}s mode={self.mode} "
                      f"front={self.front:.2f} bearing={self.best_bearing:+.2f} "
                      f"scans={self.scan_count}", flush=True)

    rclpy.init()
    node = Wanderer()
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


def save_map():
    """Save the slam_toolbox map to maps/warehouse_map.{yaml,pgm}."""
    os.makedirs(MAPS_DIR, exist_ok=True)
    # serialize pose-graph too (useful for slam_toolbox localization mode later)
    cmd = (
        f"{SRC} && ros2 run nav2_map_server map_saver_cli "
        f"-f {MAP_STEM} --ros-args -p save_map_timeout:=20.0 "
        f"-p free_thresh_default:=0.25 -p occupied_thresh_default:=0.65"
    )
    print(f"[MAP] Saving map -> {MAP_STEM}.{{yaml,pgm}}")
    r = subprocess.run(cmd, shell=True, executable="/bin/bash",
                       capture_output=True, text=True, timeout=60)
    print(r.stdout[-500:] if r.stdout else "")
    if r.returncode != 0:
        print("[MAP][ERROR] map_saver_cli failed:\n" + (r.stderr[-800:] or ""))
        return False
    ok = os.path.exists(MAP_STEM + ".pgm") and os.path.exists(MAP_STEM + ".yaml")
    print(f"[MAP] saved={ok}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive-seconds", type=int, default=240)
    args = ap.parse_args()

    runner = SimulationManager(headless=True)
    slam_proc = slam_log = None
    try:
        print("[BOOT] Starting Gazebo warehouse + robot...")
        if not runner.start_gazebo(world="warehouse.sdf"):
            print("[BOOT][ERROR] Gazebo failed to boot")
            return 1
        print("[BOOT] Gazebo up. Launching slam_toolbox...")
        slam_proc, slam_log = launch_slam()
        if not wait_for_map_topic(timeout=90):
            print("[SLAM][ERROR] /map never appeared — check /tmp/slam_mapping.log")
            return 2
        print("[SLAM] /map is live. Starting reactive wander "
              f"({args.drive_seconds}s)...")
        wander(args.drive_seconds)
        print("[SLAM] Wander done. Letting slam settle (8s)...")
        time.sleep(8)
        ok = save_map()
        return 0 if ok else 3
    finally:
        print("[CLEANUP] Stopping slam + sim...")
        if slam_proc is not None:
            try:
                os.killpg(os.getpgid(slam_proc.pid), signal.SIGKILL)
            except Exception:
                pass
        if slam_log is not None:
            try:
                slam_log.close()
            except Exception:
                pass
        try:
            runner.stop_all(reset_daemon=True)
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
