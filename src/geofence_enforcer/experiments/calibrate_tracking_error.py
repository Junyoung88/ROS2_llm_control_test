#!/usr/bin/env python3
"""
Tracking Error Calibration for RA-L Safety Margin Formula.

Measures Nav2 DWB controller path-tracking deviation at various speeds
to calibrate e_0 (static) and c_1 (velocity-proportional) parameters:

    e_track(v) = e_0 + c_1 * v

Uses the existing GazeboExperimentRunner infrastructure for reliable
Gazebo/Nav2 management (teleport, TF correction, periodic restarts).

Usage:
  python3 calibrate_tracking_error.py --seeds 5
"""

import argparse
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# ─── Configuration ───────────────────────────────────────────────────────────

WORKSPACE_DIR = str(Path(__file__).resolve().parents[3])
sys.path.insert(0, str(Path(__file__).parent))

SPEEDS = [0.0, 0.10, 0.15, 0.22, 0.30, 0.40, 0.50]
NAV_DISTANCE = 6.0
MEASURE_X_MIN = 1.0
MEASURE_X_MAX = 5.0
GZ_WORLD = "empty"
RESTART_EVERY = 3  # Restart Gazebo every N trials (DiffDrive yaw drift)

TRAJ_LOG = Path("/tmp/tracking_trajectory.jsonl")


@dataclass
class TrialResult:
    speed_setting: float
    seed: int
    actual_max_speed: float
    actual_mean_speed: float
    max_lateral_dev: float
    mean_lateral_dev: float
    p95_lateral_dev: float
    p99_lateral_dev: float
    n_samples: int
    nav_success: bool


# ─── Background Trajectory Recorder ──────────────────────────────────────────

RECORDER_SCRIPT = r'''#!/usr/bin/env python3
"""Background Gazebo pose recorder using streaming gz topic."""
import json, re, subprocess, sys, time

GZ_WORLD = sys.argv[1] if len(sys.argv) > 1 else "empty"
LOG_FILE = sys.argv[2] if len(sys.argv) > 2 else "/tmp/tracking_trajectory.jsonl"

proc = subprocess.Popen(
    ["gz", "topic", "-e", "-t", f"/world/{GZ_WORLD}/pose/info"],
    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
)

pattern = re.compile(
    r'name: "mobile_manip".*?position \{\s*x: ([\d.e+-]+)\s*y: ([\d.e+-]+)',
    re.DOTALL
)

with open(LOG_FILE, "a") as f:
    buffer = ""
    last_write = 0
    try:
        for line in proc.stdout:
            buffer += line
            if "mobile_manip" in buffer and buffer.count("position {") >= 1:
                match = pattern.search(buffer)
                if match:
                    now = time.time()
                    if now - last_write >= 0.05:  # ~20Hz
                        entry = {"t": now, "x": float(match.group(1)), "y": float(match.group(2))}
                        f.write(json.dumps(entry) + "\n")
                        f.flush()
                        last_write = now
                last_brace = buffer.rfind("}")
                if last_brace > 0:
                    buffer = buffer[last_brace+1:]
    except (KeyboardInterrupt, BrokenPipeError):
        pass
    finally:
        proc.terminate()
'''

_recorder_proc = None


def start_recorder():
    global _recorder_proc
    stop_recorder()

    script_file = Path("/tmp/tracking_recorder.py")
    script_file.write_text(RECORDER_SCRIPT)

    # Clear log
    if TRAJ_LOG.exists():
        TRAJ_LOG.unlink()

    _recorder_proc = subprocess.Popen(
        ["python3", str(script_file), GZ_WORLD, str(TRAJ_LOG)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid
    )
    time.sleep(2.0)

    if TRAJ_LOG.exists() and TRAJ_LOG.stat().st_size > 0:
        print("  Recorder: OK")
    else:
        print("  [WARN] Recorder may not be recording yet")


def stop_recorder():
    global _recorder_proc
    if _recorder_proc and _recorder_proc.poll() is None:
        try:
            os.killpg(os.getpgid(_recorder_proc.pid), signal.SIGTERM)
        except Exception:
            pass
    subprocess.run("pkill -f tracking_recorder", shell=True, capture_output=True, timeout=3)
    _recorder_proc = None


def read_trajectory(t_start: float, t_end: float) -> List[dict]:
    points = []
    if not TRAJ_LOG.exists():
        return points
    with open(TRAJ_LOG) as f:
        for line in f:
            try:
                p = json.loads(line.strip())
                if t_start <= p["t"] <= t_end:
                    points.append(p)
            except (json.JSONDecodeError, KeyError):
                continue
    return points


# ─── ROS2 Helpers ─────────────────────────────────────────────────────────────

def ros2_cmd(cmd: str, timeout: int = 15):
    full_cmd = f"source /opt/ros/jazzy/setup.bash && source {WORKSPACE_DIR}/install/setup.bash && {cmd}"
    try:
        return subprocess.run(full_cmd, shell=True, executable='/bin/bash',
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        r = subprocess.CompletedProcess(full_cmd, 1)
        r.stdout, r.stderr = "", "timeout"
        return r


def set_dwb_speed(speed: float) -> bool:
    if speed <= 0:
        return True
    for param in ["FollowPath.max_vel_x", "FollowPath.max_speed_xy"]:
        r = ros2_cmd(f"ros2 param set /controller_server {param} {speed}")
        if "timeout" in r.stderr:
            return False
    r = ros2_cmd("ros2 param get /controller_server FollowPath.max_vel_x")
    print(f"    DWB max_vel_x → {speed} ({r.stdout.strip()})")
    return True


def restore_dwb_speed():
    set_dwb_speed(0.22)


# ─── Simulation Management ───────────────────────────────────────────────────

def kill_all():
    """Kill Gazebo, Nav2, relays."""
    for pat in ["gz sim", "ruby.*gz", "parameter_bridge", "ros_gz", "image_bridge",
                "robot_state_publisher", "ekf_node", "navigation.launch",
                "relay /odom", "relay /scan", "relay /cmd_vel",
                "static_map_odom_tf", "tracking_recorder",
                "bt_navigator", "controller_server", "planner_server"]:
        subprocess.run(f"pkill -9 -f '{pat}'", shell=True, capture_output=True, timeout=5)
    time.sleep(3)


def start_gazebo() -> bool:
    """Start Gazebo with empty world."""
    kill_all()
    time.sleep(2)

    cmd = (f"source /opt/ros/jazzy/setup.bash && source {WORKSPACE_DIR}/install/setup.bash && "
           f"ros2 launch mobile_manip_moveit_config mobile_manipulator.launch.py "
           f"use_sim_time:=true world:=empty.sdf headless:=true yaw:=0")

    log = open('/tmp/gazebo_calib.log', 'w')
    proc = subprocess.Popen(cmd, shell=True, executable='/bin/bash',
                            stdout=log, stderr=log, preexec_fn=os.setsid)

    # Wait for topics
    for i in range(90):
        r = ros2_cmd("ros2 topic list 2>/dev/null | grep /odom_real")
        if '/odom_real' in r.stdout:
            r2 = ros2_cmd("timeout 5 ros2 topic echo /scan_real --once 2>/dev/null | wc -l")
            lines = int(r2.stdout.strip() or '0')
            if lines > 3:
                print(f"  Gazebo ready ({i*2}s)")

                # Start relays
                for src, dst in [("/odom_real", "/odom"), ("/scan_real", "/scan"),
                                 ("/cmd_vel_nav", "/cmd_vel")]:
                    subprocess.Popen(
                        f"source /opt/ros/jazzy/setup.bash && source {WORKSPACE_DIR}/install/setup.bash && "
                        f"ros2 run topic_tools relay {src} {dst}",
                        shell=True, executable='/bin/bash',
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        preexec_fn=os.setsid
                    )
                time.sleep(2)
                return True
        time.sleep(2)

    print("  [ERROR] Gazebo timeout")
    return False


def start_nav2() -> bool:
    """Start Nav2 without AMCL."""
    cmd = (f"source /opt/ros/jazzy/setup.bash && source {WORKSPACE_DIR}/install/setup.bash && "
           f"ros2 launch mobile_manip_moveit_config navigation.launch.py "
           f"use_sim_time:=true use_amcl:=false rviz:=false")

    proc = subprocess.Popen(cmd, shell=True, executable='/bin/bash',
                            stdout=open('/tmp/nav2_calib.log', 'w'),
                            stderr=subprocess.STDOUT, preexec_fn=os.setsid)

    for i in range(60):
        r = ros2_cmd("ros2 action list 2>/dev/null | grep navigate_to_pose")
        if "navigate_to_pose" in r.stdout:
            print(f"  Nav2 ready ({i*5}s)")
            return True
        time.sleep(5)

    print("  [ERROR] Nav2 timeout")
    return False


def teleport_robot(x: float, y: float, theta: float = 0.0) -> bool:
    """Teleport robot and fix map→odom TF."""
    qz = math.sin(theta / 2)
    qw = math.cos(theta / 2)

    # Cancel goals + stop
    ros2_cmd("ros2 topic pub --once /navigate_to_pose/_action/cancel_goal "
             "action_msgs/msg/CancelGoal '{}' 2>/dev/null", timeout=10)
    ros2_cmd("ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "
             "'{linear: {x: 0.0}, angular: {z: 0.0}}' 2>/dev/null", timeout=10)
    time.sleep(0.5)

    # Gazebo teleport
    gz_cmd = (f'gz service -s /world/{GZ_WORLD}/set_pose '
              f'--reqtype gz.msgs.Pose --reptype gz.msgs.Boolean --timeout 2000 '
              f'--req \'name: "mobile_manip", position: {{x: {x}, y: {y}, z: 0.1}}, '
              f'orientation: {{x: 0.0, y: 0.0, z: {qz}, w: {qw}}}\'')
    subprocess.run(gz_cmd, shell=True, capture_output=True, timeout=5)
    time.sleep(1.5)

    # Fix map→odom TF
    r = ros2_cmd("timeout 3 ros2 topic echo /odom --once 2>/dev/null")
    if 'position:' not in r.stdout:
        print("    [WARN] Cannot read odom")
        return False

    odom_x = odom_y = odom_qz = odom_qw = None
    in_pos = in_orient = False
    for line in r.stdout.split('\n'):
        s = line.strip()
        if s == 'position:':
            in_pos, in_orient = True, False
        elif s == 'orientation:':
            in_orient, in_pos = True, False
        elif in_pos:
            if s.startswith('x:') and odom_x is None: odom_x = float(s.split(':')[1])
            elif s.startswith('y:') and odom_y is None: odom_y = float(s.split(':')[1])
            elif s.startswith('z:'): in_pos = False
        elif in_orient:
            if s.startswith('z:') and odom_qz is None: odom_qz = float(s.split(':')[1])
            elif s.startswith('w:') and odom_qw is None:
                odom_qw = float(s.split(':')[1])
                in_orient = False

    if odom_x is None:
        return False

    odom_yaw = 2.0 * math.atan2(odom_qz or 0, odom_qw or 1)
    tf_yaw = theta - odom_yaw
    tf_x = x - (math.cos(tf_yaw) * odom_x - math.sin(tf_yaw) * odom_y)
    tf_y = y - (math.sin(tf_yaw) * odom_x + math.cos(tf_yaw) * odom_y)

    subprocess.run("pkill -f static_map_odom_tf", shell=True, capture_output=True, timeout=3)
    time.sleep(0.3)

    tf_cmd = (f"source /opt/ros/jazzy/setup.bash && "
              f"ros2 run tf2_ros static_transform_publisher "
              f"--ros-args -r __node:=static_map_odom_tf "
              f"-p use_sim_time:=true "
              f"-- {tf_x} {tf_y} 0 {tf_yaw} 0 0 map odom")
    subprocess.Popen(tf_cmd, shell=True, executable='/bin/bash',
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     preexec_fn=os.setsid)
    time.sleep(1.5)

    # Clear costmaps
    ros2_cmd("ros2 service call /local_costmap/clear_entirely_local_costmap "
             "nav2_msgs/srv/ClearEntireCostmap '{}' 2>/dev/null")
    ros2_cmd("ros2 service call /global_costmap/clear_entirely_global_costmap "
             "nav2_msgs/srv/ClearEntireCostmap '{}' 2>/dev/null")
    time.sleep(0.5)

    return True


def full_restart() -> bool:
    """Full Gazebo + Nav2 restart."""
    print("\n  [RESTART] Full simulation restart...")
    stop_recorder()

    if not start_gazebo():
        return False
    if not start_nav2():
        return False

    start_recorder()
    time.sleep(2)
    return True


# ─── Trial Runners ────────────────────────────────────────────────────────────

def run_position_hold_trial(seed: int) -> TrialResult:
    print(f"  [v=0.00] Seed {seed}: position-hold...")

    set_dwb_speed(0.22)
    teleport_robot(3.0, 0.0)
    time.sleep(1.0)

    t_start = time.time()
    time.sleep(10.0)
    t_end = time.time()

    points = read_trajectory(t_start, t_end)
    if len(points) < 5:
        return TrialResult(0.0, seed, 0, 0, 0, 0, 0, 0, len(points), False)

    ys = np.array([p["y"] for p in points])
    devs = np.abs(ys - ys[0])

    r = TrialResult(0.0, seed, 0, 0,
                    float(np.max(devs)), float(np.mean(devs)),
                    float(np.percentile(devs, 95)), float(np.percentile(devs, 99)),
                    len(points), True)
    print(f"    max|y|={r.max_lateral_dev:.4f}m, n={len(points)}")
    return r


def run_nav_trial(speed: float, seed: int) -> TrialResult:
    print(f"  [v={speed:.2f}] Seed {seed}: {NAV_DISTANCE}m straight...")

    if not set_dwb_speed(speed):
        return TrialResult(speed, seed, 0, 0, 0, 0, 0, 0, 0, False)

    if not teleport_robot(0.0, 0.0, theta=0.0):
        return TrialResult(speed, seed, 0, 0, 0, 0, 0, 0, 0, False)

    goal_x = NAV_DISTANCE
    t_start = time.time()

    # Send goal async
    goal_cmd = (
        f"source /opt/ros/jazzy/setup.bash && source {WORKSPACE_DIR}/install/setup.bash && "
        f"ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose "
        f"\"{{pose: {{header: {{frame_id: 'map'}}, "
        f"pose: {{position: {{x: {goal_x}, y: 0.0, z: 0.0}}, "
        f"orientation: {{w: 1.0}}}}}}}}\" --feedback"
    )

    timeout = min(NAV_DISTANCE / max(speed, 0.01) + 20, 90)
    nav_proc = subprocess.Popen(
        goal_cmd, shell=True, executable='/bin/bash',
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        preexec_fn=os.setsid
    )

    try:
        stdout, stderr = nav_proc.communicate(timeout=timeout)
        output = (stdout.decode() if stdout else '') + (stderr.decode() if stderr else '')
        nav_success = "SUCCEEDED" in output
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(nav_proc.pid), signal.SIGTERM)
        except Exception:
            pass
        nav_success = False
        print(f"    Timed out ({timeout:.0f}s)")

    t_end = time.time()

    # Read trajectory
    points = read_trajectory(t_start, t_end)
    meas = [p for p in points if MEASURE_X_MIN <= p["x"] <= MEASURE_X_MAX]

    if len(meas) < 5:
        # Fallback: any forward motion
        meas = [p for p in points if p["x"] > 0.3]
        if len(meas) >= 5:
            print(f"    Using fallback window: {len(meas)} pts (x>{0.3})")

    if len(meas) < 3:
        print(f"    Insufficient data: {len(meas)} meas pts ({len(points)} total)")
        return TrialResult(speed, seed, 0, 0, 0, 0, 0, 0, len(meas), nav_success)

    ys = np.abs(np.array([p["y"] for p in meas]))

    # Compute actual speed
    v_list = []
    for i in range(1, len(meas)):
        dt = meas[i]["t"] - meas[i-1]["t"]
        if dt > 0.001:
            v_list.append(abs((meas[i]["x"] - meas[i-1]["x"]) / dt))

    v_max = float(np.max(v_list)) if v_list else 0.0
    v_mean = float(np.mean(v_list)) if v_list else 0.0

    r = TrialResult(speed, seed, v_max, v_mean,
                    float(np.max(ys)), float(np.mean(ys)),
                    float(np.percentile(ys, 95)), float(np.percentile(ys, 99)),
                    len(meas), nav_success)

    status = "OK" if nav_success else "FAIL"
    print(f"    [{status}] max|y|={r.max_lateral_dev:.4f}m, p99={r.p99_lateral_dev:.4f}m, "
          f"v={v_mean:.3f}m/s, n={len(meas)}")
    return r


# ─── Analysis ─────────────────────────────────────────────────────────────────

def fit_linear(speeds, devs):
    if len(speeds) < 3:
        return 0.0, 0.0, 0.0
    A = np.vstack([np.ones_like(speeds), speeds]).T
    res = np.linalg.lstsq(A, devs, rcond=None)
    e_0, c_1 = res[0]
    pred = e_0 + c_1 * speeds
    ss_res = np.sum((devs - pred)**2)
    ss_tot = np.sum((devs - np.mean(devs))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return float(e_0), float(c_1), float(r2)


def generate_report(results: List[TrialResult], output_dir: Path):
    # Filter: successful + reasonable deviation (< 0.5m, anything larger is nav failure)
    valid = [r for r in results if r.nav_success and r.n_samples >= 5
             and r.max_lateral_dev < 0.5]

    if len(valid) < 5:
        print(f"[ERROR] Only {len(valid)} valid results (need 5+)")
        # Show what we got
        for r in results:
            print(f"  v={r.speed_setting:.2f} s={r.seed} max|y|={r.max_lateral_dev:.4f} "
                  f"ok={r.nav_success} n={r.n_samples}")
        return

    groups = {}
    for r in valid:
        groups.setdefault(r.speed_setting, []).append(r)

    print("\n" + "="*70)
    print("TRACKING ERROR CALIBRATION RESULTS")
    print("="*70)
    print(f"  Valid trials: {len(valid)}/{len(results)}")
    print(f"{'v_set':>7} {'N':>4} {'v_actual':>9} {'max|y| (m)':>11} {'±σ':>8} "
          f"{'p99|y|':>9} {'mean|y|':>10}")
    print("-"*70)

    for speed in sorted(groups.keys()):
        g = groups[speed]
        print(f"{speed:>7.2f} {len(g):>4d} "
              f"{np.mean([r.actual_mean_speed for r in g]):>9.3f} "
              f"{np.mean([r.max_lateral_dev for r in g]):>11.4f} "
              f"{np.std([r.max_lateral_dev for r in g]):>8.4f} "
              f"{np.mean([r.p99_lateral_dev for r in g]):>9.4f} "
              f"{np.mean([r.mean_lateral_dev for r in g]):>10.4f}")

    # Fits
    speeds_all = np.array([r.actual_mean_speed for r in valid])
    max_all = np.array([r.max_lateral_dev for r in valid])
    p99_all = np.array([r.p99_lateral_dev for r in valid])

    e0_max, c1_max, r2_max = fit_linear(speeds_all, max_all)
    e0_p99, c1_p99, r2_p99 = fit_linear(speeds_all, p99_all)

    print(f"\nFIT: e_track(v) = e_0 + c_1·v")
    print(f"  max|y|: e_0={e0_max:.4f}m, c_1={c1_max:.4f}s (R²={r2_max:.3f})")
    print(f"  p99|y|: e_0={e0_p99:.4f}m, c_1={c1_p99:.4f}s (R²={r2_p99:.3f})")
    print(f"\n  At v=0.5: max→{e0_max+c1_max*0.5:.4f}m, p99→{e0_p99+c1_p99*0.5:.4f}m")
    print(f"  Current:  0.03+0.04·0.5 = 0.050m")

    rec_e0 = max(e0_max, 0.001)
    rec_c1 = max(c1_max, 0.0)
    print(f"\n  RECOMMENDED: e_0={rec_e0:.4f}m, c_1={rec_c1:.4f}s")

    # Save JSON
    cal = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "method": "Nav2 DWB, Gazebo empty world, 6m straight-line, ground truth pose",
        "n_valid": len(valid), "n_total": len(results),
        "speeds_tested": sorted(groups.keys()),
        "fit_max": {"e_0": e0_max, "c_1": c1_max, "r_squared": r2_max},
        "fit_p99": {"e_0": e0_p99, "c_1": c1_p99, "r_squared": r2_p99},
        "recommendation": {"e_0": rec_e0, "c_1": rec_c1},
        "per_speed": {str(s): {
            "n": len(g), "v_actual": float(np.mean([r.actual_mean_speed for r in g])),
            "max_dev_mean": float(np.mean([r.max_lateral_dev for r in g])),
            "max_dev_std": float(np.std([r.max_lateral_dev for r in g])),
            "p99_dev_mean": float(np.mean([r.p99_lateral_dev for r in g])),
        } for s, g in sorted(groups.items())},
        "raw_results": [asdict(r) for r in valid]
    }

    with open(output_dir / "tracking_error_calibration.json", 'w') as f:
        json.dump(cal, f, indent=2)

    # Figure
    try:
        _generate_figure(valid, groups, e0_max, c1_max, e0_p99, c1_p99, output_dir)
    except Exception as e:
        print(f"  [WARN] Figure failed: {e}")


def _generate_figure(results, groups, e0_max, c1_max, e0_p99, c1_p99, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))

    speeds = [r.actual_mean_speed for r in results]
    max_devs = [r.max_lateral_dev for r in results]
    p99_devs = [r.p99_lateral_dev for r in results]

    ax.scatter(speeds, max_devs, c='#d62728', alpha=0.4, s=30, zorder=3, label='max |y|')
    ax.scatter(speeds, p99_devs, c='#1f77b4', alpha=0.4, s=30, zorder=3, label='p99 |y|')

    # Per-speed means
    sorted_s = sorted(groups.keys())
    mv = [np.mean([r.actual_mean_speed for r in groups[s]]) for s in sorted_s]
    mm = [np.mean([r.max_lateral_dev for r in groups[s]]) for s in sorted_s]
    ms = [np.std([r.max_lateral_dev for r in groups[s]]) for s in sorted_s]
    ax.errorbar(mv, mm, yerr=ms, fmt='D', color='#d62728', ms=8, capsize=4, lw=2,
                zorder=5, label='max |y| mean±σ')

    v = np.linspace(0, 0.55, 100)
    ax.plot(v, np.maximum(e0_max + c1_max * v, 0), 'r-', lw=2,
            label=f'max: {e0_max:.3f}+{c1_max:.3f}v')
    ax.plot(v, np.maximum(e0_p99 + c1_p99 * v, 0), 'b--', lw=2,
            label=f'p99: {e0_p99:.3f}+{c1_p99:.3f}v')
    ax.plot(v, 0.03 + 0.04 * v, 'k:', lw=1.5, alpha=0.5,
            label='default: 0.030+0.040v')

    ax.set_xlabel('Speed $v$ (m/s)', fontsize=13)
    ax.set_ylabel('Lateral deviation (m)', fontsize=13)
    ax.set_title('DWB Path-Tracking Error Calibration', fontsize=14)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 0.55)
    ax.set_ylim(bottom=0)
    plt.tight_layout()

    for ext in ['png', 'pdf']:
        fig.savefig(output_dir / f"tracking_error_calibration.{ext}",
                    dpi=150, bbox_inches='tight')
    print(f"  Figure saved")
    plt.close(fig)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--speeds', type=str, default=None)
    parser.add_argument('--output-dir', type=str,
                        default='experiment_results/tracking_calibration')
    args = parser.parse_args()

    speeds = SPEEDS
    if args.speeds:
        speeds = [float(s) for s in args.speeds.split(',')]

    output_dir = Path(WORKSPACE_DIR) / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    total = len(speeds) * args.seeds
    print("="*70)
    print(f"TRACKING ERROR CALIBRATION — {total} trials")
    print(f"  Speeds: {speeds}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Restart every {RESTART_EVERY} nav trials")
    print("="*70)

    # Initial start
    if not full_restart():
        print("[FATAL] Cannot start simulation")
        sys.exit(1)

    all_results: List[TrialResult] = []
    nav_trial_count = 0

    try:
        trial_num = 0
        for speed in speeds:
            for seed in range(args.seeds):
                trial_num += 1
                print(f"\n--- Trial {trial_num}/{total} ---")

                if speed == 0.0:
                    r = run_position_hold_trial(seed)
                else:
                    # Periodic restart
                    if nav_trial_count > 0 and nav_trial_count % RESTART_EVERY == 0:
                        if not full_restart():
                            print("[ERROR] Restart failed, skipping")
                            continue
                    nav_trial_count += 1
                    r = run_nav_trial(speed, seed)

                all_results.append(r)

                # Save partial
                partial = {"raw_results": [asdict(r) for r in all_results]}
                with open(output_dir / "calibration_partial.json", 'w') as f:
                    json.dump(partial, f, indent=2)

                time.sleep(1)

    except KeyboardInterrupt:
        print("\n[INTERRUPTED]")
    finally:
        restore_dwb_speed()
        stop_recorder()

    generate_report(all_results, output_dir)
    print("\nDone!")


if __name__ == "__main__":
    main()
