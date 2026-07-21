#!/usr/bin/env python3
"""
S5' Repeated Comparison Test

Run multiple trials to get statistically meaningful results.
"""

import subprocess
import time
import os
import sys
import json

WORKSPACE = "/home/jim/ros2_motion_planning_tutorials"
FORBIDDEN_ZONE_X = 4.0
GOAL_X = 3.8

# Attack parameters - slightly reduced from v3 for more consistent results
ATTACK_PARAMS = {
    'rotation': 0.10,  # Slightly less than 0.12
    'scale': 1.15,     # Slightly less than 1.20
    'noise': 0.02      # Slightly less than 0.03
}

NUM_TRIALS = 10

def run_cmd(cmd: str, timeout: int = 10) -> str:
    try:
        result = subprocess.run(
            cmd, shell=True, executable='/bin/bash',
            capture_output=True, text=True, timeout=timeout
        )
        return result.stdout + result.stderr
    except:
        return ""

def run_bg(cmd: str, log_file: str = None):
    if log_file:
        f = open(log_file, 'w')
        return subprocess.Popen(
            cmd, shell=True, executable='/bin/bash',
            stdout=f, stderr=subprocess.STDOUT, preexec_fn=os.setsid
        )
    return subprocess.Popen(
        cmd, shell=True, executable='/bin/bash',
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid
    )

def source_ros():
    return f"source /opt/ros/jazzy/setup.bash && source {WORKSPACE}/install/setup.bash"

def cleanup():
    patterns = [
        "gz sim", "gzserver", "gzclient", "ruby.*gz",
        "nav2_", "controller_server", "planner_server",
        "behavior_server", "bt_navigator", "lifecycle_manager",
        "goal_gate", "cmd_vel_guard", "hardware_geofence",
        "attack_", "relay", "ros_gz", "parameter_bridge", "amcl"
    ]
    for p in patterns:
        subprocess.run(f"pkill -9 -f '{p}'", shell=True, capture_output=True, timeout=5)
    time.sleep(2)

def activate_nav2_nodes():
    nodes = [
        'map_server', 'amcl',
        'controller_server', 'smoother_server', 'planner_server',
        'behavior_server', 'bt_navigator', 'waypoint_follower',
        'velocity_smoother', 'collision_monitor'
    ]
    for node in nodes:
        run_cmd(f"{source_ros()} && ros2 lifecycle set {node} configure 2>&1", timeout=5)
    time.sleep(2)
    for node in nodes:
        run_cmd(f"{source_ros()} && ros2 lifecycle set {node} activate 2>&1", timeout=5)
    time.sleep(3)

def set_initial_pose(x=0.0, y=0.0, yaw=0.0):
    import math
    qz = math.sin(yaw / 2.0)
    qw = math.cos(yaw / 2.0)
    cmd = f'''{source_ros()} && ros2 topic pub --once /initialpose geometry_msgs/msg/PoseWithCovarianceStamped "{{
      header: {{frame_id: 'map'}},
      pose: {{pose: {{position: {{x: {x}, y: {y}, z: 0.0}}, orientation: {{z: {qz}, w: {qw}}}}},
        covariance: [0.25,0,0,0,0,0, 0,0.25,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0.07]}}
    }}"'''
    return run_cmd(cmd, timeout=10)

def get_robot_position():
    output = run_cmd(f"{source_ros()} && ros2 topic echo /odom --once", timeout=5)
    if 'position:' in output:
        lines = output.split('\n')
        for i, line in enumerate(lines):
            if 'position:' in line and i+2 < len(lines):
                try:
                    x = float(lines[i+1].split(':')[1].strip())
                    y = float(lines[i+2].split(':')[1].strip())
                    return x, y
                except:
                    pass
    return None, None

def send_nav_goal(x: float, y: float, use_gate: bool = False):
    """Send navigation goal. If use_gate=True, send via goal gate."""
    topic = '/navigate_to_pose_safe' if use_gate else '/navigate_to_pose'
    cmd = f'''{source_ros()} && ros2 action send_goal {topic} nav2_msgs/action/NavigateToPose "{{
      pose: {{header: {{frame_id: 'map'}}, pose: {{position: {{x: {x}, y: {y}, z: 0.0}}, orientation: {{w: 1.0}}}}}}
    }}"'''
    return run_bg(cmd, '/tmp/nav_goal_repeated.log')

def run_single_trial(defense: str, trial: int, procs: list) -> dict:
    """Run single trial."""
    print(f"\n--- Trial {trial+1}/{NUM_TRIALS} for {defense} ---")

    # 1. Gazebo
    print("[1] Starting Gazebo...")
    gz_cmd = f"{source_ros()} && ros2 launch mobile_manip_moveit_config mobile_manipulator.launch.py use_sim_time:=true headless:=true"
    gz_proc = run_bg(gz_cmd, f'/tmp/gz_rep_{defense}_{trial}.log')
    procs.append(gz_proc)
    time.sleep(35)

    # 2. Odom relay
    print("[2] Starting odom relay...")
    odom_cmd = f"{source_ros()} && ros2 run topic_tools relay /odom_real /odom"
    odom_proc = run_bg(odom_cmd)
    procs.append(odom_proc)
    time.sleep(2)

    # 3. LIDAR spoofing attack
    print("[3] Starting LIDAR spoofing attack...")
    scan_cmd = f"{source_ros()} && ros2 run geofence_policy_enforcer attack_scan_spoofing --ros-args -p rotation_offset:={ATTACK_PARAMS['rotation']} -p range_scale:={ATTACK_PARAMS['scale']} -p noise_stddev:={ATTACK_PARAMS['noise']}"
    scan_proc = run_bg(scan_cmd, f'/tmp/attack_rep_{defense}_{trial}.log')
    procs.append(scan_proc)
    time.sleep(3)

    # 4. Navigation
    print("[4] Starting Navigation...")
    nav_cmd = f"{source_ros()} && ros2 launch mobile_manip_moveit_config navigation.launch.py use_sim_time:=true rviz:=false"
    nav_proc = run_bg(nav_cmd, f'/tmp/nav_rep_{defense}_{trial}.log')
    procs.append(nav_proc)
    time.sleep(35)

    # 5. Activate Nav2
    print("[5] Activating Nav2 nodes...")
    activate_nav2_nodes()

    # 6. Safety method - handle each method's original architecture
    use_gate = False

    if defense == 'none':
        # No defense - direct relay
        print("[6] No defense (direct relay)...")
        guard_cmd = f"{source_ros()} && ros2 run topic_tools relay /cmd_vel_nav /cmd_vel"
        guard_proc = run_bg(guard_cmd, f'/tmp/guard_rep_{defense}_{trial}.log')
        procs.append(guard_proc)

    elif defense == 'selp':
        # SELP: Goal Gate ONLY (planning-time), NO runtime cmd_vel check
        print("[6] SELP: Goal Gate only (no runtime check)...")
        gate_cmd = f"{source_ros()} && ros2 run geofence_policy_enforcer goal_gate_node --ros-args -p safety_method:=selp"
        gate_proc = run_bg(gate_cmd, f'/tmp/gate_rep_{defense}_{trial}.log')
        procs.append(gate_proc)
        time.sleep(3)
        # Direct relay for cmd_vel (NO runtime safety check)
        guard_cmd = f"{source_ros()} && ros2 run topic_tools relay /cmd_vel_nav /cmd_vel"
        guard_proc = run_bg(guard_cmd, f'/tmp/guard_rep_{defense}_{trial}.log')
        procs.append(guard_proc)
        use_gate = True

    elif defense in ['cbf', 'ssm']:
        # CBF/SSM: Cmd Vel Guard ONLY (runtime check, no goal gate)
        print(f"[6] {defense.upper()}: Cmd Vel Guard only (runtime check)...")
        guard_cmd = f"{source_ros()} && ros2 run geofence_policy_enforcer cmd_vel_guard_node --ros-args -p safety_method:={defense} -p input_topic:=/cmd_vel_nav -p output_topic:=/cmd_vel"
        guard_proc = run_bg(guard_cmd, f'/tmp/guard_rep_{defense}_{trial}.log')
        procs.append(guard_proc)

    elif defense == 'geofence':
        # Geofence: FULL STACK (Goal Gate + Cmd Vel Guard)
        print("[6] Geofence: Full stack (Goal Gate + Cmd Vel Guard)...")
        gate_cmd = f"{source_ros()} && ros2 run geofence_policy_enforcer goal_gate_node --ros-args -p safety_method:=geofence"
        gate_proc = run_bg(gate_cmd, f'/tmp/gate_rep_{defense}_{trial}.log')
        procs.append(gate_proc)
        time.sleep(3)
        guard_cmd = f"{source_ros()} && ros2 run geofence_policy_enforcer cmd_vel_guard_node --ros-args -p safety_method:=geofence -p input_topic:=/cmd_vel_nav -p output_topic:=/cmd_vel"
        guard_proc = run_bg(guard_cmd, f'/tmp/guard_rep_{defense}_{trial}.log')
        procs.append(guard_proc)
        use_gate = True

    time.sleep(3)

    # 7. Initial pose
    print("[7] Setting initial pose...")
    set_initial_pose(0.0, 0.0, 0.0)
    time.sleep(3)

    # 8. Send goal (via gate if applicable)
    gate_str = " via gate" if use_gate else ""
    print(f"[8] Sending goal ({GOAL_X}, 0.0){gate_str}...")
    goal_proc = send_nav_goal(GOAL_X, 0.0, use_gate=use_gate)
    procs.append(goal_proc)

    # 9. Monitor (60s)
    print("[9] Monitoring (60s)...")
    start_time = time.time()
    max_x = 0.0
    violated = False

    while time.time() - start_time < 60:
        x, y = get_robot_position()
        if x is not None:
            max_x = max(max_x, x)
            status = "VIOLATION!" if x >= FORBIDDEN_ZONE_X else ""
            print(f"  t={time.time()-start_time:.0f}s: x={x:.2f}m {status}")

            if x >= FORBIDDEN_ZONE_X:
                violated = True
                break
        time.sleep(5)

    return {
        'defense': defense,
        'trial': trial,
        'max_x': max_x,
        'violated': violated or max_x >= FORBIDDEN_ZONE_X
    }

def main():
    print("=" * 70)
    print(f"S5' REPEATED COMPARISON TEST ({NUM_TRIALS} trials each)")
    print("=" * 70)
    print(f"Attack: rotation={ATTACK_PARAMS['rotation']}, scale={ATTACK_PARAMS['scale']}, noise={ATTACK_PARAMS['noise']}")
    print(f"Goal: {GOAL_X}m | Forbidden: x>={FORBIDDEN_ZONE_X}m")
    print("=" * 70)

    defenses = ['none', 'selp', 'cbf', 'ssm', 'geofence']
    all_results = []

    for defense in defenses:
        print(f"\n{'='*60}")
        print(f"TESTING: {defense.upper()} ({NUM_TRIALS} trials)")
        print(f"{'='*60}")

        for trial in range(NUM_TRIALS):
            cleanup()
            procs = []
            try:
                result = run_single_trial(defense, trial, procs)
                all_results.append(result)
                status = "VIOLATED!" if result['violated'] else "safe"
                print(f"[TRIAL {trial+1}] {defense}: Max X={result['max_x']:.2f}m, {status}")
            except Exception as e:
                print(f"[ERROR] Trial {trial+1}: {e}")
                all_results.append({
                    'defense': defense,
                    'trial': trial,
                    'max_x': 0.0,
                    'violated': False,
                    'error': str(e)
                })
            finally:
                cleanup()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY - All Trials")
    print("=" * 70)

    for defense in defenses:
        trials = [r for r in all_results if r['defense'] == defense]
        violations = sum(1 for r in trials if r['violated'])
        max_xs = [r['max_x'] for r in trials]
        avg_max_x = sum(max_xs) / len(max_xs) if max_xs else 0

        print(f"\n{defense.upper()}:")
        print(f"  Trials: {len(trials)}")
        print(f"  Violations: {violations}/{len(trials)}")
        print(f"  Avg Max X: {avg_max_x:.2f}m")
        print(f"  Max X values: {[f'{x:.2f}m' for x in max_xs]}")

    print("\n" + "=" * 70)

    # Final verdict
    none_violations = sum(1 for r in all_results if r['defense'] == 'none' and r['violated'])
    geo_violations = sum(1 for r in all_results if r['defense'] == 'geofence' and r['violated'])

    print("FINAL VERDICT:")
    if none_violations > geo_violations:
        print(f"  ✓ Geofence provides better protection!")
        print(f"    none: {none_violations}/{NUM_TRIALS} violations")
        print(f"    geofence: {geo_violations}/{NUM_TRIALS} violations")
    elif none_violations == geo_violations:
        print(f"  = Both defenses have same violation rate")
        print(f"    {none_violations}/{NUM_TRIALS} violations each")
    else:
        print(f"  ⚠ Unexpected: Geofence has more violations")
        print(f"    none: {none_violations}/{NUM_TRIALS} violations")
        print(f"    geofence: {geo_violations}/{NUM_TRIALS} violations")

    # Save results
    with open('/tmp/s5p_repeated_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to /tmp/s5p_repeated_results.json")

    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted!")
        cleanup()
        sys.exit(1)
