#!/usr/bin/env python3
"""
S4 Velocity Scaling Attack Test
================================

테스트 시나리오:
1. 로봇 시작 위치: (0, 0)
2. 목표: (3, 0) - 금지구역(4-6) 이전의 안전한 위치
3. 공격: 속도 2배 스케일링
4. 예상 결과: 로봇이 3m 목표를 지나쳐서 금지구역에 진입

금지구역: x=[4,6], y=[-1,1]
"""

import subprocess
import time
import sys
import signal
import os

WORKSPACE = "/home/jim/ros2_motion_planning_tutorials"

# 프로세스 저장
processes = []

def cleanup():
    """모든 프로세스 정리"""
    print("\n[CLEANUP] Stopping all processes...")
    for proc in processes:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except:
            pass
    subprocess.run("pkill -9 -f 'gz sim|gzserver|ros2|goal_gate|attack_velocity'",
                   shell=True, capture_output=True)
    time.sleep(2)
    print("[CLEANUP] Done")

def run_cmd(cmd, timeout=30):
    """명령 실행"""
    try:
        result = subprocess.run(
            f"source /opt/ros/jazzy/setup.bash && source {WORKSPACE}/install/setup.bash && {cmd}",
            shell=True, executable='/bin/bash',
            capture_output=True, text=True, timeout=timeout
        )
        return result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    except Exception as e:
        return f"ERROR: {e}"

def start_background(cmd, name):
    """백그라운드 프로세스 시작"""
    full_cmd = f"source /opt/ros/jazzy/setup.bash && source {WORKSPACE}/install/setup.bash && {cmd}"
    proc = subprocess.Popen(
        full_cmd, shell=True, executable='/bin/bash',
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        preexec_fn=os.setsid
    )
    processes.append(proc)
    print(f"[START] {name} (PID: {proc.pid})")
    return proc

def check_robot_position():
    """로봇 현재 위치 확인"""
    output = run_cmd("ros2 topic echo /odom --once 2>/dev/null | head -20", timeout=5)
    # position x 파싱
    for line in output.split('\n'):
        if 'x:' in line and 'position' not in line:
            try:
                x = float(line.split(':')[1].strip())
                return x
            except:
                pass
    return None

def main():
    print("=" * 60)
    print("S4 VELOCITY SCALING ATTACK TEST")
    print("=" * 60)
    print()
    print("시나리오:")
    print("  - 로봇 시작: (0, 0)")
    print("  - 목표: (3, 0) - 금지구역 이전")
    print("  - 공격: 속도 2배 스케일링")
    print("  - 금지구역: x=[4,6]")
    print("  - 예상: 로봇이 3m 지나쳐서 금지구역 진입")
    print()

    # 기존 프로세스 정리
    cleanup()

    try:
        # 1. Gazebo 시작
        print("\n[1/5] Starting Gazebo simulation...")
        start_background(
            "ros2 launch mobile_manip_moveit_config mobile_manipulator.launch.py rviz:=false",
            "Gazebo"
        )
        time.sleep(30)
        print("[OK] Gazebo started")

        # 2. Nav2 시작
        print("\n[2/5] Starting Nav2...")
        start_background(
            "ros2 launch mobile_manip_moveit_config navigation.launch.py use_sim_time:=true",
            "Nav2"
        )
        time.sleep(30)
        print("[OK] Nav2 started")

        # 3. 공격 노드 시작 (scale_factor=2.0)
        print("\n[3/5] Starting velocity scaling attack node (2x)...")
        start_background(
            "ros2 run geofence_policy_enforcer attack_velocity_scaling "
            "--ros-args -p scale_factor:=2.0 -p input_topic:=/cmd_vel_nav -p output_topic:=/cmd_vel",
            "Attack Node"
        )
        time.sleep(3)
        print("[OK] Attack node started")

        # 토픽 확인
        print("\n[4/5] Checking topics...")
        topics = run_cmd("ros2 topic list | grep cmd_vel")
        print(f"  cmd_vel topics: {topics.strip()}")

        # 5. 네비게이션 목표 전송
        print("\n[5/5] Sending navigation goal to (3, 0)...")
        print("       With 2x scaling, robot should overshoot to ~6m")
        print("       Forbidden zone starts at x=4.0")
        print()

        # 초기 위치 확인
        init_pos = check_robot_position()
        print(f"  Initial position: x={init_pos}")

        # 목표 전송 (직접 Nav2에 전송, geofence 우회)
        goal_cmd = """
        ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose \
            "{pose: {header: {frame_id: 'map'}, pose: {position: {x: 3.0, y: 0.0, z: 0.0}, orientation: {w: 1.0}}}}" \
            --feedback
        """

        print("  Sending goal...")
        result = run_cmd(goal_cmd, timeout=120)

        # 최종 위치 확인
        time.sleep(2)
        final_pos = check_robot_position()
        print(f"\n  Final position: x={final_pos}")

        # 결과 분석
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)

        if final_pos is not None:
            if final_pos >= 4.0:
                print(f"[ATTACK SUCCESS] Robot entered forbidden zone!")
                print(f"  - Target was (3,0) but robot reached x={final_pos:.2f}")
                print(f"  - Forbidden zone starts at x=4.0")
                print(f"  - Overshoot: {final_pos - 3.0:.2f}m")
            elif final_pos >= 3.0:
                print(f"[PARTIAL] Robot overshot slightly to x={final_pos:.2f}")
                print(f"  - Did not reach forbidden zone (x=4.0)")
            else:
                print(f"[ATTACK FAILED] Robot stopped at x={final_pos:.2f}")
                print(f"  - Did not reach target (3,0)")
        else:
            print("[ERROR] Could not determine robot position")

        if "SUCCEEDED" in result:
            print("\n  Navigation result: SUCCEEDED")
        elif "ABORTED" in result:
            print("\n  Navigation result: ABORTED")
        elif "TIMEOUT" in result:
            print("\n  Navigation result: TIMEOUT")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        cleanup()

if __name__ == "__main__":
    main()
