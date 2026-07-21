#!/usr/bin/env python3
"""Debug script to test robot localization in Gazebo"""

import subprocess
import time
import json

def run_cmd(cmd, timeout=10):
    """Run a shell command and return output"""
    try:
        result = subprocess.run(
            f"source /opt/ros/jazzy/setup.bash && source ~/ros2_motion_planning_tutorials/install/setup.bash && {cmd}",
            shell=True, executable='/bin/bash',
            capture_output=True, text=True, timeout=timeout
        )
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except subprocess.TimeoutExpired:
        return "", "timeout", -1

def get_gazebo_pose():
    """Get robot pose from Gazebo ground truth"""
    stdout, stderr, rc = run_cmd(
        "ros2 topic echo /model/mobile_manip/pose --once 2>/dev/null | head -20",
        timeout=5
    )
    if "position:" in stdout:
        lines = stdout.split('\n')
        x = y = z = None
        for i, line in enumerate(lines):
            if 'position:' in line:
                for j in range(i+1, min(i+5, len(lines))):
                    if 'x:' in lines[j] and x is None:
                        x = float(lines[j].split(':')[1].strip())
                    elif 'y:' in lines[j] and y is None:
                        y = float(lines[j].split(':')[1].strip())
                    elif 'z:' in lines[j] and z is None:
                        z = float(lines[j].split(':')[1].strip())
                break
        return x, y
    return None, None

def get_amcl_pose():
    """Get robot pose from AMCL"""
    stdout, stderr, rc = run_cmd(
        "ros2 topic echo /amcl_pose --once 2>/dev/null | head -30",
        timeout=5
    )
    if "position:" in stdout:
        lines = stdout.split('\n')
        x = y = None
        for i, line in enumerate(lines):
            if 'position:' in line:
                for j in range(i+1, min(i+5, len(lines))):
                    if 'x:' in lines[j] and x is None:
                        x = float(lines[j].split(':')[1].strip())
                    elif 'y:' in lines[j] and y is None:
                        y = float(lines[j].split(':')[1].strip())
                break
        return x, y
    return None, None

def teleport_robot(x, y, theta=0.0):
    """Teleport robot in Gazebo"""
    import math
    qz = math.sin(theta / 2)
    qw = math.cos(theta / 2)

    cmd = f"""gz service -s /world/empty/set_pose \
        --reqtype gz.msgs.Pose --reptype gz.msgs.Boolean --timeout 2000 \
        --req 'name: "mobile_manip", position: {{x: {x}, y: {y}, z: 0.1}}, orientation: {{x: 0.0, y: 0.0, z: {qz}, w: {qw}}}'"""

    stdout, stderr, rc = run_cmd(cmd, timeout=5)
    return rc == 0, stderr

def publish_initial_pose(x, y, theta=0.0):
    """Publish initial pose to AMCL"""
    import math
    qz = math.sin(theta / 2)
    qw = math.cos(theta / 2)

    cmd = f"""ros2 topic pub --once /initialpose geometry_msgs/msg/PoseWithCovarianceStamped '{{
        header: {{frame_id: "map"}},
        pose: {{
            pose: {{
                position: {{x: {x}, y: {y}, z: 0.0}},
                orientation: {{x: 0.0, y: 0.0, z: {qz}, w: {qw}}}
            }},
            covariance: [0.01, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.01, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.01]
        }}
    }}'"""

    stdout, stderr, rc = run_cmd(cmd, timeout=5)
    return rc == 0

def check_topics():
    """Check if required topics are available"""
    stdout, stderr, rc = run_cmd("ros2 topic list", timeout=5)
    topics = stdout.split('\n')

    required = ['/scan', '/odom', '/amcl_pose', '/cmd_vel']
    missing = [t for t in required if t not in topics]

    return len(missing) == 0, missing

def main():
    print("=" * 60)
    print("Localization Debug Script")
    print("=" * 60)

    # 1. Check topics
    print("\n[1] Checking ROS2 topics...")
    ok, missing = check_topics()
    if ok:
        print("    All required topics available")
    else:
        print(f"    MISSING TOPICS: {missing}")
        return

    # 2. Get current poses
    print("\n[2] Getting current robot poses...")
    gz_x, gz_y = get_gazebo_pose()
    amcl_x, amcl_y = get_amcl_pose()

    print(f"    Gazebo ground truth: ({gz_x}, {gz_y})")
    print(f"    AMCL estimate:       ({amcl_x}, {amcl_y})")

    if gz_x is not None and amcl_x is not None:
        error = ((gz_x - amcl_x)**2 + (gz_y - amcl_y)**2)**0.5
        print(f"    Localization error:  {error:.3f}m")

    # 3. Teleport robot to origin
    print("\n[3] Teleporting robot to origin (0, 0)...")
    ok, err = teleport_robot(0.0, 0.0, 0.0)
    if ok:
        print("    Teleport command sent")
    else:
        print(f"    Teleport FAILED: {err}")

    time.sleep(2)

    # 4. Check Gazebo pose after teleport
    print("\n[4] Checking Gazebo pose after teleport...")
    gz_x, gz_y = get_gazebo_pose()
    print(f"    Gazebo position: ({gz_x}, {gz_y})")

    if gz_x is not None:
        error = (gz_x**2 + gz_y**2)**0.5
        if error > 0.5:
            print(f"    WARNING: Robot not at origin! Error: {error:.3f}m")
        else:
            print(f"    OK: Robot at origin (error: {error:.3f}m)")

    # 5. Publish initial pose
    print("\n[5] Publishing initial pose to AMCL...")
    ok = publish_initial_pose(0.0, 0.0, 0.0)
    if ok:
        print("    Initial pose published")
    else:
        print("    Initial pose publish FAILED")

    time.sleep(3)

    # 6. Check AMCL pose after initial pose
    print("\n[6] Checking AMCL pose after initial pose...")
    amcl_x, amcl_y = get_amcl_pose()
    print(f"    AMCL position: ({amcl_x}, {amcl_y})")

    if amcl_x is not None:
        error = (amcl_x**2 + amcl_y**2)**0.5
        if error > 0.5:
            print(f"    WARNING: AMCL not at origin! Error: {error:.3f}m")
        else:
            print(f"    OK: AMCL at origin (error: {error:.3f}m)")

    # 7. Final comparison
    print("\n[7] Final pose comparison...")
    gz_x, gz_y = get_gazebo_pose()
    amcl_x, amcl_y = get_amcl_pose()

    print(f"    Gazebo: ({gz_x}, {gz_y})")
    print(f"    AMCL:   ({amcl_x}, {amcl_y})")

    if gz_x is not None and amcl_x is not None:
        error = ((gz_x - amcl_x)**2 + (gz_y - amcl_y)**2)**0.5
        print(f"    Localization error: {error:.3f}m")

        if error > 0.5:
            print("\n    DIAGNOSIS: Localization is NOT working properly!")
            print("    Possible causes:")
            print("    - AMCL not receiving scan data")
            print("    - Map does not match environment")
            print("    - TF tree is broken")
        else:
            print("\n    DIAGNOSIS: Localization seems OK")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
