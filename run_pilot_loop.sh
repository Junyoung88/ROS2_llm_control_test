#!/bin/bash
# Simple pilot test loop using the working single trial script

WORKSPACE="/home/jim/ros2_motion_planning_tutorials/src/mobile_manipulator_tutorial"

# Test cases: method goal_x goal_y expected
declare -a TESTS=(
    "no_guard 2.0 1.0 allow"
    "selp_proper 2.0 1.0 reject"
    "cbf 2.0 1.0 reject"
    "ssm 2.0 1.0 reject"
    "geofence 2.0 1.0 reject"
)

echo "========================================"
echo "PILOT TEST LOOP"
echo "========================================"

for test in "${TESTS[@]}"; do
    read -r method gx gy expected <<< "$test"

    echo ""
    echo "Testing: $method at ($gx, $gy) - expected: $expected"
    echo "----------------------------------------"

    # Cleanup
    pkill -9 -f 'gz sim' 2>/dev/null
    pkill -9 -f 'goal_gate' 2>/dev/null
    pkill -9 -f 'nav2' 2>/dev/null
    sleep 3

    # Start Gazebo
    source /opt/ros/jazzy/setup.bash && source $WORKSPACE/install/setup.bash
    ros2 launch mobile_manip_moveit_config mobile_manipulator.launch.py headless:=true &
    sleep 20

    # Start Nav2
    ros2 launch mobile_manip_moveit_config navigation.launch.py use_sim_time:=true rviz:=false &
    sleep 12

    # Start Geofence and capture output
    ros2 launch geofence_policy_enforcer demo.launch.py safety_method:=$method use_sim_time:=true 2>&1 &
    GEOFENCE_PID=$!
    sleep 6

    # Send goal
    ros2 action send_goal /navigate_to_pose_safe nav2_msgs/action/NavigateToPose \
        "{pose: {header: {frame_id: 'map'}, pose: {position: {x: $gx, y: $gy, z: 0.0}, orientation: {w: 1.0}}}}" 2>&1 | tee /tmp/goal_result.txt

    sleep 3

    # Check ROS logs for decision
    echo ""
    echo "Checking logs..."
    ros2 topic echo /rosout --once 2>/dev/null | grep -E "(ALLOWED|REJECTED|PROJECTED)" | head -5 || echo "No decision in rosout"

    # Cleanup for next test
    pkill -9 -f 'gz sim' 2>/dev/null
    sleep 2
done

echo ""
echo "========================================"
echo "PILOT TEST COMPLETE"
echo "========================================"
