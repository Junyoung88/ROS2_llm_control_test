#!/bin/bash
cd /home/jim/ros2_motion_planning_tutorials
source /opt/ros/jazzy/setup.bash
source install/setup.bash
python3 -u src/geofence_enforcer/experiments/run_gazebo_s1_s6.py \
  --method no_guard --scenario S5 --seeds 1 --seed-offset 0 --no-sweep \
  --intensity fab_traverse \
  --output experiment_results/gazebo_s1_s6/fab/res_fab_traverse_no_guard_v0.jsonl \
  > experiment_results/gazebo_s1_s6/fab/fab_sanity_v0.log 2>&1
echo "FAB_SANITY_DONE rc=$?" >> experiment_results/gazebo_s1_s6/fab/fab_sanity_v0.log
