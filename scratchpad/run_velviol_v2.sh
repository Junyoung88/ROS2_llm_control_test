#!/bin/bash
set +e
cd /home/jim/ros2_motion_planning_tutorials
source /opt/ros/jazzy/setup.bash; source install/setup.bash
OUT=experiment_results/gazebo_s1_s6/assum_violation
mkdir -p "$OUT"
for M in no_guard static_margin; do
  echo "==== V2 $M $(date +%H:%M:%S) ===="
  pkill -9 -f "gz sim" 2>/dev/null; pkill -9 -f cmd_vel_guard 2>/dev/null; pkill -9 -f metrics_logger 2>/dev/null; sleep 3
  python3 -u src/geofence_enforcer/experiments/run_gazebo_s1_s6.py \
    --method "$M" --scenario S4 --seeds 1 --seed-offset 0 --no-sweep \
    --intensity velocity_scaling_2x --output "$OUT/v2_${M}.jsonl" > "$OUT/v2_${M}.log" 2>&1
  cp -f /tmp/position_monitor.log "$OUT/v2_posmon_${M}.log" 2>/dev/null
  echo "PROGRESS v2 $M $(date +%H:%M:%S)"
done
echo "V2 DONE $(date +%H:%M:%S)"
