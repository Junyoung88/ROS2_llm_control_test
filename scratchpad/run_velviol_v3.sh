#!/bin/bash
set +e
cd /home/jim/ros2_motion_planning_tutorials
source /opt/ros/jazzy/setup.bash; source install/setup.bash
OUT=experiment_results/gazebo_s1_s6/assum_violation
mkdir -p "$OUT"
for M in static_margin geofence no_guard; do
  echo "==== V3 $M $(date +%H:%M:%S) ===="
  pkill -9 -f "gz sim" 2>/dev/null; pkill -9 -f cmd_vel_guard 2>/dev/null; pkill -9 -f metrics_logger 2>/dev/null; sleep 3
  python3 -u src/geofence_enforcer/experiments/run_gazebo_s1_s6.py \
    --method "$M" --scenario S4 --seeds 1 --seed-offset 0 --no-sweep \
    --intensity direct_to_zone --output "$OUT/v3_${M}.jsonl" > "$OUT/v3_${M}.log" 2>&1
  cp -f /tmp/position_monitor.log "$OUT/v3_posmon_${M}.log" 2>/dev/null
  echo "PROGRESS v3 $M $(date +%H:%M:%S)"
done
echo "V3 DONE $(date +%H:%M:%S)"
