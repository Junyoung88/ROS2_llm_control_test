#!/bin/bash
set +e
cd /home/jim/ros2_motion_planning_tutorials
source /opt/ros/jazzy/setup.bash; source install/setup.bash
OUT=experiment_results/gazebo_s1_s6/assum_violation
for M in static_margin geofence; do
  echo "==== VELVIOL validate $M $(date +%H:%M:%S) ===="
  pkill -9 -f "gz sim" 2>/dev/null; pkill -9 -f cmd_vel_guard 2>/dev/null; pkill -9 -f metrics_logger 2>/dev/null; sleep 3
  python3 -u src/geofence_enforcer/experiments/run_gazebo_s1_s6.py \
    --method "$M" --scenario S4 --seeds 1 --seed-offset 0 --no-sweep \
    --intensity velocity_scaling_2x --output "$OUT/val_${M}.jsonl" > "$OUT/val_${M}.log" 2>&1
  echo "PROGRESS velviol $M $(date +%H:%M:%S)"
done
echo "VELVIOL VALIDATE DONE $(date +%H:%M:%S)"
