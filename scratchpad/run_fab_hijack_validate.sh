#!/bin/bash
set +e
cd /home/jim/ros2_motion_planning_tutorials
source /opt/ros/jazzy/setup.bash
source install/setup.bash
OUT=experiment_results/gazebo_s1_s6/fab
mkdir -p "$OUT"
for M in no_guard geofence; do
  echo "==== VALIDATE fab_spoof_hijack $M $(date +%H:%M:%S) ===="
  pkill -9 -f "gz sim" 2>/dev/null; pkill -9 -f cmd_vel_guard 2>/dev/null; pkill -9 -f metrics_logger 2>/dev/null; sleep 3
  python3 -u src/geofence_enforcer/experiments/run_gazebo_s1_s6.py \
    --method "$M" --scenario S5 --seeds 1 --seed-offset 0 --no-sweep \
    --intensity fab_spoof_hijack --output "$OUT/val_hijack_${M}.jsonl" > "$OUT/val_hijack_${M}.log" 2>&1
  cp -f /tmp/position_monitor.log "$OUT/val_posmon_${M}.log" 2>/dev/null
  echo "PROGRESS validate $M done $(date +%H:%M:%S)"
done
echo "HIJACK VALIDATE DONE $(date +%H:%M:%S)"
