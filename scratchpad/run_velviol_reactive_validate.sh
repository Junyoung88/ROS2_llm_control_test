#!/bin/bash
# Validate the reactive fixed-margin baseline (R1-③): over-speed direct_control attack drives the
# robot toward zone x[4,6]. static_reactive stops ONLY when current pos within fixed 0.55m -> at
# 1.5 m/s the braking distance exceeds the margin -> overshoots into zone. geofence (PETSE) uses
# velocity-adaptive forward re-verification -> stops early. Capture posmon to see the overshoot.
set +e
cd /home/jim/ros2_motion_planning_tutorials
source /opt/ros/jazzy/setup.bash; source install/setup.bash
OUT=experiment_results/gazebo_s1_s6/assum_violation; mkdir -p "$OUT"
for M in static_reactive geofence; do
  echo "==== VELVIOL-REACTIVE $M $(date +%H:%M:%S) ===="
  pkill -9 -f "gz sim" 2>/dev/null; pkill -9 -f cmd_vel_guard 2>/dev/null; sleep 3
  python3 -u src/geofence_enforcer/experiments/run_gazebo_s1_s6.py \
    --method "$M" --scenario S4 --seeds 1 --seed-offset 0 --no-sweep \
    --intensity direct_to_zone_overspeed --output "$OUT/val_reactive_${M}.jsonl" > "$OUT/val_reactive_${M}.log" 2>&1
  cp -f /tmp/position_monitor.log "$OUT/posmon_reactive_${M}.log" 2>/dev/null
  echo "PROGRESS reactive $M $(date +%H:%M:%S)"
  python3 -c "import json;d=json.load(open('$OUT/val_reactive_${M}.jsonl'));print('  ==>',d.get('method'),'violated=',d.get('violated'),'vc=',d.get('violation_count'),'| ',(d.get('reason') or '')[:60])" 2>/dev/null || echo "  (no result json)"
done
echo "VELVIOL-REACTIVE VALIDATE DONE $(date +%H:%M:%S)"
