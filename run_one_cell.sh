#!/bin/bash
# run_one_cell.sh MODE INTENSITY TAG
cd /home/jim/ros2_motion_planning_tutorials
source /opt/ros/jazzy/setup.bash
source install/setup.bash
MODE="${1:-cusum}"
INTENSITY="${2:-warehouse_clean6}"
TAG="${3:-verify}"
OUT=experiment_results/gazebo_s1_s6/money_2x2
mkdir -p "$OUT"
pkill -9 -f "gz sim" 2>/dev/null
pkill -9 -f "attack_scan" 2>/dev/null
pkill -9 -f "cmd_vel_guard" 2>/dev/null
sleep 3
export PETSE_ENFORCE_POSE=amcl
export PETSE_OFFSET_THRESH="${PETSE_OFFSET_THRESH:-1.35}"
export PETSE_JUMP_THRESH="${PETSE_JUMP_THRESH:-0.45}"
export PETSE_DETECTION_MODE="$MODE"
python3 -u src/geofence_enforcer/experiments/run_gazebo_s1_s6.py \
  --method geofence --scenario S5 --seeds 1 --seed-offset 0 --no-sweep \
  --intensity "$INTENSITY" \
  --output "$OUT/${TAG}.jsonl" > "$OUT/${TAG}.log" 2>&1
cp -f /tmp/position_monitor.log "$OUT/${TAG}_posmon.log" 2>/dev/null
cp -f /tmp/guard_standalone.log "$OUT/${TAG}_guard.log" 2>/dev/null
echo "CELL DONE tag=$TAG"
