#!/bin/bash
# Demonstration A (threat reality): STRONG map-consistent spoof.
#   no_guard  -> robot dragged DEEP into the forbidden zone (violation)
#   geofence  -> PETSE-CUSUM fail-stops at the boundary
# Usage: bash run_demoA.sh [SEEDS]
cd /home/jim/ros2_motion_planning_tutorials
source /opt/ros/jazzy/setup.bash
source install/setup.bash

SEEDS="${1:-3}"
INTENSITY=map_spoof_strong
OUTDIR=experiment_results/gazebo_s1_s6/money_2x2/demoA
mkdir -p "$OUTDIR"

export PETSE_ENFORCE_POSE=amcl
export PETSE_OFFSET_THRESH=0.95
export PETSE_JUMP_THRESH=0.45
export PETSE_DETECTION_MODE=cusum

run_one () {   # method tag seedslot
  local method="$1" tag="$2" S="$3"
  echo "======== $tag seed=$S (method=$method) ========"
  pkill -9 -f "gz sim" 2>/dev/null; pkill -9 -f "attack_scan" 2>/dev/null
  pkill -9 -f "cmd_vel_guard" 2>/dev/null; pkill -9 -f "metrics_logger" 2>/dev/null
  sleep 3
  python3 -u src/geofence_enforcer/experiments/run_gazebo_s1_s6.py \
    --method "$method" --scenario S5 --seeds 1 --seed-offset "$S" --no-sweep \
    --intensity "$INTENSITY" \
    --output "$OUTDIR/res_${tag}.jsonl" > "$OUTDIR/${tag}.log" 2>&1
  cp -f /tmp/position_monitor.log "$OUTDIR/posmon_${tag}.log" 2>/dev/null
  cp -f /tmp/guard_standalone.log "$OUTDIR/guard_${tag}.log" 2>/dev/null
}

for S in $(seq 0 $((SEEDS-1))); do
  run_one no_guard  "noguard_v${S}"  "$S"
  run_one geofence  "cusum_v${S}"    "$S"
done
echo "DEMO A DONE"
