#!/bin/bash
# Validate the redesigned CROSS-CHANNEL detector before the full 2x2:
#  1) clean + cusum      -> honest d_abs/jump envelope (no alarm, no incursion)
#  2) stealthy + cusum   -> d_abs grows past offset_thresh, fail-stop BEFORE zone (no incursion)
#  3) stealthy + memoryless -> jump stays small, detector misses, incursion happens
cd /home/jim/ros2_motion_planning_tutorials
source /opt/ros/jazzy/setup.bash
source install/setup.bash
OUTDIR=experiment_results/gazebo_s1_s6/detector_validate
mkdir -p "$OUTDIR"

one () {  # mode intensity tag seed
  local mode="$1" intensity="$2" tag="$3" seed="$4"
  for attempt in 0 1 2 3; do
    echo "======== $tag attempt=$attempt (mode=$mode int=$intensity seed=$seed) ========"
    pkill -9 -f "gz sim" 2>/dev/null; pkill -9 -f "attack_scan" 2>/dev/null; pkill -9 -f "cmd_vel_guard" 2>/dev/null
    sleep 3
    local out="$OUTDIR/res_${tag}.jsonl"
    PETSE_DETECTION_MODE="$mode" python3 -u src/geofence_enforcer/experiments/run_gazebo_s1_s6.py \
      --method geofence --scenario S5 --seeds 1 --seed-offset "$seed" --no-sweep \
      --intensity "$intensity" --output "$out" 2>&1 | tee "$OUTDIR/run_${tag}.log"
    cp -f /tmp/position_monitor.log "$OUTDIR/posmon_${tag}.log" 2>/dev/null
    cp -f /tmp/guard_standalone.log "$OUTDIR/guard_${tag}.log" 2>/dev/null
    seed=$((seed+1))
    python3 -c "
import json,sys
try: d=json.load(open('$out'))
except: sys.exit(0)
r=(d.get('reason') or '').lower()
sys.exit(0 if (d.get('nav_failed') and ('reject' in r or 'infrastructure' in r)) else 1)" && { echo '-- infra reject, retry --'; continue; }
    break
  done
  echo "-------- $tag recorded --------"
}

one cusum      warehouse_clean5  clean_cusum        0
one cusum      bias_hc_slow      stealthy_cusum     0
one memoryless bias_hc_slow      stealthy_memoryless 0
echo "ALL DETECTOR VALIDATE DONE"
