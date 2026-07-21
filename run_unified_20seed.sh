#!/bin/bash
# Unified 20-seed 2x2: {clean, strong map-consistent spoof} x {memoryless, CUSUM}.
# The strong spoof reliably lures (Demo A 3/3) AND is jump-stealthy during approach
# (per-update jump ~0.06-0.12 << 0.45) -> memoryless is blind, CUSUM catches the
# accumulated offset. Resumable (skips trials whose valid result already exists),
# retries flaky Nav2 goal-rejections, reaps metrics_logger leak each trial.
# Usage: bash run_unified_20seed.sh [SEEDS] [MAX_RETRY]
cd /home/jim/ros2_motion_planning_tutorials
source /opt/ros/jazzy/setup.bash
source install/setup.bash

SEEDS="${1:-20}"
MAX_RETRY="${2:-3}"
STRONG_CFG=map_spoof_strong
CLEAN_CFG=warehouse_clean6
OUTDIR=experiment_results/gazebo_s1_s6/money_2x2/unified20
mkdir -p "$OUTDIR"

export PETSE_ENFORCE_POSE=amcl
export PETSE_OFFSET_THRESH=0.95
export PETSE_JUMP_THRESH=0.45

is_infra_fail () {
  python3 -c "
import json,sys
try: d=json.load(open('$1'))
except Exception: sys.exit(0)
r=(d.get('reason') or '').lower()
navfail=d.get('nav_failed')
flaky=('nav2 path failure' in r) or ('navigation aborted' in r) or (navfail and ('reject' in r or 'infrastructure' in r))
sys.exit(0 if flaky else 1)
"
}

run_cell () {   # mode intensity tag
  local mode="$1" intensity="$2" tag="$3"
  echo "######## CELL $tag (detector=$mode, intensity=$intensity) ########"
  local valid=0 S=0
  while [ "$valid" -lt "$SEEDS" ]; do
    local out="$OUTDIR/results_${tag}_v${valid}.jsonl"
    # RESUME: skip if a valid result already exists for this slot
    if [ -f "$out" ] && ! is_infra_fail "$out"; then
      echo "-- resume: $tag v$valid already valid, skipping --"
      valid=$((valid+1)); S=$((S+1)); continue
    fi
    local attempt=0 ok=1
    while [ "$attempt" -lt "$MAX_RETRY" ]; do
      echo "======== $tag valid=$valid seedslot=$S attempt=$attempt $(date +%H:%M:%S) ========"
      pkill -9 -f "gz sim" 2>/dev/null; pkill -9 -f "attack_scan" 2>/dev/null
      pkill -9 -f "cmd_vel_guard" 2>/dev/null; pkill -9 -f "metrics_logger" 2>/dev/null
      sleep 3
      PETSE_DETECTION_MODE="$mode" python3 -u src/geofence_enforcer/experiments/run_gazebo_s1_s6.py \
        --method geofence --scenario S5 --seeds 1 --seed-offset "$S" --no-sweep \
        --intensity "$intensity" --output "$out" > "$OUTDIR/run_${tag}_v${valid}.log" 2>&1
      cp -f /tmp/position_monitor.log "$OUTDIR/posmon_${tag}_v${valid}.log" 2>/dev/null
      cp -f /tmp/guard_standalone.log "$OUTDIR/guard_${tag}_v${valid}.log" 2>/dev/null
      S=$((S+1)); attempt=$((attempt+1))
      if is_infra_fail "$out"; then
        echo "-- infra goal-rejection, retry $attempt/$MAX_RETRY --"; ok=0
      else ok=1; break; fi
    done
    [ "$ok" -eq 1 ] && echo "-- $tag v$valid recorded --" || echo "-- $tag v$valid gave up (counting) --"
    valid=$((valid+1))
    echo "PROGRESS $tag $valid/$SEEDS $(date +%H:%M:%S)"
  done
}

run_cell cusum      "$CLEAN_CFG"   clean_cusum
run_cell memoryless "$CLEAN_CFG"   clean_memoryless
run_cell cusum      "$STRONG_CFG"  spoof_cusum
run_cell memoryless "$STRONG_CFG"  spoof_memoryless
echo "UNIFIED 20-SEED DONE"
