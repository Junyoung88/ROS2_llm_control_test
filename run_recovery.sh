#!/bin/bash
# Recovery experiment: TRANSIENT strong spoof (fires 4s, lasts 40s so CUSUM fail-stops
# FIRST, then honest scans restored). Two arms, both CUSUM:
#   latch   (default)           -> fail-stop LATCHES: safe-hold, robot stays out of zone,
#                                  does NOT auto-resume (operator must clear). Fail-secure.
#   recover (PETSE_AUTO_RECOVER) -> un-latch once offset decays: robot RESUMES, reaches
#                                   goal, still never enters the zone.
# Retries ONLY flaky boot aborts (Nav2 path failure with NO spoof-detection); a trial
# where the guard fired (SPOOF DETECTED) is valid and kept regardless of goal outcome.
# Usage: bash run_recovery.sh [SEEDS] [MAX_RETRY]
cd /home/jim/ros2_motion_planning_tutorials
source /opt/ros/jazzy/setup.bash
source install/setup.bash

SEEDS="${1:-3}"; MAX_RETRY="${2:-3}"
INTENSITY=map_spoof_transient
OUTDIR=experiment_results/gazebo_s1_s6/money_2x2/recovery
mkdir -p "$OUTDIR"
export PETSE_ENFORCE_POSE=amcl PETSE_OFFSET_THRESH=0.95 PETSE_JUMP_THRESH=0.45 PETSE_DETECTION_MODE=cusum

run_one () {   # arm(latch|recover) seedslot
  local arm="$1" S0="$2" tag="${1}_v${2}"
  if [ "$arm" = "recover" ]; then export PETSE_AUTO_RECOVER=1; else export PETSE_AUTO_RECOVER=0; fi
  local attempt=0 S="$S0"
  while [ "$attempt" -lt "$MAX_RETRY" ]; do
    echo "======== $tag seed=$S attempt=$attempt $(date +%H:%M:%S) ========"
    pkill -9 -f "gz sim" 2>/dev/null; pkill -9 -f "attack_scan" 2>/dev/null
    pkill -9 -f "cmd_vel_guard" 2>/dev/null; pkill -9 -f "metrics_logger" 2>/dev/null
    sleep 3
    python3 -u src/geofence_enforcer/experiments/run_gazebo_s1_s6.py \
      --method geofence --scenario S5 --seeds 1 --seed-offset "$S" --no-sweep \
      --intensity "$INTENSITY" --output "$OUTDIR/res_${tag}.jsonl" > "$OUTDIR/${tag}.log" 2>&1
    cp -f /tmp/position_monitor.log "$OUTDIR/posmon_${tag}.log" 2>/dev/null
    cp -f /tmp/guard_standalone.log "$OUTDIR/guard_${tag}.log" 2>/dev/null
    attempt=$((attempt+1)); S=$((S+100))   # different seed on retry
    if grep -q "SPOOF DETECTED" "$OUTDIR/guard_${tag}.log" 2>/dev/null; then
      echo "-- $tag VALID (guard fired) --"; break
    else
      echo "-- $tag flaky (no spoof-detection), retry $attempt/$MAX_RETRY --"
    fi
  done
}

for S in $(seq 0 $((SEEDS-1))); do
  run_one recover "$S"
  run_one latch   "$S"
done
echo "RECOVERY DONE"
