#!/bin/bash
# Fast diagnostic of the CORE claim: under the stealthy heading-compensated spoof,
# geofence-on-AMCL with MEMORYLESS detector suffers an incursion (spoof missed),
# while CUSUM catches the spoof and fail-stops (no/less incursion).
# Retries infra goal-rejections. 2 valid trials each mode.
cd /home/jim/ros2_motion_planning_tutorials
source /opt/ros/jazzy/setup.bash
source install/setup.bash
OUTDIR=experiment_results/gazebo_s1_s6/diag_stealthy
mkdir -p "$OUTDIR"
STEALTH=bias_hc_slow
NVALID=2
MAXRETRY=4

run () {
  local mode="$1"; local valid=0 S=0
  while [ "$valid" -lt "$NVALID" ]; do
    local attempt=0
    while [ "$attempt" -lt "$MAXRETRY" ]; do
      echo "======== $mode valid=$valid slot=$S attempt=$attempt ========"
      pkill -9 -f "gz sim" 2>/dev/null; pkill -9 -f "attack_scan" 2>/dev/null; pkill -9 -f "cmd_vel_guard" 2>/dev/null
      sleep 3
      local out="$OUTDIR/res_${mode}_v${valid}.jsonl"
      PETSE_DETECTION_MODE="$mode" python3 -u src/geofence_enforcer/experiments/run_gazebo_s1_s6.py \
        --method geofence --scenario S5 --seeds 1 --seed-offset "$S" --no-sweep \
        --intensity "$STEALTH" --output "$out" 2>&1 | tee "$OUTDIR/run_${mode}_v${valid}.log"
      cp -f /tmp/position_monitor.log "$OUTDIR/posmon_${mode}_v${valid}.log" 2>/dev/null
      cp -f /tmp/guard_standalone.log "$OUTDIR/guard_${mode}_v${valid}.log" 2>/dev/null
      S=$((S+1)); attempt=$((attempt+1))
      python3 -c "
import json,sys
try: d=json.load(open('$out'))
except: sys.exit(0)
r=(d.get('reason') or '').lower()
sys.exit(0 if (d.get('nav_failed') and ('reject' in r or 'infrastructure' in r)) else 1)" && { echo '-- infra reject, retry --'; continue; }
      break
    done
    valid=$((valid+1)); echo "-------- $mode valid $valid recorded --------"
  done
}

run memoryless
run cusum
echo "ALL DIAG STEALTHY DONE"
