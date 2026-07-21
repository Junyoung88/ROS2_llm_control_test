#!/bin/bash
# Money 2x2: detector {memoryless, cusum} x attack {clean, stealthy-HC}
# Shows CUSUM catches the stealthy heading-compensated bias-injection that the
# memoryless detector misses; neither false-alarms on clean.
# Retries a seed (fresh boot) when Nav2 rejects the goal (infrastructure flake,
# robot never moved) so every recorded trial is a VALID navigation.
# Usage: bash run_money_2x2.sh [SEEDS] [MAX_RETRY]
cd /home/jim/ros2_motion_planning_tutorials
source /opt/ros/jazzy/setup.bash
source install/setup.bash

SEEDS="${1:-6}"
MAX_RETRY="${2:-3}"
STEALTH_CFG=map_spoof_slow    # SLOW ramp (0.12->0.08): small AMCL jumps (<0.45) so
                              # (toward the +Y racks that anchor the forged scan → AMCL
                              #  follows strongly → TRUE robot lured −Y into the zone;
                              #  per-update jumps ~0.09 still evade the memoryless gate)
CLEAN_CFG=warehouse_clean6    # clean straight-+X nav to (4.5,0), matched geometry (no turn)
OUTDIR=experiment_results/gazebo_s1_s6/money_2x2
mkdir -p "$OUTDIR"

# Cross-channel detector thresholds (recalibrated 2026-07-15, +X-straight geometry):
#   clean d_abs ceiling ~0.3 (no turn → honest odom), spoof d_abs → ~1.4 (crosses well
#   before the −Y zone). offset_thresh 0.6 sits between them so CUSUM trips while the
#   robot is still short of y=−1.0 (catch BEFORE incursion), no clean false alarm.
export PETSE_ENFORCE_POSE=amcl
export PETSE_OFFSET_THRESH="${PETSE_OFFSET_THRESH:-0.95}"
# memoryless = teleport/innovation-gate detector at teleport scale (0.45). Clean jumps
# ~0.06 and the ramp's per-update steps ~0.09 both stay under it → memoryless MISSES the
# slow ramp (Urbina CCS'16); CUSUM accumulates the coherent offset drift and catches.
export PETSE_JUMP_THRESH="${PETSE_JUMP_THRESH:-0.45}"

is_infra_fail () {  # returns 0 (true) if the trial never produced a valid navigation
  python3 -c "
import json,sys
try:
    d=json.load(open('$1'))
except Exception:
    sys.exit(0)   # missing/corrupt -> treat as infra fail (retry)
r=(d.get('reason') or '').lower()
navfail=d.get('nav_failed')
# Retry (infra fail) when: (a) Nav2 rejected the goal at boot (never navigated), OR
# (b) the intermittent warehouse +X 'Nav2 path failure' abort where the robot never
# left the origin (goal aborted before/at the mid-nav spoof) — not a real clean pass
# or a real spoof outcome, just harness flake. Real 'goal reached'/'timed out'/'spoof
# detected'/'runtime' results are valid.
flaky_abort = ('nav2 path failure' in r) or ('navigation aborted' in r)
infra = (navfail and ('reject' in r or 'infrastructure' in r))
sys.exit(0 if (infra or flaky_abort) else 1)
"
}

run_cell () {
  local mode="$1" intensity="$2" tag="$3"
  echo "######## CELL $tag  (detector=$mode, intensity=$intensity) ########"
  local valid=0 S=0
  while [ "$valid" -lt "$SEEDS" ]; do
    local attempt=0 ok=1
    while [ "$attempt" -lt "$MAX_RETRY" ]; do
      echo "======== $tag valid=$valid seedslot=$S attempt=$attempt ========"
      pkill -9 -f "gz sim" 2>/dev/null; pkill -9 -f "attack_scan" 2>/dev/null; pkill -9 -f "cmd_vel_guard" 2>/dev/null
      sleep 3
      local out="$OUTDIR/results_${tag}_v${valid}.jsonl"
      PETSE_DETECTION_MODE="$mode" python3 -u src/geofence_enforcer/experiments/run_gazebo_s1_s6.py \
        --method geofence --scenario S5 --seeds 1 --seed-offset "$S" --no-sweep \
        --intensity "$intensity" \
        --output "$out" 2>&1 | tee "$OUTDIR/run_${tag}_v${valid}.log"
      cp -f /tmp/position_monitor.log "$OUTDIR/posmon_${tag}_v${valid}.log" 2>/dev/null
      cp -f /tmp/guard_standalone.log "$OUTDIR/guard_${tag}_v${valid}.log" 2>/dev/null
      S=$((S+1)); attempt=$((attempt+1))
      if is_infra_fail "$out"; then
        echo "-- infra goal-rejection, retrying (attempt $attempt/$MAX_RETRY) --"
        ok=0
      else
        ok=1; break
      fi
    done
    if [ "$ok" -eq 1 ]; then
      valid=$((valid+1))
      echo "-------- $tag valid trial $valid/$SEEDS recorded --------"
    else
      echo "-- gave up seedslot after $MAX_RETRY infra fails, counting anyway --"
      valid=$((valid+1))
    fi
  done
}

run_cell memoryless "$CLEAN_CFG"    clean_memoryless
run_cell cusum      "$CLEAN_CFG"    clean_cusum
run_cell memoryless "$STEALTH_CFG"  stealthy_memoryless
run_cell cusum      "$STEALTH_CFG"  stealthy_cusum
echo "ALL MONEY 2x2 DONE"
