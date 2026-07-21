#!/bin/bash
# Coordinated-attack sweep (reviewer ①): LiDAR spoof + synchronized odom spoof holding the
# cross-channel offset c=amcl−odom near a residual ε. Sweep ε ∈ {0, 0.3, 0.6, 1.3} under
# CUSUM. Expect: ε<0.95 → PETSE EVADED (c stays low, robot enters zone = defense broken);
# ε>0.95 → PETSE detects (fail-stop). Quantifies the coordination the attacker must achieve.
# Usage: bash run_coord.sh [SEEDS] [MAX_RETRY]
cd /home/jim/ros2_motion_planning_tutorials
source /opt/ros/jazzy/setup.bash
source install/setup.bash

SEEDS="${1:-3}"; MAX_RETRY="${2:-3}"
OUTDIR=experiment_results/gazebo_s1_s6/money_2x2/coord
mkdir -p "$OUTDIR"
export PETSE_ENFORCE_POSE=amcl PETSE_OFFSET_THRESH=0.95 PETSE_JUMP_THRESH=0.45 PETSE_DETECTION_MODE=cusum

is_flaky () { ! grep -q "spoofmon" "$1" 2>/dev/null; }

run_cell () {   # intensity tag
  local intensity="$1" tag="$2" valid=0 S=0
  echo "######## $tag ($intensity) ########"
  while [ "$valid" -lt "$SEEDS" ]; do
    local attempt=0
    while [ "$attempt" -lt "$MAX_RETRY" ]; do
      echo "==== $tag v$valid seed=$S attempt=$attempt $(date +%H:%M:%S) ===="
      pkill -9 -f "gz sim" 2>/dev/null; pkill -9 -f attack_scan 2>/dev/null
      pkill -9 -f attack_odom 2>/dev/null; pkill -9 -f cmd_vel_guard 2>/dev/null
      pkill -9 -f metrics_logger 2>/dev/null; pkill -9 -f "relay /odom /odom_spoofed" 2>/dev/null
      sleep 3
      python3 -u src/geofence_enforcer/experiments/run_gazebo_s1_s6.py \
        --method geofence --scenario S5 --seeds 1 --seed-offset "$S" --no-sweep \
        --intensity "$intensity" --output "$OUTDIR/res_${tag}_v${valid}.jsonl" \
        > "$OUTDIR/${tag}_v${valid}.log" 2>&1
      cp -f /tmp/position_monitor.log "$OUTDIR/posmon_${tag}_v${valid}.log" 2>/dev/null
      cp -f /tmp/guard_standalone.log "$OUTDIR/guard_${tag}_v${valid}.log" 2>/dev/null
      S=$((S+1)); attempt=$((attempt+1))
      if is_flaky "$OUTDIR/guard_${tag}_v${valid}.log"; then
        echo "-- flaky (no spoofmon), retry $attempt --"
      else break; fi
    done
    valid=$((valid+1)); echo "PROGRESS $tag $valid/$SEEDS $(date +%H:%M:%S)"
  done
}

run_cell coord_eps00 eps00
run_cell coord_eps03 eps03
run_cell coord_eps06 eps06
run_cell coord_eps13 eps13
echo "COORD DONE"
