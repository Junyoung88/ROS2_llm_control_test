#!/bin/bash
# Realism sweep (reviewer gap ④): the map-consistent spoof restricted to a limited
# angular FoV (real spoofers can't override a full 360° scan). Sweep FoV ∈ {360,180,90,45}
# under CUSUM. Expect: lure weakens as FoV shrinks (real beams anchor AMCL), while PETSE
# still detects any residual AMCL-odom offset (detector agnostic to spoofing modality).
# Usage: bash run_realism.sh [SEEDS] [MAX_RETRY]
cd /home/jim/ros2_motion_planning_tutorials
source /opt/ros/jazzy/setup.bash
source install/setup.bash

SEEDS="${1:-3}"; MAX_RETRY="${2:-3}"
OUTDIR=experiment_results/gazebo_s1_s6/money_2x2/realism
mkdir -p "$OUTDIR"
export PETSE_ENFORCE_POSE=amcl PETSE_OFFSET_THRESH=0.95 PETSE_JUMP_THRESH=0.45 PETSE_DETECTION_MODE=cusum

is_flaky () {  # retry only Nav2 boot aborts where the guard never even saw the spoof
  ! grep -q "spoofmon" "$1" 2>/dev/null
}

run_cell () {   # intensity tag
  local intensity="$1" tag="$2" valid=0 S=0
  echo "######## $tag ($intensity) ########"
  while [ "$valid" -lt "$SEEDS" ]; do
    local attempt=0
    while [ "$attempt" -lt "$MAX_RETRY" ]; do
      echo "==== $tag v$valid seed=$S attempt=$attempt $(date +%H:%M:%S) ===="
      pkill -9 -f "gz sim" 2>/dev/null; pkill -9 -f attack_scan 2>/dev/null
      pkill -9 -f cmd_vel_guard 2>/dev/null; pkill -9 -f metrics_logger 2>/dev/null
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

run_cell map_spoof_strong  fov360   # baseline (full 360 replacement)
run_cell map_spoof_fov180  fov180
run_cell map_spoof_fov90   fov90
run_cell map_spoof_fov45   fov45
echo "REALISM DONE"
