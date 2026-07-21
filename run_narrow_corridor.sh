#!/bin/bash
# Proper narrow-corridor experiment (reviewer ④): corridor width {xwide ~1.2m safe / wide
# ~0.6m / tight overlap-unsafe} × methods {no_guard, cbf_inflated, geofence} × 5 seeds.
# Metrics: goal-reach (task_completed) + violation (in_zone>0). Shows PETSE traverses safe
# corridors and blocks unsafe ones; baselines drive through regardless.
cd /home/jim/ros2_motion_planning_tutorials
source /opt/ros/jazzy/setup.bash; source install/setup.bash
export PETSE_ENFORCE_POSE=amcl PETSE_OFFSET_THRESH=0.95 PETSE_JUMP_THRESH=0.45 PETSE_DETECTION_MODE=cusum
OUT=experiment_results/gazebo_s1_s6/narrow; mkdir -p "$OUT"
SEEDS=5; MAX_RETRY=3
is_flaky () { python3 -c "
import json,sys
try: d=json.load(open('$1'))
except: sys.exit(0)
r=(d.get('reason') or '').lower()
sys.exit(0 if ('infrastructure' in r or 'nav2 rejected' in r or 'path failure' in r) else 1)"; }
run_cell () {  # geom method tag
  local S=0 valid=0
  while [ "$valid" -lt "$SEEDS" ]; do
    local attempt=0
    while [ "$attempt" -lt "$MAX_RETRY" ]; do
      echo "==== $3 v$valid seed=$S attempt=$attempt $(date +%H:%M:%S) ===="
      pkill -9 -f "gz sim"; pkill -9 -f cmd_vel_guard; pkill -9 -f metrics_logger; sleep 3
      python3 -u src/geofence_enforcer/experiments/run_gazebo_s1_s6.py --method "$2" --scenario S5 --seeds 1 --seed-offset "$S" --no-sweep --intensity "$1" --output "$OUT/res_${3}_v${valid}.jsonl" > "$OUT/${3}_v${valid}.log" 2>&1
      cp -f /tmp/position_monitor.log "$OUT/posmon_${3}_v${valid}.log" 2>/dev/null
      S=$((S+1)); attempt=$((attempt+1))
      if [ -s "$OUT/res_${3}_v${valid}.jsonl" ] && ! is_flaky "$OUT/res_${3}_v${valid}.jsonl"; then break; else echo "-- flaky retry --"; fi
    done
    valid=$((valid+1)); echo "PROGRESS $3 $valid/$SEEDS $(date +%H:%M:%S)"
  done
}
for G in nc2_xwide nc2_wide nc2_tight; do
  for M in no_guard cbf_inflated geofence; do
    run_cell geom_$G $M ${G}_${M}
  done
done
echo "NARROW DONE"
