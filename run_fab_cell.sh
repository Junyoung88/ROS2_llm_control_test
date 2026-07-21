#!/bin/bash
# Fab-cell testbed experiment: multi-seed × multi-method, two goal types.
#   fab_traverse  — safe aisle path-through (7.5,0): does PETSE traverse a realistic
#                   fab aisle without over-blocking? baselines should reach too.
#   fab_forbidden — goal (4,-2.2) inside the south tool keep-out zone: admission-level
#                   reject for guard methods; no_guard drives in -> violation.
# Mirrors run_narrow_corridor.sh: per-seed flaky-retry, separate process per trial.
set +e
cd /home/jim/ros2_motion_planning_tutorials
source /opt/ros/jazzy/setup.bash
source install/setup.bash
OUT=experiment_results/gazebo_s1_s6/fab
mkdir -p "$OUT"
SEEDS=5
MAX_RETRY=3

is_flaky () { python3 -c "
import json,sys
try: d=json.load(open('$1'))
except: sys.exit(0)
r=(d.get('reason') or '').lower()
sys.exit(0 if ('infrastructure' in r or 'nav2 rejected' in r or 'path failure' in r) else 1)"; }

run_cell () {  # intensity method tag
  local S=0 valid=0
  while [ "$valid" -lt "$SEEDS" ]; do
    local attempt=0
    while [ "$attempt" -lt "$MAX_RETRY" ]; do
      echo "==== $3 v$valid seed=$S attempt=$attempt $(date +%H:%M:%S) ===="
      pkill -9 -f "gz sim" 2>/dev/null; pkill -9 -f cmd_vel_guard 2>/dev/null; pkill -9 -f metrics_logger 2>/dev/null; sleep 3
      python3 -u src/geofence_enforcer/experiments/run_gazebo_s1_s6.py \
        --method "$2" --scenario S5 --seeds 1 --seed-offset "$S" --no-sweep \
        --intensity "$1" --output "$OUT/res_${3}_v${valid}.jsonl" > "$OUT/${3}_v${valid}.log" 2>&1
      cp -f /tmp/position_monitor.log "$OUT/posmon_${3}_v${valid}.log" 2>/dev/null
      S=$((S+1)); attempt=$((attempt+1))
      if [ -s "$OUT/res_${3}_v${valid}.jsonl" ] && ! is_flaky "$OUT/res_${3}_v${valid}.jsonl"; then break; else echo "-- flaky retry --"; fi
    done
    valid=$((valid+1)); echo "PROGRESS $3 $valid/$SEEDS $(date +%H:%M:%S)"
  done
}

for I in fab_traverse fab_spoof_hijack; do
  for M in no_guard selp_proper cbf_inflated geofence; do
    run_cell "$I" "$M" "${I}_${M}"
  done
done
echo "FAB DONE $(date +%H:%M:%S)"
