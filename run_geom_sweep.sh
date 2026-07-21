#!/bin/bash
# Generalization sweep (reviewer ④): forbidden-zone geometry variations g1-g4 × {no_guard,
# geofence} × 3 seeds. no_guard path-through → violation; PETSE → 0 VR (goal-reject or stop).
cd /home/jim/ros2_motion_planning_tutorials
source /opt/ros/jazzy/setup.bash; source install/setup.bash
export PETSE_ENFORCE_POSE=amcl PETSE_OFFSET_THRESH=0.95 PETSE_JUMP_THRESH=0.45 PETSE_DETECTION_MODE=cusum
OUT=experiment_results/gazebo_s1_s6/geom; mkdir -p "$OUT"
SEEDS=3; MAX_RETRY=3
is_flaky () {  # retry only Nav2 boot aborts (robot never moved)
  python3 -c "
import json,sys
try: d=json.load(open('$1'))
except: sys.exit(0)
r=(d.get('reason') or '').lower()
sys.exit(0 if ('infrastructure' in r or 'nav2 rejected' in r or 'path failure' in r) else 1)
"
}
run_one () {  # geom method tag
  local S=0 valid=0
  while [ "$valid" -lt "$SEEDS" ]; do
    local attempt=0
    while [ "$attempt" -lt "$MAX_RETRY" ]; do
      echo "==== $3 v$valid seed=$S attempt=$attempt $(date +%H:%M:%S) ===="
      pkill -9 -f "gz sim"; pkill -9 -f cmd_vel_guard; pkill -9 -f metrics_logger; sleep 3
      python3 -u src/geofence_enforcer/experiments/run_gazebo_s1_s6.py --method "$2" --scenario S5 --seeds 1 --seed-offset "$S" --no-sweep --intensity "$1" --output "$OUT/res_${3}_v${valid}.jsonl" > "$OUT/${3}_v${valid}.log" 2>&1
      cp -f /tmp/position_monitor.log "$OUT/posmon_${3}_v${valid}.log" 2>/dev/null
      S=$((S+1)); attempt=$((attempt+1))
      if [ -s "$OUT/res_${3}_v${valid}.jsonl" ] && ! is_flaky "$OUT/res_${3}_v${valid}.jsonl"; then break; else echo "-- flaky, retry $attempt --"; fi
    done
    valid=$((valid+1)); echo "PROGRESS $3 $valid/$SEEDS $(date +%H:%M:%S)"
  done
}
for G in geom_g1 geom_g2 geom_g3 geom_g4; do
  run_one $G no_guard  ${G}_no_guard
  run_one $G geofence  ${G}_geofence
done
echo "GEOM SWEEP DONE"
