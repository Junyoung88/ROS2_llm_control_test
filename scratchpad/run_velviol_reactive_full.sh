#!/bin/bash
# R1-③ assumption violation (velocity bound), FULL run: over-speed direct_control (~2.7 m/s, 5.4×
# the declared v_max=0.5) drives the robot toward zone x[4,6]y[-1,1].
#   no_guard        : no execution-time defense                → enters
#   static_reactive : reactive FIXED margin (trusts declared v_max, no forward prediction)
#                     → braking distance > 0.55 m buffer → overshoots INTO zone
#   geofence (PETSE): execution-time re-verification (authorization + velocity-adaptive margin)
#                     → stops in time
# 5 seeds each, with flaky-retry (infrastructure/nav failures).
set +e
cd /home/jim/ros2_motion_planning_tutorials
source /opt/ros/jazzy/setup.bash; source install/setup.bash
OUT=experiment_results/gazebo_s1_s6/assum_violation; mkdir -p "$OUT"
SEEDS=5; MAX_RETRY=3
is_flaky(){ python3 -c "
import json,sys
try: d=json.load(open('$1'))
except: sys.exit(0)
r=(d.get('reason') or '').lower()
sys.exit(0 if ('infrastructure' in r or 'nav2 rejected' in r or 'failed to start' in r) else 1)"; }

run_cell(){  # $1=method
  local S=0 valid=0
  while [ "$valid" -lt "$SEEDS" ]; do
    local attempt=0
    while [ "$attempt" -lt "$MAX_RETRY" ]; do
      echo "==== $1 v$valid seed=$S $(date +%H:%M:%S) ===="
      pkill -9 -f "gz sim" 2>/dev/null; pkill -9 -f cmd_vel_guard 2>/dev/null; sleep 3
      python3 -u src/geofence_enforcer/experiments/run_gazebo_s1_s6.py \
        --method "$1" --scenario S4 --seeds 1 --seed-offset "$S" --no-sweep \
        --intensity direct_to_zone_overspeed --output "$OUT/res_velreact_${1}_v${valid}.jsonl" \
        > "$OUT/velreact_${1}_v${valid}.log" 2>&1
      S=$((S+1)); attempt=$((attempt+1))
      if [ -s "$OUT/res_velreact_${1}_v${valid}.jsonl" ] && ! is_flaky "$OUT/res_velreact_${1}_v${valid}.jsonl"; then break; else echo "-- flaky retry --"; fi
    done
    python3 -c "import json;d=json.load(open('$OUT/res_velreact_${1}_v${valid}.jsonl'));print(f'  v${valid}: violated={d.get(\"violated\")} vc={d.get(\"violation_count\")}')" 2>/dev/null
    valid=$((valid+1)); echo "PROGRESS $1 $valid/$SEEDS $(date +%H:%M:%S)"
  done
}
for M in no_guard static_reactive geofence; do
  run_cell "$M"
done
echo "VELREACT FULL DONE $(date +%H:%M:%S)"
