#!/bin/bash
# R4: fill the combined-ablation grid with REAL runs for the "planning guard only"
# condition (PETSE's goal/path admission gate with the runtime monitor disabled:
# PETSE_SPATIAL_CHECK=false, PETSE_SPOOF_DET=false). We run it on a PLAN-TIME attack
# (S1, goal inside zone) and an EXECUTION-TIME attack (S4, TOCTOU pose offset) to
# show the phase-dependence directly: planning-only catches plan-time but breaches
# execution-time. Full PETSE (=0 everywhere) is Table~II; no_guard is Table~II too.
set +e
cd /home/jim/ros2_motion_planning_tutorials
source /opt/ros/jazzy/setup.bash; source install/setup.bash
export PETSE_DETECTION_MODE=cusum
export PETSE_SPATIAL_CHECK=false PETSE_SPOOF_DET=false   # planning-guard-only
OUT=experiment_results/gazebo_s1_s6/r4_planning; mkdir -p "$OUT"
SEEDS=3; MAX_RETRY=3
is_flaky(){ python3 -c "
import json,sys
try: d=json.load(open('$1'))
except: sys.exit(0)
r=(d.get('reason') or '').lower()
sys.exit(0 if ('infrastructure' in r or 'nav2 rejected' in r or 'failed to start' in r or \"didn't move\" in r or 'marking infra' in r) else 1)"; }

run_cell(){  # $1=scenario $2=tag
  local i=0
  while [ "$i" -lt "$SEEDS" ]; do
    local S=$i attempt=0
    while [ "$attempt" -lt "$MAX_RETRY" ]; do
      echo "==== $2 v$i seed=$S attempt=$attempt $(date +%H:%M:%S) ===="
      pkill -9 -f "gz sim" 2>/dev/null; pkill -9 -f cmd_vel_guard 2>/dev/null; pkill -9 -f metrics_logger 2>/dev/null; sleep 3
      python3 -u src/geofence_enforcer/experiments/run_gazebo_s1_s6.py \
        --method geofence --scenario "$1" --seeds 1 --seed-offset "$S" --no-sweep \
        --output "$OUT/res_${2}_v${i}.jsonl" > "$OUT/${2}_v${i}.log" 2>&1
      S=$((S+1)); attempt=$((attempt+1))
      if [ -s "$OUT/res_${2}_v${i}.jsonl" ] && ! is_flaky "$OUT/res_${2}_v${i}.jsonl"; then break; else echo "-- flaky retry --"; fi
    done
    python3 -c "import json;d=json.load(open('$OUT/res_${2}_v${i}.jsonl'));print(f'  ==> $2 v${i}: violated={d.get(\"violated\")} decision={d.get(\"decision\")} vc={d.get(\"violation_count\")}')" 2>/dev/null
    i=$((i+1)); echo "PROGRESS $2 $i/$SEEDS $(date +%H:%M:%S)"
  done
}
run_cell S1 plan_S1     # plan-time attack: goal inside zone
run_cell S4 exec_S4     # execution-time attack: TOCTOU pose offset
echo "R4_PLANNING_DONE $(date +%H:%M:%S)"
