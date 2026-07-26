#!/bin/bash
# 0.09 m/s boundary point for the slow-attacker hijack sweep (between wh_hijack 0.12 = attack
# succeeds and 0.06 = attack fails to reach the zone). no_guard proves attack validity; geofence
# tests PETSE defense.
set +e
cd /home/jim/ros2_motion_planning_tutorials
source /opt/ros/jazzy/setup.bash; source install/setup.bash
export PETSE_DETECTION_MODE=cusum
OUT=experiment_results/gazebo_s1_s6/hijack_rate; mkdir -p "$OUT"
SEEDS=2; MAX_RETRY=3
is_flaky(){ python3 -c "
import json,sys
try: d=json.load(open('$1'))
except: sys.exit(0)
r=(d.get('reason') or '').lower()
sys.exit(0 if ('infrastructure' in r or 'nav2 rejected' in r or 'failed to start' in r or \"didn't move\" in r) else 1)"; }
run_cell(){  # $1=method $2=tag
  local S=0 valid=0
  while [ "$valid" -lt "$SEEDS" ]; do
    local attempt=0
    while [ "$attempt" -lt "$MAX_RETRY" ]; do
      echo "==== $2 v$valid seed=$S attempt=$attempt $(date +%H:%M:%S) ===="
      pkill -9 -f "gz sim" 2>/dev/null; pkill -9 -f cmd_vel_guard 2>/dev/null; pkill -9 -f metrics_logger 2>/dev/null; sleep 3
      python3 -u src/geofence_enforcer/experiments/run_gazebo_s1_s6.py \
        --method "$1" --scenario S5 --seeds 1 --seed-offset "$S" --no-sweep \
        --intensity wh_hijack_r09 --output "$OUT/res_${2}_v${valid}.jsonl" > "$OUT/${2}_v${valid}.log" 2>&1
      cp -f /tmp/position_monitor.log "$OUT/posmon_${2}_v${valid}.log" 2>/dev/null
      S=$((S+1)); attempt=$((attempt+1))
      if [ -s "$OUT/res_${2}_v${valid}.jsonl" ] && ! is_flaky "$OUT/res_${2}_v${valid}.jsonl"; then break; else echo "-- flaky retry --"; fi
    done
    python3 -c "import json;d=json.load(open('$OUT/res_${2}_v${valid}.jsonl'));print(f'  ==> $2 v${valid}: violated={d.get(\"violated\")} vc={d.get(\"violation_count\")} pmin={d.get(\"path_min_distance\")} | {(d.get(\"reason\") or \"\")[:48]}')" 2>/dev/null
    valid=$((valid+1)); echo "PROGRESS $2 $valid/$SEEDS $(date +%H:%M:%S)"
  done
}
run_cell no_guard ng_r09
run_cell geofence gf_r09
echo "HIJACK R09 DONE $(date +%H:%M:%S)"
