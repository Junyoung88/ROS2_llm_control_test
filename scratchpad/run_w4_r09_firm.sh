#!/bin/bash
# W4 confirmatory: firm up the discriminating slow-spoof cell (ramp 0.09 m/s) to 5 seeds
# per method, AND capture the guard's cross-channel detection log per trial so we can
# report WHICH gate fired and the detection delay. Fresh seed-offsets (10-14) and a new
# output dir so the existing 2-seed sweep is untouched.
set +e
cd /home/jim/ros2_motion_planning_tutorials
source /opt/ros/jazzy/setup.bash; source install/setup.bash
export PETSE_DETECTION_MODE=cusum
OUT=experiment_results/gazebo_s1_s6/hijack_rate_w4; mkdir -p "$OUT"
SEEDS=5; MAX_RETRY=3; BASE=10
is_flaky(){ python3 -c "
import json,sys
try: d=json.load(open('$1'))
except: sys.exit(0)
r=(d.get('reason') or '').lower()
sys.exit(0 if ('infrastructure' in r or 'nav2 rejected' in r or 'failed to start' in r or \"didn't move\" in r) else 1)"; }
run_cell(){  # $1=method $2=tag
  local i=0
  while [ "$i" -lt "$SEEDS" ]; do
    local S=$((BASE+i)) attempt=0
    while [ "$attempt" -lt "$MAX_RETRY" ]; do
      echo "==== $2 v$i seed=$S attempt=$attempt $(date +%H:%M:%S) ===="
      pkill -9 -f "gz sim" 2>/dev/null; pkill -9 -f cmd_vel_guard 2>/dev/null; pkill -9 -f trusted_cmd_mux 2>/dev/null; pkill -9 -f metrics_logger 2>/dev/null; sleep 3
      python3 -u src/geofence_enforcer/experiments/run_gazebo_s1_s6.py \
        --method "$1" --scenario S5 --seeds 1 --seed-offset "$S" --no-sweep \
        --intensity wh_hijack_r09 --output "$OUT/res_${2}_v${i}.jsonl" > "$OUT/${2}_v${i}.log" 2>&1
      cp -f /tmp/position_monitor.log "$OUT/posmon_${2}_v${i}.log" 2>/dev/null
      cp -f /tmp/guard_standalone.log "$OUT/guard_${2}_v${i}.log" 2>/dev/null
      attempt=$((attempt+1))
      if [ -s "$OUT/res_${2}_v${i}.jsonl" ] && ! is_flaky "$OUT/res_${2}_v${i}.jsonl"; then break; else echo "-- flaky retry --"; S=$((S+100)); fi
    done
    python3 -c "import json;d=json.load(open('$OUT/res_${2}_v${i}.jsonl'));print(f'  ==> $2 v${i}: violated={d.get(\"violated\")} vc={d.get(\"violation_count\")} pmin={d.get(\"path_min_distance\")}')" 2>/dev/null
    i=$((i+1)); echo "PROGRESS $2 $i/$SEEDS $(date +%H:%M:%S)"
  done
}
run_cell no_guard ng_r09
run_cell geofence gf_r09
echo "W4_R09_FIRM_DONE $(date +%H:%M:%S)"
