#!/bin/bash
# A: cluttered multi-zone. (1) clutter_path: geofence 4 + no_guard 3 (path threads 3 zones).
# (2) clutter_spoof: geofence 4 + no_guard 3 (S5 spoof amid clutter). Closes the paper's
# self-admitted 'cluttered multi-zone remains future work' caveat.
set +e
cd /home/jim/ros2_motion_planning_tutorials
source /opt/ros/jazzy/setup.bash; source install/setup.bash
export PETSE_DETECTION_MODE=cusum
WT=/home/jim/ros2_motion_planning_tutorials/.claude/worktrees/fix-poscheck-infra
RUNNER="$WT/src/geofence_enforcer/experiments/run_gazebo_s1_s6.py"
OUT="$WT/experiment_results/gazebo_s1_s6/clutter"; mkdir -p "$OUT"
MAX_RETRY=3
is_flaky(){ python3 -c "
import json,sys
try: d=json.load(open('$1'))
except: sys.exit(0)
if not d.get('is_valid_result',True) or d.get('is_infra_failure'): sys.exit(0)
r=(d.get('reason') or '').lower()
sys.exit(0 if ('infrastructure' in r or 'failed to start' in r or 'nav2 rejected' in r or 'timed out' in r) else 1)"; }
run_cell(){  # $1=intensity $2=method $3=tag $4=n
  local S=0 valid=0 N="$4"
  while [ "$valid" -lt "$N" ]; do
    local a=0
    while [ "$a" -lt "$MAX_RETRY" ]; do
      echo "==== $3 v$valid seed=$S a=$a $(date +%H:%M:%S) ===="
      pkill -9 -f 'gz sim' 2>/dev/null; pkill -9 -f ros_gz 2>/dev/null; pkill -9 -f cmd_vel_guard 2>/dev/null
      pkill -9 -f run_gazebo_s1_s6 2>/dev/null; rm -f /tmp/guard_standalone.log /tmp/position_monitor.log; sleep 4
      timeout 900 python3 -u "$RUNNER" --method "$2" --scenario S5 --seeds 1 --seed-offset "$S" --no-sweep \
        --intensity "$1" --output "$OUT/res_${3}_v${valid}.jsonl" > "$OUT/run_${3}_v${valid}.log" 2>&1
      cp -f /tmp/guard_standalone.log "$OUT/guard_${3}_v${valid}.log" 2>/dev/null
      cp -f /tmp/position_monitor.log "$OUT/posmon_${3}_v${valid}.log" 2>/dev/null
      S=$((S+1)); a=$((a+1))
      if [ -s "$OUT/res_${3}_v${valid}.jsonl" ] && ! is_flaky "$OUT/res_${3}_v${valid}.jsonl"; then break; else echo "-- flaky --"; fi
    done
    python3 -c "import json;d=json.load(open('$OUT/res_${3}_v${valid}.jsonl'));print(f'  ==> $3 v${valid}: violated={d.get(\"violated\")} vc={d.get(\"violation_count\")} pmin={d.get(\"path_min_distance\")} | {(d.get(\"reason\") or \"\")[:40]}')" 2>/dev/null
    valid=$((valid+1)); echo "PROGRESS $3 $valid/$N $(date +%H:%M:%S)"
  done
}
echo "CLUTTER_START $(date +%H:%M:%S)"
run_cell clutter_path  geofence gf_path  4
run_cell clutter_path  no_guard ng_path  3
run_cell clutter_spoof geofence gf_spoof 4
run_cell clutter_spoof no_guard ng_spoof 3
echo "CLUTTER_DONE $(date +%H:%M:%S)"
