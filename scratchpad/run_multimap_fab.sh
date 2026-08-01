#!/bin/bash
# Multi-map robustness: re-run the S5 cross-channel defense on a SECOND map (fab_cell.sdf,
# different zone geometry + spoof direction) to close the single-map caveat. Same cross-channel
# CUSUM gate as the warehouse S5. geofence 5 seeds (defense) + no_guard 3 seeds (attack validity).
set +e
cd /home/jim/ros2_motion_planning_tutorials
source /opt/ros/jazzy/setup.bash; source install/setup.bash
export PETSE_DETECTION_MODE=cusum
WT=/home/jim/ros2_motion_planning_tutorials/.claude/worktrees/fix-poscheck-infra
RUNNER="$WT/src/geofence_enforcer/experiments/run_gazebo_s1_s6.py"
OUT="$WT/experiment_results/gazebo_s1_s6/multimap_fab"; mkdir -p "$OUT"
MAX_RETRY=3

is_flaky(){ python3 -c "
import json,sys
try: d=json.load(open('$1'))
except: sys.exit(0)
if not d.get('is_valid_result',True) or d.get('is_infra_failure'): sys.exit(0)
r=(d.get('reason') or '').lower()
sys.exit(0 if ('infrastructure' in r or 'failed to start' in r or 'nav2 rejected' in r or 'timed out' in r) else 1)"; }

run_cell(){  # $1=method $2=tag $3=nseeds
  local S=0 valid=0 N="$3"
  while [ "$valid" -lt "$N" ]; do
    local attempt=0
    while [ "$attempt" -lt "$MAX_RETRY" ]; do
      echo "==== $2 v$valid seed=$S attempt=$attempt $(date +%H:%M:%S) ===="
      pkill -9 -f 'gz sim' 2>/dev/null; pkill -9 -f ros_gz 2>/dev/null; pkill -9 -f cmd_vel_guard 2>/dev/null
      pkill -9 -f run_gazebo_s1_s6 2>/dev/null; pkill -9 -f metrics_logger 2>/dev/null
      rm -f /tmp/guard_standalone.log /tmp/position_monitor.log; sleep 4
      timeout 900 python3 -u "$RUNNER" \
        --method "$1" --scenario S5 --seeds 1 --seed-offset "$S" --no-sweep \
        --intensity fab_spoof_hijack --output "$OUT/res_${2}_v${valid}.jsonl" > "$OUT/run_${2}_v${valid}.log" 2>&1
      cp -f /tmp/guard_standalone.log "$OUT/guard_${2}_v${valid}.log" 2>/dev/null
      cp -f /tmp/position_monitor.log "$OUT/posmon_${2}_v${valid}.log" 2>/dev/null
      S=$((S+1)); attempt=$((attempt+1))
      if [ -s "$OUT/res_${2}_v${valid}.jsonl" ] && ! is_flaky "$OUT/res_${2}_v${valid}.jsonl"; then break; else echo "-- flaky retry --"; fi
    done
    python3 -c "
import json,os
d=json.load(open('$OUT/res_${2}_v${valid}.jsonl'))
g='$OUT/guard_${2}_v${valid}.log'; fs=os.path.exists(g) and 'spoof fail-stop' in open(g,errors='ignore').read().lower()
print(f'  ==> $2 v${valid}: violated={d.get(\"violated\")} clearance={d.get(\"path_min_distance\")} spoof_failstop={fs} | {(d.get(\"reason\") or \"\")[:40]}')" 2>/dev/null
    valid=$((valid+1)); echo "PROGRESS $2 $valid/$N $(date +%H:%M:%S)"
  done
}

echo "MULTIMAP_FAB_START $(date +%H:%M:%S)"
run_cell geofence gf 5   # PETSE defense on the second map
run_cell no_guard ng 3   # attack-validity control
echo "MULTIMAP_FAB_DONE $(date +%H:%M:%S)"
