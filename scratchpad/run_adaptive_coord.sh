#!/bin/bash
# W4-Q4: adaptive-coordination attack. A CUSUM-aware attacker keeps perfect dual-channel
# coordination (ε=0) AND ramps the offset ever more slowly (0.06/0.03/0.015 m/s) to keep the
# accumulated cross-channel evidence under the detector until it reaches the zone. Tests whether
# slowing the ramp lets a coordinated spoof out-run the accumulator. geofence 4 seeds/rate
# (defense) + no_guard 2 seeds/rate (does the slow coord spoof even reach the zone unguarded?).
set +e
cd /home/jim/ros2_motion_planning_tutorials
source /opt/ros/jazzy/setup.bash; source install/setup.bash
export PETSE_DETECTION_MODE=cusum
WT=/home/jim/ros2_motion_planning_tutorials/.claude/worktrees/fix-poscheck-infra
RUNNER="$WT/src/geofence_enforcer/experiments/run_gazebo_s1_s6.py"
OUT="$WT/experiment_results/gazebo_s1_s6/adaptive_coord"; mkdir -p "$OUT"
MAX_RETRY=3

is_flaky(){ python3 -c "
import json,sys
try: d=json.load(open('$1'))
except: sys.exit(0)
if not d.get('is_valid_result',True) or d.get('is_infra_failure'): sys.exit(0)
r=(d.get('reason') or '').lower()
sys.exit(0 if ('infrastructure' in r or 'failed to start' in r or 'nav2 rejected' in r or 'timed out' in r) else 1)"; }

run_cell(){  # $1=intensity $2=method $3=tag $4=nseeds
  local S=0 valid=0 N="$4"
  while [ "$valid" -lt "$N" ]; do
    local attempt=0
    while [ "$attempt" -lt "$MAX_RETRY" ]; do
      echo "==== $3 v$valid seed=$S attempt=$attempt $(date +%H:%M:%S) ===="
      pkill -9 -f 'gz sim' 2>/dev/null; pkill -9 -f ros_gz 2>/dev/null; pkill -9 -f cmd_vel_guard 2>/dev/null
      pkill -9 -f attack_odom_spoofing 2>/dev/null; pkill -9 -f run_gazebo_s1_s6 2>/dev/null
      rm -f /tmp/guard_standalone.log /tmp/position_monitor.log; sleep 4
      timeout 900 python3 -u "$RUNNER" \
        --method "$2" --scenario S5 --seeds 1 --seed-offset "$S" --no-sweep \
        --intensity "$1" --output "$OUT/res_${3}_v${valid}.jsonl" > "$OUT/run_${3}_v${valid}.log" 2>&1
      cp -f /tmp/guard_standalone.log "$OUT/guard_${3}_v${valid}.log" 2>/dev/null
      cp -f /tmp/position_monitor.log "$OUT/posmon_${3}_v${valid}.log" 2>/dev/null
      S=$((S+1)); attempt=$((attempt+1))
      if [ -s "$OUT/res_${3}_v${valid}.jsonl" ] && ! is_flaky "$OUT/res_${3}_v${valid}.jsonl"; then break; else echo "-- flaky retry --"; fi
    done
    python3 -c "
import json,os
d=json.load(open('$OUT/res_${3}_v${valid}.jsonl'))
g='$OUT/guard_${3}_v${valid}.log'; fs=os.path.exists(g) and 'spoof fail-stop' in open(g,errors='ignore').read().lower()
print(f'  ==> $3 v${valid}: violated={d.get(\"violated\")} clearance={d.get(\"path_min_distance\")} spoof_failstop={fs} | {(d.get(\"reason\") or \"\")[:38]}')" 2>/dev/null
    valid=$((valid+1)); echo "PROGRESS $3 $valid/$N $(date +%H:%M:%S)"
  done
}

echo "ADAPTIVE_COORD_START $(date +%H:%M:%S)"
for rate in slow vslow xslow; do
  run_cell "adcoord_${rate}" geofence "gf_${rate}" 4    # PETSE defense
  run_cell "adcoord_${rate}" no_guard "ng_${rate}" 2    # attack-validity control
done
echo "ADAPTIVE_COORD_DONE $(date +%H:%M:%S)"
