#!/bin/bash
# §3/§4 seed expansion: coordinated dual-channel (LiDAR+odom) boundary sweep.
# The paper's "Coordinated Dual-Channel Boundary" appendix currently has only 3 seeds/epsilon.
# This bumps every epsilon to 6 fresh seeds so each incursion rate gets a Wilson CI and the
# evaded->caught transition at tau_c=0.95 is statistically pinned. Attacker synchronizes an odom
# spoof with the LiDAR spoof holding c=amcl-odom near epsilon; PETSE (cusum) is evaded iff the
# OBSERVED residual stays under tau_c. Uses the WORKTREE runner (POS_CHECK-INFRA fix + coord code).
set +e
cd /home/jim/ros2_motion_planning_tutorials
source /opt/ros/jazzy/setup.bash; source install/setup.bash
export PETSE_DETECTION_MODE=cusum
WT=/home/jim/ros2_motion_planning_tutorials/.claude/worktrees/fix-poscheck-infra
RUNNER="$WT/src/geofence_enforcer/experiments/run_gazebo_s1_s6.py"
OUT=experiment_results/gazebo_s1_s6/coord_boundary; mkdir -p "$OUT"
SEEDS=6; MAX_RETRY=3

is_flaky(){ python3 -c "
import json,sys
try: d=json.load(open('$1'))
except: sys.exit(0)
if not d.get('is_valid_result', True): sys.exit(0)
if d.get('is_infra_failure'): sys.exit(0)
r=(d.get('reason') or '').lower()
sys.exit(0 if ('infrastructure' in r or 'nav2 rejected' in r or 'failed to start' in r or 'did' in r and 'move' in r) else 1)"; }

run_cell(){  # $1=intensity  $2=tag
  local S=0 valid=0
  while [ "$valid" -lt "$SEEDS" ]; do
    local attempt=0
    while [ "$attempt" -lt "$MAX_RETRY" ]; do
      echo "==== $2 v$valid seed=$S attempt=$attempt $(date +%H:%M:%S) ===="
      pkill -9 -f "gz sim" 2>/dev/null; pkill -9 -f cmd_vel_guard 2>/dev/null
      pkill -9 -f attack_odom_spoofing 2>/dev/null; pkill -9 -f metrics_logger 2>/dev/null; sleep 3
      python3 -u "$RUNNER" \
        --method geofence --scenario S5 --seeds 1 --seed-offset "$S" --no-sweep \
        --intensity "$1" --output "$OUT/res_${2}_v${valid}.jsonl" > "$OUT/${2}_v${valid}.log" 2>&1
      cp -f /tmp/guard_standalone.log "$OUT/guard_${2}_v${valid}.log" 2>/dev/null
      cp -f /tmp/odom_coord.log "$OUT/odom_${2}_v${valid}.log" 2>/dev/null
      cp -f /tmp/position_monitor.log "$OUT/posmon_${2}_v${valid}.log" 2>/dev/null
      S=$((S+1)); attempt=$((attempt+1))
      if [ -s "$OUT/res_${2}_v${valid}.jsonl" ] && ! is_flaky "$OUT/res_${2}_v${valid}.jsonl"; then break; else echo "-- flaky retry --"; fi
    done
    python3 -c "import json;d=json.load(open('$OUT/res_${2}_v${valid}.jsonl'));print(f'  ==> $2 v${valid}: violated={d.get(\"violated\")} vc={d.get(\"violation_count\")} pmin={d.get(\"path_min_distance\")} valid={d.get(\"is_valid_result\")} infra={d.get(\"is_infra_failure\")} | {(d.get(\"reason\") or \"\")[:46]}')" 2>/dev/null
    valid=$((valid+1)); echo "PROGRESS $2 $valid/$SEEDS $(date +%H:%M:%S)"
  done
}

# 7 epsilon points matching the paper's boundary sweep.
run_cell coord_eps00  eps00
run_cell coord_eps03  eps03
run_cell coord_eps06  eps06
run_cell coord_eps08  eps08
run_cell coord_eps095 eps095
run_cell coord_eps11  eps11
run_cell coord_eps13  eps13
echo "COORD_BOUNDARY_EXPAND_DONE $(date +%H:%M:%S)"
