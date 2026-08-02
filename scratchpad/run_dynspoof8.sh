#!/bin/bash
# dyn_spoof to 8 VALID (navigating) seeds + no_guard 4. Key: treat a NON-NAVIGATING trial
# (robot never moved, "Navigation aborted", ground-truth max_x<0.3) as flaky and RETRY, so we
# collect real defense outcomes not stalls. yaml restored to default zone before each trial.
set +e
cd /home/jim/ros2_motion_planning_tutorials
source /opt/ros/jazzy/setup.bash; source install/setup.bash
export PETSE_DETECTION_MODE=cusum
WT=/home/jim/ros2_motion_planning_tutorials/.claude/worktrees/fix-poscheck-infra
RUNNER="$WT/src/geofence_enforcer/experiments/run_gazebo_s1_s6.py"
OUT="$WT/experiment_results/gazebo_s1_s6/dynspoof8"; mkdir -p "$OUT"
MAX_RETRY=4
restore_yaml(){ python3 -c "
import glob,os
WS='/home/jim/ros2_motion_planning_tutorial'.replace('tutorial','tutorials')
hdr='uncertainty:\n  k_sigma: 3.0\n  localization_sigma: 0.15\n  tracking_error: 0.05\n  v_max: 0.5\n  latency: 0.1\n\nzones:\n'
a,b,c,d=1.5,4.5,-4.0,-1.0
body='  - name: \"wh_zone_0\"\n    type: \"forbidden\"\n    priority: 10\n    vertices:\n      - {x: %s, y: %s}\n      - {x: %s, y: %s}\n      - {x: %s, y: %s}\n      - {x: %s, y: %s}\n'%(a,c,b,c,b,d,a,d)
for p in glob.glob(os.path.join(WS,'**','warehouse_geofence.yaml'),recursive=True):
    if '/build/' in p: continue
    open(p,'w').write(hdr+body)
"; }
# a trial is USABLE only if the robot actually navigated (ground-truth max_x>0.3)
navigated(){ python3 -c "
import json,os,sys
p='$1'
if not os.path.exists(p): sys.exit(1)
mx=max((json.loads(l).get('x',0) for l in open(p) if l.strip()),default=0)
sys.exit(0 if mx>0.3 else 1)"; }
run_cell(){  # $1=intensity $2=method $3=tag $4=n_valid
  local S=200 valid=0 N="$4"
  while [ "$valid" -lt "$N" ]; do
    local a=0 ok=0
    while [ "$a" -lt "$MAX_RETRY" ]; do
      echo "==== $3 v$valid seed=$S a=$a $(date +%H:%M:%S) ===="
      restore_yaml
      python3 -c "import subprocess;[subprocess.run(['pkill','-9','-f',p]) for p in ['gz sim','ros_gz','cmd_vel_guard','run_gazebo_s1_s6','position_monitor_node']]" 2>/dev/null
      sleep 4; rm -f /tmp/guard_standalone.log /tmp/position_monitor.log
      timeout 750 python3 -u "$RUNNER" --method "$2" --scenario S5 --seeds 1 --seed-offset "$S" --no-sweep \
        --intensity "$1" --output "$OUT/res_${3}_v${valid}.jsonl" > "$OUT/run_${3}_v${valid}.log" 2>&1
      cp -f /tmp/guard_standalone.log "$OUT/guard_${3}_v${valid}.log" 2>/dev/null
      cp -f /tmp/position_monitor.log "$OUT/posmon_${3}_v${valid}.log" 2>/dev/null
      S=$((S+1)); a=$((a+1))
      if navigated "$OUT/posmon_${3}_v${valid}.log"; then ok=1; break; else echo "-- did not navigate, retry --"; fi
    done
    if [ "$ok" -eq 1 ]; then
      python3 -c "import json,os;d=json.load(open('$OUT/res_${3}_v${valid}.jsonl'));p='$OUT/posmon_${3}_v${valid}.log';mx=max((__import__('json').loads(l).get('x',0) for l in open(p) if l.strip()),default=0);g='$OUT/guard_${3}_v${valid}.log';fs=os.path.exists(g) and 'spoof fail-stop' in open(g,errors='ignore').read().lower();print(f'  ==> $3 v${valid}: violated={d.get(\"violated\")} moved_x={round(mx,2)} spoof_failstop={fs} pmin={round(d.get(\"path_min_distance\") or -1,2)}')" 2>/dev/null
      valid=$((valid+1)); echo "PROGRESS $3 $valid/$N $(date +%H:%M:%S)"
    else
      echo "SKIP $3 v$valid: no navigation after $MAX_RETRY retries"; valid=$((valid+1))
    fi
  done
}
echo "DYNSPOOF8_START $(date +%H:%M:%S)"
run_cell dyn_spoof geofence gf 8
run_cell dyn_spoof no_guard ng 4
echo "DYNSPOOF8_DONE $(date +%H:%M:%S)"
