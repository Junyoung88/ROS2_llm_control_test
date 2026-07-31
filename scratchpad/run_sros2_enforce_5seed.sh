#!/bin/bash
# W2 replication: PETSE S5 defense under whole-graph SROS2 Enforce, 5 seeds.
# Keystore is already in the success state (access-control-only governance + wildcard
# rcl-service perms + deny_rule(rt/cmd_vel) + rt/* allow); nodock nav2 launch is
# PETSE_SROS2-conditional. Each seed: robot should navigate under the S5 spoof, PETSE's
# cross-channel gate should fail-stop it before the zone. Collect violated + clearance
# (path_min_distance) + whether the spoof fail-stop fired. Flaky (gz/EKF) trials retry.
set +e
cd /home/jim/ros2_motion_planning_tutorials
source /opt/ros/jazzy/setup.bash
source install/setup.bash
WT=/home/jim/ros2_motion_planning_tutorials/.claude/worktrees/fix-poscheck-infra
export PETSE_DETECTION_MODE=cusum
export PETSE_USE_MUX=1
export PETSE_SROS2=enforce
export PETSE_SROS2_KEYSTORE="$WT/sros2_full/keystore"
unset FASTRTPS_DEFAULT_PROFILES_FILE
RUNNER="$WT/src/geofence_enforcer/experiments/run_gazebo_s1_s6.py"
OUT="$WT/experiment_results/sros2_full/enforce_5seed"; mkdir -p "$OUT"
SEEDS=5; MAX_RETRY=3

is_flaky(){ python3 -c "
import json,sys
try: d=json.load(open('$1'))
except: sys.exit(0)
if not d.get('is_valid_result',True) or d.get('is_infra_failure'): sys.exit(0)
r=(d.get('reason') or '').lower()
# gz/EKF/nav bringup flakiness -> retry; a real defend/abort is NOT flaky
sys.exit(0 if ('infrastructure' in r or 'failed to start' in r or 'nav2 rejected' in r or 'timed out' in r) else 1)"; }

echo "ENFORCE_5SEED_START $(date +%H:%M:%S)"
S=0; valid=0
while [ "$valid" -lt "$SEEDS" ]; do
  attempt=0
  while [ "$attempt" -lt "$MAX_RETRY" ]; do
    echo "==== v$valid seed=$S attempt=$attempt $(date +%H:%M:%S) ===="
    pkill -9 -f 'gz sim' 2>/dev/null; pkill -9 -f ros_gz 2>/dev/null; pkill -9 -f cmd_vel_guard 2>/dev/null
    pkill -9 -f trusted_cmd_mux 2>/dev/null; pkill -9 -f run_gazebo_s1_s6 2>/dev/null; pkill -9 -f opennav_docking 2>/dev/null
    rm -f /tmp/guard_standalone.log /tmp/position_monitor.log
    sleep 4
    timeout 900 python3 -u "$RUNNER" \
      --method geofence --scenario S5 --seeds 1 --seed-offset "$S" --no-sweep \
      --intensity wh_hijack --output "$OUT/res_v${valid}.jsonl" > "$OUT/run_v${valid}.log" 2>&1
    cp -f /tmp/guard_standalone.log "$OUT/guard_v${valid}.log" 2>/dev/null
    cp -f /tmp/position_monitor.log "$OUT/posmon_v${valid}.log" 2>/dev/null
    S=$((S+1)); attempt=$((attempt+1))
    if [ -s "$OUT/res_v${valid}.jsonl" ] && ! is_flaky "$OUT/res_v${valid}.jsonl"; then break; else echo "-- flaky retry --"; fi
  done
  # per-seed summary: violated, clearance, spoof-fail-stop, ground-truth max x
  python3 -c "
import json,os,re
d=json.load(open('$OUT/res_v${valid}.jsonl'))
g='$OUT/guard_v${valid}.log'; fs=os.path.exists(g) and 'spoof fail-stop' in open(g,errors='ignore').read().lower()
p='$OUT/posmon_v${valid}.log'; mx=0.0
if os.path.exists(p):
    for l in open(p):
        try: mx=max(mx, json.loads(l).get('x',0))
        except: pass
print(f'  ==> v${valid}: violated={d.get(\"violated\")} clearance(pmin)={d.get(\"path_min_distance\")} spoof_failstop={fs} gt_max_x={round(mx,2)} | {(d.get(\"reason\") or \"\")[:40]}')
" 2>/dev/null
  valid=$((valid+1)); echo "PROGRESS $valid/$SEEDS $(date +%H:%M:%S)"
done
echo "ENFORCE_5SEED_DONE $(date +%H:%M:%S)"
