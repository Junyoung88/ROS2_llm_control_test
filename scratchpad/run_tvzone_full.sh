#!/bin/bash
# Time-varying forbidden zone (R3-4): a keep-out is ACTIVATED mid-navigation, AFTER the goal was
# approved. Approval-time-only methods (no_guard, SELP) keep driving in; methods with continuous
# runtime re-verification (PETSE geofence; also the guard-equipped CBF) enforce the just-activated
# zone -> fail-stop. Goal (6.5,0); geofence.yaml starts with NO forbidden zone (goal_gate admits
# for all); external injector activates x[4,6] via /petse/inject_zone when robot x in [1.2,3.7].
set +e
cd /home/jim/ros2_motion_planning_tutorials
source /opt/ros/jazzy/setup.bash; source install/setup.bash
OUT=experiment_results/gazebo_s1_s6/tvzone; mkdir -p "$OUT"
SEEDS=5; MAX_RETRY=3

NOZONE='uncertainty:
  k_sigma: 3.0
  localization_sigma: 0.15
  tracking_error: 0.05
  v_max: 0.5
  latency: 0.1

zones: []
'
mapfile -t CFGS < <(find /home/jim/ros2_motion_planning_tutorials -name geofence.yaml -path '*geofence_policy_enforcer/config*' -not -path '*build*')
for c in "${CFGS[@]}"; do cp -f "$c" "$c.tvbak"; printf '%s' "$NOZONE" > "$c"; done
echo "wrote empty-forbidden geofence.yaml to ${#CFGS[@]} copies"
restore_cfg(){ for c in "${CFGS[@]}"; do [ -f "$c.tvbak" ] && mv -f "$c.tvbak" "$c"; done; echo "restored geofence.yaml"; }
trap restore_cfg EXIT

injector(){
  local log=/tmp/position_monitor.log
  for i in $(seq 1 400); do
    x=$(tail -1 "$log" 2>/dev/null | python3 -c "import sys,json
try: print(json.loads(sys.stdin.readline()).get('x',''))
except: print('')" 2>/dev/null)
    if [ -n "$x" ]; then
      ok=$(python3 -c "print(1 if 0.5 < ${x:-99} < 2.8 else 0)" 2>/dev/null)
      if [ "$ok" = "1" ]; then
        ros2 topic pub --once /petse/inject_zone std_msgs/msg/String "{data: '4.0,6.0,-1.0,1.0'}" >/dev/null 2>&1
        echo "  [INJECTOR] activated x[4,6] at x=$x"; return; fi
    fi
    sleep 0.5
  done
  echo "  [INJECTOR] timed out; last x=$x"
}
is_flaky(){ python3 -c "
import json,sys
try: d=json.load(open('$1'))
except: sys.exit(0)
r=(d.get('reason') or '').lower()
sys.exit(0 if ('infrastructure' in r or 'nav2 rejected' in r) else 1)"; }

run_cell(){  # method tag
  local S=0 valid=0
  while [ "$valid" -lt "$SEEDS" ]; do
    local attempt=0
    while [ "$attempt" -lt "$MAX_RETRY" ]; do
      echo "==== $2 v$valid seed=$S $(date +%H:%M:%S) ===="
      pkill -9 -f "gz sim" 2>/dev/null; pkill -9 -f cmd_vel_guard 2>/dev/null; pkill -9 -f metrics_logger 2>/dev/null; sleep 3
      ( sleep 40; injector ) & INJ=$!
      python3 -u src/geofence_enforcer/experiments/run_gazebo_s1_s6.py \
        --method "$1" --scenario S4 --seeds 1 --seed-offset "$S" --no-sweep \
        --intensity tvzone --output "$OUT/res_${2}_v${valid}.jsonl" > "$OUT/${2}_v${valid}.log" 2>&1
      kill $INJ 2>/dev/null
      S=$((S+1)); attempt=$((attempt+1))
      if [ -s "$OUT/res_${2}_v${valid}.jsonl" ] && ! is_flaky "$OUT/res_${2}_v${valid}.jsonl"; then break; else echo "-- flaky retry --"; fi
    done
    valid=$((valid+1)); echo "PROGRESS $2 $valid/$SEEDS $(date +%H:%M:%S)"
  done
}
for M in no_guard selp_proper cbf_inflated geofence; do
  run_cell "$M" "tvzone_${M}"
done
echo "TVZONE FULL DONE $(date +%H:%M:%S)"
