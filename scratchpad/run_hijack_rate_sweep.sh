#!/bin/bash
# Defense-aware slow-attacker sweep, REDESIGNED on the stable map-consistent hijack (wh_hijack,
# which is 5/5 reliable) instead of the flaky bias_injection. The attacker ramps the map-consistent
# spoof ever more slowly (bias_rate 0.06 -> 0.03 -> 0.015 m/s) to keep each cross-channel residual
# jump under PETSE's CUSUM. Run BOTH no_guard (proves the attack still drives the true robot into
# the zone) and geofence (does PETSE's accumulated-residual detector still fail-stop in time?).
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

run_cell(){  # $1=intensity $2=method $3=tag
  local S=0 valid=0
  while [ "$valid" -lt "$SEEDS" ]; do
    local attempt=0
    while [ "$attempt" -lt "$MAX_RETRY" ]; do
      echo "==== $3 v$valid seed=$S attempt=$attempt $(date +%H:%M:%S) ===="
      pkill -9 -f "gz sim" 2>/dev/null; pkill -9 -f cmd_vel_guard 2>/dev/null; pkill -9 -f metrics_logger 2>/dev/null; sleep 3
      python3 -u src/geofence_enforcer/experiments/run_gazebo_s1_s6.py \
        --method "$2" --scenario S5 --seeds 1 --seed-offset "$S" --no-sweep \
        --intensity "$1" --output "$OUT/res_${3}_v${valid}.jsonl" > "$OUT/${3}_v${valid}.log" 2>&1
      cp -f /tmp/position_monitor.log "$OUT/posmon_${3}_v${valid}.log" 2>/dev/null
      S=$((S+1)); attempt=$((attempt+1))
      if [ -s "$OUT/res_${3}_v${valid}.jsonl" ] && ! is_flaky "$OUT/res_${3}_v${valid}.jsonl"; then break; else echo "-- flaky retry --"; fi
    done
    python3 -c "import json;d=json.load(open('$OUT/res_${3}_v${valid}.jsonl'));print(f'  ==> $3 v${valid}: violated={d.get(\"violated\")} vc={d.get(\"violation_count\")} pmin={d.get(\"path_min_distance\")} | {(d.get(\"reason\") or \"\")[:48]}')" 2>/dev/null
    valid=$((valid+1)); echo "PROGRESS $3 $valid/$SEEDS $(date +%H:%M:%S)"
  done
}
for rate in r06 r03 r015; do
  run_cell "wh_hijack_${rate}" no_guard "ng_${rate}"
  run_cell "wh_hijack_${rate}" geofence "gf_${rate}"
done
echo "HIJACK RATE SWEEP DONE $(date +%H:%M:%S)"
