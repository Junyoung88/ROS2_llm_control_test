#!/bin/bash
# Defense-aware slow-attacker sweep (NDSS adaptive-attacker concern): an attacker who knows
# PETSE's cross-channel CUSUM detector ramps the localization bias ever more slowly to keep each
# per-update residual jump under the detector before reaching the zone. We sweep the bias rate
# (0.15 -> 0.08 -> 0.04 -> 0.02 m/s, heading-compensated world +X lure) against PETSE (cusum) and
# measure whether the accumulated residual still fail-stops the robot before zone entry, and the
# minimum clearance / in-zone samples. Warehouse map (AMCL), S5 scan-spoofing.
set +e
cd /home/jim/ros2_motion_planning_tutorials
source /opt/ros/jazzy/setup.bash; source install/setup.bash
export PETSE_DETECTION_MODE=cusum
OUT=experiment_results/gazebo_s1_s6/slowbias; mkdir -p "$OUT"
SEEDS=2; MAX_RETRY=3
is_flaky(){ python3 -c "
import json,sys
try: d=json.load(open('$1'))
except: sys.exit(0)
r=(d.get('reason') or '').lower()
sys.exit(0 if ('infrastructure' in r or 'nav2 rejected' in r or 'failed to start' in r) else 1)"; }

run_cell(){  # $1=intensity  $2=tag
  local S=0 valid=0
  while [ "$valid" -lt "$SEEDS" ]; do
    local attempt=0
    while [ "$attempt" -lt "$MAX_RETRY" ]; do
      echo "==== $2 v$valid seed=$S attempt=$attempt $(date +%H:%M:%S) ===="
      pkill -9 -f "gz sim" 2>/dev/null; pkill -9 -f cmd_vel_guard 2>/dev/null; pkill -9 -f metrics_logger 2>/dev/null; sleep 3
      python3 -u src/geofence_enforcer/experiments/run_gazebo_s1_s6.py \
        --method geofence --scenario S5 --seeds 1 --seed-offset "$S" --no-sweep \
        --intensity "$1" --output "$OUT/res_${2}_v${valid}.jsonl" > "$OUT/${2}_v${valid}.log" 2>&1
      cp -f /tmp/position_monitor.log "$OUT/posmon_${2}_v${valid}.log" 2>/dev/null
      S=$((S+1)); attempt=$((attempt+1))
      if [ -s "$OUT/res_${2}_v${valid}.jsonl" ] && ! is_flaky "$OUT/res_${2}_v${valid}.jsonl"; then break; else echo "-- flaky retry --"; fi
    done
    python3 -c "import json;d=json.load(open('$OUT/res_${2}_v${valid}.jsonl'));print(f'  ==> $2 v${valid}: violated={d.get(\"violated\")} vc={d.get(\"violation_count\")} pmin={d.get(\"path_min_distance\")} | {(d.get(\"reason\") or \"\")[:50]}')" 2>/dev/null
    valid=$((valid+1)); echo "PROGRESS $2 $valid/$SEEDS $(date +%H:%M:%S)"
  done
}
run_cell bias_hc_mid   rate015
run_cell bias_hc_slow  rate008
run_cell bias_hc_vslow rate004
run_cell bias_hc_xslow rate002
echo "SLOWBIAS SWEEP DONE $(date +%H:%M:%S)"
