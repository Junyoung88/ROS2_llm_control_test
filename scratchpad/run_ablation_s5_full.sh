#!/bin/bash
# Component-ablation FULL, S5 (localization-spoofing hijack), 4 conditions x 20 seeds.
# Isolates the cross-channel gate's independent contribution on a sensor-spoofing attack.
#   planning : spatial=off cross=off  (goal/path gate only)
#   runtime  : spatial=on  cross=off  (spatial runtime envelope only)
#   odomgate : spatial=off cross=on   (AMCL-odom cross-channel gate only)
#   full     : spatial=on  cross=on   (full PETSE)
# Expected: cross-channel gate (odomgate, full) defends the spoof; planning is breached.
set +e
cd /home/jim/ros2_motion_planning_tutorials
source /opt/ros/jazzy/setup.bash; source install/setup.bash
export PETSE_DETECTION_MODE=cusum
OUT=experiment_results/gazebo_s1_s6/ablation_s5; mkdir -p "$OUT"
SEEDS=20; MAX_RETRY=3
is_flaky(){ python3 -c "
import json,sys
try: d=json.load(open('$1'))
except: sys.exit(0)
r=(d.get('reason') or '').lower()
sys.exit(0 if ('infrastructure' in r or 'nav2 rejected' in r or 'failed to start' in r or \"didn't move\" in r) else 1)"; }

run_cond(){  # $1=tag $2=SPOOF_DET $3=SPATIAL_CHECK
  export PETSE_SPOOF_DET="$2" PETSE_SPATIAL_CHECK="$3"
  local S=0 valid=0
  while [ "$valid" -lt "$SEEDS" ]; do
    local attempt=0
    while [ "$attempt" -lt "$MAX_RETRY" ]; do
      echo "==== $1 v$valid seed=$S (spoof=$2 spatial=$3) $(date +%H:%M:%S) ===="
      pkill -9 -f "gz sim" 2>/dev/null; pkill -9 -f cmd_vel_guard 2>/dev/null; pkill -9 -f metrics_logger 2>/dev/null; sleep 3
      python3 -u src/geofence_enforcer/experiments/run_gazebo_s1_s6.py \
        --method geofence --scenario S5 --seeds 1 --seed-offset "$S" --no-sweep \
        --intensity wh_hijack --output "$OUT/res_${1}_v${valid}.jsonl" > "$OUT/${1}_v${valid}.log" 2>&1
      cp -f /tmp/position_monitor.log "$OUT/posmon_${1}_v${valid}.log" 2>/dev/null
      S=$((S+1)); attempt=$((attempt+1))
      if [ -s "$OUT/res_${1}_v${valid}.jsonl" ] && ! is_flaky "$OUT/res_${1}_v${valid}.jsonl"; then break; else echo "-- flaky retry --"; fi
    done
    python3 -c "import json;d=json.load(open('$OUT/res_${1}_v${valid}.jsonl'));print(f'  ==> $1 v${valid}: violated={d.get(\"violated\")} vc={d.get(\"violation_count\")} pmin={d.get(\"path_min_distance\")}')" 2>/dev/null
    valid=$((valid+1)); echo "PROGRESS $1 $valid/$SEEDS $(date +%H:%M:%S)"
  done
  unset PETSE_SPOOF_DET PETSE_SPATIAL_CHECK
}
run_cond planning  false false
run_cond runtime   false true
run_cond odomgate  true  false
run_cond full      true  true
echo "ABLATION S5 FULL DONE $(date +%H:%M:%S)"
