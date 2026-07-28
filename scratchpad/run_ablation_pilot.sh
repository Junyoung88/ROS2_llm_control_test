#!/bin/bash
# Component-ablation PILOT: 4 conditions x S5 wh_hijack, 1 seed each, to verify the wiring
# (spatial/cross-channel independent flags). Expected on S5 (localization spoof):
#   planning-only (spatial=off, spoof=off): breached
#   runtime-only  (spatial=on,  spoof=off): breached (spatial trusts the spoofed pose)
#   odom-gate     (spatial=off, spoof=on):  defended (cross-channel catches spoof)
#   full          (spatial=on,  spoof=on):  defended
set +e
cd /home/jim/ros2_motion_planning_tutorials
source /opt/ros/jazzy/setup.bash; source install/setup.bash
export PETSE_DETECTION_MODE=cusum
OUT=experiment_results/gazebo_s1_s6/ablation_pilot; mkdir -p "$OUT"
run_cond(){  # $1=tag $2=SPOOF_DET $3=SPATIAL_CHECK
  export PETSE_SPOOF_DET="$2" PETSE_SPATIAL_CHECK="$3"
  echo "==== $1 (spoof=$2 spatial=$3) $(date +%H:%M:%S) ===="
  pkill -9 -f "gz sim" 2>/dev/null; pkill -9 -f cmd_vel_guard 2>/dev/null; sleep 3
  python3 -u src/geofence_enforcer/experiments/run_gazebo_s1_s6.py \
    --method geofence --scenario S5 --seeds 1 --seed-offset 0 --no-sweep \
    --intensity wh_hijack --output "$OUT/res_${1}.jsonl" > "$OUT/${1}.log" 2>&1
  cp -f /tmp/position_monitor.log "$OUT/posmon_${1}.log" 2>/dev/null
  python3 -c "import json;d=json.load(open('$OUT/res_${1}.jsonl'));print(f'  ==> $1: violated={d.get(\"violated\")} vc={d.get(\"violation_count\")} pmin={d.get(\"path_min_distance\")} | {(d.get(\"reason\") or \"\")[:45]}')" 2>/dev/null
  unset PETSE_SPOOF_DET PETSE_SPATIAL_CHECK
}
run_cond planning  false false
run_cond runtime   false true
run_cond odomgate  true  false
run_cond full      true  true
echo "ABLATION PILOT DONE $(date +%H:%M:%S)"
