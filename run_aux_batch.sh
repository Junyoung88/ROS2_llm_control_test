#!/bin/bash
# 3rd-channel (aux) defense batch: coordinated attack (coord_eps00) WITH aux × 3 seeds
# (expect aux to fire when the lure succeeds → 0 incursion) + clean WITH aux × 2 (must NOT
# false-alarm → robot reaches goal). Guard polls gz true pose internally (PETSE_ENABLE_AUX).
cd /home/jim/ros2_motion_planning_tutorials
source /opt/ros/jazzy/setup.bash; source install/setup.bash
export PETSE_ENFORCE_POSE=amcl PETSE_OFFSET_THRESH=0.95 PETSE_JUMP_THRESH=0.45 PETSE_DETECTION_MODE=cusum
export PETSE_ENABLE_AUX=1 PETSE_AUX_THRESH=0.95
OUT=experiment_results/gazebo_s1_s6/money_2x2/coord; mkdir -p "$OUT"

run_one () {  # intensity tag seed
  echo "==== $2 seed=$3 $(date +%H:%M:%S) ===="
  pkill -9 -f "gz sim"; pkill -9 -f attack_scan; pkill -9 -f attack_odom; pkill -9 -f cmd_vel_guard; pkill -9 -f metrics_logger; pkill -9 -f "relay /odom /odom_spoofed"; sleep 3
  python3 -u src/geofence_enforcer/experiments/run_gazebo_s1_s6.py --method geofence --scenario S5 --seeds 1 --seed-offset "$3" --no-sweep --intensity "$1" --output "$OUT/res_$2.jsonl" > "$OUT/$2.log" 2>&1
  cp -f /tmp/position_monitor.log "$OUT/posmon_$2.log" 2>/dev/null
  cp -f /tmp/guard_standalone.log "$OUT/guard_$2.log" 2>/dev/null
  echo "-- $2 done --"
}

run_one coord_eps00     auxcoord_v0  0
run_one coord_eps00     auxcoord_v1  1
run_one coord_eps00     auxcoord_v2  2
run_one warehouse_clean6 auxclean_v0 0
run_one warehouse_clean6 auxclean_v1 1
echo "AUX BATCH DONE"
