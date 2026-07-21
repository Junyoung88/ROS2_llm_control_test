#!/bin/bash
# Boundary-sharpening: coordinated attack at ε near τ_c=0.95 (0.8, 0.95, 1.1) × 3 seeds,
# to pin the evaded→caught transition exactly at the consistency threshold.
cd /home/jim/ros2_motion_planning_tutorials
source /opt/ros/jazzy/setup.bash; source install/setup.bash
export PETSE_ENFORCE_POSE=amcl PETSE_OFFSET_THRESH=0.95 PETSE_JUMP_THRESH=0.45 PETSE_DETECTION_MODE=cusum
OUT=experiment_results/gazebo_s1_s6/money_2x2/coord; mkdir -p "$OUT"
SEEDS=3; MAX_RETRY=3
is_flaky () { ! grep -q "spoofmon" "$1" 2>/dev/null; }
run_cell () {  # intensity tag
  local valid=0 S=0
  while [ "$valid" -lt "$SEEDS" ]; do
    local attempt=0
    while [ "$attempt" -lt "$MAX_RETRY" ]; do
      echo "==== $2 v$valid seed=$S attempt=$attempt $(date +%H:%M:%S) ===="
      pkill -9 -f "gz sim"; pkill -9 -f attack_scan; pkill -9 -f attack_odom; pkill -9 -f cmd_vel_guard; pkill -9 -f metrics_logger; pkill -9 -f "relay /odom /odom_spoofed"; sleep 3
      python3 -u src/geofence_enforcer/experiments/run_gazebo_s1_s6.py --method geofence --scenario S5 --seeds 1 --seed-offset "$S" --no-sweep --intensity "$1" --output "$OUT/res_$2_v$valid.jsonl" > "$OUT/$2_v$valid.log" 2>&1
      cp -f /tmp/position_monitor.log "$OUT/posmon_$2_v$valid.log" 2>/dev/null
      cp -f /tmp/guard_standalone.log "$OUT/guard_$2_v$valid.log" 2>/dev/null
      S=$((S+1)); attempt=$((attempt+1))
      if is_flaky "$OUT/guard_$2_v$valid.log"; then echo "-- flaky retry $attempt --"; else break; fi
    done
    valid=$((valid+1)); echo "PROGRESS $2 $valid/$SEEDS $(date +%H:%M:%S)"
  done
}
run_cell coord_eps08  eps08
run_cell coord_eps095 eps095
run_cell coord_eps11  eps11
echo "COORD FINE DONE"
