#!/bin/bash
# phi-direction calibration: find which laser-frame phi lures the TRUE robot into
# the warehouse forbidden zone x[1.3,3.0] y[-0.7,0.7]. no_guard, 1 seed each.
cd /home/jim/ros2_motion_planning_tutorials
source /opt/ros/jazzy/setup.bash
source install/setup.bash
OUTDIR=experiment_results/gazebo_s1_s6/phi_calib
mkdir -p "$OUTDIR"
for PHI in phi0 phi90 phi180 phi270; do
  echo "======== bias_cal_$PHI ========"
  pkill -9 -f "gz sim" 2>/dev/null; pkill -9 -f "attack_scan" 2>/dev/null; pkill -9 -f "cmd_vel_guard" 2>/dev/null
  sleep 3
  python3 -u src/geofence_enforcer/experiments/run_gazebo_s1_s6.py \
    --method no_guard --scenario S5 --seeds 1 --no-sweep \
    --intensity "bias_cal_$PHI" \
    --output "$OUTDIR/results_$PHI.jsonl" 2>&1 | tee "$OUTDIR/run_$PHI.log"
  cp -f /tmp/position_monitor.log "$OUTDIR/posmon_$PHI.log" 2>/dev/null || echo "no posmon log"
  echo "======== done $PHI ========"
done
echo "ALL PHI CALIB DONE"
