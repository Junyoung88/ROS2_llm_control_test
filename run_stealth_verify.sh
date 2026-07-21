#!/bin/bash
# Verify the STEALTHY-rate bias-injection (phi=0) still lures the TRUE robot into
# the forbidden zone under no_guard. If slow(0.08) reliably lures, it is the money
# config (evades memoryless, caught by cusum). Otherwise fall back to mid(0.15).
cd /home/jim/ros2_motion_planning_tutorials
source /opt/ros/jazzy/setup.bash
source install/setup.bash
OUTDIR=experiment_results/gazebo_s1_s6/stealth_verify
mkdir -p "$OUTDIR"
for CFG in bias_stealth_slow bias_stealth_mid; do
  for S in 0 1; do
    echo "======== $CFG seed $S ========"
    pkill -9 -f "gz sim" 2>/dev/null; pkill -9 -f "attack_scan" 2>/dev/null; pkill -9 -f "cmd_vel_guard" 2>/dev/null
    sleep 3
    python3 -u src/geofence_enforcer/experiments/run_gazebo_s1_s6.py \
      --method no_guard --scenario S5 --seeds 1 --seed-offset "$S" --no-sweep \
      --intensity "$CFG" \
      --output "$OUTDIR/results_${CFG}_s${S}.jsonl" 2>&1 | tee "$OUTDIR/run_${CFG}_s${S}.log"
    cp -f /tmp/position_monitor.log "$OUTDIR/posmon_${CFG}_s${S}.log" 2>/dev/null
    echo "-------- done $CFG s$S --------"
  done
done
echo "ALL STEALTH VERIFY DONE"
