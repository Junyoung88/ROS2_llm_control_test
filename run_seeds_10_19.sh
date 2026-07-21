#!/bin/bash
# Additional seeds 10-19 for all methods, S1/S3/S4/S5 (no S2)
set -e
cd /home/jim/ros2_motion_planning_tutorials
source install/setup.bash

LOG="experiment_results/gazebo_s1_s6/seeds_10_19.log"
echo "=== Seeds 10-19 started at $(date) ===" | tee -a "$LOG"

for METHOD in roboguard geofence cbf cbf_inflated ssm selp_proper no_guard; do
  echo "[$(date)] $METHOD S1,S3,S4,S5 seed 10-19" | tee -a "$LOG"
  python3 src/geofence_enforcer/experiments/run_gazebo_s1_s6.py \
    --method "$METHOD" --scenario S1,S3,S4,S5 --seeds 10 --seed-offset 10 \
    --no-sweep --append 2>&1 | tee -a "$LOG"
done

echo "=== Seeds 10-19 completed at $(date) ===" | tee -a "$LOG"
