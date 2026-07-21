#!/bin/bash
# Additional seed experiments - run after current roboguard finishes
# S2 excluded from all runs

set -e
cd /home/jim/ros2_motion_planning_tutorials
source install/setup.bash

LOG="experiment_results/gazebo_s1_s6/additional_seeds.log"
echo "=== Additional seeds started at $(date) ===" | tee -a "$LOG"

# --- Phase 1: Roboguard additions ---
echo "[Phase 1a] Roboguard S1,S4,S5 seed 5-9 (60 trials)" | tee -a "$LOG"
python3 src/geofence_enforcer/experiments/run_gazebo_s1_s6.py \
  --method roboguard --scenario S1,S4,S5 --seeds 5 --seed-offset 5 \
  --no-sweep --append 2>&1 | tee -a "$LOG"

echo "[Phase 1b] Roboguard S3 seed 0-9 (40 trials)" | tee -a "$LOG"
python3 src/geofence_enforcer/experiments/run_gazebo_s1_s6.py \
  --method roboguard --scenario S3 --seeds 10 --seed-offset 0 \
  --no-sweep --append 2>&1 | tee -a "$LOG"

# --- Phase 2: Existing methods seed 5-9 additions (S1,S3,S4,S5) ---
for METHOD in geofence cbf ssm selp_proper; do
  echo "[Phase 2] $METHOD S1,S3,S4,S5 seed 5-9" | tee -a "$LOG"
  python3 src/geofence_enforcer/experiments/run_gazebo_s1_s6.py \
    --method "$METHOD" --scenario S1,S3,S4,S5 --seeds 5 --seed-offset 5 \
    --no-sweep --append 2>&1 | tee -a "$LOG"
done

echo "=== All additional seeds completed at $(date) ===" | tee -a "$LOG"
