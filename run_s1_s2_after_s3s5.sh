#!/bin/bash
source /opt/ros/jazzy/setup.bash
source /home/jim/ros2_motion_planning_tutorials/install/setup.bash 2>/dev/null
cd /home/jim/ros2_motion_planning_tutorials

echo "[$(date)] Waiting for S3-S5 experiment to finish..."

# Monitor the S3-S5 process
while pgrep -f "run_gazebo_s1_s6.*S3,S4,S5" > /dev/null 2>&1; do
    sleep 60
done

echo "[$(date)] S3-S5 experiment finished!"

# BACKUP before doing anything
BACKUP_NAME="results_pre_s1s2_rerun_$(date +%Y%m%d_%H%M%S).jsonl"
cp experiment_results/gazebo_s1_s6/results.jsonl "experiment_results/gazebo_s1_s6/${BACKUP_NAME}"
echo "[$(date)] Backup saved: ${BACKUP_NAME}"
echo "[$(date)] Backup line count: $(wc -l < experiment_results/gazebo_s1_s6/${BACKUP_NAME})"

echo "[$(date)] Waiting 30s for cleanup..."
sleep 30

# Kill any leftover processes
pkill -9 -f 'gz sim' 2>/dev/null
pkill -9 -f gzserver 2>/dev/null
pkill -9 -f ros2 2>/dev/null
sleep 10

echo "[$(date)] Starting S1/S2 experiment (seeds=10, append mode)..."
python3 -u src/geofence_enforcer/experiments/run_gazebo_s1_s6.py \
    --scenario S1,S2 \
    --seeds 10 \
    --append \
    2>&1 | tee experiment_results/gazebo_s1_s6/s1_s2_rerun.log

echo "[$(date)] S1/S2 experiment complete!"
echo "[$(date)] Final results count: $(wc -l < experiment_results/gazebo_s1_s6/results.jsonl)"
