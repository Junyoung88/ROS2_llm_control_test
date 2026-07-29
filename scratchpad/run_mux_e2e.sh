#!/bin/bash
# Single end-to-end Gazebo trial with the trusted command mux wired in
# (PETSE_USE_MUX=1). S5 wh_hijack: localization spoof should trip the guard's
# fail-stop → /petse/stop_latch → mux latches zero as the sole /cmd_vel writer.
cd /home/jim/ros2_motion_planning_tutorials/.claude/worktrees/fix-poscheck-infra
source /opt/ros/jazzy/setup.bash
source /home/jim/ros2_motion_planning_tutorials/install/setup.bash
export PETSE_DETECTION_MODE=cusum
export PETSE_USE_MUX=1
pkill -9 -f "gz sim" 2>/dev/null; pkill -9 -f cmd_vel_guard 2>/dev/null
pkill -9 -f trusted_cmd_mux 2>/dev/null; pkill -9 -f metrics_logger 2>/dev/null
sleep 3
python3 -u src/geofence_enforcer/experiments/run_gazebo_s1_s6.py \
  --method geofence --scenario S5 --seeds 1 --seed-offset 0 --no-sweep \
  --intensity wh_hijack --output experiment_results/mux_e2e/res_mux_e2e.jsonl \
  > experiment_results/mux_e2e/run.log 2>&1
echo "MUX_E2E_DONE $(date +%H:%M:%S)" >> experiment_results/mux_e2e/run.log
