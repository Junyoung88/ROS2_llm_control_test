#!/bin/bash
# Whole-graph SROS2 trial: full Nav2+Gazebo under DDS security, mux in its own
# enclave as the only party permitted to publish /cmd_vel. STRATEGY arg:
#   permissive (default) — logs violations, does not block (find policy gaps)
#   enforce               — blocks violations (may break nodes with policy gaps)
STRAT="${1:-permissive}"
cd /home/jim/ros2_motion_planning_tutorials/.claude/worktrees/fix-poscheck-infra
source /opt/ros/jazzy/setup.bash
source /home/jim/ros2_motion_planning_tutorials/install/setup.bash
export PETSE_DETECTION_MODE=cusum
export PETSE_USE_MUX=1
export PETSE_SROS2="$STRAT"
export PETSE_SROS2_KEYSTORE="$PWD/sros2_full/keystore"
# Larger DDS buffers/fragments so the bigger signed messages under Enforce do not
# overrun the transport ("[RTPS_WRITER] Buffer too small"), which stalled Nav2
# lifecycle activation in the first Enforce attempt.
export FASTRTPS_DEFAULT_PROFILES_FILE="$PWD/sros2_full/fastdds_bigbuf.xml"
pkill -9 -f "gz sim" 2>/dev/null; pkill -9 -f cmd_vel_guard 2>/dev/null
pkill -9 -f trusted_cmd_mux 2>/dev/null; pkill -9 -f metrics_logger 2>/dev/null
sleep 3
mkdir -p experiment_results/sros2_full
python3 -u src/geofence_enforcer/experiments/run_gazebo_s1_s6.py \
  --method geofence --scenario S5 --seeds 1 --seed-offset 0 --no-sweep \
  --intensity wh_hijack --output experiment_results/sros2_full/res_${STRAT}.jsonl \
  > experiment_results/sros2_full/run_${STRAT}.log 2>&1
echo "SROS2_${STRAT}_DONE $(date +%H:%M:%S)" >> experiment_results/sros2_full/run_${STRAT}.log
