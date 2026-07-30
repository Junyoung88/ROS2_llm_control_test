#!/bin/bash
# One corrected whole-graph SROS2 Enforce attempt using fastdds_secfrag.xml
# (small maxMessageSize + async + flow control) to fix the CDRMessage buffer overflow
# that stalled Nav2 lifecycle activation. Success == Nav2 reaches "ready" and the S5
# trial navigates (then the mux/latch defense can actually run under Enforce).
set +e
cd /home/jim/ros2_motion_planning_tutorials
source /opt/ros/jazzy/setup.bash
source install/setup.bash
WT=/home/jim/ros2_motion_planning_tutorials/.claude/worktrees/fix-poscheck-infra
export PETSE_DETECTION_MODE=cusum
export PETSE_USE_MUX=1
export PETSE_SROS2=enforce
export PETSE_SROS2_KEYSTORE="$WT/sros2_full/keystore"
export FASTRTPS_DEFAULT_PROFILES_FILE="$WT/sros2_full/fastdds_secfrag.xml"
pkill -9 -f "gz sim" 2>/dev/null; pkill -9 -f cmd_vel_guard 2>/dev/null
pkill -9 -f trusted_cmd_mux 2>/dev/null; pkill -9 -f metrics_logger 2>/dev/null
sleep 3
OUT="$WT/experiment_results/sros2_full"; mkdir -p "$OUT"
echo "SECFRAG_ENFORCE_START $(date +%H:%M:%S)"
timeout 900 python3 -u "$WT/src/geofence_enforcer/experiments/run_gazebo_s1_s6.py" \
  --method geofence --scenario S5 --seeds 1 --seed-offset 0 --no-sweep \
  --intensity wh_hijack --output "$OUT/res_secfrag_enforce.jsonl" \
  > "$OUT/run_secfrag_enforce.log" 2>&1
echo "SECFRAG_ENFORCE_EXIT rc=$? $(date +%H:%M:%S)"
# quick verdict: did Nav2 activate / any buffer-too-small errors?
echo "--- buffer-too-small count: $(grep -c 'Buffer too small' "$OUT/run_secfrag_enforce.log" 2>/dev/null) ---"
grep -iE "Nav2 (is )?ready|lifecycle not (ready|active)|controller_server.*(active|not)|Timeout waiting for Nav2" "$OUT/run_secfrag_enforce.log" 2>/dev/null | tail -5
echo "SECFRAG_ENFORCE_DONE $(date +%H:%M:%S)"
