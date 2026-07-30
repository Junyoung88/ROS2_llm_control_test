#!/bin/bash
# W2 experiment: whole-graph SROS2 Enforce with an ACCESS-CONTROL-ONLY governance
# (join + read/write access control kept; all protection_kind=NONE). Removing the
# crypto MAC/encryption removes the fragmentation overhead that overflowed the CDR
# buffer, so Nav2 should come up under Enforce while the /cmd_vel allow-rule is still
# enforced (sole-writer exclusivity). Default transport (no custom fastdds profile).
set +e
cd /home/jim/ros2_motion_planning_tutorials
source /opt/ros/jazzy/setup.bash
source install/setup.bash
WT=/home/jim/ros2_motion_planning_tutorials/.claude/worktrees/fix-poscheck-infra
export PETSE_DETECTION_MODE=cusum
export PETSE_USE_MUX=1
export PETSE_SROS2=enforce
export PETSE_SROS2_KEYSTORE="$WT/sros2_full/keystore"
unset FASTRTPS_DEFAULT_PROFILES_FILE   # default transport; no encryption -> no CDR overflow
pkill -9 -f "gz sim" 2>/dev/null; pkill -9 -f cmd_vel_guard 2>/dev/null
pkill -9 -f trusted_cmd_mux 2>/dev/null; pkill -9 -f metrics_logger 2>/dev/null
sleep 3
OUT="$WT/experiment_results/sros2_full"; mkdir -p "$OUT"
echo "NOENC_ENFORCE_START $(date +%H:%M:%S)"
timeout 900 python3 -u "$WT/src/geofence_enforcer/experiments/run_gazebo_s1_s6.py" \
  --method geofence --scenario S5 --seeds 1 --seed-offset 0 --no-sweep \
  --intensity wh_hijack --output "$OUT/res_noenc_enforce.jsonl" \
  > "$OUT/run_noenc_enforce.log" 2>&1
echo "NOENC_ENFORCE_EXIT rc=$? $(date +%H:%M:%S)"
echo "--- Buffer too small count: $(grep -c 'Buffer too small' "$OUT/run_noenc_enforce.log" 2>/dev/null) ---"
echo "--- SECURITY access-control denials (expected: only attacker/unpermitted): ---"
grep -c 'not found in allow rule' "$OUT/run_noenc_enforce.log" 2>/dev/null
grep -iE "Nav2 (is )?ready|Timeout waiting for Nav2|controller_server.*(active|not active)|Navigation (succeeded|completed)|robot_moved|Goal reached|runtime_reject|violated" "$OUT/run_noenc_enforce.log" 2>/dev/null | tail -6
echo "--- trial result: ---"
python3 -c "import json;d=json.load(open('$OUT/res_noenc_enforce.jsonl'));print('violated=',d.get('violated'),'decision=',d.get('decision'),'robot_moved=',d.get('robot_moved'),'reason=',(d.get('reason') or '')[:55])" 2>/dev/null || echo "(empty result)"
echo "NOENC_ENFORCE_DONE $(date +%H:%M:%S)"
