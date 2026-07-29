#!/bin/bash
# W2: collect AMCL (p_hat, Sigma) during warehouse navigation; ground truth comes
# from the runner's position-monitor log (copied per trial), matched offline by time.
# The recorder is named gtcov_rec to survive the runner's 'amcl' cleanup pattern.
set +e
cd /home/jim/ros2_motion_planning_tutorials
source /opt/ros/jazzy/setup.bash; source install/setup.bash
export PETSE_DETECTION_MODE=cusum
OUT=experiment_results/gazebo_s1_s6/poscov; mkdir -p "$OUT"
SEEDS=12; BASE=20
REC=/home/jim/ros2_motion_planning_tutorials/.claude/worktrees/fix-poscheck-infra/scratchpad/gtcov_rec.py

for i in $(seq 0 $((SEEDS-1))); do
  S=$((BASE+i))
  echo "==== cov seed=$S (v$i) $(date +%H:%M:%S) ===="
  pkill -9 -f "gz sim" 2>/dev/null; pkill -9 -f cmd_vel_guard 2>/dev/null
  pkill -9 -f gtcov_rec 2>/dev/null; pkill -9 -f metrics_logger 2>/dev/null; sleep 3
  # Background poller: wait until AMCL actually publishes /amcl_pose, THEN start the
  # recorder (avoids the late-join case where a pre-existing subscriber never matches
  # the later-appearing TRANSIENT_LOCAL publisher). exec so RECPID is the recorder.
  ( for k in $(seq 1 90); do
      ros2 topic info /amcl_pose 2>/dev/null | grep -qE "Publisher count: [1-9]" && break
      sleep 3
    done
    exec python3 "$REC" --out "$OUT/cov_v${i}.jsonl" ) > "$OUT/rec_v${i}.log" 2>&1 &
  RECPID=$!
  python3 -u src/geofence_enforcer/experiments/run_gazebo_s1_s6.py \
    --method geofence --scenario S5 --seeds 1 --seed-offset "$S" --no-sweep \
    --intensity wh_hijack --output "$OUT/res_v${i}.jsonl" > "$OUT/trial_v${i}.log" 2>&1
  kill -INT "$RECPID" 2>/dev/null; sleep 2; kill -9 "$RECPID" 2>/dev/null
  cp -f /tmp/position_monitor.log "$OUT/posmon_v${i}.log" 2>/dev/null
  grep -oE "\[[0-9-]+ [0-9:]+\].*Injecting scan spoof" "$OUT/trial_v${i}.log" | head -1 > "$OUT/inject_v${i}.txt"
  echo "  amcl_samples=$(wc -l < "$OUT/cov_v${i}.jsonl" 2>/dev/null) posmon_lines=$(wc -l < "$OUT/posmon_v${i}.log" 2>/dev/null)"
  echo "PROGRESS cov $((i+1))/$SEEDS $(date +%H:%M:%S)"
done
echo "W2_COV_COLLECT_DONE $(date +%H:%M:%S)"
