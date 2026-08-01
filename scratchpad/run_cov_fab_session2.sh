#!/bin/bash
# W2 covariance calibration — SECOND session on a DIFFERENT map (fab_cell), to close the
# single-map/single-AMCL-session caveat. Collects AMCL (p_hat, Sigma) during fab_cell
# navigation; ground truth from the runner's position-monitor log, matched offline by time.
# Same recorder (gtcov_rec) + analysis as the warehouse session.
set +e
cd /home/jim/ros2_motion_planning_tutorials
source /opt/ros/jazzy/setup.bash; source install/setup.bash
export PETSE_DETECTION_MODE=cusum
WT=/home/jim/ros2_motion_planning_tutorials/.claude/worktrees/fix-poscheck-infra
RUNNER="$WT/src/geofence_enforcer/experiments/run_gazebo_s1_s6.py"
OUT="$WT/experiment_results/gazebo_s1_s6/poscov_fab"; mkdir -p "$OUT"
REC="$WT/scratchpad/gtcov_rec.py"
SEEDS=8; BASE=60

for i in $(seq 0 $((SEEDS-1))); do
  S=$((BASE+i))
  echo "==== covfab seed=$S (v$i) $(date +%H:%M:%S) ===="
  pkill -9 -f "gz sim" 2>/dev/null; pkill -9 -f ros_gz 2>/dev/null; pkill -9 -f cmd_vel_guard 2>/dev/null
  pkill -9 -f gtcov_rec 2>/dev/null; pkill -9 -f run_gazebo_s1_s6 2>/dev/null; sleep 4
  # start recorder only after AMCL publishes /amcl_pose (late-join match)
  ( for k in $(seq 1 90); do
      ros2 topic info /amcl_pose 2>/dev/null | grep -qE "Publisher count: [1-9]" && break
      sleep 3
    done
    exec python3 "$REC" --out "$OUT/cov_v${i}.jsonl" ) > "$OUT/rec_v${i}.log" 2>&1 &
  RECPID=$!
  timeout 900 python3 -u "$RUNNER" \
    --method geofence --scenario S5 --seeds 1 --seed-offset "$S" --no-sweep \
    --intensity fab_spoof_hijack --output "$OUT/res_v${i}.jsonl" > "$OUT/trial_v${i}.log" 2>&1
  kill -INT "$RECPID" 2>/dev/null; sleep 2; kill -9 "$RECPID" 2>/dev/null
  cp -f /tmp/position_monitor.log "$OUT/posmon_v${i}.log" 2>/dev/null
  echo "  amcl_samples=$(wc -l < "$OUT/cov_v${i}.jsonl" 2>/dev/null) posmon_lines=$(wc -l < "$OUT/posmon_v${i}.log" 2>/dev/null)"
  echo "PROGRESS covfab $((i+1))/$SEEDS $(date +%H:%M:%S)"
done
echo "COV_FAB_SESSION2_DONE $(date +%H:%M:%S)"
