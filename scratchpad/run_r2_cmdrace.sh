#!/bin/bash
# R2: in-situ command race in Gazebo. Run an S5 wh_hijack trial with the mux
# (PETSE_USE_MUX=1); a co-running attacker floods /cmd_vel_proposed at 100 Hz the
# moment PETSE latches the stop. Measure malicious commands reaching the actuator
# after the latch (should be 0) and the robot's stopping clearance.
set +e
cd /home/jim/ros2_motion_planning_tutorials
source /opt/ros/jazzy/setup.bash; source install/setup.bash
export PETSE_DETECTION_MODE=cusum PETSE_USE_MUX=1
OUT=experiment_results/gazebo_s1_s6/cmdrace; mkdir -p "$OUT"
RACE=/home/jim/ros2_motion_planning_tutorials/.claude/worktrees/fix-poscheck-infra/scratchpad/cmdrace_gz.py
SEEDS=2; BASE=40

for i in $(seq 0 $((SEEDS-1))); do
  S=$((BASE+i))
  echo "==== r2 seed=$S (v$i) $(date +%H:%M:%S) ===="
  pkill -9 -f "gz sim" 2>/dev/null; pkill -9 -f cmd_vel_guard 2>/dev/null
  pkill -9 -f trusted_cmd_mux 2>/dev/null; pkill -9 -f cmdrace_gz 2>/dev/null
  pkill -9 -f metrics_logger 2>/dev/null; sleep 3
  # Start the attacker AFTER the mux is up (so its DDS participant is created once
  # the graph is stable and discovers every node -- an early participant created
  # before the runner's daemon resets/full restart can stay isolated). Latch is
  # detected via the mux's PERIODIC /petse/mux_metrics (state=='latched'), so a
  # late start still catches it.
  ( for k in $(seq 1 100); do
      ros2 node list 2>/dev/null | grep -q trusted_cmd_mux && break
      sleep 3
    done
    sleep 2
    exec python3 "$RACE" --out "$OUT/race_v${i}.json" --flood-hz 100 --dur 25 ) \
        > "$OUT/race_v${i}.log" 2>&1 &
  RPID=$!
  python3 -u .claude/worktrees/fix-poscheck-infra/src/geofence_enforcer/experiments/run_gazebo_s1_s6.py \
    --method geofence --scenario S5 --seeds 1 --seed-offset "$S" --no-sweep \
    --intensity wh_hijack --output "$OUT/res_v${i}.jsonl" > "$OUT/trial_v${i}.log" 2>&1
  kill -INT "$RPID" 2>/dev/null; sleep 2; kill -9 "$RPID" 2>/dev/null
  cp -f /tmp/guard_standalone.log "$OUT/guard_v${i}.log" 2>/dev/null
  cp -f /tmp/trusted_mux.log "$OUT/mux_v${i}.log" 2>/dev/null
  CLR=$(python3 -c "import json;d=json.load(open('$OUT/res_v${i}.jsonl'));print(round(d.get('path_min_distance') or -1,3),d.get('violated'))" 2>/dev/null)
  echo "  race=$(cat $OUT/race_v${i}.json 2>/dev/null | tr -d '\n' | cut -c1-160)  clearance/violated=$CLR"
  echo "PROGRESS r2 $((i+1))/$SEEDS $(date +%H:%M:%S)"
done
echo "R2_CMDRACE_DONE $(date +%H:%M:%S)"
