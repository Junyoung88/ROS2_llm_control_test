#!/bin/bash
WT=/home/jim/ros2_motion_planning_tutorials/.claude/worktrees/fix-poscheck-infra
CL="$WT/experiment_results/gazebo_s1_s6/clutter_driver.log"
while ! grep -q "CLUTTER_DONE" "$CL" 2>/dev/null; do
  pgrep -f 'run_clutter.sh' >/dev/null || break
  sleep 60
done
sleep 10
bash "$WT/scratchpad/run_dynamic.sh"
