#!/bin/bash
WT=/home/jim/ros2_motion_planning_tutorials/.claude/worktrees/fix-poscheck-infra
MLOG="$WT/experiment_results/gazebo_s1_s6/multimap_fab_driver.log"
echo "CHAIN_ADAPTIVE waiting for multimap $(date +%H:%M:%S)"
while ! grep -q "MULTIMAP_FAB_DONE" "$MLOG" 2>/dev/null; do
  pgrep -f 'run_multimap_fab.sh' >/dev/null || { grep -q MULTIMAP_FAB_DONE "$MLOG" 2>/dev/null || echo "multimap exited w/o DONE"; break; }
  sleep 60
done
sleep 10
bash "$WT/scratchpad/run_adaptive_coord.sh"
