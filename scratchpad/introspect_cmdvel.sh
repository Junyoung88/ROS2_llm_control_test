#!/bin/bash
# Live introspection of the /cmd_vel chain under SROS2 Enforce. Waits until Nav2 is
# navigating (guard nav_approved=True), then captures pub/sub counts, QoS, and actual
# values on each hop: controller->/cmd_vel_nav->guard->/cmd_vel_proposed->mux->/cmd_vel.
# Runs with the SAME security env so its CLI participant joins the enclosed domain.
WT=/home/jim/ros2_motion_planning_tutorials/.claude/worktrees/fix-poscheck-infra
source /opt/ros/jazzy/setup.bash
source /home/jim/ros2_motion_planning_tutorials/install/setup.bash
export ROS_SECURITY_ENABLE=true
export ROS_SECURITY_STRATEGY=Enforce
export ROS_SECURITY_KEYSTORE="$WT/sros2_full/keystore"
export PETSE_SROS2=enforce
OUT="$WT/experiment_results/sros2_full/cmdvel_introspect.log"
: > "$OUT"
echo "waiting for nav_approved=True ..." >> "$OUT"
for i in $(seq 1 150); do
  grep -q "nav_approved=True" /tmp/guard_standalone.log 2>/dev/null && { echo "nav active at $(date +%H:%M:%S) (iter $i)" >> "$OUT"; break; }
  sleep 4
done
# give the controller a moment to be commanding motion
sleep 3
for t in /cmd_vel_nav /cmd_vel_proposed /cmd_vel; do
  echo "" >> "$OUT"; echo "========== $t ==========" >> "$OUT"
  echo "--- info -v ---" >> "$OUT"; timeout 12 ros2 topic info -v "$t" >> "$OUT" 2>&1
  echo "--- hz (6s) ---" >> "$OUT"; timeout 8 ros2 topic hz "$t" >> "$OUT" 2>&1
  echo "--- echo --once ---" >> "$OUT"; timeout 8 ros2 topic echo --once "$t" >> "$OUT" 2>&1
done
echo "" >> "$OUT"; echo "=== node list (mux/guard/bridge present?) ===" >> "$OUT"
timeout 10 ros2 node list >> "$OUT" 2>&1
echo "INTROSPECT_DONE $(date +%H:%M:%S)" >> "$OUT"
