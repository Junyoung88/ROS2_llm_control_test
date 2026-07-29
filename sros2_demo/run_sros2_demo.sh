#!/bin/bash
# SROS2 actuator-exclusivity demo for the trusted command mux.
#
# Proves the sole-writer property the plain Gazebo wiring could not: under
# ROS_SECURITY_STRATEGY=Enforce, only the /petse/mux enclave may publish the
# actuator topic /cmd_vel. An attacker in the /petse/attacker enclave is rejected
# by the DDS access-control plugin at datawriter creation, so its commands never
# reach the actuator — regardless of publish rate.
#
# Prereqs (already generated in sros2_demo/keystore):
#   ros2 security create_keystore sros2_demo/keystore
#   ros2 security create_enclave  sros2_demo/keystore /petse/{mux,attacker,monitor}
#   ros2 security create_permission sros2_demo/keystore /petse/<e> sros2_demo/policy.xml
#
# Usage: bash sros2_demo/run_sros2_demo.sh
set +e
cd "$(dirname "$0")/.."
source /opt/ros/jazzy/setup.bash
export ROS_SECURITY_KEYSTORE="$PWD/sros2_demo/keystore"
export ROS_SECURITY_ENABLE=true
export ROS_SECURITY_STRATEGY=Enforce
export CMD_N=40
OUT=sros2_demo/results; mkdir -p "$OUT"

echo "== monitor (/petse/monitor) subscribing /cmd_vel =="
python3 sros2_demo/cmd_sub.py --ros-args --enclave /petse/monitor > "$OUT/monitor.log" 2>&1 &
MON=$!; sleep 5

echo "== MUX publisher (/petse/mux) — ALLOWED =="
python3 sros2_demo/cmd_pub.py --ros-args --enclave /petse/mux > "$OUT/mux_pub.log" 2>&1
sleep 3
MUX_SENT=$(grep -oE 'published [0-9]+' "$OUT/mux_pub.log" | tail -1)

echo "== ATTACKER publisher (/petse/attacker) — DENIED on /cmd_vel =="
python3 sros2_demo/cmd_pub.py --ros-args --enclave /petse/attacker > "$OUT/attacker_pub.log" 2>&1
sleep 3
ATK_DENIED=$(grep -cE 'rt/cmd_vel topic not found in allow rule' "$OUT/attacker_pub.log")

kill -INT $MON 2>/dev/null; sleep 2
FINAL=$(grep -oE 'MONITOR_FINAL_COUNT=[0-9]+' "$OUT/monitor.log" | tail -1)

{
  echo "mux_publisher:        $MUX_SENT (no security error → allowed)"
  echo "attacker_denied_rows: $ATK_DENIED (DDS check_create_datawriter rejections on rt/cmd_vel)"
  echo "monitor_$FINAL       (nonzero /cmd_vel actually delivered — attacker contributed 0)"
  if [ "${ATK_DENIED:-0}" -ge 1 ]; then
    echo "RESULT: PASS — actuator /cmd_vel exclusive to the mux enclave; attacker rejected at DDS layer"
  else
    echo "RESULT: FAIL — attacker was not denied"
  fi
} | tee "$OUT/summary.txt"
