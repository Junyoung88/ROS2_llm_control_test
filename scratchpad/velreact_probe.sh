#!/bin/bash
# Restore the velocity cap, clean DDS/gazebo state, and run ONE static_reactive trial to confirm
# gazebo can start again (isolate whether cap=3.0 broke the spawn).
set +e
cd /home/jim/ros2_motion_planning_tutorials
# restore cap 1.5 from backups
for f in \
  src/mobile_manipulator_tutorial/src/mobile_manip_moveit_config/urdf/include/mobile_manip.gazebo \
  src/mobile_manipulator_tutorial/install/mobile_manip_moveit_config/share/mobile_manip_moveit_config/urdf/include/mobile_manip.gazebo \
  install/mobile_manip_moveit_config/share/mobile_manip_moveit_config/urdf/include/mobile_manip.gazebo ; do
  [ -f "$f.velcap_bak" ] && cp -f "$f.velcap_bak" "$f"
done
echo "cap restored: $(grep -h max_linear_velocity install/mobile_manip_moveit_config/share/mobile_manip_moveit_config/urdf/include/mobile_manip.gazebo)"
# thorough cleanup
pkill -9 -f "gz sim" 2>/dev/null; pkill -9 -f gz 2>/dev/null; pkill -9 -f cmd_vel_guard 2>/dev/null
pkill -9 -f goal_gate 2>/dev/null; pkill -9 -f metrics_logger 2>/dev/null; pkill -9 -f ros2-daemon 2>/dev/null
rm -rf /dev/shm/fastrtps_* /dev/shm/sem.fastrtps_* /tmp/fastrtps_* 2>/dev/null
sleep 5
source /opt/ros/jazzy/setup.bash; source install/setup.bash
O=experiment_results/gazebo_s1_s6/assum_violation
echo "probe start $(date +%H:%M:%S)"
python3 -u src/geofence_enforcer/experiments/run_gazebo_s1_s6.py \
  --method static_reactive --scenario S4 --seeds 1 --seed-offset 0 --no-sweep \
  --intensity direct_to_zone_overspeed --output "$O/probe_sr.jsonl" > "$O/probe_sr.log" 2>&1
echo "probe exit=$? $(date +%H:%M:%S)"
cp -f /tmp/position_monitor.log "$O/posmon_probe_sr.log" 2>/dev/null
if [ -s "$O/probe_sr.jsonl" ]; then
  python3 -c "
import json
d=json.load(open('$O/probe_sr.jsonl'))
xs=[]
for line in open('$O/posmon_probe_sr.log'):
  try:xs.append(json.loads(line).get('x'))
  except:pass
xs=[x for x in xs if x is not None]
print('PROBE RESULT: violated=',d.get('violated'),'vc=',d.get('violation_count'),'xmax=%.2f'%(max(xs) if xs else -9),'|',(d.get('reason') or '')[:45])"
else
  echo "PROBE: empty jsonl — gazebo issue persists:"; tail -6 "$O/probe_sr.log"
fi
echo "PROBE DONE $(date +%H:%M:%S)"
