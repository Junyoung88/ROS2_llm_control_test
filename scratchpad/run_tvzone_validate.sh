#!/bin/bash
# Time-varying zone (R3-4): goal (6.5,0) approved on a clear path (forbidden zone removed from
# the geofence config so goal_gate admits it for ALL methods). Mid-navigation, an external
# process activates the zone x[4,6]y[-1,1] via /petse/inject_zone. PETSE's runtime guard then
# enforces the just-activated zone (continuous re-verification) -> stops before it; approval-time
# methods (no_guard) keep driving in. Violations are counted by the position monitor vs ZONES x[4,6].
set +e
cd /home/jim/ros2_motion_planning_tutorials
source /opt/ros/jazzy/setup.bash; source install/setup.bash
OUT=experiment_results/gazebo_s1_s6/tvzone; mkdir -p "$OUT"

# ---- write geofence.yaml with NO forbidden zone (so goal_gate approves the drive-through) ----
NOZONE='uncertainty:
  k_sigma: 3.0
  localization_sigma: 0.15
  tracking_error: 0.05
  v_max: 0.5
  latency: 0.1

zones: []
'
mapfile -t CFGS < <(find /home/jim/ros2_motion_planning_tutorials -name geofence.yaml -path '*geofence_policy_enforcer/config*' -not -path '*build*')
for c in "${CFGS[@]}"; do cp -f "$c" "$c.tvbak"; printf '%s' "$NOZONE" > "$c"; done
echo "wrote empty-forbidden geofence.yaml to ${#CFGS[@]} copies"

restore_cfg(){ for c in "${CFGS[@]}"; do [ -f "$c.tvbak" ] && mv -f "$c.tvbak" "$c"; done; echo "restored geofence.yaml"; }
trap restore_cfg EXIT

# background injector: wait until the robot has started moving (x>1.5) but is still short of the
# zone (x<3.5), then activate x[4,6] once.
injector(){
  local log=/tmp/position_monitor.log
  for i in $(seq 1 400); do
    x=$(tail -1 "$log" 2>/dev/null | python3 -c "import sys,json
try: print(json.loads(sys.stdin.readline()).get('x',''))
except: print('')" 2>/dev/null)
    if [ -n "$x" ]; then
      ok=$(python3 -c "print(1 if 1.2 < ${x:-99} < 3.7 else 0)" 2>/dev/null)
      if [ "$ok" = "1" ]; then
        ros2 topic pub --once /petse/inject_zone std_msgs/msg/String "{data: '4.0,6.0,-1.0,1.0'}" >/dev/null 2>&1
        echo "  [INJECTOR] activated zone x[4,6] at robot x=$x"
        return
      fi
    fi
    sleep 0.5
  done
  echo "  [INJECTOR] timed out (robot never in [1.2,3.7]); last x=$x"
}

for M in no_guard geofence; do
  echo "==== TVZONE $M $(date +%H:%M:%S) ===="
  pkill -9 -f "gz sim" 2>/dev/null; pkill -9 -f cmd_vel_guard 2>/dev/null; sleep 3
  ( sleep 40; injector ) &   # give the sim time to boot before watching position
  INJ=$!
  python3 -u src/geofence_enforcer/experiments/run_gazebo_s1_s6.py \
    --method "$M" --scenario S4 --seeds 1 --seed-offset 0 --no-sweep \
    --intensity tvzone --output "$OUT/val_${M}.jsonl" > "$OUT/val_${M}.log" 2>&1
  kill $INJ 2>/dev/null
  cp -f /tmp/position_monitor.log "$OUT/posmon_${M}.log" 2>/dev/null
  echo "PROGRESS tvzone $M $(date +%H:%M:%S)"
done
echo "TVZONE VALIDATE DONE $(date +%H:%M:%S)"
