#!/usr/bin/env python3
"""
S6b: Braking Term Necessity — Gazebo Direct-Drive Experiment

Without braking term: M=0.55m → 약한 제동력이면 구역 침입
With braking term:    M=0.55+v²/(2a) → 항상 안전하게 정지

아키텍처:
  orchestration mode (default): Gazebo 관리 + 텔레포트 + trial subprocess 호출
  trial mode (--trial): 순수 rclpy 드라이브 루프
"""
import sys, os, subprocess, time, json, signal, math

ZONE_X_MIN = 4.0
V          = 0.5          # approach velocity (m/s)
TAU        = 0.1          # guard latency (s)
M_STATIC   = 3*0.15+0.05  # k_sigma*sigma + e_track = 0.50 m
GZ_WORLD   = 'empty'
SETUP      = 'source /home/jim/ros2_motion_planning_tutorials/install/setup.bash 2>/dev/null && '
OUT_FILE   = '/home/jim/ros2_motion_planning_tutorials/experiment_results/gazebo_s1_s6/s6b_braking_results.json'

def M_no_brake():    return M_STATIC + V*TAU           # 0.55 m
def M_with_brake(a): return M_STATIC + V*TAU + V**2/(2*a)
def expected_pen(a): return -M_STATIC + V**2/(2*a)    # positive = violation

CONDITIONS = [
    (0.1, False), (0.2, False), (0.5, False), (1.0, False),
    (0.1, True),  (0.2, True),  (0.5, True),  (1.0, True),
]
SEEDS = 3

# ══════════════════════════════════════════════════════════════
# TRIAL MODE  — rclpy 드라이브 루프 (subprocess로 호출됨)
# ══════════════════════════════════════════════════════════════
if '--trial' in sys.argv:
    idx       = sys.argv.index('--trial')
    a_max     = float(sys.argv[idx+1])
    use_brake = sys.argv[idx+2].lower() == 'true'

    signal.signal(signal.SIGTERM, signal.SIG_IGN)

    import rclpy
    from rclpy.node import Node
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import Twist

    class DriveNode(Node):
        def __init__(self):
            super().__init__('s6b_drive')
            self.x, self._got = 0.0, False
            self.sub = self.create_subscription(Odometry, '/odom', self._cb, 10)
            self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        def _cb(self, msg):
            self.x = msg.pose.pose.position.x
            self._got = True
        def stop(self):   self.pub.publish(Twist())
        def spin(self):
            try: rclpy.spin_once(self, timeout_sec=0)
            except Exception: pass

    rclpy.init()
    nd = DriveNode()

    # Wait for /odom
    t0 = time.time()
    while not nd._got and time.time()-t0 < 15:
        nd.spin(); time.sleep(0.05)
    if not nd._got:
        print(json.dumps({'error': 'no_odom'})); sys.exit(1)

    M       = M_with_brake(a_max) if use_brake else M_no_brake()
    trig_x  = ZONE_X_MIN - M

    nd.stop(); time.sleep(0.2)

    vel             = V
    guard_triggered = False
    guard_t         = None
    braking         = False
    t_start         = time.time()
    prev_loop_t     = t_start

    while True:
        nd.spin()
        now  = time.time()
        dt   = now - prev_loop_t      # actual elapsed time (accurate!)
        prev_loop_t = now
        t    = now - t_start
        x    = nd.x

        if not guard_triggered and x >= trig_x:
            guard_triggered = True; guard_t = t

        if guard_triggered and not braking and t >= guard_t + TAU:
            braking = True

        if braking:
            vel = max(0.0, vel - a_max * dt)

        msg = Twist(); msg.linear.x = vel
        nd.pub.publish(msg)

        if vel < 5e-4 or x > ZONE_X_MIN+1.5 or t > 40:
            break
        time.sleep(0.02)

    nd.stop(); time.sleep(0.3)
    nd.spin()
    fx = nd.x

    print(json.dumps({
        'a_max':            a_max,
        'use_braking_term': use_brake,
        'M_guard':          round(M, 4),
        'trigger_x':        round(trig_x, 4),
        'final_x':          round(fx, 4),
        'violation':        fx > ZONE_X_MIN,
        'penetration_m':    round(fx - ZONE_X_MIN, 4),
        'expected_pen_m':   round(expected_pen(a_max), 4),
    }))
    nd.destroy_node(); rclpy.shutdown(); sys.exit(0)


# ══════════════════════════════════════════════════════════════
# ORCHESTRATION MODE
# ══════════════════════════════════════════════════════════════
procs = []

def gz(cmd):
    return subprocess.run(['bash','-c', SETUP+cmd],
                          capture_output=True, text=True, timeout=10)

def start_gazebo():
    print('[S6b] Cleaning up old processes...')
    for pat in ['gz sim','robot_state_publisher','ros2_control',
                'topic_tools relay','parameter_bridge','ros_gz_bridge',
                'run_s6b']:
        subprocess.run(['pkill','-f',pat], capture_output=True)
    time.sleep(5)

    print('[S6b] Starting Gazebo (empty world)...')
    p = subprocess.Popen(
        ['bash','-c', SETUP +
         'ros2 launch mobile_manip_moveit_config world.launch.py world:=empty.sdf'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid)
    procs.append(p)
    print('[S6b] Waiting 90s...')
    time.sleep(90)

    p2 = subprocess.Popen(
        ['bash','-c', SETUP+'ros2 run topic_tools relay /odom_real /odom'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid)
    procs.append(p2)
    time.sleep(8)

    # verify /odom
    for _ in range(10):
        r = gz('ros2 topic list 2>/dev/null | grep "^/odom$"')
        if '/odom' in r.stdout: break
        time.sleep(3)
    print('[S6b] Gazebo ready.')

def teleport(x=0.0, y=0.0):
    """텔레포트: orchestration mode에서 직접 실행 (rclpy 없음)"""
    # stop cmd_vel first
    gz('ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{}"')
    time.sleep(0.3)
    cmd = (f'gz service -s /world/{GZ_WORLD}/set_pose '
           f'--reqtype gz.msgs.Pose --reptype gz.msgs.Boolean --timeout 3000 '
           f'--req \'name: "mobile_manip", '
           f'position: {{x: {x}, y: {y}, z: 0.1}}, '
           f'orientation: {{x: 0.0, y: 0.0, z: 0.0, w: 1.0}}\'')
    r = gz(cmd)
    ok = 'true' in r.stdout.lower()
    if not ok:
        print(f'[S6b][WARN] teleport may have failed: {r.stderr[:80]}')
    time.sleep(2.0)
    return ok

def verify_position():
    """현재 robot x 위치 확인 (gz topic)"""
    r = gz(f'gz topic -e -n 1 -t /world/{GZ_WORLD}/pose/info 2>/dev/null')
    for line in r.stdout.split('\n'):
        if 'x:' in line:
            try: return float(line.split('x:')[1].strip().split()[0])
            except Exception: pass
    return None

def run_trial(a_max, use_brake):
    cmd = SETUP + f'python3 -u {os.path.abspath(__file__)} --trial {a_max} {use_brake}'
    try:
        r = subprocess.run(['bash','-c',cmd],
                           capture_output=True, text=True, timeout=60)
        for line in reversed(r.stdout.strip().split('\n')):
            line = line.strip()
            if line.startswith('{'):
                return json.loads(line)
        stderr_snip = r.stderr.strip()[-200:] if r.stderr else ''
        print(f'  [WARN] no JSON in output. stderr: {stderr_snip}')
        return None
    except subprocess.TimeoutExpired:
        print('  [ERROR] trial timeout')
        return None

def stop_all():
    for p in procs:
        try: os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except Exception: pass

def main():
    sys.stdout.reconfigure(line_buffering=True)

    start_gazebo()

    results = []
    total = len(CONDITIONS)*SEEDS
    idx   = 0

    for seed in range(SEEDS):
        for (a_max, use_brake) in CONDITIONS:
            idx += 1
            label = ('with_brake' if use_brake else 'no_brake') + f'_a{a_max}_s{seed}'
            M_str = f'{M_with_brake(a_max):.3f}' if use_brake else f'{M_no_brake():.3f}'
            print(f'\n[S6b] Trial {idx}/{total}: {label}')
            print(f'       M={M_str}m  expected_pen={expected_pen(a_max):+.3f}m')

            # ── 텔레포트 (orchestration에서 직접) ──────────────
            ok = teleport(0.0, 0.0)
            # 위치 확인
            pos = verify_position()
            if pos is not None and abs(pos) > 1.0:
                print(f'  [WARN] robot at x={pos:.2f} after teleport, retrying...')
                teleport(0.0, 0.0)
                time.sleep(2.0)

            # ── Trial subprocess ────────────────────────────────
            result = run_trial(a_max, use_brake)
            if result is None:
                result = {'a_max':a_max,'use_braking_term':use_brake,'error':'failed'}
            result['trial_id'] = label
            result['seed']     = seed
            results.append(result)

            if 'violation' in result and result['violation'] is not None:
                status = 'VIOLATION' if result['violation'] else 'safe'
                print(f'       final_x={result["final_x"]:.4f}m  '
                      f'pen={result["penetration_m"]:+.4f}m  [{status}]')

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE,'w') as f: json.dump(results, f, indent=2)
    print(f'\n[S6b] Saved → {OUT_FILE}')

    # ── Summary ─────────────────────────────────────────────────
    print('\n=== S6b Results (seed=0) ===')
    print(f"{'Condition':<22} {'a_max':>6} {'M(m)':>7} "
          f"{'pen_actual':>11} {'pen_expected':>13} {'Result':>10}")
    print('-'*75)
    for r in [x for x in results if x.get('seed',0)==0 and 'error' not in x]:
        cond = 'with_brake' if r['use_braking_term'] else 'no_brake '
        st   = 'VIOLATION' if r['violation'] else 'safe'
        print(f"  {cond}  a={r['a_max']:<5}  {r['M_guard']:>6.3f}  "
              f"{r['penetration_m']:>+10.4f}m  {r['expected_pen_m']:>+12.4f}m  {st:>10}")

    stop_all()
    print('[S6b] Done.')

if __name__ == '__main__':
    main()
