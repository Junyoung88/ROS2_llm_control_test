#!/usr/bin/env python3
"""
Persistent stealthy pose-spoofing magnitude sweep (bounded-spoof threat model).

Reviewer concern (TII S4/code-S5): the paper claims "the covariance-based
margin absorbed the spoofing offset", but a persistent fixed offset that keeps
the reported covariance small is NOT reflected in lambda_max(Sigma), so the
runtime monitor trusts the spoofed position. This experiment characterizes
exactly how much persistent spoof the margin absorbs.

Physics (constant-velocity approach, 1D, guard at ZONE_X_MIN - M):
  A persistent offset delta makes the guard perceive x_hat = x_true - delta
  (robot appears delta further from the zone). The guard fires when
  x_hat >= trig_x = ZONE_X_MIN - M, i.e. at x_true = ZONE_X_MIN - M + delta.
  Latency + braking then add M_lat + M_brake of true travel, so
    final_x = ZONE_X_MIN - (M_est+M_track) + delta
    penetration = delta - (M_est + M_track).
  => Certified spoof budget  Delta_spoof = M_est + M_track.
     delta <= Delta_spoof  -> absorbed (safe);  delta > Delta_spoof -> violation.

This is the honest, precise version of the paper's claim: only the SPATIAL
margin terms (estimation + tracking) budget a persistent spoof; the kinematic
terms (latency, braking) are consumed by real travel and give no spoof slack.

Modes:
  (default)      orchestration: manage Gazebo, teleport, call --trial subprocess
  --trial D      drive loop with persistent spoof offset D applied to the guard
  --analytical   no Gazebo; emit the exact kinematic prediction
  --plot         render figure from the JSON results

Stealthy vs. detectable: a spoof that inflates lambda_max(Sigma) > lambda_bar
would trigger PETSE's fail-stop regardless of magnitude. This sweep models the
STEALTHY case (covariance held small) -- the adversarially relevant one. The
residual gap beyond Delta_spoof motivates an odom-AMCL consistency detector
(future work).
"""
import sys, os, subprocess, time, json, signal

# ---- Margin parameters (match paper simulation nominal, M = 0.562 m) --------
ZONE_X_MIN = 4.0
V          = 0.5            # approach velocity (m/s)
TAU        = 0.1            # guard/actuation latency (s)
A_MAX      = 2.5            # deceleration (m/s^2)
SIGMA      = 0.15           # 1D localization std (m)
Z_EPS      = 2.748         # z_{1-eps}, eps=0.003 (Abramowitz-Stegun, matches guard)
E0, C1     = 0.03, 0.04     # tracking envelope: e0 + c1*v

M_EST      = Z_EPS * SIGMA              # 0.412
M_TRACK    = E0 + C1 * V                # 0.05
M_LAT      = V * TAU                    # 0.05
M_BRAKE    = V * V / (2 * A_MAX)        # 0.05
M_FULL     = M_EST + M_TRACK + M_LAT + M_BRAKE   # 0.562
SPOOF_BUDGET = M_EST + M_TRACK          # 0.462  <-- certified Delta_spoof

# Spoof magnitudes bracketing the 0.462 threshold
DELTAS = [0.0, 0.1, 0.25, 0.4, 0.462, 0.5, 0.6, 0.75, 1.0]
SEEDS  = 3

GZ_WORLD = 'empty'
SETUP    = ('source /home/jim/ros2_motion_planning_tutorials/install/setup.bash '
            '2>/dev/null && ')
OUT_FILE = ('/home/jim/ros2_motion_planning_tutorials/experiment_results/'
            'gazebo_s1_s6/spoof_sweep_results.json')
FIG_FILE = ('/home/jim/ros2_motion_planning_tutorials/figures/'
            'spoof_budget_sweep.png')


def expected_pen(delta):
    """Exact kinematic penetration prediction (m). positive = violation."""
    return delta - SPOOF_BUDGET


# ==============================================================================
# TRIAL MODE  -- rclpy drive loop with persistent spoof (subprocess)
# ==============================================================================
if '--trial' in sys.argv:
    idx   = sys.argv.index('--trial')
    delta = float(sys.argv[idx + 1])

    signal.signal(signal.SIGTERM, signal.SIG_IGN)

    import rclpy
    from rclpy.node import Node
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import Twist

    class DriveNode(Node):
        def __init__(self):
            super().__init__('spoof_drive')
            self.x, self._got = 0.0, False
            self.sub = self.create_subscription(Odometry, '/odom', self._cb, 10)
            self.pub = self.create_publisher(Twist, '/cmd_vel', 10)

        def _cb(self, msg):
            self.x = msg.pose.pose.position.x
            self._got = True

        def stop(self):
            self.pub.publish(Twist())

        def spin(self):
            try:
                rclpy.spin_once(self, timeout_sec=0)
            except Exception:
                pass

    rclpy.init()
    nd = DriveNode()

    t0 = time.time()
    while not nd._got and time.time() - t0 < 15:
        nd.spin(); time.sleep(0.05)
    if not nd._got:
        print(json.dumps({'error': 'no_odom'})); sys.exit(1)

    # gz set_pose teleports the robot physically but does NOT reset the
    # diff-drive odom integrator, so nd.x carries a stale offset between
    # trials. Track DISPLACEMENT from this trial's start instead: the robot
    # is physically at the origin now, so x_rel = nd.x - x_start is its true
    # distance travelled toward the zone (robust to the odom offset).
    x_start = nd.x

    trig_rel = ZONE_X_MIN - M_FULL   # trigger in displacement (relative) coords

    nd.stop(); time.sleep(0.2)

    vel             = V
    guard_triggered = False
    guard_t         = None
    braking         = False
    t_start         = time.time()
    prev_loop_t     = t_start

    while True:
        nd.spin()
        now = time.time()
        dt  = now - prev_loop_t
        prev_loop_t = now
        t = now - t_start
        x_rel = nd.x - x_start                # true distance travelled
        x_perceived = x_rel - delta           # persistent stealthy spoof

        if not guard_triggered and x_perceived >= trig_rel:
            guard_triggered = True; guard_t = t

        if guard_triggered and not braking and t >= guard_t + TAU:
            braking = True

        if braking:
            vel = max(0.0, vel - A_MAX * dt)

        msg = Twist(); msg.linear.x = vel
        nd.pub.publish(msg)

        if vel < 5e-4 or x_rel > ZONE_X_MIN + 1.5 or t > 40:
            break
        time.sleep(0.02)

    nd.stop(); time.sleep(0.3)
    nd.spin()
    fx_rel = nd.x - x_start

    print(json.dumps({
        'delta_spoof':     delta,
        'M_full':          round(M_FULL, 4),
        'spoof_budget':    round(SPOOF_BUDGET, 4),
        'x_start_odom':    round(x_start, 4),
        'final_x':         round(fx_rel, 4),    # displacement from spawn
        'violation':       fx_rel > ZONE_X_MIN,
        'penetration_m':   round(fx_rel - ZONE_X_MIN, 4),
        'expected_pen_m':  round(expected_pen(delta), 4),
    }))
    nd.destroy_node(); rclpy.shutdown(); sys.exit(0)


# ==============================================================================
# ANALYTICAL MODE -- exact kinematic prediction, no Gazebo
# ==============================================================================
def run_analytical():
    results = []
    for seed in range(SEEDS):
        for delta in DELTAS:
            pen = expected_pen(delta)
            results.append({
                'delta_spoof':    delta,
                'M_full':         round(M_FULL, 4),
                'spoof_budget':   round(SPOOF_BUDGET, 4),
                'final_x':        round(ZONE_X_MIN + pen, 4),
                'violation':      pen > 0,
                'penetration_m':  round(pen, 4),
                'expected_pen_m': round(pen, 4),
                'trial_id':       f'analytical_d{delta}_s{seed}',
                'seed':           seed,
                'source':         'analytical',
            })
    _save(results)
    _summary(results)
    return results


# ==============================================================================
# PLOT MODE
# ==============================================================================
def run_plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from collections import defaultdict

    with open(OUT_FILE) as f:
        results = json.load(f)

    by_delta = defaultdict(list)
    for r in results:
        if 'penetration_m' in r and r.get('penetration_m') is not None:
            by_delta[r['delta_spoof']].append(r['penetration_m'])

    deltas = sorted(by_delta)
    means  = [sum(by_delta[d]) / len(by_delta[d]) for d in deltas]
    sds    = [(sum((v - means[i]) ** 2 for v in by_delta[d])
               / len(by_delta[d])) ** 0.5 for i, d in enumerate(deltas)]
    pred   = [expected_pen(d) for d in deltas]
    src    = results[0].get('source', 'gazebo')

    # Empirical threshold: linear-interpolate where mean penetration crosses 0
    emp = None
    for i in range(len(deltas) - 1):
        if means[i] <= 0 < means[i + 1]:
            f = (0 - means[i]) / (means[i + 1] - means[i])
            emp = deltas[i] + f * (deltas[i + 1] - deltas[i])
            break

    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    ax.axhspan(-1, 0, color='#e8f5e9', zorder=0)
    ax.axhspan(0, 1.2, color='#fdecea', zorder=0)
    ax.axhline(0, color='#888', lw=0.8)
    ax.axvline(SPOOF_BUDGET, color='#1565c0', lw=1.2, ls='--',
               label=f'certified $\\Delta_{{spoof}}=M_{{est}}{{+}}M_{{track}}'
                     f'={SPOOF_BUDGET:.3f}$ m')
    if emp is not None:
        ax.axvline(emp, color='#6a1b9a', lw=1.2, ls=':',
                   label=f'empirical threshold $\\approx {emp:.3f}$ m')
    ax.plot(deltas, pred, '--', color='#555', lw=1.0,
            label='kinematic prediction $\\delta-\\Delta_{spoof}$')
    ax.errorbar(deltas, means, yerr=sds, fmt='o-', color='#c62828',
                lw=1.8, ms=6, capsize=3,
                label=f'measured penetration ({src}, 3 seeds)')

    ax.set_xlabel('persistent spoof offset $\\delta$ (m)')
    ax.set_ylabel('true zone penetration (m)')
    ax.set_title('Persistent stealthy pose spoofing vs. safety margin')
    ax.text(0.02, -0.85, 'SAFE (absorbed)', color='#2e7d32',
            transform=ax.get_yaxis_transform(), fontsize=9, ha='left')
    ax.text(0.02, 0.20, 'VIOLATION', color='#c62828',
            transform=ax.get_yaxis_transform(), fontsize=9, ha='left')
    ax.grid(True, alpha=0.25)
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
    ax.set_ylim(min(-0.55, min(means) - 0.05), max(0.65, max(means) + 0.05))
    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG_FILE), exist_ok=True)
    fig.savefig(FIG_FILE, dpi=150)
    print(f'[spoof] Figure -> {FIG_FILE}')


# ==============================================================================
# ORCHESTRATION MODE  (Gazebo, modeled on run_s6b_braking.py)
# ==============================================================================
procs = []


class _Stub:
    stdout = ''
    stderr = ''


def gz(cmd, timeout=15):
    """Run a shell/ros2 helper command. Never raises: a slow ros2 CLI call
    (discovery latency) must not crash the whole orchestrator."""
    try:
        return subprocess.run(['bash', '-c', SETUP + cmd],
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return _Stub()
    except Exception:
        return _Stub()


def start_gazebo():
    print('[spoof] Cleaning up old processes...')
    # NOTE: do NOT pkill 'run_spoof' -- that pattern matches this orchestrator's
    # own command line and would kill ourselves. Trial subprocesses are managed
    # synchronously (subprocess.run) so they never leak between runs.
    for pat in ['gz sim', 'robot_state_publisher', 'ros2_control',
                'topic_tools relay', 'parameter_bridge', 'ros_gz_bridge']:
        subprocess.run(['pkill', '-f', pat], capture_output=True)
    time.sleep(5)

    print('[spoof] Starting Gazebo (empty world)...')
    # mobile_manipulator.launch.py = world + robot spawn + parameter_bridge
    # (world.launch.py alone does NOT publish /odom_real -- no robot spawned).
    p = subprocess.Popen(
        ['bash', '-c', SETUP +
         'ros2 launch mobile_manip_moveit_config mobile_manipulator.launch.py '
         'world:=empty.sdf'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid)
    procs.append(p)
    print('[spoof] Waiting 90s...')
    time.sleep(90)

    p2 = subprocess.Popen(
        ['bash', '-c', SETUP + 'ros2 run topic_tools relay /odom_real /odom'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid)
    procs.append(p2)
    time.sleep(8)

    # Verify /odom actually carries DATA (topic existing is not enough --
    # a dead relay still advertises the topic with zero messages).
    ready = False
    for _ in range(15):
        r = gz('timeout 5 ros2 topic echo /odom --once 2>/dev/null '
               '| grep -m1 "position"', timeout=8)
        if 'position' in r.stdout:
            ready = True
            break
        time.sleep(3)
    print(f'[spoof] Gazebo ready (odom data: {ready}).')
    if not ready:
        print('[spoof][WARN] /odom has no data; trials will likely fail.')


def teleport(x=0.0, y=0.0):
    # best-effort stop (bounded; gz() swallows a slow/timed-out CLI call)
    gz('timeout 4 ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{}"',
       timeout=6)
    time.sleep(0.3)
    cmd = (f'gz service -s /world/{GZ_WORLD}/set_pose '
           f'--reqtype gz.msgs.Pose --reptype gz.msgs.Boolean --timeout 3000 '
           f'--req \'name: "mobile_manip", '
           f'position: {{x: {x}, y: {y}, z: 0.1}}, '
           f'orientation: {{x: 0.0, y: 0.0, z: 0.0, w: 1.0}}\'')
    r = gz(cmd)
    if 'true' not in r.stdout.lower():
        print(f'[spoof][WARN] teleport may have failed: {r.stderr[:80]}')
    time.sleep(2.0)


def run_trial(delta):
    cmd = SETUP + f'python3 -u {os.path.abspath(__file__)} --trial {delta}'
    try:
        r = subprocess.run(['bash', '-c', cmd],
                           capture_output=True, text=True, timeout=60)
        for line in reversed(r.stdout.strip().split('\n')):
            line = line.strip()
            if line.startswith('{'):
                return json.loads(line)
        print(f'  [WARN] no JSON. stderr: {r.stderr.strip()[-200:]}')
        return None
    except subprocess.TimeoutExpired:
        print('  [ERROR] trial timeout')
        return None


def stop_all():
    for p in procs:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except Exception:
            pass


def _save(results):
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'[spoof] Saved -> {OUT_FILE}')


def _summary(results):
    print(f'\n=== Spoof budget sweep (Delta_spoof = M_est+M_track '
          f'= {SPOOF_BUDGET:.3f} m) ===')
    print(f"{'delta(m)':>9} {'pen_actual(m)':>14} {'pen_pred(m)':>12} "
          f"{'result':>10}")
    print('-' * 50)
    from collections import defaultdict
    by_delta = defaultdict(list)
    for r in results:
        if 'penetration_m' in r:
            by_delta[r['delta_spoof']].append(r)
    for d in sorted(by_delta):
        rs = by_delta[d]
        pen = sum(x['penetration_m'] for x in rs) / len(rs)
        viol = sum(1 for x in rs if x['violation'])
        st = f'{viol}/{len(rs)} VIOL' if viol else 'safe'
        print(f'{d:>9.3f} {pen:>+14.4f} {expected_pen(d):>+12.4f} {st:>10}')


def main():
    sys.stdout.reconfigure(line_buffering=True)

    results = []
    total = len(DELTAS) * SEEDS
    idx = 0
    for seed in range(SEEDS):
        # Restart Gazebo per seed: diff-drive yaw/odom drift accumulates across
        # trials (documented "restart every few trials" issue), so a fresh sim
        # per seed keeps each seed's odom clean and each seed independent.
        print(f'\n[spoof] === Seed {seed}: fresh Gazebo ===')
        start_gazebo()

        for delta in DELTAS:
            idx += 1
            print(f'\n[spoof] Trial {idx}/{total}: delta={delta} seed={seed} '
                  f'(pred pen={expected_pen(delta):+.3f}m)')
            teleport(0.0, 0.0)
            res = run_trial(delta)
            if res is None:
                res = {'delta_spoof': delta, 'error': 'failed'}
            res['trial_id'] = f'spoof_d{delta}_s{seed}'
            res['seed'] = seed
            res['source'] = 'gazebo'
            results.append(res)
            if res.get('violation') is not None and 'error' not in res:
                st = 'VIOLATION' if res['violation'] else 'safe'
                print(f'       final_x={res["final_x"]:.4f}  '
                      f'pen={res["penetration_m"]:+.4f}m  [{st}]')
            elif 'error' in res:
                print(f'       [INVALID] {res["error"]} '
                      f'{res.get("x_start", "")}')
        stop_all()
        del procs[:]

    _save(results)
    _summary(results)
    print('[spoof] Done.')


if __name__ == '__main__':
    if '--analytical' in sys.argv:
        run_analytical()
    elif '--plot' in sys.argv:
        run_plot()
    else:
        main()
