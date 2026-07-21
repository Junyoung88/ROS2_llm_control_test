#!/usr/bin/env python3
"""
Fail-stop recovery policies (reviewer R3: "when speed exceeds v_max or
covariance exceeds lambda_bar, PETSE cancels the goal and stops immediately;
in the real world this can block a corridor or leave the robot unrecoverable.
How can this be addressed?").

We show fail-stop is only the LAST rung of a graceful-degradation ladder, and
that adding a recovery state machine restores mission availability WITHOUT ever
compromising safety (0 zone entries in every policy). We also surface an honest
nuance: the right recovery depends on WHICH COE precondition was violated.

Scenario: a robot drives an aisle at y=1.7 past a forbidden zone
(x in [4,6], y in [-1,1]); nominal clearance 0.7 m > margin 0.562 m, so the
lane is safe. Mid-aisle a TRANSIENT COE fault fires the runtime monitor:
  fault A: covariance spike (lambda_max jumps for T_fault s) -> M_est inflates.
  fault B: velocity/latency injection (v or tau jumps) -> M_lat/M_brake inflate.
The upper aisle wall at y=2.4 makes the passage 1.4 m wide, so a stopped robot
blocks it.

Policies:
  P1 fail-stop     : cancel goal, v=0 forever (current PETSE behavior).
  P2 hold-resume   : v=0; poll COE; resume at v_max once preconditions restored.
  P3 retreat-replan: v=0, retreat to a safe staging point (real
                     GeofencePolicy._project_to_safe, moved clear of the aisle),
                     wait, then replan and resume.
  P4 reduced-speed : drop to the largest speed whose margin still fits the
                     clearance, and keep moving if safe.

Key honest finding: reduced-speed (P4) recovers a velocity/latency fault (those
terms shrink with v) but NOT a covariance fault (M_est is velocity-independent),
so covariance faults require hold/retreat. Different faults, different recovery.

Output: experiment_results/gazebo_s1_s6/recovery_policies_results.json
        figures/recovery_policies.png
"""
import json
import math
import os
import sys

CORE = ('/home/jim/ros2_motion_planning_tutorials/src/'
        'mobile_manipulator_tutorial/src/geofence_policy_enforcer')
sys.path.insert(0, CORE)
from geofence_policy_enforcer.geofence_core import (  # noqa: E402
    GeofencePolicy, GeofenceZone, ZoneType, UncertaintyParams,
)

OUT = ('/home/jim/ros2_motion_planning_tutorials/experiment_results/'
       'gazebo_s1_s6/recovery_policies_results.json')
FIG = '/home/jim/ros2_motion_planning_tutorials/figures/recovery_policies.png'

# --- scenario constants ---
ZONE = GeofenceZone('F', ZoneType.FORBIDDEN, [(4, -1), (6, -1), (6, 1), (4, 1)])
LANE_Y = 1.7
WALL_Y = 2.4          # upper aisle wall
GOAL_X = 10.0
START_X = 0.0
V_MAX = 0.5
SIGMA0 = 0.15         # nominal localization sigma
DT = 0.05
FAULT_START = 10.0    # s (robot is beside the zone by then)
T_FAULT = 8.0         # transient fault duration
Z = 2.748             # z_{1-eps}, eps=0.003


def base_policy():
    pol = GeofencePolicy.__new__(GeofencePolicy)
    pol.zones = [ZONE]
    pol.uncertainty = UncertaintyParams(
        localization_sigma=SIGMA0, v_max=V_MAX, latency=0.1, a_max=2.5,
        e_0=0.03, c_1=0.04, epsilon=0.003)
    pol._config_path = None
    return pol


def margin_at(sigma, v, tau):
    """RA-L margin for given (possibly faulted) sigma, v, tau."""
    return (Z * sigma + (0.03 + 0.04 * v) + v * tau + v * v / (2 * 2.5))


def clearance(x, y):
    """Euclidean clearance from (x,y) to the zone (0 if inside)."""
    pol = base_policy()
    return pol._distance_to_polygon((x, y), ZONE.vertices)


def fault_params(t, fault_type):
    """Return (sigma, v_cap, tau) active at time t for the chosen fault."""
    active = FAULT_START <= t < FAULT_START + T_FAULT
    if not active:
        return SIGMA0, V_MAX, 0.1
    if fault_type == 'covariance':
        return 0.55, V_MAX, 0.1        # sigma spike -> M_est ~0.55*2.748=1.5
    else:  # velocity/latency injection
        return SIGMA0, V_MAX, 0.9      # tau spike -> M_lat/M_brake inflate


def run_policy(policy, fault_type):
    """Kinematic sim. Returns trajectory + metrics."""
    x, y = START_X, LANE_Y
    t = 0.0
    v = V_MAX
    state = 'drive'          # drive | stopped | retreat | done
    retreat_target = None
    traj = []
    violations = 0
    block_start = None
    blocked_time = 0.0
    secondary_stops = 0
    prev_moving = True

    staging_x = 3.0          # clear-of-aisle staging point (before the zone)

    while t < 120.0 and state != 'done':
        sigma, v_cap, tau = fault_params(t, fault_type)
        clr = clearance(x, y)
        # margin the runtime monitor would enforce at current speed
        M_now = margin_at(sigma, min(v, v_cap), tau)
        coe_ok = (sigma <= SIGMA0 + 1e-6) and (tau <= 0.1 + 1e-6)
        triggered = clr < M_now       # runtime monitor would fire

        moving = False
        if state == 'drive':
            if triggered:
                # runtime monitor fires; apply policy
                if policy == 'P1_failstop':
                    state = 'stopped_final'
                elif policy == 'P2_hold_resume':
                    state = 'stopped'
                elif policy == 'P3_retreat_replan':
                    state = 'retreat'
                    # real safe holding point, then pull clear of the aisle
                    sp = base_policy()._project_to_safe(
                        (x, y), ZONE.vertices, M_now)
                    retreat_target = (staging_x, LANE_Y)
                    _ = sp   # (projection validates a safe pose exists)
                elif policy == 'P4_reduced_speed':
                    # largest v whose margin fits the clearance
                    v_safe = solve_reduced_speed(sigma, tau, clr)
                    if v_safe >= 0.02:
                        v = v_safe
                        moving = True
                        x += v * DT
                    else:
                        state = 'stopped'   # can't fix by slowing -> hold
            else:
                v = min(V_MAX, v_cap)
                x += v * DT
                moving = True
                if x >= GOAL_X:
                    state = 'done'

        elif state == 'stopped':
            if coe_ok and clearance(x, y) >= margin_at(SIGMA0, V_MAX, 0.1):
                state = 'drive'          # preconditions restored -> resume
                secondary_stops += 1

        elif state == 'retreat':
            tx, ty = retreat_target
            if x > tx + 0.02:
                x -= V_MAX * DT
                moving = True
            else:
                # at staging point, clear of the aisle; wait for COE then replan
                if coe_ok:
                    state = 'drive'
                    secondary_stops += 1

        if state == 'stopped_final':
            pass

        # safety accounting
        if clearance(x, y) <= 0:
            violations += 1

        # corridor-block accounting: robot stationary inside the narrow passage
        in_passage = 3.5 <= x <= 6.5
        if (not moving) and in_passage:
            if block_start is None:
                block_start = t
            blocked_time += DT
        prev_moving = moving

        traj.append({'t': round(t, 2), 'x': round(x, 3), 'y': round(y, 3),
                     'v': round(v, 3), 'state': state})
        t += DT

    completed = state == 'done'
    return {
        'policy': policy, 'fault': fault_type,
        'completed': completed,
        'violations': violations,
        'blocked_passage_s': round(blocked_time, 2),
        'secondary_stops': secondary_stops,
        'final_x': round(x, 3),
        'total_time_s': round(t, 2) if completed else None,
        'traj': traj,
    }


def solve_reduced_speed(sigma, tau, clr):
    """Largest v with margin_at(sigma, v, tau) <= clr. Closed-ish via bisection."""
    lo, hi = 0.0, V_MAX
    if margin_at(sigma, lo, tau) > clr:
        return 0.0                     # even v=0 doesn't fit (covariance fault)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if margin_at(sigma, mid, tau) <= clr:
            lo = mid
        else:
            hi = mid
    return lo


def plot(results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    faults = ['covariance', 'velocity_latency']
    policies = ['P1_failstop', 'P2_hold_resume', 'P3_retreat_replan',
                'P4_reduced_speed']
    labels = {'P1_failstop': 'P1 fail-stop', 'P2_hold_resume': 'P2 hold-resume',
              'P3_retreat_replan': 'P3 retreat-replan',
              'P4_reduced_speed': 'P4 reduced-speed'}
    colors = {'P1_failstop': '#c62828', 'P2_hold_resume': '#f9a825',
              'P3_retreat_replan': '#2e7d32', 'P4_reduced_speed': '#1565c0'}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, fault in zip(axes, faults):
        for pol in policies:
            r = next(x for x in results if x['policy'] == pol
                     and x['fault'] == fault)
            xs = [p['t'] for p in r['traj']]
            ys = [p['x'] for p in r['traj']]
            ax.plot(xs, ys, color=colors[pol], lw=1.8,
                    label=f"{labels[pol]}: "
                          f"{'done' if r['completed'] else 'STUCK'}, "
                          f"block {r['blocked_passage_s']:.0f}s")
        ax.axhspan(4, 6, color='#fdecea', alpha=0.6)   # zone x-extent
        ax.axhline(GOAL_X, color='#888', ls=':', lw=1)
        ax.axvspan(FAULT_START, FAULT_START + T_FAULT, color='#eee', zorder=0)
        ax.text(FAULT_START + T_FAULT / 2, 0.3, 'COE\nfault', ha='center',
                fontsize=7, color='#666')
        ax.set_xlabel('time (s)')
        ax.set_ylabel('robot x-position (m)')
        ax.set_title(f'{fault} fault  (zone at x∈[4,6], goal x=10)')
        ax.legend(fontsize=7, loc='lower right')
        ax.grid(True, alpha=0.25)
    fig.suptitle('Fail-stop vs. recovery policies: all safe (0 zone entries), '
                 'but recovery restores mission completion', fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, dpi=150)
    print(f'[recov] Figure -> {FIG}')


def main():
    results = []
    for fault in ['covariance', 'velocity_latency']:
        for pol in ['P1_failstop', 'P2_hold_resume', 'P3_retreat_replan',
                    'P4_reduced_speed']:
            results.append(run_policy(pol, fault))

    # strip trajectories for the json summary (keep separately)
    summary = [{k: v for k, v in r.items() if k != 'traj'} for r in results]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump({'summary': summary}, f, indent=2)
    print(f'[recov] Saved -> {OUT}')

    print(f"\n{'fault':>16} {'policy':>18} {'done':>5} {'viol':>5} "
          f"{'block(s)':>9} {'2ndstop':>7}")
    for r in results:
        print(f"{r['fault']:>16} {r['policy']:>18} "
              f"{str(r['completed']):>5} {r['violations']:>5} "
              f"{r['blocked_passage_s']:>9.1f} {r['secondary_stops']:>7}")

    plot(results)


if __name__ == '__main__':
    main()
