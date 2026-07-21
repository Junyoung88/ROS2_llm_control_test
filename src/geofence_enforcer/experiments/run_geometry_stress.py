#!/usr/bin/env python3
"""
Geometry stress tests for reviewer 3: narrow corridors + adjacent/overlapping zones.

Drives the REAL enforcement geometry (geofence_core.GeofencePolicy:
evaluate_point / evaluate_segment / _segment_intersects_polygon), not a
re-implementation, so the results reflect Algorithm 1 exactly.

R3 concern 1 (narrow corridor false rejection): a straight path between two
forbidden zones is rejected when it grazes the inflated boundary. We sweep
corridor width and report where PETSE transitions from reject to allow, and
quantify how much of that is genuinely-unsafe (robot footprint would clip the
zone) vs. conservative.

R3 concern 2 (adjacent/overlapping zones -> margin summing blocks the robot):
we show the enforced unsafe region is the UNION of per-zone inflations, not a
sum. A point between two zones is rejected iff it is inside a SINGLE zone's
margin; margins do not add. We sweep zone separation and confirm the free gap
equals separation - 2*margin (union), never separation - 4*margin (sum).

Output: experiment_results/gazebo_s1_s6/geometry_stress_results.json
        figures/geometry_stress.png
"""
import json
import os
import sys

# Import the real enforcement core
CORE_DIR = ('/home/jim/ros2_motion_planning_tutorials/src/'
            'mobile_manipulator_tutorial/src/geofence_policy_enforcer')
sys.path.insert(0, CORE_DIR)
from geofence_policy_enforcer.geofence_core import (   # noqa: E402
    GeofencePolicy, GeofenceZone, ZoneType, UncertaintyParams, PolicyAction,
)

OUT = ('/home/jim/ros2_motion_planning_tutorials/experiment_results/'
       'gazebo_s1_s6/geometry_stress_results.json')
FIG = '/home/jim/ros2_motion_planning_tutorials/figures/geometry_stress.png'

ROBOT_RADIUS = 0.22   # half-footprint of the diff-drive base (m)


def make_policy():
    """GeofencePolicy with paper-simulation margin (M = 0.562 m)."""
    pol = GeofencePolicy.__new__(GeofencePolicy)
    pol.zones = []
    pol.uncertainty = UncertaintyParams(
        localization_sigma=0.15, v_max=0.5, latency=0.1, a_max=2.5,
        e_0=0.03, c_1=0.04, epsilon=0.003, use_epsilon_formulation=True,
    )
    pol._config_path = None
    return pol


def rect(name, cx, cy, w, h):
    """Axis-aligned rectangle zone centered at (cx, cy)."""
    hw, hh = w / 2.0, h / 2.0
    return GeofenceZone(name=name, zone_type=ZoneType.FORBIDDEN, vertices=[
        (cx - hw, cy - hh), (cx + hw, cy - hh),
        (cx + hw, cy + hh), (cx - hw, cy + hh)])


# ==========================================================================
# ③a  Narrow corridor: two walls, sweep the free width
# ==========================================================================
def corridor_sweep():
    """Two forbidden walls form a vertical corridor along y. A straight path
    goes up the centerline (x=0). Sweep the wall gap and record PETSE's
    decision, the true centerline clearance, and the footprint clearance."""
    pol = make_policy()
    M = pol.uncertainty.compute_margin(velocity=pol.uncertainty.v_max)

    # walls span y in [-2, 2], each 1 m thick, placed symmetrically at +-(gap/2 + 0.5)
    WALL_THICK = 1.0
    gaps = [round(g, 3) for g in
            [0.4, 0.6, 0.8, 1.0, 1.1, 1.124, 1.2, 1.4, 1.6, 2.0, 2.5]]
    rows = []
    for gap in gaps:
        pol.zones = [
            rect('wall_L', -(gap / 2 + WALL_THICK / 2), 0.0, WALL_THICK, 4.0),
            rect('wall_R', +(gap / 2 + WALL_THICK / 2), 0.0, WALL_THICK, 4.0),
        ]
        # centerline path from below to above the corridor
        dec = pol.evaluate_segment((0.0, -3.0), (0.0, 3.0))
        allowed = dec.action == PolicyAction.ALLOW
        centerline_clearance = gap / 2.0          # wall face to centerline
        footprint_clearance = gap / 2.0 - ROBOT_RADIUS
        rows.append({
            'gap': gap,
            'margin': round(M, 4),
            'centerline_clearance': round(centerline_clearance, 4),
            'footprint_clearance': round(footprint_clearance, 4),
            'petse_allows': allowed,
            # genuinely unsafe iff the robot body would clip a wall
            'genuinely_unsafe': footprint_clearance < 0,
            'reason': dec.reason,
        })
    # theoretical allow threshold: centerline clearance > margin -> gap > 2M
    return {'test': 'narrow_corridor', 'margin': round(M, 4),
            'robot_radius': ROBOT_RADIUS,
            'allow_threshold_gap': round(2 * M, 4), 'rows': rows}


# ==========================================================================
# ③b  Adjacent/overlapping zones: union vs sum
# ==========================================================================
def adjacent_zones_sweep():
    """Two square zones separated along x by a gap between their near faces.
    Probe the midpoint: PETSE rejects it iff it is within ONE zone's margin
    (union), never requiring the SUM of both margins. Sweep the separation and
    record the free-gap width the enforcer actually leaves."""
    pol = make_policy()
    M = pol.uncertainty.compute_margin(velocity=pol.uncertainty.v_max)
    ZW = 1.0  # zone side length

    seps = [round(s, 3) for s in
            [0.2, 0.5, 0.9, 1.0, 1.124, 1.2, 1.5, 2.0, 2.5, 3.0]]
    rows = []
    for sep in seps:
        # near faces at x = -sep/2 and +sep/2
        cxL = -(sep / 2 + ZW / 2)
        cxR = +(sep / 2 + ZW / 2)
        pol.zones = [rect('zA', cxL, 0.0, ZW, ZW),
                     rect('zB', cxR, 0.0, ZW, ZW)]

        # midpoint between the two zones
        mid = pol.evaluate_point(0.0, 0.0)
        mid_rejected = mid.action == PolicyAction.REJECT

        # measure the actual free gap along x by scanning the corridor centre
        free = 0.0
        x = -sep / 2
        step = 0.005
        while x <= sep / 2:
            if pol.evaluate_point(x, 0.0).action == PolicyAction.ALLOW:
                free += step
            x += step
        free = round(free, 3)

        # union prediction: free = max(0, sep - 2M);  sum would be sep - 4M
        union_pred = max(0.0, sep - 2 * M)
        sum_pred = max(0.0, sep - 4 * M)
        rows.append({
            'separation': sep,
            'margin': round(M, 4),
            'midpoint_rejected': mid_rejected,
            'midpoint_min_dist': round(mid.min_distance_to_forbidden, 4),
            'free_gap_measured': free,
            'free_gap_union_pred': round(union_pred, 4),
            'free_gap_sum_pred': round(sum_pred, 4),
        })
    return {'test': 'adjacent_zones', 'margin': round(M, 4),
            'zone_width': ZW, 'rows': rows}


# ==========================================================================
# Plot
# ==========================================================================
def plot(data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    corr = next(d for d in data if d['test'] == 'narrow_corridor')
    adj = next(d for d in data if d['test'] == 'adjacent_zones')
    M = corr['margin']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # --- ③a corridor ---
    gaps = [r['gap'] for r in corr['rows']]
    cl = [r['centerline_clearance'] for r in corr['rows']]
    fc = [r['footprint_clearance'] for r in corr['rows']]
    allow = [r['petse_allows'] for r in corr['rows']]
    ax1.axvline(corr['allow_threshold_gap'], color='#1565c0', ls='--', lw=1.2,
                label=f'allow threshold gap $=2M={corr["allow_threshold_gap"]}$ m')
    ax1.axhline(M, color='#888', ls=':', lw=1.0, label=f'margin $M={M}$ m')
    ax1.plot(gaps, cl, 'o-', color='#c62828', label='centerline clearance')
    ax1.plot(gaps, fc, 's--', color='#2e7d32', label='footprint clearance')
    for g, a, c in zip(gaps, allow, cl):
        ax1.annotate('ALLOW' if a else 'REJECT', (g, c), fontsize=6,
                     color=('#2e7d32' if a else '#c62828'),
                     xytext=(0, 6), textcoords='offset points', ha='center')
    ax1.set_xlabel('corridor gap width (m)')
    ax1.set_ylabel('clearance (m)')
    ax1.set_title('(a) Narrow corridor: reject $\\Leftrightarrow$ gap $< 2M$')
    ax1.grid(True, alpha=0.25); ax1.legend(fontsize=7, loc='upper left')

    # --- ③b adjacent zones: union vs sum ---
    seps = [r['separation'] for r in adj['rows']]
    meas = [r['free_gap_measured'] for r in adj['rows']]
    up = [r['free_gap_union_pred'] for r in adj['rows']]
    sp = [r['free_gap_sum_pred'] for r in adj['rows']]
    ax2.plot(seps, up, '-', color='#1565c0', lw=1.4,
             label='union prediction $\\max(0,\\,s-2M)$')
    ax2.plot(seps, sp, ':', color='#999', lw=1.4,
             label='(sum would be $s-4M$)')
    ax2.plot(seps, meas, 'o', color='#c62828', ms=6,
             label='measured free gap (real code)')
    ax2.set_xlabel('zone separation $s$ (m)')
    ax2.set_ylabel('free passage width (m)')
    ax2.set_title('(b) Adjacent zones: enforced region is the UNION')
    ax2.grid(True, alpha=0.25); ax2.legend(fontsize=7, loc='upper left')

    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, dpi=150)
    print(f'[geom] Figure -> {FIG}')


def main():
    data = [corridor_sweep(), adjacent_zones_sweep()]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(data, f, indent=2)
    print(f'[geom] Saved -> {OUT}')

    # console summary
    corr = data[0]
    print(f"\n=== (a) Narrow corridor (M={corr['margin']} m, "
          f"allow threshold gap=2M={corr['allow_threshold_gap']} m) ===")
    print(f"{'gap':>6} {'centerline':>11} {'footprint':>10} "
          f"{'PETSE':>7} {'genuinely_unsafe':>16}")
    for r in corr['rows']:
        print(f"{r['gap']:>6.3f} {r['centerline_clearance']:>11.3f} "
              f"{r['footprint_clearance']:>10.3f} "
              f"{'ALLOW' if r['petse_allows'] else 'REJECT':>7} "
              f"{str(r['genuinely_unsafe']):>16}")

    adj = data[1]
    print(f"\n=== (b) Adjacent zones (M={adj['margin']} m) — "
          f"union vs sum ===")
    print(f"{'sep':>6} {'mid_reject':>11} {'free_meas':>10} "
          f"{'union_pred':>11} {'sum_pred':>9}")
    for r in adj['rows']:
        print(f"{r['separation']:>6.3f} {str(r['midpoint_rejected']):>11} "
              f"{r['free_gap_measured']:>10.3f} "
              f"{r['free_gap_union_pred']:>11.3f} "
              f"{r['free_gap_sum_pred']:>9.3f}")

    plot(data)


if __name__ == '__main__':
    main()
