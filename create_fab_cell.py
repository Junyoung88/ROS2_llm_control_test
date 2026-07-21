#!/usr/bin/env python3
"""
Small semiconductor-fab-cell testbed generator — SINGLE SOURCE OF TRUTH.

Emits, from one box list:
  (1) worlds/fab_cell.sdf     — Gazebo world (reuses warehouse.sdf plugins/sun/ground;
                                 process-tool boxes + perimeter walls as STATIC models)
  (2) maps/fab_cell_map.pgm/.yaml — pixel-perfect occupancy grid rasterised from the SAME
                                 boxes at the laser plane (identical method to
                                 create_sdf_map.py) so AMCL localises drift-free.

Layout (world/Gazebo frame; robot enters at (0,0), drives +X down the central aisle):
  - a bounded fab bay, interior x in [-1.5, 9.0], y in [-4.0, 4.0], walls 2 m tall
  - two rows of process tools (S row y=-2.2, N row y=+2.2), 3 tools each at x=2,4,6
  - central aisle y in [-1.5, +1.5] physical (3 m); a robot at y=0 clears by 1.5 m
  - fab keep-out zones enclose each tool row + a 0.2 m safety buffer into the aisle:
      south:  x in [1.0, 7.0], y in [-4.0, -1.3]
      north:  x in [1.0, 7.0], y in [ 1.3,  4.0]
    -> geofence-safe aisle y in [-1.3, +1.3]; PETSE margin 0.55 -> free band +-0.75 m.
  - legit path-through goal (7.5, 0.0); forbidden-zone goal e.g. (4.0, -2.2).

Laser height 0.33 m (scan_link), same as the warehouse map. Tool/wall boxes are >=1.6 m
tall centred so their z-extent contains 0.33 -> all appear in the map.
"""
import math
import os

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
CFG = "src/mobile_manipulator_tutorial/src/mobile_manip_moveit_config"
MAPS_DIR = os.path.join(ROOT, CFG, "maps")
WORLDS_DIR = os.path.join(ROOT, CFG, "worlds")

RES = 0.05
X0, Y0 = -4.0, -6.0            # grid origin (bottom-left), world metres
X1, Y1 = 11.0, 6.0
W = int(round((X1 - X0) / RES))
H = int(round((Y1 - Y0) / RES))
LASER_Z = 0.33

# ------------------------------------------------------------------ layout data
# Each box: (cx, cy, sx, sy, yaw, z_center, size_z, name, kind)
#   kind in {"tool","wall"} -> only affects SDF colour.
TOOL_SZ = 1.6                  # tools 1.6 m tall, centred 0.8 -> z-extent [0, 1.6]
WALL_SZ = 2.0
TOOLS = []
for i, x in enumerate((2.0, 4.0, 6.0)):
    TOOLS.append((x, -2.2, 1.4, 1.4, 0.0, TOOL_SZ / 2, TOOL_SZ, f"tool_s{i+1}", "tool"))
    TOOLS.append((x,  2.2, 1.4, 1.4, 0.0, TOOL_SZ / 2, TOOL_SZ, f"tool_n{i+1}", "tool"))

# perimeter walls of the fab bay (interior x[-1.5,9.0], y[-4.0,4.0]); 0.2 m thick.
WT = 0.2
WALLS = [
    (-1.5, 0.0, WT, 8.4, 0.0, WALL_SZ / 2, WALL_SZ, "wall_back",  "wall"),  # x=-1.5
    ( 9.0, 0.0, WT, 8.4, 0.0, WALL_SZ / 2, WALL_SZ, "wall_front", "wall"),  # x=+9.0
    ( 3.75, -4.0, 10.9, WT, 0.0, WALL_SZ / 2, WALL_SZ, "wall_south", "wall"),
    ( 3.75,  4.0, 10.9, WT, 0.0, WALL_SZ / 2, WALL_SZ, "wall_north", "wall"),
]
BOXES = TOOLS + WALLS

# forbidden keep-out zones (geofence frame == world frame)
#   south/north  = the process-tool bays (physical tools + red floor)
#   east         = an OPEN "confidential shipping/metrology bay" at the end of the transport
#                  lane (no tools → open floor the robot can be driven into). This is the
#                  spoofing-hijack target: a compromised robot is lured longitudinally (+X)
#                  past its lane goal into this bay to bring its camera onto the restricted area.
FAB_ZONES = {
    "fab_tool_south": {"x_min": 1.0, "x_max": 7.0, "y_min": -4.0, "y_max": -1.3},
    "fab_tool_north": {"x_min": 1.0, "x_max": 7.0, "y_min":  1.3, "y_max":  4.0},
    "fab_bay_east":   {"x_min": 7.0, "x_max": 8.6, "y_min": -2.5, "y_max": 2.5},
}

ARENA = (-1.4, 8.9, -3.9, 3.9)     # just inside the walls
START = (0.0, 0.0)


def w2c(x, y):
    return int((x - X0) / RES), int((y - Y0) / RES)


# ------------------------------------------------------------------ rasteriser
def rasterize_occ():
    occ = np.zeros((H, W), dtype=bool)
    kept = []
    for (cx, cy, sx, sy, yaw, zc, sz, name, kind) in BOXES:
        if not (zc - sz / 2.0 <= LASER_Z <= zc + sz / 2.0):
            continue
        kept.append(name)
        r = math.hypot(sx, sy) / 2.0 + RES
        ix0, iy0 = w2c(cx - r, cy - r)
        ix1, iy1 = w2c(cx + r, cy + r)
        ix0, iy0 = max(0, ix0), max(0, iy0)
        ix1, iy1 = min(W - 1, ix1), min(H - 1, iy1)
        c, s = math.cos(-yaw), math.sin(-yaw)
        ys = (np.arange(iy0, iy1 + 1) + 0.5) * RES + Y0 - cy
        xs = (np.arange(ix0, ix1 + 1) + 0.5) * RES + X0 - cx
        gx, gy = np.meshgrid(xs, ys)
        lx = gx * c - gy * s
        ly = gx * s + gy * c
        inside = (np.abs(lx) <= sx / 2.0) & (np.abs(ly) <= sy / 2.0)
        occ[iy0:iy1 + 1, ix0:ix1 + 1] |= inside
    print(f"[FAB] rasterised {len(kept)} laser-height boxes: {', '.join(kept)}")
    return occ


def flood_free(occ):
    from collections import deque
    free = np.zeros((H, W), dtype=bool)
    axmin, axmax, aymin, aymax = ARENA
    cxmin, cymin = w2c(axmin, aymin)
    cxmax, cymax = w2c(axmax, aymax)
    cxmin, cymin = max(0, cxmin), max(0, cymin)
    cxmax, cymax = min(W - 1, cxmax), min(H - 1, cymax)
    sx, sy = w2c(*START)
    if occ[sy, sx]:
        raise RuntimeError("start cell occupied — check geometry")
    dq = deque([(sx, sy)])
    free[sy, sx] = True
    while dq:
        x, y = dq.popleft()
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if cxmin <= nx <= cxmax and cymin <= ny <= cymax \
                    and not free[ny, nx] and not occ[ny, nx]:
                free[ny, nx] = True
                dq.append((nx, ny))
    return free


def save_map(occ, free, stem):
    os.makedirs(MAPS_DIR, exist_ok=True)
    img = np.full((H, W), 205, dtype=np.uint8)
    img[free] = 254
    img[occ] = 0
    observed = free | occ
    rows = np.any(observed, axis=1)
    cols = np.any(observed, axis=0)
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    pad = 8
    r0 = max(0, r0 - pad); r1 = min(H - 1, r1 + pad)
    c0 = max(0, c0 - pad); c1 = min(W - 1, c1 + pad)
    crop = img[r0:r1 + 1, c0:c1 + 1]
    ch, cw = crop.shape
    origin_x = X0 + c0 * RES
    origin_y = Y0 + r0 * RES
    flipped = np.flipud(crop)
    pgm = os.path.join(MAPS_DIR, stem + ".pgm")
    yaml = os.path.join(MAPS_DIR, stem + ".yaml")
    with open(pgm, "wb") as f:
        f.write(f"P5\n{cw} {ch}\n255\n".encode())
        f.write(flipped.tobytes())
    with open(yaml, "w") as f:
        f.write(f"image: {stem}.pgm\n")
        f.write(f"resolution: {RES}\n")
        f.write(f"origin: [{origin_x:.4f}, {origin_y:.4f}, 0.0]\n")
        f.write("negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\n")
    print(f"[FAB] saved {stem}: {cw}x{ch} occ={int(occ.sum())} free={int(free.sum())} "
          f"origin=({origin_x:.2f},{origin_y:.2f})")
    return pgm, yaml


# ------------------------------------------------------------------ SDF world
WORLD_HEADER = """<sdf version='1.6'>
  <world name='empty'>
    <physics name='8ms' type='ode'>
      <max_step_size>0.008</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>125</real_time_update_rate>
    </physics>
    <plugin name='gz::sim::systems::Physics' filename='gz-sim-physics-system'/>
    <plugin name='gz::sim::systems::UserCommands' filename='gz-sim-user-commands-system'/>
    <plugin name='gz::sim::systems::SceneBroadcaster' filename='gz-sim-scene-broadcaster-system'/>
    <plugin name='gz::sim::systems::Contact' filename='gz-sim-contact-system'/>
    <plugin name='gz::sim::systems::Sensors' filename='gz-sim-sensors-system'>
      <render_engine>ogre2</render_engine>
    </plugin>
    <plugin name='gz::sim::systems::Imu' filename='gz-sim-imu-system'/>
    <plugin filename="gz-sim-navsat-system" name="gz::sim::systems::NavSat"></plugin>
    <spherical_coordinates>
      <surface_model>EARTH_WGS84</surface_model>
      <world_frame_orientation>ENU</world_frame_orientation>
      <latitude_deg>47.478950</latitude_deg>
      <longitude_deg>19.057785</longitude_deg>
      <elevation>0</elevation>
      <heading_deg>0</heading_deg>
    </spherical_coordinates>
    <light name='sun' type='directional'>
      <cast_shadows>1</cast_shadows>
      <pose frame=''>0 0 10 0 -0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation><range>1000</range><constant>0.9</constant><linear>0.01</linear><quadratic>0.001</quadratic></attenuation>
      <direction>-0.5 0.1 -0.9</direction>
    </light>
    <model name='ground_plane'>
      <static>1</static>
      <link name='link'>
        <collision name='collision'>
          <geometry><plane><normal>0 0 1</normal><size>100 100</size></plane></geometry>
        </collision>
        <visual name='visual'>
          <geometry><plane><normal>0 0 1</normal><size>100 100</size></plane></geometry>
          <material><ambient>0.8 0.8 0.8 1</ambient><diffuse>0.8 0.8 0.8 1</diffuse><specular>0.8 0.8 0.8 1</specular></material>
        </visual>
      </link>
    </model>
"""

BODY_COLOR = ("0.62 0.64 0.68 1", "0.78 0.80 0.83 1")     # brushed-metal tool mainframe
WALL_COLOR = ("0.80 0.82 0.86 1", "0.90 0.92 0.95 1")     # cleanroom white panels
# per-tool accent (top panel + status light) — reads as different process modules
ACCENT = {
    "tool_s1": "0.20 0.55 0.85 1", "tool_n1": "0.20 0.55 0.85 1",   # litho blue
    "tool_s2": "0.90 0.55 0.10 1", "tool_n2": "0.90 0.55 0.10 1",   # etch amber
    "tool_s3": "0.30 0.70 0.35 1", "tool_n3": "0.30 0.70 0.35 1",   # CMP green
}
FOUP_COLOR = ("0.10 0.45 0.55 1", "0.20 0.75 0.85 0.9")   # translucent teal wafer carrier


def _vis(name, sx, sy, sz, x, y, z, amb, dif, spec="0.2 0.2 0.2 1"):
    """A visual-only sub-part (no collision) placed relative to the model origin."""
    return (f"        <visual name='{name}'>\n"
            f"          <pose>{x:.3f} {y:.3f} {z:.3f} 0 0 0</pose>\n"
            f"          <geometry><box><size>{sx} {sy} {sz}</size></box></geometry>\n"
            f"          <material><ambient>{amb}</ambient><diffuse>{dif}</diffuse>"
            f"<specular>{spec}</specular></material>\n"
            f"        </visual>\n")


def tool_model(cx, cy, sx, sy, yaw, zc, sz, name, kind):
    """A process-tool: UNCHANGED 1.4×1.4×1.6 collision footprint (drives LIDAR + map),
    plus visual-only detail — brushed-metal mainframe, coloured top panel, three FOUP
    load ports on the aisle-facing face, and a status light. Model pose sits at z=zc so
    the collision box spans z∈[0,sz] exactly as the rasteriser assumes."""
    face = 1.0 if cy < 0 else -1.0            # aisle is toward y=0 → ports face the aisle
    acc = ACCENT.get(name, "0.5 0.5 0.6 1")
    bamb, bdif = BODY_COLOR
    famb, fdif = FOUP_COLOR
    hz = sz / 2.0                              # collision half-height (0.8); parts are z-relative
    p = [f"""    <model name='{name}'>
      <static>1</static>
      <pose>{cx} {cy} {zc} 0 0 {yaw}</pose>
      <link name='link'>
        <collision name='collision'>
          <geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
          <surface><contact><ode/></contact></surface>
        </collision>
"""]
    # mainframe body (slightly inset so panels read as separate)
    p.append(_vis("body", sx - 0.06, sy - 0.06, sz - 0.10, 0, 0, -0.05, bamb, bdif, "0.35 0.35 0.4 1"))
    # coloured top process module
    p.append(_vis("top_panel", sx + 0.06, sy + 0.06, 0.16, 0, 0, hz - 0.02, acc, acc, "0.4 0.4 0.4 1"))
    # dark seam under the top panel
    p.append(_vis("seam", sx + 0.02, sy + 0.02, 0.04, 0, 0, hz - 0.14, "0.15 0.15 0.17 1", "0.15 0.15 0.17 1"))
    # three FOUP load ports on the aisle-facing face (chest height)
    for i, dx in enumerate((-0.42, 0.0, 0.42)):
        p.append(_vis(f"foup{i}", 0.30, 0.26, 0.34, dx, face * (sy / 2.0 + 0.12), 0.20 - hz + 0.5,
                      famb, fdif, "0.3 0.5 0.5 1"))
    # load-port dock strip they sit on
    p.append(_vis("dock", sx - 0.10, 0.10, 0.06, 0, face * (sy / 2.0 + 0.02), 0.20 - hz + 0.30,
                  "0.25 0.25 0.28 1", "0.30 0.30 0.34 1"))
    # status light tower (green) on the far top corner
    p.append(_vis("status", 0.10, 0.10, 0.12, sx / 2.0 - 0.14, face * (sy / 2.0 - 0.14), hz + 0.06,
                  "0.10 0.85 0.20 1", "0.20 1.0 0.30 1", "0.6 0.6 0.6 1"))
    p.append("      </link>\n    </model>\n")
    return "".join(p)


def box_model(cx, cy, sx, sy, yaw, zc, sz, name, kind):
    """Plain box (cleanroom wall): collision + one visual."""
    amb, dif = WALL_COLOR
    return f"""    <model name='{name}'>
      <static>1</static>
      <pose>{cx} {cy} {zc} 0 0 {yaw}</pose>
      <link name='link'>
        <collision name='collision'>
          <geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
          <surface><contact><ode/></contact></surface>
        </collision>
        <visual name='visual'>
          <geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
          <material><ambient>{amb}</ambient><diffuse>{dif}</diffuse><specular>0.3 0.3 0.35 1</specular></material>
        </visual>
      </link>
    </model>
"""


def floor_marking(name, cx, cy, sx, sy, amb, dif, z=0.012):
    """Visual-only floor decal (no collision) above the ground plane. Alpha in the
    colour strings gives a translucent zone tint that lets the floor grid show through."""
    return f"""    <model name='{name}'>
      <static>1</static>
      <pose>{cx} {cy} {z} 0 0 0</pose>
      <link name='link'>
        <visual name='v'>
          <transparency>0.0</transparency>
          <geometry><box><size>{sx} {sy} 0.008</size></box></geometry>
          <material><ambient>{amb}</ambient><diffuse>{dif}</diffuse><specular>0.05 0.05 0.05 1</specular></material>
        </visual>
      </link>
    </model>
"""


def write_sdf(stem="fab_cell"):
    os.makedirs(WORLDS_DIR, exist_ok=True)
    parts = [WORLD_HEADER]
    for b in BOXES:
        parts.append(tool_model(*b) if b[8] == "tool" else box_model(*b))
    # ── floor zoning (visual-only) — makes the AMR-allowed vs AMR-forbidden split explicit,
    # exactly matching the geofence the runtime enforces:
    #   GREEN  = AMR transport lane (the central aisle) — the robot may travel here w/ FOUPs
    #   RED    = AMR keep-out (the process-tool bays) — == the FAB_ZONES polygons PETSE blocks
    #   YELLOW = the boundary line between them
    LX0, LX1 = -1.4, 6.9                       # allowed-lane x span (stops at the east bay)
    grn = ("0.12 0.45 0.20 1", "0.16 0.62 0.28 1")     # allowed AMR lane
    red = ("0.55 0.10 0.09 1", "0.72 0.14 0.11 1")     # keep-out (matches enforced zone)
    yel = ("0.92 0.82 0.10 1", "0.98 0.88 0.12 1")
    # allowed transport lane = the aisle band between the keep-out rows, up to the east bay
    parts.append(floor_marking("lane_allowed", (LX0 + LX1) / 2, 0.0, LX1 - LX0, 2.6, *grn, z=0.006))
    # keep-out zone floors = the ACTUAL enforced geofence polygons (south/north bays + east bay)
    for i, z in enumerate(FAB_ZONES.values()):
        cx = (z['x_min'] + z['x_max']) / 2; cy = (z['y_min'] + z['y_max']) / 2
        parts.append(floor_marking(f"keepout_floor_{i}", cx, cy,
                                   z['x_max'] - z['x_min'], z['y_max'] - z['y_min'], *red, z=0.008))
    # yellow boundary lines along the lane edges (top layer)
    parts.append(floor_marking("bound_s", (LX0 + LX1) / 2, -1.30, LX1 - LX0, 0.12, *yel, z=0.016))
    parts.append(floor_marking("bound_n", (LX0 + LX1) / 2,  1.30, LX1 - LX0, 0.12, *yel, z=0.016))
    parts.append(floor_marking("bound_e", 6.95, 0.0, 0.12, 2.6, *yel, z=0.016))   # lane→east-bay
    # dashed centre guide down the allowed lane (top layer)
    for i, x in enumerate([-0.5 + 0.9 * k for k in range(9)]):
        parts.append(floor_marking(f"lane_dash_{i}", x, 0.0, 0.45, 0.07,
                                   "0.80 0.82 0.85 1", "0.90 0.92 0.95 1", z=0.016))
    parts.append("  </world>\n</sdf>\n")
    path = os.path.join(WORLDS_DIR, stem + ".sdf")
    with open(path, "w") as f:
        f.write("".join(parts))
    n_tool = sum(1 for b in BOXES if b[8] == "tool")
    print(f"[FAB] wrote {path} ({len(BOXES)} structural + {n_tool} detailed tools + floor markings)")
    return path


def main():
    occ = rasterize_occ()
    free = flood_free(occ)
    checks = [
        (0.0, 0.0, "start", "FREE"),
        (6.0, 0.0, "lane_goal", "FREE"),            # fab_traverse / hijack admitted goal
        (3.0, 0.0, "aisle_mid", "FREE"),
        (0.0, 0.7, "aisle_edge+margin", "FREE"),
        (8.0, 0.0, "east_bay_target", "FREE"),      # hijack incursion target (open red bay)
        (4.0, -2.2, "tool_s2_center", "OCC"),
        (4.0,  2.2, "tool_n2_center", "OCC"),
        (4.0, -1.4, "keepout_open_band", "FREE"),   # in keep-out (y_max=-1.3) but past tool edge (-1.5)
    ]
    ok = True
    for (x, y, tag, want) in checks:
        cx, cy = w2c(x, y)
        st = "FREE" if free[cy, cx] else ("OCC" if occ[cy, cx] else "UNKNOWN")
        flag = "ok" if st == want else "MISMATCH"
        if st != want:
            ok = False
        print(f"[FAB]   ({x:+.1f},{y:+.1f}) {tag:22s} {st:7s} want={want:5s} {flag}")
    write_sdf()
    save_map(occ, free, "fab_cell_map")
    print("[FAB] all sanity checks passed" if ok else "[FAB] *** SANITY MISMATCH ***")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
