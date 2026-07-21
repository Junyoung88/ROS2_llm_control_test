#!/usr/bin/env python3
"""
Build an EXACT occupancy-grid map of warehouse.sdf directly from the SDF collision
geometry — no Gazebo, no driving, no scan registration, no pose estimation.

Why: every scan-registration attempt needed the robot's TRUE world pose, and the
ros_gz Pose_V->TF bridge gives neither a reliable model-root pose (the mobile_manip
canonical pose does NOT track the moving base) nor world-frame link poses
(base_footprint is reported model-relative ~= 0). So we skip robot poses entirely:
the warehouse is a handful of STATIC boxes at known world poses, and the laser sits
at a fixed z. Rasterising the box footprints that intersect the laser plane yields a
pixel-perfect, drift-free, world-aligned map. AMCL against this localises cleanly, so
the LiDAR-spoof cross-channel detector separates spoof (drift to metres) from clean.

Laser height: base_footprint(0) -> base_link_mobile(0) -> scan_link(+0.33) = 0.33 m.
A box is "seen" iff its world z-extent [zc - sz_z/2, zc + sz_z/2] contains 0.33.

Coordinates are world/Gazebo frame == the frame goals & the forbidden zone use.
"""
import math
import os

import numpy as np

MAPS_DIR = os.path.join(
    os.path.dirname(__file__),
    "src/mobile_manipulator_tutorial/src/mobile_manip_moveit_config/maps")

RES = 0.05
X0, Y0 = -11.0, -12.0          # grid origin (bottom-left), world metres
X1, Y1 = 11.0, 10.0
W = int(round((X1 - X0) / RES))
H = int(round((Y1 - Y0) / RES))
LASER_Z = 0.33                 # scan_link height above the floor

# workcell model is at this world offset (yaw 0); its collision <pose>s are relative.
WC = (-0.040107, 0.052292)

# Each obstacle: (world_cx, world_cy, size_x, size_y, yaw, z_center, size_z, name).
# Only boxes whose z-extent contains LASER_Z end up occupied.
def wc(rx, ry, sx, sy, zc, sz, name):
    return (rx + WC[0], ry + WC[1], sx, sy, 0.0, zc, sz, name)

BOXES = [
    # --- workcell outer walls (tall) ---
    wc(4.13686, -10.0212, 12.0768, 0.72902, 3.48855, 6.97771, "wall1"),
    wc(-6.58151, -8.924, 10.5921, 5.00198, 3.48855, 6.97771, "wall2"),
    wc(-9.25721, 0.6135, 0.72902, 15.2908, 3.48855, 6.97771, "wall3"),
    # --- racks (2.83 tall) ---
    wc(7.35056, 1.7443, 0.84444, 1.95581, 1.41643, 2.83286, "rack1"),
    wc(5.36795, 1.7443, 0.84444, 1.95581, 1.41643, 2.83286, "rack2"),
    wc(7.00142, -7.6467, 0.84444, 3.91162, 1.41643, 2.83286, "rack3"),
    wc(4.97338, -7.6467, 0.84444, 3.91162, 1.41643, 2.83286, "rack4"),
    wc(2.66949, -7.6467, 0.84444, 3.91162, 1.41643, 2.83286, "rack5"),
    # --- pallets ---
    wc(-3.33609, 6.31883, 2.14726, 1.79349, 1.04257, 2.08514, "large_pallet1"),
    wc(-5.74492, 6.31883, 2.14726, 1.79349, 1.04257, 2.08514, "large_pallet2"),
    wc(-7.55868, 1.91644, 2.14726, 1.79432, 1.04257, 2.08514, "large_pallet3"),
    wc(-1.09806, 6.31883, 2.14726, 1.79349, 0.61958, 1.23916, "box_pallet1"),
    # empty_pallet1 top = 0.327 < 0.33 -> laser passes over it -> OMITTED
    # --- poles (6 m) ---
    wc(5.02174, -2.39478, 0.49161, 0.49161, 3.02208, 6.04415, "pole1"),
    wc(0.47071, -2.39478, 0.49161, 0.49161, 3.02208, 6.04415, "pole2"),
    wc(-3.93786, -2.39478, 0.49161, 0.49161, 3.02208, 6.04415, "pole3"),
    # --- control panel ---
    wc(2.07015, 1.16224, 0.86733, 0.6001, 0.561315, 1.12263, "control_panel"),
    # --- perimeter walls (separate models) ---
    (9.8, -0.5, 0.5, 17.5, 0.0, 1.0, 2.0, "wall_side"),
    (0.0, 8.0, 0.5, 19.0, 1.57, 1.0, 2.0, "wall_side2"),
    # --- bins (0.6 x 0.6, body z[0.175,0.725]) ---
    (4.0, 6.0, 0.6, 0.6, 0.0, 0.45, 0.55, "bin0"),
    (5.0, 6.0, 0.6, 0.6, 0.0, 0.45, 0.55, "bin1"),
    (6.0, 6.0, 0.6, 0.6, 0.0, 0.45, 0.55, "bin2"),
    (7.0, 6.0, 0.6, 0.6, 0.0, 0.45, 0.55, "bin3"),
    (8.0, 6.0, 0.6, 0.6, 0.0, 0.45, 0.55, "bin4"),
]

# Flood-fill free space from the robot start, bounded to just inside the arena so a
# gap in the perimeter can't leak "free" across the whole plane.
ARENA = (-9.3, 9.7, -9.9, 7.8)   # xmin, xmax, ymin, ymax (world metres)
START = (0.0, 0.0)               # robot spawn (known free)


def w2c(x, y):
    return int((x - X0) / RES), int((y - Y0) / RES)


def rasterize_occ():
    occ = np.zeros((H, W), dtype=bool)
    kept = []
    for (cx, cy, sx, sy, yaw, zc, sz, name) in BOXES:
        if not (zc - sz / 2.0 <= LASER_Z <= zc + sz / 2.0):
            continue                      # laser plane misses this box
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
    print(f"[SDFMAP] rasterised {len(kept)} laser-height boxes: {', '.join(kept)}")
    return occ


def flood_free(occ):
    """4-connected flood-fill of free space from START, bounded to ARENA rect."""
    from collections import deque
    free = np.zeros((H, W), dtype=bool)
    axmin, axmax, aymin, aymax = ARENA
    cxmin, cymin = w2c(axmin, aymin)
    cxmax, cymax = w2c(axmax, aymax)
    cxmin, cymin = max(0, cxmin), max(0, cymin)
    cxmax, cymax = min(W - 1, cxmax), min(H - 1, cymax)
    sx, sy = w2c(*START)
    if occ[sy, sx]:
        raise RuntimeError("start cell is occupied — check geometry")
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
    img = np.full((H, W), 205, dtype=np.uint8)   # unknown
    img[free] = 254                               # free
    img[occ] = 0                                  # occupied (drawn last = boundaries)
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
    flipped = np.flipud(crop)                     # nav2 origin = bottom-left
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
    print(f"[SDFMAP] saved {stem}: {cw}x{ch} occ={int(occ.sum())} free={int(free.sum())} "
          f"origin=({origin_x:.2f},{origin_y:.2f})")
    return pgm, yaml


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="warehouse_map_sdf")
    args = ap.parse_args()
    occ = rasterize_occ()
    free = flood_free(occ)
    # sanity: forbidden zone + clean goal must be free
    for (x, y, tag) in [(0, 0, "start"), (0, 5, "clean_goal"),
                        (2.15, 0.0, "zone_center"), (3.0, 0.0, "zone_far_edge")]:
        cx, cy = w2c(x, y)
        st = "FREE" if free[cy, cx] else ("OCC" if occ[cy, cx] else "UNKNOWN")
        print(f"[SDFMAP]   ({x},{y}) {tag}: {st}")
    save_map(occ, free, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
