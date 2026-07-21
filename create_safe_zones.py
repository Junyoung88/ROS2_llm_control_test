#!/usr/bin/env python3
"""
Create Safe Test Zones

This script analyzes the static map to find completely free areas
and suggests new zone definitions for experiments.

Run: python3 create_safe_zones.py
"""

import struct
import math

# Map parameters
MAP_PATH = "/home/jim/ros2_motion_planning_tutorials/src/mobile_manipulator_tutorial/src/mobile_manip_moveit_config/maps/my_map.pgm"
RESOLUTION = 0.05  # m/pixel
ORIGIN_X = -8.21
ORIGIN_Y = -9.59

def read_pgm():
    """Read PGM file and return pixel array."""
    with open(MAP_PATH, "rb") as f:
        magic = f.readline().decode().strip()
        dims = f.readline().decode().strip()
        width, height = map(int, dims.split())
        max_val = int(f.readline().decode().strip())
        data = f.read()
        pixels = list(data)
    return pixels, width, height

def world_to_pixel(x, y, height):
    px = int((x - ORIGIN_X) / RESOLUTION)
    py = int(height - (y - ORIGIN_Y) / RESOLUTION)
    return px, py

def pixel_to_world(px, py, height):
    x = px * RESOLUTION + ORIGIN_X
    y = (height - py) * RESOLUTION + ORIGIN_Y
    return x, y

def check_zone_free(pixels, width, height, min_x, max_x, min_y, max_y):
    """Check if a rectangular zone is completely free."""
    px_min, py_max = world_to_pixel(min_x, min_y, height)
    px_max, py_min = world_to_pixel(max_x, max_y, height)

    total = 0
    free = 0
    occupied = 0

    for py in range(py_min, py_max + 1):
        for px in range(px_min, px_max + 1):
            if 0 <= px < width and 0 <= py < height:
                val = pixels[py * width + px]
                total += 1
                if val > 200:  # Free
                    free += 1
                elif val < 50:  # Occupied
                    occupied += 1

    if total == 0:
        return 0, 0
    return free / total * 100, occupied / total * 100


def find_free_zones(pixels, width, height, zone_size=1.0, step=0.5):
    """Find completely free rectangular zones."""
    free_zones = []

    # Search area (known navigable bounds from the home.sdf world)
    search_x_min, search_x_max = -6.0, 5.0
    search_y_min, search_y_max = -6.0, 5.0

    for x in [i * step for i in range(int(search_x_min / step), int(search_x_max / step))]:
        for y in [i * step for i in range(int(search_y_min / step), int(search_y_max / step))]:
            min_x = x
            max_x = x + zone_size
            min_y = y
            max_y = y + zone_size

            free_pct, occupied_pct = check_zone_free(
                pixels, width, height, min_x, max_x, min_y, max_y
            )

            if free_pct == 100.0 and occupied_pct == 0:
                # Also check with margin
                margin_free, _ = check_zone_free(
                    pixels, width, height,
                    min_x - 0.3, max_x + 0.3, min_y - 0.3, max_y + 0.3
                )
                if margin_free > 95:  # At least 95% free with margin
                    free_zones.append({
                        "min_x": min_x,
                        "max_x": max_x,
                        "min_y": min_y,
                        "max_y": max_y,
                        "center_x": (min_x + max_x) / 2,
                        "center_y": (min_y + max_y) / 2,
                        "margin_free": margin_free
                    })

    return free_zones


def main():
    print("=" * 70)
    print("Finding Free Zones for Experiments")
    print("=" * 70)

    pixels, width, height = read_pgm()
    print(f"Map: {width}x{height} pixels, resolution={RESOLUTION}m/px")
    print(f"Origin: ({ORIGIN_X}, {ORIGIN_Y})")

    # Current zone definitions
    current_zones = {
        "zone_A": {"min_x": 0.5, "max_x": 1.5, "min_y": 1.5, "max_y": 2.5},
        "zone_B": {"min_x": -2.0, "max_x": -1.0, "min_y": 0.5, "max_y": 1.5},
        "zone_C": {"min_x": 0.5, "max_x": 1.5, "min_y": -1.5, "max_y": -0.5},
        "zone_D": {"min_x": 1.7, "max_x": 2.3, "min_y": 0.0, "max_y": 0.8},
    }

    print("\n" + "-" * 70)
    print("Current Zone Analysis (static map only, ignoring inflation):")
    print("-" * 70)

    for name, zone in current_zones.items():
        free_pct, occupied_pct = check_zone_free(
            pixels, width, height,
            zone["min_x"], zone["max_x"], zone["min_y"], zone["max_y"]
        )
        status = "OK" if free_pct == 100 else "HAS OBSTACLES"
        print(f"{name}: free={free_pct:.1f}%, occupied={occupied_pct:.1f}% - {status}")

    print("\n" + "-" * 70)
    print("Finding Completely Free 1m x 1m Zones:")
    print("-" * 70)

    free_zones = find_free_zones(pixels, width, height, zone_size=1.0, step=0.5)

    # Filter to get well-distributed zones
    # Group by quadrant
    quadrants = {
        "Q1 (x>0, y>0)": [],
        "Q2 (x<0, y>0)": [],
        "Q3 (x<0, y<0)": [],
        "Q4 (x>0, y<0)": []
    }

    for z in free_zones:
        cx, cy = z["center_x"], z["center_y"]
        if cx >= 0 and cy >= 0:
            quadrants["Q1 (x>0, y>0)"].append(z)
        elif cx < 0 and cy >= 0:
            quadrants["Q2 (x<0, y>0)"].append(z)
        elif cx < 0 and cy < 0:
            quadrants["Q3 (x<0, y<0)"].append(z)
        else:
            quadrants["Q4 (x>0, y<0)"].append(z)

    print(f"\nFound {len(free_zones)} completely free zones")
    for qname, zones in quadrants.items():
        print(f"  {qname}: {len(zones)} zones")

    # Select best zones from each quadrant (closest to origin but not too close)
    selected_zones = []
    for qname, zones in quadrants.items():
        if zones:
            # Sort by distance from origin (prefer closer ones but not at origin)
            zones_sorted = sorted(
                zones,
                key=lambda z: abs(z["center_x"]) + abs(z["center_y"])
            )
            # Filter out zones too close to origin (robot starting point)
            zones_filtered = [z for z in zones_sorted
                           if abs(z["center_x"]) > 0.5 or abs(z["center_y"]) > 0.5]
            if zones_filtered:
                selected_zones.append((qname, zones_filtered[0]))

    print("\n" + "-" * 70)
    print("RECOMMENDED NEW ZONE DEFINITIONS:")
    print("-" * 70)

    zone_names = ["zone_A", "zone_B", "zone_C", "zone_D"]
    i = 0

    print("\n# Copy this to geofence.yaml:")
    print("zones:")

    for qname, z in selected_zones[:4]:
        name = zone_names[i]
        print(f"""  - name: {name}
    type: forbidden
    description: "Test zone in {qname}"
    vertices:
      - [{z['min_x']:.1f}, {z['min_y']:.1f}]
      - [{z['max_x']:.1f}, {z['min_y']:.1f}]
      - [{z['max_x']:.1f}, {z['max_y']:.1f}]
      - [{z['min_x']:.1f}, {z['max_y']:.1f}]
""")
        i += 1

    print("\n" + "-" * 70)
    print("SIMPLIFIED ZONES (away from walls):")
    print("-" * 70)

    # Manually define safe zones based on map analysis
    # These are in the central open area of the home environment
    safe_zones = [
        {"name": "zone_A", "min_x": 2.0, "max_x": 3.0, "min_y": 2.5, "max_y": 3.5, "desc": "Upper right"},
        {"name": "zone_B", "min_x": 2.0, "max_x": 3.0, "min_y": -2.5, "max_y": -1.5, "desc": "Lower right"},
        {"name": "zone_C", "min_x": -5.0, "max_x": -4.0, "min_y": 2.0, "max_y": 3.0, "desc": "Upper left"},
        {"name": "zone_D", "min_x": 3.5, "max_x": 4.5, "min_y": 0.0, "max_y": 1.0, "desc": "Right center"},
    ]

    for z in safe_zones:
        free_pct, occupied_pct = check_zone_free(
            pixels, width, height,
            z["min_x"], z["max_x"], z["min_y"], z["max_y"]
        )
        status = "OK" if free_pct == 100 else f"WARNING: {100-free_pct:.1f}% not free"
        print(f"{z['name']} ({z['desc']}): x=[{z['min_x']}, {z['max_x']}], y=[{z['min_y']}, {z['max_y']}] - {status}")


if __name__ == "__main__":
    main()
