#!/usr/bin/env python3
"""
Costmap Mask Generation Test

Tests the costmap mask generation logic without ROS2 dependencies.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from geofence_enforcer.geofence_geometry import GeofenceGeometry


def test_mask_generation():
    """Test that forbidden zones are correctly converted to costmap mask."""
    print("=" * 60)
    print(" Costmap Mask Generation Test ".center(60))
    print("=" * 60)

    # 1. Setup geofence
    print("\n[1] Setting up geofence...")
    geofence = GeofenceGeometry()
    geofence.add_zone(
        "workcell",
        [(4.5, 4.5), (6.5, 4.5), (6.5, 6.5), (4.5, 6.5)],
        metadata={'type': 'forbidden'}
    )
    print(f"    Zone: workcell")
    print(f"    Vertices: (4.5, 4.5) to (6.5, 6.5)")

    # 2. Generate mask grid
    print("\n[2] Generating costmap mask...")
    resolution = 0.1  # 10cm resolution for faster test
    map_origin_x = 0.0
    map_origin_y = 0.0
    map_width = 10.0
    map_height = 10.0

    width_cells = int(map_width / resolution)
    height_cells = int(map_height / resolution)

    print(f"    Grid size: {width_cells}x{height_cells} cells")
    print(f"    Resolution: {resolution}m")

    # Create grid
    from shapely.geometry import Point, Polygon

    grid = np.zeros((height_cells, width_cells), dtype=np.int8)

    # Get forbidden zone vertices
    zone = geofence._zones['workcell']
    polygon = Polygon(zone.vertices)

    # Mark cells
    marked_count = 0
    for row in range(height_cells):
        for col in range(width_cells):
            x = map_origin_x + (col + 0.5) * resolution
            y = map_origin_y + (row + 0.5) * resolution
            point = Point(x, y)

            if polygon.contains(point) or polygon.touches(point):
                grid[row, col] = 100  # Lethal obstacle
                marked_count += 1

    print(f"    Marked cells: {marked_count}")
    print(f"    Expected area: 2x2m = 4m² → ~{int(4 / (resolution**2))} cells")

    # 3. Verify mask
    print("\n[3] Verifying mask...")

    # Check center of forbidden zone (should be 100)
    center_col = int((5.5 - map_origin_x) / resolution)
    center_row = int((5.5 - map_origin_y) / resolution)
    center_value = grid[center_row, center_col]
    print(f"    Center (5.5, 5.5) value: {center_value}")
    assert center_value == 100, f"Center should be 100, got {center_value}"

    # Check outside forbidden zone (should be 0)
    outside_col = int((2.0 - map_origin_x) / resolution)
    outside_row = int((2.0 - map_origin_y) / resolution)
    outside_value = grid[outside_row, outside_col]
    print(f"    Outside (2.0, 2.0) value: {outside_value}")
    assert outside_value == 0, f"Outside should be 0, got {outside_value}"

    # Check edge of forbidden zone
    edge_col = int((4.5 - map_origin_x) / resolution)
    edge_row = int((4.5 - map_origin_y) / resolution)
    edge_value = grid[edge_row, edge_col]
    print(f"    Edge (4.5, 4.5) value: {edge_value}")

    # 4. Visualize (ASCII art)
    print("\n[4] Mask visualization (every 10th cell):")
    print("    " + "-" * 22)
    for row in range(height_cells - 1, -1, -10):
        line = "    |"
        for col in range(0, width_cells, 10):
            if grid[row, col] == 100:
                line += "##"
            else:
                line += "  "
        line += "|"
        print(line)
    print("    " + "-" * 22)
    print("    (## = forbidden zone)")

    # 5. Test result
    print("\n[5] Test Result:")
    expected_cells = int(4 / (resolution ** 2))  # 2x2m area
    tolerance = expected_cells * 0.1  # 10% tolerance

    if abs(marked_count - expected_cells) <= tolerance:
        print(f"    ✅ PASS: Marked {marked_count} cells (expected ~{expected_cells})")
        return True
    else:
        print(f"    ❌ FAIL: Marked {marked_count} cells (expected ~{expected_cells})")
        return False


def test_yaml_loading():
    """Test loading geofence from YAML file."""
    print("\n" + "=" * 60)
    print(" YAML Loading Test ".center(60))
    print("=" * 60)

    config_path = os.path.join(
        os.path.dirname(__file__),
        '..', 'config', 'workcell_geofence.yaml'
    )

    if not os.path.exists(config_path):
        print(f"    ⚠️  Config file not found: {config_path}")
        return True  # Skip test

    print(f"\n[1] Loading: {config_path}")
    try:
        geofence = GeofenceGeometry.from_yaml(config_path)
        zones = geofence.get_zone_names()
        print(f"    Loaded {len(zones)} zones: {zones}")

        forbidden = geofence.get_forbidden_zone_names()
        print(f"    Forbidden zones: {forbidden}")

        buffer_zones = geofence.get_buffer_zone_names()
        print(f"    Buffer zones: {buffer_zones}")

        print(f"\n[2] Test Result:")
        if 'workcell' in zones:
            print(f"    ✅ PASS: workcell zone loaded successfully")
            return True
        else:
            print(f"    ❌ FAIL: workcell zone not found")
            return False

    except Exception as e:
        print(f"    ❌ FAIL: {e}")
        return False


if __name__ == "__main__":
    results = []

    results.append(("Mask Generation", test_mask_generation()))
    results.append(("YAML Loading", test_yaml_loading()))

    print("\n" + "=" * 60)
    print(" Test Summary ".center(60))
    print("=" * 60)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name}: {status}")

    all_passed = all(r[1] for r in results)
    print(f"\nOverall: {'✅ All tests passed' if all_passed else '❌ Some tests failed'}")

    sys.exit(0 if all_passed else 1)
