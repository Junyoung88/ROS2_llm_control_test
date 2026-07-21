#!/usr/bin/env python3
"""
Verify Zone Access Analysis

This script analyzes why the robot cannot enter forbidden zones
and provides recommendations for fixing the experiment setup.

Run: python3 verify_zone_access.py
"""

import math

# Zone definitions
ZONES = {
    "zone_A": {"min_x": 0.5, "max_x": 1.5, "min_y": 1.5, "max_y": 2.5},
    "zone_B": {"min_x": -2.0, "max_x": -1.0, "min_y": 0.5, "max_y": 1.5},
    "zone_C": {"min_x": 0.5, "max_x": 1.5, "min_y": -1.5, "max_y": -0.5},
    "zone_D": {"min_x": 1.7, "max_x": 2.3, "min_y": 0.0, "max_y": 0.8},
}

# Key obstacles from Gazebo world (home.sdf)
OBSTACLES = {
    "Wall_22": {"x": -0.014, "y_min": 0.88, "y_max": 6.13, "width": 0.15},  # Vertical wall at x≈0
    "Wall_20": {"x": 0.997, "y_min": -0.74, "y_max": 0.26, "width": 0.15},  # Vertical wall at x≈1
    "fire_hydrant": {"x": 0.45, "y": -1.66, "radius": 0.15},
}

# Current Nav2 parameters
INFLATION_RADIUS = 0.7
ROBOT_RADIUS = 0.22

def analyze_zone_access(zone_name, zone, inflation_radius):
    """Analyze if a zone is accessible given the inflation radius."""
    print(f"\n{zone_name}: x=[{zone['min_x']}, {zone['max_x']}], y=[{zone['min_y']}, {zone['max_y']}]")

    blocking_obstacles = []

    # Check Wall_22 (vertical wall at x≈0)
    wall = OBSTACLES["Wall_22"]
    inflated_x = wall["x"] + wall["width"]/2 + inflation_radius
    if zone["min_y"] <= wall["y_max"] and zone["max_y"] >= wall["y_min"]:
        if inflated_x >= zone["min_x"]:
            overlap = inflated_x - zone["min_x"]
            blocking_obstacles.append(f"Wall_22 (inflated to x={inflated_x:.2f}m, overlaps {overlap:.2f}m into zone)")

    # Check Wall_20 (vertical wall at x≈1)
    wall = OBSTACLES["Wall_20"]
    if zone["min_y"] <= wall["y_max"] and zone["max_y"] >= wall["y_min"]:
        inflated_x_min = wall["x"] - wall["width"]/2 - inflation_radius
        inflated_x_max = wall["x"] + wall["width"]/2 + inflation_radius
        if inflated_x_min <= zone["max_x"] and inflated_x_max >= zone["min_x"]:
            blocking_obstacles.append(f"Wall_20 (inflated x=[{inflated_x_min:.2f}, {inflated_x_max:.2f}])")

    # Check fire hydrant
    fh = OBSTACLES["fire_hydrant"]
    fh_dist_to_zone_x = max(zone["min_x"] - fh["x"], fh["x"] - zone["max_x"], 0)
    fh_dist_to_zone_y = max(zone["min_y"] - fh["y"], fh["y"] - zone["max_y"], 0)
    fh_dist = math.sqrt(fh_dist_to_zone_x**2 + fh_dist_to_zone_y**2)

    if fh_dist < inflation_radius + fh["radius"]:
        blocking_obstacles.append(f"fire_hydrant (at ({fh['x']:.2f}, {fh['y']:.2f}), dist to zone: {fh_dist:.2f}m)")

    if blocking_obstacles:
        print(f"  BLOCKED by:")
        for obs in blocking_obstacles:
            print(f"    - {obs}")
        return False
    else:
        print(f"  ACCESSIBLE")
        return True


def main():
    print("=" * 70)
    print("Zone Access Analysis")
    print("=" * 70)
    print(f"\nCurrent Nav2 Parameters:")
    print(f"  inflation_radius = {INFLATION_RADIUS}m")
    print(f"  robot_radius = {ROBOT_RADIUS}m")

    print("\n" + "-" * 70)
    print("Zone Accessibility with CURRENT inflation_radius:")
    print("-" * 70)

    accessible = {}
    for zone_name, zone in ZONES.items():
        accessible[zone_name] = analyze_zone_access(zone_name, zone, INFLATION_RADIUS)

    # Test with reduced inflation radius
    print("\n" + "-" * 70)
    print("Zone Accessibility with REDUCED inflation_radius = 0.25m:")
    print("-" * 70)

    accessible_reduced = {}
    for zone_name, zone in ZONES.items():
        accessible_reduced[zone_name] = analyze_zone_access(zone_name, zone, 0.25)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    current_accessible = sum(accessible.values())
    reduced_accessible = sum(accessible_reduced.values())

    print(f"\nWith current inflation_radius (0.7m): {current_accessible}/4 zones accessible")
    print(f"With reduced inflation_radius (0.25m): {reduced_accessible}/4 zones accessible")

    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    print("""
To enable zone entry testing, modify navigation.yaml:

1. Edit: src/mobile_manip_moveit_config/config/navigation.yaml
2. Change inflation_radius from 0.7 to 0.25 in BOTH:
   - local_costmap.local_costmap.ros__parameters.inflation_layer.inflation_radius
   - global_costmap.global_costmap.ros__parameters.inflation_layer.inflation_radius

3. Restart navigation stack

This will allow the robot to navigate closer to walls and enter the
forbidden zones when no_guard method is used.

Alternatively, use zones that are farther from walls:
  - Zone A center (1.0, 2.0) is 0.5m from Wall_22
  - Consider moving zones to x >= 1.5 to avoid Wall_22 inflation
""")

    print("\n" + "=" * 70)
    print("QUICK FIX COMMAND")
    print("=" * 70)
    print("""
# Reduce inflation radius to 0.25m
sed -i 's/inflation_radius: 0.7/inflation_radius: 0.25/g' \\
    /home/jim/ros2_motion_planning_tutorials/src/mobile_manipulator_tutorial/src/mobile_manip_moveit_config/config/navigation.yaml

# Then rebuild and restart navigation:
cd /home/jim/ros2_motion_planning_tutorials/src/mobile_manipulator_tutorial
colcon build --packages-select mobile_manip_moveit_config
source install/setup.bash
# Restart navigation launch
""")


if __name__ == "__main__":
    main()
