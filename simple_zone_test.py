#!/usr/bin/env python3
"""
Simple zone violation test using the experiment framework.
Tests that no_guard allows robot to enter forbidden zones.
"""
import sys
import time
sys.path.insert(0, 'src/geofence_enforcer/experiments')

from run_gazebo_s1_s6 import SimulationManager, PositionMonitor, GoalSender, ZONES, ProcessManager

def main():
    print("=" * 60)
    print("Simple Zone Violation Test")
    print("=" * 60)

    # Show zone_C bounds
    zone_c = ZONES.get('zone_C')
    if zone_c:
        print(f"\nZone C: x=[{zone_c['x_min']}, {zone_c['x_max']}], y=[{zone_c['y_min']}, {zone_c['y_max']}]")
    else:
        print("\nWARNING: zone_C not found!")

    # Use a goal that's INSIDE zone_C to ensure violation
    # Zone C is x=[-4.5, -3.5], y=[-3, 3], center is (-4, 0)
    goal_x, goal_y = -4.0, 0.0  # Inside zone_C!
    print(f"Goal: ({goal_x}, {goal_y}) - INSIDE zone C!\n")

    sim = SimulationManager()

    # Start simulation with no_guard method
    print("[1] Starting simulation with no_guard method...")
    success = sim.restart_with_method("no_guard")
    if not success:
        print("ERROR: Failed to start simulation")
        return 1
    print("    Simulation ready!")

    # Reset robot to origin
    print("\n[2] Resetting robot to origin...")
    sim.reset_robot_pose(0.0, 0.0, 0.0)
    time.sleep(2)

    # Start position monitor
    print("\n[3] Starting position monitor...")
    # Convert ZONES dict to list format for PositionMonitor
    zones_list = []
    for name, z in ZONES.items():
        zones_list.append({
            'name': name,
            'vertices': [
                {'x': z['x_min'], 'y': z['y_min']},
                {'x': z['x_max'], 'y': z['y_min']},
                {'x': z['x_max'], 'y': z['y_max']},
                {'x': z['x_min'], 'y': z['y_max']}
            ]
        })

    monitor = PositionMonitor(zones_list)
    monitor.start()
    time.sleep(1)

    # Send navigation goal
    print(f"\n[4] Sending navigation goal to ({goal_x}, {goal_y})...")

    decision, reason = GoalSender.send_goal(goal_x, goal_y, timeout=90, safety_method="no_guard")

    # Stop monitor and get results
    time.sleep(1)
    violations = monitor.stop()

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"Navigation: {decision} - {reason}")
    print(f"\nViolation data:")
    print(f"  Total samples: {violations.get('total_samples', 0)}")
    print(f"  Violation count: {violations.get('violation_count', 0)}")
    print(f"  Violation duration: {violations.get('violation_duration_s', 0):.2f}s")
    print(f"  Violated zones: {violations.get('violated_zones', [])}")
    print(f"  Min distance to zones: {violations.get('path_min_distance', 'N/A')}")

    # Check if zone_C was violated
    violated = violations.get('violated_zones', [])
    if 'zone_C' in violated:
        print("\n  SUCCESS: Robot entered zone_C (expected for no_guard)")
        return 0
    elif violations.get('violation_count', 0) > 0:
        print(f"\n  Robot entered zones: {violated} (but not zone_C?)")
        return 1
    else:
        print("\n  FAILURE: Robot did NOT enter any forbidden zone")
        print("  -> This is unexpected - goal was INSIDE zone_C!")
        return 1

if __name__ == "__main__":
    try:
        ret = main()
        # Cleanup
        print("\n[5] Cleaning up...")
        ProcessManager.cleanup_all()
        sys.exit(ret)
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
