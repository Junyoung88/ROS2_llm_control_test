# W2 nodock fix (whole-graph SROS2 Enforce, full trial)

Under SROS2 Enforce the `/cmd_vel` deny_rule (mux = sole writer) also denies nav2's
stock `docking_server`, which crashes at activation and aborts the navigation
lifecycle bring-up so bt_navigator never activates. Fix: drop `docking_server` from
the nav2 navigation lifecycle, ONLY under Enforce.

Runtime files (MAIN checkout, symlink-install so source edits are live):
  src/mobile_manipulator_tutorial/src/mobile_manip_moveit_config/launch/
    navigation_nodock_launch.py       (copy of nav2_bringup navigation_launch.py,
                                        'docking_server' removed from lifecycle_nodes)
    navigation.launch.py              (patched: uses nodock launch iff PETSE_SROS2 set;
                                        backup navigation.launch.py.preNODOCK)

Result (v11): docking-abort gone, bt_navigator activates, goal_gate approves
(nav_approved=True), Nav2 plans a path, guard passes commands to the mux, 0 buffer
errors, 0 access-control denials, /cmd_vel exclusivity preserved (deny_rule). Remaining:
the base does not yet physically move (final actuation hop under Enforce, still tracing).

Copies here are for the record; the .patch is the conditional edit applied to the main
navigation.launch.py.
