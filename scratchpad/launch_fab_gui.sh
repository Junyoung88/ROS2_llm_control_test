#!/bin/bash
export DISPLAY=:1
export GZ_PARTITION=fabgui_view
export QT_QPA_PLATFORM=xcb
source /opt/ros/jazzy/setup.bash
exec gz sim "/home/jim/ros2_motion_planning_tutorials/src/mobile_manipulator_tutorial/src/mobile_manip_moveit_config/worlds/fab_cell.sdf" --force-version 8 -v1
