#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Geofence Policy Enforcer Demo Launch File

Launches all PEP nodes for geofence enforcement.
Can be used standalone (assuming Nav2 is already running) or
can include Nav2 bringup.
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Package paths
    pkg_share = FindPackageShare('geofence_policy_enforcer')

    # Launch arguments
    geofence_config_arg = DeclareLaunchArgument(
        'geofence_config',
        default_value=PathJoinSubstitution([pkg_share, 'config', 'geofence.yaml']),
        description='Path to geofence configuration file'
    )

    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=PathJoinSubstitution([pkg_share, 'config', 'geofence_params.yaml']),
        description='Path to node parameters file'
    )

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time'
    )

    enable_goal_gate_arg = DeclareLaunchArgument(
        'enable_goal_gate',
        default_value='true',
        description='Enable Goal Gate PEP'
    )

    enable_cmd_vel_guard_arg = DeclareLaunchArgument(
        'enable_cmd_vel_guard',
        default_value='true',
        description='Enable Cmd Vel Guard PEP'
    )

    enable_path_watchdog_arg = DeclareLaunchArgument(
        'enable_path_watchdog',
        default_value='true',
        description='Enable Path Watchdog PEP'
    )

    enable_metrics_arg = DeclareLaunchArgument(
        'enable_metrics',
        default_value='true',
        description='Enable Metrics Logger'
    )

    csv_output_arg = DeclareLaunchArgument(
        'csv_output',
        default_value='/tmp/geofence_metrics.csv',
        description='CSV output file path'
    )

    override_strategy_arg = DeclareLaunchArgument(
        'override_strategy',
        default_value='stop',
        description='Cmd vel override strategy: stop, scale, redirect'
    )

    safety_method_arg = DeclareLaunchArgument(
        'safety_method',
        default_value='geofence',
        description='Safety method: no_guard, geofence, safetychip, selp'
    )

    # Get launch configurations
    geofence_config = LaunchConfiguration('geofence_config')
    params_file = LaunchConfiguration('params_file')
    use_sim_time = LaunchConfiguration('use_sim_time')
    enable_goal_gate = LaunchConfiguration('enable_goal_gate')
    enable_cmd_vel_guard = LaunchConfiguration('enable_cmd_vel_guard')
    enable_path_watchdog = LaunchConfiguration('enable_path_watchdog')
    enable_metrics = LaunchConfiguration('enable_metrics')
    csv_output = LaunchConfiguration('csv_output')
    override_strategy = LaunchConfiguration('override_strategy')
    safety_method = LaunchConfiguration('safety_method')

    # Goal Gate Node
    goal_gate_node = Node(
        package='geofence_policy_enforcer',
        executable='goal_gate_node',
        name='goal_gate',
        parameters=[
            params_file,
            {
                'use_sim_time': use_sim_time,
                'geofence_config': geofence_config,
                'safety_method': safety_method,
            }
        ],
        output='screen',
        condition=IfCondition(enable_goal_gate),
    )

    # Cmd Vel Guard Node
    cmd_vel_guard_node = Node(
        package='geofence_policy_enforcer',
        executable='cmd_vel_guard_node',
        name='cmd_vel_guard',
        parameters=[
            params_file,
            {
                'use_sim_time': use_sim_time,
                'geofence_config': geofence_config,
                'override_strategy': override_strategy,
            }
        ],
        output='screen',
        condition=IfCondition(enable_cmd_vel_guard),
    )

    # Path Watchdog Node
    path_watchdog_node = Node(
        package='geofence_policy_enforcer',
        executable='path_watchdog_node',
        name='path_watchdog',
        parameters=[
            params_file,
            {
                'use_sim_time': use_sim_time,
                'geofence_config': geofence_config,
            }
        ],
        output='screen',
        condition=IfCondition(enable_path_watchdog),
    )

    # Metrics Logger Node
    metrics_logger_node = Node(
        package='geofence_policy_enforcer',
        executable='metrics_logger_node',
        name='metrics_logger',
        parameters=[
            params_file,
            {
                'use_sim_time': use_sim_time,
                'geofence_config': geofence_config,
                'csv_output_file': csv_output,
            }
        ],
        output='screen',
        condition=IfCondition(enable_metrics),
    )

    return LaunchDescription([
        # Launch arguments
        geofence_config_arg,
        params_file_arg,
        use_sim_time_arg,
        enable_goal_gate_arg,
        enable_cmd_vel_guard_arg,
        enable_path_watchdog_arg,
        enable_metrics_arg,
        csv_output_arg,
        override_strategy_arg,
        safety_method_arg,

        # Nodes
        goal_gate_node,
        cmd_vel_guard_node,
        path_watchdog_node,
        metrics_logger_node,
    ])
