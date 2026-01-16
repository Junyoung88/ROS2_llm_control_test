#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attack Test Launch File

Launches attack nodes for testing geofence enforcement.
Run this AFTER launching the demo.launch.py with PEP nodes.
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Package paths
    pkg_share = FindPackageShare('geofence_policy_enforcer')

    # Launch arguments
    attack_type_arg = DeclareLaunchArgument(
        'attack_type',
        default_value='stepwise',
        description='Attack type: stepwise or cmd_vel'
    )

    target_x_arg = DeclareLaunchArgument(
        'target_x',
        default_value='3.0',
        description='Target X coordinate (forbidden zone)'
    )

    target_y_arg = DeclareLaunchArgument(
        'target_y',
        default_value='3.0',
        description='Target Y coordinate (forbidden zone)'
    )

    use_safe_arg = DeclareLaunchArgument(
        'use_safe',
        default_value='true',
        description='Use safe (guarded) action/topic'
    )

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time'
    )

    # Get launch configurations
    target_x = LaunchConfiguration('target_x')
    target_y = LaunchConfiguration('target_y')
    use_safe = LaunchConfiguration('use_safe')
    use_sim_time = LaunchConfiguration('use_sim_time')

    # Stepwise Goals Attack Node
    stepwise_attack_node = Node(
        package='geofence_policy_enforcer',
        executable='attack_stepwise_goals',
        name='attack_stepwise_goals',
        parameters=[{
            'use_sim_time': use_sim_time,
            'target_x': target_x,
            'target_y': target_y,
            'step_size': 0.5,
            'num_steps': 20,
            'use_safe_action': use_safe,
            'delay_between_steps': 1.0,
        }],
        output='screen',
    )

    # Cmd Vel Attack Node
    cmd_vel_attack_node = Node(
        package='geofence_policy_enforcer',
        executable='attack_cmd_vel',
        name='attack_cmd_vel',
        parameters=[{
            'use_sim_time': use_sim_time,
            'target_x': target_x,
            'target_y': target_y,
            'attack_velocity': 0.3,
            'attack_duration': 30.0,
            'publish_rate': 10.0,
            'use_guarded_topic': use_safe,
        }],
        output='screen',
    )

    return LaunchDescription([
        # Launch arguments
        attack_type_arg,
        target_x_arg,
        target_y_arg,
        use_safe_arg,
        use_sim_time_arg,

        # Uncomment the attack you want to run:
        stepwise_attack_node,
        # cmd_vel_attack_node,
    ])
