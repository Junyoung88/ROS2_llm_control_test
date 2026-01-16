#!/usr/bin/env python3
"""
Gazebo Integration Test Launch File

이 launch 파일은 다음을 실행합니다:
1. Gazebo + TurtleBot3 시뮬레이션 (nav2_bringup 사용)
2. Nav2 Navigation Stack
3. Geofence Integration Test Node

사용법:
    ros2 launch geofence_enforcer gazebo_integration_test.launch.py
    ros2 launch geofence_enforcer gazebo_integration_test.launch.py trials:=20
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # 패키지 경로
    pkg_geofence = get_package_share_directory('geofence_enforcer')
    pkg_nav2_bringup = get_package_share_directory('nav2_bringup')

    # Launch arguments
    trials_arg = DeclareLaunchArgument(
        'trials',
        default_value='5',
        description='Number of trials per configuration'
    )

    output_dir_arg = DeclareLaunchArgument(
        'output_dir',
        default_value='/tmp/geofence_gazebo_test',
        description='Output directory for results'
    )

    headless_arg = DeclareLaunchArgument(
        'headless',
        default_value='False',
        description='Run Gazebo headless'
    )

    # Nav2 TurtleBot3 Simulation (Gazebo + Nav2 + AMCL 포함)
    tb3_simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_nav2_bringup, 'launch', 'tb3_simulation_launch.py')
        ),
        launch_arguments={
            'headless': LaunchConfiguration('headless'),
        }.items()
    )

    # Geofence Integration Test Node (Nav2 초기화 후 시작)
    test_node = TimerAction(
        period=20.0,  # Nav2 + Gazebo 초기화 대기
        actions=[
            Node(
                package='geofence_enforcer',
                executable='gazebo_integration_test',
                name='gazebo_integration_test',
                output='screen',
                parameters=[{
                    'use_sim_time': True,
                    'geofence_config': os.path.join(pkg_geofence, 'config', 'geofence_gazebo_test.yaml'),
                    'output_dir': LaunchConfiguration('output_dir'),
                    'trials_per_config': LaunchConfiguration('trials'),
                    'alpha': 0.99,
                }],
            )
        ]
    )

    return LaunchDescription([
        # Arguments
        trials_arg,
        output_dir_arg,
        headless_arg,

        # TurtleBot3 Simulation (Gazebo + Nav2)
        tb3_simulation,

        # Test Node (지연 시작)
        test_node,
    ])
