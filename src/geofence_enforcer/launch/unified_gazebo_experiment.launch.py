#!/usr/bin/env python3
"""
Unified S1-S7 Gazebo Experiment Launch File

실제 Gazebo + Nav2 환경에서 S1-S7 공격 시나리오를 테스트합니다.

Usage:
    ros2 launch geofence_enforcer unified_gazebo_experiment.launch.py
    ros2 launch geofence_enforcer unified_gazebo_experiment.launch.py seeds:=50
    ros2 launch geofence_enforcer unified_gazebo_experiment.launch.py headless:=True seeds:=100

Arguments:
    seeds: Number of random seeds per scenario/method (default: 10)
    headless: Run Gazebo without GUI (default: False)
    output_dir: Directory for results (default: /tmp/unified_gazebo_results)
    scenarios: Comma-separated scenario list (default: S1,S2,S3,S4,S5,S6,S7)
    methods: Comma-separated method list (default: no_guard,safetychip,selp,geofence)
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
    LogInfo,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    # Package paths
    pkg_geofence = get_package_share_directory('geofence_enforcer')
    pkg_nav2_bringup = get_package_share_directory('nav2_bringup')

    # Launch arguments
    seeds_arg = DeclareLaunchArgument(
        'seeds',
        default_value='10',
        description='Number of random seeds per scenario/method combination'
    )

    headless_arg = DeclareLaunchArgument(
        'headless',
        default_value='False',
        description='Run Gazebo without GUI for faster execution'
    )

    output_dir_arg = DeclareLaunchArgument(
        'output_dir',
        default_value='/tmp/unified_gazebo_results',
        description='Output directory for experiment results'
    )

    scenarios_arg = DeclareLaunchArgument(
        'scenarios',
        default_value="['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7']",
        description='List of scenarios to run'
    )

    methods_arg = DeclareLaunchArgument(
        'methods',
        default_value="['no_guard', 'safetychip', 'selp', 'geofence']",
        description='List of methods to test'
    )

    # Ablation study arguments
    ablation_arg = DeclareLaunchArgument(
        'ablation_condition',
        default_value='full',
        description='Ablation condition: full, no_estimation, no_tracking, no_latency'
    )

    use_dynamic_margin_arg = DeclareLaunchArgument(
        'use_dynamic_margin',
        default_value='True',
        description='Use dynamic margin from policy enforcer node'
    )

    # Individual ablation term enable/disable arguments
    enable_estimation_arg = DeclareLaunchArgument(
        'enable_estimation_term',
        default_value='True',
        description='Enable estimation uncertainty term in margin calculation'
    )

    enable_tracking_arg = DeclareLaunchArgument(
        'enable_tracking_term',
        default_value='True',
        description='Enable tracking error term in margin calculation'
    )

    enable_latency_arg = DeclareLaunchArgument(
        'enable_latency_term',
        default_value='True',
        description='Enable latency compensation term in margin calculation'
    )

    # Log experiment info
    log_info = LogInfo(msg=[
        '\n',
        '=' * 70, '\n',
        'UNIFIED S1-S7 GAZEBO EXPERIMENT\n',
        '=' * 70, '\n',
        'Seeds: ', LaunchConfiguration('seeds'), '\n',
        'Headless: ', LaunchConfiguration('headless'), '\n',
        'Output: ', LaunchConfiguration('output_dir'), '\n',
        'Ablation: ', LaunchConfiguration('ablation_condition'), '\n',
        'Dynamic Margin: ', LaunchConfiguration('use_dynamic_margin'), '\n',
        '=' * 70, '\n',
    ])

    # Nav2 TurtleBot3 Simulation (includes Gazebo + Nav2 + AMCL)
    # AMCL 모드 사용 - 사전 제작된 맵으로 정확한 위치 추정 가능
    tb3_simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_nav2_bringup, 'launch', 'tb3_simulation_launch.py')
        ),
        launch_arguments={
            'headless': LaunchConfiguration('headless'),
            'slam': 'False',  # AMCL 모드 - 텔레포트 후 재위치 추정 지원
        }.items()
    )

    # Convert ablation_condition to enable_* parameters
    # full: all enabled
    # no_estimation: disable estimation term
    # no_tracking: disable tracking term
    # no_latency: disable latency term

    # Dynamic Policy Enforcer Node (delayed start for Nav2 initialization)
    geofence_config_path = os.path.join(pkg_geofence, 'config', 'factory_s1_s7_zones.yaml')

    dynamic_enforcer_node = TimerAction(
        period=30.0,  # Start before experiment node
        actions=[
            Node(
                package='geofence_enforcer',
                executable='dynamic_policy_enforcer',
                name='dynamic_policy_enforcer',
                output='screen',
                parameters=[{
                    'use_sim_time': True,
                    'alpha': 0.95,
                    'v_max': 0.22,  # TurtleBot3 Waffle max speed
                    'use_dynamic_v_max': True,
                    'default_e_track': 0.05,
                    'default_tau': 0.1,
                    # Ablation parameters from launch arguments
                    'enable_estimation_term': LaunchConfiguration('enable_estimation_term'),
                    'enable_tracking_term': LaunchConfiguration('enable_tracking_term'),
                    'enable_latency_term': LaunchConfiguration('enable_latency_term'),
                    'geofence_config': geofence_config_path,
                }],
            )
        ]
    )

    # Unified Gazebo Experiment Node (delayed start for Nav2 initialization)
    experiment_node = TimerAction(
        period=35.0,  # Wait for Gazebo + Nav2 + SLAM initialization
        actions=[
            Node(
                package='geofence_enforcer',
                executable='unified_gazebo_experiment',
                name='unified_gazebo_experiment',
                output='screen',
                parameters=[{
                    'use_sim_time': True,
                    'config_file': geofence_config_path,
                    'output_dir': LaunchConfiguration('output_dir'),
                    'seeds': LaunchConfiguration('seeds'),
                    'scenarios': LaunchConfiguration('scenarios'),
                    'methods': LaunchConfiguration('methods'),
                    'use_dynamic_margin_node': LaunchConfiguration('use_dynamic_margin'),
                    'ablation_condition': LaunchConfiguration('ablation_condition'),
                }],
            )
        ]
    )

    return LaunchDescription([
        # Arguments
        seeds_arg,
        headless_arg,
        output_dir_arg,
        scenarios_arg,
        methods_arg,
        ablation_arg,
        use_dynamic_margin_arg,
        enable_estimation_arg,
        enable_tracking_arg,
        enable_latency_arg,

        # Info
        log_info,

        # TurtleBot3 Simulation (Gazebo + Nav2)
        tb3_simulation,

        # Dynamic Policy Enforcer Node (delayed start)
        dynamic_enforcer_node,

        # Experiment Node (delayed start)
        experiment_node,
    ])
