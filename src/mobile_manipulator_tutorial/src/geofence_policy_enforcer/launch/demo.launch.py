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

    # Cmd vel guard topic configuration (for hardware-level interception)
    cmd_vel_input_topic_arg = DeclareLaunchArgument(
        'cmd_vel_input_topic',
        default_value='/cmd_vel_raw',
        description='Input topic for cmd_vel_guard (use /cmd_vel for full interception)'
    )

    cmd_vel_output_topic_arg = DeclareLaunchArgument(
        'cmd_vel_output_topic',
        default_value='/cmd_vel',
        description='Output topic for cmd_vel_guard (use /cmd_vel_safe with hw_guard bridge)'
    )

    safety_method_arg = DeclareLaunchArgument(
        'safety_method',
        default_value='geofence',
        description='Safety method: no_guard, geofence, safetychip, selp'
    )

    # Ablation study arguments
    ablation_condition_arg = DeclareLaunchArgument(
        'ablation_condition',
        default_value='full',
        description='Ablation condition: full, no_estimation, no_tracking, no_latency'
    )

    enable_estimation_arg = DeclareLaunchArgument(
        'enable_estimation_term',
        default_value='true',
        description='Enable estimation uncertainty term in margin calculation'
    )

    enable_tracking_arg = DeclareLaunchArgument(
        'enable_tracking_term',
        default_value='true',
        description='Enable tracking error term in margin calculation'
    )

    enable_latency_arg = DeclareLaunchArgument(
        'enable_latency_term',
        default_value='true',
        description='Enable latency compensation term in margin calculation'
    )

    use_dynamic_v_max_arg = DeclareLaunchArgument(
        'use_dynamic_v_max',
        default_value='true',
        description='Use dynamic v_max estimation from observed velocities'
    )

    use_dynamic_tau_arg = DeclareLaunchArgument(
        'use_dynamic_tau',
        default_value='true',
        description='Use dynamic latency (tau) estimation from cmd_vel/odom timing'
    )

    use_dynamic_e_track_arg = DeclareLaunchArgument(
        'use_dynamic_e_track',
        default_value='true',
        description='Use dynamic tracking error estimation from path following'
    )

    # Uncertainty/margin parameters (can be overridden for geofence_no_margin)
    k_sigma_arg = DeclareLaunchArgument(
        'k_sigma',
        default_value='3.0',
        description='Confidence multiplier for localization uncertainty'
    )

    localization_sigma_arg = DeclareLaunchArgument(
        'localization_sigma',
        default_value='0.15',
        description='Localization uncertainty in meters'
    )

    tracking_error_arg = DeclareLaunchArgument(
        'tracking_error',
        default_value='0.05',
        description='Path tracking error in meters'
    )

    v_max_arg = DeclareLaunchArgument(
        'v_max',
        default_value='0.5',
        description='Maximum velocity in m/s'
    )

    latency_arg = DeclareLaunchArgument(
        'latency',
        default_value='0.1',
        description='System latency in seconds'
    )

    a_max_arg = DeclareLaunchArgument(
        'a_max',
        default_value='2.5',
        description='Maximum deceleration magnitude in m/s² (for v²/2a braking term)'
    )

    enable_braking_term_arg = DeclareLaunchArgument(
        'enable_braking_term',
        default_value='true',
        description='Enable braking distance term v²/(2*a_max) in margin calculation'
    )

    # RA-L epsilon formulation arguments
    use_epsilon_formulation_arg = DeclareLaunchArgument(
        'use_epsilon_formulation',
        default_value='true',
        description='Use ε-based margin formula (True) or legacy k_σ formula (False)'
    )

    epsilon_arg = DeclareLaunchArgument(
        'epsilon',
        default_value='0.003',
        description='Safety risk budget ε (0.003 ≈ 99.7%)'
    )

    e_0_arg = DeclareLaunchArgument(
        'e_0',
        default_value='0.03',
        description='Static tracking error e₀ (meters)'
    )

    c_1_arg = DeclareLaunchArgument(
        'c_1',
        default_value='0.04',
        description='Velocity-proportional tracking coefficient c₁'
    )

    c_2_arg = DeclareLaunchArgument(
        'c_2',
        default_value='0.0',
        description='Curvature-dependent tracking coefficient c₂'
    )

    # Runtime monitoring arguments (for CBF/SSM)
    enable_runtime_monitoring_arg = DeclareLaunchArgument(
        'enable_runtime_monitoring',
        default_value='false',
        description='Enable runtime safety monitoring for CBF/SSM methods'
    )

    runtime_monitoring_rate_arg = DeclareLaunchArgument(
        'runtime_monitoring_rate',
        default_value='10.0',
        description='Runtime monitoring rate in Hz'
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
    cmd_vel_input_topic = LaunchConfiguration('cmd_vel_input_topic')
    cmd_vel_output_topic = LaunchConfiguration('cmd_vel_output_topic')
    ablation_condition = LaunchConfiguration('ablation_condition')
    enable_estimation_term = LaunchConfiguration('enable_estimation_term')
    enable_tracking_term = LaunchConfiguration('enable_tracking_term')
    enable_latency_term = LaunchConfiguration('enable_latency_term')
    use_dynamic_v_max = LaunchConfiguration('use_dynamic_v_max')
    use_dynamic_tau = LaunchConfiguration('use_dynamic_tau')
    use_dynamic_e_track = LaunchConfiguration('use_dynamic_e_track')
    # Uncertainty parameters
    k_sigma = LaunchConfiguration('k_sigma')
    localization_sigma = LaunchConfiguration('localization_sigma')
    tracking_error = LaunchConfiguration('tracking_error')
    v_max = LaunchConfiguration('v_max')
    latency = LaunchConfiguration('latency')
    a_max = LaunchConfiguration('a_max')
    enable_braking_term = LaunchConfiguration('enable_braking_term')
    # RA-L epsilon formulation parameters
    use_epsilon_formulation = LaunchConfiguration('use_epsilon_formulation')
    epsilon = LaunchConfiguration('epsilon')
    e_0 = LaunchConfiguration('e_0')
    c_1 = LaunchConfiguration('c_1')
    c_2 = LaunchConfiguration('c_2')
    # Runtime monitoring parameters
    enable_runtime_monitoring = LaunchConfiguration('enable_runtime_monitoring')
    runtime_monitoring_rate = LaunchConfiguration('runtime_monitoring_rate')

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
                # Ablation study parameters
                'ablation_condition': ablation_condition,
                'enable_estimation_term': enable_estimation_term,
                'enable_tracking_term': enable_tracking_term,
                'enable_latency_term': enable_latency_term,
                # Dynamic measurement parameters
                'use_dynamic_v_max': use_dynamic_v_max,
                'use_dynamic_tau': use_dynamic_tau,
                'use_dynamic_e_track': use_dynamic_e_track,
                # Uncertainty/margin parameters
                'k_sigma': k_sigma,
                'localization_sigma': localization_sigma,
                'tracking_error': tracking_error,
                'v_max': v_max,
                'latency': latency,
                'a_max': a_max,
                'enable_braking_term': enable_braking_term,
                # RA-L epsilon formulation
                'use_epsilon_formulation': use_epsilon_formulation,
                'epsilon': epsilon,
                'e_0': e_0,
                'c_1': c_1,
                'c_2': c_2,
                # Runtime monitoring for CBF/SSM
                'enable_runtime_monitoring': enable_runtime_monitoring,
                'runtime_monitoring_rate': runtime_monitoring_rate,
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
                'input_topic': cmd_vel_input_topic,
                'output_topic': cmd_vel_output_topic,
                'safety_method': safety_method,
                'odom_topic': '/odom',
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
        cmd_vel_input_topic_arg,
        cmd_vel_output_topic_arg,
        # Ablation study arguments
        ablation_condition_arg,
        enable_estimation_arg,
        enable_tracking_arg,
        enable_latency_arg,
        # Dynamic measurement arguments
        use_dynamic_v_max_arg,
        use_dynamic_tau_arg,
        use_dynamic_e_track_arg,
        # Uncertainty/margin arguments
        k_sigma_arg,
        localization_sigma_arg,
        tracking_error_arg,
        v_max_arg,
        latency_arg,
        a_max_arg,
        enable_braking_term_arg,
        # RA-L epsilon formulation arguments
        use_epsilon_formulation_arg,
        epsilon_arg,
        e_0_arg,
        c_1_arg,
        c_2_arg,
        # Runtime monitoring arguments
        enable_runtime_monitoring_arg,
        runtime_monitoring_rate_arg,

        # Nodes
        goal_gate_node,
        cmd_vel_guard_node,
        path_watchdog_node,
        metrics_logger_node,
    ])
