"""
Launch file for the Probabilistic Geofence Policy Enforcer.

This launch file starts the policy enforcer node with configurable
parameters for the safety margin computation.

Usage:
    ros2 launch geofence_enforcer policy_enforcer.launch.py

    With custom parameters:
    ros2 launch geofence_enforcer policy_enforcer.launch.py \\
        alpha:=0.99 \\
        e_track:=0.1 \\
        v_max:=1.0 \\
        tau:=0.15

Parameters:
    alpha: Confidence level for chi-square bound (default: 0.95)
    e_track: Tracking/execution error buffer in meters (default: 0.05)
    v_max: Maximum translational velocity in m/s (default: 0.5)
    tau: End-to-end system latency in seconds (default: 0.1)
    geofence_config: Path to YAML config file
    enable_geofence: Enable/disable enforcement (default: true)
    enable_projection: Enable goal projection (default: true)
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for the policy enforcer node."""

    # Get package share directory
    pkg_share = get_package_share_directory('geofence_enforcer')

    # Default config path
    default_config = os.path.join(pkg_share, 'config', 'geofence.yaml')

    # =========================================================================
    # Launch Arguments
    # =========================================================================

    # Margin calculation parameters
    alpha_arg = DeclareLaunchArgument(
        'alpha',
        default_value='0.95',
        description='Confidence level for chi-square uncertainty bound (0 < alpha < 1)'
    )

    e_track_arg = DeclareLaunchArgument(
        'e_track',
        default_value='0.05',
        description='Execution/tracking error buffer in meters'
    )

    v_max_arg = DeclareLaunchArgument(
        'v_max',
        default_value='0.5',
        description='Maximum translational velocity in m/s'
    )

    tau_arg = DeclareLaunchArgument(
        'tau',
        default_value='0.1',
        description='End-to-end system latency in seconds'
    )

    # Geofence behavior parameters
    geofence_config_arg = DeclareLaunchArgument(
        'geofence_config',
        default_value=default_config,
        description='Path to YAML file defining forbidden zones'
    )

    enable_geofence_arg = DeclareLaunchArgument(
        'enable_geofence',
        default_value='true',
        description='Enable or disable geofence enforcement'
    )

    enable_projection_arg = DeclareLaunchArgument(
        'enable_projection',
        default_value='true',
        description='Enable projection of unsafe goals to safe locations'
    )

    # Node behavior parameters
    check_rate_arg = DeclareLaunchArgument(
        'check_rate_hz',
        default_value='10.0',
        description='Rate for pose checking in Hz'
    )

    publish_viz_arg = DeclareLaunchArgument(
        'publish_visualization',
        default_value='true',
        description='Publish visualization markers'
    )

    # =========================================================================
    # Node Configuration
    # =========================================================================

    policy_enforcer_node = Node(
        package='geofence_enforcer',
        executable='policy_enforcer_node',
        name='policy_enforcer',
        output='screen',
        parameters=[{
            # Margin calculation parameters
            'alpha': LaunchConfiguration('alpha'),
            'e_track': LaunchConfiguration('e_track'),
            'v_max': LaunchConfiguration('v_max'),
            'tau': LaunchConfiguration('tau'),

            # Geofence behavior
            'enable_geofence': LaunchConfiguration('enable_geofence'),
            'enable_projection': LaunchConfiguration('enable_projection'),
            'geofence_config': LaunchConfiguration('geofence_config'),

            # Node behavior
            'check_rate_hz': LaunchConfiguration('check_rate_hz'),
            'publish_visualization': LaunchConfiguration('publish_visualization'),
            'safety_stop_topic': '/cmd_vel',
            'visualization_rate_hz': 1.0,
            'projection_safety_buffer': 0.01,
        }],
        remappings=[
            # Standard Nav2 remappings
            ('/amcl_pose', '/amcl_pose'),
            ('/odom', '/odom'),
            ('/cmd_vel', '/cmd_vel'),
            ('/goal_pose', '/goal_pose'),
        ],
    )

    # =========================================================================
    # Launch Description
    # =========================================================================

    return LaunchDescription([
        # Declare arguments
        alpha_arg,
        e_track_arg,
        v_max_arg,
        tau_arg,
        geofence_config_arg,
        enable_geofence_arg,
        enable_projection_arg,
        check_rate_arg,
        publish_viz_arg,

        # Launch node
        policy_enforcer_node,
    ])
