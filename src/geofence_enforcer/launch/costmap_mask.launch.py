"""
Costmap Mask Publisher Launch File

Launches the costmap mask publisher node that publishes geofence zones
as a Nav2 KeepoutFilter mask.

Usage:
    ros2 launch geofence_enforcer costmap_mask.launch.py

    # With custom config:
    ros2 launch geofence_enforcer costmap_mask.launch.py \
        geofence_config:=/path/to/geofence.yaml

This node publishes:
    /keepout_filter_mask (nav_msgs/OccupancyGrid) - The costmap mask
    /costmap_filter_info (nav2_msgs/CostmapFilterInfo) - Filter metadata

Nav2 KeepoutFilter subscribes to /costmap_filter_info to get the mask topic,
then subscribes to /keepout_filter_mask to get the actual mask data.
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Package directories
    pkg_dir = get_package_share_directory('geofence_enforcer')

    # Default config path
    default_config = os.path.join(pkg_dir, 'config', 'workcell_geofence.yaml')

    # Launch arguments
    geofence_config_arg = DeclareLaunchArgument(
        'geofence_config',
        default_value=default_config,
        description='Path to geofence YAML configuration file'
    )

    resolution_arg = DeclareLaunchArgument(
        'resolution',
        default_value='0.05',
        description='Costmap resolution in meters'
    )

    map_width_arg = DeclareLaunchArgument(
        'map_width',
        default_value='20.0',
        description='Map width in meters'
    )

    map_height_arg = DeclareLaunchArgument(
        'map_height',
        default_value='20.0',
        description='Map height in meters'
    )

    map_origin_x_arg = DeclareLaunchArgument(
        'map_origin_x',
        default_value='-10.0',
        description='Map origin X coordinate'
    )

    map_origin_y_arg = DeclareLaunchArgument(
        'map_origin_y',
        default_value='-10.0',
        description='Map origin Y coordinate'
    )

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time'
    )

    # Costmap Mask Publisher Node
    costmap_mask_publisher_node = Node(
        package='geofence_enforcer',
        executable='costmap_mask_publisher',
        name='costmap_mask_publisher',
        output='screen',
        parameters=[{
            'geofence_config': LaunchConfiguration('geofence_config'),
            'resolution': LaunchConfiguration('resolution'),
            'map_width': LaunchConfiguration('map_width'),
            'map_height': LaunchConfiguration('map_height'),
            'map_origin_x': LaunchConfiguration('map_origin_x'),
            'map_origin_y': LaunchConfiguration('map_origin_y'),
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'publish_rate': 1.0,
            'inflation_radius': 0.1,
        }],
    )

    return LaunchDescription([
        geofence_config_arg,
        resolution_arg,
        map_width_arg,
        map_height_arg,
        map_origin_x_arg,
        map_origin_y_arg,
        use_sim_time_arg,
        costmap_mask_publisher_node,
    ])
