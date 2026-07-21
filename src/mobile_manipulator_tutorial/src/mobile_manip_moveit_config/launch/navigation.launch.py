import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, Command
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    pkg_navigation_stack = get_package_share_directory('mobile_manip_moveit_config')

    # gazebo_models_path, ignore_last_dir = os.path.split(pkg_navigation_stack)
    # os.environ["GZ_SIM_RESOURCE_PATH"] += os.pathsep + gazebo_models_path
    os.environ["GZ_SIM_RESOURCE_PATH"] += os.pathsep + pkg_navigation_stack + "/gazebo_models/"

    rviz_launch_arg = DeclareLaunchArgument(
        'rviz', default_value='true',
        description='Open RViz'
    )

    rviz_config_arg = DeclareLaunchArgument(
        'rviz_config', default_value='navigation.rviz',
        description='RViz config file'
    )

    sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='True',
        description='Flag to enable use_sim_time'
    )

    # AMCL localization enable/disable (for dead reckoning experiments)
    use_amcl_arg = DeclareLaunchArgument(
        'use_amcl', default_value='true',
        description='Enable AMCL localization. Set to false for dead reckoning only.'
    )

    # Generate path to config file
    interactive_marker_config_file_path = os.path.join(
        get_package_share_directory('interactive_marker_twist_server'),
        'config',
        'linear.yaml'
    )

    # Path to the Slam Toolbox launch file
    nav2_localization_launch_path = os.path.join(
        get_package_share_directory('nav2_bringup'),
        'launch',
        'localization_launch.py'
    )

    nav2_navigation_launch_path = os.path.join(
        get_package_share_directory('nav2_bringup'),
        'launch',
        'navigation_launch.py'
    )

    localization_params_path = os.path.join(
        get_package_share_directory('mobile_manip_moveit_config'),
        'config',
        'amcl_localization.yaml'
    )

    navigation_params_path = os.path.join(
        get_package_share_directory('mobile_manip_moveit_config'),
        'config',
        'navigation.yaml'
    )

    # Default map is the blank empty_map (empty-world experiments); warehouse
    # experiments pass map:=<...>/warehouse_map.yaml so AMCL has a real occupancy
    # grid to localize against.
    default_map_path = os.path.join(
        get_package_share_directory('mobile_manip_moveit_config'),
        'maps',
        'empty_map.yaml'
    )
    map_arg = DeclareLaunchArgument(
        'map', default_value=default_map_path,
        description='Path to the map yaml for AMCL/map_server.'
    )
    map_file_path = LaunchConfiguration('map')

    # Launch rviz
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', PathJoinSubstitution([pkg_navigation_stack, 'rviz', LaunchConfiguration('rviz_config')])],
        condition=IfCondition(LaunchConfiguration('rviz')),
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
        ]
    )

    localization_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(nav2_localization_launch_path),
        launch_arguments={
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'params_file': localization_params_path,
                'map': map_file_path,
        }.items(),
        condition=IfCondition(LaunchConfiguration('use_amcl'))
    )

    navigation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(nav2_navigation_launch_path),
        launch_arguments={
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'params_file': navigation_params_path,
        }.items()
    )

    # When AMCL is disabled (dead reckoning mode), we need:
    # 1. Static transform from map -> odom (identity, assuming start at origin)
    # 2. Map server to publish the map
    static_tf_map_odom = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_map_odom_tf',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        condition=UnlessCondition(LaunchConfiguration('use_amcl'))
    )

    # Map server for dead reckoning mode (since localization_launch includes map_server)
    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        parameters=[
            {'yaml_filename': map_file_path},
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ],
        condition=UnlessCondition(LaunchConfiguration('use_amcl'))
    )

    # Lifecycle manager for map_server when AMCL is disabled
    map_lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_map',
        parameters=[
            {'autostart': True},
            {'node_names': ['map_server']},
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ],
        condition=UnlessCondition(LaunchConfiguration('use_amcl'))
    )

    launchDescriptionObject = LaunchDescription()

    launchDescriptionObject.add_action(rviz_launch_arg)
    launchDescriptionObject.add_action(rviz_config_arg)
    launchDescriptionObject.add_action(sim_time_arg)
    launchDescriptionObject.add_action(use_amcl_arg)
    launchDescriptionObject.add_action(map_arg)
    launchDescriptionObject.add_action(rviz_node)
    #launchDescriptionObject.add_action(interactive_marker_twist_server_node)
    launchDescriptionObject.add_action(localization_launch)
    launchDescriptionObject.add_action(navigation_launch)
    # Dead reckoning support (when AMCL disabled)
    launchDescriptionObject.add_action(static_tf_map_odom)
    launchDescriptionObject.add_action(map_server_node)
    launchDescriptionObject.add_action(map_lifecycle_manager)

    return launchDescriptionObject
