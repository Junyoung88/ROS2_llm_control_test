import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, TextSubstitution


def generate_launch_description():

    world_arg = DeclareLaunchArgument(
        'world', default_value='world.sdf',
        description='Name of the Gazebo world file to load'
    )

    headless_arg = DeclareLaunchArgument(
        'headless', default_value='false',
        description='Run Gazebo in headless mode (no GUI, -s flag)'
    )

    pkg_navigation_stack = get_package_share_directory('mobile_manip_moveit_config')
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')

    # Add gazebo models path from package
    gazebo_models_path = os.path.join(pkg_navigation_stack, 'gazebo_models')
    os.environ["GZ_SIM_RESOURCE_PATH"] += os.pathsep + gazebo_models_path

    # Gazebo with GUI
    gazebo_launch_gui = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py'),
        ),
        launch_arguments={'gz_args': [PathJoinSubstitution([
            pkg_navigation_stack,
            'worlds',
            LaunchConfiguration('world')
        ]),
        TextSubstitution(text=' -r -v -v1 --render-engine ogre --render-engine-gui-api-backend opengl')],
        'on_exit_shutdown': 'true'}.items(),
        condition=UnlessCondition(LaunchConfiguration('headless'))
    )

    # Gazebo headless (server only, no GUI)
    gazebo_launch_headless = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py'),
        ),
        launch_arguments={'gz_args': [PathJoinSubstitution([
            pkg_navigation_stack,
            'worlds',
            LaunchConfiguration('world')
        ]),
        TextSubstitution(text=' -r -s -v1 --headless-rendering')],
        'on_exit_shutdown': 'true'}.items(),
        condition=IfCondition(LaunchConfiguration('headless'))
    )

    launchDescriptionObject = LaunchDescription()

    launchDescriptionObject.add_action(world_arg)
    launchDescriptionObject.add_action(headless_arg)
    launchDescriptionObject.add_action(gazebo_launch_gui)
    launchDescriptionObject.add_action(gazebo_launch_headless)

    return launchDescriptionObject