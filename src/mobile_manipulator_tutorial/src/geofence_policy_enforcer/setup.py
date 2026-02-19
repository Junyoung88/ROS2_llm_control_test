from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'geofence_policy_enforcer'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Robot Security Team',
    maintainer_email='security@robot.local',
    description='Security-oriented geofence policy enforcement framework for Nav2',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'goal_gate_node = geofence_policy_enforcer.goal_gate_node:main',
            'cmd_vel_guard_node = geofence_policy_enforcer.cmd_vel_guard_node:main',
            'path_watchdog_node = geofence_policy_enforcer.path_watchdog_node:main',
            'metrics_logger_node = geofence_policy_enforcer.metrics_logger_node:main',
            'attack_stepwise_goals = geofence_policy_enforcer.attack_stepwise_goals:main',
            'attack_cmd_vel = geofence_policy_enforcer.attack_cmd_vel:main',
            'attack_velocity_amplify = geofence_policy_enforcer.attack_velocity_amplify:main',
            'attack_pose_spoofing = geofence_policy_enforcer.attack_pose_spoofing:main',
            'attack_direct_control = geofence_policy_enforcer.attack_direct_control:main',
            'attack_prompt_injection = geofence_policy_enforcer.attack_prompt_injection:main',
            'attack_velocity_scaling = geofence_policy_enforcer.attack_velocity_scaling:main',
            'attack_odom_spoofing = geofence_policy_enforcer.attack_odom_spoofing:main',
            'attack_scan_spoofing = geofence_policy_enforcer.attack_scan_spoofing:main',
            'scan_relay = geofence_policy_enforcer.scan_relay:main',
            'hardware_geofence_guard = geofence_policy_enforcer.hardware_geofence_guard:main',
        ],
    },
)
