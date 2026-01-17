"""
Setup script for the Probabilistic Geofence Policy Enforcer package.

This package provides research-grade geofence enforcement with dynamic
safety margins that adapt to localization uncertainty.
"""

from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'geofence_enforcer'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        # Package index
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        # Package manifest
        ('share/' + package_name, ['package.xml']),
        # Launch files
        ('share/' + package_name + '/launch',
            glob('launch/*.py') + glob('launch/*.xml') + glob('launch/*.yaml')),
        # Config files
        ('share/' + package_name + '/config',
            glob('config/*.yaml')),
    ],
    install_requires=[
        'setuptools',
        'numpy',
        'pyyaml',
        'shapely',
    ],
    zip_safe=True,
    maintainer='Research Team',
    maintainer_email='research@example.com',
    description='Probabilistic Geofence Policy Enforcer with dynamic safety margins',
    license='BSD-3-Clause',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'policy_enforcer_node = geofence_enforcer.policy_enforcer_node:main',
            'dynamic_policy_enforcer = geofence_enforcer.dynamic_policy_enforcer_node:main',
            'costmap_mask_publisher = geofence_enforcer.costmap_mask_publisher:main',
            'runtime_monitor = geofence_enforcer.runtime_monitor:main',
            'gazebo_integration_test = geofence_enforcer.gazebo_integration_test:main',
            'unified_gazebo_experiment = geofence_enforcer.unified_gazebo_experiment:main',
        ],
    },
    python_requires='>=3.8',
)
