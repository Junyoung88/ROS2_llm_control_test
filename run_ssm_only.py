#!/usr/bin/env python3
"""SSM-only experiment runner"""
import subprocess
import sys
import os

os.chdir('/home/jim/ros2_motion_planning_tutorials')

# Run experiment with only SSM method
cmd = [
    'python3', '-u',
    'src/geofence_enforcer/experiments/run_gazebo_s1_s6.py',
    '--quick', '--seeds', '1',
    '--methods', 'ssm'
]

print("Starting SSM-only experiment...")
subprocess.run(cmd)
