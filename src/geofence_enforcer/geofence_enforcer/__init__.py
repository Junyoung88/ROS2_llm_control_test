# Probabilistic Geofence Policy Enforcer
# Research-grade implementation for ROS2
#
# This package implements a dynamic safety margin computation based on:
#   1) State-estimation uncertainty (from AMCL/SLAM covariance)
#   2) Execution/tracking error buffer (measured in real-time)
#   3) System latency compensation (measured in real-time)
#
# Safety Margin Formula:
#   margin_α = √(χ²₂(α) · λ_max(Σ_xy)) + e_track + v · τ
#
# All parameters can be dynamically updated from sensor data!
#
# Reference: Probabilistic safety margins for mobile robot navigation
# Author: Research Implementation

from .margin_calculator import MarginCalculator, MarginComponents
from .geofence_geometry import GeofenceGeometry, GeofenceCheckResult
from .latency_monitor import LatencyMonitor, LatencyStats
from .tracking_error_monitor import TrackingErrorMonitor, TrackingErrorStats

__all__ = [
    'MarginCalculator',
    'MarginComponents',
    'GeofenceGeometry',
    'GeofenceCheckResult',
    'LatencyMonitor',
    'LatencyStats',
    'TrackingErrorMonitor',
    'TrackingErrorStats',
]
