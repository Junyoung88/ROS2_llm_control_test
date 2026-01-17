"""
Real-time System Latency Monitor

This module measures end-to-end system latency by:
1. Command-to-execution latency: Time from cmd_vel publish to actual motion
2. Sensor latency: Timestamp delays in sensor messages
3. Round-trip latency: Echo-based measurement

The measured latency τ is used in the safety margin formula:
    margin_α = √(χ²₂(α) · λ_max(Σ_xy)) + e_track + v_max · τ

Methods:
--------
1. Sensor Timestamp Analysis:
   - Compare message timestamps with current time
   - Accounts for sensor processing + communication delay

2. Command Echo Measurement:
   - Publish velocity command, measure time until odom reflects it
   - Captures full actuation pipeline delay

3. Moving Average Filter:
   - Smooth latency estimates over time window
   - Reject outliers for robust estimation
"""

import time
import math
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Callable, Tuple
from threading import Lock
from enum import Enum, auto


class SafetyLevel(Enum):
    """Safety level based on latency status"""
    NORMAL = auto()      # Normal operation
    CAUTION = auto()     # Elevated latency - increase margins
    EMERGENCY = auto()   # Severe latency spike - recommend stop


@dataclass
class LatencyAdaptiveParams:
    """Dynamic parameters adjusted based on measured latency"""
    base_margin: float = 0.5          # Base safety margin (m)
    latency: float = 0.0              # Measured latency (s)
    max_acceleration: float = 2.0     # Robot max deceleration (m/s^2)
    margin_multiplier: float = 1.0    # Multiplier for base margin
    velocity_factor: float = 1.0      # Multiplier for velocity-based margin
    safety_level: SafetyLevel = SafetyLevel.NORMAL


@dataclass
class LatencyStats:
    """Statistics for measured latency values."""
    current: float = 0.0          # Most recent measurement
    mean: float = 0.0             # Moving average
    std: float = 0.0              # Standard deviation
    min_val: float = float('inf') # Minimum observed
    max_val: float = 0.0          # Maximum observed
    sample_count: int = 0         # Number of samples


class LatencyMonitor:
    """
    Real-time system latency monitor for robotics applications.

    Measures multiple sources of latency:
    - Sensor message delays
    - Command-to-actuation delays
    - Processing pipeline delays

    Parameters:
        window_size: Number of samples for moving average (default: 50)
        outlier_threshold: Reject samples > threshold * std from mean (default: 3.0)
        default_latency: Initial latency estimate [s] (default: 0.1)
    """

    def __init__(
        self,
        window_size: int = 50,
        outlier_threshold: float = 3.0,
        default_latency: float = 0.1
    ):
        self._window_size = window_size
        self._outlier_threshold = outlier_threshold
        self._default_latency = default_latency

        # Separate buffers for different latency sources
        self._sensor_latencies: deque = deque(maxlen=window_size)
        self._command_latencies: deque = deque(maxlen=window_size)
        self._combined_latencies: deque = deque(maxlen=window_size)

        # Thread safety
        self._lock = Lock()

        # Command echo state
        self._pending_command_time: Optional[float] = None
        self._pending_command_value: Optional[float] = None

        # Statistics
        self._stats = LatencyStats()
        self._stats.mean = default_latency

    def record_sensor_latency(self, msg_stamp_sec: float, msg_stamp_nanosec: float) -> float:
        """
        Record latency from a sensor message timestamp.

        Measures the delay between when the sensor captured data
        and when we received it.

        Args:
            msg_stamp_sec: Message timestamp seconds
            msg_stamp_nanosec: Message timestamp nanoseconds

        Returns:
            Measured latency in seconds
        """
        current_time = time.time()
        msg_time = msg_stamp_sec + msg_stamp_nanosec * 1e-9
        latency = current_time - msg_time

        # Sanity check (reject negative or huge latencies)
        if 0 < latency < 5.0:
            with self._lock:
                self._sensor_latencies.append(latency)
                self._update_combined()

        return latency

    def record_sensor_latency_from_header(self, header) -> float:
        """
        Record latency from a ROS message header.

        Args:
            header: std_msgs/Header with stamp field

        Returns:
            Measured latency in seconds
        """
        return self.record_sensor_latency(
            header.stamp.sec,
            header.stamp.nanosec
        )

    def start_command_measurement(self, command_value: float) -> None:
        """
        Start measuring command-to-actuation latency.

        Call this when publishing a velocity command.

        Args:
            command_value: The commanded velocity value to track
        """
        with self._lock:
            self._pending_command_time = time.time()
            self._pending_command_value = command_value

    def complete_command_measurement(
        self,
        observed_value: float,
        tolerance: float = 0.05
    ) -> Optional[float]:
        """
        Complete command latency measurement when motion is observed.

        Call this from odometry callback to detect when robot responds.

        Args:
            observed_value: Current velocity from odometry
            tolerance: How close observed must be to commanded

        Returns:
            Measured latency if measurement completed, None otherwise
        """
        with self._lock:
            if self._pending_command_time is None:
                return None

            if self._pending_command_value is None:
                return None

            # Check if robot has responded to command
            if abs(observed_value - self._pending_command_value) < tolerance:
                latency = time.time() - self._pending_command_time

                # Reset pending state
                self._pending_command_time = None
                self._pending_command_value = None

                # Record if reasonable
                if 0 < latency < 2.0:
                    self._command_latencies.append(latency)
                    self._update_combined()
                    return latency

        return None

    def record_custom_latency(self, latency: float, source: str = "custom") -> None:
        """
        Record a custom latency measurement.

        Args:
            latency: Measured latency in seconds
            source: Identifier for the latency source
        """
        if 0 < latency < 5.0:
            with self._lock:
                self._combined_latencies.append(latency)
                self._update_stats()

    def _update_combined(self) -> None:
        """Update combined latency estimate from all sources."""
        # Weighted combination of sensor and command latencies
        all_latencies = []

        if self._sensor_latencies:
            # Sensor latency is typically more frequent
            all_latencies.extend(self._sensor_latencies)

        if self._command_latencies:
            # Command latency is more accurate for actuation delay
            # Weight it more heavily
            all_latencies.extend(self._command_latencies * 2)

        if all_latencies:
            # Apply outlier rejection
            arr = np.array(all_latencies)
            mean = np.mean(arr)
            std = np.std(arr)

            if std > 0:
                mask = np.abs(arr - mean) < self._outlier_threshold * std
                filtered = arr[mask]
                if len(filtered) > 0:
                    arr = filtered

            # Store filtered values
            self._combined_latencies.clear()
            self._combined_latencies.extend(arr[-self._window_size:])

        self._update_stats()

    def _update_stats(self) -> None:
        """Update statistics from combined measurements."""
        if not self._combined_latencies:
            return

        arr = np.array(self._combined_latencies)
        self._stats.current = arr[-1]
        self._stats.mean = float(np.mean(arr))
        self._stats.std = float(np.std(arr))
        self._stats.min_val = float(np.min(arr))
        self._stats.max_val = float(np.max(arr))
        self._stats.sample_count = len(arr)

    def get_latency(self) -> float:
        """
        Get current latency estimate (τ).

        Returns the mean latency with a safety factor based on variance.
        Uses: τ = mean + k * std for conservative estimate.

        Returns:
            Estimated system latency in seconds
        """
        with self._lock:
            if self._stats.sample_count < 5:
                return self._default_latency

            # Conservative estimate: mean + 1 std
            return self._stats.mean + self._stats.std

    def get_latency_mean(self) -> float:
        """Get mean latency without safety factor."""
        with self._lock:
            if self._stats.sample_count < 5:
                return self._default_latency
            return self._stats.mean

    def get_stats(self) -> LatencyStats:
        """Get full latency statistics."""
        with self._lock:
            return LatencyStats(
                current=self._stats.current,
                mean=self._stats.mean,
                std=self._stats.std,
                min_val=self._stats.min_val,
                max_val=self._stats.max_val,
                sample_count=self._stats.sample_count
            )

    def reset(self) -> None:
        """Reset all measurements."""
        with self._lock:
            self._sensor_latencies.clear()
            self._command_latencies.clear()
            self._combined_latencies.clear()
            self._stats = LatencyStats()
            self._stats.mean = self._default_latency
            self._pending_command_time = None
            self._pending_command_value = None

    # =========================================================================
    # Realtime Adaptive Methods (for latency-tolerant geofence)
    # =========================================================================

    def is_spike_detected(self, threshold: float = 2.0) -> bool:
        """
        Detect if current latency is a spike (significantly above average).

        Args:
            threshold: Ratio to average that triggers spike detection

        Returns:
            True if current latency is spike (> threshold * average)
        """
        with self._lock:
            if self._stats.sample_count < 5:
                return False

            if self._stats.std < 0.001:
                return False

            # Check if current is significantly above mean
            return self._stats.current > self._stats.mean + threshold * self._stats.std

    def is_emergency(self, threshold: float = 3.0, max_acceptable: float = 2.0) -> bool:
        """
        Check if latency is at emergency level.

        Args:
            threshold: Ratio to average that triggers emergency
            max_acceptable: Absolute maximum acceptable latency (seconds)

        Returns:
            True if emergency stop is recommended
        """
        with self._lock:
            # Absolute check
            if self._stats.current > max_acceptable:
                return True

            # Relative check
            if self._stats.sample_count >= 5 and self._stats.std > 0.001:
                return self._stats.current > self._stats.mean + threshold * self._stats.std

            return False

    def get_safety_level(
        self,
        spike_threshold: float = 2.0,
        emergency_threshold: float = 3.0,
        max_acceptable: float = 2.0
    ) -> SafetyLevel:
        """
        Get current safety level based on latency status.

        Args:
            spike_threshold: Ratio for caution level
            emergency_threshold: Ratio for emergency level
            max_acceptable: Absolute max latency for emergency

        Returns:
            SafetyLevel enum value
        """
        if self.is_emergency(emergency_threshold, max_acceptable):
            return SafetyLevel.EMERGENCY
        elif self.is_spike_detected(spike_threshold):
            return SafetyLevel.CAUTION
        return SafetyLevel.NORMAL

    def get_adaptive_params(
        self,
        base_margin: float = 0.5,
        max_acceleration: float = 2.0,
    ) -> LatencyAdaptiveParams:
        """
        Get safety parameters adapted to current latency.

        Adjusts margins based on safety level:
        - NORMAL: Use estimated latency, standard margins
        - CAUTION: Use higher latency estimate, 50% margin increase
        - EMERGENCY: Maximum latency, doubled margins

        Args:
            base_margin: Base safety margin (meters)
            max_acceleration: Robot max deceleration (m/s^2)

        Returns:
            LatencyAdaptiveParams with adjusted values
        """
        safety_level = self.get_safety_level()
        stats = self.get_stats()

        if safety_level == SafetyLevel.EMERGENCY:
            # Use maximum observed latency with buffer
            latency = stats.max_val * 1.5 if stats.max_val > 0 else self._default_latency * 3
            margin_mult = 2.0
            vel_factor = 2.0

        elif safety_level == SafetyLevel.CAUTION:
            # Use conservative estimate with buffer
            latency = (stats.mean + 2 * stats.std) * 1.2 if stats.sample_count >= 5 else self._default_latency * 1.5
            margin_mult = 1.5
            vel_factor = 1.5

        else:  # NORMAL
            latency = self.get_latency()  # mean + 1 std
            margin_mult = 1.0
            vel_factor = 1.0

        return LatencyAdaptiveParams(
            base_margin=base_margin * margin_mult,
            latency=latency,
            max_acceleration=max_acceleration,
            margin_multiplier=margin_mult,
            velocity_factor=vel_factor,
            safety_level=safety_level,
        )

    def compute_dynamic_margin(
        self,
        velocity: float,
        base_margin: float = 0.5,
        max_acceleration: float = 2.0,
    ) -> float:
        """
        Compute dynamic safety margin based on velocity and latency.

        Formula:
            margin = base_margin * multiplier + v * tau * factor + 0.5 * a * tau^2

        This accounts for:
        - Distance traveled during latency
        - Stopping distance at current velocity
        - Safety buffer based on latency stability

        Args:
            velocity: Current robot speed (m/s)
            base_margin: Base safety margin (meters)
            max_acceleration: Robot max deceleration (m/s^2)

        Returns:
            Dynamic safety margin in meters
        """
        params = self.get_adaptive_params(base_margin, max_acceleration)

        # Distance traveled during latency
        velocity_margin = velocity * params.latency * params.velocity_factor

        # Additional distance due to deceleration time
        decel_margin = 0.5 * max_acceleration * params.latency ** 2

        total_margin = params.base_margin + velocity_margin + decel_margin

        return total_margin

    def should_emergency_stop(self) -> bool:
        """
        Check if emergency stop is recommended.

        Returns:
            True if robot should perform emergency stop due to latency
        """
        return self.get_safety_level() == SafetyLevel.EMERGENCY


class LatencyMeasurementNode:
    """
    ROS2 integration helper for latency measurement.

    Provides callbacks that can be connected to ROS subscribers
    to automatically measure latencies.
    """

    def __init__(self, monitor: LatencyMonitor):
        self._monitor = monitor
        self._last_cmd_vel_time: Optional[float] = None

    def odom_callback(self, msg) -> None:
        """
        Process odometry message for latency measurement.

        Records sensor latency and checks for command response.
        """
        # Record sensor latency
        self._monitor.record_sensor_latency_from_header(msg.header)

        # Check for command response
        vx = msg.twist.twist.linear.x
        self._monitor.complete_command_measurement(vx)

    def scan_callback(self, msg) -> None:
        """Record latency from laser scan message."""
        self._monitor.record_sensor_latency_from_header(msg.header)

    def image_callback(self, msg) -> None:
        """Record latency from image message."""
        self._monitor.record_sensor_latency_from_header(msg.header)

    def cmd_vel_callback(self, msg) -> None:
        """
        Track when velocity commands are sent.

        Call this from a subscriber or after publishing cmd_vel.
        """
        vx = msg.linear.x
        if abs(vx) > 0.01:  # Only track non-zero commands
            self._monitor.start_command_measurement(vx)


# =============================================================================
# Latency-Aware Geofence Wrapper
# =============================================================================

class LatencyAwareGeofence:
    """
    Wrapper that adds latency-aware margin adjustment to GeofenceGeometry.

    This class integrates the LatencyMonitor with GeofenceGeometry to provide
    real-time latency-compensated safety margins. When latency increases,
    the safety margin automatically grows to ensure the robot can stop
    before entering a forbidden zone.

    Usage:
        from geofence_geometry import GeofenceGeometry
        from latency_monitor import LatencyAwareGeofence, LatencyMonitor

        geometry = GeofenceGeometry.from_yaml("geofence.yaml")
        monitor = LatencyMonitor()
        latency_geofence = LatencyAwareGeofence(geometry, monitor)

        # In your control loop:
        latency_geofence.update_latency(measured_rtt)
        result = latency_geofence.check_point_adaptive(x, y, velocity)

        if latency_geofence.should_emergency_stop():
            robot.emergency_stop()
    """

    def __init__(
        self,
        geometry,  # GeofenceGeometry instance
        latency_monitor: LatencyMonitor,
        base_margin: float = 0.5,
        max_acceleration: float = 2.0,
    ):
        """
        Initialize latency-aware geofence.

        Args:
            geometry: GeofenceGeometry instance
            latency_monitor: LatencyMonitor instance for measuring latency
            base_margin: Base safety margin when latency is zero (meters)
            max_acceleration: Robot maximum deceleration capability (m/s^2)
        """
        self.geometry = geometry
        self.latency_monitor = latency_monitor
        self.base_margin = base_margin
        self.max_acceleration = max_acceleration

    def update_latency(self, measured_latency: float) -> None:
        """
        Update with a new latency measurement.

        This should be called whenever you receive a response that allows
        you to measure round-trip latency (e.g., heartbeat response).

        Args:
            measured_latency: Measured round-trip latency (seconds)
        """
        self.latency_monitor.record_custom_latency(measured_latency, source="heartbeat")

    def get_adaptive_margin(self, velocity: float = 0.0) -> float:
        """
        Get the current adaptive safety margin.

        The margin increases with:
        - Higher latency
        - Higher velocity
        - Unstable latency (spikes)

        Args:
            velocity: Current robot speed (m/s)

        Returns:
            Computed safety margin in meters
        """
        return self.latency_monitor.compute_dynamic_margin(
            velocity=velocity,
            base_margin=self.base_margin,
            max_acceleration=self.max_acceleration,
        )

    def check_point_adaptive(self, x: float, y: float, velocity: float = 0.0):
        """
        Check a point with latency-adaptive margin.

        The margin used for checking is dynamically computed based on
        current latency and robot velocity.

        Args:
            x: X coordinate
            y: Y coordinate
            velocity: Current robot speed (m/s)

        Returns:
            GeofenceCheckResult with adaptive margin applied
        """
        margin = self.get_adaptive_margin(velocity)
        return self.geometry.check_point(x, y, margin)

    def check_point_3d_adaptive(
        self,
        x: float,
        y: float,
        z: float,
        velocity: float = 0.0,
    ):
        """
        Check a 3D point with latency-adaptive margin.

        Args:
            x: X coordinate
            y: Y coordinate
            z: Z coordinate (height)
            velocity: Current robot speed (m/s)

        Returns:
            GeofenceCheckResult with adaptive margin applied
        """
        margin = self.get_adaptive_margin(velocity)
        return self.geometry.check_point_3d(x, y, z, margin)

    def is_safe(self, x: float, y: float, velocity: float = 0.0) -> bool:
        """
        Quick check if a point is safe with current latency adaptation.

        Args:
            x: X coordinate
            y: Y coordinate
            velocity: Current robot speed (m/s)

        Returns:
            True if point is safe
        """
        result = self.check_point_adaptive(x, y, velocity)
        return not result.is_violated

    def should_emergency_stop(self) -> bool:
        """
        Check if emergency stop is recommended due to latency.

        This should be checked frequently and the robot should
        stop immediately if True.

        Returns:
            True if robot should perform emergency stop
        """
        return self.latency_monitor.should_emergency_stop()

    def get_safety_level(self) -> SafetyLevel:
        """
        Get current safety level.

        Returns:
            SafetyLevel enum (NORMAL, CAUTION, or EMERGENCY)
        """
        return self.latency_monitor.get_safety_level()

    def get_status(self) -> dict:
        """
        Get current safety status including latency information.

        Returns:
            Dictionary with safety status details
        """
        params = self.latency_monitor.get_adaptive_params(
            self.base_margin, self.max_acceleration
        )
        stats = self.latency_monitor.get_stats()

        return {
            'safety_level': params.safety_level.name,
            'latency_ms': params.latency * 1000,
            'base_margin': self.base_margin,
            'current_margin_at_rest': params.base_margin,
            'margin_multiplier': params.margin_multiplier,
            'should_stop': params.safety_level == SafetyLevel.EMERGENCY,
            'latency_stats': {
                'mean_ms': stats.mean * 1000,
                'std_ms': stats.std * 1000,
                'max_ms': stats.max_val * 1000,
                'sample_count': stats.sample_count,
            }
        }


# =============================================================================
# STANDALONE TEST
# =============================================================================

def _test_latency_monitor():
    """Test the latency monitor with simulated data."""
    monitor = LatencyMonitor(window_size=20)

    # Simulate sensor latencies (50-150ms range)
    import random
    for _ in range(30):
        # Simulate message from 50-150ms ago
        delay = random.uniform(0.05, 0.15)
        fake_msg_time = time.time() - delay
        monitor.record_sensor_latency(int(fake_msg_time), int((fake_msg_time % 1) * 1e9))

    stats = monitor.get_stats()
    print(f"Latency Stats:")
    print(f"  Mean: {stats.mean*1000:.1f} ms")
    print(f"  Std:  {stats.std*1000:.1f} ms")
    print(f"  Min:  {stats.min_val*1000:.1f} ms")
    print(f"  Max:  {stats.max_val*1000:.1f} ms")
    print(f"  Conservative τ: {monitor.get_latency()*1000:.1f} ms")

    assert 0.05 < stats.mean < 0.15, "Mean should be in expected range"
    print("✓ Latency monitor test passed")


def _test_adaptive_methods():
    """Test the adaptive margin calculation."""
    monitor = LatencyMonitor(window_size=10, default_latency=0.1)

    # Add normal latency measurements (100ms)
    for i in range(10):
        monitor.record_custom_latency(0.1)

    # Check normal safety level
    assert monitor.get_safety_level() == SafetyLevel.NORMAL
    print(f"Safety Level (normal): {monitor.get_safety_level().name}")

    # Test dynamic margin at rest
    margin_rest = monitor.compute_dynamic_margin(velocity=0.0, base_margin=0.5)
    print(f"Margin at rest: {margin_rest:.3f}m")
    assert margin_rest >= 0.5, "Margin should be at least base margin"

    # Test dynamic margin at 2 m/s
    margin_moving = monitor.compute_dynamic_margin(velocity=2.0, base_margin=0.5)
    print(f"Margin at 2 m/s: {margin_moving:.3f}m")
    assert margin_moving > margin_rest, "Margin should increase with velocity"

    # Add a spike
    monitor.record_custom_latency(0.5)  # 500ms spike

    # Check caution level
    level = monitor.get_safety_level()
    print(f"Safety Level (after spike): {level.name}")
    assert level in [SafetyLevel.CAUTION, SafetyLevel.EMERGENCY]

    # Check margin increased
    margin_spike = monitor.compute_dynamic_margin(velocity=0.0, base_margin=0.5)
    print(f"Margin after spike: {margin_spike:.3f}m")
    assert margin_spike > margin_rest, "Margin should increase after spike"

    print("✓ Adaptive methods test passed")


def _test_latency_aware_geofence():
    """Test the LatencyAwareGeofence wrapper."""
    # Create a mock geometry class
    class MockGeometry:
        def check_point(self, x, y, margin):
            class Result:
                is_violated = (x < margin or y < margin)
                violated_zones = ["zone1"] if is_violated else []
            return Result()

        def check_point_3d(self, x, y, z, margin):
            return self.check_point(x, y, margin)

    geometry = MockGeometry()
    monitor = LatencyMonitor(default_latency=0.1)
    geofence = LatencyAwareGeofence(geometry, monitor, base_margin=0.5)

    # Record some latencies
    for _ in range(10):
        geofence.update_latency(0.1)  # 100ms

    # Test safe point
    assert geofence.is_safe(5.0, 5.0, velocity=0.0), "Point should be safe"

    # Get status
    status = geofence.get_status()
    print(f"Status: {status}")
    assert status['safety_level'] == 'NORMAL'

    # Test margin calculation
    margin = geofence.get_adaptive_margin(velocity=2.0)
    print(f"Adaptive margin at 2 m/s: {margin:.3f}m")
    assert margin > 0.5, "Margin should be greater than base"

    print("✓ LatencyAwareGeofence test passed")


if __name__ == "__main__":
    _test_latency_monitor()
    _test_adaptive_methods()
    _test_latency_aware_geofence()
    print("\n All latency monitor tests passed!")
