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
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Callable
from threading import Lock


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


if __name__ == "__main__":
    _test_latency_monitor()
