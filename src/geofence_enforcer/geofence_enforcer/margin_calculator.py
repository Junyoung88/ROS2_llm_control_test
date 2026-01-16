"""
Probabilistic Safety Margin Calculator

This module computes the dynamic safety margin for geofence enforcement
based on three distinct uncertainty sources:

    margin_α = √(χ²₂(α) · λ_max(Σ_xy)) + e_track + v_max · τ

Components:
-----------
1. Estimation Uncertainty Term: √(χ²₂(α) · λ_max(Σ_xy))
   - Captures the spatial uncertainty from state estimation (AMCL/SLAM)
   - Uses chi-square distribution with 2 DOF for 2D position
   - λ_max is the maximum eigenvalue of the 2x2 position covariance

2. Execution/Tracking Error: e_track
   - Constant buffer for controller tracking imperfections
   - Accounts for the gap between commanded and actual trajectories

3. Latency Compensation: v_max · τ
   - Distance the robot could travel during system latency
   - Ensures safety even with delayed perception-action loops

Mathematical Background:
------------------------
For a 2D Gaussian distribution N(μ, Σ), the probability that a sample
lies within an ellipse defined by (x-μ)ᵀ Σ⁻¹ (x-μ) ≤ c follows a
chi-square distribution with 2 degrees of freedom.

The closed-form inverse CDF for χ²(2) is:
    χ²₂⁻¹(α) = -2 · ln(1 - α)

This avoids dependency on scipy.stats for the chi-square quantile.
"""

import math
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class MarginComponents:
    """
    Breakdown of safety margin components for analysis and logging.

    Attributes:
        estimation_term: √(χ²₂(α) · λ_max(Σ_xy)) - uncertainty from state estimation
        tracking_error: e_track - execution/controller error buffer
        latency_term: v_max · τ - distance traveled during system latency
        total_margin: Sum of all components
    """
    estimation_term: float
    tracking_error: float
    latency_term: float
    total_margin: float

    # Additional diagnostic information
    chi2_quantile: float
    lambda_max: float
    lambda_min: float


class MarginCalculator:
    """
    Computes probabilistic safety margins for geofence enforcement.

    The margin dynamically adapts to:
    - Current localization uncertainty (from pose covariance)
    - Fixed tracking error buffer (parameter)
    - Maximum velocity and system latency (parameters)

    Parameters:
        alpha (float): Confidence level in [0, 1), e.g., 0.95 or 0.99
        e_track (float): Execution/tracking error buffer in meters
        v_max (float): Maximum translational velocity in m/s
        tau (float): End-to-end system latency in seconds
    """

    def __init__(
        self,
        alpha: float = 0.95,
        e_track: float = 0.05,
        v_max: float = 0.5,
        tau: float = 0.1
    ):
        """
        Initialize the margin calculator with safety parameters.

        Args:
            alpha: Confidence level for the chi-square quantile.
                   Higher values (e.g., 0.99) yield larger, more conservative margins.
            e_track: Tracking/execution error buffer [m].
                     Accounts for controller imperfections.
            v_max: Maximum translational velocity [m/s].
                   Used for latency compensation.
            tau: System latency [s].
                 Total delay from sensing to actuation.

        Raises:
            ValueError: If parameters are outside valid ranges.
        """
        # Validate confidence level
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        # Validate non-negative parameters
        if e_track < 0:
            raise ValueError(f"e_track must be non-negative, got {e_track}")
        if v_max < 0:
            raise ValueError(f"v_max must be non-negative, got {v_max}")
        if tau < 0:
            raise ValueError(f"tau must be non-negative, got {tau}")

        self._alpha = alpha
        self._e_track = e_track
        self._v_max = v_max
        self._tau = tau

        # Precompute chi-square quantile (constant for fixed alpha)
        self._chi2_quantile = self._compute_chi2_quantile(alpha)

    @staticmethod
    def _compute_chi2_quantile(alpha: float) -> float:
        """
        Compute the chi-square quantile for 2 degrees of freedom.

        For χ²(df=2), the inverse CDF has a closed-form solution:
            χ²₂⁻¹(α) = -2 · ln(1 - α)

        This is derived from the CDF: F(x) = 1 - exp(-x/2)
        Solving for x: x = -2 · ln(1 - α)

        Args:
            alpha: Probability level (e.g., 0.95)

        Returns:
            The chi-square quantile value

        Note:
            For α = 0.95: χ²₂⁻¹(0.95) ≈ 5.991
            For α = 0.99: χ²₂⁻¹(0.99) ≈ 9.210
        """
        # Guard against numerical issues near α = 1
        if alpha >= 1.0 - 1e-15:
            return float('inf')

        return -2.0 * math.log(1.0 - alpha)

    @staticmethod
    def extract_position_covariance(pose_covariance: np.ndarray) -> np.ndarray:
        """
        Extract the 2x2 position covariance (Σ_xy) from a 6x6 pose covariance.

        The 6x6 covariance matrix follows the ROS convention:
            [x, y, z, roll, pitch, yaw]

        We extract the upper-left 2x2 block corresponding to (x, y).

        Args:
            pose_covariance: Either a 6x6 matrix or a flat 36-element array
                            (as provided by geometry_msgs/PoseWithCovariance)

        Returns:
            2x2 numpy array representing Σ_xy

        Raises:
            ValueError: If the input shape is invalid
        """
        cov = np.asarray(pose_covariance)

        # Handle flat array from ROS message (row-major order)
        if cov.shape == (36,):
            cov = cov.reshape(6, 6)
        elif cov.shape != (6, 6):
            raise ValueError(
                f"Expected 6x6 matrix or 36-element array, got shape {cov.shape}"
            )

        # Extract the (x, y) covariance block
        sigma_xy = cov[0:2, 0:2].copy()

        # Ensure symmetry (numerical robustness)
        sigma_xy = (sigma_xy + sigma_xy.T) / 2.0

        return sigma_xy

    @staticmethod
    def compute_eigenvalues(sigma_xy: np.ndarray) -> Tuple[float, float]:
        """
        Compute eigenvalues of the 2x2 position covariance matrix.

        For a 2x2 symmetric matrix [[a, b], [b, c]], the eigenvalues are:
            λ = ((a+c) ± √((a-c)² + 4b²)) / 2

        This closed-form avoids numpy.linalg.eigvalsh for efficiency,
        though we use numpy for robustness.

        Args:
            sigma_xy: 2x2 position covariance matrix

        Returns:
            Tuple of (lambda_min, lambda_max)

        Raises:
            ValueError: If eigenvalues are negative (invalid covariance)
        """
        # Use numpy's eigenvalue solver for numerical stability
        eigenvalues = np.linalg.eigvalsh(sigma_xy)

        lambda_min = float(eigenvalues[0])
        lambda_max = float(eigenvalues[1])

        # Covariance matrices must be positive semi-definite
        if lambda_min < -1e-10:
            raise ValueError(
                f"Invalid covariance: negative eigenvalue λ_min = {lambda_min}"
            )

        # Clamp small negative values to zero (numerical tolerance)
        lambda_min = max(0.0, lambda_min)
        lambda_max = max(0.0, lambda_max)

        return lambda_min, lambda_max

    def compute_margin(
        self,
        pose_covariance: np.ndarray,
        current_velocity: Optional[float] = None
    ) -> MarginComponents:
        """
        Compute the total safety margin and its components.

        The margin formula:
            margin_α = √(χ²₂(α) · λ_max(Σ_xy)) + e_track + v · τ

        Where v is either the current velocity (if provided) or v_max.

        Args:
            pose_covariance: 6x6 pose covariance matrix or 36-element flat array
            current_velocity: Optional current translational velocity [m/s].
                             If None, uses v_max for conservative estimate.

        Returns:
            MarginComponents dataclass with breakdown of all terms
        """
        # Extract 2x2 position covariance
        sigma_xy = self.extract_position_covariance(pose_covariance)

        # Compute eigenvalues
        lambda_min, lambda_max = self.compute_eigenvalues(sigma_xy)

        # ===== TERM 1: Estimation Uncertainty =====
        # This term represents the radius of the confidence ellipse
        # √(χ²₂(α) · λ_max) gives the semi-major axis of the α-confidence ellipse
        estimation_term = math.sqrt(self._chi2_quantile * lambda_max)

        # ===== TERM 2: Tracking/Execution Error =====
        # Constant buffer for controller imperfections
        tracking_error = self._e_track

        # ===== TERM 3: Latency Compensation =====
        # Distance traveled during system latency
        # Use current velocity if available, otherwise v_max (conservative)
        velocity = current_velocity if current_velocity is not None else self._v_max
        velocity = min(velocity, self._v_max)  # Cap at v_max
        latency_term = velocity * self._tau

        # ===== TOTAL MARGIN =====
        total_margin = estimation_term + tracking_error + latency_term

        return MarginComponents(
            estimation_term=estimation_term,
            tracking_error=tracking_error,
            latency_term=latency_term,
            total_margin=total_margin,
            chi2_quantile=self._chi2_quantile,
            lambda_max=lambda_max,
            lambda_min=lambda_min
        )

    def compute_margin_value(
        self,
        pose_covariance: np.ndarray,
        current_velocity: Optional[float] = None
    ) -> float:
        """
        Convenience method to get just the total margin value.

        Args:
            pose_covariance: 6x6 pose covariance or 36-element array
            current_velocity: Optional current velocity [m/s]

        Returns:
            Total safety margin in meters
        """
        return self.compute_margin(pose_covariance, current_velocity).total_margin

    @property
    def alpha(self) -> float:
        """Confidence level for uncertainty bound."""
        return self._alpha

    @property
    def e_track(self) -> float:
        """Tracking/execution error buffer [m]."""
        return self._e_track

    @property
    def v_max(self) -> float:
        """Maximum translational velocity [m/s]."""
        return self._v_max

    @property
    def tau(self) -> float:
        """System latency [s]."""
        return self._tau

    @property
    def chi2_quantile(self) -> float:
        """Precomputed chi-square quantile for current alpha."""
        return self._chi2_quantile

    def update_parameters(
        self,
        alpha: Optional[float] = None,
        e_track: Optional[float] = None,
        v_max: Optional[float] = None,
        tau: Optional[float] = None
    ) -> None:
        """
        Update margin calculation parameters at runtime.

        Args:
            alpha: New confidence level (optional)
            e_track: New tracking error [m] (optional)
            v_max: New maximum velocity [m/s] (optional)
            tau: New system latency [s] (optional)
        """
        if alpha is not None:
            if not (0.0 < alpha < 1.0):
                raise ValueError(f"alpha must be in (0, 1), got {alpha}")
            self._alpha = alpha
            self._chi2_quantile = self._compute_chi2_quantile(alpha)

        if e_track is not None:
            if e_track < 0:
                raise ValueError(f"e_track must be non-negative, got {e_track}")
            self._e_track = e_track

        if v_max is not None:
            if v_max < 0:
                raise ValueError(f"v_max must be non-negative, got {v_max}")
            self._v_max = v_max

        if tau is not None:
            if tau < 0:
                raise ValueError(f"tau must be non-negative, got {tau}")
            self._tau = tau


# =============================================================================
# UNIT TESTS / VERIFICATION
# =============================================================================

def _verify_chi2_quantiles():
    """Verify chi-square quantile computation against known values."""
    # Known values for χ²(df=2)
    test_cases = [
        (0.90, 4.605),
        (0.95, 5.991),
        (0.99, 9.210),
    ]

    for alpha, expected in test_cases:
        computed = MarginCalculator._compute_chi2_quantile(alpha)
        assert abs(computed - expected) < 0.01, \
            f"χ²₂⁻¹({alpha}) = {computed}, expected ≈ {expected}"

    print("✓ Chi-square quantile verification passed")


def _verify_margin_computation():
    """Verify margin computation with a simple example."""
    # Example: isotropic covariance with σ² = 0.01 (σ = 0.1m)
    sigma_sq = 0.01
    cov_6x6 = np.zeros((6, 6))
    cov_6x6[0, 0] = sigma_sq  # x variance
    cov_6x6[1, 1] = sigma_sq  # y variance

    calc = MarginCalculator(
        alpha=0.95,
        e_track=0.05,
        v_max=0.5,
        tau=0.1
    )

    result = calc.compute_margin(cov_6x6)

    # Expected: √(5.991 * 0.01) + 0.05 + 0.5*0.1
    #         = √0.05991 + 0.05 + 0.05
    #         ≈ 0.245 + 0.05 + 0.05 = 0.345
    expected_estimation = math.sqrt(5.991 * 0.01)
    expected_total = expected_estimation + 0.05 + 0.05

    assert abs(result.estimation_term - expected_estimation) < 0.01
    assert abs(result.total_margin - expected_total) < 0.01

    print(f"✓ Margin verification passed: {result.total_margin:.4f}m")


if __name__ == "__main__":
    _verify_chi2_quantiles()
    _verify_margin_computation()
    print("\nAll verifications passed!")
