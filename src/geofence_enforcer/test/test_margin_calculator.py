"""
Unit tests for the Margin Calculator module.

These tests verify the correctness of:
1. Chi-square quantile computation (closed-form for df=2)
2. Eigenvalue extraction from covariance matrices
3. Safety margin computation formula

The safety margin formula:
    margin_α = √(χ²₂(α) · λ_max(Σ_xy)) + e_track + v_max · τ
"""

import pytest
import numpy as np
import math

from geofence_enforcer.margin_calculator import MarginCalculator, MarginComponents


class TestChiSquareQuantile:
    """Tests for chi-square quantile computation with df=2."""

    def test_chi2_quantile_095(self):
        """Verify χ²₂⁻¹(0.95) ≈ 5.991."""
        # Known value from chi-square tables
        expected = 5.991
        computed = MarginCalculator._compute_chi2_quantile(0.95)
        assert abs(computed - expected) < 0.01, \
            f"χ²₂⁻¹(0.95) = {computed}, expected ≈ {expected}"

    def test_chi2_quantile_099(self):
        """Verify χ²₂⁻¹(0.99) ≈ 9.210."""
        expected = 9.210
        computed = MarginCalculator._compute_chi2_quantile(0.99)
        assert abs(computed - expected) < 0.01, \
            f"χ²₂⁻¹(0.99) = {computed}, expected ≈ {expected}"

    def test_chi2_quantile_090(self):
        """Verify χ²₂⁻¹(0.90) ≈ 4.605."""
        expected = 4.605
        computed = MarginCalculator._compute_chi2_quantile(0.90)
        assert abs(computed - expected) < 0.01, \
            f"χ²₂⁻¹(0.90) = {computed}, expected ≈ {expected}"

    def test_chi2_quantile_formula(self):
        """Verify closed-form: χ²₂⁻¹(α) = -2·ln(1-α)."""
        for alpha in [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]:
            expected = -2.0 * math.log(1.0 - alpha)
            computed = MarginCalculator._compute_chi2_quantile(alpha)
            assert abs(computed - expected) < 1e-10, \
                f"Formula mismatch at α={alpha}"


class TestCovarianceExtraction:
    """Tests for extracting 2x2 position covariance from 6x6."""

    def test_extract_from_6x6(self):
        """Extract Σ_xy from a 6x6 covariance matrix."""
        cov_6x6 = np.eye(6) * 0.1
        cov_6x6[0, 0] = 0.01  # x variance
        cov_6x6[1, 1] = 0.02  # y variance
        cov_6x6[0, 1] = 0.005  # xy covariance
        cov_6x6[1, 0] = 0.005  # yx covariance (symmetric)

        sigma_xy = MarginCalculator.extract_position_covariance(cov_6x6)

        assert sigma_xy.shape == (2, 2)
        assert abs(sigma_xy[0, 0] - 0.01) < 1e-10
        assert abs(sigma_xy[1, 1] - 0.02) < 1e-10
        assert abs(sigma_xy[0, 1] - 0.005) < 1e-10

    def test_extract_from_flat_array(self):
        """Extract Σ_xy from flat 36-element array (ROS format)."""
        cov_flat = np.zeros(36)
        cov_flat[0] = 0.01   # cov[0,0] = x variance
        cov_flat[7] = 0.02   # cov[1,1] = y variance
        cov_flat[1] = 0.003  # cov[0,1] = xy covariance
        cov_flat[6] = 0.003  # cov[1,0] = yx covariance

        sigma_xy = MarginCalculator.extract_position_covariance(cov_flat)

        assert sigma_xy.shape == (2, 2)
        assert abs(sigma_xy[0, 0] - 0.01) < 1e-10
        assert abs(sigma_xy[1, 1] - 0.02) < 1e-10

    def test_symmetry_enforcement(self):
        """Verify that extracted covariance is symmetric."""
        cov_flat = np.zeros(36)
        cov_flat[0] = 0.01
        cov_flat[7] = 0.02
        cov_flat[1] = 0.003
        cov_flat[6] = 0.004  # Intentionally asymmetric

        sigma_xy = MarginCalculator.extract_position_covariance(cov_flat)

        # Should be averaged to ensure symmetry
        assert abs(sigma_xy[0, 1] - sigma_xy[1, 0]) < 1e-10


class TestEigenvalueComputation:
    """Tests for eigenvalue computation of Σ_xy."""

    def test_diagonal_covariance(self):
        """Eigenvalues of diagonal matrix equal diagonal elements."""
        sigma_xy = np.array([[0.01, 0.0], [0.0, 0.04]])

        lambda_min, lambda_max = MarginCalculator.compute_eigenvalues(sigma_xy)

        assert abs(lambda_min - 0.01) < 1e-10
        assert abs(lambda_max - 0.04) < 1e-10

    def test_isotropic_covariance(self):
        """Isotropic covariance has equal eigenvalues."""
        sigma_xy = np.eye(2) * 0.01

        lambda_min, lambda_max = MarginCalculator.compute_eigenvalues(sigma_xy)

        assert abs(lambda_min - 0.01) < 1e-10
        assert abs(lambda_max - 0.01) < 1e-10

    def test_general_covariance(self):
        """Test eigenvalue computation for general 2x2 symmetric matrix."""
        # Covariance with correlation
        sigma_xy = np.array([[0.04, 0.02], [0.02, 0.04]])

        lambda_min, lambda_max = MarginCalculator.compute_eigenvalues(sigma_xy)

        # For this matrix: λ = 0.04 ± 0.02
        assert abs(lambda_min - 0.02) < 1e-10
        assert abs(lambda_max - 0.06) < 1e-10

    def test_positive_semidefinite(self):
        """Eigenvalues must be non-negative for valid covariance."""
        sigma_xy = np.array([[0.01, 0.0], [0.0, 0.0]])  # Rank-deficient

        lambda_min, lambda_max = MarginCalculator.compute_eigenvalues(sigma_xy)

        assert lambda_min >= 0
        assert lambda_max >= 0


class TestMarginComputation:
    """Tests for the complete margin computation."""

    def test_margin_isotropic_covariance(self):
        """Test margin with isotropic (circular) uncertainty."""
        # σ² = 0.01 → σ = 0.1m
        cov_6x6 = np.zeros((6, 6))
        cov_6x6[0, 0] = 0.01
        cov_6x6[1, 1] = 0.01

        calc = MarginCalculator(
            alpha=0.95,
            e_track=0.05,
            v_max=0.5,
            tau=0.1
        )

        result = calc.compute_margin(cov_6x6)

        # Expected components:
        # χ²₂(0.95) ≈ 5.991
        # λ_max = 0.01
        # estimation_term = √(5.991 * 0.01) ≈ 0.245
        # tracking_error = 0.05
        # latency_term = 0.5 * 0.1 = 0.05
        # total ≈ 0.345

        assert abs(result.estimation_term - 0.245) < 0.01
        assert abs(result.tracking_error - 0.05) < 1e-10
        assert abs(result.latency_term - 0.05) < 1e-10
        assert abs(result.total_margin - 0.345) < 0.01

    def test_margin_with_velocity(self):
        """Test margin with explicit current velocity."""
        cov_6x6 = np.zeros((6, 6))
        cov_6x6[0, 0] = 0.01
        cov_6x6[1, 1] = 0.01

        calc = MarginCalculator(
            alpha=0.95,
            e_track=0.05,
            v_max=0.5,
            tau=0.1
        )

        # With lower velocity
        result_slow = calc.compute_margin(cov_6x6, current_velocity=0.2)
        assert abs(result_slow.latency_term - 0.02) < 1e-10

        # With velocity above v_max (should be capped)
        result_fast = calc.compute_margin(cov_6x6, current_velocity=1.0)
        assert abs(result_fast.latency_term - 0.05) < 1e-10  # Capped at v_max * tau

    def test_margin_high_uncertainty(self):
        """Test margin with high localization uncertainty."""
        cov_6x6 = np.zeros((6, 6))
        cov_6x6[0, 0] = 0.25  # σ = 0.5m
        cov_6x6[1, 1] = 0.25

        calc = MarginCalculator(alpha=0.99)

        result = calc.compute_margin(cov_6x6)

        # With high uncertainty, estimation term dominates
        # √(9.21 * 0.25) ≈ 1.517
        assert result.estimation_term > 1.0

    def test_margin_zero_covariance(self):
        """Test margin with zero covariance (perfect localization)."""
        cov_6x6 = np.zeros((6, 6))

        calc = MarginCalculator(
            alpha=0.95,
            e_track=0.05,
            v_max=0.5,
            tau=0.1
        )

        result = calc.compute_margin(cov_6x6)

        # Only e_track and latency term remain
        assert abs(result.estimation_term) < 1e-10
        assert abs(result.total_margin - 0.10) < 1e-10

    def test_margin_components_dataclass(self):
        """Verify MarginComponents has all required fields."""
        cov_6x6 = np.eye(6) * 0.01

        calc = MarginCalculator(alpha=0.95)
        result = calc.compute_margin(cov_6x6)

        assert hasattr(result, 'estimation_term')
        assert hasattr(result, 'tracking_error')
        assert hasattr(result, 'latency_term')
        assert hasattr(result, 'total_margin')
        assert hasattr(result, 'chi2_quantile')
        assert hasattr(result, 'lambda_max')
        assert hasattr(result, 'lambda_min')


class TestParameterValidation:
    """Tests for parameter validation."""

    def test_invalid_alpha(self):
        """Alpha must be in (0, 1)."""
        with pytest.raises(ValueError):
            MarginCalculator(alpha=0.0)
        with pytest.raises(ValueError):
            MarginCalculator(alpha=1.0)
        with pytest.raises(ValueError):
            MarginCalculator(alpha=-0.5)
        with pytest.raises(ValueError):
            MarginCalculator(alpha=1.5)

    def test_invalid_e_track(self):
        """e_track must be non-negative."""
        with pytest.raises(ValueError):
            MarginCalculator(e_track=-0.1)

    def test_invalid_v_max(self):
        """v_max must be non-negative."""
        with pytest.raises(ValueError):
            MarginCalculator(v_max=-0.5)

    def test_invalid_tau(self):
        """tau must be non-negative."""
        with pytest.raises(ValueError):
            MarginCalculator(tau=-0.1)

    def test_zero_parameters_allowed(self):
        """Zero values are valid for e_track, v_max, tau."""
        calc = MarginCalculator(
            alpha=0.95,
            e_track=0.0,
            v_max=0.0,
            tau=0.0
        )
        assert calc.e_track == 0.0
        assert calc.v_max == 0.0
        assert calc.tau == 0.0


class TestParameterUpdate:
    """Tests for runtime parameter updates."""

    def test_update_alpha(self):
        """Test updating alpha at runtime."""
        calc = MarginCalculator(alpha=0.95)
        old_chi2 = calc.chi2_quantile

        calc.update_parameters(alpha=0.99)

        assert calc.alpha == 0.99
        assert calc.chi2_quantile > old_chi2

    def test_update_multiple(self):
        """Test updating multiple parameters."""
        calc = MarginCalculator()

        calc.update_parameters(
            e_track=0.1,
            v_max=1.0,
            tau=0.2
        )

        assert calc.e_track == 0.1
        assert calc.v_max == 1.0
        assert calc.tau == 0.2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
