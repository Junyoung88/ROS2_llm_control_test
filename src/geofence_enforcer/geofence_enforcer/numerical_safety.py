#!/usr/bin/env python3
"""
Numerical Safety Module - 수학적/알고리즘적 취약점 방어

취약점:
1. Chi-square 가정 위반: 비정규분포 오차 시 신뢰구간 부정확
2. Concave 폴리곤: Shapely buffer()가 예상치 못한 결과 생성
3. 수치적 불안정성: 공분산 행렬이 singular에 가까울 때 고유값 계산 오류
4. 이산화 오차: 연속 경로를 이산 점으로 검사 시 틈새 발생

방어 전략:
1. 분포 검정 + 견고한(robust) 통계 + 보수적 fallback
2. 폴리곤 검증 + convex hull fallback + 보수적 buffer
3. 조건수 검사 + 정규화(regularization) + SVD 기반 계산
4. 경로 세그먼트 검사 + 최소 샘플링 밀도 + 연속 충돌 탐지
"""

import numpy as np
from scipy import stats
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, LineString, Point
from shapely.validation import make_valid
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import IntEnum
import warnings


class NumericalAlertLevel(IntEnum):
    """수치적 안전성 경고 수준"""
    SAFE = 0
    WARNING = 1      # 약간의 불확실성
    DEGRADED = 2     # 보수적 모드로 전환
    CRITICAL = 3     # 즉시 조치 필요


@dataclass
class NumericalSafetyReport:
    """수치적 안전성 검사 결과"""
    alert_level: NumericalAlertLevel
    is_safe: bool
    warnings: List[str]
    corrections_applied: List[str]
    original_value: float
    corrected_value: float
    confidence_degradation: float  # 0.0 ~ 1.0


# =============================================================================
# 1. Chi-Square 가정 검증 및 견고한 대안
# =============================================================================

class DistributionValidator:
    """
    오차 분포 검증기

    Chi-square 기반 신뢰구간은 정규분포를 가정함.
    실제 오차가 비정규 (heavy-tail, skewed)면 마진이 과소 추정됨.

    방어:
    1. Shapiro-Wilk 정규성 검정
    2. 비정규 시 보수적 배수 적용
    3. 견고한 통계(median, MAD) 사용 옵션
    """

    # 정규성 검정 임계값
    NORMALITY_THRESHOLD = 0.05  # p-value 이하면 비정규

    # 비정규 분포에 대한 보수적 배수
    # Chebyshev 부등식: P(|X-μ| ≥ kσ) ≤ 1/k²
    # 99% 신뢰구간 for unknown dist: k = 10 (1% = 1/100)
    CHEBYSHEV_MULTIPLIER = 10.0

    # Heavy-tail 분포에 대한 추가 배수
    HEAVY_TAIL_MULTIPLIER = 1.5

    def __init__(self, history_size: int = 100):
        self._history_size = history_size
        self._error_history: List[Tuple[float, float]] = []  # (x_error, y_error)
        self._last_normality_result: Optional[bool] = None

    def add_error_sample(self, x_error: float, y_error: float):
        """오차 샘플 추가"""
        self._error_history.append((x_error, y_error))
        if len(self._error_history) > self._history_size:
            self._error_history.pop(0)

    def test_normality(self) -> Tuple[bool, float, List[str]]:
        """
        정규성 검정 수행

        Returns:
            (is_normal, p_value, warnings)
        """
        warnings_list = []

        if len(self._error_history) < 20:
            warnings_list.append(
                f"Insufficient samples ({len(self._error_history)}) for normality test"
            )
            return True, 1.0, warnings_list  # 충분하지 않으면 정규 가정

        # X, Y 오차 분리
        x_errors = [e[0] for e in self._error_history]
        y_errors = [e[1] for e in self._error_history]

        # Shapiro-Wilk 검정 (최대 5000 샘플)
        try:
            _, p_x = stats.shapiro(x_errors[-5000:])
            _, p_y = stats.shapiro(y_errors[-5000:])
            p_value = min(p_x, p_y)
        except Exception as e:
            warnings_list.append(f"Normality test failed: {e}")
            return True, 1.0, warnings_list

        is_normal = p_value >= self.NORMALITY_THRESHOLD

        if not is_normal:
            warnings_list.append(
                f"Non-normal distribution detected (p={p_value:.4f}). "
                f"Chi-square confidence interval may be inaccurate."
            )

            # Heavy-tail 검사 (kurtosis)
            kurtosis_x = stats.kurtosis(x_errors)
            kurtosis_y = stats.kurtosis(y_errors)

            if kurtosis_x > 3 or kurtosis_y > 3:
                warnings_list.append(
                    f"Heavy-tail distribution detected (kurtosis: x={kurtosis_x:.2f}, y={kurtosis_y:.2f})"
                )

        self._last_normality_result = is_normal
        return is_normal, p_value, warnings_list

    def get_robust_margin_multiplier(self) -> Tuple[float, List[str]]:
        """
        견고한 마진 배수 계산

        정규분포면 chi-square 기반 배수 사용
        비정규면 Chebyshev 부등식 기반 보수적 배수 사용
        """
        warnings_list = []

        is_normal, p_value, normality_warnings = self.test_normality()
        warnings_list.extend(normality_warnings)

        if is_normal:
            # Chi-square 기반 (정규분포)
            # 99% 신뢰구간: sqrt(χ²₂(0.99)) ≈ 3.03
            return 3.03, warnings_list

        # 비정규분포: Chebyshev 부등식 사용
        # 99% 신뢰구간: k where 1/k² = 0.01 → k = 10
        multiplier = self.CHEBYSHEV_MULTIPLIER

        # Heavy-tail이면 추가 배수
        if len(self._error_history) >= 20:
            x_errors = [e[0] for e in self._error_history]
            kurtosis = stats.kurtosis(x_errors)
            if kurtosis > 3:
                multiplier *= self.HEAVY_TAIL_MULTIPLIER
                warnings_list.append(
                    f"Heavy-tail adjustment applied: multiplier = {multiplier:.1f}"
                )

        warnings_list.append(
            f"Using Chebyshev-based multiplier ({multiplier:.1f}) instead of chi-square (3.03)"
        )

        return multiplier, warnings_list


# =============================================================================
# 2. 폴리곤 검증 및 안전한 처리
# =============================================================================

class PolygonValidator:
    """
    폴리곤 검증 및 안전한 처리

    Concave 폴리곤에서 buffer()가 예상치 못한 결과 생성 가능:
    - Self-intersection
    - 내부 구멍 생성
    - 영역 축소

    방어:
    1. 폴리곤 유효성 검사
    2. 문제 발생 시 make_valid() 또는 convex hull fallback
    3. Buffer 결과 검증
    """

    @staticmethod
    def validate_polygon(vertices: List[Tuple[float, float]]) -> Tuple[bool, List[str]]:
        """
        폴리곤 유효성 검사

        Returns:
            (is_valid, warnings)
        """
        warnings_list = []

        if len(vertices) < 3:
            warnings_list.append(f"Polygon has only {len(vertices)} vertices (minimum 3)")
            return False, warnings_list

        try:
            poly = Polygon(vertices)

            # 유효성 검사
            if not poly.is_valid:
                warnings_list.append(f"Invalid polygon: {poly.is_valid}")
                return False, warnings_list

            # Self-intersection 검사
            if not poly.is_simple:
                warnings_list.append("Polygon has self-intersection")
                return False, warnings_list

            # 면적 검사 (너무 작으면 수치 오류 가능)
            if poly.area < 1e-10:
                warnings_list.append(f"Polygon area too small: {poly.area}")
                return False, warnings_list

            # Concave 검사 (convex hull과 비교)
            hull = poly.convex_hull
            if hull.area > poly.area * 1.01:  # 1% 이상 차이
                warnings_list.append(
                    f"Concave polygon detected (area ratio: {poly.area/hull.area:.3f})"
                )
                # Concave는 경고만 (유효하지만 주의 필요)

            return True, warnings_list

        except Exception as e:
            warnings_list.append(f"Polygon validation failed: {e}")
            return False, warnings_list

    @staticmethod
    def create_safe_polygon(
        vertices: List[Tuple[float, float]],
        use_convex_hull_fallback: bool = True
    ) -> Tuple[Polygon, List[str]]:
        """
        안전한 폴리곤 생성

        문제 발생 시 보수적 fallback (convex hull) 사용
        """
        warnings_list = []

        try:
            poly = Polygon(vertices)

            if not poly.is_valid:
                # make_valid 시도
                poly = make_valid(poly)
                warnings_list.append("Applied make_valid() to fix invalid polygon")

                if not poly.is_valid:
                    if use_convex_hull_fallback:
                        poly = poly.convex_hull
                        warnings_list.append(
                            "Fallback to convex hull (more conservative)"
                        )
                    else:
                        raise ValueError("Cannot create valid polygon")

            return poly, warnings_list

        except Exception as e:
            warnings_list.append(f"Polygon creation failed: {e}")

            if use_convex_hull_fallback:
                # 최후의 수단: convex hull
                from scipy.spatial import ConvexHull
                points = np.array(vertices)
                hull = ConvexHull(points)
                hull_vertices = [tuple(points[i]) for i in hull.vertices]
                warnings_list.append("Created convex hull from vertices as fallback")
                return Polygon(hull_vertices), warnings_list

            raise

    @staticmethod
    def safe_buffer(
        poly: Polygon,
        distance: float,
        resolution: int = 16
    ) -> Tuple[Polygon, List[str]]:
        """
        안전한 buffer 연산

        Buffer 결과를 검증하고 문제 시 보수적 대안 사용
        """
        warnings_list = []

        if distance <= 0:
            return poly, warnings_list

        try:
            # Buffer 수행
            buffered = poly.buffer(distance, resolution=resolution)

            # 결과 검증
            if not buffered.is_valid:
                buffered = make_valid(buffered)
                warnings_list.append("Applied make_valid() to buffer result")

            # 면적 검증 (buffer 후 면적이 줄면 문제)
            expected_min_area = poly.area
            if buffered.area < expected_min_area:
                warnings_list.append(
                    f"Buffer reduced area: {poly.area:.4f} → {buffered.area:.4f}. "
                    "Using convex hull buffer instead."
                )
                # Convex hull의 buffer 사용 (더 보수적)
                buffered = poly.convex_hull.buffer(distance, resolution=resolution)

            # MultiPolygon이면 가장 큰 것만 사용
            if buffered.geom_type == 'MultiPolygon':
                warnings_list.append(
                    f"Buffer created MultiPolygon ({len(buffered.geoms)} parts). "
                    "Using largest component."
                )
                buffered = max(buffered.geoms, key=lambda g: g.area)

            return buffered, warnings_list

        except Exception as e:
            warnings_list.append(f"Buffer failed: {e}. Using convex hull fallback.")
            return poly.convex_hull.buffer(distance), warnings_list


# =============================================================================
# 3. 수치적 안정성 보장
# =============================================================================

class NumericalStabilizer:
    """
    수치적 안정성 보장

    공분산 행렬이 singular에 가까울 때:
    - 고유값 계산 실패/부정확
    - 역행렬 계산 불가
    - 마진이 0, NaN, Inf가 될 수 있음

    방어:
    1. 조건수(condition number) 검사
    2. 정규화(regularization) 적용
    3. SVD 기반 안정적 계산
    4. 실패 시 보수적 기본값
    """

    # 조건수 임계값 (이 이상이면 불안정)
    CONDITION_NUMBER_THRESHOLD = 1e10

    # 정규화 상수 (대각선에 추가)
    REGULARIZATION_EPSILON = 1e-6

    # 최소 허용 고유값
    MIN_EIGENVALUE = 1e-8

    # 수치 불안정 시 최소 마진 (보수적)
    MIN_SAFE_MARGIN = 0.5  # meters

    # 실패 시 기본 마진 (보수적)
    DEFAULT_MARGIN = 1.0  # meters

    @staticmethod
    def check_matrix_stability(
        covariance: np.ndarray
    ) -> Tuple[bool, float, List[str]]:
        """
        공분산 행렬 안정성 검사

        Returns:
            (is_stable, condition_number, warnings)
        """
        warnings_list = []

        # 2x2 또는 6x6 행렬 처리
        if covariance.ndim == 1:
            if covariance.size == 4:
                cov = covariance.reshape(2, 2)
            elif covariance.size == 36:
                cov = covariance.reshape(6, 6)[:2, :2]
            else:
                warnings_list.append(f"Unexpected covariance size: {covariance.size}")
                return False, float('inf'), warnings_list
        else:
            cov = covariance[:2, :2] if covariance.shape[0] > 2 else covariance

        try:
            # 대칭성 검사
            if not np.allclose(cov, cov.T, rtol=1e-5):
                warnings_list.append("Covariance matrix is not symmetric")
                cov = (cov + cov.T) / 2  # 대칭화

            # 조건수 계산
            cond = np.linalg.cond(cov)

            if cond > NumericalStabilizer.CONDITION_NUMBER_THRESHOLD:
                warnings_list.append(
                    f"Ill-conditioned matrix (cond={cond:.2e}). "
                    "Numerical instability likely."
                )
                return False, cond, warnings_list

            # 양정치 검사
            eigenvalues = np.linalg.eigvalsh(cov)
            if np.any(eigenvalues < 0):
                warnings_list.append(
                    f"Non-positive-definite matrix (min eigenvalue: {np.min(eigenvalues):.2e})"
                )
                return False, cond, warnings_list

            # 매우 작은 고유값 검사
            if np.any(eigenvalues < NumericalStabilizer.MIN_EIGENVALUE):
                warnings_list.append(
                    f"Near-singular matrix (min eigenvalue: {np.min(eigenvalues):.2e})"
                )
                return False, cond, warnings_list

            return True, cond, warnings_list

        except np.linalg.LinAlgError as e:
            warnings_list.append(f"Matrix computation failed: {e}")
            return False, float('inf'), warnings_list

    @staticmethod
    def regularize_covariance(
        covariance: np.ndarray,
        epsilon: float = REGULARIZATION_EPSILON
    ) -> Tuple[np.ndarray, List[str]]:
        """
        공분산 행렬 정규화 (안정화)

        대각선에 작은 값을 추가하여 조건수 개선
        """
        warnings_list = []

        # 형태 정리
        if covariance.ndim == 1:
            if covariance.size == 36:
                cov = covariance.reshape(6, 6)
            elif covariance.size == 4:
                cov = covariance.reshape(2, 2)
            else:
                warnings_list.append(f"Cannot regularize: unexpected size {covariance.size}")
                return covariance, warnings_list
        else:
            cov = covariance.copy()

        # 정규화: C' = C + εI
        regularized = cov + np.eye(cov.shape[0]) * epsilon
        warnings_list.append(f"Applied regularization (ε={epsilon})")

        return regularized.flatten() if covariance.ndim == 1 else regularized, warnings_list

    @staticmethod
    def safe_eigenvalue_computation(
        covariance: np.ndarray
    ) -> Tuple[float, List[str]]:
        """
        안전한 최대 고유값 계산

        SVD 기반으로 더 안정적인 계산 수행
        실패 시 보수적 기본값 반환
        """
        warnings_list = []

        # 2x2 xy 공분산 추출
        if covariance.ndim == 1:
            if covariance.size == 36:
                cov = covariance.reshape(6, 6)[:2, :2]
            elif covariance.size == 4:
                cov = covariance.reshape(2, 2)
            else:
                warnings_list.append(f"Unexpected covariance, using default margin")
                return NumericalStabilizer.DEFAULT_MARGIN**2 / 9.21, warnings_list
        else:
            cov = covariance[:2, :2] if covariance.shape[0] > 2 else covariance

        try:
            # 방법 1: SVD 기반 (더 안정적)
            U, S, Vh = np.linalg.svd(cov)
            lambda_max = np.max(S)

            # 결과 검증
            if np.isnan(lambda_max) or np.isinf(lambda_max):
                raise ValueError(f"Invalid eigenvalue: {lambda_max}")

            if lambda_max < NumericalStabilizer.MIN_EIGENVALUE:
                warnings_list.append(
                    f"Eigenvalue too small ({lambda_max:.2e}), using minimum"
                )
                lambda_max = NumericalStabilizer.MIN_EIGENVALUE

            return lambda_max, warnings_list

        except Exception as e:
            warnings_list.append(f"SVD failed: {e}. Using default margin.")

            # Fallback: 대각 요소의 최대값 사용
            try:
                lambda_max = max(cov[0, 0], cov[1, 1])
                if lambda_max > 0:
                    warnings_list.append(f"Using diagonal max as fallback: {lambda_max:.4f}")
                    return lambda_max, warnings_list
            except:
                pass

            # 최후의 수단: 보수적 기본값
            default_lambda = (NumericalStabilizer.DEFAULT_MARGIN**2) / 9.21
            warnings_list.append(f"Using conservative default: λ_max = {default_lambda:.4f}")
            return default_lambda, warnings_list


# =============================================================================
# 4. 이산화 오차 방지
# =============================================================================

class PathDiscretizationGuard:
    """
    이산화 오차 방지

    연속 경로를 이산 점으로 검사할 때 틈새가 발생할 수 있음:
    - 점 사이로 금지구역 통과 가능
    - 빠른 속도에서 샘플링 부족

    방어:
    1. 최소 샘플링 밀도 보장
    2. 선분(segment) 기반 충돌 검사
    3. 안전 버퍼로 틈새 커버
    """

    # 최소 샘플링 간격 (meters)
    MIN_SAMPLING_DISTANCE = 0.05  # 5cm

    # 안전 버퍼 (샘플링 간격의 배수)
    DISCRETIZATION_BUFFER_MULTIPLIER = 1.5

    def __init__(self, forbidden_zones: List[Polygon]):
        self._zones = forbidden_zones
        self._combined_zone = None

        if forbidden_zones:
            # 모든 금지구역 합치기
            from shapely.ops import unary_union
            self._combined_zone = unary_union(forbidden_zones)

    def check_path_segment(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        safety_margin: float = 0.0
    ) -> Tuple[bool, List[str]]:
        """
        경로 세그먼트(선분) 검사

        점이 아닌 전체 선분이 금지구역과 교차하는지 검사
        """
        warnings_list = []

        if self._combined_zone is None:
            return True, warnings_list

        # 선분 생성
        segment = LineString([start, end])

        # 안전 마진 적용
        if safety_margin > 0:
            buffered_zone = self._combined_zone.buffer(safety_margin)
        else:
            buffered_zone = self._combined_zone

        # 교차 검사
        if segment.intersects(buffered_zone):
            warnings_list.append(
                f"Path segment ({start} → {end}) intersects forbidden zone"
            )
            return False, warnings_list

        # 선분과 금지구역 사이 거리 검사
        distance = segment.distance(self._combined_zone)
        if distance < safety_margin:
            warnings_list.append(
                f"Path segment too close to forbidden zone "
                f"(distance: {distance:.3f}m, required: {safety_margin:.3f}m)"
            )
            return False, warnings_list

        return True, warnings_list

    def check_continuous_path(
        self,
        waypoints: List[Tuple[float, float]],
        safety_margin: float = 0.0,
        ensure_minimum_density: bool = True
    ) -> Tuple[bool, List[str], List[Tuple[float, float]]]:
        """
        연속 경로 검사

        Args:
            waypoints: 경로 웨이포인트
            safety_margin: 안전 마진
            ensure_minimum_density: 최소 샘플링 밀도 보장 여부

        Returns:
            (is_safe, warnings, interpolated_points)
        """
        warnings_list = []

        if len(waypoints) < 2:
            return True, warnings_list, waypoints

        # 최소 밀도 보장
        if ensure_minimum_density:
            dense_path = self._ensure_density(waypoints)
            if len(dense_path) > len(waypoints):
                warnings_list.append(
                    f"Interpolated {len(dense_path) - len(waypoints)} points "
                    f"for minimum sampling density"
                )
        else:
            dense_path = waypoints

        # 각 세그먼트 검사
        all_safe = True
        for i in range(len(dense_path) - 1):
            segment_safe, segment_warnings = self.check_path_segment(
                dense_path[i],
                dense_path[i + 1],
                safety_margin
            )
            warnings_list.extend(segment_warnings)

            if not segment_safe:
                all_safe = False

        return all_safe, warnings_list, dense_path

    def _ensure_density(
        self,
        waypoints: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """최소 샘플링 밀도 보장을 위한 보간"""
        result = []

        for i, point in enumerate(waypoints):
            result.append(point)

            if i < len(waypoints) - 1:
                next_point = waypoints[i + 1]
                distance = np.sqrt(
                    (next_point[0] - point[0])**2 +
                    (next_point[1] - point[1])**2
                )

                if distance > self.MIN_SAMPLING_DISTANCE:
                    # 보간 필요
                    num_interpolations = int(distance / self.MIN_SAMPLING_DISTANCE)

                    for j in range(1, num_interpolations + 1):
                        t = j / (num_interpolations + 1)
                        interp_x = point[0] + t * (next_point[0] - point[0])
                        interp_y = point[1] + t * (next_point[1] - point[1])
                        result.append((interp_x, interp_y))

        return result

    def get_discretization_buffer(self, velocity: float) -> float:
        """
        이산화 오차를 커버하기 위한 추가 버퍼 계산

        속도가 빠를수록 샘플 사이 간격이 넓어지므로 더 큰 버퍼 필요
        """
        # 기본 샘플링 주기 (예: 10Hz = 0.1초)
        sampling_period = 0.1

        # 샘플 사이 이동 거리
        sample_distance = velocity * sampling_period

        # 안전 버퍼
        buffer = max(
            sample_distance * self.DISCRETIZATION_BUFFER_MULTIPLIER,
            self.MIN_SAMPLING_DISTANCE
        )

        return buffer


# =============================================================================
# 통합 수치 안전성 검사기
# =============================================================================

class NumericalSafetyChecker:
    """
    통합 수치 안전성 검사기

    모든 수학적/알고리즘적 취약점을 검사하고
    필요 시 보수적 보정을 적용
    """

    def __init__(
        self,
        forbidden_zones: Optional[List[Polygon]] = None,
        enable_distribution_check: bool = True,
        enable_polygon_validation: bool = True,
        enable_numerical_stabilization: bool = True,
        enable_path_discretization_guard: bool = True
    ):
        self._dist_validator = DistributionValidator() if enable_distribution_check else None
        self._poly_validator = PolygonValidator() if enable_polygon_validation else None
        self._num_stabilizer = NumericalStabilizer() if enable_numerical_stabilization else None
        self._path_guard = PathDiscretizationGuard(forbidden_zones or []) if enable_path_discretization_guard else None

        # 통계
        self._total_checks = 0
        self._warnings_count = 0
        self._corrections_count = 0

    def validate_and_correct_margin(
        self,
        covariance: np.ndarray,
        chi_square_quantile: float = 9.21,  # 99% confidence
        base_margin: float = 0.0
    ) -> NumericalSafetyReport:
        """
        안전 마진 계산 검증 및 보정

        1. 공분산 행렬 안정성 검사
        2. 분포 가정 검증
        3. 보수적 마진 계산
        """
        self._total_checks += 1
        warnings_list = []
        corrections = []
        alert_level = NumericalAlertLevel.SAFE

        # 원본 마진 계산 시도
        try:
            if covariance.size == 36:
                cov_xy = covariance.reshape(6, 6)[:2, :2]
            elif covariance.size == 4:
                cov_xy = covariance.reshape(2, 2)
            else:
                cov_xy = np.eye(2) * 0.1

            lambda_max = np.max(np.linalg.eigvalsh(cov_xy))
            original_margin = np.sqrt(chi_square_quantile * lambda_max) + base_margin
        except:
            original_margin = NumericalStabilizer.DEFAULT_MARGIN

        corrected_margin = original_margin
        confidence_degradation = 0.0

        # 1. 수치적 안정성 검사
        if self._num_stabilizer:
            is_stable, cond, stability_warnings = NumericalStabilizer.check_matrix_stability(covariance)
            warnings_list.extend(stability_warnings)

            if not is_stable:
                alert_level = max(alert_level, NumericalAlertLevel.DEGRADED)

                # 정규화 적용
                reg_cov, reg_warnings = NumericalStabilizer.regularize_covariance(covariance)
                corrections.extend(reg_warnings)

                # 안정적인 고유값 계산
                lambda_max, eig_warnings = NumericalStabilizer.safe_eigenvalue_computation(reg_cov)
                warnings_list.extend(eig_warnings)

                corrected_margin = np.sqrt(chi_square_quantile * lambda_max) + base_margin
                corrections.append(f"Margin recalculated with stabilized matrix")

                # 수치 불안정 시 최소 안전 마진 보장
                if corrected_margin < NumericalStabilizer.MIN_SAFE_MARGIN:
                    corrections.append(
                        f"Margin {corrected_margin:.4f}m below minimum safe threshold. "
                        f"Enforcing MIN_SAFE_MARGIN={NumericalStabilizer.MIN_SAFE_MARGIN}m"
                    )
                    corrected_margin = NumericalStabilizer.MIN_SAFE_MARGIN
                    alert_level = max(alert_level, NumericalAlertLevel.CRITICAL)

                confidence_degradation += 0.2

        # 2. 분포 검증
        if self._dist_validator:
            multiplier, dist_warnings = self._dist_validator.get_robust_margin_multiplier()
            warnings_list.extend(dist_warnings)

            # 기본 chi-square 배수(3.03)보다 크면 보정 필요
            if multiplier > 3.03:
                alert_level = max(alert_level, NumericalAlertLevel.WARNING)

                # 마진 배수 적용
                margin_ratio = multiplier / 3.03
                corrected_margin *= margin_ratio
                corrections.append(
                    f"Applied robust multiplier: margin × {margin_ratio:.2f}"
                )

                confidence_degradation += 0.1

        # 3. 최종 검증
        if np.isnan(corrected_margin) or np.isinf(corrected_margin):
            alert_level = NumericalAlertLevel.CRITICAL
            corrected_margin = NumericalStabilizer.DEFAULT_MARGIN
            corrections.append(f"Invalid margin replaced with default: {corrected_margin}m")
            confidence_degradation = 0.5

        # 통계 업데이트
        if warnings_list:
            self._warnings_count += 1
        if corrections:
            self._corrections_count += 1

        return NumericalSafetyReport(
            alert_level=alert_level,
            is_safe=alert_level < NumericalAlertLevel.CRITICAL,
            warnings=warnings_list,
            corrections_applied=corrections,
            original_value=original_margin,
            corrected_value=corrected_margin,
            confidence_degradation=confidence_degradation
        )

    def validate_polygon(
        self,
        vertices: List[Tuple[float, float]]
    ) -> Tuple[Polygon, List[str]]:
        """폴리곤 검증 및 안전한 생성"""
        if self._poly_validator:
            return PolygonValidator.create_safe_polygon(vertices)
        return Polygon(vertices), []

    def validate_path(
        self,
        waypoints: List[Tuple[float, float]],
        safety_margin: float
    ) -> Tuple[bool, List[str]]:
        """경로 검증 (이산화 오차 방지)"""
        if self._path_guard:
            is_safe, warnings, _ = self._path_guard.check_continuous_path(
                waypoints, safety_margin
            )
            return is_safe, warnings
        return True, []

    def get_statistics(self) -> dict:
        return {
            'total_checks': self._total_checks,
            'warnings_count': self._warnings_count,
            'corrections_count': self._corrections_count,
            'correction_rate': self._corrections_count / max(1, self._total_checks)
        }
