from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math
import random
from typing import Callable, Sequence


def mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def pearson_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Inputs must have the same length.")
    if len(left) < 2:
        return 0.0

    left_mean = mean(left)
    right_mean = mean(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]

    covariance = sum(a * b for a, b in zip(left_centered, right_centered))
    left_norm = math.sqrt(sum(value * value for value in left_centered))
    right_norm = math.sqrt(sum(value * value for value in right_centered))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return covariance / (left_norm * right_norm)


def _validate_design_matrix(rows: Sequence[Sequence[float]]) -> None:
    if not rows:
        raise ValueError("X must contain at least one observation.")
    width = len(rows[0])
    if width == 0:
        raise ValueError("X must contain at least one feature.")
    for row in rows:
        if len(row) != width:
            raise ValueError("All rows in X must have the same number of features.")


def _select_columns(rows: Sequence[Sequence[float]], feature_indices: Sequence[int]) -> list[list[float]]:
    return [[row[index] for index in feature_indices] for row in rows]


def _vector_mean(rows: Sequence[Sequence[float]]) -> list[float]:
    width = len(rows[0])
    return [sum(row[index] for row in rows) / len(rows) for index in range(width)]


def _identity_matrix(size: int) -> list[list[float]]:
    return [[1.0 if row == column else 0.0 for column in range(size)] for row in range(size)]


def _covariance_matrix(rows: Sequence[Sequence[float]], ridge: float) -> list[list[float]]:
    width = len(rows[0])
    if len(rows) < 2:
        matrix = _identity_matrix(width)
        for index in range(width):
            matrix[index][index] *= max(ridge, 1.0)
        return matrix

    averages = _vector_mean(rows)
    matrix = [[0.0 for _ in range(width)] for _ in range(width)]
    scale = 1.0 / (len(rows) - 1)
    for row in rows:
        centered = [value - average for value, average in zip(row, averages)]
        for left in range(width):
            for right in range(width):
                matrix[left][right] += centered[left] * centered[right] * scale

    for index in range(width):
        matrix[index][index] += ridge
    return matrix


def _invert_matrix(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    size = len(matrix)
    augmented = [
        list(row) + [1.0 if row_index == column else 0.0 for column in range(size)]
        for row_index, row in enumerate(matrix)
    ]

    for pivot_column in range(size):
        pivot_row = max(range(pivot_column, size), key=lambda index: abs(augmented[index][pivot_column]))
        pivot_value = augmented[pivot_row][pivot_column]
        if abs(pivot_value) < 1e-12:
            raise ValueError("Covariance matrix is singular; raise ridge regularization.")

        if pivot_row != pivot_column:
            augmented[pivot_column], augmented[pivot_row] = augmented[pivot_row], augmented[pivot_column]

        pivot_value = augmented[pivot_column][pivot_column]
        augmented[pivot_column] = [value / pivot_value for value in augmented[pivot_column]]

        for row_index in range(size):
            if row_index == pivot_column:
                continue
            factor = augmented[row_index][pivot_column]
            if factor == 0.0:
                continue
            augmented[row_index] = [
                current - factor * pivot
                for current, pivot in zip(augmented[row_index], augmented[pivot_column])
            ]

    return [row[size:] for row in augmented]


def _quadratic_form(vector: Sequence[float], matrix: Sequence[Sequence[float]]) -> float:
    total = 0.0
    for row_index, row in enumerate(matrix):
        total += vector[row_index] * sum(value * coefficient for value, coefficient in zip(vector, row))
    return total


def _sample_variance_from_zero(values: Sequence[float], mask: Sequence[bool] | None = None) -> float:
    if mask is None:
        sample = list(values)
    else:
        sample = [value for value, keep in zip(values, mask) if keep]
    if len(sample) < 2:
        return 0.0
    return sum(value * value for value in sample) / (len(sample) - 1)


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("Cannot compute a quantile of an empty sample.")
    if probability <= 0.0:
        return float("-inf")
    if probability >= 1.0:
        return max(values)

    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


@dataclass(frozen=True)
class GridCellSpec:
    feature_indices: tuple[int, ...]
    censoring_kind: str
    threshold_percentile: float


@dataclass
class CellResult:
    spec: GridCellSpec
    prediction: float
    fit: float
    asymmetry: float
    adjusted_fit: float
    observation_weights: list[float]
    retained_indices: list[int]
    relevance_scores: list[float]
    similarity_scores: list[float]


@dataclass
class PredictionResult:
    prediction: float
    fit: float
    observation_weights: list[float]
    cell_results: list[CellResult]
    variable_importance: dict[str, float]

    def top_observations(self, count: int = 5) -> dict[str, list[tuple[int, float]]]:
        ranked = sorted(enumerate(self.observation_weights), key=lambda item: item[1], reverse=True)
        lowest = sorted(enumerate(self.observation_weights), key=lambda item: item[1])
        return {
            "most_relevant": ranked[:count],
            "least_relevant": lowest[:count],
        }


class RelevanceBasedPredictor:
    def __init__(
        self,
        *,
        thresholds: Sequence[float] = (0.0, 0.2, 0.5, 0.8),
        include_similarity_cells: bool = True,
        random_cells: int = 100,
        ridge: float = 1e-6,
        seed: int = 0,
    ) -> None:
        self.thresholds = tuple(thresholds)
        self.include_similarity_cells = include_similarity_cells
        self.random_cells = random_cells
        self.ridge = ridge
        self.seed = seed

        self.X: list[list[float]] | None = None
        self.y: list[float] | None = None
        self.feature_names: list[str] | None = None
        self._grid_specs: list[GridCellSpec] = []

    def fit(
        self,
        X: Sequence[Sequence[float]],
        y: Sequence[float],
        *,
        feature_names: Sequence[str] | None = None,
    ) -> "RelevanceBasedPredictor":
        _validate_design_matrix(X)
        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of observations.")

        self.X = [list(map(float, row)) for row in X]
        self.y = [float(value) for value in y]
        feature_count = len(self.X[0])
        if feature_names is None:
            self.feature_names = [f"x{index + 1}" for index in range(feature_count)]
        else:
            if len(feature_names) != feature_count:
                raise ValueError("feature_names must match the number of columns in X.")
            self.feature_names = list(feature_names)

        self._grid_specs = self._build_sparse_grid(feature_count)
        return self

    def predict_one(self, x_t: Sequence[float]) -> PredictionResult:
        if self.X is None or self.y is None or self.feature_names is None:
            raise ValueError("Call fit() before predict_one().")
        if len(x_t) != len(self.feature_names):
            raise ValueError("x_t must have the same number of features used in fit().")

        cell_results: list[CellResult] = []
        for spec in self._grid_specs:
            cell = self._evaluate_cell(spec, list(map(float, x_t)))
            if cell is not None:
                cell_results.append(cell)

        if not cell_results:
            raise ValueError("No valid grid cells were produced for this prediction.")

        adjusted_fit_sum = sum(cell.adjusted_fit for cell in cell_results)
        if adjusted_fit_sum == 0.0:
            cell_weights = [1.0 / len(cell_results)] * len(cell_results)
        else:
            cell_weights = [cell.adjusted_fit / adjusted_fit_sum for cell in cell_results]

        observation_weights = [0.0] * len(self.y)
        for grid_weight, cell in zip(cell_weights, cell_results):
            for index, weight in enumerate(cell.observation_weights):
                observation_weights[index] += grid_weight * weight

        prediction = sum(weight * outcome for weight, outcome in zip(observation_weights, self.y))
        fit = pearson_correlation(observation_weights, self.y) ** 2
        variable_importance = self._variable_importance(cell_results)
        return PredictionResult(
            prediction=prediction,
            fit=fit,
            observation_weights=observation_weights,
            cell_results=cell_results,
            variable_importance=variable_importance,
        )

    def predict_many(self, rows: Sequence[Sequence[float]]) -> list[PredictionResult]:
        return [self.predict_one(row) for row in rows]

    def _evaluate_cell(self, spec: GridCellSpec, x_t: list[float]) -> CellResult | None:
        assert self.X is not None and self.y is not None
        X_subset = _select_columns(self.X, spec.feature_indices)
        x_t_subset = [x_t[index] for index in spec.feature_indices]

        covariance = _covariance_matrix(X_subset, self.ridge)
        inverse_covariance = _invert_matrix(covariance)
        mean_vector = _vector_mean(X_subset)

        similarity_scores: list[float] = []
        informativeness_scores: list[float] = []
        for row in X_subset:
            diff_to_task = [row_value - task_value for row_value, task_value in zip(row, x_t_subset)]
            diff_to_mean = [row_value - avg_value for row_value, avg_value in zip(row, mean_vector)]
            similarity_scores.append(-0.5 * _quadratic_form(diff_to_task, inverse_covariance))
            informativeness_scores.append(_quadratic_form(diff_to_mean, inverse_covariance))

        task_informativeness = _quadratic_form(
            [value - avg_value for value, avg_value in zip(x_t_subset, mean_vector)],
            inverse_covariance,
        )
        relevance_scores = [
            similarity + 0.5 * (info_observation + task_informativeness)
            for similarity, info_observation in zip(similarity_scores, informativeness_scores)
        ]

        if spec.censoring_kind == "similarity":
            base_scores = similarity_scores
        else:
            base_scores = relevance_scores

        threshold = _quantile(base_scores, spec.threshold_percentile)
        mask = [score >= threshold for score in base_scores]
        retained_indices = [index for index, keep in enumerate(mask) if keep]
        if len(retained_indices) < 2:
            return None

        observation_weights = self._observation_weights(relevance_scores, mask)
        prediction = sum(weight * outcome for weight, outcome in zip(observation_weights, self.y))

        fit = pearson_correlation(observation_weights, self.y) ** 2
        positive_count = len(retained_indices)
        negative_mask = [not keep for keep in mask]
        negative_count = len(mask) - positive_count
        if positive_count < 2 or negative_count < 2:
            asymmetry = 0.0
        else:
            positive_weights = self._observation_weights(relevance_scores, mask)
            negative_weights = self._observation_weights(relevance_scores, negative_mask)
            asymmetry = 0.5 * (
                pearson_correlation(positive_weights, self.y) - pearson_correlation(negative_weights, self.y)
            ) ** 2
        adjusted_fit = len(spec.feature_indices) * (fit + asymmetry)

        return CellResult(
            spec=spec,
            prediction=prediction,
            fit=fit,
            asymmetry=asymmetry,
            adjusted_fit=adjusted_fit,
            observation_weights=observation_weights,
            retained_indices=retained_indices,
            relevance_scores=relevance_scores,
            similarity_scores=similarity_scores,
        )

    def _observation_weights(self, relevance_scores: Sequence[float], mask: Sequence[bool]) -> list[float]:
        observation_count = len(relevance_scores)
        retained_count = sum(1 for keep in mask if keep)
        if retained_count < 2:
            return [1.0 / observation_count] * observation_count

        phi = retained_count / observation_count
        retained_average = sum(score for score, keep in zip(relevance_scores, mask) if keep) / retained_count
        full_variance = _sample_variance_from_zero(relevance_scores)
        partial_variance = _sample_variance_from_zero(relevance_scores, mask)
        lambda_squared = 0.0 if partial_variance == 0.0 else full_variance / partial_variance

        baseline = 1.0 / observation_count
        scale = lambda_squared / (retained_count - 1)
        return [
            baseline + scale * (((score if keep else 0.0) - phi * retained_average))
            for score, keep in zip(relevance_scores, mask)
        ]

    def _variable_importance(self, cell_results: Sequence[CellResult]) -> dict[str, float]:
        assert self.feature_names is not None
        importance: dict[str, float] = {}
        for feature_index, feature_name in enumerate(self.feature_names):
            included = [
                cell.adjusted_fit
                for cell in cell_results
                if feature_index in cell.spec.feature_indices
            ]
            excluded = [
                cell.adjusted_fit
                for cell in cell_results
                if feature_index not in cell.spec.feature_indices
            ]
            importance[feature_name] = mean(included) - mean(excluded)
        return importance

    def _build_sparse_grid(self, feature_count: int) -> list[GridCellSpec]:
        base_specs = {
            GridCellSpec(tuple(range(feature_count)), "relevance", 0.0),
            *[
                GridCellSpec((feature_index,), "relevance", 0.0)
                for feature_index in range(feature_count)
            ],
        }

        remaining_specs: list[GridCellSpec] = []
        threshold_kinds: list[str] = ["relevance"]
        if self.include_similarity_cells:
            threshold_kinds.append("similarity")

        total_subsets = (1 << feature_count) - 1
        approximate_total_grid = total_subsets * len(self.thresholds) * len(threshold_kinds)

        if approximate_total_grid <= 200_000:
            for subset_size in range(1, feature_count + 1):
                for subset in combinations(range(feature_count), subset_size):
                    for threshold in self.thresholds:
                        kinds = ["relevance"] if threshold == 0.0 else threshold_kinds
                        for kind in kinds:
                            spec = GridCellSpec(subset, kind, threshold)
                            if spec not in base_specs:
                                remaining_specs.append(spec)
        else:
            sampler = random.Random(self.seed)
            seen = set(base_specs)
            while len(remaining_specs) < self.random_cells:
                subset_size = sampler.randint(1, feature_count)
                subset = tuple(sorted(sampler.sample(range(feature_count), subset_size)))
                threshold = sampler.choice(self.thresholds)
                kinds = ["relevance"] if threshold == 0.0 else threshold_kinds
                kind = sampler.choice(kinds)
                spec = GridCellSpec(subset, kind, threshold)
                if spec in seen:
                    continue
                seen.add(spec)
                remaining_specs.append(spec)
            return list(base_specs) + remaining_specs

        sampler = random.Random(self.seed)
        if len(remaining_specs) <= self.random_cells:
            sampled_specs = remaining_specs
        else:
            sampled_specs = sampler.sample(remaining_specs, self.random_cells)
        return list(base_specs) + sampled_specs


def rolling_predictions(
    X: Sequence[Sequence[float]],
    y: Sequence[float],
    *,
    initial_window: int,
    predictor_factory: Callable[[], RelevanceBasedPredictor] | None = None,
    feature_names: Sequence[str] | None = None,
) -> list[PredictionResult]:
    if initial_window < 2:
        raise ValueError("initial_window must be at least 2.")
    if initial_window >= len(X):
        raise ValueError("initial_window must be smaller than the sample size.")

    results: list[PredictionResult] = []
    for stop in range(initial_window, len(X)):
        predictor = predictor_factory() if predictor_factory is not None else RelevanceBasedPredictor()
        predictor.fit(X[:stop], y[:stop], feature_names=feature_names)
        results.append(predictor.predict_one(X[stop]))
    return results
