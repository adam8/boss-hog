from __future__ import annotations

import unittest

from example_rbp import make_synthetic_data
from rbp import GridCellSpec, RelevanceBasedPredictor, _invert_matrix, pearson_correlation, rolling_predictions


class RBPTests(unittest.TestCase):
    def test_zero_threshold_cells_match_ols_without_asymmetry_boost(self) -> None:
        X = [
            [1.0, 2.0, -1.0],
            [2.0, 0.0, 0.5],
            [3.0, 1.0, 1.5],
            [4.0, 3.0, -0.5],
            [5.0, 2.0, 0.0],
            [6.0, 4.0, 1.0],
        ]
        y = [2.0 + 1.5 * row[0] - 0.7 * row[1] + 0.3 * row[2] for row in X]
        x_t = [2.5, 1.5, 0.25]

        predictor = RelevanceBasedPredictor(random_cells=20, seed=5).fit(
            X,
            y,
            feature_names=["trend", "spread", "stress"],
        )
        cell = predictor._evaluate_cell(GridCellSpec((0, 1, 2), "relevance", 0.0), x_t)

        self.assertIsNotNone(cell)
        assert cell is not None
        self.assertAlmostEqual(cell.asymmetry, 0.0, places=8)
        self.assertAlmostEqual(cell.adjusted_fit, 3.0 * cell.fit, places=8)
        self.assertAlmostEqual(cell.prediction, self._ols_prediction(X, y, x_t), delta=1e-5)

    def test_cell_and_grid_weights_sum_to_one(self) -> None:
        X, y = make_synthetic_data(count=60, seed=3)
        predictor = RelevanceBasedPredictor(random_cells=20, seed=5).fit(
            X[:40],
            y[:40],
            feature_names=["trend", "spread", "stress"],
        )
        result = predictor.predict_one(X[40])

        self.assertAlmostEqual(sum(result.observation_weights), 1.0, places=8)
        self.assertGreater(len(result.cell_results), 0)
        for cell in result.cell_results:
            self.assertAlmostEqual(sum(cell.observation_weights), 1.0, places=8)

    def test_variable_importance_covers_every_feature(self) -> None:
        X, y = make_synthetic_data(count=70, seed=9)
        predictor = RelevanceBasedPredictor(random_cells=20, seed=2).fit(
            X[:50],
            y[:50],
            feature_names=["trend", "spread", "stress"],
        )
        result = predictor.predict_one(X[50])
        self.assertEqual(set(result.variable_importance), {"trend", "spread", "stress"})

    def test_rolling_predictions_have_signal_on_synthetic_data(self) -> None:
        X, y = make_synthetic_data(count=140, seed=7)
        results = rolling_predictions(
            X,
            y,
            initial_window=70,
            predictor_factory=lambda: RelevanceBasedPredictor(random_cells=20, seed=11),
            feature_names=["trend", "spread", "stress"],
        )
        predictions = [result.prediction for result in results]
        actuals = y[70:]
        self.assertGreater(pearson_correlation(predictions, actuals), 0.35)

    @staticmethod
    def _ols_prediction(
        X: list[list[float]],
        y: list[float],
        x_t: list[float],
    ) -> float:
        design = [[1.0] + row for row in X]
        width = len(design[0])
        xtx = [
            [sum(row[left] * row[right] for row in design) for right in range(width)]
            for left in range(width)
        ]
        xty = [sum(row[index] * outcome for row, outcome in zip(design, y)) for index in range(width)]
        coefficients = [
            sum(value * target for value, target in zip(row, xty))
            for row in _invert_matrix(xtx)
        ]
        return coefficients[0] + sum(weight * value for weight, value in zip(coefficients[1:], x_t))


if __name__ == "__main__":
    unittest.main()
