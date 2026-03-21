from __future__ import annotations

import unittest

from example_rbp import make_synthetic_data
from rbp import RelevanceBasedPredictor, pearson_correlation, rolling_predictions


class RBPTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
