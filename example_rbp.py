from __future__ import annotations

import math
import random

from rbp import RelevanceBasedPredictor, pearson_correlation, rolling_predictions


def make_synthetic_data(count: int = 180, seed: int = 7) -> tuple[list[list[float]], list[float]]:
    rng = random.Random(seed)
    rows: list[list[float]] = []
    targets: list[float] = []

    for index in range(count):
        trend = math.sin(index / 10.0) + rng.gauss(0.0, 0.15)
        spread = math.cos(index / 14.0) + rng.gauss(0.0, 0.2)
        stress = math.sin(index / 27.0 + 1.5) + rng.gauss(0.0, 0.1)

        if stress > 0.35:
            target = 0.3 * trend + 1.2 * abs(spread) + 0.5 * trend * spread
        else:
            target = -0.4 * trend + 0.8 * spread - 0.2 * trend * spread

        target += 0.3 * stress + rng.gauss(0.0, 0.08)
        rows.append([trend, spread, stress])
        targets.append(target)

    return rows, targets


def run_demo() -> None:
    X, y = make_synthetic_data()
    feature_names = ["trend", "spread", "stress"]

    def predictor_factory() -> RelevanceBasedPredictor:
        return RelevanceBasedPredictor(random_cells=40, seed=11)

    results = rolling_predictions(
        X,
        y,
        initial_window=80,
        predictor_factory=predictor_factory,
        feature_names=feature_names,
    )
    predictions = [result.prediction for result in results]
    actuals = y[80:]
    fits = [result.fit for result in results]

    print("RBP rolling demo")
    print(f"Predictions: {len(predictions)}")
    print(f"Prediction/actual correlation: {pearson_correlation(predictions, actuals):.3f}")
    print(f"Average ex-ante fit: {sum(fits) / len(fits):.3f}")

    final_result = results[-1]
    print("\nFinal prediction")
    print(f"Prediction: {final_result.prediction:.3f}")
    print(f"Actual: {actuals[-1]:.3f}")
    print(f"Fit: {final_result.fit:.3f}")

    print("\nVariable importance")
    for name, importance in sorted(
        final_result.variable_importance.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(f"{name:>8}: {importance:.4f}")

    top_observations = final_result.top_observations(count=3)
    print("\nMost relevant observations")
    for index, weight in top_observations["most_relevant"]:
        print(f"train_index={index:>3} weight={weight: .4f} target={y[index]: .3f}")

    print("\nLeast relevant observations")
    for index, weight in top_observations["least_relevant"]:
        print(f"train_index={index:>3} weight={weight: .4f} target={y[index]: .3f}")


if __name__ == "__main__":
    run_demo()
