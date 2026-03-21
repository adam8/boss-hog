# Relevance-Based Prediction in Plain Python

This repository contains a small reference implementation of the RBP routine described in:

Megan Czasonis, Mark Kritzman, and David Turkington, "A Transparent Alternative to Neural Networks With an Application to Predicting Volatility," *Journal of Investment Management*, Third Quarter 2025.

The goal here is clarity, not speed. The implementation uses only the Python standard library so the paper's mechanics stay visible.

## What is included

- `rbp.py`: core RBP implementation
- `example_rbp.py`: rolling demo on synthetic nonlinear data
- `tests/test_rbp.py`: smoke tests for weights, diagnostics, and predictive signal

## How the code maps to the paper

For a single prediction task `x_t`, the implementation follows the paper's structure:

1. Compute similarity and informativeness with Mahalanobis-distance terms.
2. Combine them into a relevance score for each historical observation.
3. Build grid cells from:
   - subsets of features
   - censoring thresholds such as `0.0`, `0.2`, `0.5`, `0.8`
   - censoring by relevance or similarity
4. Form observation weights for each cell using Equation (1).
5. Turn each cell into a prediction and an adjusted-fit score.
6. Average the cells using adjusted fit as the grid weight.

The code also exposes two paper-style diagnostics:

- observation-level transparency via `PredictionResult.top_observations()`
- variable importance via the difference in average adjusted fit across cells that include or exclude each variable

## Run it

```bash
python3 example_rbp.py
python3 -m unittest discover -s tests -v
```

## Minimal usage

```python
from rbp import RelevanceBasedPredictor

X_train = [
    [0.1, 1.2, -0.4],
    [0.0, 1.0, -0.2],
    [0.7, 0.1, 0.8],
]
y_train = [0.5, 0.4, 1.1]

model = RelevanceBasedPredictor(random_cells=20, seed=11)
model.fit(X_train, y_train, feature_names=["trend", "spread", "stress"])

result = model.predict_one([0.6, 0.2, 0.7])
print(result.prediction)
print(result.fit)
print(result.variable_importance)
print(result.top_observations())
```

## Notes

- This is a reference translation, not a production implementation.
- For larger datasets, replace the hand-written linear algebra with NumPy and cache covariance work more aggressively.
- The sparse grid defaults mirror the paper's idea, but you should tune the grid for your own use case.
