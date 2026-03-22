# Relevance-Based Prediction in Plain Python

This repository contains a small reference implementation of the RBP routine described in:

Megan Czasonis, Mark Kritzman, and David Turkington, "A Transparent Alternative to Neural Networks With an Application to Predicting Volatility," *Journal of Investment Management*, Third Quarter 2025.

The goal here is clarity, not speed. The implementation uses only the Python standard library so the paper's mechanics stay visible.

## What is included

- `rbp.py`: core RBP implementation
- `example_rbp.py`: rolling demo on synthetic nonlinear data
- `hog_price_baseline.py`: USDA AMS direct-hog baseline with optional same-report fundamentals
- `tests/test_rbp.py`: smoke tests for weights, diagnostics, and predictive signal
- `tests/test_hog_price_baseline.py`: offline tests for the hog data loader and feature builder

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
python3 hog_price_baseline.py --max-observations 240 --initial-window 120 --random-cells 20
python3 hog_price_baseline.py --feature-pack core_fundamentals --max-observations 240 --initial-window 120 --random-cells 20
python3 -m unittest discover -s tests -v
```

## Historical hog baseline

`hog_price_baseline.py` downloads USDA AMS direct hog history for `Prod. Sold (All Purchase Types)`, caches a versioned daily CSV locally, and aggregates the daily report to monthly arithmetic averages before feature building.

The cache schema is:

- `schema_version`
- `date`
- `avg_net_price`
- `head_count`
- `avg_live_weight`
- `avg_carcass_weight`
- `avg_sort_loss`
- `avg_backfat`
- `avg_loin_depth`
- `avg_lean_percent`

The default `price_only` feature pack uses:

- 1, 3, 6, and 12 month log-return momentum
- 3 and 12 month moving-average gaps
- 3 and 12 month realized volatility
- month-of-year seasonality encoded as sine/cosine

The optional `core_fundamentals` feature pack appends these monthly level features in the same training row:

- `head_count_avg`
- `live_weight_avg`
- `carcass_weight_avg`
- `sort_loss_avg`
- `backfat_avg`
- `loin_depth_avg`
- `lean_percent_avg`

The target is still the next month's log return. The script runs a rolling out-of-sample RBP backtest and prints the selected feature pack, overall metrics, averaged variable importance across the rolling predictions, and the final predicted versus realized next-month move.

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
- Zero-threshold linear cells treat asymmetry as `0.0` when there is no censored complement, so their reliability weight does not get an artificial boost.
- The first exogenous-feature pack intentionally stays inside the same USDA AMS direct-hog report so the common history stays long and the diagnostics stay readable.
- Pork cutout and primal-value features from USDA AMS report `2498` are deferred for now because they materially shorten the common history relative to the direct-hog report.
