# Relevance-Based Prediction in Plain Python

This repository contains a small reference implementation of the RBP routine described in:

Megan Czasonis, Mark Kritzman, and David Turkington, "A Transparent Alternative to Neural Networks With an Application to Predicting Volatility," *Journal of Investment Management*, Third Quarter 2025.

The goal here is clarity, not speed. The implementation uses only the Python standard library so the paper's mechanics stay visible.

## What is included

- `rbp.py`: core RBP implementation
- `example_rbp.py`: rolling demo on synthetic nonlinear data
- `hog_price_baseline.py`: USDA AMS direct-hog baseline with optional same-report fundamentals
- `hog_backtest_service.py`: bounded request and JSON payload layer shared by the Worker API and local tools
- `hog_ui.py`: minimal local web UI for the hog backtest
- `cf_api_worker.py`: Python Cloudflare Worker entrypoint for the live bounded backtest API
- `cloudflare/ui-worker/`: UI Worker with static assets, API proxying, and Access token validation
- `tests/test_rbp.py`: smoke tests for weights, diagnostics, and predictive signal
- `tests/test_hog_price_baseline.py`: offline tests for the hog data loader and feature builder
- `tests/test_hog_ui.py`: render-level smoke tests for the UI
- `tests/test_hog_backtest_service.py`: request validation and API-payload regression tests

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
python3 hog_ui.py
python3 -m unittest discover -s tests -v
```

## Cloudflare deployment

The deployed app is now split into two Workers:

- `boss-hog-api`: a Python Worker that refreshes USDA data on request, stores the normalized daily dataset in KV, and returns bounded JSON payloads
- `boss-hog-ui`: a JavaScript Worker that serves the browser app, proxies `/api/*` to the Python Worker through a service binding, and validates Cloudflare Access JWTs on dynamic routes

### Worker layout

- [`wrangler.jsonc`](/Users/adamjones/Development/boss-hog/wrangler.jsonc): Python API Worker config
- [`pyproject.toml`](/Users/adamjones/Development/boss-hog/pyproject.toml): Python Worker tooling with `workers-py`
- [`package.json`](/Users/adamjones/Development/boss-hog/package.json): Python Worker deploy scripts
- [`cloudflare/ui-worker/wrangler.jsonc`](/Users/adamjones/Development/boss-hog/cloudflare/ui-worker/wrangler.jsonc): UI Worker config
- [`cloudflare/ui-worker/public/app/index.html`](/Users/adamjones/Development/boss-hog/cloudflare/ui-worker/public/app/index.html): static app shell

### Local Cloudflare tooling

```bash
# Root Python API Worker
npm install
uv sync --group dev

# UI Worker
cd cloudflare/ui-worker
npm install
npm test
```

### Required Cloudflare resources

1. Create a KV namespace for `HOG_DATA_CACHE`.
2. Replace `REPLACE_WITH_HOG_DATA_CACHE_NAMESPACE_ID` in [`wrangler.jsonc`](/Users/adamjones/Development/boss-hog/wrangler.jsonc).
3. Deploy the Python API Worker.
4. Deploy the UI Worker.
5. Enable Cloudflare Access on the UI Worker hostname.
6. Configure one-time PIN login.
7. Create an allow policy for your email address and the second approved user.
8. Set the UI Worker runtime secrets or vars:
   - `ACCESS_TEAM_DOMAIN`
   - `ACCESS_AUD`

### GitHub Actions secrets

The deployment workflows expect:

- `CLOUDFLARE_API_TOKEN`
- `CLOUDFLARE_ACCOUNT_ID`

### Zero Trust Access

The repository can stay public while the deployed site remains private.

Use Cloudflare Access with:

- the UI Worker hostname
- one-time PIN login
- an allow policy limited to your email and the second approved user

The UI Worker also validates `CF-Access-Jwt-Assertion` at runtime on `/` and `/api/*`. That protects the dynamic paths even if the edge-side Access policy is loosened later.

## Local UI

`hog_ui.py` starts a small local server on `http://127.0.0.1:8000` with:

- a compact control panel for the existing backtest parameters
- inline info icons that expand to explain the controls and metrics
- summary cards for correlation, directional accuracy, ex-ante fit, and feature pack
- ranked average feature-importance views

The UI calls the same `hog_price_baseline.py` pipeline under the hood, so it is a thin interface over the existing monthly RBP backtest rather than a separate model path.

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
- The Cloudflare deployment is private-by-default: the public repo can remain open while the site itself stays behind Cloudflare Access.
