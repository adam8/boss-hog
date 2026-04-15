from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from io import StringIO
import json
import math
from pathlib import Path
from statistics import mean
from typing import Iterator, Sequence
from urllib.parse import urlencode
from urllib.request import urlopen

from rbp import RelevanceBasedPredictor, rolling_predictions


AMS_REPORT_ID = "2511"
AMS_API_URL = f"https://mpr.datamart.ams.usda.gov/services/v1.1/reports/{AMS_REPORT_ID}/"
CACHE_SCHEMA_VERSION = "2"
DEFAULT_PURCHASE_TYPE = "Prod. Sold (All Purchase Types)"
DEFAULT_START_DATE = date(2002, 1, 1)
DEFAULT_CACHE_PATH = Path(__file__).resolve().parent / "data" / "usda_ams_direct_hog_daily_v2.csv"
MAX_API_RANGE_DAYS = 180
PRICE_ONLY_FEATURE_PACK = "price_only"
CORE_FUNDAMENTALS_FEATURE_PACK = "core_fundamentals"
PRICE_FEATURE_NAMES = [
    "ret_1m",
    "ret_3m",
    "ret_6m",
    "ret_12m",
    "ma_gap_3m",
    "ma_gap_12m",
    "vol_3m",
    "vol_12m",
    "month_sin",
    "month_cos",
]
CORE_FUNDAMENTAL_FEATURES = [
    ("head_count_avg", "head_count"),
    ("live_weight_avg", "avg_live_weight"),
    ("carcass_weight_avg", "avg_carcass_weight"),
    ("sort_loss_avg", "avg_sort_loss"),
    ("backfat_avg", "avg_backfat"),
    ("loin_depth_avg", "avg_loin_depth"),
    ("lean_percent_avg", "avg_lean_percent"),
]
CACHE_NUMERIC_FIELDS = [
    "avg_net_price",
    "head_count",
    "avg_live_weight",
    "avg_carcass_weight",
    "avg_sort_loss",
    "avg_backfat",
    "avg_loin_depth",
    "avg_lean_percent",
]
FEATURE_PACKS = {
    PRICE_ONLY_FEATURE_PACK: (),
    CORE_FUNDAMENTALS_FEATURE_PACK: tuple(CORE_FUNDAMENTAL_FEATURES),
}


@dataclass(frozen=True)
class HogObservation:
    date: str
    avg_net_price: float | None
    head_count: float | None = None
    avg_live_weight: float | None = None
    avg_carcass_weight: float | None = None
    avg_sort_loss: float | None = None
    avg_backfat: float | None = None
    avg_loin_depth: float | None = None
    avg_lean_percent: float | None = None


@dataclass(frozen=True)
class SupervisedDataset:
    X: list[list[float]]
    y: list[float]
    dates: list[str]
    current_prices: list[float]
    next_prices: list[float]
    feature_names: list[str]


@dataclass(frozen=True)
class ForecastFeatureRow:
    row: list[float]
    feature_names: list[str]
    starting_month_bucket: str
    target_month_bucket: str
    starting_month_price_average: float


@dataclass(frozen=True)
class BacktestSummary:
    series_name: str
    feature_pack: str
    feature_names: list[str]
    source_path: Path
    observation_count: int
    train_window: int
    prediction_dates: list[str]
    predictions: list[float]
    actuals: list[float]
    implied_next_prices: list[float]
    current_prices: list[float]
    next_prices: list[float]
    correlation: float
    directional_accuracy: float
    average_fit: float
    average_variable_importance: dict[str, float]
    average_exogenous_variable_importance: dict[str, float]


def download_direct_hog_history(
    cache_path: Path = DEFAULT_CACHE_PATH,
    *,
    purchase_type: str = DEFAULT_PURCHASE_TYPE,
    start_date: date = DEFAULT_START_DATE,
    end_date: date | None = None,
    force: bool = False,
) -> Path:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not force:
        return cache_path

    observations = fetch_direct_hog_history(
        purchase_type=purchase_type,
        start_date=start_date,
        end_date=end_date,
    )
    write_cached_daily_series(cache_path, observations)
    return cache_path


def load_cached_daily_series(path: Path) -> list[HogObservation]:
    return deserialize_cached_daily_series(path.read_text(encoding="utf-8"), source=str(path))


def write_cached_daily_series(path: Path, series: Sequence[HogObservation]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialize_cached_daily_series(series), encoding="utf-8")


def fetch_direct_hog_history(
    *,
    purchase_type: str = DEFAULT_PURCHASE_TYPE,
    start_date: date = DEFAULT_START_DATE,
    end_date: date | None = None,
) -> list[HogObservation]:
    final_date = date.today() if end_date is None else end_date
    if start_date > final_date:
        raise ValueError("start_date must be on or before end_date.")

    observed_rows: dict[str, HogObservation] = {}
    for chunk_start, chunk_end in _date_chunks(start_date, final_date):
        for observation in _fetch_direct_hog_chunk(
            purchase_type=purchase_type,
            start_date=chunk_start,
            end_date=chunk_end,
        ):
            observed_rows[observation.date] = observation

    if not observed_rows:
        raise ValueError(f"No direct hog prices found for purchase type {purchase_type!r}.")

    return [observed_rows[point_date] for point_date in sorted(observed_rows)]


def serialize_cached_daily_series(series: Sequence[HogObservation]) -> str:
    handle = StringIO()
    writer = csv.writer(handle)
    writer.writerow(["schema_version", "date", *CACHE_NUMERIC_FIELDS])
    for observation in sorted(series, key=lambda point: point.date):
        writer.writerow(
            [
                CACHE_SCHEMA_VERSION,
                observation.date,
                *[_format_optional_float(getattr(observation, field_name)) for field_name in CACHE_NUMERIC_FIELDS],
            ]
        )
    return handle.getvalue()


def deserialize_cached_daily_series(payload: str, *, source: str = "<memory>") -> list[HogObservation]:
    reader = csv.DictReader(StringIO(payload))
    if reader.fieldnames is None:
        raise ValueError(f"Missing CSV header in {source}.")

    expected_columns = {"schema_version", "date", *CACHE_NUMERIC_FIELDS}
    missing_columns = expected_columns.difference(reader.fieldnames)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required cache columns in {source}: {missing}.")

    series: list[HogObservation] = []
    for row in reader:
        if row["schema_version"] != CACHE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported cache schema version {row['schema_version']!r} in {source}; "
                f"expected {CACHE_SCHEMA_VERSION!r}."
            )
        if not row["date"]:
            continue
        series.append(
            HogObservation(
                date=row["date"],
                avg_net_price=_parse_optional_float(row["avg_net_price"]),
                head_count=_parse_optional_float(row["head_count"]),
                avg_live_weight=_parse_optional_float(row["avg_live_weight"]),
                avg_carcass_weight=_parse_optional_float(row["avg_carcass_weight"]),
                avg_sort_loss=_parse_optional_float(row["avg_sort_loss"]),
                avg_backfat=_parse_optional_float(row["avg_backfat"]),
                avg_loin_depth=_parse_optional_float(row["avg_loin_depth"]),
                avg_lean_percent=_parse_optional_float(row["avg_lean_percent"]),
            )
        )

    if not series:
        raise ValueError(f"No cached direct hog rows found in {source}.")
    return sorted(series, key=lambda point: point.date)


def aggregate_monthly_average(series: Sequence[HogObservation]) -> list[HogObservation]:
    grouped: dict[str, list[HogObservation]] = {}
    for observation in series:
        month_key = observation.date[:7]
        grouped.setdefault(month_key, []).append(observation)

    monthly_rows: list[HogObservation] = []
    for month_key, rows in sorted(grouped.items()):
        monthly_rows.append(
            HogObservation(
                date=f"{month_key}-01",
                avg_net_price=_mean_optional(row.avg_net_price for row in rows),
                head_count=_mean_optional(row.head_count for row in rows),
                avg_live_weight=_mean_optional(row.avg_live_weight for row in rows),
                avg_carcass_weight=_mean_optional(row.avg_carcass_weight for row in rows),
                avg_sort_loss=_mean_optional(row.avg_sort_loss for row in rows),
                avg_backfat=_mean_optional(row.avg_backfat for row in rows),
                avg_loin_depth=_mean_optional(row.avg_loin_depth for row in rows),
                avg_lean_percent=_mean_optional(row.avg_lean_percent for row in rows),
            )
        )
    return monthly_rows


def build_monthly_dataset(
    series: Sequence[HogObservation],
    *,
    lookback: int = 12,
    feature_pack: str = PRICE_ONLY_FEATURE_PACK,
) -> SupervisedDataset:
    _validate_feature_pack(feature_pack)
    if lookback < 12:
        raise ValueError("lookback must be at least 12.")
    if len(series) <= lookback + 1:
        raise ValueError("Series is too short for the requested lookback.")

    observations = sorted(series, key=lambda point: point.date)
    feature_names = _feature_names_for_pack(feature_pack)
    selected_fields = [field_name for _, field_name in FEATURE_PACKS[feature_pack]]
    rows: list[list[float]] = []
    targets: list[float] = []
    dates: list[str] = []
    current_prices: list[float] = []
    next_prices: list[float] = []

    for index in range(lookback, len(observations) - 1):
        window = observations[index - lookback : index + 2]
        if not _has_consecutive_months(window):
            continue

        history = observations[index - lookback : index + 1]
        current = observations[index]
        target = observations[index + 1]
        target_price = target.avg_net_price
        if target_price is None or target_price <= 0.0:
            continue
        row_context = _build_feature_row_context(history, current, selected_fields)
        if row_context is None:
            continue

        rows.append(row_context.row)
        targets.append(math.log(float(target_price) / row_context.starting_month_price_average))
        dates.append(target.date)
        current_prices.append(row_context.starting_month_price_average)
        next_prices.append(float(target_price))

    if not rows:
        raise ValueError("No supervised rows were produced for the selected feature pack.")

    return SupervisedDataset(
        X=rows,
        y=targets,
        dates=dates,
        current_prices=current_prices,
        next_prices=next_prices,
        feature_names=feature_names,
    )


def build_price_only_dataset(
    series: Sequence[HogObservation],
    *,
    lookback: int = 12,
) -> SupervisedDataset:
    return build_monthly_dataset(series, lookback=lookback, feature_pack=PRICE_ONLY_FEATURE_PACK)


def build_current_forecast_row(
    series: Sequence[HogObservation],
    *,
    lookback: int = 12,
    feature_pack: str = PRICE_ONLY_FEATURE_PACK,
) -> ForecastFeatureRow:
    _validate_feature_pack(feature_pack)
    if lookback < 12:
        raise ValueError("lookback must be at least 12.")
    observations = sorted(series, key=lambda point: point.date)
    if len(observations) < lookback + 1:
        raise ValueError("Series is too short for a current forecast row.")

    history = observations[-(lookback + 1) :]
    if not _has_consecutive_months(history):
        raise ValueError("Latest monthly window is not consecutive.")

    current = observations[-1]
    selected_fields = [field_name for _, field_name in FEATURE_PACKS[feature_pack]]
    row_context = _build_feature_row_context(history, current, selected_fields)
    if row_context is None:
        raise ValueError("Latest completed month cannot produce a current forecast row.")
    return ForecastFeatureRow(
        row=row_context.row,
        feature_names=_feature_names_for_pack(feature_pack),
        starting_month_bucket=current.date,
        target_month_bucket=_next_month_bucket(current.date),
        starting_month_price_average=row_context.starting_month_price_average,
    )


def run_monthly_backtest(
    series: Sequence[HogObservation],
    *,
    series_name: str = DEFAULT_PURCHASE_TYPE,
    source_path: Path = DEFAULT_CACHE_PATH,
    lookback: int = 12,
    initial_window: int = 120,
    random_cells: int = 30,
    seed: int = 11,
    feature_pack: str = PRICE_ONLY_FEATURE_PACK,
) -> BacktestSummary:
    dataset = build_monthly_dataset(series, lookback=lookback, feature_pack=feature_pack)
    if initial_window >= len(dataset.X):
        raise ValueError("initial_window must be smaller than the supervised sample size.")

    results = rolling_predictions(
        dataset.X,
        dataset.y,
        initial_window=initial_window,
        predictor_factory=lambda: RelevanceBasedPredictor(random_cells=random_cells, seed=seed),
        feature_names=dataset.feature_names,
    )
    predictions = [result.prediction for result in results]
    actuals = dataset.y[initial_window:]
    current_prices = dataset.current_prices[initial_window:]
    next_prices = dataset.next_prices[initial_window:]
    implied_next_prices = [
        current_price * math.exp(prediction)
        for current_price, prediction in zip(current_prices, predictions)
    ]
    directional_accuracy = mean(
        1.0 if (prediction >= 0.0) == (actual >= 0.0) else 0.0
        for prediction, actual in zip(predictions, actuals)
    )
    average_fit = mean(result.fit for result in results)
    average_variable_importance = {
        feature_name: mean(result.variable_importance[feature_name] for result in results)
        for feature_name in dataset.feature_names
    }
    exogenous_feature_names = [
        feature_name
        for feature_name in dataset.feature_names
        if feature_name not in PRICE_FEATURE_NAMES
    ]
    average_exogenous_variable_importance = {
        feature_name: average_variable_importance[feature_name]
        for feature_name in exogenous_feature_names
    }

    return BacktestSummary(
        series_name=series_name,
        feature_pack=feature_pack,
        feature_names=dataset.feature_names,
        source_path=source_path,
        observation_count=len(series),
        train_window=initial_window,
        prediction_dates=dataset.dates[initial_window:],
        predictions=predictions,
        actuals=actuals,
        implied_next_prices=implied_next_prices,
        current_prices=current_prices,
        next_prices=next_prices,
        correlation=_pearson_correlation(predictions, actuals),
        directional_accuracy=directional_accuracy,
        average_fit=average_fit,
        average_variable_importance=average_variable_importance,
        average_exogenous_variable_importance=average_exogenous_variable_importance,
    )


def run_price_only_backtest(
    series: Sequence[HogObservation],
    *,
    series_name: str = DEFAULT_PURCHASE_TYPE,
    source_path: Path = DEFAULT_CACHE_PATH,
    lookback: int = 12,
    initial_window: int = 120,
    random_cells: int = 30,
    seed: int = 11,
) -> BacktestSummary:
    return run_monthly_backtest(
        series,
        series_name=series_name,
        source_path=source_path,
        lookback=lookback,
        initial_window=initial_window,
        random_cells=random_cells,
        seed=seed,
        feature_pack=PRICE_ONLY_FEATURE_PACK,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch USDA AMS direct hog data and run a monthly RBP backtest.",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=DEFAULT_CACHE_PATH,
        help="Where to cache the versioned daily direct hog CSV.",
    )
    parser.add_argument(
        "--purchase-type",
        default=DEFAULT_PURCHASE_TYPE,
        help="USDA AMS purchase_type to extract from the direct hog report.",
    )
    parser.add_argument(
        "--start-date",
        default=DEFAULT_START_DATE.isoformat(),
        help="Inclusive start date for the AMS history pull, in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Optional inclusive end date for the AMS history pull, in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Refetch the AMS history even if a cached CSV exists.",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=12,
        help="Trailing months used for the longest lag and volatility features.",
    )
    parser.add_argument(
        "--initial-window",
        type=int,
        default=120,
        help="Number of supervised rows used before the first out-of-sample prediction.",
    )
    parser.add_argument(
        "--random-cells",
        type=int,
        default=30,
        help="Additional sparse-grid cells to sample in each RBP fit.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=11,
        help="Random seed for the sparse-grid sampler.",
    )
    parser.add_argument(
        "--feature-pack",
        choices=sorted(FEATURE_PACKS),
        default=PRICE_ONLY_FEATURE_PACK,
        help="Feature set to include in the monthly supervised dataset.",
    )
    parser.add_argument(
        "--max-observations",
        type=int,
        default=None,
        help="Optionally keep only the most recent N monthly observations before feature building.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_date = _parse_iso_date(args.start_date)
    end_date = None if args.end_date is None else _parse_iso_date(args.end_date)
    csv_path = download_direct_hog_history(
        args.cache_path,
        purchase_type=args.purchase_type,
        start_date=start_date,
        end_date=end_date,
        force=args.force_download,
    )
    daily_series = load_cached_daily_series(csv_path)
    series = aggregate_monthly_average(daily_series)
    if args.max_observations is not None:
        if args.max_observations < 14:
            raise ValueError("max_observations must be at least 14.")
        series = series[-args.max_observations :]
    summary = run_monthly_backtest(
        series,
        series_name=args.purchase_type,
        source_path=csv_path,
        lookback=args.lookback,
        initial_window=args.initial_window,
        random_cells=args.random_cells,
        seed=args.seed,
        feature_pack=args.feature_pack,
    )

    print("Historical hog price baseline")
    print("Source series: USDA AMS direct hog avg_net_price")
    print(f"Purchase type: {args.purchase_type}")
    print(f"Feature pack: {summary.feature_pack}")
    print(f"Cached CSV: {csv_path}")
    print(f"Daily observations cached: {len(daily_series)}")
    print(f"Monthly observations: {summary.observation_count}")
    print(f"Out-of-sample predictions: {len(summary.predictions)}")
    print(f"Prediction/actual correlation: {summary.correlation:.3f}")
    print(f"Directional accuracy: {summary.directional_accuracy:.3f}")
    print(f"Average ex-ante fit: {summary.average_fit:.3f}")
    _print_importances("Top average feature importances", summary.average_variable_importance)
    if summary.average_exogenous_variable_importance:
        _print_importances(
            "Top average exogenous feature importances",
            summary.average_exogenous_variable_importance,
        )

    last_index = len(summary.predictions) - 1
    print("\nFinal out-of-sample month")
    print(f"Date: {summary.prediction_dates[last_index]}")
    print(f"Predicted next-month log return: {summary.predictions[last_index]:.4f}")
    print(f"Actual next-month log return: {summary.actuals[last_index]:.4f}")
    print(f"Current monthly average direct hog price: {summary.current_prices[last_index]:.2f}")
    print(f"Predicted next monthly average price: {summary.implied_next_prices[last_index]:.2f}")
    print(f"Actual next monthly average price: {summary.next_prices[last_index]:.2f}")


def _fetch_direct_hog_chunk(
    *,
    purchase_type: str,
    start_date: date,
    end_date: date,
) -> list[HogObservation]:
    query = urlencode(
        {
            "q": f"report_date={_format_ams_date(start_date)}:{_format_ams_date(end_date)}",
            "allSections": "true",
        }
    )
    with urlopen(f"{AMS_API_URL}?{query}") as response:
        payload = json.loads(response.read().decode("utf-8", "ignore"))

    if isinstance(payload, str):
        return []

    for section in payload:
        if section.get("reportSection") != "Barrows/Gilts":
            continue
        return [
            HogObservation(
                date=_normalize_report_date(row["report_date"]),
                avg_net_price=_parse_optional_float(row.get("avg_net_price")),
                head_count=_parse_optional_float(row.get("head_count")),
                avg_live_weight=_parse_optional_float(row.get("avg_live_weight")),
                avg_carcass_weight=_parse_optional_float(row.get("avg_carcass_weight")),
                avg_sort_loss=_parse_optional_float(row.get("avg_sort_loss")),
                avg_backfat=_parse_optional_float(row.get("avg_backfat")),
                avg_loin_depth=_parse_optional_float(row.get("avg_loin_depth")),
                avg_lean_percent=_parse_optional_float(row.get("avg_lean_percent")),
            )
            for row in section.get("results", [])
            if row.get("purchase_type") == purchase_type and row.get("avg_net_price")
        ]
    return []


def _date_chunks(start_date: date, end_date: date) -> Iterator[tuple[date, date]]:
    cursor = start_date
    while cursor <= end_date:
        chunk_end = min(cursor + timedelta(days=MAX_API_RANGE_DAYS - 1), end_date)
        yield cursor, chunk_end
        cursor = chunk_end + timedelta(days=1)


def _mean_optional(values: Sequence[float | None] | Iterator[float | None]) -> float | None:
    sample = [value for value in values if value is not None]
    if not sample:
        return None
    return mean(sample)


def _parse_optional_float(raw_value: str | None) -> float | None:
    if raw_value is None:
        return None
    normalized = raw_value.replace(",", "").strip()
    if not normalized:
        return None
    return float(normalized)


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.4f}"


@dataclass(frozen=True)
class _FeatureRowContext:
    row: list[float]
    starting_month_price_average: float


def _feature_names_for_pack(feature_pack: str) -> list[str]:
    return PRICE_FEATURE_NAMES + [public_name for public_name, _ in FEATURE_PACKS[feature_pack]]


def _build_feature_row_context(
    history: Sequence[HogObservation],
    current: HogObservation,
    selected_fields: Sequence[str],
) -> _FeatureRowContext | None:
    price_window = [point.avg_net_price for point in history]
    if any(price is None or price <= 0.0 for price in price_window):
        return None
    if any(getattr(current, field_name) is None for field_name in selected_fields):
        return None

    current_prices_window = [float(price) for price in price_window]
    trailing_returns = [
        math.log(current_price / previous_price)
        for previous_price, current_price in zip(current_prices_window, current_prices_window[1:])
    ]
    month_number = int(current.date[5:7])
    row = [
        trailing_returns[-1],
        sum(trailing_returns[-3:]),
        sum(trailing_returns[-6:]),
        sum(trailing_returns[-12:]),
        current_prices_window[-1] / mean(current_prices_window[-3:]) - 1.0,
        current_prices_window[-1] / mean(current_prices_window[-12:]) - 1.0,
        _sample_standard_deviation(trailing_returns[-3:]),
        _sample_standard_deviation(trailing_returns[-12:]),
        math.sin(2.0 * math.pi * month_number / 12.0),
        math.cos(2.0 * math.pi * month_number / 12.0),
    ]
    row.extend(float(getattr(current, field_name)) for field_name in selected_fields)
    return _FeatureRowContext(
        row=row,
        starting_month_price_average=current_prices_window[-1],
    )


def _validate_feature_pack(feature_pack: str) -> None:
    if feature_pack not in FEATURE_PACKS:
        valid = ", ".join(sorted(FEATURE_PACKS))
        raise ValueError(f"Unknown feature_pack {feature_pack!r}; expected one of {valid}.")


def _has_consecutive_months(series: Sequence[HogObservation]) -> bool:
    month_indices = [_month_index(point.date) for point in series]
    return all(current == previous + 1 for previous, current in zip(month_indices, month_indices[1:]))


def _month_index(date_string: str) -> int:
    year = int(date_string[:4])
    month = int(date_string[5:7])
    return year * 12 + month


def _next_month_bucket(bucket_date: str) -> str:
    year = int(bucket_date[:4])
    month = int(bucket_date[5:7])
    if month == 12:
        return f"{year + 1:04d}-01-01"
    return f"{year:04d}-{month + 1:02d}-01"


def _normalize_report_date(raw_date: str) -> str:
    return datetime.strptime(raw_date, "%m/%d/%Y").strftime("%Y-%m-%d")


def _format_ams_date(value: date) -> str:
    return value.strftime("%m/%d/%Y")


def _parse_iso_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _sample_standard_deviation(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    average = mean(values)
    variance = sum((value - average) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def _pearson_correlation(left: Sequence[float], right: Sequence[float]) -> float:
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


def _print_importances(title: str, importances: dict[str, float], *, count: int = 5) -> None:
    if not importances:
        return
    print(f"\n{title}")
    for feature_name, importance in sorted(
        importances.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:count]:
        print(f"{feature_name:>20}: {importance:.4f}")


if __name__ == "__main__":
    main()
