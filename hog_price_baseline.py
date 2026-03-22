from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import date, datetime, timedelta
import json
import math
from pathlib import Path
from statistics import mean
from typing import Iterator, Sequence
from urllib.parse import urlencode
from urllib.request import urlopen

from rbp import RelevanceBasedPredictor, pearson_correlation, rolling_predictions


AMS_REPORT_ID = "2511"
AMS_API_URL = f"https://mpr.datamart.ams.usda.gov/services/v1.1/reports/{AMS_REPORT_ID}/"
DEFAULT_PURCHASE_TYPE = "Prod. Sold (All Purchase Types)"
DEFAULT_START_DATE = date(2002, 1, 1)
DEFAULT_CACHE_PATH = Path(__file__).resolve().parent / "data" / "usda_ams_direct_hog_daily.csv"
MAX_API_RANGE_DAYS = 180


@dataclass(frozen=True)
class PricePoint:
    date: str
    price: float


@dataclass(frozen=True)
class SupervisedDataset:
    X: list[list[float]]
    y: list[float]
    dates: list[str]
    current_prices: list[float]
    next_prices: list[float]
    feature_names: list[str]


@dataclass(frozen=True)
class BacktestSummary:
    series_name: str
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

    final_date = date.today() if end_date is None else end_date
    if start_date > final_date:
        raise ValueError("start_date must be on or before end_date.")

    observed_prices: dict[str, float] = {}
    for chunk_start, chunk_end in _date_chunks(start_date, final_date):
        for point in _fetch_direct_hog_chunk(
            purchase_type=purchase_type,
            start_date=chunk_start,
            end_date=chunk_end,
        ):
            observed_prices[point.date] = point.price

    if not observed_prices:
        raise ValueError(f"No direct hog prices found for purchase type {purchase_type!r}.")

    with cache_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "avg_net_price"])
        for point_date in sorted(observed_prices):
            writer.writerow([point_date, f"{observed_prices[point_date]:.4f}"])

    return cache_path


def load_cached_daily_series(path: Path) -> list[PricePoint]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        series = [
            PricePoint(date=row["date"], price=float(row["avg_net_price"]))
            for row in reader
            if row["date"] and row["avg_net_price"]
        ]

    if not series:
        raise ValueError(f"No cached direct hog rows found in {path}.")
    return sorted(series, key=lambda point: point.date)


def aggregate_monthly_average(series: Sequence[PricePoint]) -> list[PricePoint]:
    grouped: dict[str, list[float]] = {}
    for point in series:
        month_key = point.date[:7]
        grouped.setdefault(month_key, []).append(point.price)

    return [
        PricePoint(date=f"{month_key}-01", price=mean(prices))
        for month_key, prices in sorted(grouped.items())
    ]


def build_price_only_dataset(
    series: Sequence[PricePoint],
    *,
    lookback: int = 12,
) -> SupervisedDataset:
    if lookback < 12:
        raise ValueError("lookback must be at least 12.")
    if len(series) <= lookback + 1:
        raise ValueError("Series is too short for the requested lookback.")

    prices = [point.price for point in series]
    returns = [0.0]
    for previous, current in zip(prices, prices[1:]):
        if previous <= 0.0 or current <= 0.0:
            raise ValueError("Prices must be positive to compute log returns.")
        returns.append(math.log(current / previous))

    feature_names = [
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
    rows: list[list[float]] = []
    targets: list[float] = []
    dates: list[str] = []
    current_prices: list[float] = []
    next_prices: list[float] = []

    for index in range(lookback, len(series) - 1):
        trailing_returns = returns[1 : index + 1]
        ret_3m_window = trailing_returns[-3:]
        ret_6m_window = trailing_returns[-6:]
        ret_12m_window = trailing_returns[-12:]
        trailing_prices = prices[: index + 1]
        price_3m_window = trailing_prices[-3:]
        price_12m_window = trailing_prices[-12:]
        month_number = int(series[index].date[5:7])

        rows.append(
            [
                returns[index],
                sum(ret_3m_window),
                sum(ret_6m_window),
                sum(ret_12m_window),
                prices[index] / mean(price_3m_window) - 1.0,
                prices[index] / mean(price_12m_window) - 1.0,
                _sample_standard_deviation(ret_3m_window),
                _sample_standard_deviation(ret_12m_window),
                math.sin(2.0 * math.pi * month_number / 12.0),
                math.cos(2.0 * math.pi * month_number / 12.0),
            ]
        )
        targets.append(returns[index + 1])
        dates.append(series[index + 1].date)
        current_prices.append(prices[index])
        next_prices.append(prices[index + 1])

    return SupervisedDataset(
        X=rows,
        y=targets,
        dates=dates,
        current_prices=current_prices,
        next_prices=next_prices,
        feature_names=feature_names,
    )


def run_price_only_backtest(
    series: Sequence[PricePoint],
    *,
    series_name: str = DEFAULT_PURCHASE_TYPE,
    source_path: Path = DEFAULT_CACHE_PATH,
    lookback: int = 12,
    initial_window: int = 120,
    random_cells: int = 30,
    seed: int = 11,
) -> BacktestSummary:
    dataset = build_price_only_dataset(series, lookback=lookback)
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

    return BacktestSummary(
        series_name=series_name,
        source_path=source_path,
        observation_count=len(series),
        train_window=initial_window,
        prediction_dates=dataset.dates[initial_window:],
        predictions=predictions,
        actuals=actuals,
        implied_next_prices=implied_next_prices,
        current_prices=current_prices,
        next_prices=next_prices,
        correlation=pearson_correlation(predictions, actuals),
        directional_accuracy=directional_accuracy,
        average_fit=average_fit,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch USDA AMS direct hog prices and run a monthly price-only RBP backtest.",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=DEFAULT_CACHE_PATH,
        help="Where to cache the normalized daily direct hog CSV.",
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
    summary = run_price_only_backtest(
        series,
        series_name=args.purchase_type,
        source_path=csv_path,
        lookback=args.lookback,
        initial_window=args.initial_window,
        random_cells=args.random_cells,
        seed=args.seed,
    )

    print("Historical hog price baseline")
    print("Source series: USDA AMS direct hog avg_net_price")
    print(f"Purchase type: {args.purchase_type}")
    print(f"Cached CSV: {csv_path}")
    print(f"Daily observations cached: {len(daily_series)}")
    print(f"Monthly observations: {summary.observation_count}")
    print(f"Out-of-sample predictions: {len(summary.predictions)}")
    print(f"Prediction/actual correlation: {summary.correlation:.3f}")
    print(f"Directional accuracy: {summary.directional_accuracy:.3f}")
    print(f"Average ex-ante fit: {summary.average_fit:.3f}")

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
) -> list[PricePoint]:
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
            PricePoint(
                date=_normalize_report_date(row["report_date"]),
                price=float(row["avg_net_price"].replace(",", "")),
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


if __name__ == "__main__":
    main()
