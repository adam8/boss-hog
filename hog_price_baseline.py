from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path
from statistics import mean
from typing import Sequence
from urllib.request import urlopen

from rbp import RelevanceBasedPredictor, pearson_correlation, rolling_predictions


ERS_HISTORY_URL = (
    "https://www.ers.usda.gov/media/5028/"
    "historical-monthly-price-spread-data-for-beef-pork-broilers.csv?v=90046"
)
DEFAULT_DATA_ITEM = "Pork gross farm value"
DEFAULT_CACHE_PATH = Path(__file__).resolve().parent / "data" / "ers_meat_history.csv"


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
    data_item: str
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


def download_ers_history(
    cache_path: Path = DEFAULT_CACHE_PATH,
    *,
    force: bool = False,
) -> Path:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not force:
        return cache_path

    with urlopen(ERS_HISTORY_URL) as response:
        cache_path.write_bytes(response.read())
    return cache_path


def load_ers_price_series(
    path: Path,
    *,
    data_item: str = DEFAULT_DATA_ITEM,
) -> list[PricePoint]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        points = [
            PricePoint(
                date=f"{int(row['Year']):04d}-{int(row['Month-number']):02d}-01",
                price=float(row["Value"]),
            )
            for row in reader
            if row["Data_Item"] == data_item and row["Value"]
        ]

    if not points:
        raise ValueError(f"No rows found for data item {data_item!r} in {path}.")
    return sorted(points, key=lambda point: point.date)


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
    data_item: str = DEFAULT_DATA_ITEM,
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
        data_item=data_item,
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
        description="Fetch USDA ERS monthly pork data and run a price-only RBP backtest.",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=DEFAULT_CACHE_PATH,
        help="Where to cache the ERS CSV.",
    )
    parser.add_argument(
        "--data-item",
        default=DEFAULT_DATA_ITEM,
        help="ERS Data_Item value to extract. Defaults to a hog-price proxy.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Redownload the ERS CSV even if a cached copy exists.",
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
    csv_path = download_ers_history(args.cache_path, force=args.force_download)
    series = load_ers_price_series(csv_path, data_item=args.data_item)
    if args.max_observations is not None:
        if args.max_observations < 14:
            raise ValueError("max_observations must be at least 14.")
        series = series[-args.max_observations :]
    summary = run_price_only_backtest(
        series,
        data_item=args.data_item,
        source_path=csv_path,
        lookback=args.lookback,
        initial_window=args.initial_window,
        random_cells=args.random_cells,
        seed=args.seed,
    )

    print("Historical hog price baseline")
    print(f"Source item: {args.data_item}")
    print(f"Cached CSV: {csv_path}")
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
    print(f"Current price proxy: {summary.current_prices[last_index]:.2f}")
    print(f"Predicted next price proxy: {summary.implied_next_prices[last_index]:.2f}")
    print(f"Actual next price proxy: {summary.next_prices[last_index]:.2f}")


def _sample_standard_deviation(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    average = mean(values)
    variance = sum((value - average) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


if __name__ == "__main__":
    main()
