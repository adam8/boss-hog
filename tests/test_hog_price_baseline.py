from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from hog_price_baseline import (
    PricePoint,
    aggregate_monthly_average,
    build_price_only_dataset,
    load_cached_daily_series,
)


class HogPriceBaselineTests(unittest.TestCase):
    def test_load_cached_daily_series_and_monthly_aggregation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "direct_hog.csv"
            csv_path.write_text(
                "date,avg_net_price\n"
                "2024-01-02,80.0\n"
                "2024-01-03,82.0\n"
                "2024-02-01,90.0\n",
                encoding="utf-8",
            )

            daily_series = load_cached_daily_series(csv_path)
            monthly_series = aggregate_monthly_average(daily_series)

        self.assertEqual(
            daily_series,
            [
                PricePoint(date="2024-01-02", price=80.0),
                PricePoint(date="2024-01-03", price=82.0),
                PricePoint(date="2024-02-01", price=90.0),
            ],
        )
        self.assertEqual(
            monthly_series,
            [
                PricePoint(date="2024-01-01", price=81.0),
                PricePoint(date="2024-02-01", price=90.0),
            ],
        )

    def test_build_price_only_dataset_creates_expected_rows(self) -> None:
        series = [
            PricePoint(
                date=f"{2023 + (month - 1) // 12:04d}-{((month - 1) % 12) + 1:02d}-01",
                price=100.0 + 2.0 * month,
            )
            for month in range(1, 16)
        ]

        dataset = build_price_only_dataset(series, lookback=12)

        self.assertEqual(len(dataset.X), 2)
        self.assertEqual(dataset.feature_names[0], "ret_1m")
        self.assertEqual(dataset.feature_names[-1], "month_cos")
        self.assertEqual(dataset.dates, ["2024-02-01", "2024-03-01"])
        self.assertEqual(dataset.current_prices, [126.0, 128.0])
        self.assertEqual(dataset.next_prices, [128.0, 130.0])

        first_row = dataset.X[0]
        expected_ret_1m = math.log(126.0 / 124.0)
        expected_ret_3m = math.log(126.0 / 120.0)
        expected_ma_gap_3m = 126.0 / ((122.0 + 124.0 + 126.0) / 3.0) - 1.0
        self.assertAlmostEqual(first_row[0], expected_ret_1m, places=8)
        self.assertAlmostEqual(first_row[1], expected_ret_3m, places=8)
        self.assertAlmostEqual(first_row[4], expected_ma_gap_3m, places=8)
        self.assertAlmostEqual(dataset.y[0], math.log(128.0 / 126.0), places=8)


if __name__ == "__main__":
    unittest.main()
