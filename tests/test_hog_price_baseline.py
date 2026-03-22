from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from hog_price_baseline import (
    CORE_FUNDAMENTALS_FEATURE_PACK,
    CORE_FUNDAMENTAL_FEATURES,
    PRICE_FEATURE_NAMES,
    PRICE_ONLY_FEATURE_PACK,
    HogObservation,
    aggregate_monthly_average,
    build_monthly_dataset,
    build_price_only_dataset,
    load_cached_daily_series,
    run_monthly_backtest,
)


class HogPriceBaselineTests(unittest.TestCase):
    def test_load_cached_daily_series_and_monthly_aggregation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "direct_hog_v2.csv"
            csv_path.write_text(
                "schema_version,date,avg_net_price,head_count,avg_live_weight,avg_carcass_weight,"
                "avg_sort_loss,avg_backfat,avg_loin_depth,avg_lean_percent\n"
                "2,2024-01-02,80.0,100,280,210,-2.0,0.60,2.80,56.0\n"
                "2,2024-01-03,82.0,140,282,212,-1.0,0.70,,57.0\n"
                "2,2024-02-01,90.0,160,285,214,-1.5,0.65,2.90,58.0\n",
                encoding="utf-8",
            )

            daily_series = load_cached_daily_series(csv_path)
            monthly_series = aggregate_monthly_average(daily_series)

        self._assert_observation(
            daily_series[0],
            HogObservation(
                date="2024-01-02",
                avg_net_price=80.0,
                head_count=100.0,
                avg_live_weight=280.0,
                avg_carcass_weight=210.0,
                avg_sort_loss=-2.0,
                avg_backfat=0.60,
                avg_loin_depth=2.80,
                avg_lean_percent=56.0,
            ),
        )
        self._assert_observation(
            monthly_series[0],
            HogObservation(
                date="2024-01-01",
                avg_net_price=81.0,
                head_count=120.0,
                avg_live_weight=281.0,
                avg_carcass_weight=211.0,
                avg_sort_loss=-1.5,
                avg_backfat=0.65,
                avg_loin_depth=2.80,
                avg_lean_percent=56.5,
            ),
        )
        self._assert_observation(
            monthly_series[1],
            HogObservation(
                date="2024-02-01",
                avg_net_price=90.0,
                head_count=160.0,
                avg_live_weight=285.0,
                avg_carcass_weight=214.0,
                avg_sort_loss=-1.5,
                avg_backfat=0.65,
                avg_loin_depth=2.90,
                avg_lean_percent=58.0,
            ),
        )

    def test_build_price_only_dataset_regression_shape_and_values(self) -> None:
        series = self._make_monthly_series(count=15)

        dataset = build_price_only_dataset(series, lookback=12)

        self.assertEqual(len(dataset.X), 2)
        self.assertEqual(dataset.feature_names, PRICE_FEATURE_NAMES)
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

    def test_build_core_fundamentals_dataset_order_and_alignment(self) -> None:
        series = self._make_monthly_series(count=15)

        dataset = build_monthly_dataset(
            series,
            lookback=12,
            feature_pack=CORE_FUNDAMENTALS_FEATURE_PACK,
        )

        self.assertEqual(
            dataset.feature_names,
            PRICE_FEATURE_NAMES + [public_name for public_name, _ in CORE_FUNDAMENTAL_FEATURES],
        )
        self.assertEqual(dataset.dates, ["2024-02-01", "2024-03-01"])
        self.assertEqual(dataset.current_prices, [126.0, 128.0])
        self.assertEqual(dataset.next_prices, [128.0, 130.0])
        expected_tail = [1130.0, 283.0, 213.0, -1.7, 0.63, 2.93, 56.3]
        for observed, expected in zip(dataset.X[0][-7:], expected_tail):
            self.assertAlmostEqual(observed, expected, places=8)
        self.assertAlmostEqual(dataset.y[0], math.log(128.0 / 126.0), places=8)

    def test_core_fundamentals_backtest_smoke(self) -> None:
        series = self._make_monthly_series(count=36)

        summary = run_monthly_backtest(
            series,
            feature_pack=CORE_FUNDAMENTALS_FEATURE_PACK,
            initial_window=12,
            random_cells=8,
            seed=3,
        )

        self.assertEqual(summary.feature_pack, CORE_FUNDAMENTALS_FEATURE_PACK)
        self.assertGreater(len(summary.predictions), 0)
        self.assertEqual(set(summary.average_variable_importance), set(summary.feature_names))
        self.assertEqual(
            set(summary.average_exogenous_variable_importance),
            {public_name for public_name, _ in CORE_FUNDAMENTAL_FEATURES},
        )
        self.assertGreater(len(summary.average_exogenous_variable_importance), 0)

    @staticmethod
    def _make_monthly_series(count: int) -> list[HogObservation]:
        rows: list[HogObservation] = []
        for month in range(1, count + 1):
            year = 2023 + (month - 1) // 12
            month_number = ((month - 1) % 12) + 1
            rows.append(
                HogObservation(
                    date=f"{year:04d}-{month_number:02d}-01",
                    avg_net_price=100.0 + 2.0 * month,
                    head_count=1000.0 + 10.0 * month,
                    avg_live_weight=270.0 + month,
                    avg_carcass_weight=200.0 + month,
                    avg_sort_loss=-3.0 + 0.1 * month,
                    avg_backfat=0.50 + 0.01 * month,
                    avg_loin_depth=2.80 + 0.01 * month,
                    avg_lean_percent=55.0 + 0.1 * month,
                )
            )
        return rows

    def _assert_observation(self, observed: HogObservation, expected: HogObservation) -> None:
        self.assertEqual(observed.date, expected.date)
        self.assertAlmostEqual(observed.avg_net_price, expected.avg_net_price, places=8)
        self.assertAlmostEqual(observed.head_count, expected.head_count, places=8)
        self.assertAlmostEqual(observed.avg_live_weight, expected.avg_live_weight, places=8)
        self.assertAlmostEqual(observed.avg_carcass_weight, expected.avg_carcass_weight, places=8)
        self.assertAlmostEqual(observed.avg_sort_loss, expected.avg_sort_loss, places=8)
        self.assertAlmostEqual(observed.avg_backfat, expected.avg_backfat, places=8)
        self.assertAlmostEqual(observed.avg_loin_depth, expected.avg_loin_depth, places=8)
        self.assertAlmostEqual(observed.avg_lean_percent, expected.avg_lean_percent, places=8)


if __name__ == "__main__":
    unittest.main()
