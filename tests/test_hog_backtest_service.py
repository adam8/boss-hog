from __future__ import annotations

import unittest

from hog_backtest_service import (
    BacktestRequest,
    aggregate_request_from_daily_series,
)
from hog_price_baseline import HogObservation


class HogBacktestServiceTests(unittest.TestCase):
    def test_backtest_request_enforces_bounds(self) -> None:
        request = BacktestRequest.from_mapping(
            {
                "feature_pack": "core_fundamentals",
                "max_observations": "180",
                "initial_window": "90",
                "random_cells": "25",
                "seed": "99",
            }
        )
        self.assertEqual(request.feature_pack, "core_fundamentals")
        self.assertEqual(request.max_observations, 180)
        self.assertEqual(request.initial_window, 90)
        self.assertEqual(request.random_cells, 25)
        self.assertEqual(request.seed, 99)

        with self.assertRaisesRegex(ValueError, "initial_window must be smaller"):
            BacktestRequest.from_mapping({"max_observations": "120", "initial_window": "120"})

        with self.assertRaisesRegex(ValueError, "feature_pack must be one of"):
            BacktestRequest.from_mapping({"feature_pack": "nope"})

    def test_aggregate_request_builds_api_payload(self) -> None:
        request = BacktestRequest.from_mapping(
            {
                "feature_pack": "core_fundamentals",
                "max_observations": "120",
                "initial_window": "60",
                "random_cells": "12",
                "seed": "7",
            }
        )
        payload = aggregate_request_from_daily_series(
            self._build_daily_series(),
            request,
            refreshed_at="2026-04-11T16:00:00+00:00",
        )

        self.assertEqual(payload["request"]["feature_pack"], "core_fundamentals")
        self.assertEqual(payload["data_status"]["data_as_of"], "2012-12-03")
        self.assertEqual(payload["data_status"]["refreshed_at"], "2026-04-11T16:00:00+00:00")
        self.assertEqual(payload["metrics"]["feature_pack"], "core_fundamentals")
        self.assertGreater(payload["metrics"]["out_of_sample_prediction_count"], 0)
        self.assertIn("prediction_date", payload["final_month"])
        self.assertTrue(payload["average_feature_importance"])
        self.assertTrue(payload["average_exogenous_importance"])

    def _build_daily_series(self) -> list[HogObservation]:
        observations: list[HogObservation] = []
        month_index = 0
        for year in range(2003, 2013):
            for month in range(1, 13):
                month_index += 1
                base_price = 70.0 + month_index * 0.5 + (month % 4) * 0.4
                day_one = f"{year:04d}-{month:02d}-01"
                day_two = f"{year:04d}-{month:02d}-03"
                observations.append(
                    HogObservation(
                        date=day_one,
                        avg_net_price=base_price,
                        head_count=1900.0 + month_index * 5.0,
                        avg_live_weight=280.0 + month_index * 0.2,
                        avg_carcass_weight=210.0 + month_index * 0.18,
                        avg_sort_loss=1.0 + (month % 3) * 0.02,
                        avg_backfat=0.7 + (month % 4) * 0.01,
                        avg_loin_depth=2.4 + month_index * 0.01,
                        avg_lean_percent=55.0 + (month % 5) * 0.08,
                    )
                )
                observations.append(
                    HogObservation(
                        date=day_two,
                        avg_net_price=base_price + 0.6,
                        head_count=1915.0 + month_index * 5.0,
                        avg_live_weight=280.3 + month_index * 0.2,
                        avg_carcass_weight=210.3 + month_index * 0.18,
                        avg_sort_loss=1.02 + (month % 3) * 0.02,
                        avg_backfat=0.72 + (month % 4) * 0.01,
                        avg_loin_depth=2.43 + month_index * 0.01,
                        avg_lean_percent=55.05 + (month % 5) * 0.08,
                    )
                )
        return observations


if __name__ == "__main__":
    unittest.main()
