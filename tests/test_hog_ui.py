from __future__ import annotations

from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from hog_price_baseline import HogObservation
from hog_ui import APP_ASSET_DIR, DEFAULT_PURCHASE_TYPE, LOGO_PATH, resolve_asset, run_local_backtest


class HogUITests(TestCase):
    def test_resolve_asset_uses_shared_worker_frontend_files(self) -> None:
        index_asset = resolve_asset("/")
        logo_asset = resolve_asset("/app/logo.png")

        self.assertIsNotNone(index_asset)
        self.assertIsNotNone(logo_asset)
        self.assertEqual(index_asset[0], APP_ASSET_DIR / "index.html")
        self.assertEqual(index_asset[1], "text/html; charset=utf-8")
        self.assertEqual(logo_asset[0], LOGO_PATH)
        self.assertEqual(logo_asset[1], "image/png")

    def test_run_local_backtest_uses_shared_payload_builder(self) -> None:
        fake_series = [HogObservation(date="2026-03-31", avg_net_price=90.0)]
        fake_payload = {"metrics": {"feature_pack": "price_only"}}

        with (
            patch("hog_ui.download_direct_hog_history", return_value=Path("/tmp/hog-cache.csv")) as download_mock,
            patch("hog_ui.load_cached_daily_series", return_value=fake_series) as load_mock,
            patch("hog_ui._timestamp_from_path", return_value="2026-04-14T12:00:00+00:00") as timestamp_mock,
            patch("hog_ui.aggregate_request_from_daily_series", return_value=fake_payload) as aggregate_mock,
        ):
            payload = run_local_backtest({"feature_pack": ["price_only"]})

        self.assertEqual(payload, fake_payload)
        download_mock.assert_called_once()
        load_mock.assert_called_once_with(Path("/tmp/hog-cache.csv"))
        timestamp_mock.assert_called_once_with(Path("/tmp/hog-cache.csv"))
        aggregate_mock.assert_called_once()
        _, request = aggregate_mock.call_args.args[:2]
        self.assertEqual(request.feature_pack, "price_only")
        self.assertEqual(aggregate_mock.call_args.kwargs["purchase_type"], DEFAULT_PURCHASE_TYPE)
        self.assertEqual(aggregate_mock.call_args.kwargs["source"], "USDA AMS direct hog avg_net_price")
        self.assertEqual(aggregate_mock.call_args.kwargs["refreshed_at"], "2026-04-14T12:00:00+00:00")
