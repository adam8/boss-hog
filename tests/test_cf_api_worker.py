from __future__ import annotations

from datetime import datetime, timezone
import json
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

workers_stub = ModuleType("workers")
workers_stub.Request = object
workers_stub.Response = object
workers_stub.WorkerEntrypoint = object
workers_stub.fetch = None
sys.modules.setdefault("workers", workers_stub)

from cf_api_worker import Default, _meta_cache_key, _series_cache_key
from hog_price_baseline import HogObservation, serialize_cached_daily_series


class FakeKVNamespace:
    def __init__(self, initial: dict[str, str] | None = None) -> None:
        self.storage = dict(initial or {})

    async def get(self, key: str) -> str | None:
        return self.storage.get(key)

    async def put(self, key: str, value: str) -> None:
        self.storage[key] = value


class IncrementalRefreshWorker(Default):
    def __init__(self, kv_namespace: FakeKVNamespace, incremental_series: list[HogObservation]) -> None:
        self.env = SimpleNamespace(HOG_DATA_CACHE=kv_namespace)
        self.incremental_series = incremental_series
        self.fetch_calls: list[tuple[str, object, object, bool]] = []

    async def _fetch_direct_hog_history(
        self,
        purchase_type: str,
        *,
        start_date=None,
        end_date=None,
        allow_empty: bool = False,
    ) -> list[HogObservation]:
        self.fetch_calls.append((purchase_type, start_date, end_date, allow_empty))
        return list(self.incremental_series)


class ApiWorkerCacheRefreshTests(unittest.IsolatedAsyncioTestCase):
    async def test_stale_cache_refreshes_incrementally_from_last_cached_date(self) -> None:
        purchase_type = "Prod. Sold (All Purchase Types)"
        cache_key = _series_cache_key(purchase_type)
        meta_key = _meta_cache_key(purchase_type)
        cached_series = [
            HogObservation(
                date="2026-06-08",
                avg_net_price=100.0,
                head_count=2000.0,
                avg_live_weight=290.0,
                avg_carcass_weight=220.0,
                avg_sort_loss=1.0,
                avg_backfat=0.7,
                avg_loin_depth=2.5,
                avg_lean_percent=55.0,
            )
        ]
        kv_namespace = FakeKVNamespace(
            {
                cache_key: serialize_cached_daily_series(cached_series),
                meta_key: json.dumps(
                    {
                        "cache_day": "2026-06-09",
                        "refreshed_at": "2026-06-09T06:00:00+00:00",
                        "data_as_of": "2026-06-08",
                    }
                ),
            }
        )
        incremental_series = [
            HogObservation(
                date="2026-06-09",
                avg_net_price=101.0,
                head_count=2010.0,
                avg_live_weight=291.0,
                avg_carcass_weight=221.0,
                avg_sort_loss=1.1,
                avg_backfat=0.71,
                avg_loin_depth=2.6,
                avg_lean_percent=55.2,
            )
        ]
        worker = IncrementalRefreshWorker(kv_namespace, incremental_series)

        fake_now = datetime(2026, 6, 10, 12, 0, tzinfo=timezone.utc)
        with patch("cf_api_worker._utc_now", return_value=fake_now):
            refreshed_series, refreshed_at = await worker._load_series(purchase_type)

        self.assertEqual(refreshed_at, "2026-06-10T12:00:00+00:00")
        self.assertEqual([observation.date for observation in refreshed_series], ["2026-06-08", "2026-06-09"])
        self.assertEqual(len(worker.fetch_calls), 1)
        _, start_date, end_date, allow_empty = worker.fetch_calls[0]
        self.assertEqual(str(start_date), "2026-06-09")
        self.assertEqual(str(end_date), "2026-06-10")
        self.assertTrue(allow_empty)

        updated_meta = json.loads(kv_namespace.storage[meta_key])
        self.assertEqual(updated_meta["cache_day"], "2026-06-10")
        self.assertEqual(updated_meta["data_as_of"], "2026-06-09")


if __name__ == "__main__":
    unittest.main()
