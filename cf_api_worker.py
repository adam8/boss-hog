from __future__ import annotations

from datetime import datetime, timezone
import json
import re
from urllib.parse import parse_qs, urlencode, urlparse

from workers import Request, Response, WorkerEntrypoint, fetch

from hog_backtest_service import BacktestRequest, aggregate_request_from_daily_series
from hog_price_baseline import (
    AMS_API_URL,
    CACHE_SCHEMA_VERSION,
    DEFAULT_PURCHASE_TYPE,
    HogObservation,
    _date_chunks,
    _format_ams_date,
    _normalize_report_date,
    _parse_optional_float,
    deserialize_cached_daily_series,
    serialize_cached_daily_series,
)


CACHE_PREFIX = f"hog-direct-cache-v{CACHE_SCHEMA_VERSION}"


class Default(WorkerEntrypoint):
    async def fetch(self, request: Request):
        url = urlparse(request.url)
        if request.method != "GET":
            return _json_response({"error": "Method not allowed."}, status=405)
        if url.path == "/health":
            return _json_response({"ok": True, "service": "boss-hog-api"})
        if url.path != "/backtest":
            return _json_response({"error": "Not found."}, status=404)

        try:
            params = parse_qs(url.query)
            backtest_request = BacktestRequest.from_mapping(params)
        except ValueError as error:
            return _json_response({"error": str(error)}, status=400)

        try:
            daily_series, refreshed_at = await self._load_series(DEFAULT_PURCHASE_TYPE)
            payload = aggregate_request_from_daily_series(
                daily_series,
                backtest_request,
                purchase_type=DEFAULT_PURCHASE_TYPE,
                source="USDA AMS direct hog avg_net_price",
                refreshed_at=refreshed_at,
            )
            return _json_response(payload)
        except Exception as error:  # pragma: no cover - exercised in worker runtime
            print(json.dumps({"event": "backtest_error", "message": str(error)}))
            return _json_response({"error": "Backtest run failed."}, status=500)

    async def _load_series(self, purchase_type: str) -> tuple[list[HogObservation], str]:
        cache_key = _series_cache_key(purchase_type)
        meta_key = _meta_cache_key(purchase_type)
        cached_csv = await self.env.HOG_DATA_CACHE.get(cache_key)
        cached_meta_text = await self.env.HOG_DATA_CACHE.get(meta_key)
        today_utc = _utc_now().date().isoformat()

        if cached_csv and cached_meta_text:
            meta = _decode_meta(cached_meta_text)
            if meta.get("cache_day") == today_utc:
                return (
                    deserialize_cached_daily_series(cached_csv, source=cache_key),
                    str(meta["refreshed_at"]),
                )

        try:
            fresh_series = await self._fetch_direct_hog_history(purchase_type)
            refreshed_at = _utc_now().replace(microsecond=0).isoformat()
            meta = {
                "cache_day": today_utc,
                "refreshed_at": refreshed_at,
                "data_as_of": max(observation.date for observation in fresh_series),
            }
            await self.env.HOG_DATA_CACHE.put(cache_key, serialize_cached_daily_series(fresh_series))
            await self.env.HOG_DATA_CACHE.put(meta_key, json.dumps(meta))
            return fresh_series, refreshed_at
        except Exception:
            if cached_csv and cached_meta_text:
                meta = _decode_meta(cached_meta_text)
                return (
                    deserialize_cached_daily_series(cached_csv, source=f"{cache_key}:stale"),
                    str(meta.get("refreshed_at", _utc_now().replace(microsecond=0).isoformat())),
                )
            raise

    async def _fetch_direct_hog_history(self, purchase_type: str) -> list[HogObservation]:
        final_date = _utc_now().date()
        observed_rows: dict[str, HogObservation] = {}
        for chunk_start, chunk_end in _date_chunks(datetime(2002, 1, 1).date(), final_date):
            for observation in await self._fetch_direct_hog_chunk(
                purchase_type=purchase_type,
                start_date=chunk_start,
                end_date=chunk_end,
            ):
                observed_rows[observation.date] = observation
        if not observed_rows:
            raise ValueError(f"No direct hog prices found for purchase type {purchase_type!r}.")
        return [observed_rows[point_date] for point_date in sorted(observed_rows)]

    async def _fetch_direct_hog_chunk(
        self,
        *,
        purchase_type: str,
        start_date,
        end_date,
    ) -> list[HogObservation]:
        query = urlencode(
            {
                "q": f"report_date={_format_ams_date(start_date)}:{_format_ams_date(end_date)}",
                "allSections": "true",
            }
        )
        response = await fetch(f"{AMS_API_URL}?{query}")
        if response.status >= 400:
            raise ValueError(f"USDA AMS request failed with status {response.status}.")
        payload_text = await response.text()
        payload = json.loads(payload_text)
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


def _json_response(payload: dict[str, object], *, status: int = 200) -> Response:
    return Response(
        json.dumps(payload),
        status=status,
        headers={
            "content-type": "application/json; charset=utf-8",
            "cache-control": "no-store",
        },
    )


def _series_cache_key(purchase_type: str) -> str:
    return f"{CACHE_PREFIX}:{_slugify(purchase_type)}:csv"


def _meta_cache_key(purchase_type: str) -> str:
    return f"{CACHE_PREFIX}:{_slugify(purchase_type)}:meta"


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _decode_meta(raw_value: str) -> dict[str, str]:
    decoded = json.loads(raw_value)
    if not isinstance(decoded, dict):
        raise ValueError("Invalid cache metadata.")
    return decoded


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
