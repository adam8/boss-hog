from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Mapping, Sequence

from hog_price_baseline import (
    CORE_FUNDAMENTALS_FEATURE_PACK,
    DEFAULT_CACHE_PATH,
    DEFAULT_PURCHASE_TYPE,
    PRICE_ONLY_FEATURE_PACK,
    BacktestSummary,
    FEATURE_PACKS,
    HogObservation,
    aggregate_monthly_average,
    run_monthly_backtest,
)


DEFAULT_MAX_OBSERVATIONS = 240
DEFAULT_INITIAL_WINDOW = 120
DEFAULT_RANDOM_CELLS = 20
DEFAULT_SEED = 11
MIN_MAX_OBSERVATIONS = 120
MAX_MAX_OBSERVATIONS = 300
MIN_INITIAL_WINDOW = 60
MAX_INITIAL_WINDOW = 180
MIN_RANDOM_CELLS = 10
MAX_RANDOM_CELLS = 40
MIN_SEED = 0
MAX_SEED = 9999


@dataclass(frozen=True)
class BacktestRequest:
    feature_pack: str = PRICE_ONLY_FEATURE_PACK
    max_observations: int = DEFAULT_MAX_OBSERVATIONS
    initial_window: int = DEFAULT_INITIAL_WINDOW
    random_cells: int = DEFAULT_RANDOM_CELLS
    seed: int = DEFAULT_SEED

    @classmethod
    def from_mapping(cls, raw_params: Mapping[str, object]) -> "BacktestRequest":
        feature_pack = _read_feature_pack(raw_params.get("feature_pack"), PRICE_ONLY_FEATURE_PACK)
        max_observations = _read_int(
            raw_params.get("max_observations"),
            "max_observations",
            DEFAULT_MAX_OBSERVATIONS,
            minimum=MIN_MAX_OBSERVATIONS,
            maximum=MAX_MAX_OBSERVATIONS,
        )
        initial_window = _read_int(
            raw_params.get("initial_window"),
            "initial_window",
            DEFAULT_INITIAL_WINDOW,
            minimum=MIN_INITIAL_WINDOW,
            maximum=MAX_INITIAL_WINDOW,
        )
        random_cells = _read_int(
            raw_params.get("random_cells"),
            "random_cells",
            DEFAULT_RANDOM_CELLS,
            minimum=MIN_RANDOM_CELLS,
            maximum=MAX_RANDOM_CELLS,
        )
        seed = _read_int(
            raw_params.get("seed"),
            "seed",
            DEFAULT_SEED,
            minimum=MIN_SEED,
            maximum=MAX_SEED,
        )
        if initial_window >= max_observations:
            raise ValueError("initial_window must be smaller than max_observations.")
        return cls(
            feature_pack=feature_pack,
            max_observations=max_observations,
            initial_window=initial_window,
            random_cells=random_cells,
            seed=seed,
        )

    def as_payload(self) -> dict[str, int | str]:
        return {
            "feature_pack": self.feature_pack,
            "max_observations": self.max_observations,
            "initial_window": self.initial_window,
            "random_cells": self.random_cells,
            "seed": self.seed,
        }


def run_request_against_monthly_series(
    series: Sequence[HogObservation],
    request: BacktestRequest,
    *,
    series_name: str = DEFAULT_PURCHASE_TYPE,
    source_path: str | None = None,
) -> BacktestSummary:
    monthly_series = list(series)
    if request.max_observations:
        monthly_series = monthly_series[-request.max_observations :]
    summary = run_monthly_backtest(
        monthly_series,
        series_name=series_name,
        source_path=DEFAULT_CACHE_PATH if source_path is None else DEFAULT_CACHE_PATH.__class__(source_path),
        initial_window=request.initial_window,
        random_cells=request.random_cells,
        seed=request.seed,
        feature_pack=request.feature_pack,
    )
    return summary


def build_payload(
    request: BacktestRequest,
    summary: BacktestSummary,
    *,
    purchase_type: str = DEFAULT_PURCHASE_TYPE,
    source: str = "USDA AMS direct hog avg_net_price",
    data_as_of: str,
    refreshed_at: str | None = None,
) -> dict[str, object]:
    last_index = len(summary.predictions) - 1
    return {
        "request": request.as_payload(),
        "data_status": {
            "source": source,
            "purchase_type": purchase_type,
            "data_as_of": data_as_of,
            "refreshed_at": refreshed_at or _utc_now_iso(),
        },
        "metrics": {
            "prediction_actual_correlation": round(summary.correlation, 6),
            "directional_accuracy": round(summary.directional_accuracy, 6),
            "average_ex_ante_fit": round(summary.average_fit, 6),
            "monthly_observation_count": summary.observation_count,
            "out_of_sample_prediction_count": len(summary.predictions),
            "feature_pack": summary.feature_pack,
        },
        "final_month": {
            "prediction_date": summary.prediction_dates[last_index],
            "predicted_next_month_log_return": round(summary.predictions[last_index], 6),
            "actual_next_month_log_return": round(summary.actuals[last_index], 6),
            "starting_month_price_average": round(summary.current_prices[last_index], 4),
            "predicted_next_month_price_average": round(summary.implied_next_prices[last_index], 4),
            "actual_next_month_price_average": round(summary.next_prices[last_index], 4),
        },
        "average_feature_importance": _sorted_importances(summary.average_variable_importance),
        "average_exogenous_importance": _sorted_importances(summary.average_exogenous_variable_importance),
    }


def latest_observation_date(series: Sequence[HogObservation]) -> str:
    if not series:
        raise ValueError("Series must contain at least one observation.")
    return max(observation.date for observation in series)


def aggregate_request_from_daily_series(
    series: Sequence[HogObservation],
    request: BacktestRequest,
    *,
    purchase_type: str = DEFAULT_PURCHASE_TYPE,
    source: str = "USDA AMS direct hog avg_net_price",
    refreshed_at: str | None = None,
) -> dict[str, object]:
    monthly_series = aggregate_monthly_average(series)
    summary = run_request_against_monthly_series(
        monthly_series,
        request,
        series_name=purchase_type,
        source_path=str(DEFAULT_CACHE_PATH),
    )
    return build_payload(
        request,
        summary,
        purchase_type=purchase_type,
        source=source,
        data_as_of=latest_observation_date(series),
        refreshed_at=refreshed_at,
    )


def _sorted_importances(importances: Mapping[str, float]) -> list[dict[str, float | str]]:
    return [
        {"feature": feature_name, "importance": round(importance, 6)}
        for feature_name, importance in sorted(
            importances.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    ]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _read_feature_pack(raw_value: object, default: str) -> str:
    if raw_value is None:
        return default
    value = _normalize_scalar(raw_value)
    if not isinstance(value, str):
        raise ValueError("feature_pack must be a string.")
    feature_pack = value.strip()
    if feature_pack not in FEATURE_PACKS:
        valid = ", ".join(sorted(FEATURE_PACKS))
        raise ValueError(f"feature_pack must be one of: {valid}.")
    return feature_pack


def _read_int(
    raw_value: object,
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if raw_value is None:
        return default
    value = _normalize_scalar(raw_value)
    try:
        parsed = int(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if parsed < minimum or parsed > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")
    return parsed


def _normalize_scalar(raw_value: object) -> object:
    if isinstance(raw_value, (list, tuple)):
        if not raw_value:
            return None
        return raw_value[0]
    return raw_value
