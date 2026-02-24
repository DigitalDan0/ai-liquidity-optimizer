from __future__ import annotations

import re
from typing import Any

from ai_liquidity_optimizer.http import JsonHttpClient
from ai_liquidity_optimizer.models import (
    SynthLpBoundForecast,
    SynthLpProbabilitiesSnapshot,
    SynthLpProbabilityPoint,
    SynthPredictionPercentilesSnapshot,
)


class SynthInsightsClient:
    """Synth Insights API client for LP forecast bounds."""

    def __init__(self, base_url: str, api_key: str, http_client: JsonHttpClient | None = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.http = http_client or JsonHttpClient()

    def get_lp_bounds(self, asset: str, horizon: str = "24h", days: int = 30, limit: int = 20) -> list[SynthLpBoundForecast]:
        url = f"{self.base_url}/insights/lp-bounds"
        payload = self.http.get_json(
            url,
            params={"asset": asset, "horizon": horizon, "days": days, "limit": limit},
            headers={"Authorization": f"Apikey {self.api_key}"},
        )

        raw_items = _extract_data_list(payload)
        forecasts: list[SynthLpBoundForecast] = []
        for item in raw_items:
            forecasts.append(_parse_lp_bound_item(item, horizon=horizon))
        forecasts.sort(key=lambda f: f.width_pct)
        return forecasts

    def get_lp_probabilities(
        self,
        asset: str,
        horizon: str = "24h",
        days: int = 30,
    ) -> SynthLpProbabilitiesSnapshot:
        url = f"{self.base_url}/insights/lp-probabilities"
        payload = self.http.get_json(
            url,
            params={"asset": asset, "horizon": horizon, "days": days},
            headers={"Authorization": f"Apikey {self.api_key}"},
        )
        return _parse_lp_probabilities_payload(payload, asset=asset, horizon=horizon)

    def get_prediction_percentiles(
        self,
        asset: str,
        step_minutes: int = 5,
    ) -> SynthPredictionPercentilesSnapshot:
        """Optional path forecast endpoint used for time-decayed occupancy weighting.

        Docs page: /prediction-percentiles. Endpoint payload is normalized defensively.
        """
        url = f"{self.base_url}/insights/prediction-percentiles"
        payload = self.http.get_json(
            url,
            params={"asset": asset},
            headers={"Authorization": f"Apikey {self.api_key}"},
        )
        return _parse_prediction_percentiles_payload(payload, asset=asset, step_minutes=step_minutes)


def _extract_data_list(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
    raise RuntimeError("Unexpected Synth lp-bounds response format")


def _parse_lp_bound_item(item: dict[str, Any], horizon: str) -> SynthLpBoundForecast:
    interval = item.get("interval") if isinstance(item.get("interval"), dict) else {}

    lower = _coerce_float(item.get("lower_bound"))
    if lower is None:
        lower = _coerce_float(interval.get("lower_bound"))
    upper = _coerce_float(item.get("upper_bound"))
    if upper is None:
        upper = _coerce_float(interval.get("upper_bound"))
    if lower is None or upper is None:
        raise RuntimeError(f"Synth lp-bounds item missing bounds: {item}")

    width_pct = _coerce_float(item.get("width_pct"))
    if width_pct is None:
        width_pct = _coerce_percent(interval.get("full_width"))
    if width_pct is None:
        mid = (lower + upper) / 2.0
        width_pct = ((upper - lower) / mid) * 100.0 if mid > 0 else 0.0

    prob = _coerce_horizon_value(item.get("probability_to_stay_in_interval"), horizon=horizon)
    if prob is None:
        # Some variants nest by "probability_to_stay_in_interval_by_horizon"
        prob = _coerce_horizon_value(item.get("probability_to_stay_in_interval_by_horizon"), horizon=horizon)
    if prob is None:
        raise RuntimeError(f"Synth lp-bounds item missing probability_to_stay_in_interval: {item}")

    expected_time = _coerce_horizon_value(item.get("expected_time_in_interval"), horizon=horizon)
    if expected_time is None:
        expected_time = _coerce_horizon_value(item.get("expected_time_in_interval_minutes"), horizon=horizon)
    if expected_time is None:
        raise RuntimeError(f"Synth lp-bounds item missing expected_time_in_interval: {item}")

    expected_il = _coerce_horizon_value(item.get("expected_impermanent_loss"), horizon=horizon)
    if expected_il is None:
        # Fallback to horizonless scalar if present
        expected_il = _coerce_float(item.get("expected_il"))
    if expected_il is None:
        raise RuntimeError(f"Synth lp-bounds item missing expected_impermanent_loss: {item}")

    return SynthLpBoundForecast(
        width_pct=float(width_pct),
        lower_bound=float(lower),
        upper_bound=float(upper),
        probability_to_stay_in_interval=float(prob),
        expected_time_in_interval_minutes=float(expected_time),
        expected_impermanent_loss=float(expected_il),
    )


def _parse_lp_probabilities_payload(payload: Any, asset: str, horizon: str) -> SynthLpProbabilitiesSnapshot:
    root = payload if isinstance(payload, dict) else {"data": payload}
    container = root.get("data")
    if not isinstance(container, dict):
        if isinstance(payload, dict):
            container = payload
        else:
            raise RuntimeError("Unexpected Synth lp-probabilities response format")

    horizon_payload = None
    if isinstance(container.get(horizon), dict):
        horizon_payload = container[horizon]
    else:
        # docs examples often use "24h"; some variants might use "24"
        for key in _horizon_candidate_keys(horizon):
            if isinstance(container.get(key), dict):
                horizon_payload = container[key]
                break
    if horizon_payload is None and any(k in container for k in ("probability_below", "probability_above")):
        horizon_payload = container
    if horizon_payload is None:
        # If response nests under another key, select first dict containing probabilities.
        for value in container.values():
            if isinstance(value, dict) and any(k in value for k in ("probability_below", "probability_above")):
                horizon_payload = value
                break
    if not isinstance(horizon_payload, dict):
        raise RuntimeError("Could not locate horizon payload in Synth lp-probabilities response")

    below_map = horizon_payload.get("probability_below")
    above_map = horizon_payload.get("probability_above")
    if not isinstance(below_map, dict) and not isinstance(above_map, dict):
        raise RuntimeError("Synth lp-probabilities response missing probability_below/probability_above maps")

    points_by_price: dict[float, SynthLpProbabilityPoint] = {}
    if isinstance(below_map, dict):
        for k, v in below_map.items():
            price = _coerce_float(k)
            prob = _coerce_float(v)
            if price is None or prob is None:
                continue
            points_by_price.setdefault(price, SynthLpProbabilityPoint(price=price)).probability_below = prob
    if isinstance(above_map, dict):
        for k, v in above_map.items():
            price = _coerce_float(k)
            prob = _coerce_float(v)
            if price is None or prob is None:
                continue
            points_by_price.setdefault(price, SynthLpProbabilityPoint(price=price)).probability_above = prob

    points = sorted(points_by_price.values(), key=lambda p: p.price)
    if not points:
        raise RuntimeError("Synth lp-probabilities response did not contain parseable threshold points")

    return SynthLpProbabilitiesSnapshot(
        asset=str(root.get("asset") or asset).upper(),
        horizon=horizon,
        points=points,
        current_price=_coerce_float(root.get("current_price")),
        as_of=str(root.get("as_of") or root.get("asof")) if root.get("as_of") or root.get("asof") else None,
        raw=root,
    )


def _parse_prediction_percentiles_payload(
    payload: Any,
    asset: str,
    step_minutes: int,
) -> SynthPredictionPercentilesSnapshot:
    if not isinstance(payload, dict):
        raise RuntimeError("Unexpected Synth prediction-percentiles response format")
    future = payload.get("forecast_future")
    if not isinstance(future, dict):
        raise RuntimeError("Synth prediction-percentiles missing forecast_future")
    rows = future.get("percentiles")
    if not isinstance(rows, list):
        raise RuntimeError("Synth prediction-percentiles missing forecast_future.percentiles list")

    parsed_rows: list[dict[float, float]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        parsed_row: dict[float, float] = {}
        for k, v in row.items():
            p_key = _coerce_float(k)
            p_val = _coerce_float(v)
            if p_key is None or p_val is None:
                continue
            parsed_row[p_key] = p_val
        if parsed_row:
            parsed_rows.append(parsed_row)
    if not parsed_rows:
        raise RuntimeError("Synth prediction-percentiles contained no parseable percentile rows")

    return SynthPredictionPercentilesSnapshot(
        asset=str(payload.get("asset") or asset).upper(),
        percentiles_by_step=parsed_rows,
        current_price=_coerce_float(payload.get("current_price")),
        step_minutes=step_minutes,
        as_of=str(payload.get("as_of") or payload.get("asof")) if payload.get("as_of") or payload.get("asof") else None,
        raw=payload,
    )


def _coerce_horizon_value(value: Any, horizon: str) -> float | None:
    scalar = _coerce_float(value)
    if scalar is not None:
        return scalar
    if not isinstance(value, dict):
        return None

    for key in _horizon_candidate_keys(horizon):
        if key in value:
            parsed = _coerce_float(value.get(key))
            if parsed is not None:
                return parsed

    # fall back to first parseable numeric value
    for candidate in value.values():
        parsed = _coerce_float(candidate)
        if parsed is not None:
            return parsed
    return None


def _horizon_candidate_keys(horizon: str) -> list[str]:
    keys = [horizon]
    m = re.fullmatch(r"(\d+)([hd])", horizon.strip().lower())
    if m:
        qty = int(m.group(1))
        unit = m.group(2)
        keys.append(str(qty))
        if unit == "d":
            keys.append(f"{qty}d")
            keys.append(str(qty * 24))
            keys.append(f"{qty * 24}h")
        elif unit == "h":
            keys.append(f"{qty}h")
    return list(dict.fromkeys(keys))


def _coerce_percent(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if text.endswith("%"):
            return _coerce_float(text[:-1])
        return _coerce_float(text)
    return None


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None
