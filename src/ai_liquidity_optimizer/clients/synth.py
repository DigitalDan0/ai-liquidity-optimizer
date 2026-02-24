from __future__ import annotations

from typing import Any

from ai_liquidity_optimizer.http import JsonHttpClient
from ai_liquidity_optimizer.models import SynthLpBoundForecast


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
            forecasts.append(
                SynthLpBoundForecast(
                    width_pct=float(item["width_pct"]),
                    lower_bound=float(item["lower_bound"]),
                    upper_bound=float(item["upper_bound"]),
                    probability_to_stay_in_interval=float(item["probability_to_stay_in_interval"]),
                    expected_time_in_interval_minutes=float(item["expected_time_in_interval"]),
                    expected_impermanent_loss=float(item["expected_impermanent_loss"]),
                )
            )
        forecasts.sort(key=lambda f: f.width_pct)
        return forecasts


def _extract_data_list(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
    raise RuntimeError("Unexpected Synth lp-bounds response format")

