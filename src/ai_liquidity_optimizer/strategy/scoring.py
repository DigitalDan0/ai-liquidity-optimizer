from __future__ import annotations

import re

from ai_liquidity_optimizer.compat import dataclass
from ai_liquidity_optimizer.models import MeteoraPoolSnapshot, ScoredCandidate, StrategyDecision, SynthLpBoundForecast, normalize_fraction


def horizon_to_minutes(horizon: str) -> int:
    text = horizon.strip().lower()
    m = re.fullmatch(r"(\d+)([hd])", text)
    if not m:
        raise ValueError(f"Unsupported horizon format: {horizon!r} (expected like '24h' or '7d')")
    qty = int(m.group(1))
    unit = m.group(2)
    return qty * (60 if unit == "h" else 24 * 60)


@dataclass(slots=True)
class StrategyScorer:
    min_stay_probability: float = 0.02

    def rank_candidates(
        self,
        forecasts: list[SynthLpBoundForecast],
        pool: MeteoraPoolSnapshot,
        horizon: str,
        max_candidates: int | None = None,
    ) -> StrategyDecision:
        horizon_minutes = horizon_to_minutes(horizon)
        base_fee_return = normalize_fraction(pool.fee_return_fraction_24h())

        filtered = [f for f in forecasts if f.probability_to_stay_in_interval >= self.min_stay_probability]
        if not filtered:
            filtered = forecasts[:]
        if max_candidates is not None:
            filtered = filtered[:max_candidates]
        if not filtered:
            raise RuntimeError("No Synth LP bounds candidates available to score")

        ranked = [
            self._score_candidate(forecast=f, horizon_minutes=horizon_minutes, base_fee_return_fraction=base_fee_return)
            for f in filtered
        ]
        ranked.sort(
            key=lambda c: (
                c.score,
                c.forecast.probability_to_stay_in_interval,
                -c.forecast.width_pct,  # prefer narrower range when score/probability tie
            ),
            reverse=True,
        )
        return StrategyDecision(chosen=ranked[0], ranked=ranked, horizon=horizon)

    def _score_candidate(
        self,
        forecast: SynthLpBoundForecast,
        horizon_minutes: int,
        base_fee_return_fraction: float,
    ) -> ScoredCandidate:
        expected_active_fraction = min(max(forecast.expected_time_in_interval_minutes / float(horizon_minutes), 0.0), 1.0)
        stay_prob = min(max(forecast.probability_to_stay_in_interval, 0.0), 1.0)
        # Risk-adjusted fee proxy:
        # - active fraction captures expected time inside range
        # - stay probability acts as confidence that range remains valid for full horizon
        confidence_multiplier = 0.25 + 0.75 * stay_prob
        expected_fee_return_fraction = base_fee_return_fraction * expected_active_fraction * confidence_multiplier
        expected_net_return_fraction = expected_fee_return_fraction - max(forecast.expected_impermanent_loss, 0.0)
        return ScoredCandidate(
            forecast=forecast,
            expected_active_fraction=expected_active_fraction,
            expected_fee_return_fraction=expected_fee_return_fraction,
            confidence_multiplier=confidence_multiplier,
            expected_net_return_fraction=expected_net_return_fraction,
            score=expected_net_return_fraction,
        )


def relative_range_change_bps(
    current_lower: float,
    current_upper: float,
    new_lower: float,
    new_upper: float,
) -> float:
    current_mid = (current_lower + current_upper) / 2.0
    if current_mid <= 0:
        return 10_000.0
    lower_change = abs(new_lower - current_lower) / current_mid
    upper_change = abs(new_upper - current_upper) / current_mid
    return max(lower_change, upper_change) * 10_000.0
