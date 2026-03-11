from __future__ import annotations

from typing import Iterable

from ai_liquidity_optimizer.compat import dataclass
from ai_liquidity_optimizer.models import (
    MeteoraPoolSnapshot,
    SynthHorizonState,
    SynthLpBoundForecast,
    SynthMarketState,
    SynthPredictionPercentilesSnapshot,
)
from ai_liquidity_optimizer.strategy.ev import EvLpScorer, clamp
from ai_liquidity_optimizer.strategy.scoring import StrategyScorer, horizon_to_minutes


@dataclass(slots=True)
class SynthFusionConfig:
    horizons: list[str]
    range_max_center_drift_ratio: float
    trend_min_center_drift_ratio: float
    trend_min_onesided_prob: float
    min_agreement_score: float
    uncertain_width_expansion: float
    entry_conflict_threshold: float
    size_low_confidence: float
    size_medium_confidence: float
    size_full_confidence: float


def build_synthetic_forecasts_from_prediction_percentiles(
    *,
    prediction_percentiles: SynthPredictionPercentilesSnapshot | None,
    horizon_minutes: int,
    current_price: float | None = None,
    coverage_levels: list[float] | None = None,
) -> list[SynthLpBoundForecast]:
    if prediction_percentiles is None or not prediction_percentiles.percentiles_by_step:
        return []

    target_minutes = max(int(horizon_minutes), int(prediction_percentiles.step_minutes))
    parsed_rows: list[tuple[float, list[tuple[float, float]]]] = []
    for step_index, row in enumerate(prediction_percentiles.percentiles_by_step):
        t_minutes = float(step_index * prediction_percentiles.step_minutes)
        if t_minutes > float(target_minutes):
            break
        quantiles = _parse_prediction_quantiles(row)
        if not quantiles:
            continue
        parsed_rows.append((t_minutes, quantiles))

    if len(parsed_rows) > 1 and parsed_rows[0][0] <= 0.0:
        parsed_rows = parsed_rows[1:]
    if not parsed_rows:
        return []

    terminal_points = parsed_rows[-1][1]
    reference_price = float(current_price or prediction_percentiles.current_price or _price_at_quantile(terminal_points, 0.5))
    if reference_price <= 0:
        reference_price = _price_at_quantile(terminal_points, 0.5)
    if reference_price <= 0:
        return []

    coverages = coverage_levels or [0.55, 0.70, 0.85]
    forecasts: list[SynthLpBoundForecast] = []
    covered_minutes = max(parsed_rows[-1][0], float(prediction_percentiles.step_minutes))
    for coverage in coverages:
        coverage = clamp(float(coverage), 0.05, 0.98)
        lower_q = max(0.0, (1.0 - coverage) * 0.5)
        upper_q = min(1.0, 1.0 - lower_q)
        lower = _price_at_quantile(terminal_points, lower_q)
        upper = _price_at_quantile(terminal_points, upper_q)
        if lower <= 0 or upper <= lower:
            continue

        occupancy_values = []
        for _t_minutes, points in parsed_rows:
            p_in = clamp(
                _cdf_at_price(points, upper) - _cdf_at_price(points, lower),
                0.0,
                1.0,
            )
            occupancy_values.append(p_in)
        if not occupancy_values:
            continue

        avg_occupancy = sum(occupancy_values) / float(len(occupancy_values))
        terminal_occupancy = clamp(
            _cdf_at_price(terminal_points, upper) - _cdf_at_price(terminal_points, lower),
            0.0,
            1.0,
        )
        probability_to_stay = clamp(0.65 * terminal_occupancy + 0.35 * avg_occupancy, 0.01, 0.995)
        expected_time_in_interval_minutes = clamp(avg_occupancy * covered_minutes, 0.0, float(target_minutes))
        mid_price = max((lower + upper) * 0.5, 1e-9)
        width_pct = max(((upper - lower) / mid_price) * 100.0, 0.05)
        out_prob = 1.0 - avg_occupancy
        expected_il = clamp(out_prob * (0.0005 + 0.0025 / max(width_pct, 0.25)), 0.00005, 0.02)
        forecasts.append(
            SynthLpBoundForecast(
                width_pct=width_pct,
                lower_bound=lower,
                upper_bound=upper,
                probability_to_stay_in_interval=probability_to_stay,
                expected_time_in_interval_minutes=expected_time_in_interval_minutes,
                expected_impermanent_loss=expected_il,
            )
        )

    forecasts.sort(key=lambda forecast: (forecast.width_pct, -forecast.probability_to_stay_in_interval))
    return forecasts


def build_synth_market_state(
    *,
    representative_pool: MeteoraPoolSnapshot,
    forecasts_by_horizon: dict[str, list[SynthLpBoundForecast]],
    prediction_percentiles: SynthPredictionPercentilesSnapshot | None,
    scorer: StrategyScorer,
    ev_scorer: EvLpScorer,
    config: SynthFusionConfig,
) -> SynthMarketState | None:
    if not config.horizons:
        return None

    states: dict[str, SynthHorizonState] = {}
    pool_spot = representative_pool.current_price_sol_usdc()

    for horizon in config.horizons:
        forecasts = forecasts_by_horizon.get(horizon) or []
        if not forecasts:
            continue
        ranked = scorer.rank_candidates(
            forecasts=forecasts,
            pool=representative_pool,
            horizon=horizon,
            max_candidates=1,
        )
        chosen = ranked.chosen
        forecast = chosen.forecast
        spot = pool_spot or forecast.mid_price
        occupancy_fraction = clamp(
            forecast.expected_time_in_interval_minutes / max(float(horizon_to_minutes(horizon)), 1.0),
            0.0,
            1.0,
        )
        risk = ev_scorer.path_drift_risk_metrics(
            pool=representative_pool,
            forecast=forecast,
            prediction_percentiles=prediction_percentiles,
            fallback_range_occupancy=occupancy_fraction,
        )
        center = forecast.mid_price
        signed_bps = 0.0
        abs_bps = 0.0
        if spot > 0:
            signed_bps = ((center - spot) / spot) * 10_000.0
            abs_bps = abs(signed_bps)
        reentry_probability = ev_scorer.compute_recovery_probability(
            range_active_occupancy_15m=occupancy_fraction,
            one_sided_break_prob=float(risk.get("one_sided_break_prob") or 0.0),
            directional_confidence=float(risk.get("directional_confidence") or 0.0),
        )
        states[horizon] = SynthHorizonState(
            horizon=horizon,
            forecast=forecast,
            score=float(chosen.score),
            center_price=center,
            center_drift_bps=abs_bps,
            center_drift_signed_bps=signed_bps,
            width_pct=float(forecast.width_pct),
            occupancy_fraction=occupancy_fraction,
            probability_to_stay=float(forecast.probability_to_stay_in_interval),
            expected_il=float(forecast.expected_impermanent_loss),
            out_of_range_prob_15m=float(risk.get("out_of_range_prob_15m") or 0.0),
            one_sided_break_prob=float(risk.get("one_sided_break_prob") or 0.0),
            directional_confidence=float(risk.get("directional_confidence") or 0.0),
            reentry_probability=float(reentry_probability),
            expected_out_of_range_minutes_15m=float(risk.get("expected_out_of_range_minutes_15m") or 0.0),
            current_price=spot,
        )

    if not states:
        return None

    short_state = _pick_horizon_state(states, preferred="15m")
    medium_state = _pick_horizon_state(states, preferred="1h", fallback_after=short_state)
    long_state = _pick_horizon_state(states, preferred="4h", fallback_after=medium_state or short_state)

    short_signal = _signal_ratio(short_state)
    medium_signal = _signal_ratio(medium_state or short_state)
    agreement_score = _agreement_score(short_signal, medium_signal)
    conflict_score = 1.0 - agreement_score
    width_term_expansion = _width_term_expansion(short_state, long_state or medium_state)
    uncertainty_score = clamp(
        0.60 * conflict_score + 0.40 * max(0.0, width_term_expansion),
        0.0,
        1.0,
    )

    max_onesided = max(
        float((short_state or medium_state).one_sided_break_prob if (short_state or medium_state) else 0.0),
        float((medium_state or short_state).one_sided_break_prob if (medium_state or short_state) else 0.0),
    )
    range_like = (
        max(abs(short_signal), abs(medium_signal)) <= float(config.range_max_center_drift_ratio)
        and min(
            float((short_state or medium_state).probability_to_stay if (short_state or medium_state) else 0.0),
            float((medium_state or short_state).probability_to_stay if (medium_state or short_state) else 0.0),
        )
        >= 0.45
        and max_onesided <= float(config.trend_min_onesided_prob)
    )
    trend_like = (
        agreement_score >= float(config.min_agreement_score)
        and max(abs(short_signal), abs(medium_signal)) >= float(config.trend_min_center_drift_ratio)
        and max_onesided >= float(config.trend_min_onesided_prob)
    )
    if range_like:
        market_regime = "range"
    elif trend_like:
        market_regime = "trend_up" if (short_signal + medium_signal) >= 0 else "trend_down"
    else:
        market_regime = "uncertain"

    regime_confidence = _regime_confidence(
        market_regime=market_regime,
        short_state=short_state,
        medium_state=medium_state or short_state,
        agreement_score=agreement_score,
        uncertainty_score=uncertainty_score,
        signal_strength=max(abs(short_signal), abs(medium_signal)),
    )
    fused_center_price = _fused_center_price(
        market_regime=market_regime,
        short_state=short_state,
        medium_state=medium_state or short_state,
    )
    fused_width_pct = _fused_width_pct(
        market_regime=market_regime,
        short_state=short_state,
        medium_state=medium_state or short_state,
        long_state=long_state,
    )
    size_multiplier = _size_multiplier(
        market_regime=market_regime,
        regime_confidence=regime_confidence,
        agreement_score=agreement_score,
        uncertainty_score=uncertainty_score,
        config=config,
    )

    return SynthMarketState(
        horizons=states,
        market_regime=market_regime,
        regime_confidence=regime_confidence,
        horizon_agreement_score=agreement_score,
        short_medium_conflict_score=conflict_score,
        width_term_expansion=width_term_expansion,
        uncertainty_score=uncertainty_score,
        fused_center_price=fused_center_price,
        fused_width_pct=fused_width_pct,
        reentry_prob_15m=states.get("15m").reentry_probability if states.get("15m") is not None else None,
        reentry_prob_1h=states.get("1h").reentry_probability if states.get("1h") is not None else None,
        size_multiplier=size_multiplier,
        current_price=pool_spot,
    )


def build_candidate_ladder(market_state: SynthMarketState) -> list[tuple[str, SynthLpBoundForecast]]:
    short_state = market_state.horizons.get("15m") or next(iter(market_state.horizons.values()))
    medium_state = market_state.horizons.get("1h") or short_state
    long_state = market_state.horizons.get("4h") or market_state.horizons.get("24h") or medium_state

    ladder: list[tuple[str, SynthLpBoundForecast]] = []
    ladder.append(("short_anchor", short_state.forecast))
    if medium_state is not short_state:
        ladder.append(("medium_anchor", medium_state.forecast))
    if long_state is not None and long_state is not medium_state:
        ladder.append(("defensive_anchor", long_state.forecast))

    sources = [short_state, medium_state]
    fused = synthesize_forecast(
        center_price=market_state.fused_center_price,
        width_pct=market_state.fused_width_pct,
        sources=sources,
    )
    ladder.append(("fused_symmetric", fused))

    direction = 0.0
    if market_state.market_regime == "trend_up":
        direction = 1.0
    elif market_state.market_regime == "trend_down":
        direction = -1.0
    skew_shift = min(0.0025, max(0.0005, market_state.fused_width_pct / 100.0 * 0.18))
    trend_center = market_state.fused_center_price * (1.0 + direction * skew_shift)
    trend_width = market_state.fused_width_pct * (1.10 if direction else 1.05)
    ladder.append(
        (
            "trend_skewed",
            synthesize_forecast(center_price=trend_center, width_pct=trend_width, sources=sources),
        )
    )

    defensive_sources = [medium_state]
    if long_state is not None:
        defensive_sources.append(long_state)
    defensive_center = _weighted_mean([s.center_price for s in defensive_sources], [0.55, 0.45][: len(defensive_sources)])
    defensive_width = max(
        market_state.fused_width_pct * 1.25,
        medium_state.width_pct,
        long_state.width_pct if long_state is not None else 0.0,
    )
    ladder.append(
        (
            "defensive_wide",
            synthesize_forecast(center_price=defensive_center, width_pct=defensive_width, sources=defensive_sources),
        )
    )
    return _dedupe_forecast_ladder(ladder)


def synthesize_forecast(
    *,
    center_price: float,
    width_pct: float,
    sources: Iterable[SynthHorizonState],
) -> SynthLpBoundForecast:
    source_list = [s for s in sources if s is not None]
    if not source_list:
        raise RuntimeError("Cannot synthesize forecast without source horizon states")
    center_price = max(float(center_price), 1e-9)
    width_pct = max(float(width_pct), 0.05)
    half_width = center_price * (width_pct / 200.0)
    lower = max(center_price - half_width, 1e-9)
    upper = max(center_price + half_width, lower * (1.0 + 1e-9))
    base_width = max(_weighted_mean([s.width_pct for s in source_list], _default_weights(len(source_list))), 1e-9)
    width_scale = clamp(width_pct / base_width, 0.50, 1.60)
    base_prob = _weighted_mean([s.probability_to_stay for s in source_list], _default_weights(len(source_list)))
    base_time = _weighted_mean(
        [s.forecast.expected_time_in_interval_minutes for s in source_list],
        _default_weights(len(source_list)),
    )
    base_il = _weighted_mean([s.expected_il for s in source_list], _default_weights(len(source_list)))
    stay_prob = clamp(base_prob * (width_scale ** 0.45), 0.01, 0.995)
    expected_time = max(0.0, base_time * clamp(width_scale ** 0.55, 0.70, 1.75))
    expected_il = max(0.0, base_il * clamp((1.0 / width_scale) ** 0.35, 0.70, 1.35))
    return SynthLpBoundForecast(
        width_pct=width_pct,
        lower_bound=lower,
        upper_bound=upper,
        probability_to_stay_in_interval=stay_prob,
        expected_time_in_interval_minutes=expected_time,
        expected_impermanent_loss=expected_il,
    )


def _pick_horizon_state(
    states: dict[str, SynthHorizonState],
    *,
    preferred: str,
    fallback_after: SynthHorizonState | None = None,
) -> SynthHorizonState:
    if preferred in states:
        return states[preferred]
    if fallback_after is not None:
        return fallback_after
    ordered = sorted(states.values(), key=lambda s: horizon_to_minutes(s.horizon))
    return ordered[0]


def _signal_ratio(state: SynthHorizonState) -> float:
    width_bps = max(state.width_pct * 100.0, 1e-9)
    return float(state.center_drift_signed_bps) / width_bps


def _agreement_score(short_signal: float, medium_signal: float) -> float:
    same_sign = 1.0 if short_signal * medium_signal >= 0 else 0.0
    magnitude_gap = abs(short_signal - medium_signal)
    magnitude_match = 1.0 - min(1.0, magnitude_gap / 1.25)
    return clamp((0.60 * same_sign) + (0.40 * magnitude_match), 0.0, 1.0)


def _width_term_expansion(
    short_state: SynthHorizonState,
    long_state: SynthHorizonState | None,
) -> float:
    if long_state is None or short_state.width_pct <= 0:
        return 0.0
    return (float(long_state.width_pct) - float(short_state.width_pct)) / max(float(short_state.width_pct), 1e-9)


def _regime_confidence(
    *,
    market_regime: str,
    short_state: SynthHorizonState,
    medium_state: SynthHorizonState,
    agreement_score: float,
    uncertainty_score: float,
    signal_strength: float,
) -> float:
    if market_regime == "range":
        return clamp(
            0.40 * min(short_state.probability_to_stay, medium_state.probability_to_stay)
            + 0.35 * agreement_score
            + 0.25 * (1.0 - max(short_state.one_sided_break_prob, medium_state.one_sided_break_prob)),
            0.0,
            1.0,
        )
    if market_regime.startswith("trend"):
        return clamp(
            0.35 * agreement_score
            + 0.35 * max(short_state.one_sided_break_prob, medium_state.one_sided_break_prob)
            + 0.30 * min(1.0, signal_strength / 0.75),
            0.0,
            1.0,
        )
    return clamp(0.50 * (1.0 - uncertainty_score) + 0.50 * agreement_score, 0.0, 0.70)


def _fused_center_price(
    *,
    market_regime: str,
    short_state: SynthHorizonState,
    medium_state: SynthHorizonState,
) -> float:
    if market_regime == "range":
        return _weighted_mean([short_state.center_price, medium_state.center_price], [0.65, 0.35])
    if market_regime.startswith("trend"):
        return _weighted_mean([short_state.center_price, medium_state.center_price], [0.40, 0.60])
    return _weighted_mean([short_state.center_price, medium_state.center_price], [0.50, 0.50])


def _fused_width_pct(
    *,
    market_regime: str,
    short_state: SynthHorizonState,
    medium_state: SynthHorizonState,
    long_state: SynthHorizonState | None,
) -> float:
    if market_regime == "range":
        return max(float(short_state.width_pct), float(medium_state.width_pct) * 0.80)
    if market_regime.startswith("trend"):
        long_width = float(long_state.width_pct) if long_state is not None else float(medium_state.width_pct) * 1.15
        return max(float(medium_state.width_pct), long_width * 0.85)
    long_width = float(long_state.width_pct) if long_state is not None else float(medium_state.width_pct) * 1.30
    return max(float(medium_state.width_pct) * 1.15, long_width)


def _size_multiplier(
    *,
    market_regime: str,
    regime_confidence: float,
    agreement_score: float,
    uncertainty_score: float,
    config: SynthFusionConfig,
) -> float:
    if market_regime == "uncertain" and uncertainty_score >= 0.50:
        return 0.0
    if (
        regime_confidence >= float(config.size_full_confidence)
        and agreement_score >= float(config.min_agreement_score)
        and uncertainty_score <= float(config.uncertain_width_expansion)
    ):
        return 1.0
    if regime_confidence >= float(config.size_medium_confidence):
        return 0.66
    if regime_confidence >= float(config.size_low_confidence):
        return 0.33
    return 0.0


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    total_weight = sum(weights)
    if total_weight <= 0:
        return float(values[0] if values else 0.0)
    return sum(v * w for v, w in zip(values, weights)) / total_weight


def _default_weights(count: int) -> list[float]:
    if count <= 1:
        return [1.0]
    if count == 2:
        return [0.60, 0.40]
    return [1.0 / float(count)] * count


def _dedupe_forecast_ladder(
    ladder: list[tuple[str, SynthLpBoundForecast]],
) -> list[tuple[str, SynthLpBoundForecast]]:
    deduped: list[tuple[str, SynthLpBoundForecast]] = []
    seen: set[tuple[float, float]] = set()
    for family, forecast in ladder:
        key = (round(forecast.lower_bound, 6), round(forecast.upper_bound, 6))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((family, forecast))
    return deduped


def _parse_prediction_quantiles(row: dict[float, float]) -> list[tuple[float, float]]:
    parsed: list[tuple[float, float]] = []
    for raw_p, raw_v in row.items():
        try:
            p = float(raw_p)
            value = float(raw_v)
        except (TypeError, ValueError):
            continue
        if value <= 0:
            continue
        parsed.append((p, value))
    if len(parsed) < 2:
        return []
    parsed.sort(key=lambda item: item[0])
    scale = 1.0 if parsed[-1][0] <= 1.0 + 1e-9 else 100.0
    normalized: list[tuple[float, float]] = []
    last_p = -1.0
    last_value = 0.0
    for raw_p, value in parsed:
        p = raw_p / scale
        if p < 0.0 or p > 1.0 or p <= last_p + 1e-12:
            continue
        monotonic_value = max(value, last_value)
        normalized.append((p, monotonic_value))
        last_p = p
        last_value = monotonic_value
    return normalized if len(normalized) >= 2 else []


def _cdf_at_price(points: list[tuple[float, float]], price: float) -> float:
    if not points or price <= 0:
        return 0.0
    if price <= points[0][1]:
        return clamp(points[0][0], 0.0, 1.0)
    if price >= points[-1][1]:
        return clamp(points[-1][0], 0.0, 1.0)
    for idx in range(len(points) - 1):
        p0, q0 = points[idx]
        p1, q1 = points[idx + 1]
        if q1 <= q0:
            continue
        if q0 <= price <= q1:
            t = (price - q0) / max(q1 - q0, 1e-12)
            return clamp(p0 + t * (p1 - p0), 0.0, 1.0)
    return clamp(points[-1][0], 0.0, 1.0)


def _price_at_quantile(points: list[tuple[float, float]], target_p: float) -> float:
    if not points:
        return 0.0
    target = clamp(float(target_p), 0.0, 1.0)
    if target <= points[0][0]:
        return points[0][1]
    if target >= points[-1][0]:
        return points[-1][1]
    for idx in range(len(points) - 1):
        p0, q0 = points[idx]
        p1, q1 = points[idx + 1]
        if p1 <= p0:
            continue
        if p0 <= target <= p1:
            t = (target - p0) / max(p1 - p0, 1e-12)
            return max(0.0, q0 + t * (q1 - q0))
    return points[-1][1]
