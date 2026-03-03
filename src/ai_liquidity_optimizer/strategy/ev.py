from __future__ import annotations

import math
from statistics import median

from ai_liquidity_optimizer.compat import dataclass
from ai_liquidity_optimizer.models import (
    ActivePositionState,
    EvComponentBreakdown,
    EvScoredCandidate,
    MeteoraPoolSnapshot,
    PoolPreScore,
    SynthLpBoundForecast,
    SynthPredictionPercentilesSnapshot,
    WeightedBinPlan,
)
from ai_liquidity_optimizer.strategy.bin_weights import compute_active_occupancy_metrics_from_percentiles
from ai_liquidity_optimizer.strategy.scoring import horizon_to_minutes, relative_range_change_bps


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass(slots=True)
class EvLpScorer:
    ev_horizon_minutes: int = 15
    rebalance_cost_usd: float = 0.50
    pool_switch_extra_cost_usd: float = 1.00
    min_ev_improvement_usd: float = 0.25
    ev_percentile_decay_half_life_minutes: int = 15
    ev_concentration_gamma: float = 0.6
    ev_concentration_min: float = 0.70
    ev_concentration_max: float = 2.25
    ev_capture_kappa: float = 0.7
    ev_capture_min: float = 0.60
    ev_capture_max: float = 1.05
    ev_capture_eps: float = 1e-9
    ev_oor_penalty_enabled: bool = True
    ev_oor_penalize_hold_only: bool = True
    ev_oor_deadband_bps: float = 5.0
    ev_oor_ref_bps: float = 50.0
    ev_oor_base_penalty_fraction_15m: float = 0.00025
    ev_oor_max_penalty_fraction_15m: float = 0.0015
    ev_oor_persistence_step: float = 0.20
    ev_oor_persistence_cap_cycles: int = 12
    ev_il_drift_alpha: float = 0.60
    ev_il_oor_beta: float = 1.00
    ev_il_onesided_gamma: float = 0.75
    ev_il_persistence_delta: float = 0.50
    ev_il_mult_min: float = 1.0
    ev_il_mult_max: float = 3.0
    ev_il_drift_ref_bps: float = 50.0
    ev_il_drift_horizon_minutes: int = 30
    min_stay_probability: float = 0.0

    def pool_pre_rank(
        self,
        pools: list[MeteoraPoolSnapshot],
        limit: int,
    ) -> list[PoolPreScore]:
        if not pools:
            return []
        rows = []
        for pool in pools:
            tvl = max(pool.tvl_usd(), 1.0)
            realized_fee_rate = pool.realized_fee_rate_15m_fraction_proxy()
            vol_1h = pool.volume_window("1h")
            vol_24h = pool.volume_window("24h")
            volume_utilization = 0.7 * (vol_1h / tvl if vol_1h > 0 else 0.0) + 0.3 * ((vol_24h / tvl) / 24.0 if vol_24h > 0 else 0.0)
            absolute_tvl = math.log1p(max(pool.tvl_usd(), 0.0))
            bin_step_fit = 1.0 / max(pool.bin_step_bps or 10.0, 1e-9)
            total_fee_pct = max(0.0, (pool.base_fee_pct or 0.0) + (pool.dynamic_fee_pct or 0.0))
            fee_config_signal = 1.0 / (1.0 + total_fee_pct)
            rows.append(
                {
                    "pool": pool,
                    "realized_fee_rate": realized_fee_rate,
                    "volume_utilization": volume_utilization,
                    "absolute_tvl": absolute_tvl,
                    "bin_step_fit": bin_step_fit,
                    "fee_config_signal": fee_config_signal,
                }
            )

        for key in ("realized_fee_rate", "volume_utilization", "absolute_tvl", "bin_step_fit", "fee_config_signal"):
            _apply_rank_normalization(rows, key)

        scores: list[PoolPreScore] = []
        for row in rows:
            score = (
                0.40 * row["realized_fee_rate_rank"]
                + 0.25 * row["volume_utilization_rank"]
                + 0.20 * row["absolute_tvl_rank"]
                + 0.10 * row["bin_step_fit_rank"]
                + 0.05 * row["fee_config_signal_rank"]
            )
            pool = row["pool"]
            scores.append(
                PoolPreScore(
                    pool_address=pool.address,
                    pool_name=pool.name,
                    score=float(score),
                    components={
                        "realized_fee_rate_rank": float(row["realized_fee_rate_rank"]),
                        "volume_utilization_rank": float(row["volume_utilization_rank"]),
                        "absolute_tvl_rank": float(row["absolute_tvl_rank"]),
                        "bin_step_fit_rank": float(row["bin_step_fit_rank"]),
                        "fee_config_signal_rank": float(row["fee_config_signal_rank"]),
                        "realized_fee_rate_15m_fraction": float(row["realized_fee_rate"]),
                        "volume_utilization": float(row["volume_utilization"]),
                        "tvl_usd": float(pool.tvl_usd()),
                        "bin_step_bps": float(pool.bin_step_bps or 0.0),
                    },
                )
            )

        scores.sort(key=lambda s: s.score, reverse=True)
        return scores[: max(1, limit)]

    def compute_capital_usd(self, pool: MeteoraPoolSnapshot, deposit_sol_amount: float, deposit_usdc_amount: float) -> float:
        spot = max(pool.current_price_sol_usdc(), 0.0)
        return max(0.0, float(deposit_sol_amount) * spot + float(deposit_usdc_amount))

    @staticmethod
    def out_of_range_bps(*, spot: float, lower: float, upper: float) -> float:
        lo = min(lower, upper)
        hi = max(lower, upper)
        if spot <= 0 or lo <= 0 or hi <= 0:
            return 0.0
        if lo <= spot <= hi:
            return 0.0
        nearest = lo if spot < lo else hi
        if nearest <= 0:
            return 0.0
        # Fixed reference: distance from nearest bound, scaled in bps.
        return abs(spot - nearest) / nearest * 10_000.0

    def out_of_range_penalty_fraction_15m(self, *, out_of_range_bps: float, out_of_range_cycles: int) -> float:
        if not self.ev_oor_penalty_enabled:
            return 0.0
        if out_of_range_bps <= self.ev_oor_deadband_bps:
            return 0.0
        z = max(0.0, out_of_range_bps - self.ev_oor_deadband_bps) / max(self.ev_oor_ref_bps, 1e-9)
        base_frac = min(self.ev_oor_max_penalty_fraction_15m, z * self.ev_oor_base_penalty_fraction_15m)
        capped_cycles = max(0, min(int(out_of_range_cycles), int(self.ev_oor_persistence_cap_cycles)))
        persistence_mult = 1.0 + capped_cycles * self.ev_oor_persistence_step
        return min(self.ev_oor_max_penalty_fraction_15m, base_frac * persistence_mult)

    def compute_recovery_probability(
        self,
        *,
        range_active_occupancy_15m: float,
        one_sided_break_prob: float,
        directional_confidence: float,
    ) -> float:
        recovery = (
            0.50 * (1.0 - one_sided_break_prob)
            + 0.35 * range_active_occupancy_15m
            + 0.15 * (1.0 - directional_confidence)
        )
        return clamp(recovery, 0.0, 1.0)

    def compute_trend_continuation_probability(
        self,
        *,
        one_sided_break_prob: float,
        directional_confidence: float,
    ) -> float:
        trend = 0.70 * one_sided_break_prob + 0.30 * directional_confidence
        return clamp(trend, 0.0, 1.0)

    def concentration_factor(
        self,
        *,
        forecast: SynthLpBoundForecast,
        width_ref_pct: float,
        pool: MeteoraPoolSnapshot,
        bin_step_ref_bps: float,
    ) -> float:
        width = max(forecast.width_pct, 1e-9)
        width_ref = max(width_ref_pct, 1e-9)
        raw_conc = (width_ref / width) ** self.ev_concentration_gamma
        raw_conc = clamp(raw_conc, self.ev_concentration_min, self.ev_concentration_max)

        bin_step = pool.bin_step_bps or bin_step_ref_bps or 10.0
        bin_step_ref = bin_step_ref_bps or 10.0
        bin_step_adj = clamp((max(bin_step_ref, 1e-9) / max(bin_step, 1e-9)) ** 0.15, 0.90, 1.15)
        return max(0.0, raw_conc * bin_step_adj)

    def active_occupancy_15m(
        self,
        *,
        forecast: SynthLpBoundForecast,
        synth_horizon: str,
        weighted_bin_plan: WeightedBinPlan,
        prediction_percentiles: SynthPredictionPercentilesSnapshot | None,
    ) -> tuple[float, float | None, str]:
        if prediction_percentiles is not None:
            metrics = compute_active_occupancy_metrics_from_percentiles(
                bin_edges=weighted_bin_plan.bin_edges,
                prediction_percentiles=prediction_percentiles,
                tau_half_minutes=self.ev_percentile_decay_half_life_minutes,
                max_horizon_minutes=self.ev_horizon_minutes,
                bin_weights=weighted_bin_plan.weights,
            )
            if metrics is not None:
                range_occ = float(metrics.get("range_active_occupancy") or 0.0)
                align = metrics.get("weight_alignment_score")
                align_float = float(align) if align is not None else None
                return clamp(range_occ, 0.0, 1.0), align_float, "prediction_percentiles_range_occupancy"

        # Fallback to terminal mass proxy restricted to the selected range.
        mass_in_range = weighted_bin_plan.diagnostics.mass_in_range
        if mass_in_range > 0:
            return clamp(mass_in_range, 0.0, 1.0), None, "lp_probabilities_terminal_mass"

        # Final fallback to Synth lp-bounds expected active fraction over the Synth horizon.
        horizon_minutes = max(float(horizon_to_minutes(synth_horizon)), 1.0)
        t_frac = forecast.expected_time_in_interval_minutes / horizon_minutes
        return clamp(t_frac, 0.0, 1.0), None, "lp_bounds_expected_time_fraction"

    def path_drift_risk_metrics(
        self,
        *,
        pool: MeteoraPoolSnapshot,
        forecast: SynthLpBoundForecast,
        prediction_percentiles: SynthPredictionPercentilesSnapshot | None,
        fallback_range_occupancy: float,
    ) -> dict[str, float]:
        spot = max(pool.current_price_sol_usdc(), 0.0)
        if prediction_percentiles is None or not prediction_percentiles.percentiles_by_step:
            out_prob = clamp(1.0 - fallback_range_occupancy, 0.0, 1.0)
            mid = max((forecast.lower_bound + forecast.upper_bound) * 0.5, 1e-9)
            drift_bps = abs(spot - mid) / max(spot, 1e-9) * 10_000.0
            directional_conf = clamp(abs(spot - mid) / mid, 0.0, 1.0)
            return {
                "p50_drift_bps": drift_bps,
                "out_of_range_prob_15m": out_prob,
                "one_sided_break_prob": 0.5 * out_prob,
                "directional_confidence": directional_conf,
                "expected_out_of_range_minutes_15m": out_prob * float(self.ev_horizon_minutes),
            }

        ln2 = math.log(2.0)
        half_life = max(float(self.ev_percentile_decay_half_life_minutes), 1.0)
        max_horizon = max(float(self.ev_horizon_minutes), float(self.ev_il_drift_horizon_minutes))
        w_sum = 0.0
        drift_sum = 0.0
        out_sum = 0.0
        one_side_sum = 0.0
        dir_sum = 0.0

        for step_index, row in enumerate(prediction_percentiles.percentiles_by_step):
            if not isinstance(row, dict):
                continue
            t_minutes = float(step_index * prediction_percentiles.step_minutes)
            if t_minutes > max_horizon:
                break
            quantiles = _parse_row_quantiles_normalized(row)
            if len(quantiles) < 2:
                continue
            cdf_lower = _cdf_at_price_from_quantiles(quantiles, forecast.lower_bound)
            cdf_upper = _cdf_at_price_from_quantiles(quantiles, forecast.upper_bound)
            cdf_spot = _cdf_at_price_from_quantiles(quantiles, spot)
            p50 = _price_at_quantile(quantiles, 0.5)
            p_out = clamp(cdf_lower + (1.0 - cdf_upper), 0.0, 1.0)
            p_one_side = clamp(max(cdf_lower, 1.0 - cdf_upper), 0.0, 1.0)
            direction_conf = clamp(abs(1.0 - 2.0 * cdf_spot), 0.0, 1.0)
            drift_bps = abs(p50 - spot) / max(spot, 1e-9) * 10_000.0
            decay = math.exp(-ln2 * (t_minutes / half_life))
            w_sum += decay
            drift_sum += decay * drift_bps
            out_sum += decay * p_out
            one_side_sum += decay * p_one_side
            dir_sum += decay * direction_conf

        if w_sum <= 0:
            return self.path_drift_risk_metrics(
                pool=pool,
                forecast=forecast,
                prediction_percentiles=None,
                fallback_range_occupancy=fallback_range_occupancy,
            )

        p50_drift_bps = drift_sum / w_sum
        out_prob = clamp(out_sum / w_sum, 0.0, 1.0)
        one_sided = clamp(one_side_sum / w_sum, 0.0, 1.0)
        directional_conf = clamp(dir_sum / w_sum, 0.0, 1.0)
        return {
            "p50_drift_bps": p50_drift_bps,
            "out_of_range_prob_15m": out_prob,
            "one_sided_break_prob": one_sided,
            "directional_confidence": directional_conf,
            "expected_out_of_range_minutes_15m": out_prob * float(self.ev_horizon_minutes),
        }

    def il_15m_fraction(
        self,
        *,
        forecast: SynthLpBoundForecast,
        synth_horizon: str,
    ) -> float:
        horizon_minutes = max(float(horizon_to_minutes(synth_horizon)), 1.0)
        scaled = max(forecast.expected_impermanent_loss, 0.0) * (self.ev_horizon_minutes / horizon_minutes)
        return max(0.0, scaled)

    def score_pool_range_ev_15m(
        self,
        *,
        pool: MeteoraPoolSnapshot,
        forecast: SynthLpBoundForecast,
        weighted_bin_plan: WeightedBinPlan,
        synth_horizon: str,
        prediction_percentiles: SynthPredictionPercentilesSnapshot | None,
        deposit_sol_amount: float,
        deposit_usdc_amount: float,
        width_ref_pct: float,
        bin_step_ref_bps: float,
        active_position: ActivePositionState | None,
        range_change_threshold_bps: float,
        apply_rebalance_costs: bool,
        baseline_mode: str | None = None,
        pre_rank_score: float | None = None,
        out_of_range_cycles: int = 0,
        action_type: str = "rebalance",
    ) -> EvScoredCandidate:
        capital_usd = self.compute_capital_usd(pool, deposit_sol_amount, deposit_usdc_amount)
        fee_rate_15m = max(0.0, pool.realized_fee_rate_15m_fraction_proxy())
        active_occupancy, weight_alignment_score, occupancy_source = self.active_occupancy_15m(
            forecast=forecast,
            synth_horizon=synth_horizon,
            weighted_bin_plan=weighted_bin_plan,
            prediction_percentiles=prediction_percentiles,
        )
        concentration = self.concentration_factor(
            forecast=forecast,
            width_ref_pct=width_ref_pct,
            pool=pool,
            bin_step_ref_bps=bin_step_ref_bps,
        )
        utilization_ratio = None
        if weight_alignment_score is not None:
            utilization_ratio = clamp(
                float(weight_alignment_score) / max(float(active_occupancy), self.ev_capture_eps),
                0.0,
                1.0,
            )
        fee_capture_factor = 1.0
        if utilization_ratio is not None:
            fee_capture_factor = clamp(
                utilization_ratio ** self.ev_capture_kappa,
                self.ev_capture_min,
                self.ev_capture_max,
            )
        expected_fees_usd = capital_usd * fee_rate_15m * active_occupancy * concentration * fee_capture_factor
        risk_metrics = self.path_drift_risk_metrics(
            pool=pool,
            forecast=forecast,
            prediction_percentiles=prediction_percentiles,
            fallback_range_occupancy=active_occupancy,
        )
        p50_drift_bps = float(risk_metrics["p50_drift_bps"])
        out_prob = float(risk_metrics["out_of_range_prob_15m"])
        one_sided_prob = float(risk_metrics["one_sided_break_prob"])
        directional_confidence = float(risk_metrics["directional_confidence"])
        expected_oor_minutes = float(risk_metrics["expected_out_of_range_minutes_15m"])
        persistence_norm = (
            max(0.0, float(out_of_range_cycles))
            / max(1.0, float(self.ev_oor_persistence_cap_cycles))
        )
        il_multiplier = clamp(
            1.0
            + self.ev_il_drift_alpha * (p50_drift_bps / max(self.ev_il_drift_ref_bps, 1e-9))
            + self.ev_il_oor_beta * out_prob
            + self.ev_il_onesided_gamma * one_sided_prob
            + self.ev_il_persistence_delta * persistence_norm,
            self.ev_il_mult_min,
            self.ev_il_mult_max,
        )
        il_baseline_fraction = self.il_15m_fraction(forecast=forecast, synth_horizon=synth_horizon)
        il_baseline_usd = capital_usd * il_baseline_fraction
        spot = max(pool.current_price_sol_usdc(), 0.0)
        out_bps = self.out_of_range_bps(
            spot=spot,
            lower=forecast.lower_bound,
            upper=forecast.upper_bound,
        )
        apply_oor_penalty = self.ev_oor_penalty_enabled and (
            not self.ev_oor_penalize_hold_only or str(baseline_mode or "").startswith("current_hold")
        )
        oor_penalty_fraction = (
            self.out_of_range_penalty_fraction_15m(
                out_of_range_bps=out_bps,
                out_of_range_cycles=out_of_range_cycles,
            )
            if apply_oor_penalty
            else 0.0
        )
        il_penalty_usd = capital_usd * oor_penalty_fraction
        il_multiplier_penalty_usd = il_baseline_usd * max(0.0, il_multiplier - 1.0)
        il_state_penalty_usd = il_multiplier_penalty_usd + il_penalty_usd
        il_fraction = (il_baseline_fraction * il_multiplier) + oor_penalty_fraction
        expected_il_usd = il_baseline_usd + il_state_penalty_usd

        cost_info = self._candidate_costs(
            pool=pool,
            forecast=forecast,
            active_position=active_position,
            threshold_bps=range_change_threshold_bps,
            apply_rebalance_costs=apply_rebalance_costs,
        )
        ev_15m = expected_fees_usd - expected_il_usd - cost_info["rebalance_cost_usd"] - cost_info["pool_switch_extra_cost_usd"]
        components = EvComponentBreakdown(
            expected_fees_usd=float(expected_fees_usd),
            expected_il_usd=float(expected_il_usd),
            rebalance_cost_usd=float(cost_info["rebalance_cost_usd"]),
            pool_switch_extra_cost_usd=float(cost_info["pool_switch_extra_cost_usd"]),
            active_occupancy_15m=float(active_occupancy),
            range_active_occupancy_15m=float(active_occupancy),
            weight_alignment_score=float(weight_alignment_score) if weight_alignment_score is not None else None,
            utilization_ratio=float(utilization_ratio) if utilization_ratio is not None else None,
            fee_capture_factor=float(fee_capture_factor),
            concentration_factor=float(concentration),
            fee_rate_15m_fraction=float(fee_rate_15m),
            capital_usd=float(capital_usd),
            il_15m_fraction=float(il_fraction),
            occupancy_source=occupancy_source,
            baseline_mode=baseline_mode or occupancy_source,
            il_baseline_usd=float(il_baseline_usd),
            il_state_penalty_usd=float(il_state_penalty_usd),
            out_of_range_bps=float(out_bps),
            out_of_range_cycles=int(out_of_range_cycles),
            out_of_range_penalty_fraction_15m=float(oor_penalty_fraction),
            il_multiplier=float(il_multiplier),
            p50_drift_bps=p50_drift_bps,
            out_of_range_prob_15m=out_prob,
            one_sided_break_prob=one_sided_prob,
            directional_confidence=directional_confidence,
            expected_out_of_range_minutes_15m=expected_oor_minutes,
        )
        return EvScoredCandidate(
            pool_address=pool.address,
            pool_name=pool.name,
            pool_symbol_pair=f"{pool.symbol_x}/{pool.symbol_y}",
            pool_current_price_sol_usdc=pool.current_price_sol_usdc(),
            forecast=forecast,
            weighted_bin_plan=weighted_bin_plan,
            ev_15m_usd=float(ev_15m),
            ev_components=components,
            pre_rank_score=pre_rank_score,
            rebalance_structural_change=bool(cost_info["structural_change"]),
            pool_switch=bool(cost_info["pool_switch"]),
            range_change_bps_vs_active=cost_info["range_change_bps_vs_active"],
            action_type=action_type,
        )

    def nearest_forecast_to_active_range(
        self,
        forecasts: list[SynthLpBoundForecast],
        active_position: ActivePositionState,
    ) -> SynthLpBoundForecast:
        if not forecasts:
            raise RuntimeError("No forecasts available for baseline comparison")
        return min(
            forecasts,
            key=lambda f: (
                relative_range_change_bps(
                    active_position.lower_price,
                    active_position.upper_price,
                    f.lower_bound,
                    f.upper_bound,
                ),
                abs(f.width_pct - active_position.width_pct),
            ),
        )

    def _candidate_costs(
        self,
        *,
        pool: MeteoraPoolSnapshot,
        forecast: SynthLpBoundForecast,
        active_position: ActivePositionState | None,
        threshold_bps: float,
        apply_rebalance_costs: bool,
    ) -> dict[str, float | bool | None]:
        if active_position is None:
            rebalance_cost = self.rebalance_cost_usd if apply_rebalance_costs else 0.0
            return {
                "rebalance_cost_usd": rebalance_cost,
                "pool_switch_extra_cost_usd": 0.0,
                "pool_switch": False,
                "structural_change": True,
                "range_change_bps_vs_active": None,
            }

        pool_switch = active_position.pool_address != pool.address
        range_change_bps = None
        structural_change = pool_switch
        if not pool_switch:
            range_change_bps = relative_range_change_bps(
                active_position.lower_price,
                active_position.upper_price,
                forecast.lower_bound,
                forecast.upper_bound,
            )
            structural_change = range_change_bps >= threshold_bps

        rebalance_cost = 0.0
        pool_switch_extra = 0.0
        if apply_rebalance_costs and structural_change:
            rebalance_cost = self.rebalance_cost_usd
            if pool_switch:
                pool_switch_extra = self.pool_switch_extra_cost_usd

        return {
            "rebalance_cost_usd": rebalance_cost,
            "pool_switch_extra_cost_usd": pool_switch_extra,
            "pool_switch": pool_switch,
            "structural_change": structural_change,
            "range_change_bps_vs_active": range_change_bps,
        }


def median_width_pct(forecasts: list[SynthLpBoundForecast]) -> float:
    widths = [max(1e-9, f.width_pct) for f in forecasts]
    return float(median(widths)) if widths else 10.0


def median_bin_step_bps(pools: list[MeteoraPoolSnapshot]) -> float:
    values = [float(p.bin_step_bps) for p in pools if p.bin_step_bps is not None and p.bin_step_bps > 0]
    return float(median(values)) if values else 10.0


def _apply_rank_normalization(rows: list[dict[str, object]], key: str) -> None:
    indexed = []
    for idx, row in enumerate(rows):
        try:
            value = float(row.get(key, 0.0))
        except (TypeError, ValueError):
            value = 0.0
        indexed.append((idx, value))
    indexed.sort(key=lambda t: t[1])
    n = len(indexed)
    if n <= 1:
        for row in rows:
            row[f"{key}_rank"] = 1.0
        return

    # Average rank for ties.
    rank_values = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(indexed[j + 1][1] - indexed[i][1]) <= 1e-12:
            j += 1
        avg_rank = (i + j) / 2.0
        norm = avg_rank / (n - 1)
        for k in range(i, j + 1):
            rank_values[k] = norm
        i = j + 1

    for order_idx, (row_idx, _value) in enumerate(indexed):
        rows[row_idx][f"{key}_rank"] = rank_values[order_idx]


def _parse_row_quantiles_normalized(row: dict[float, float]) -> list[tuple[float, float]]:
    parsed: list[tuple[float, float]] = []
    for raw_p, raw_v in row.items():
        try:
            p = float(raw_p)
            q = float(raw_v)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(p) or not math.isfinite(q) or q <= 0:
            continue
        parsed.append((p, q))
    if len(parsed) < 2:
        return []
    parsed.sort(key=lambda t: t[0])
    max_p = parsed[-1][0]
    scale = 1.0 if max_p <= 1.0 + 1e-9 else 100.0
    normalized: list[tuple[float, float]] = []
    last_p = -1.0
    last_q = 0.0
    for raw_p, raw_q in parsed:
        p = raw_p / scale
        if p < 0.0 or p > 1.0:
            continue
        if p <= last_p + 1e-12:
            # drop duplicate/non-monotonic percentile keys
            continue
        q = max(raw_q, last_q)
        normalized.append((p, q))
        last_p = p
        last_q = q
    return normalized if len(normalized) >= 2 else []


def _cdf_at_price_from_quantiles(points: list[tuple[float, float]], price: float) -> float:
    if not points or price <= 0:
        return 0.0
    if price <= points[0][1]:
        return clamp(points[0][0], 0.0, 1.0)
    if price >= points[-1][1]:
        return clamp(points[-1][0], 0.0, 1.0)
    for i in range(len(points) - 1):
        p0, q0 = points[i]
        p1, q1 = points[i + 1]
        if q1 <= q0:
            continue
        if q0 <= price <= q1:
            t = (price - q0) / max(q1 - q0, 1e-12)
            return clamp(p0 + t * (p1 - p0), 0.0, 1.0)
    return clamp(points[-1][0], 0.0, 1.0)


def _price_at_quantile(points: list[tuple[float, float]], target_p: float) -> float:
    if not points:
        return 0.0
    p = clamp(float(target_p), 0.0, 1.0)
    if p <= points[0][0]:
        return points[0][1]
    if p >= points[-1][0]:
        return points[-1][1]
    for i in range(len(points) - 1):
        p0, q0 = points[i]
        p1, q1 = points[i + 1]
        if p1 <= p0:
            continue
        if p0 <= p <= p1:
            t = (p - p0) / max(p1 - p0, 1e-12)
            return max(0.0, q0 + t * (q1 - q0))
    return points[-1][1]
