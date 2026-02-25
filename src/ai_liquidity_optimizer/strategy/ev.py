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
from ai_liquidity_optimizer.strategy.bin_weights import compute_weighted_active_occupancy_from_percentiles
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
    ) -> tuple[float, str]:
        if prediction_percentiles is not None:
            weighted_occ = compute_weighted_active_occupancy_from_percentiles(
                bin_edges=weighted_bin_plan.bin_edges,
                bin_weights=weighted_bin_plan.weights,
                prediction_percentiles=prediction_percentiles,
                tau_half_minutes=self.ev_percentile_decay_half_life_minutes,
                max_horizon_minutes=self.ev_horizon_minutes,
            )
            if weighted_occ is not None:
                return clamp(weighted_occ, 0.0, 1.0), "prediction_percentiles_weighted"

        # Fallback to terminal mass proxy restricted to the selected range.
        mass_in_range = weighted_bin_plan.diagnostics.mass_in_range
        if mass_in_range > 0:
            return clamp(mass_in_range, 0.0, 1.0), "lp_probabilities_terminal_mass"

        # Final fallback to Synth lp-bounds expected active fraction over the Synth horizon.
        horizon_minutes = max(float(horizon_to_minutes(synth_horizon)), 1.0)
        t_frac = forecast.expected_time_in_interval_minutes / horizon_minutes
        return clamp(t_frac, 0.0, 1.0), "lp_bounds_expected_time_fraction"

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
    ) -> EvScoredCandidate:
        capital_usd = self.compute_capital_usd(pool, deposit_sol_amount, deposit_usdc_amount)
        fee_rate_15m = max(0.0, pool.realized_fee_rate_15m_fraction_proxy())
        active_occupancy, occupancy_source = self.active_occupancy_15m(
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
        expected_fees_usd = capital_usd * fee_rate_15m * active_occupancy * concentration
        il_fraction = self.il_15m_fraction(forecast=forecast, synth_horizon=synth_horizon)
        expected_il_usd = capital_usd * il_fraction

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
            concentration_factor=float(concentration),
            fee_rate_15m_fraction=float(fee_rate_15m),
            capital_usd=float(capital_usd),
            il_15m_fraction=float(il_fraction),
            occupancy_source=occupancy_source,
            baseline_mode=baseline_mode or occupancy_source,
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
