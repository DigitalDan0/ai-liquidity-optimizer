from __future__ import annotations

import logging
import time
from typing import Any

from ai_liquidity_optimizer.clients.meteora import MeteoraDlmmApiClient
from ai_liquidity_optimizer.clients.synth import SynthInsightsClient
from ai_liquidity_optimizer.config import Settings
from ai_liquidity_optimizer.execution.base import PositionExecutor
from ai_liquidity_optimizer.models import BotState, EvScoredCandidate, ExecutionApplyRequest, MeteoraPoolSnapshot, utc_now_iso
from ai_liquidity_optimizer.state_store import JsonStateStore
from ai_liquidity_optimizer.strategy.bin_weights import compute_bin_weights_for_range, derive_mvp_bin_edges_for_range
from ai_liquidity_optimizer.strategy.ev import EvLpScorer, median_bin_step_bps, median_width_pct
from ai_liquidity_optimizer.strategy.scoring import StrategyScorer, relative_range_change_bps


LOGGER = logging.getLogger(__name__)


class OptimizerOrchestrator:
    def __init__(
        self,
        settings: Settings,
        synth_client: SynthInsightsClient,
        meteora_client: MeteoraDlmmApiClient,
        scorer: StrategyScorer,
        ev_scorer: EvLpScorer | None,
        executor: PositionExecutor,
        state_store: JsonStateStore,
    ):
        self.settings = settings
        self.synth_client = synth_client
        self.meteora_client = meteora_client
        self.scorer = scorer
        self.ev_scorer = ev_scorer
        self.executor = executor
        self.state_store = state_store

    def run_forever(self) -> None:
        interval_seconds = self.settings.rebalance_interval_minutes * 60
        while True:
            self.run_once()
            time.sleep(interval_seconds)

    def run_once(self) -> dict:
        state = self.state_store.load()
        if self.settings.ev_mode:
            if self.ev_scorer is None:
                raise RuntimeError("EV_MODE=true but no EvLpScorer was configured")
            return self._run_once_ev(state)
        return self._run_once_proxy(state)

    def _run_once_proxy(self, state: BotState) -> dict:
        pool = self.meteora_client.find_sol_usdc_pool(
            pool_address=self.settings.meteora_pool_address,
            query=self.settings.meteora_pool_query,
        )
        forecasts = self.synth_client.get_lp_bounds(
            asset=self.settings.synth_asset,
            horizon=self.settings.synth_horizon,
            days=self.settings.synth_days,
            limit=self.settings.synth_limit,
        )
        decision = self.scorer.rank_candidates(
            forecasts=forecasts,
            pool=pool,
            horizon=self.settings.synth_horizon,
            max_candidates=self.settings.max_candidates,
        )

        chosen = decision.chosen
        weighted_bin_plan = None
        try:
            lp_probabilities = self.synth_client.get_lp_probabilities(
                asset=self.settings.synth_asset,
                horizon=self.settings.synth_horizon,
                days=self.settings.synth_days,
            )

            prediction_percentiles = None
            try:
                prediction_percentiles = self.synth_client.get_prediction_percentiles(asset=self.settings.synth_asset)
            except Exception as exc:  # pragma: no cover - best-effort optional endpoint
                LOGGER.warning("prediction-percentiles unavailable; using terminal-only bin weights: %s", exc)

            bin_edges, binning_mode = derive_mvp_bin_edges_for_range(
                pool=pool,
                range_lower=chosen.forecast.lower_bound,
                range_upper=chosen.forecast.upper_bound,
                target_bin_count=24,
            )
            weighted_bin_plan = compute_bin_weights_for_range(
                forecast=chosen.forecast,
                horizon=self.settings.synth_horizon,
                bin_edges=bin_edges,
                current_price=pool.current_price_sol_usdc(),
                lp_probabilities=lp_probabilities,
                prediction_percentiles=prediction_percentiles,
                binning_mode=binning_mode,
            )
            top_weight = max(weighted_bin_plan.weights) if weighted_bin_plan.weights else 0.0
            LOGGER.info(
                "Computed %d bin weights (mode=%s, fallback=%s, max_w=%.4f, path=%s)",
                len(weighted_bin_plan.weights),
                weighted_bin_plan.diagnostics.binning_mode,
                weighted_bin_plan.diagnostics.fallback_reason,
                top_weight,
                weighted_bin_plan.diagnostics.used_prediction_percentiles,
            )
        except Exception as exc:
            LOGGER.warning("Failed to compute Synth-driven bin weights; continuing without them: %s", exc)

        should_rebalance, reason = _should_rebalance(
            state=state,
            pool_address=pool.address,
            new_lower=chosen.forecast.lower_bound,
            new_upper=chosen.forecast.upper_bound,
            threshold_bps=self.settings.range_change_threshold_bps,
        )

        LOGGER.info(
            "Selected range %.4f-%.4f (width=%s%%, score=%.6f, p_stay=%.4f, il=%.4f), rebalance=%s (%s)",
            chosen.forecast.lower_bound,
            chosen.forecast.upper_bound,
            chosen.forecast.width_pct,
            chosen.score,
            chosen.forecast.probability_to_stay_in_interval,
            chosen.forecast.expected_impermanent_loss,
            should_rebalance,
            reason,
        )

        execution_result = None
        if should_rebalance:
            execution_result = self.executor.apply_target_range(
                ExecutionApplyRequest(
                    pool=pool,
                    target_forecast=chosen.forecast,
                    target_lower_price=chosen.forecast.lower_bound,
                    target_upper_price=chosen.forecast.upper_bound,
                    deposit_sol_amount=self.settings.deposit_sol_amount,
                    deposit_usdc_amount=self.settings.deposit_usdc_amount,
                    existing_position=state.active_position,
                    target_bin_edges=weighted_bin_plan.bin_edges if weighted_bin_plan else None,
                    target_bin_weights=weighted_bin_plan.weights if weighted_bin_plan else None,
                )
            )
            if execution_result.active_position is not None:
                state.active_position = execution_result.active_position
            LOGGER.info("Execution completed: changed=%s txs=%s", execution_result.changed, execution_result.tx_signatures)

        state.last_decision = {
            "timestamp": decision.generated_at,
            "pool": {
                "address": pool.address,
                "name": pool.name,
                "current_price": pool.current_price,
                "symbol_x": pool.symbol_x,
                "symbol_y": pool.symbol_y,
            },
            "horizon": decision.horizon,
            "chosen": chosen.to_dict(),
            "bin_weight_plan": weighted_bin_plan.to_dict() if weighted_bin_plan else None,
            "top_candidates": [c.to_dict() for c in decision.ranked[:5]],
            "ev_mode": False,
            "rebalance": {
                "should_rebalance": should_rebalance,
                "reason": reason,
                "executor": self.settings.executor,
                "tx_signatures": execution_result.tx_signatures if execution_result else [],
            },
        }
        self.state_store.save(state)

        return state.last_decision

    def _run_once_ev(self, state: BotState) -> dict:
        assert self.ev_scorer is not None
        pools = self._load_pool_candidates()
        if not pools:
            raise RuntimeError("No eligible SOL/USDC Meteora pools after filtering")

        forecasts = self.synth_client.get_lp_bounds(
            asset=self.settings.synth_asset,
            horizon=self.settings.synth_horizon,
            days=self.settings.synth_days,
            limit=self.settings.synth_limit,
        )
        lp_probabilities = self.synth_client.get_lp_probabilities(
            asset=self.settings.synth_asset,
            horizon=self.settings.synth_horizon,
            days=self.settings.synth_days,
        )

        prediction_percentiles = None
        try:
            prediction_percentiles = self.synth_client.get_prediction_percentiles(asset=self.settings.synth_asset)
        except Exception as exc:  # pragma: no cover - optional path endpoint
            LOGGER.warning("prediction-percentiles unavailable; EV occupancy will use fallbacks: %s", exc)

        pre_scores = self.ev_scorer.pool_pre_rank(pools, limit=self.settings.pool_candidate_limit)
        pre_score_by_addr = {p.pool_address: p for p in pre_scores}
        pruned_pools = [p for p in pools if p.address in pre_score_by_addr]
        if not pruned_pools:
            pruned_pools = pools[: self.settings.pool_candidate_limit]
        pruned_pools.sort(key=lambda p: pre_score_by_addr.get(p.address).score if p.address in pre_score_by_addr else -1.0, reverse=True)

        width_ref_pct = median_width_pct(forecasts)
        bin_step_ref_bps = median_bin_step_bps(pruned_pools)

        ev_candidates: list[EvScoredCandidate] = []
        pools_by_address: dict[str, MeteoraPoolSnapshot] = {p.address: p for p in pruned_pools}
        range_failures = 0
        for pool in pruned_pools:
            proxy_ranked = self.scorer.rank_candidates(
                forecasts=forecasts,
                pool=pool,
                horizon=self.settings.synth_horizon,
                max_candidates=self.settings.max_candidates,
            ).ranked
            top_range_candidates = proxy_ranked[: max(1, self.settings.top_k_ranges_per_pool)]
            for scored in top_range_candidates:
                try:
                    weighted_bin_plan = self._compute_weighted_bin_plan(
                        pool=pool,
                        lower=scored.forecast.lower_bound,
                        upper=scored.forecast.upper_bound,
                        forecast=scored.forecast,
                        lp_probabilities=lp_probabilities,
                        prediction_percentiles=prediction_percentiles,
                    )
                    ev_candidate = self.ev_scorer.score_pool_range_ev_15m(
                        pool=pool,
                        forecast=scored.forecast,
                        weighted_bin_plan=weighted_bin_plan,
                        synth_horizon=self.settings.synth_horizon,
                        prediction_percentiles=prediction_percentiles,
                        deposit_sol_amount=self.settings.deposit_sol_amount,
                        deposit_usdc_amount=self.settings.deposit_usdc_amount,
                        width_ref_pct=width_ref_pct,
                        bin_step_ref_bps=bin_step_ref_bps,
                        active_position=state.active_position,
                        range_change_threshold_bps=self.settings.range_change_threshold_bps,
                        apply_rebalance_costs=True,
                        pre_rank_score=pre_score_by_addr.get(pool.address).score if pool.address in pre_score_by_addr else None,
                    )
                    ev_candidates.append(ev_candidate)
                except Exception as exc:
                    range_failures += 1
                    LOGGER.warning(
                        "Skipping EV candidate pool=%s range=%.4f-%.4f due to error: %s",
                        pool.address,
                        scored.forecast.lower_bound,
                        scored.forecast.upper_bound,
                        exc,
                    )

        if not ev_candidates:
            raise RuntimeError(f"No EV candidates could be scored (range_failures={range_failures})")

        ev_candidates.sort(
            key=lambda c: (
                c.ev_15m_usd,
                c.ev_components.active_occupancy_15m,
                -c.forecast.width_pct,
            ),
            reverse=True,
        )
        best = ev_candidates[0]
        selected_pool = pools_by_address.get(best.pool_address)
        if selected_pool is None:
            selected_pool = self.meteora_client.get_pool(best.pool_address)
            pools_by_address[selected_pool.address] = selected_pool

        ev_current_hold = self._compute_current_hold_ev(
            state=state,
            forecasts=forecasts,
            lp_probabilities=lp_probabilities,
            prediction_percentiles=prediction_percentiles,
            width_ref_pct=width_ref_pct,
            bin_step_ref_bps=bin_step_ref_bps,
            pools_by_address=pools_by_address,
            pre_score_by_addr=pre_score_by_addr,
        )
        ev_delta_usd = best.ev_15m_usd if ev_current_hold is None else (best.ev_15m_usd - ev_current_hold.ev_15m_usd)

        should_rebalance, reason, rebalance_gate = _should_rebalance_ev(
            state=state,
            best=best,
            ev_delta_usd=ev_delta_usd,
            min_ev_improvement_usd=self.settings.min_ev_improvement_usd,
        )

        LOGGER.info(
            (
                "EV best pool=%s range=%.4f-%.4f EV15m=$%.4f "
                "(fees=$%.4f il=$%.4f costs=$%.4f%s occ=%.3f conc=%.3f src=%s) "
                "hold=%s delta=%s rebalance=%s (%s)"
            ),
            best.pool_name,
            best.forecast.lower_bound,
            best.forecast.upper_bound,
            best.ev_15m_usd,
            best.ev_components.expected_fees_usd,
            best.ev_components.expected_il_usd,
            best.ev_components.rebalance_cost_usd,
            f"+${best.ev_components.pool_switch_extra_cost_usd:.4f}" if best.ev_components.pool_switch_extra_cost_usd else "",
            best.ev_components.active_occupancy_15m,
            best.ev_components.concentration_factor,
            best.ev_components.occupancy_source or best.ev_components.baseline_mode,
            f"${ev_current_hold.ev_15m_usd:.4f}" if ev_current_hold else "n/a",
            f"${ev_delta_usd:.4f}" if ev_delta_usd is not None else "n/a",
            should_rebalance,
            reason,
        )

        execution_result = None
        if should_rebalance:
            execution_result = self.executor.apply_target_range(
                ExecutionApplyRequest(
                    pool=selected_pool,
                    target_forecast=best.forecast,
                    target_lower_price=best.forecast.lower_bound,
                    target_upper_price=best.forecast.upper_bound,
                    deposit_sol_amount=self.settings.deposit_sol_amount,
                    deposit_usdc_amount=self.settings.deposit_usdc_amount,
                    existing_position=state.active_position,
                    target_bin_edges=best.weighted_bin_plan.bin_edges,
                    target_bin_weights=best.weighted_bin_plan.weights,
                )
            )
            if execution_result.active_position is not None:
                state.active_position = execution_result.active_position
            LOGGER.info("Execution completed: changed=%s txs=%s", execution_result.changed, execution_result.tx_signatures)

        state.last_decision = self._build_ev_last_decision(
            best=best,
            selected_pool=selected_pool,
            ev_current_hold=ev_current_hold,
            ev_delta_usd=ev_delta_usd,
            ev_candidates=ev_candidates,
            pre_scores=pre_scores,
            rebalance_gate=rebalance_gate,
            should_rebalance=should_rebalance,
            reason=reason,
            execution_result=execution_result,
        )
        self.state_store.save(state)
        return state.last_decision

    def _load_pool_candidates(self) -> list[MeteoraPoolSnapshot]:
        if self.settings.meteora_pool_address:
            return [
                self.meteora_client.find_sol_usdc_pool(
                    pool_address=self.settings.meteora_pool_address,
                    query=self.settings.meteora_pool_query,
                )
            ]
        return self.meteora_client.list_sol_usdc_pool_candidates(
            query=self.settings.meteora_pool_query,
            per_page=max(200, self.settings.pool_candidate_limit * 4),
            min_tvl_usd=self.settings.min_pool_tvl_usd,
        )

    def _compute_weighted_bin_plan(
        self,
        *,
        pool: MeteoraPoolSnapshot,
        lower: float,
        upper: float,
        forecast,
        lp_probabilities,
        prediction_percentiles,
    ):
        bin_edges, binning_mode = derive_mvp_bin_edges_for_range(
            pool=pool,
            range_lower=lower,
            range_upper=upper,
            target_bin_count=24,
        )
        weighted_bin_plan = compute_bin_weights_for_range(
            forecast=forecast,
            horizon=self.settings.synth_horizon,
            bin_edges=bin_edges,
            current_price=pool.current_price_sol_usdc(),
            lp_probabilities=lp_probabilities,
            prediction_percentiles=prediction_percentiles,
            binning_mode=binning_mode,
        )
        top_weight = max(weighted_bin_plan.weights) if weighted_bin_plan.weights else 0.0
        LOGGER.info(
            "Computed %d bin weights (mode=%s, fallback=%s, max_w=%.4f, path=%s)",
            len(weighted_bin_plan.weights),
            weighted_bin_plan.diagnostics.binning_mode,
            weighted_bin_plan.diagnostics.fallback_reason,
            top_weight,
            weighted_bin_plan.diagnostics.used_prediction_percentiles,
        )
        return weighted_bin_plan

    def _compute_current_hold_ev(
        self,
        *,
        state: BotState,
        forecasts,
        lp_probabilities,
        prediction_percentiles,
        width_ref_pct: float,
        bin_step_ref_bps: float,
        pools_by_address: dict[str, MeteoraPoolSnapshot],
        pre_score_by_addr: dict[str, Any],
    ) -> EvScoredCandidate | None:
        if state.active_position is None:
            return None
        active = state.active_position
        pool = pools_by_address.get(active.pool_address)
        if pool is None:
            try:
                pool = self.meteora_client.get_pool(active.pool_address)
                pools_by_address[pool.address] = pool
            except Exception as exc:
                LOGGER.warning("Failed to load active pool for EV hold baseline (%s): %s", active.pool_address, exc)
                return None
        try:
            forecast = self.ev_scorer.nearest_forecast_to_active_range(forecasts, active)
            weighted_bin_plan = self._compute_weighted_bin_plan(
                pool=pool,
                lower=forecast.lower_bound,
                upper=forecast.upper_bound,
                forecast=forecast,
                lp_probabilities=lp_probabilities,
                prediction_percentiles=prediction_percentiles,
            )
            return self.ev_scorer.score_pool_range_ev_15m(
                pool=pool,
                forecast=forecast,
                weighted_bin_plan=weighted_bin_plan,
                synth_horizon=self.settings.synth_horizon,
                prediction_percentiles=prediction_percentiles,
                deposit_sol_amount=self.settings.deposit_sol_amount,
                deposit_usdc_amount=self.settings.deposit_usdc_amount,
                width_ref_pct=width_ref_pct,
                bin_step_ref_bps=bin_step_ref_bps,
                active_position=active,
                range_change_threshold_bps=self.settings.range_change_threshold_bps,
                apply_rebalance_costs=False,
                baseline_mode="current_hold",
                pre_rank_score=pre_score_by_addr.get(pool.address).score if pool.address in pre_score_by_addr else None,
            )
        except Exception as exc:
            LOGGER.warning("Failed to compute EV hold baseline; proceeding without it: %s", exc)
            return None

    def _build_ev_last_decision(
        self,
        *,
        best: EvScoredCandidate,
        selected_pool: MeteoraPoolSnapshot,
        ev_current_hold: EvScoredCandidate | None,
        ev_delta_usd: float | None,
        ev_candidates: list[EvScoredCandidate],
        pre_scores,
        rebalance_gate: dict[str, Any],
        should_rebalance: bool,
        reason: str,
        execution_result,
    ) -> dict[str, Any]:
        chosen = {
            "width_pct": best.forecast.width_pct,
            "lower_bound": best.forecast.lower_bound,
            "upper_bound": best.forecast.upper_bound,
            "probability_to_stay_in_interval": best.forecast.probability_to_stay_in_interval,
            "expected_time_in_interval_minutes": best.forecast.expected_time_in_interval_minutes,
            "expected_impermanent_loss": best.forecast.expected_impermanent_loss,
            "score": best.ev_15m_usd,
            "ev_15m_usd": best.ev_15m_usd,
            "ev_components": best.ev_components.to_dict(),
        }
        pool_dict = {
            "address": selected_pool.address,
            "name": selected_pool.name,
            "current_price": selected_pool.current_price,
            "symbol_x": selected_pool.symbol_x,
            "symbol_y": selected_pool.symbol_y,
            "tvl": selected_pool.tvl_usd(),
            "bin_step_bps": selected_pool.bin_step_bps,
            "base_fee_pct": selected_pool.base_fee_pct,
            "dynamic_fee_pct": selected_pool.dynamic_fee_pct,
        }
        return {
            "timestamp": utc_now_iso(),
            "ev_mode": True,
            "pool": pool_dict,
            "selected_pool": pool_dict,
            "horizon": self.settings.synth_horizon,
            "chosen": chosen,
            "bin_weight_plan": best.weighted_bin_plan.to_dict(),
            "top_candidates": [_summarize_ev_candidate(c) for c in ev_candidates[:5]],
            "ev_best_candidate": best.to_dict(),
            "ev_current_hold": ev_current_hold.to_dict() if ev_current_hold else None,
            "ev_delta_usd": ev_delta_usd,
            "rebalance_gate": rebalance_gate,
            "pool_rankings": {
                "pre_ranked": [p.to_dict() for p in pre_scores[: min(10, len(pre_scores))]],
                "evaluated_candidate_count": len(ev_candidates),
                "evaluated_pool_count": len({c.pool_address for c in ev_candidates}),
            },
            "rebalance": {
                "should_rebalance": should_rebalance,
                "reason": reason,
                "executor": self.settings.executor,
                "tx_signatures": execution_result.tx_signatures if execution_result else [],
            },
        }


def _should_rebalance(
    state: BotState,
    pool_address: str,
    new_lower: float,
    new_upper: float,
    threshold_bps: float,
) -> tuple[bool, str]:
    if state.active_position is None:
        return True, "no_active_position"
    if state.active_position.pool_address != pool_address:
        return True, "pool_changed"
    delta_bps = relative_range_change_bps(
        current_lower=state.active_position.lower_price,
        current_upper=state.active_position.upper_price,
        new_lower=new_lower,
        new_upper=new_upper,
    )
    if delta_bps >= threshold_bps:
        return True, f"range_changed_{delta_bps:.2f}bps"
    return False, f"range_change_below_threshold_{delta_bps:.2f}bps"


def _should_rebalance_ev(
    *,
    state: BotState,
    best: EvScoredCandidate,
    ev_delta_usd: float | None,
    min_ev_improvement_usd: float,
) -> tuple[bool, str, dict[str, Any]]:
    if state.active_position is None:
        gate = {
            "baseline_present": False,
            "ev_delta_usd": ev_delta_usd,
            "min_ev_improvement_usd": min_ev_improvement_usd,
            "ev_threshold_passed": True,
            "structural_change_passed": True,
            "pool_switch": best.pool_switch,
            "range_change_bps_vs_active": best.range_change_bps_vs_active,
        }
        return True, "no_active_position", gate

    delta = float(ev_delta_usd or 0.0)
    ev_threshold_passed = delta >= float(min_ev_improvement_usd)
    structural_change_passed = bool(best.rebalance_structural_change)

    gate = {
        "baseline_present": True,
        "ev_delta_usd": delta,
        "min_ev_improvement_usd": float(min_ev_improvement_usd),
        "ev_threshold_passed": ev_threshold_passed,
        "structural_change_passed": structural_change_passed,
        "pool_switch": best.pool_switch,
        "range_change_bps_vs_active": best.range_change_bps_vs_active,
    }
    if not ev_threshold_passed:
        return False, f"ev_delta_below_threshold_{delta:.4f}", gate
    if not structural_change_passed:
        if best.range_change_bps_vs_active is None:
            return False, "no_structural_change", gate
        return False, f"range_change_below_threshold_{best.range_change_bps_vs_active:.2f}bps", gate
    if best.pool_switch:
        return True, f"pool_switch_ev_gain_{delta:.4f}", gate
    return True, f"range_ev_gain_{delta:.4f}", gate


def _summarize_ev_candidate(candidate: EvScoredCandidate) -> dict[str, Any]:
    return {
        "pool_address": candidate.pool_address,
        "pool_name": candidate.pool_name,
        "ev_15m_usd": candidate.ev_15m_usd,
        "pool_switch": candidate.pool_switch,
        "range_change_bps_vs_active": candidate.range_change_bps_vs_active,
        "rebalance_structural_change": candidate.rebalance_structural_change,
        "forecast": {
            "width_pct": candidate.forecast.width_pct,
            "lower_bound": candidate.forecast.lower_bound,
            "upper_bound": candidate.forecast.upper_bound,
            "probability_to_stay_in_interval": candidate.forecast.probability_to_stay_in_interval,
            "expected_time_in_interval_minutes": candidate.forecast.expected_time_in_interval_minutes,
            "expected_impermanent_loss": candidate.forecast.expected_impermanent_loss,
        },
        "ev_components": candidate.ev_components.to_dict(),
        "pre_rank_score": candidate.pre_rank_score,
    }
