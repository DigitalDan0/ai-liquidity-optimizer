from __future__ import annotations

import json
import logging
import time
import traceback
from datetime import datetime, timezone
from statistics import median
from typing import Any

from ai_liquidity_optimizer.clients.meteora import MeteoraDlmmApiClient
from ai_liquidity_optimizer.clients.synth import SynthInsightsClient
from ai_liquidity_optimizer.config import Settings
from ai_liquidity_optimizer.execution.base import PositionExecutor
from ai_liquidity_optimizer.models import (
    ActivePositionState,
    BinWeightingDiagnostics,
    BotState,
    EvComponentBreakdown,
    EvScoredCandidate,
    ExecutionApplyRequest,
    ExecutorRangeBinQuote,
    MeteoraPoolSnapshot,
    SynthLpBoundForecast,
    WeightedBinPlan,
    utc_now_iso,
)
from ai_liquidity_optimizer.state_store import JsonStateStore
from ai_liquidity_optimizer.strategy.bin_weights import (
    compute_bin_weights_for_range,
    compute_exact_sdk_bin_odds_weight_plan,
    derive_mvp_bin_edges_for_range,
)
from ai_liquidity_optimizer.strategy.ev import EvLpScorer, median_bin_step_bps, median_width_pct
from ai_liquidity_optimizer.strategy.realism import (
    CalibrationSnapshot,
    build_calibration_snapshot_from_journal,
    default_calibration_snapshot,
)
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
        retryable_error_delay_seconds = min(45.0, max(5.0, interval_seconds / 6.0))
        while True:
            cycle_started = time.time()
            sleep_seconds = interval_seconds
            try:
                self.run_once()
            except KeyboardInterrupt:  # pragma: no cover - interactive shutdown
                raise
            except Exception as exc:
                LOGGER.exception("Run cycle failed; skipping this interval and continuing")
                if self._is_retryable_loop_error(exc):
                    sleep_seconds = retryable_error_delay_seconds
                    LOGGER.warning(
                        "Retryable runtime error detected; scheduling early retry in %.1fs",
                        sleep_seconds,
                    )
            elapsed = time.time() - cycle_started
            time.sleep(max(0.0, sleep_seconds - elapsed))

    def run_once(self) -> dict:
        state = self.state_store.load()
        state = self._reconcile_local_position_state(state)
        try:
            if self.settings.ev_mode:
                if self.ev_scorer is None:
                    raise RuntimeError("EV_MODE=true but no EvLpScorer was configured")
                result = self._run_once_ev(state)
            else:
                result = self._run_once_proxy(state)
            self._append_trade_journal_decision_entry(result=result)
            return result
        except Exception as exc:
            self._append_trade_journal_error_entry(exc=exc, state=state)
            raise

    def _reconcile_local_position_state(self, state: BotState) -> BotState:
        current = state.active_position
        if current is None:
            return state
        reconciled = self.executor.reconcile_active_position(current)
        if reconciled is current:
            return state
        if reconciled is None:
            LOGGER.warning(
                "Tracked active position %s is not present on-chain/executor anymore; clearing local state",
                current.position_pubkey or "<unknown>",
            )
            self._reset_position_lifecycle_state(state)
        state.active_position = reconciled
        if (
            reconciled is not None
            and current is not None
            and current.position_pubkey
            and reconciled.position_pubkey
            and current.position_pubkey != reconciled.position_pubkey
        ):
            # Different on-chain position identity means a new lifecycle.
            self._reset_position_lifecycle_state(state)
        # Persist immediately so a later transient API failure doesn't leave stale state behind.
        self.state_store.save(state)
        return state

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
                max_single_bin=self.settings.synth_weight_max_single_bin,
                max_top3=self.settings.synth_weight_max_top3,
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
            had_existing_position = state.active_position is not None
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
            _log_execution_details(execution_result.details)

        onchain_snapshot, onchain_delta = self._capture_onchain_snapshot(state=state, pool=pool)
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
            "onchain_snapshot": onchain_snapshot,
            "onchain_delta": onchain_delta,
        }
        self.state_store.save(state)

        return state.last_decision

    def _run_once_ev(self, state: BotState) -> dict:
        assert self.ev_scorer is not None
        pools = self._load_pool_candidates(state=state)
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

        cost_model_info = self._apply_tx_cost_model(pools=pruned_pools)

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
                        action_type="rebalance",
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

        scoring_objective_used = "hybrid"
        rescored_candidate_count = 0
        if self._scoring_uses_exact_objective():
            ev_candidates, rescored_candidate_count = self._rescore_candidates_exact(
                candidates=ev_candidates,
                pools_by_address=pools_by_address,
                lp_probabilities=lp_probabilities,
                prediction_percentiles=prediction_percentiles,
                width_ref_pct=width_ref_pct,
                bin_step_ref_bps=bin_step_ref_bps,
                active_position=state.active_position,
                pre_score_by_addr=pre_score_by_addr,
            )
            if rescored_candidate_count > 0:
                scoring_objective_used = "odds_15m_exact"
            else:
                scoring_objective_used = "hybrid_fallback"

        realism_snapshot = self._build_realism_snapshot()
        use_adjusted_for_decisions = bool(self.settings.ev_realism_enabled) and not bool(self.settings.ev_realism_shadow_mode)
        for candidate in ev_candidates:
            self._apply_realism_adjustments(
                candidate=candidate,
                snapshot=realism_snapshot,
                apply_to_decision=use_adjusted_for_decisions,
            )

        ev_candidates.sort(
            key=lambda c: (
                self._candidate_effective_ev(c, use_adjusted=use_adjusted_for_decisions),
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
        hold_oor_cycles = self._update_oor_hold_state(
            state=state,
            pools_by_address=pools_by_address,
        )

        ev_current_hold = self._compute_current_hold_ev(
            state=state,
            forecasts=forecasts,
            lp_probabilities=lp_probabilities,
            prediction_percentiles=prediction_percentiles,
            width_ref_pct=width_ref_pct,
            bin_step_ref_bps=bin_step_ref_bps,
            pools_by_address=pools_by_address,
            pre_score_by_addr=pre_score_by_addr,
            scoring_objective=scoring_objective_used,
            hold_oor_cycles=hold_oor_cycles,
        )
        if ev_current_hold is not None:
            self._apply_realism_adjustments(
                candidate=ev_current_hold,
                snapshot=realism_snapshot,
                apply_to_decision=use_adjusted_for_decisions,
            )
        active_pool_for_snapshot = self._pool_for_active_position(
            state=state,
            pools_by_address=pools_by_address,
            fallback_pool=selected_pool,
        )
        pre_onchain_snapshot, pre_onchain_delta = self._capture_onchain_snapshot(
            state=state,
            pool=active_pool_for_snapshot,
        )
        lifecycle_metrics = self._update_position_lifecycle_state(
            state=state,
            active_pool=active_pool_for_snapshot,
            onchain_snapshot=pre_onchain_snapshot,
            reset_baseline=False,
        )
        prior_protective_breach_count = int((state.strategy_state or {}).get("protective_breach_count", 0))
        idle_candidate = self._compute_idle_ev_candidate(
            state=state,
            pools_by_address=pools_by_address,
            fallback_pool=selected_pool,
        )
        self._apply_realism_adjustments(
            candidate=idle_candidate,
            snapshot=realism_snapshot,
            apply_to_decision=use_adjusted_for_decisions,
        )
        effective_min_delta_usd = self._effective_min_delta_usd(
            realism_snapshot,
            use_adjusted=use_adjusted_for_decisions,
        )
        action_cooldown_remaining = max(0, int((state.strategy_state or {}).get("action_cooldown_remaining", 0)))
        if action_cooldown_remaining > 0:
            action_cooldown_remaining -= 1
        state.strategy_state["action_cooldown_remaining"] = action_cooldown_remaining
        (
            selected_action,
            should_rebalance,
            should_close_to_idle,
            reason,
            rebalance_gate,
            protective_breach_count,
            ev_delta_usd,
            action_scores,
        ) = self._decide_ev_action(
            state=state,
            best=best,
            ev_current_hold=ev_current_hold,
            idle_candidate=idle_candidate,
            prior_protective_breach_count=prior_protective_breach_count,
            hold_oor_cycles=hold_oor_cycles,
            action_cooldown_remaining=action_cooldown_remaining,
            lifecycle_metrics=lifecycle_metrics,
            use_adjusted_ev=use_adjusted_for_decisions,
            effective_min_delta_usd=effective_min_delta_usd,
        )
        state.strategy_state["protective_breach_count"] = 0 if should_rebalance else protective_breach_count

        best_raw_ev = self._candidate_raw_ev(best)
        best_adjusted_ev = best.ev_components.adjusted_ev_15m_usd
        hold_raw_ev = self._candidate_raw_ev(ev_current_hold) if ev_current_hold is not None else None
        hold_adjusted_ev = ev_current_hold.ev_components.adjusted_ev_15m_usd if ev_current_hold is not None else None
        LOGGER.info(
            (
                "EV best pool=%s spot=%.4f range=%.4f-%.4f "
                "EVraw=$%.4f EVadj=%s "
                "(fees_raw=$%.4f fees_adj=%s il=$%.4f[base=$%.4f pen=$%.4f] costs=$%.4f%s drag=%s unc=%s occ_range=%.3f align=%s util=%s capture=%s conc=%.3f src=%s) "
                "active_range=%s hold_raw=%s hold_adj=%s hold_util=%s hold_out_bps=%s hold_oor_cycles=%s hold_exact_bounds=%s "
                "lifecycle_pnl=%s lifecycle_pct=%s policy=%s "
                "delta=%s min_delta_eff=%s cal=%s/%s action=%s gate=%s breach=%s rebalance=%s close_idle=%s (%s)"
            ),
            best.pool_name,
            selected_pool.current_price_sol_usdc(),
            best.forecast.lower_bound,
            best.forecast.upper_bound,
            best_raw_ev,
            f"${best_adjusted_ev:.4f}" if best_adjusted_ev is not None else "n/a",
            best.ev_components.raw_expected_fees_usd if best.ev_components.raw_expected_fees_usd is not None else best.ev_components.expected_fees_usd,
            (
                f"${best.ev_components.adjusted_expected_fees_usd:.4f}"
                if best.ev_components.adjusted_expected_fees_usd is not None
                else "n/a"
            ),
            best.ev_components.expected_il_usd,
            best.ev_components.il_baseline_usd or 0.0,
            best.ev_components.il_state_penalty_usd or 0.0,
            best.ev_components.rebalance_cost_usd,
            f"+${best.ev_components.pool_switch_extra_cost_usd:.4f}" if best.ev_components.pool_switch_extra_cost_usd else "",
            f"{best.ev_components.execution_drag_usd:.4f}" if best.ev_components.execution_drag_usd is not None else "n/a",
            f"{best.ev_components.uncertainty_penalty_usd:.4f}" if best.ev_components.uncertainty_penalty_usd is not None else "n/a",
            best.ev_components.range_active_occupancy_15m or best.ev_components.active_occupancy_15m,
            f"{best.ev_components.weight_alignment_score:.3f}" if best.ev_components.weight_alignment_score is not None else "n/a",
            f"{best.ev_components.utilization_ratio:.3f}" if best.ev_components.utilization_ratio is not None else "n/a",
            f"{best.ev_components.fee_capture_factor:.3f}" if best.ev_components.fee_capture_factor is not None else "n/a",
            best.ev_components.concentration_factor,
            best.ev_components.occupancy_source or best.ev_components.baseline_mode,
            (
                f"[{state.active_position.lower_price:.4f},{state.active_position.upper_price:.4f}]"
                if state.active_position is not None
                else "n/a"
            ),
            f"${hold_raw_ev:.4f}" if hold_raw_ev is not None else "n/a",
            f"${hold_adjusted_ev:.4f}" if hold_adjusted_ev is not None else "n/a",
            f"{ev_current_hold.ev_components.utilization_ratio:.3f}"
            if ev_current_hold and ev_current_hold.ev_components.utilization_ratio is not None
            else "n/a",
            f"{ev_current_hold.ev_components.out_of_range_bps:.2f}"
            if ev_current_hold and ev_current_hold.ev_components.out_of_range_bps is not None
            else "n/a",
            ev_current_hold.ev_components.out_of_range_cycles if ev_current_hold else "n/a",
            (
                str((ev_current_hold.ev_components.baseline_mode or "").startswith("current_hold_exact_bounds")).lower()
                if ev_current_hold
                else "n/a"
            ),
            f"${lifecycle_metrics.get('lifecycle_pnl_usd'):.4f}"
            if lifecycle_metrics.get("lifecycle_pnl_usd") is not None
            else "n/a",
            f"{100.0 * lifecycle_metrics.get('lifecycle_pnl_pct'):.2f}%"
            if lifecycle_metrics.get("lifecycle_pnl_pct") is not None
            else "n/a",
            rebalance_gate.get("policy_selected_override", "none"),
            f"${ev_delta_usd:.4f}" if ev_delta_usd is not None else "n/a",
            f"${effective_min_delta_usd:.4f}",
            realism_snapshot.sample_count,
            realism_snapshot.mode,
            selected_action,
            rebalance_gate.get("gate_mode"),
            rebalance_gate.get("protective_breach_count"),
            should_rebalance,
            should_close_to_idle,
            reason,
        )

        execution_result = None
        execution_bin_weight_plan: WeightedBinPlan | None = None
        execution_bin_quote: ExecutorRangeBinQuote | None = None
        execution_weight_objective = (
            self.settings.synth_weight_objective
            if self.settings.meteora_liquidity_mode == "synth_weights"
            else "hybrid"
        )
        if should_rebalance:
            execution_bin_weight_plan = best.weighted_bin_plan
            if (
                self.settings.meteora_liquidity_mode == "synth_weights"
                and self.settings.synth_weight_objective == "odds_15m_exact"
            ):
                try:
                    execution_bin_weight_plan, execution_bin_quote = self._compute_exact_execution_weight_plan(
                        pool=selected_pool,
                        forecast=best.forecast,
                        prediction_percentiles=prediction_percentiles,
                        lp_probabilities=lp_probabilities,
                    )
                except Exception as exc:
                    # Fail closed for synth_weights execution, but do not crash the loop.
                    should_rebalance = False
                    reason = "exact_execution_plan_unavailable"
                    rebalance_gate["execution_plan_available"] = False
                    rebalance_gate["execution_plan_error"] = str(exc)
                    LOGGER.error(
                        "Skipping rebalance this cycle (fail-closed): exact execution plan unavailable: %s",
                        exc,
                    )
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
                        target_bin_ids=execution_bin_quote.bin_ids if execution_bin_quote else None,
                        target_bin_edges=execution_bin_weight_plan.bin_edges if execution_bin_weight_plan else best.weighted_bin_plan.bin_edges,
                        target_bin_weights=execution_bin_weight_plan.weights if execution_bin_weight_plan else best.weighted_bin_plan.weights,
                    )
                )
                if execution_result.active_position is not None:
                    state.active_position = execution_result.active_position
                LOGGER.info("Execution completed: changed=%s txs=%s", execution_result.changed, execution_result.tx_signatures)
                _log_execution_details(execution_result.details)
                state.strategy_state["action_cooldown_remaining"] = int(self.settings.ev_action_cooldown_cycles)
        elif should_close_to_idle:
            execution_result = self.executor.close_position(state.active_position)
            state.active_position = execution_result.active_position
            LOGGER.info("Execution completed: changed=%s txs=%s", execution_result.changed, execution_result.tx_signatures)
            _log_execution_details(execution_result.details)
            state.strategy_state["action_cooldown_remaining"] = int(self.settings.ev_action_cooldown_cycles)

        post_snapshot_pool = self._pool_for_active_position(
            state=state,
            pools_by_address=pools_by_address,
            fallback_pool=selected_pool,
        )
        onchain_snapshot = pre_onchain_snapshot
        onchain_delta = pre_onchain_delta
        if execution_result is not None or onchain_snapshot is None:
            onchain_snapshot, onchain_delta = self._capture_onchain_snapshot(state=state, pool=post_snapshot_pool)

        if execution_result is not None and should_rebalance:
            lifecycle_metrics = self._update_position_lifecycle_state(
                state=state,
                active_pool=post_snapshot_pool,
                onchain_snapshot=onchain_snapshot,
                reset_baseline=True,
            )
        elif execution_result is not None and should_close_to_idle:
            self._reset_position_lifecycle_state(state)
            lifecycle_metrics = self._empty_lifecycle_metrics()

        state.last_decision = self._build_ev_last_decision(
            state=state,
            best=best,
            selected_pool=selected_pool,
            ev_current_hold=ev_current_hold,
            ev_delta_usd=ev_delta_usd,
            ev_candidates=ev_candidates,
            pre_scores=pre_scores,
            rebalance_gate=rebalance_gate,
            should_rebalance=should_rebalance,
            should_close_to_idle=should_close_to_idle,
            selected_action=selected_action,
            reason=reason,
            execution_result=execution_result,
            cost_model_info=cost_model_info,
            execution_bin_weight_plan=execution_bin_weight_plan,
            execution_bin_quote=execution_bin_quote,
            execution_weight_objective=execution_weight_objective,
            scoring_objective_used=scoring_objective_used,
            rescored_candidate_count=rescored_candidate_count,
            idle_candidate=idle_candidate,
            action_scores=action_scores,
            lifecycle_metrics=lifecycle_metrics,
            onchain_snapshot=onchain_snapshot,
            onchain_delta=onchain_delta,
            realism_snapshot=realism_snapshot,
            use_adjusted_for_decisions=use_adjusted_for_decisions,
            effective_min_delta_usd=effective_min_delta_usd,
        )
        self.state_store.save(state)
        return state.last_decision

    def _load_pool_candidates(self, *, state: BotState | None = None) -> list[MeteoraPoolSnapshot]:
        try:
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
        except Exception as exc:
            fallback = self._fallback_pool_candidates_from_state(state=state)
            if fallback:
                LOGGER.warning(
                    "Using stale pool snapshot fallback from local state due to Meteora API failure: %s",
                    exc,
                )
                return fallback
            synth_fallback = self._fallback_pool_candidates_from_synth(state=state, cause=exc)
            if synth_fallback:
                return synth_fallback
            raise

    def _fallback_pool_candidates_from_state(self, *, state: BotState | None) -> list[MeteoraPoolSnapshot]:
        if state is None or not isinstance(state.last_decision, dict):
            return []
        selected_pool = state.last_decision.get("selected_pool")
        if not isinstance(selected_pool, dict):
            selected_pool = state.last_decision.get("pool")
        if not isinstance(selected_pool, dict):
            return []

        address = str(selected_pool.get("address") or "").strip()
        if not address:
            return []
        if self.settings.meteora_pool_address and address != self.settings.meteora_pool_address:
            return []

        symbol_x = str(selected_pool.get("symbol_x") or "SOL").upper()
        symbol_y = str(selected_pool.get("symbol_y") or "USDC").upper()
        if {symbol_x, symbol_y} != {"SOL", "USDC"}:
            return []

        current_price = self._safe_float(selected_pool.get("current_price")) or 0.0
        tvl = self._safe_float(selected_pool.get("tvl")) or 0.0
        bin_step_bps = self._safe_float(selected_pool.get("bin_step_bps"))
        base_fee_pct = self._safe_float(selected_pool.get("base_fee_pct"))
        dynamic_fee_pct = self._safe_float(selected_pool.get("dynamic_fee_pct"))

        fee_tvl_ratio_24h: float | None = None
        ev_best = state.last_decision.get("ev_best_candidate")
        if isinstance(ev_best, dict):
            ev_components = ev_best.get("ev_components")
            if isinstance(ev_components, dict):
                fee_rate_15m = self._safe_float(ev_components.get("fee_rate_15m_fraction"))
                if fee_rate_15m is not None and fee_rate_15m > 0:
                    fee_tvl_ratio_24h = max(0.0, fee_rate_15m * (1440.0 / 15.0))

        fallback_pool = MeteoraPoolSnapshot(
            address=address,
            name=str(selected_pool.get("name") or "SOL-USDC (state-fallback)"),
            mint_x="",
            mint_y="",
            symbol_x=symbol_x,
            symbol_y=symbol_y,
            decimals_x=9 if symbol_x == "SOL" else 6,
            decimals_y=6 if symbol_y == "USDC" else 9,
            current_price=current_price,
            liquidity=tvl,
            volume_24h=0.0,
            fees_24h=0.0,
            fee_tvl_ratio_24h=fee_tvl_ratio_24h,
            raw={"fallback_source": "state.last_decision.selected_pool"},
            tvl=tvl,
            bin_step_bps=bin_step_bps,
            base_fee_pct=base_fee_pct,
            dynamic_fee_pct=dynamic_fee_pct,
            is_blacklisted=False,
        )
        return [fallback_pool]

    def _fallback_pool_candidates_from_synth(
        self,
        *,
        state: BotState | None,
        cause: Exception,
    ) -> list[MeteoraPoolSnapshot]:
        pool_address = str(self.settings.meteora_pool_address or "").strip()
        if not pool_address:
            return []

        spot = self._fetch_synth_spot_for_pool_fallback()
        if spot is None or spot <= 0:
            spot = self._fallback_spot_from_state(state=state)
        if spot is None or spot <= 0:
            return []

        selected_pool = self._extract_selected_pool_from_state(state=state)
        tvl = self._safe_float(selected_pool.get("tvl")) if isinstance(selected_pool, dict) else None
        bin_step_bps = self._safe_float(selected_pool.get("bin_step_bps")) if isinstance(selected_pool, dict) else None
        base_fee_pct = self._safe_float(selected_pool.get("base_fee_pct")) if isinstance(selected_pool, dict) else None
        dynamic_fee_pct = self._safe_float(selected_pool.get("dynamic_fee_pct")) if isinstance(selected_pool, dict) else None

        fee_tvl_ratio_24h: float | None = None
        if state is not None and isinstance(state.last_decision, dict):
            ev_best = state.last_decision.get("ev_best_candidate")
            if isinstance(ev_best, dict):
                ev_components = ev_best.get("ev_components")
                if isinstance(ev_components, dict):
                    fee_rate_15m = self._safe_float(ev_components.get("fee_rate_15m_fraction"))
                    if fee_rate_15m is not None and fee_rate_15m > 0:
                        fee_tvl_ratio_24h = max(0.0, fee_rate_15m * (1440.0 / 15.0))

        fallback_pool = MeteoraPoolSnapshot(
            address=pool_address,
            name=str((selected_pool or {}).get("name") or "SOL-USDC (synth-fallback)"),
            mint_x="",
            mint_y="",
            symbol_x="SOL",
            symbol_y="USDC",
            decimals_x=9,
            decimals_y=6,
            current_price=float(spot),
            liquidity=float(tvl or 0.0),
            volume_24h=0.0,
            fees_24h=0.0,
            fee_tvl_ratio_24h=fee_tvl_ratio_24h,
            raw={
                "fallback_source": "synth.spot",
                "fallback_cause": str(cause),
            },
            tvl=float(tvl or 0.0),
            bin_step_bps=bin_step_bps,
            base_fee_pct=base_fee_pct,
            dynamic_fee_pct=dynamic_fee_pct,
            is_blacklisted=False,
        )
        LOGGER.warning(
            "Using Synth-derived pool snapshot fallback for %s due to Meteora API failure: %s",
            pool_address,
            cause,
        )
        return [fallback_pool]

    def _fetch_synth_spot_for_pool_fallback(self) -> float | None:
        try:
            lp_probabilities = self.synth_client.get_lp_probabilities(
                asset=self.settings.synth_asset,
                horizon=self.settings.synth_horizon,
                days=self.settings.synth_days,
            )
        except Exception as exc:
            LOGGER.warning("Synth lp-probabilities spot fallback unavailable: %s", exc)
            return None
        spot = self._safe_float(getattr(lp_probabilities, "current_price", None))
        if spot is None or spot <= 0:
            return None
        return float(spot)

    def _fallback_spot_from_state(self, *, state: BotState | None) -> float | None:
        selected_pool = self._extract_selected_pool_from_state(state=state)
        if isinstance(selected_pool, dict):
            spot = self._safe_float(selected_pool.get("current_price"))
            if spot is not None and spot > 0:
                return float(spot)
        if state is not None and state.active_position is not None:
            lower = float(min(state.active_position.lower_price, state.active_position.upper_price))
            upper = float(max(state.active_position.lower_price, state.active_position.upper_price))
            mid = (lower + upper) / 2.0
            if mid > 0:
                return mid
        if state is not None and isinstance(state.last_decision, dict):
            chosen = state.last_decision.get("chosen")
            if isinstance(chosen, dict):
                lower = self._safe_float(chosen.get("lower_bound"))
                upper = self._safe_float(chosen.get("upper_bound"))
                if lower is not None and upper is not None and upper > 0:
                    return (lower + upper) / 2.0
        return None

    @staticmethod
    def _extract_selected_pool_from_state(*, state: BotState | None) -> dict[str, Any] | None:
        if state is None or not isinstance(state.last_decision, dict):
            return None
        selected_pool = state.last_decision.get("selected_pool")
        if isinstance(selected_pool, dict):
            return selected_pool
        pool = state.last_decision.get("pool")
        if isinstance(pool, dict):
            return pool
        return None

    def _scoring_uses_exact_objective(self) -> bool:
        return (
            self.settings.meteora_liquidity_mode == "synth_weights"
            and self.settings.synth_weight_objective == "odds_15m_exact"
        )

    def _rescore_candidates_exact(
        self,
        *,
        candidates: list[EvScoredCandidate],
        pools_by_address: dict[str, MeteoraPoolSnapshot],
        lp_probabilities,
        prediction_percentiles,
        width_ref_pct: float,
        bin_step_ref_bps: float,
        active_position,
        pre_score_by_addr: dict[str, Any],
    ) -> tuple[list[EvScoredCandidate], int]:
        if not candidates:
            return candidates, 0
        top_n = max(1, int(self.settings.ev_exact_rescoring_top_n))
        ranked = sorted(
            candidates,
            key=lambda c: (
                c.ev_15m_usd,
                c.ev_components.active_occupancy_15m,
                -c.forecast.width_pct,
            ),
            reverse=True,
        )
        to_rescore = ranked[: min(top_n, len(ranked))]
        rescored_by_key: dict[tuple[str, float, float], EvScoredCandidate] = {}
        rescored_count = 0
        for candidate in to_rescore:
            pool = pools_by_address.get(candidate.pool_address)
            if pool is None:
                continue
            try:
                exact_plan, _ = self._compute_exact_execution_weight_plan(
                    pool=pool,
                    forecast=candidate.forecast,
                    prediction_percentiles=prediction_percentiles,
                    lp_probabilities=lp_probabilities,
                )
                rescored = self.ev_scorer.score_pool_range_ev_15m(
                    pool=pool,
                    forecast=candidate.forecast,
                    weighted_bin_plan=exact_plan,
                    synth_horizon=self.settings.synth_horizon,
                    prediction_percentiles=prediction_percentiles,
                    deposit_sol_amount=self.settings.deposit_sol_amount,
                    deposit_usdc_amount=self.settings.deposit_usdc_amount,
                    width_ref_pct=width_ref_pct,
                    bin_step_ref_bps=bin_step_ref_bps,
                    active_position=active_position,
                    range_change_threshold_bps=self.settings.range_change_threshold_bps,
                    apply_rebalance_costs=True,
                    pre_rank_score=pre_score_by_addr.get(pool.address).score if pool.address in pre_score_by_addr else None,
                    action_type="rebalance",
                )
                rescored_by_key[_ev_candidate_key(candidate)] = rescored
                rescored_count += 1
            except Exception as exc:
                LOGGER.warning(
                    "Exact rescoring failed for pool=%s range=%.4f-%.4f; keeping provisional score: %s",
                    candidate.pool_address,
                    candidate.forecast.lower_bound,
                    candidate.forecast.upper_bound,
                    exc,
                )
        if rescored_count == 0:
            return candidates, 0
        merged = [rescored_by_key.get(_ev_candidate_key(c), c) for c in candidates]
        return merged, rescored_count

    def _apply_tx_cost_model(self, *, pools: list[MeteoraPoolSnapshot]) -> dict[str, Any]:
        assert self.ev_scorer is not None
        representative_spot = _representative_sol_spot(pools)
        fixed_cost_usd = float(self.settings.rebalance_cost_usd)

        info: dict[str, Any] = {
            "mode": self.settings.tx_cost_mode,
            "representative_spot_sol_usdc": representative_spot,
            "fixed_rebalance_cost_usd": fixed_cost_usd,
            "fixed_pool_switch_extra_cost_usd": float(self.settings.pool_switch_extra_cost_usd),
            "applied_rebalance_cost_usd": fixed_cost_usd,
            "applied_pool_switch_extra_cost_usd": float(self.settings.pool_switch_extra_cost_usd),
            "source": "fixed",
            "applied_total_fee_lamports": None,
            "configured_rebalance_cost_lamports": int(self.settings.rebalance_cost_lamports),
        }

        # Pool-switch extra cost remains a separate policy/friction knob.
        self.ev_scorer.pool_switch_extra_cost_usd = float(self.settings.pool_switch_extra_cost_usd)

        if self.settings.tx_cost_mode == "fixed":
            self.ev_scorer.rebalance_cost_usd = fixed_cost_usd
            return info

        # `observed` is retained as a backward-compatible alias for the new simple fixed-lamports mode.
        lamports = int(self.settings.rebalance_cost_lamports)
        rebalance_cost_usd = _lamports_to_usd(lamports, representative_spot)
        self.ev_scorer.rebalance_cost_usd = rebalance_cost_usd
        info["applied_rebalance_cost_usd"] = rebalance_cost_usd
        info["applied_total_fee_lamports"] = lamports
        info["source"] = "fixed_lamports"
        return info

    def _build_realism_snapshot(self) -> CalibrationSnapshot:
        snapshot = default_calibration_snapshot(
            fee_realism_prior=float(self.settings.ev_fee_realism_prior),
            fee_realism_min=float(self.settings.ev_fee_realism_min),
            fee_realism_max=float(self.settings.ev_fee_realism_max),
            rebalance_drag_prior_usd=float(self.settings.ev_rebalance_drag_prior_usd),
            rebalance_drag_min_usd=float(self.settings.ev_rebalance_drag_min_usd),
            rebalance_drag_max_usd=float(self.settings.ev_rebalance_drag_max_usd),
        )
        if not bool(self.settings.ev_realism_enabled):
            return snapshot
        try:
            return build_calibration_snapshot_from_journal(
                journal_path=self.settings.trade_journal_path,
                window_hours=int(self.settings.ev_realism_window_hours),
                min_samples=int(self.settings.ev_realism_min_samples),
                fee_realism_prior=float(self.settings.ev_fee_realism_prior),
                fee_realism_min=float(self.settings.ev_fee_realism_min),
                fee_realism_max=float(self.settings.ev_fee_realism_max),
                rebalance_drag_prior_usd=float(self.settings.ev_rebalance_drag_prior_usd),
                rebalance_drag_min_usd=float(self.settings.ev_rebalance_drag_min_usd),
                rebalance_drag_max_usd=float(self.settings.ev_rebalance_drag_max_usd),
            )
        except Exception as exc:
            LOGGER.warning("Realism calibration failed; using priors: %s", exc)
            return snapshot

    @staticmethod
    def _candidate_raw_ev(candidate: EvScoredCandidate) -> float:
        explicit_costs = float(candidate.ev_components.rebalance_cost_usd) + float(candidate.ev_components.pool_switch_extra_cost_usd)
        return float(candidate.ev_components.expected_fees_usd) - float(candidate.ev_components.expected_il_usd) - explicit_costs

    def _apply_realism_adjustments(
        self,
        *,
        candidate: EvScoredCandidate,
        snapshot: CalibrationSnapshot,
        apply_to_decision: bool,
    ) -> None:
        raw_fees = float(candidate.ev_components.expected_fees_usd)
        raw_ev = self._candidate_raw_ev(candidate)
        explicit_costs = float(candidate.ev_components.rebalance_cost_usd) + float(candidate.ev_components.pool_switch_extra_cost_usd)
        fee_multiplier = float(snapshot.fee_realism_multiplier)
        adjusted_fees = raw_fees * fee_multiplier
        execution_drag = float(snapshot.rebalance_drag_usd) if candidate.action_type == "rebalance" else 0.0
        uncertainty_penalty = float(self.settings.ev_uncertainty_k) * float(snapshot.model_rmse_usd)
        adjusted_ev = adjusted_fees - float(candidate.ev_components.expected_il_usd) - explicit_costs - execution_drag - uncertainty_penalty

        candidate.ev_components.raw_expected_fees_usd = raw_fees
        candidate.ev_components.fee_realism_multiplier = fee_multiplier
        candidate.ev_components.adjusted_expected_fees_usd = adjusted_fees
        candidate.ev_components.execution_drag_usd = execution_drag
        candidate.ev_components.uncertainty_penalty_usd = uncertainty_penalty
        candidate.ev_components.adjusted_ev_15m_usd = adjusted_ev
        candidate.ev_components.calibration_sample_count = int(snapshot.sample_count)
        candidate.ev_components.calibration_mode = snapshot.mode

        if apply_to_decision:
            candidate.ev_15m_usd = float(adjusted_ev)
        else:
            candidate.ev_15m_usd = float(raw_ev)

    @staticmethod
    def _candidate_effective_ev(candidate: EvScoredCandidate, *, use_adjusted: bool) -> float:
        if use_adjusted and candidate.ev_components.adjusted_ev_15m_usd is not None:
            return float(candidate.ev_components.adjusted_ev_15m_usd)
        return float(candidate.ev_15m_usd)

    def _effective_min_delta_usd(self, snapshot: CalibrationSnapshot, *, use_adjusted: bool) -> float:
        base = float(self.settings.min_ev_improvement_usd)
        if not use_adjusted:
            return base
        if not bool(self.settings.ev_dynamic_gate_enabled):
            return base
        dynamic = float(self.settings.ev_dynamic_gate_base_margin_usd) + (
            float(self.settings.ev_dynamic_gate_sigma_mult) * float(snapshot.model_rmse_usd)
        )
        return max(base, dynamic)

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
            max_single_bin=self.settings.synth_weight_max_single_bin,
            max_top3=self.settings.synth_weight_max_top3,
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

    def _compute_exact_execution_weight_plan(
        self,
        *,
        pool: MeteoraPoolSnapshot,
        forecast,
        prediction_percentiles,
        lp_probabilities,
    ) -> tuple[WeightedBinPlan, ExecutorRangeBinQuote]:
        quote = self.executor.quote_target_bins(
            pool_address=pool.address,
            symbol_x=pool.symbol_x,
            symbol_y=pool.symbol_y,
            api_current_price=pool.current_price,
            target_lower_price=forecast.lower_bound,
            target_upper_price=forecast.upper_bound,
        )
        if quote is None:
            failure_reason = None
            if hasattr(self.executor, "last_quote_failure_reason"):
                try:
                    candidate_reason = self.executor.last_quote_failure_reason()
                except Exception:
                    candidate_reason = None
                if candidate_reason:
                    failure_reason = str(candidate_reason)
            raise RuntimeError(
                "synth_weights + odds_15m_exact requires executor bin quote support, "
                "but quote_target_bins returned no data"
                + (f" ({failure_reason})" if failure_reason else "")
            )
        if not quote.bin_ids or not quote.bin_prices_sol_usdc:
            raise RuntimeError("Executor bin quote returned empty bin_ids/bin_prices_sol_usdc")
        if len(quote.bin_ids) != len(quote.bin_prices_sol_usdc):
            raise RuntimeError(
                "Executor bin quote length mismatch: "
                f"ids={len(quote.bin_ids)} prices={len(quote.bin_prices_sol_usdc)}"
            )
        quote_min = min(quote.bin_prices_sol_usdc)
        quote_max = max(quote.bin_prices_sol_usdc)
        if quote_max < (forecast.lower_bound / 10.0) or quote_min > (forecast.upper_bound * 10.0):
            raise RuntimeError(
                "Executor bin quote appears to be on the wrong price scale: "
                f"quote_min={quote_min:.8f}, quote_max={quote_max:.8f}, "
                f"forecast_range=[{forecast.lower_bound:.8f},{forecast.upper_bound:.8f}]"
            )
        weighted_bin_plan = compute_exact_sdk_bin_odds_weight_plan(
            range_lower=forecast.lower_bound,
            range_upper=forecast.upper_bound,
            sdk_bin_prices_sol_usdc=quote.bin_prices_sol_usdc,
            prediction_percentiles=prediction_percentiles,
            lp_probabilities=lp_probabilities,
            ev_horizon_minutes=self.settings.ev_horizon_minutes,
            tau_half_minutes=self.settings.ev_percentile_decay_half_life_minutes,
            beta=self.settings.synth_weight_odds_beta,
            eps=self.settings.synth_weight_odds_eps,
            max_single_bin=self.settings.synth_weight_max_single_bin,
            max_top3=self.settings.synth_weight_max_top3,
        )
        if len(weighted_bin_plan.weights) != len(quote.bin_ids):
            raise RuntimeError(
                "Exact execution weight plan length mismatch: "
                f"weights={len(weighted_bin_plan.weights)} quoted_bins={len(quote.bin_ids)}"
            )
        LOGGER.info(
            (
                "Computed exact SDK-bin execution weights: bins=%d objective=odds_15m_exact "
                "source=%s path=%s fallback=%s max_w=%.4f"
            ),
            len(weighted_bin_plan.weights),
            weighted_bin_plan.distribution_components.get("source"),
            weighted_bin_plan.diagnostics.used_prediction_percentiles,
            weighted_bin_plan.diagnostics.fallback_reason,
            max(weighted_bin_plan.weights) if weighted_bin_plan.weights else 0.0,
        )
        return weighted_bin_plan, quote

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
        scoring_objective: str,
        hold_oor_cycles: int,
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
            nearest_forecast = self.ev_scorer.nearest_forecast_to_active_range(forecasts, active)
            hold_forecast = self._build_hold_forecast_from_active(active=active, nearest_forecast=nearest_forecast)
            used_exact_bounds = False
            if scoring_objective == "odds_15m_exact":
                try:
                    weighted_bin_plan, _ = self._compute_exact_execution_weight_plan(
                        pool=pool,
                        forecast=hold_forecast,
                        prediction_percentiles=prediction_percentiles,
                        lp_probabilities=lp_probabilities,
                    )
                    used_exact_bounds = True
                except Exception as exc:
                    LOGGER.warning(
                        "Exact hold scoring on active bounds unavailable; falling back to hybrid hold bins: %s",
                        exc,
                    )
                    weighted_bin_plan = self._compute_weighted_bin_plan(
                        pool=pool,
                        lower=hold_forecast.lower_bound,
                        upper=hold_forecast.upper_bound,
                        forecast=hold_forecast,
                        lp_probabilities=lp_probabilities,
                        prediction_percentiles=prediction_percentiles,
                    )
            else:
                weighted_bin_plan = self._compute_weighted_bin_plan(
                    pool=pool,
                    lower=hold_forecast.lower_bound,
                    upper=hold_forecast.upper_bound,
                    forecast=hold_forecast,
                    lp_probabilities=lp_probabilities,
                    prediction_percentiles=prediction_percentiles,
                )
                used_exact_bounds = True
            return self.ev_scorer.score_pool_range_ev_15m(
                pool=pool,
                forecast=hold_forecast,
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
                baseline_mode="current_hold_exact_bounds" if used_exact_bounds else "current_hold_hybrid_bounds",
                pre_rank_score=pre_score_by_addr.get(pool.address).score if pool.address in pre_score_by_addr else None,
                out_of_range_cycles=hold_oor_cycles,
                action_type="hold",
            )
        except Exception as exc:
            LOGGER.warning("Failed to compute EV hold baseline; proceeding without it: %s", exc)
            return None

    def _compute_idle_ev_candidate(
        self,
        *,
        state: BotState,
        pools_by_address: dict[str, MeteoraPoolSnapshot],
        fallback_pool: MeteoraPoolSnapshot,
    ) -> EvScoredCandidate:
        active = state.active_position
        pool = fallback_pool
        if active is not None and active.pool_address in pools_by_address:
            pool = pools_by_address[active.pool_address]
        spot = max(pool.current_price_sol_usdc(), 0.0)
        if active is not None:
            lower = min(active.lower_price, active.upper_price)
            upper = max(active.lower_price, active.upper_price)
        else:
            lower = max(spot * 0.999, 1e-9)
            upper = max(spot * 1.001, lower * (1.0 + 1e-9))
        if upper <= lower:
            upper = lower * (1.0 + 1e-9)
        width_pct = ((upper - lower) / ((upper + lower) * 0.5)) * 100.0
        forecast = SynthLpBoundForecast(
            width_pct=width_pct,
            lower_bound=lower,
            upper_bound=upper,
            probability_to_stay_in_interval=0.0,
            expected_time_in_interval_minutes=0.0,
            expected_impermanent_loss=0.0,
        )
        plan = WeightedBinPlan(
            range_lower=lower,
            range_upper=upper,
            bin_edges=[lower, upper],
            weights=[1.0],
            diagnostics=BinWeightingDiagnostics(
                mass_in_range=0.0,
                used_prediction_percentiles=False,
                fallback_reason="idle_action",
                confidence_factor=1.0,
                t_frac=0.0,
                entropy=0.0,
                terminal_cdf_points=0,
                num_bins=1,
                binning_mode="idle",
            ),
            distribution_components={"source": "idle"},
        )
        close_cost = self.ev_scorer.rebalance_cost_usd if active is not None else 0.0
        components = EvComponentBreakdown(
            expected_fees_usd=0.0,
            expected_il_usd=0.0,
            rebalance_cost_usd=float(close_cost),
            pool_switch_extra_cost_usd=0.0,
            active_occupancy_15m=0.0,
            range_active_occupancy_15m=0.0,
            concentration_factor=1.0,
            fee_rate_15m_fraction=0.0,
            capital_usd=self.ev_scorer.compute_capital_usd(
                pool,
                self.settings.deposit_sol_amount,
                self.settings.deposit_usdc_amount,
            ),
            il_15m_fraction=0.0,
            utilization_ratio=0.0,
            fee_capture_factor=1.0,
            occupancy_source="idle",
            baseline_mode="idle",
            il_baseline_usd=0.0,
            il_state_penalty_usd=0.0,
            out_of_range_bps=0.0,
            out_of_range_cycles=0,
            out_of_range_penalty_fraction_15m=0.0,
            il_multiplier=1.0,
            p50_drift_bps=0.0,
            out_of_range_prob_15m=0.0,
            one_sided_break_prob=0.0,
            directional_confidence=0.0,
            expected_out_of_range_minutes_15m=0.0,
        )
        return EvScoredCandidate(
            pool_address=pool.address,
            pool_name=pool.name,
            pool_symbol_pair=f"{pool.symbol_x}/{pool.symbol_y}",
            pool_current_price_sol_usdc=pool.current_price_sol_usdc(),
            forecast=forecast,
            weighted_bin_plan=plan,
            ev_15m_usd=float(-close_cost),
            ev_components=components,
            pre_rank_score=None,
            rebalance_structural_change=active is not None,
            pool_switch=False,
            range_change_bps_vs_active=None,
            action_type="idle",
        )

    def _decide_ev_action(
        self,
        *,
        state: BotState,
        best: EvScoredCandidate,
        ev_current_hold: EvScoredCandidate | None,
        idle_candidate: EvScoredCandidate,
        prior_protective_breach_count: int,
        hold_oor_cycles: int,
        action_cooldown_remaining: int,
        lifecycle_metrics: dict[str, Any] | None = None,
        use_adjusted_ev: bool = False,
        effective_min_delta_usd: float | None = None,
    ) -> tuple[str, bool, bool, str, dict[str, Any], int, float, dict[str, float | None]]:
        strategy_state = state.strategy_state
        lifecycle_metrics = lifecycle_metrics or self._empty_lifecycle_metrics()
        lifecycle_pnl_usd = self._safe_float(lifecycle_metrics.get("lifecycle_pnl_usd"))
        lifecycle_pnl_pct = self._safe_float(lifecycle_metrics.get("lifecycle_pnl_pct"))
        hold_ev = self._candidate_effective_ev(ev_current_hold, use_adjusted=use_adjusted_ev) if ev_current_hold is not None else None
        rebalance_ev = self._candidate_effective_ev(best, use_adjusted=use_adjusted_ev)
        idle_ev = self._candidate_effective_ev(idle_candidate, use_adjusted=use_adjusted_ev)
        configured_min_delta = float(self.settings.min_ev_improvement_usd)
        effective_min_delta = (
            float(effective_min_delta_usd)
            if effective_min_delta_usd is not None
            else configured_min_delta
        )
        action_scores: dict[str, float | None] = {
            "hold": hold_ev,
            "rebalance": rebalance_ev,
            "idle": idle_ev,
        }

        idle_entry_count = int(strategy_state.get("idle_entry_confirm_count", 0) or 0)
        idle_exit_count = int(strategy_state.get("idle_exit_confirm_count", 0) or 0)
        strategy_state["loss_exit_breach_count"] = int(strategy_state.get("loss_exit_breach_count", 0) or 0)

        if state.active_position is None:
            strategy_state["idle_entry_confirm_count"] = 0
            strategy_state["loss_exit_breach_count"] = 0
            open_delta = rebalance_ev - idle_ev
            idle_open_threshold = max(float(self.settings.ev_idle_exit_threshold_usd), effective_min_delta)
            if self.settings.ev_idle_enabled and open_delta >= idle_open_threshold:
                idle_exit_count += 1
            else:
                idle_exit_count = 0
            strategy_state["idle_exit_confirm_count"] = idle_exit_count
            gate = {
                "baseline_present": False,
                "ev_delta_usd": open_delta,
                "min_ev_improvement_usd": configured_min_delta,
                "effective_min_delta_usd": effective_min_delta,
                "ev_threshold_passed": open_delta >= idle_open_threshold,
                "structural_change_passed": True,
                "pool_switch": best.pool_switch,
                "range_change_bps_vs_active": best.range_change_bps_vs_active,
                "gate_mode": "idle_exit" if self.settings.ev_idle_enabled else "ev_only",
                "protective_breach_count": 0,
                "protective_gate_details": None,
                "action_cooldown_remaining": action_cooldown_remaining,
                "idle_exit_confirm_count": idle_exit_count,
                "idle_exit_confirm_needed": int(self.settings.ev_idle_confirm_cycles),
                "selected_action": "idle",
                "lifecycle_pnl_usd": lifecycle_pnl_usd,
                "lifecycle_pnl_pct": lifecycle_pnl_pct,
                "lifecycle_open_position_usd": lifecycle_metrics.get("lifecycle_open_position_usd"),
                "policy_profit_triggered": False,
                "policy_loss_recovery_hold_triggered": False,
                "policy_loss_trend_cut_triggered": False,
                "policy_recovery_prob": None,
                "policy_trend_prob": None,
                "policy_loss_exit_breach_count": 0,
                "policy_selected_override": "none",
                "policy_oor_grace_triggered": False,
                "policy_oor_reentry_hold_triggered": False,
                "policy_reentry_prob": None,
                "use_adjusted_ev": use_adjusted_ev,
            }
            if not self.settings.ev_idle_enabled:
                if (
                    use_adjusted_ev
                    and bool(getattr(self.settings, "ev_adjusted_ev_require_positive", True))
                    and rebalance_ev <= 0.0
                ):
                    gate["selected_action"] = "idle"
                    gate["adjusted_ev_positive_required"] = True
                    gate["adjusted_ev_positive_passed"] = False
                    return "idle", False, False, "adjusted_ev_nonpositive", gate, 0, open_delta, action_scores
                gate["selected_action"] = "rebalance"
                return "rebalance", True, False, "no_active_position", gate, 0, open_delta, action_scores
            if action_cooldown_remaining > 0:
                return (
                    "idle",
                    False,
                    False,
                    f"action_cooldown_{action_cooldown_remaining}",
                    gate,
                    0,
                    open_delta,
                    action_scores,
                )
            if idle_exit_count >= int(self.settings.ev_idle_confirm_cycles):
                if (
                    use_adjusted_ev
                    and bool(getattr(self.settings, "ev_adjusted_ev_require_positive", True))
                    and rebalance_ev <= 0.0
                ):
                    gate["selected_action"] = "idle"
                    gate["adjusted_ev_positive_required"] = True
                    gate["adjusted_ev_positive_passed"] = False
                    return "idle", False, False, "adjusted_ev_nonpositive", gate, 0, open_delta, action_scores
                gate["selected_action"] = "rebalance"
                return "rebalance", True, False, f"idle_exit_open_{open_delta:.4f}", gate, 0, open_delta, action_scores
            return (
                "idle",
                False,
                False,
                f"idle_wait_open_{idle_exit_count}/{int(self.settings.ev_idle_confirm_cycles)}_{open_delta:.4f}",
                gate,
                0,
                open_delta,
                action_scores,
            )

        strategy_state["idle_exit_confirm_count"] = 0
        delta_vs_hold = rebalance_ev if hold_ev is None else (rebalance_ev - hold_ev)
        should_rebalance, rebalance_reason, rebalance_gate, protective_breach_count = _should_rebalance_ev(
            state=state,
            best=best,
            ev_current_hold=ev_current_hold,
            ev_delta_usd=delta_vs_hold,
            min_ev_improvement_usd=effective_min_delta,
            prior_protective_breach_count=prior_protective_breach_count,
            ev_utilization_floor=self.settings.ev_utilization_floor,
            ev_min_utilization_gain=self.settings.ev_min_utilization_gain,
            ev_max_protective_ev_slip_usd=self.settings.ev_max_protective_ev_slip_usd,
            ev_protective_breach_cycles=self.settings.ev_protective_breach_cycles,
        )
        rebalance_gate["configured_min_ev_improvement_usd"] = configured_min_delta
        rebalance_gate["effective_min_delta_usd"] = effective_min_delta
        rebalance_gate["use_adjusted_ev"] = use_adjusted_ev
        if (
            not should_rebalance
            and str(rebalance_reason).startswith("ev_delta_below_threshold_")
            and effective_min_delta > configured_min_delta
        ):
            rebalance_reason = f"delta_below_dynamic_gate_{delta_vs_hold:.4f}"
            rebalance_gate["dynamic_gate_blocked"] = True
        else:
            rebalance_gate["dynamic_gate_blocked"] = False
        if action_cooldown_remaining > 0 and should_rebalance:
            should_rebalance = False
            rebalance_reason = f"action_cooldown_{action_cooldown_remaining}"

        hold_one_sided = (
            ev_current_hold.ev_components.one_sided_break_prob
            if ev_current_hold and ev_current_hold.ev_components.one_sided_break_prob is not None
            else 0.0
        )
        trend_stop = bool(
            self.settings.ev_trend_stop_enabled
            and hold_oor_cycles >= int(self.settings.ev_trend_stop_oor_cycles)
            and hold_one_sided >= float(self.settings.ev_trend_stop_onesided_prob)
        )

        idle_delta = idle_ev - (hold_ev if hold_ev is not None else 0.0)
        if self.settings.ev_idle_enabled and idle_delta >= float(self.settings.ev_idle_entry_threshold_usd):
            idle_entry_count += 1
        else:
            idle_entry_count = 0
        strategy_state["idle_entry_confirm_count"] = idle_entry_count

        idle_ready = bool(
            self.settings.ev_idle_enabled
            and idle_entry_count >= int(self.settings.ev_idle_confirm_cycles)
        )
        if trend_stop and self.settings.ev_idle_enabled:
            idle_ready = True
        if action_cooldown_remaining > 0 and idle_ready and not trend_stop:
            idle_ready = False

        options: list[tuple[str, float]] = []
        if hold_ev is not None:
            options.append(("hold", hold_ev))
        if should_rebalance:
            options.append(("rebalance", rebalance_ev))
        if idle_ready:
            options.append(("idle", idle_ev))
        if not options:
            options = [("hold", hold_ev if hold_ev is not None else rebalance_ev)]
        selected_action, selected_ev = max(options, key=lambda t: t[1])
        selected_reason = rebalance_reason
        selected_should_rebalance = False
        selected_should_close_to_idle = False
        selected_protective_breach = protective_breach_count
        if selected_action == "rebalance":
            selected_should_rebalance = True
            selected_reason = rebalance_reason
            selected_protective_breach = 0
        elif selected_action == "idle":
            selected_should_close_to_idle = True
            selected_reason = "trend_stop_to_idle" if trend_stop else f"idle_ev_gain_{idle_delta:.4f}"
            selected_protective_breach = 0
        else:
            selected_reason = rebalance_reason
            if should_rebalance and selected_ev >= rebalance_ev:
                selected_reason = "hold_beats_rebalance_ev"
            elif idle_ready and selected_ev >= idle_ev:
                selected_reason = "hold_beats_idle_ev"

        policy_profit_triggered = False
        policy_loss_recovery_hold_triggered = False
        policy_loss_trend_cut_triggered = False
        policy_oor_grace_triggered = False
        policy_oor_reentry_hold_triggered = False
        policy_recovery_prob: float | None = None
        policy_trend_prob: float | None = None
        policy_reentry_prob: float | None = None
        policy_selected_override = "none"
        loss_exit_breach_count = int(strategy_state.get("loss_exit_breach_count", 0) or 0)
        if self.settings.ev_policy_lifecycle_enabled and ev_current_hold is not None and self.ev_scorer is not None:
            hold_components = ev_current_hold.ev_components
            hold_il_15m = float(hold_components.il_15m_fraction or 0.0)
            hold_occ = float(
                hold_components.range_active_occupancy_15m
                if hold_components.range_active_occupancy_15m is not None
                else hold_components.active_occupancy_15m
            )
            hold_one_sided_prob = float(hold_components.one_sided_break_prob or 0.0)
            hold_directional_conf = float(hold_components.directional_confidence or 0.0)
            policy_recovery_prob = self.ev_scorer.compute_recovery_probability(
                range_active_occupancy_15m=hold_occ,
                one_sided_break_prob=hold_one_sided_prob,
                directional_confidence=hold_directional_conf,
            )
            policy_trend_prob = self.ev_scorer.compute_trend_continuation_probability(
                one_sided_break_prob=hold_one_sided_prob,
                directional_confidence=hold_directional_conf,
            )
            has_loss_trigger = (
                lifecycle_pnl_pct is not None
                and lifecycle_pnl_pct <= -float(self.settings.ev_loss_recovery_trigger_pct)
                and hold_il_15m >= float(self.settings.ev_high_il_15m_fraction)
            )
            if has_loss_trigger and policy_trend_prob >= float(self.settings.ev_trend_continue_exit_prob):
                loss_exit_breach_count += 1
            else:
                loss_exit_breach_count = 0

            if (
                lifecycle_pnl_pct is not None
                and lifecycle_pnl_pct >= float(self.settings.ev_profit_take_pct)
                and bool(best.rebalance_structural_change)
                and hold_ev is not None
                and rebalance_ev >= (hold_ev - float(self.settings.ev_profit_rebalance_max_slip_usd))
            ):
                policy_profit_triggered = True
                if selected_action == "hold":
                    selected_action = "rebalance"
                    selected_should_rebalance = True
                    selected_should_close_to_idle = False
                    selected_reason = "profit_lock_rotate"
                    selected_protective_breach = 0
                    policy_selected_override = "profit_lock"

            if has_loss_trigger and policy_recovery_prob >= float(self.settings.ev_recovery_min_prob):
                policy_loss_recovery_hold_triggered = True
                if selected_action != "hold":
                    selected_action = "hold"
                    selected_should_rebalance = False
                    selected_should_close_to_idle = False
                    selected_reason = "loss_recovery_hold"
                    selected_protective_breach = protective_breach_count
                    policy_selected_override = "loss_hold"

            if (
                has_loss_trigger
                and policy_trend_prob >= float(self.settings.ev_trend_continue_exit_prob)
                and loss_exit_breach_count >= int(self.settings.ev_trend_exit_persistence_cycles)
            ):
                policy_loss_trend_cut_triggered = True
                prefer_idle = (
                    bool(self.settings.ev_idle_enabled)
                    and idle_ev >= (rebalance_ev + float(self.settings.ev_idle_preference_edge_usd))
                )
                if prefer_idle:
                    selected_action = "idle"
                    selected_should_rebalance = False
                    selected_should_close_to_idle = True
                    selected_reason = "loss_trend_cut_idle"
                    policy_selected_override = "loss_cut_idle"
                else:
                    selected_action = "rebalance"
                    selected_should_rebalance = True
                    selected_should_close_to_idle = False
                    selected_reason = "loss_trend_cut_rotate"
                    policy_selected_override = "loss_cut_rotate"
                selected_protective_breach = 0
                loss_exit_breach_count = 0
        else:
            loss_exit_breach_count = 0

        if self.settings.ev_policy_lifecycle_enabled and ev_current_hold is not None:
            hold_components = ev_current_hold.ev_components
            hold_out_bps = float(hold_components.out_of_range_bps or 0.0)
            hold_out_prob = float(hold_components.out_of_range_prob_15m or 0.0)
            policy_reentry_prob = max(0.0, min(1.0, 1.0 - hold_out_prob))
            oor_deadband = float(getattr(self.settings, "ev_oor_deadband_bps", 0.0))
            is_out_of_range = bool(hold_oor_cycles > 0 or hold_out_bps > oor_deadband)
            loss_trigger_pct = float(getattr(self.settings, "ev_oor_loss_trigger_pct", 0.0))
            is_lifecycle_loss = (
                lifecycle_pnl_pct is not None
                and lifecycle_pnl_pct <= -loss_trigger_pct
            )
            if (
                bool(getattr(self.settings, "ev_oor_grace_enabled", True))
                and selected_action in {"rebalance", "idle"}
                and is_out_of_range
                and is_lifecycle_loss
            ):
                grace_cycles = max(1, int(getattr(self.settings, "ev_oor_grace_cycles", 2)))
                reentry_min_prob = float(getattr(self.settings, "ev_oor_reentry_min_prob", 0.60))
                if hold_oor_cycles < grace_cycles:
                    policy_oor_grace_triggered = True
                    selected_action = "hold"
                    selected_should_rebalance = False
                    selected_should_close_to_idle = False
                    selected_reason = f"oor_grace_hold_{hold_oor_cycles}/{grace_cycles}"
                    selected_protective_breach = protective_breach_count
                    policy_selected_override = "oor_grace_hold"
                elif policy_reentry_prob >= reentry_min_prob:
                    policy_oor_reentry_hold_triggered = True
                    selected_action = "hold"
                    selected_should_rebalance = False
                    selected_should_close_to_idle = False
                    selected_reason = f"oor_reentry_hold_{policy_reentry_prob:.3f}"
                    selected_protective_breach = protective_breach_count
                    policy_selected_override = "oor_reentry_hold"

        adjusted_ev_positive_required = bool(getattr(self.settings, "ev_adjusted_ev_require_positive", True))
        adjusted_ev_positive_passed = True
        if (
            use_adjusted_ev
            and adjusted_ev_positive_required
            and selected_action == "rebalance"
            and rebalance_ev <= 0.0
        ):
            adjusted_ev_positive_passed = False
            hold_candidate_ev = hold_ev if hold_ev is not None else float("-inf")
            if idle_ready and idle_ev > hold_candidate_ev:
                selected_action = "idle"
                selected_should_rebalance = False
                selected_should_close_to_idle = True
            else:
                selected_action = "hold"
                selected_should_rebalance = False
                selected_should_close_to_idle = False
            selected_reason = "adjusted_ev_nonpositive"
            selected_protective_breach = 0

        strategy_state["loss_exit_breach_count"] = loss_exit_breach_count

        rebalance_gate["action_cooldown_remaining"] = action_cooldown_remaining
        rebalance_gate["idle_entry_confirm_count"] = idle_entry_count
        rebalance_gate["idle_entry_confirm_needed"] = int(self.settings.ev_idle_confirm_cycles)
        rebalance_gate["idle_delta_vs_hold"] = idle_delta
        rebalance_gate["trend_stop_condition"] = trend_stop
        rebalance_gate["selected_action"] = selected_action
        rebalance_gate["lifecycle_pnl_usd"] = lifecycle_pnl_usd
        rebalance_gate["lifecycle_pnl_pct"] = lifecycle_pnl_pct
        rebalance_gate["lifecycle_open_position_usd"] = lifecycle_metrics.get("lifecycle_open_position_usd")
        rebalance_gate["policy_profit_triggered"] = policy_profit_triggered
        rebalance_gate["policy_loss_recovery_hold_triggered"] = policy_loss_recovery_hold_triggered
        rebalance_gate["policy_loss_trend_cut_triggered"] = policy_loss_trend_cut_triggered
        rebalance_gate["policy_recovery_prob"] = policy_recovery_prob
        rebalance_gate["policy_trend_prob"] = policy_trend_prob
        rebalance_gate["policy_reentry_prob"] = policy_reentry_prob
        rebalance_gate["policy_loss_exit_breach_count"] = loss_exit_breach_count
        rebalance_gate["policy_selected_override"] = policy_selected_override
        rebalance_gate["policy_oor_grace_triggered"] = policy_oor_grace_triggered
        rebalance_gate["policy_oor_reentry_hold_triggered"] = policy_oor_reentry_hold_triggered
        rebalance_gate["adjusted_ev_positive_required"] = adjusted_ev_positive_required if use_adjusted_ev else False
        rebalance_gate["adjusted_ev_positive_passed"] = adjusted_ev_positive_passed if use_adjusted_ev else True
        if policy_selected_override != "none":
            rebalance_gate["gate_mode"] = "policy_lifecycle"
        return (
            selected_action,
            selected_should_rebalance,
            selected_should_close_to_idle,
            selected_reason,
            rebalance_gate,
            selected_protective_breach,
            delta_vs_hold,
            action_scores,
        )

    def _build_hold_forecast_from_active(
        self,
        *,
        active: ActivePositionState,
        nearest_forecast: SynthLpBoundForecast,
    ) -> SynthLpBoundForecast:
        lower = min(float(active.lower_price), float(active.upper_price))
        upper = max(float(active.lower_price), float(active.upper_price))
        mid = (lower + upper) / 2.0
        width_pct = ((upper - lower) / mid) * 100.0 if mid > 0 else 0.0
        return SynthLpBoundForecast(
            width_pct=width_pct,
            lower_bound=lower,
            upper_bound=upper,
            probability_to_stay_in_interval=nearest_forecast.probability_to_stay_in_interval,
            expected_time_in_interval_minutes=nearest_forecast.expected_time_in_interval_minutes,
            expected_impermanent_loss=nearest_forecast.expected_impermanent_loss,
        )

    def _update_oor_hold_state(
        self,
        *,
        state: BotState,
        pools_by_address: dict[str, MeteoraPoolSnapshot],
    ) -> int:
        strategy_state = state.strategy_state
        active = state.active_position
        if active is None:
            strategy_state["oor_hold_cycles"] = 0
            strategy_state["last_active_spot"] = None
            strategy_state["last_active_in_range"] = None
            return 0

        pool = pools_by_address.get(active.pool_address)
        if pool is None:
            try:
                pool = self.meteora_client.get_pool(active.pool_address)
                pools_by_address[pool.address] = pool
            except Exception as exc:
                LOGGER.warning("Failed to load active pool for OOR state update (%s): %s", active.pool_address, exc)
                return int(strategy_state.get("oor_hold_cycles", 0) or 0)

        spot = pool.current_price_sol_usdc()
        out_bps = self.ev_scorer.out_of_range_bps(
            spot=spot,
            lower=active.lower_price,
            upper=active.upper_price,
        )
        in_range = out_bps <= 0.0
        prev = int(strategy_state.get("oor_hold_cycles", 0) or 0)
        if out_bps > self.settings.ev_oor_deadband_bps:
            cycles = prev + 1
        else:
            cycles = 0
        strategy_state["oor_hold_cycles"] = cycles
        strategy_state["last_active_spot"] = spot
        strategy_state["last_active_in_range"] = in_range
        return cycles

    def _pool_for_active_position(
        self,
        *,
        state: BotState,
        pools_by_address: dict[str, MeteoraPoolSnapshot],
        fallback_pool: MeteoraPoolSnapshot,
    ) -> MeteoraPoolSnapshot:
        active = state.active_position
        if active is None:
            return fallback_pool
        pool = pools_by_address.get(active.pool_address)
        if pool is not None:
            return pool
        try:
            pool = self.meteora_client.get_pool(active.pool_address)
            pools_by_address[pool.address] = pool
            return pool
        except Exception as exc:
            LOGGER.warning("Failed to load active pool for snapshot/lifecycle (%s): %s", active.pool_address, exc)
            return fallback_pool

    def _empty_lifecycle_metrics(self) -> dict[str, Any]:
        return {
            "lifecycle_position_id": None,
            "lifecycle_pnl_usd": None,
            "lifecycle_pnl_pct": None,
            "lifecycle_open_position_usd": None,
            "lifecycle_open_total_usd": None,
            "lifecycle_current_position_usd": None,
            "lifecycle_current_total_usd": None,
            "lifecycle_valuation_source": None,
        }

    def _reset_position_lifecycle_state(self, state: BotState) -> None:
        strategy_state = state.strategy_state
        for key in (
            "position_lifecycle_position_pubkey",
            "position_lifecycle_opened_at",
            "position_lifecycle_open_position_usd",
            "position_lifecycle_open_total_usd",
            "position_lifecycle_last_position_usd",
            "position_lifecycle_last_total_usd",
            "position_lifecycle_last_source",
            "position_lifecycle_last_pnl_usd",
            "position_lifecycle_last_pnl_pct",
            "position_lifecycle_last_snapshot_at",
        ):
            strategy_state.pop(key, None)
        strategy_state["loss_exit_breach_count"] = 0

    @staticmethod
    def _position_lifecycle_id(active: ActivePositionState) -> str:
        if active.position_pubkey:
            return str(active.position_pubkey)
        lower = min(float(active.lower_price), float(active.upper_price))
        upper = max(float(active.lower_price), float(active.upper_price))
        return f"{active.pool_address}:{lower:.8f}:{upper:.8f}"

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _extract_onchain_equity_values(cls, onchain_snapshot: dict[str, Any] | None) -> tuple[float | None, float | None]:
        if not isinstance(onchain_snapshot, dict):
            return None, None
        position_total = cls._safe_float(onchain_snapshot.get("position_total_usd_est"))
        total = cls._safe_float(onchain_snapshot.get("total_usd_est"))
        if total is None:
            wallet_total = cls._safe_float(onchain_snapshot.get("wallet_total_usd_est"))
            if wallet_total is not None or position_total is not None:
                total = float(wallet_total or 0.0) + float(position_total or 0.0)
        return position_total, total

    def _update_position_lifecycle_state(
        self,
        *,
        state: BotState,
        active_pool: MeteoraPoolSnapshot,
        onchain_snapshot: dict[str, Any] | None,
        reset_baseline: bool,
    ) -> dict[str, Any]:
        if not self.settings.ev_policy_lifecycle_enabled:
            state.strategy_state["loss_exit_breach_count"] = 0
            return self._empty_lifecycle_metrics()

        active = state.active_position
        if active is None:
            self._reset_position_lifecycle_state(state)
            return self._empty_lifecycle_metrics()

        strategy_state = state.strategy_state
        lifecycle_id = self._position_lifecycle_id(active)
        tracked_id = strategy_state.get("position_lifecycle_position_pubkey")
        current_position_usd, current_total_usd = self._extract_onchain_equity_values(onchain_snapshot)
        modeled_capital_usd = (
            self.ev_scorer.compute_capital_usd(
                active_pool,
                self.settings.deposit_sol_amount,
                self.settings.deposit_usdc_amount,
            )
            if self.ev_scorer is not None
            else None
        )
        if current_position_usd is None and modeled_capital_usd is not None:
            current_position_usd = float(modeled_capital_usd)
        if current_total_usd is None and current_position_usd is not None:
            current_total_usd = float(current_position_usd)
        if current_total_usd is None and modeled_capital_usd is not None:
            current_total_usd = float(modeled_capital_usd)

        lifecycle_changed = tracked_id != lifecycle_id
        if lifecycle_changed:
            strategy_state["loss_exit_breach_count"] = 0
        if reset_baseline or lifecycle_changed or strategy_state.get("position_lifecycle_opened_at") is None:
            baseline_position = current_position_usd
            baseline_total = current_total_usd
            if baseline_position is None and baseline_total is not None:
                baseline_position = float(baseline_total)
            if baseline_position is None and modeled_capital_usd is not None:
                baseline_position = float(modeled_capital_usd)
            if baseline_position is None:
                baseline_position = 0.0
            if baseline_total is None:
                baseline_total = float(baseline_position)
            strategy_state["position_lifecycle_position_pubkey"] = lifecycle_id
            strategy_state["position_lifecycle_opened_at"] = utc_now_iso()
            strategy_state["position_lifecycle_open_position_usd"] = float(baseline_position)
            strategy_state["position_lifecycle_open_total_usd"] = float(baseline_total)

        open_position_usd = self._safe_float(strategy_state.get("position_lifecycle_open_position_usd"))
        open_total_usd = self._safe_float(strategy_state.get("position_lifecycle_open_total_usd"))
        if open_position_usd is None and current_position_usd is not None:
            open_position_usd = float(current_position_usd)
            strategy_state["position_lifecycle_open_position_usd"] = open_position_usd
        if open_total_usd is None and current_total_usd is not None:
            open_total_usd = float(current_total_usd)
            strategy_state["position_lifecycle_open_total_usd"] = open_total_usd
        if open_position_usd is None:
            open_position_usd = float(open_total_usd or modeled_capital_usd or 0.0)
        if open_total_usd is None:
            open_total_usd = float(open_position_usd)

        valuation_source = "position_total_usd_est"
        current_equity = current_position_usd
        open_equity = open_position_usd
        if current_equity is None:
            valuation_source = "total_usd_est"
            current_equity = current_total_usd
            open_equity = open_total_usd
        if current_equity is None:
            valuation_source = "modeled_capital"
            current_equity = float(modeled_capital_usd or 0.0)
            open_equity = float(open_position_usd or 0.0)

        lifecycle_pnl_usd = float(current_equity - open_equity)
        lifecycle_pnl_pct = float(lifecycle_pnl_usd / max(open_equity, 1e-9))

        strategy_state["position_lifecycle_last_position_usd"] = current_position_usd
        strategy_state["position_lifecycle_last_total_usd"] = current_total_usd
        strategy_state["position_lifecycle_last_source"] = valuation_source
        strategy_state["position_lifecycle_last_pnl_usd"] = lifecycle_pnl_usd
        strategy_state["position_lifecycle_last_pnl_pct"] = lifecycle_pnl_pct
        strategy_state["position_lifecycle_last_snapshot_at"] = (
            onchain_snapshot.get("snapshot_at") if isinstance(onchain_snapshot, dict) else None
        )

        return {
            "lifecycle_position_id": lifecycle_id,
            "lifecycle_pnl_usd": lifecycle_pnl_usd,
            "lifecycle_pnl_pct": lifecycle_pnl_pct,
            "lifecycle_open_position_usd": open_position_usd,
            "lifecycle_open_total_usd": open_total_usd,
            "lifecycle_current_position_usd": current_position_usd,
            "lifecycle_current_total_usd": current_total_usd,
            "lifecycle_valuation_source": valuation_source,
        }

    def _build_ev_last_decision(
        self,
        *,
        state: BotState,
        best: EvScoredCandidate,
        selected_pool: MeteoraPoolSnapshot,
        ev_current_hold: EvScoredCandidate | None,
        ev_delta_usd: float | None,
        ev_candidates: list[EvScoredCandidate],
        pre_scores,
        rebalance_gate: dict[str, Any],
        should_rebalance: bool,
        should_close_to_idle: bool,
        selected_action: str,
        reason: str,
        execution_result,
        cost_model_info: dict[str, Any] | None,
        execution_bin_weight_plan: WeightedBinPlan | None,
        execution_bin_quote: ExecutorRangeBinQuote | None,
        execution_weight_objective: str,
        scoring_objective_used: str,
        rescored_candidate_count: int,
        idle_candidate: EvScoredCandidate,
        action_scores: dict[str, float | None],
        lifecycle_metrics: dict[str, Any] | None,
        onchain_snapshot: dict[str, Any] | None,
        onchain_delta: dict[str, Any] | None,
        realism_snapshot: CalibrationSnapshot,
        use_adjusted_for_decisions: bool,
        effective_min_delta_usd: float,
    ) -> dict[str, Any]:
        lifecycle_metrics = lifecycle_metrics or self._empty_lifecycle_metrics()
        best_raw_ev = self._candidate_raw_ev(best)
        best_adjusted_ev = best.ev_components.adjusted_ev_15m_usd
        hold_raw_ev = self._candidate_raw_ev(ev_current_hold) if ev_current_hold else None
        hold_adjusted_ev = ev_current_hold.ev_components.adjusted_ev_15m_usd if ev_current_hold else None
        idle_raw_ev = self._candidate_raw_ev(idle_candidate)
        idle_adjusted_ev = idle_candidate.ev_components.adjusted_ev_15m_usd
        chosen = {
            "width_pct": best.forecast.width_pct,
            "lower_bound": best.forecast.lower_bound,
            "upper_bound": best.forecast.upper_bound,
            "probability_to_stay_in_interval": best.forecast.probability_to_stay_in_interval,
            "expected_time_in_interval_minutes": best.forecast.expected_time_in_interval_minutes,
            "expected_impermanent_loss": best.forecast.expected_impermanent_loss,
            "score": self._candidate_effective_ev(best, use_adjusted=use_adjusted_for_decisions),
            "ev_15m_usd": self._candidate_effective_ev(best, use_adjusted=use_adjusted_for_decisions),
            "raw_ev_15m_usd": best_raw_ev,
            "adjusted_ev_15m_usd": best_adjusted_ev,
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
        plan_for_metrics = execution_bin_weight_plan or best.weighted_bin_plan
        top1_share = max(plan_for_metrics.weights) if plan_for_metrics.weights else None
        top3_share = sum(sorted(plan_for_metrics.weights, reverse=True)[:3]) if plan_for_metrics.weights else None
        return {
            "timestamp": utc_now_iso(),
            "ev_mode": True,
            "pool": pool_dict,
            "selected_pool": pool_dict,
            "active_position": state.active_position.to_dict() if state.active_position else None,
            "strategy_state": dict(state.strategy_state),
            "horizon": self.settings.synth_horizon,
            "chosen": chosen,
            "bin_weight_plan": best.weighted_bin_plan.to_dict(),
            "execution_bin_weight_plan": execution_bin_weight_plan.to_dict() if execution_bin_weight_plan else None,
            "execution_bin_quote": execution_bin_quote.to_dict() if execution_bin_quote else None,
            "top_candidates": [_summarize_ev_candidate(c) for c in ev_candidates[:5]],
            "ev_best_candidate": best.to_dict(),
            "ev_current_hold": ev_current_hold.to_dict() if ev_current_hold else None,
            "ev_idle_candidate": idle_candidate.to_dict(),
            "ev_best_raw_usd": best_raw_ev,
            "ev_best_adjusted_usd": best_adjusted_ev,
            "ev_hold_raw_usd": hold_raw_ev,
            "ev_hold_adjusted_usd": hold_adjusted_ev,
            "ev_idle_raw_usd": idle_raw_ev,
            "ev_idle_adjusted_usd": idle_adjusted_ev,
            "lifecycle_pnl_usd": lifecycle_metrics.get("lifecycle_pnl_usd"),
            "lifecycle_pnl_pct": lifecycle_metrics.get("lifecycle_pnl_pct"),
            "lifecycle_open_position_usd": lifecycle_metrics.get("lifecycle_open_position_usd"),
            "lifecycle_open_total_usd": lifecycle_metrics.get("lifecycle_open_total_usd"),
            "lifecycle_current_position_usd": lifecycle_metrics.get("lifecycle_current_position_usd"),
            "lifecycle_current_total_usd": lifecycle_metrics.get("lifecycle_current_total_usd"),
            "policy_profit_triggered": rebalance_gate.get("policy_profit_triggered"),
            "policy_loss_recovery_hold_triggered": rebalance_gate.get("policy_loss_recovery_hold_triggered"),
            "policy_loss_trend_cut_triggered": rebalance_gate.get("policy_loss_trend_cut_triggered"),
            "policy_oor_grace_triggered": rebalance_gate.get("policy_oor_grace_triggered"),
            "policy_oor_reentry_hold_triggered": rebalance_gate.get("policy_oor_reentry_hold_triggered"),
            "policy_recovery_prob": rebalance_gate.get("policy_recovery_prob"),
            "policy_trend_prob": rebalance_gate.get("policy_trend_prob"),
            "policy_reentry_prob": rebalance_gate.get("policy_reentry_prob"),
            "policy_loss_exit_breach_count": rebalance_gate.get("policy_loss_exit_breach_count"),
            "policy_selected_override": rebalance_gate.get("policy_selected_override", "none"),
            "hold_out_bps": (
                ev_current_hold.ev_components.out_of_range_bps if ev_current_hold else None
            ),
            "hold_oor_cycles": (
                ev_current_hold.ev_components.out_of_range_cycles if ev_current_hold else None
            ),
            "hold_exact_bounds": (
                bool((ev_current_hold.ev_components.baseline_mode or "").startswith("current_hold_exact_bounds"))
                if ev_current_hold
                else None
            ),
            "ev_delta_usd": ev_delta_usd,
            "selected_action": selected_action,
            "action_scores": action_scores,
            "effective_min_delta_usd": effective_min_delta_usd,
            "realism": {
                "enabled": bool(self.settings.ev_realism_enabled),
                "shadow_mode": bool(self.settings.ev_realism_shadow_mode),
                "use_adjusted_for_decisions": use_adjusted_for_decisions,
                "snapshot": realism_snapshot.to_dict(),
            },
            "scoring_objective_used": scoring_objective_used,
            "rescored_candidate_count": rescored_candidate_count,
            "rebalance_gate": rebalance_gate,
            "gate_mode": rebalance_gate.get("gate_mode"),
            "protective_gate_details": rebalance_gate.get("protective_gate_details"),
            "top1_weight_share": top1_share,
            "top3_weight_share": top3_share,
            "pool_rankings": {
                "pre_ranked": [p.to_dict() for p in pre_scores[: min(10, len(pre_scores))]],
                "evaluated_candidate_count": len(ev_candidates),
                "evaluated_pool_count": len({c.pool_address for c in ev_candidates}),
            },
            "cost_model": cost_model_info or None,
            "execution_config": {
                "executor": self.settings.executor,
                "meteora_liquidity_mode": self.settings.meteora_liquidity_mode,
                "max_custom_weight_position_bins": self.settings.max_custom_weight_position_bins,
                "synth_weight_active_bin_floor_bps": self.settings.synth_weight_active_bin_floor_bps,
                "synth_weight_max_bin_bps_per_side": self.settings.synth_weight_max_bin_bps_per_side,
                "synth_weight_max_single_bin": self.settings.synth_weight_max_single_bin,
                "synth_weight_max_top3": self.settings.synth_weight_max_top3,
                "synth_weight_objective": execution_weight_objective,
                "synth_weight_odds_beta": self.settings.synth_weight_odds_beta,
                "synth_weight_odds_eps": self.settings.synth_weight_odds_eps,
                "ev_exact_rescoring_top_n": self.settings.ev_exact_rescoring_top_n,
                "ev_oor_penalty_enabled": self.settings.ev_oor_penalty_enabled,
                "ev_oor_penalize_hold_only": self.settings.ev_oor_penalize_hold_only,
                "ev_oor_deadband_bps": self.settings.ev_oor_deadband_bps,
                "ev_oor_ref_bps": self.settings.ev_oor_ref_bps,
                "ev_oor_base_penalty_fraction_15m": self.settings.ev_oor_base_penalty_fraction_15m,
                "ev_oor_max_penalty_fraction_15m": self.settings.ev_oor_max_penalty_fraction_15m,
                "ev_oor_persistence_step": self.settings.ev_oor_persistence_step,
                "ev_oor_persistence_cap_cycles": self.settings.ev_oor_persistence_cap_cycles,
                "ev_oor_grace_enabled": self.settings.ev_oor_grace_enabled,
                "ev_oor_grace_cycles": self.settings.ev_oor_grace_cycles,
                "ev_oor_reentry_min_prob": self.settings.ev_oor_reentry_min_prob,
                "ev_oor_loss_trigger_pct": self.settings.ev_oor_loss_trigger_pct,
                "ev_idle_enabled": self.settings.ev_idle_enabled,
                "ev_idle_entry_threshold_usd": self.settings.ev_idle_entry_threshold_usd,
                "ev_idle_exit_threshold_usd": self.settings.ev_idle_exit_threshold_usd,
                "ev_idle_confirm_cycles": self.settings.ev_idle_confirm_cycles,
                "ev_il_drift_alpha": self.settings.ev_il_drift_alpha,
                "ev_il_oor_beta": self.settings.ev_il_oor_beta,
                "ev_il_onesided_gamma": self.settings.ev_il_onesided_gamma,
                "ev_il_persistence_delta": self.settings.ev_il_persistence_delta,
                "ev_il_mult_min": self.settings.ev_il_mult_min,
                "ev_il_mult_max": self.settings.ev_il_mult_max,
                "ev_il_drift_ref_bps": self.settings.ev_il_drift_ref_bps,
                "ev_il_drift_horizon_minutes": self.settings.ev_il_drift_horizon_minutes,
                "ev_trend_stop_enabled": self.settings.ev_trend_stop_enabled,
                "ev_trend_stop_oor_cycles": self.settings.ev_trend_stop_oor_cycles,
                "ev_trend_stop_onesided_prob": self.settings.ev_trend_stop_onesided_prob,
                "ev_action_cooldown_cycles": self.settings.ev_action_cooldown_cycles,
                "ev_policy_lifecycle_enabled": self.settings.ev_policy_lifecycle_enabled,
                "ev_profit_take_pct": self.settings.ev_profit_take_pct,
                "ev_profit_rebalance_max_slip_usd": self.settings.ev_profit_rebalance_max_slip_usd,
                "ev_loss_recovery_trigger_pct": self.settings.ev_loss_recovery_trigger_pct,
                "ev_high_il_15m_fraction": self.settings.ev_high_il_15m_fraction,
                "ev_recovery_min_prob": self.settings.ev_recovery_min_prob,
                "ev_trend_continue_exit_prob": self.settings.ev_trend_continue_exit_prob,
                "ev_trend_exit_persistence_cycles": self.settings.ev_trend_exit_persistence_cycles,
                "ev_idle_preference_edge_usd": self.settings.ev_idle_preference_edge_usd,
                "ev_realism_enabled": self.settings.ev_realism_enabled,
                "ev_realism_shadow_mode": self.settings.ev_realism_shadow_mode,
                "ev_realism_window_hours": self.settings.ev_realism_window_hours,
                "ev_realism_min_samples": self.settings.ev_realism_min_samples,
                "ev_fee_realism_prior": self.settings.ev_fee_realism_prior,
                "ev_fee_realism_min": self.settings.ev_fee_realism_min,
                "ev_fee_realism_max": self.settings.ev_fee_realism_max,
                "ev_rebalance_drag_prior_usd": self.settings.ev_rebalance_drag_prior_usd,
                "ev_rebalance_drag_min_usd": self.settings.ev_rebalance_drag_min_usd,
                "ev_rebalance_drag_max_usd": self.settings.ev_rebalance_drag_max_usd,
                "ev_uncertainty_k": self.settings.ev_uncertainty_k,
                "ev_dynamic_gate_enabled": self.settings.ev_dynamic_gate_enabled,
                "ev_dynamic_gate_base_margin_usd": self.settings.ev_dynamic_gate_base_margin_usd,
                "ev_dynamic_gate_sigma_mult": self.settings.ev_dynamic_gate_sigma_mult,
                "ev_adjusted_ev_require_positive": self.settings.ev_adjusted_ev_require_positive,
            },
            "rebalance": {
                "should_rebalance": should_rebalance,
                "reason": reason,
                "executor": self.settings.executor,
                "should_close_to_idle": should_close_to_idle,
                "tx_signatures": execution_result.tx_signatures if execution_result else [],
                "details": execution_result.details if execution_result else {},
            },
            "onchain_snapshot": onchain_snapshot,
            "onchain_delta": onchain_delta,
        }

    def _capture_onchain_snapshot(
        self,
        *,
        state: BotState,
        pool: MeteoraPoolSnapshot,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        previous = state.strategy_state.get("last_onchain_snapshot")
        if not isinstance(previous, dict):
            previous = None
        try:
            current = self.executor.get_onchain_snapshot(pool=pool, active_position=state.active_position)
        except Exception as exc:
            LOGGER.warning("On-chain snapshot capture failed; continuing without snapshot: %s", exc)
            return None, None
        if not isinstance(current, dict):
            return None, None
        if not current.get("snapshot_at"):
            current["snapshot_at"] = utc_now_iso()
        delta = self._compute_onchain_delta(previous=previous, current=current)
        state.strategy_state["last_onchain_snapshot"] = current
        state.strategy_state["last_onchain_snapshot_at"] = current.get("snapshot_at")
        return current, delta

    @staticmethod
    def _compute_onchain_delta(
        *,
        previous: dict[str, Any] | None,
        current: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if not previous or not current:
            return None

        def _num(obj: dict[str, Any], key: str) -> float | None:
            try:
                value = obj.get(key)
                if value is None:
                    return None
                return float(value)
            except (TypeError, ValueError):
                return None

        prev_sol = _num(previous, "sol_balance")
        prev_usdc = _num(previous, "usdc_balance")
        prev_wallet_total = _num(previous, "wallet_total_usd_est")
        prev_position_total = _num(previous, "position_total_usd_est")
        prev_total = _num(previous, "total_usd_est")
        if prev_total is None and (prev_wallet_total is not None or prev_position_total is not None):
            prev_total = float(prev_wallet_total or 0.0) + float(prev_position_total or 0.0)
        cur_sol = _num(current, "sol_balance")
        cur_usdc = _num(current, "usdc_balance")
        cur_wallet_total = _num(current, "wallet_total_usd_est")
        cur_position_total = _num(current, "position_total_usd_est")
        cur_total = _num(current, "total_usd_est")
        if cur_total is None and (cur_wallet_total is not None or cur_position_total is not None):
            cur_total = float(cur_wallet_total or 0.0) + float(cur_position_total or 0.0)
        delta: dict[str, Any] = {
            "from_snapshot_at": previous.get("snapshot_at"),
            "to_snapshot_at": current.get("snapshot_at"),
        }
        if prev_sol is not None and cur_sol is not None:
            delta["delta_sol"] = cur_sol - prev_sol
        if prev_usdc is not None and cur_usdc is not None:
            delta["delta_usdc"] = cur_usdc - prev_usdc
        if prev_wallet_total is not None and cur_wallet_total is not None:
            delta["delta_wallet_total_usd_est"] = cur_wallet_total - prev_wallet_total
        if prev_position_total is not None and cur_position_total is not None:
            delta["delta_position_total_usd_est"] = cur_position_total - prev_position_total
        if prev_total is not None and cur_total is not None:
            delta["delta_total_usd_est"] = cur_total - prev_total
        try:
            from_ts = _parse_iso_timestamp(previous.get("snapshot_at"))
            to_ts = _parse_iso_timestamp(current.get("snapshot_at"))
            if from_ts is not None and to_ts is not None:
                delta["minutes_elapsed"] = max(0.0, (to_ts - from_ts) / 60.0)
        except Exception:
            pass
        return delta

    def _append_trade_journal_decision_entry(self, *, result: dict[str, Any]) -> None:
        if not self.settings.trade_journal_enabled:
            return
        if not isinstance(result, dict):
            return
        pool = result.get("selected_pool") if isinstance(result.get("selected_pool"), dict) else {}
        if not pool:
            pool = result.get("pool") if isinstance(result.get("pool"), dict) else {}
        rebalance = result.get("rebalance") if isinstance(result.get("rebalance"), dict) else {}
        rebalance_gate = result.get("rebalance_gate") if isinstance(result.get("rebalance_gate"), dict) else {}
        chosen = result.get("chosen") if isinstance(result.get("chosen"), dict) else {}
        best = result.get("ev_best_candidate") if isinstance(result.get("ev_best_candidate"), dict) else {}
        best_components = best.get("ev_components") if isinstance(best.get("ev_components"), dict) else {}
        hold = result.get("ev_current_hold") if isinstance(result.get("ev_current_hold"), dict) else {}
        hold_components = hold.get("ev_components") if isinstance(hold.get("ev_components"), dict) else {}
        idle = result.get("ev_idle_candidate") if isinstance(result.get("ev_idle_candidate"), dict) else {}
        idle_components = idle.get("ev_components") if isinstance(idle.get("ev_components"), dict) else {}
        onchain_snapshot = result.get("onchain_snapshot") if isinstance(result.get("onchain_snapshot"), dict) else {}
        onchain_delta = result.get("onchain_delta") if isinstance(result.get("onchain_delta"), dict) else {}
        tx_signatures_raw = rebalance.get("tx_signatures")
        tx_signatures = [str(x) for x in tx_signatures_raw if isinstance(x, str)] if isinstance(tx_signatures_raw, list) else []
        should_rebalance = bool(rebalance.get("should_rebalance"))
        should_close_to_idle = bool(rebalance.get("should_close_to_idle"))
        selected_action = str(
            result.get("selected_action")
            or (
                "rebalance"
                if should_rebalance
                else ("idle" if should_close_to_idle else "hold")
            )
        )
        selected_ev = (
            best.get("ev_15m_usd", chosen.get("ev_15m_usd", chosen.get("score")))
            if selected_action == "rebalance"
            else (
                hold.get("ev_15m_usd")
                if selected_action == "hold"
                else idle.get("ev_15m_usd")
            )
        )
        selected_components = (
            best_components
            if selected_action == "rebalance"
            else (hold_components if selected_action == "hold" else idle_components)
        )
        selected_raw_ev = (
            result.get("ev_best_raw_usd")
            if selected_action == "rebalance"
            else (
                result.get("ev_hold_raw_usd")
                if selected_action == "hold"
                else result.get("ev_idle_raw_usd")
            )
        )
        selected_adjusted_ev = (
            result.get("ev_best_adjusted_usd")
            if selected_action == "rebalance"
            else (
                result.get("ev_hold_adjusted_usd")
                if selected_action == "hold"
                else result.get("ev_idle_adjusted_usd")
            )
        )
        selected_fees = selected_components.get("expected_fees_usd")
        selected_il = selected_components.get("expected_il_usd")
        selected_cost = (selected_components.get("rebalance_cost_usd") or 0.0) + (
            selected_components.get("pool_switch_extra_cost_usd") or 0.0
        )
        model_net_from_components = None
        if selected_fees is not None and selected_il is not None:
            try:
                model_net_from_components = float(selected_fees) - float(selected_il) - float(selected_cost)
            except (TypeError, ValueError):
                model_net_from_components = None
        if selected_raw_ev is None:
            selected_raw_ev = model_net_from_components
        if selected_adjusted_ev is None:
            selected_adjusted_ev = selected_components.get("adjusted_ev_15m_usd")
        execution_attempted = bool(should_rebalance or should_close_to_idle)
        execution_plan_error = rebalance_gate.get("execution_plan_error")
        execution_status = "not_attempted"
        if execution_attempted:
            if tx_signatures:
                execution_status = "success"
            elif execution_plan_error:
                execution_status = "blocked_before_execute"
            else:
                execution_status = "attempted_no_tx"

        entry = {
            "event_type": "decision",
            "recorded_at": utc_now_iso(),
            "decision_timestamp": result.get("timestamp"),
            "mode": "ev" if bool(result.get("ev_mode")) else "proxy",
            "horizon": result.get("horizon"),
            "selected_action": selected_action,
            "pool_address": pool.get("address"),
            "pool_name": pool.get("name"),
            "spot_price_sol_usdc": pool.get("current_price"),
            "target_range_lower": chosen.get("lower_bound"),
            "target_range_upper": chosen.get("upper_bound"),
            "target_width_pct": chosen.get("width_pct"),
            "active_range_lower": (result.get("active_position") or {}).get("lower_price")
            if isinstance(result.get("active_position"), dict)
            else None,
            "active_range_upper": (result.get("active_position") or {}).get("upper_price")
            if isinstance(result.get("active_position"), dict)
            else None,
            "ev_best_usd": best.get("ev_15m_usd", chosen.get("ev_15m_usd", chosen.get("score"))),
            "ev_hold_usd": hold.get("ev_15m_usd"),
            "ev_idle_usd": idle.get("ev_15m_usd"),
            "ev_delta_usd": result.get("ev_delta_usd"),
            "ev_selected_usd": selected_ev,
            "effective_min_delta_usd": result.get("effective_min_delta_usd"),
            "lifecycle_pnl_usd": result.get("lifecycle_pnl_usd"),
            "lifecycle_pnl_pct": result.get("lifecycle_pnl_pct"),
            "lifecycle_open_position_usd": result.get("lifecycle_open_position_usd"),
            "lifecycle_open_total_usd": result.get("lifecycle_open_total_usd"),
            "lifecycle_current_position_usd": result.get("lifecycle_current_position_usd"),
            "lifecycle_current_total_usd": result.get("lifecycle_current_total_usd"),
            "policy_profit_triggered": bool(result.get("policy_profit_triggered")),
            "policy_loss_recovery_hold_triggered": bool(result.get("policy_loss_recovery_hold_triggered")),
            "policy_loss_trend_cut_triggered": bool(result.get("policy_loss_trend_cut_triggered")),
            "policy_oor_grace_triggered": bool(result.get("policy_oor_grace_triggered")),
            "policy_oor_reentry_hold_triggered": bool(result.get("policy_oor_reentry_hold_triggered")),
            "policy_recovery_prob": result.get("policy_recovery_prob"),
            "policy_trend_prob": result.get("policy_trend_prob"),
            "policy_reentry_prob": result.get("policy_reentry_prob"),
            "policy_loss_exit_breach_count": result.get("policy_loss_exit_breach_count"),
            "policy_selected_override": result.get("policy_selected_override"),
            "expected_fees_usd": best_components.get("expected_fees_usd"),
            "expected_il_usd": best_components.get("expected_il_usd"),
            "il_baseline_usd": best_components.get("il_baseline_usd"),
            "il_state_penalty_usd": best_components.get("il_state_penalty_usd"),
            "expected_rebalance_cost_usd": best_components.get("rebalance_cost_usd"),
            "expected_pool_switch_cost_usd": best_components.get("pool_switch_extra_cost_usd"),
            "expected_total_cost_usd": (best_components.get("rebalance_cost_usd") or 0.0)
            + (best_components.get("pool_switch_extra_cost_usd") or 0.0),
            "expected_net_usd": (
                best.get("ev_15m_usd", chosen.get("ev_15m_usd", chosen.get("score")))
            ),
            "raw_expected_net_usd": selected_raw_ev,
            "adjusted_expected_net_usd": selected_adjusted_ev,
            "selected_expected_fees_usd": selected_fees,
            "selected_expected_il_usd": selected_il,
            "selected_expected_rebalance_cost_usd": selected_components.get("rebalance_cost_usd"),
            "selected_expected_pool_switch_cost_usd": selected_components.get("pool_switch_extra_cost_usd"),
            "selected_expected_total_cost_usd": selected_cost,
            "selected_expected_net_usd": selected_ev,
            "selected_raw_expected_net_usd": selected_raw_ev,
            "selected_adjusted_expected_net_usd": selected_adjusted_ev,
            "selected_expected_net_from_components_usd": model_net_from_components,
            "selected_il_baseline_usd": selected_components.get("il_baseline_usd"),
            "selected_il_state_penalty_usd": selected_components.get("il_state_penalty_usd"),
            "selected_raw_expected_fees_usd": selected_components.get("raw_expected_fees_usd"),
            "selected_adjusted_expected_fees_usd": selected_components.get("adjusted_expected_fees_usd"),
            "selected_fee_realism_multiplier": selected_components.get("fee_realism_multiplier"),
            "selected_execution_drag_usd": selected_components.get("execution_drag_usd"),
            "selected_uncertainty_penalty_usd": selected_components.get("uncertainty_penalty_usd"),
            "selected_adjusted_ev_15m_usd": selected_components.get("adjusted_ev_15m_usd"),
            "selected_calibration_sample_count": selected_components.get("calibration_sample_count"),
            "selected_calibration_mode": selected_components.get("calibration_mode"),
            "utilization_ratio": best_components.get("utilization_ratio"),
            "fee_capture_factor": best_components.get("fee_capture_factor"),
            "occ_range": best_components.get("range_active_occupancy_15m", best_components.get("active_occupancy_15m")),
            "weight_alignment_score": best_components.get("weight_alignment_score"),
            "hold_utilization_ratio": hold_components.get("utilization_ratio"),
            "hold_out_of_range_bps": hold_components.get("out_of_range_bps"),
            "hold_out_of_range_cycles": hold_components.get("out_of_range_cycles"),
            "rebalance_should": should_rebalance,
            "close_to_idle_should": should_close_to_idle,
            "rebalance_reason": rebalance.get("reason"),
            "gate_mode": result.get("gate_mode") or rebalance_gate.get("gate_mode"),
            "execution_attempted": execution_attempted,
            "execution_status": execution_status,
            "execution_tx_count": len(tx_signatures),
            "execution_tx_signatures": tx_signatures,
            "execution_plan_error": str(execution_plan_error) if execution_plan_error else None,
            "realism_enabled": (result.get("realism") or {}).get("enabled") if isinstance(result.get("realism"), dict) else None,
            "realism_shadow_mode": (result.get("realism") or {}).get("shadow_mode") if isinstance(result.get("realism"), dict) else None,
            "realism_use_adjusted_for_decisions": (result.get("realism") or {}).get("use_adjusted_for_decisions")
            if isinstance(result.get("realism"), dict)
            else None,
            "realism_fee_realism_multiplier": ((result.get("realism") or {}).get("snapshot") or {}).get("fee_realism_multiplier")
            if isinstance((result.get("realism") or {}).get("snapshot"), dict)
            else None,
            "realism_rebalance_drag_usd": ((result.get("realism") or {}).get("snapshot") or {}).get("rebalance_drag_usd")
            if isinstance((result.get("realism") or {}).get("snapshot"), dict)
            else None,
            "realism_model_rmse_usd": ((result.get("realism") or {}).get("snapshot") or {}).get("model_rmse_usd")
            if isinstance((result.get("realism") or {}).get("snapshot"), dict)
            else None,
            "realism_calibration_sample_count": ((result.get("realism") or {}).get("snapshot") or {}).get("sample_count")
            if isinstance((result.get("realism") or {}).get("snapshot"), dict)
            else None,
            "realism_calibration_mode": ((result.get("realism") or {}).get("snapshot") or {}).get("mode")
            if isinstance((result.get("realism") or {}).get("snapshot"), dict)
            else None,
            "onchain_snapshot_at": onchain_snapshot.get("snapshot_at"),
            "onchain_wallet_pubkey": onchain_snapshot.get("wallet_pubkey"),
            "onchain_sol_balance": onchain_snapshot.get("sol_balance"),
            "onchain_usdc_balance": onchain_snapshot.get("usdc_balance"),
            "onchain_native_sol_balance": onchain_snapshot.get("native_sol_balance"),
            "onchain_wallet_sol_token_balance": onchain_snapshot.get("wallet_sol_token_balance"),
            "onchain_wallet_sol_total_balance": onchain_snapshot.get("wallet_sol_total_balance"),
            "onchain_wallet_usdc_total_balance": onchain_snapshot.get("wallet_usdc_total_balance"),
            "onchain_wallet_total_usd_est": onchain_snapshot.get("wallet_total_usd_est"),
            "onchain_position_total_usd_est": onchain_snapshot.get("position_total_usd_est"),
            "onchain_total_usd_est": onchain_snapshot.get("total_usd_est"),
            "onchain_spot_price_sol_usdc": onchain_snapshot.get("spot_price_sol_usdc"),
            "onchain_active_position_exists": onchain_snapshot.get("active_position_exists"),
            "onchain_delta_sol": onchain_delta.get("delta_sol"),
            "onchain_delta_usdc": onchain_delta.get("delta_usdc"),
            "onchain_delta_wallet_total_usd_est": onchain_delta.get("delta_wallet_total_usd_est"),
            "onchain_delta_position_total_usd_est": onchain_delta.get("delta_position_total_usd_est"),
            "onchain_delta_total_usd_est": onchain_delta.get("delta_total_usd_est"),
            "onchain_delta_minutes_elapsed": onchain_delta.get("minutes_elapsed"),
        }
        self._append_trade_journal_entry(entry)

    def _append_trade_journal_error_entry(self, *, exc: Exception, state: BotState) -> None:
        if not self.settings.trade_journal_enabled:
            return
        active = state.active_position
        last = state.last_decision if isinstance(state.last_decision, dict) else {}
        message = str(exc)
        entry = {
            "event_type": "cycle_error",
            "recorded_at": utc_now_iso(),
            "mode": "ev" if self.settings.ev_mode else "proxy",
            "error_type": exc.__class__.__name__,
            "error_category": self._classify_error_category(message),
            "error_message": message,
            "error_traceback_tail": "".join(traceback.format_exception_only(type(exc), exc)).strip(),
            "active_pool_address": active.pool_address if active else None,
            "active_range_lower": active.lower_price if active else None,
            "active_range_upper": active.upper_price if active else None,
            "last_selected_action": last.get("selected_action"),
            "last_rebalance_reason": (last.get("rebalance") or {}).get("reason")
            if isinstance(last.get("rebalance"), dict)
            else None,
        }
        self._append_trade_journal_entry(entry)

    def _append_trade_journal_entry(self, payload: dict[str, Any]) -> None:
        try:
            path = self.settings.trade_journal_path
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True))
                handle.write("\n")
        except Exception:
            LOGGER.exception("Failed to append trade journal entry")

    @staticmethod
    def _classify_error_category(message: str) -> str:
        lowered = (message or "").lower()
        if "429" in lowered or "too many requests" in lowered:
            return "rpc_rate_limit"
        if "insufficient lamports" in lowered or "insufficient funds" in lowered:
            return "insufficient_funds"
        if "timed out" in lowered or "timeout" in lowered:
            return "upstream_timeout"
        if "execution plan unavailable" in lowered or "quote_target_bins returned no data" in lowered:
            return "execution_plan_unavailable"
        if "invalid strategy parameters" in lowered:
            return "invalid_strategy_parameters"
        return "unknown"

    @classmethod
    def _is_retryable_loop_error(cls, exc: Exception) -> bool:
        category = cls._classify_error_category(str(exc))
        return category in {"rpc_rate_limit", "upstream_timeout"}


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


def _lamports_to_usd(lamports: int | float, sol_usdc_price: float) -> float:
    try:
        return max(0.0, float(lamports)) / 1e9 * max(0.0, float(sol_usdc_price))
    except (TypeError, ValueError):
        return 0.0


def _representative_sol_spot(pools: list[MeteoraPoolSnapshot]) -> float:
    spots = [p.current_price_sol_usdc() for p in pools if p.current_price_sol_usdc() > 0]
    if not spots:
        return 0.0
    return float(median(spots))


def _should_rebalance_ev(
    *,
    state: BotState,
    best: EvScoredCandidate,
    ev_current_hold: EvScoredCandidate | None = None,
    ev_delta_usd: float | None,
    min_ev_improvement_usd: float,
    prior_protective_breach_count: int = 0,
    ev_utilization_floor: float = 0.40,
    ev_min_utilization_gain: float = 0.15,
    ev_max_protective_ev_slip_usd: float = 0.002,
    ev_protective_breach_cycles: int = 2,
) -> tuple[bool, str, dict[str, Any], int]:
    if state.active_position is None:
        gate = {
            "baseline_present": False,
            "ev_delta_usd": ev_delta_usd,
            "min_ev_improvement_usd": min_ev_improvement_usd,
            "ev_threshold_passed": True,
            "structural_change_passed": True,
            "pool_switch": best.pool_switch,
            "range_change_bps_vs_active": best.range_change_bps_vs_active,
            "gate_mode": "ev_only",
            "protective_breach_count": 0,
            "protective_gate_details": None,
        }
        return True, "no_active_position", gate, 0

    delta = float(ev_delta_usd or 0.0)
    ev_threshold_passed = delta >= float(min_ev_improvement_usd)
    structural_change_passed = bool(best.rebalance_structural_change)
    hold_utilization = None
    if ev_current_hold is not None:
        hold_utilization = ev_current_hold.ev_components.utilization_ratio
    candidate_utilization = best.ev_components.utilization_ratio

    gate = {
        "baseline_present": True,
        "ev_delta_usd": delta,
        "min_ev_improvement_usd": float(min_ev_improvement_usd),
        "ev_threshold_passed": ev_threshold_passed,
        "structural_change_passed": structural_change_passed,
        "pool_switch": best.pool_switch,
        "range_change_bps_vs_active": best.range_change_bps_vs_active,
        "gate_mode": "ev_only",
        "protective_breach_count": 0,
        "protective_gate_details": {
            "hold_utilization": hold_utilization,
            "candidate_utilization": candidate_utilization,
            "ev_utilization_floor": float(ev_utilization_floor),
            "ev_min_utilization_gain": float(ev_min_utilization_gain),
            "ev_max_protective_ev_slip_usd": float(ev_max_protective_ev_slip_usd),
            "ev_protective_breach_cycles": int(ev_protective_breach_cycles),
        },
    }
    if ev_threshold_passed and structural_change_passed:
        gate["protective_breach_count"] = 0
        return (
            True,
            f"pool_switch_ev_gain_{delta:.4f}" if best.pool_switch else f"range_ev_gain_{delta:.4f}",
            gate,
            0,
        )

    if not structural_change_passed:
        gate["protective_breach_count"] = 0
        if best.range_change_bps_vs_active is None:
            return False, "no_structural_change", gate, 0
        return False, f"range_change_below_threshold_{best.range_change_bps_vs_active:.2f}bps", gate, 0

    protective_conditions = (
        hold_utilization is not None
        and candidate_utilization is not None
        and hold_utilization < float(ev_utilization_floor)
        and (candidate_utilization - hold_utilization) >= float(ev_min_utilization_gain)
        and delta >= -float(ev_max_protective_ev_slip_usd)
    )
    breach_count = max(0, int(prior_protective_breach_count))
    if protective_conditions:
        breach_count += 1
    else:
        breach_count = 0
    gate["protective_breach_count"] = breach_count
    gate["gate_mode"] = "protective_utilization" if protective_conditions else "ev_only"
    if isinstance(gate.get("protective_gate_details"), dict):
        gate["protective_gate_details"]["condition_passed"] = protective_conditions
        gate["protective_gate_details"]["breach_count"] = breach_count

    if protective_conditions and breach_count >= max(1, int(ev_protective_breach_cycles)):
        return True, "protective_utilization_rebalance", gate, breach_count

    if not ev_threshold_passed:
        if protective_conditions:
            return (
                False,
                f"protective_wait_{breach_count}/{max(1, int(ev_protective_breach_cycles))}",
                gate,
                breach_count,
            )
        return False, f"ev_delta_below_threshold_{delta:.4f}", gate, breach_count
    if best.pool_switch:
        return True, f"pool_switch_ev_gain_{delta:.4f}", gate, 0
    return True, f"range_ev_gain_{delta:.4f}", gate, 0


def _summarize_ev_candidate(candidate: EvScoredCandidate) -> dict[str, Any]:
    return {
        "pool_address": candidate.pool_address,
        "pool_name": candidate.pool_name,
        "ev_15m_usd": candidate.ev_15m_usd,
        "raw_ev_15m_usd": (
            float(candidate.ev_components.expected_fees_usd)
            - float(candidate.ev_components.expected_il_usd)
            - float(candidate.ev_components.rebalance_cost_usd)
            - float(candidate.ev_components.pool_switch_extra_cost_usd)
        ),
        "adjusted_ev_15m_usd": candidate.ev_components.adjusted_ev_15m_usd,
        "pool_switch": candidate.pool_switch,
        "range_change_bps_vs_active": candidate.range_change_bps_vs_active,
        "rebalance_structural_change": candidate.rebalance_structural_change,
        "action_type": candidate.action_type,
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


def _ev_candidate_key(candidate: EvScoredCandidate) -> tuple[str, float, float]:
    return (
        candidate.pool_address,
        round(float(candidate.forecast.lower_bound), 8),
        round(float(candidate.forecast.upper_bound), 8),
    )


def _log_execution_details(details: dict[str, Any] | None) -> None:
    if not isinstance(details, dict):
        return
    execution = details.get("execution")
    if not isinstance(execution, dict):
        return
    if execution.get("liquidity_mode") != "synth_weights":
        return
    validation = execution.get("custom_weight_validation")
    if not isinstance(validation, dict):
        validation = {}
    x_constraints = validation.get("x_constraints") if isinstance(validation.get("x_constraints"), dict) else {}
    y_constraints = validation.get("y_constraints") if isinstance(validation.get("y_constraints"), dict) else {}
    preview = execution.get("custom_weight_distribution_preview")
    preview_str = "n/a"
    if isinstance(preview, list) and preview:
        rows = []
        for row in preview[:5]:
            if not isinstance(row, dict):
                continue
            rows.append(
                f"bin{row.get('binId')} x={row.get('x_bps')} y={row.get('y_bps')}"
            )
        if rows:
            preview_str = ";".join(rows)
    LOGGER.info(
        (
            "Synth-weight execution: bins=%s active_bin=%s tx_count=%s "
            "x_side_bins=%s y_side_bins=%s method=%s "
            "x_cap=%s y_cap=%s x_floor=%s y_floor=%s preview=%s"
        ),
        validation.get("bin_count", execution.get("custom_weight_num_bins")),
        validation.get("active_bin_id", execution.get("active_bin_id")),
        execution.get("sdk_tx_count"),
        validation.get("x_side_bins"),
        validation.get("y_side_bins"),
        execution.get("sdk_method_used"),
        x_constraints.get("cap_reason"),
        y_constraints.get("cap_reason"),
        x_constraints.get("active_floor_applied_bps"),
        y_constraints.get("active_floor_applied_bps"),
        preview_str,
    )


def _parse_iso_timestamp(raw: Any) -> float | None:
    if not isinstance(raw, str) or not raw:
        return None
    text = raw
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()
