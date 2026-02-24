from __future__ import annotations

import logging
import time
from dataclasses import asdict

from ai_liquidity_optimizer.clients.meteora import MeteoraDlmmApiClient
from ai_liquidity_optimizer.clients.synth import SynthInsightsClient
from ai_liquidity_optimizer.config import Settings
from ai_liquidity_optimizer.execution.base import PositionExecutor
from ai_liquidity_optimizer.models import BotState, ExecutionApplyRequest
from ai_liquidity_optimizer.state_store import JsonStateStore
from ai_liquidity_optimizer.strategy.scoring import StrategyScorer, relative_range_change_bps


LOGGER = logging.getLogger(__name__)


class OptimizerOrchestrator:
    def __init__(
        self,
        settings: Settings,
        synth_client: SynthInsightsClient,
        meteora_client: MeteoraDlmmApiClient,
        scorer: StrategyScorer,
        executor: PositionExecutor,
        state_store: JsonStateStore,
    ):
        self.settings = settings
        self.synth_client = synth_client
        self.meteora_client = meteora_client
        self.scorer = scorer
        self.executor = executor
        self.state_store = state_store

    def run_forever(self) -> None:
        interval_seconds = self.settings.rebalance_interval_minutes * 60
        while True:
            self.run_once()
            time.sleep(interval_seconds)

    def run_once(self) -> dict:
        state = self.state_store.load()
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
            "top_candidates": [c.to_dict() for c in decision.ranked[:5]],
            "rebalance": {
                "should_rebalance": should_rebalance,
                "reason": reason,
                "executor": self.settings.executor,
                "tx_signatures": execution_result.tx_signatures if execution_result else [],
            },
        }
        self.state_store.save(state)

        return state.last_decision


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

