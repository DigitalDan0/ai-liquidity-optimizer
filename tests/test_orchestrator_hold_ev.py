from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

from ai_liquidity_optimizer.execution.base import PositionExecutor
from ai_liquidity_optimizer.models import (
    ActivePositionState,
    BotState,
    ExecutionApplyRequest,
    ExecutionApplyResult,
    MeteoraPoolSnapshot,
    SynthLpBoundForecast,
    SynthLpProbabilitiesSnapshot,
    SynthLpProbabilityPoint,
)
from ai_liquidity_optimizer.orchestrator import OptimizerOrchestrator
from ai_liquidity_optimizer.strategy.ev import EvLpScorer
from ai_liquidity_optimizer.strategy.scoring import StrategyScorer


class _StubExecutor(PositionExecutor):
    def apply_target_range(self, request: ExecutionApplyRequest) -> ExecutionApplyResult:
        raise RuntimeError("not used in this test")

    def quote_target_bins(
        self,
        *,
        pool_address: str,
        symbol_x: str,
        symbol_y: str,
        api_current_price: float,
        target_lower_price: float,
        target_upper_price: float,
    ):
        return None


class _StubStateStore:
    def load(self):  # pragma: no cover - not used
        return BotState()

    def save(self, state):  # pragma: no cover - not used
        return None


class _StubMeteoraClient:
    def get_pool(self, pool_address: str):  # pragma: no cover - not expected
        raise RuntimeError("unexpected get_pool call")


def _pool(current_price: float = 100.0) -> MeteoraPoolSnapshot:
    return MeteoraPoolSnapshot(
        address="PoolA",
        name="SOL-USDC",
        mint_x="x",
        mint_y="y",
        symbol_x="SOL",
        symbol_y="USDC",
        decimals_x=9,
        decimals_y=6,
        current_price=current_price,
        liquidity=1_000_000.0,
        volume_24h=1_000_000.0,
        fees_24h=0.0,
        fee_tvl_ratio_24h=0.08,
        tvl=1_000_000.0,
        bin_step_bps=10.0,
        base_fee_pct=0.2,
        dynamic_fee_pct=0.05,
        fee_tvl_ratio_by_window={"1h": 0.02, "24h": 0.08},
        volume_by_window={"1h": 200_000.0, "24h": 1_000_000.0},
        fees_by_window={},
        raw={"pool_config": {"bin_step": 10.0}},
    )


class OrchestratorHoldEvTests(unittest.TestCase):
    def _orchestrator(self, pool: MeteoraPoolSnapshot) -> OptimizerOrchestrator:
        settings = SimpleNamespace(
            synth_horizon="1h",
            deposit_sol_amount=1.0,
            deposit_usdc_amount=100.0,
            range_change_threshold_bps=10.0,
            synth_weight_max_single_bin=0.30,
            synth_weight_max_top3=0.65,
            ev_oor_deadband_bps=5.0,
            min_ev_improvement_usd=0.01,
            ev_utilization_floor=0.4,
            ev_min_utilization_gain=0.15,
            ev_max_protective_ev_slip_usd=0.002,
            ev_protective_breach_cycles=2,
            ev_idle_enabled=True,
            ev_idle_entry_threshold_usd=0.0,
            ev_idle_exit_threshold_usd=0.01,
            ev_idle_confirm_cycles=2,
            ev_trend_stop_enabled=True,
            ev_trend_stop_oor_cycles=3,
            ev_trend_stop_onesided_prob=0.65,
            ev_action_cooldown_cycles=1,
        )
        return OptimizerOrchestrator(
            settings=settings,
            synth_client=SimpleNamespace(),
            meteora_client=_StubMeteoraClient(),
            scorer=StrategyScorer(),
            ev_scorer=EvLpScorer(),
            executor=_StubExecutor(),
            state_store=_StubStateStore(),
        )

    def test_hold_scoring_uses_active_bounds_and_falls_back_when_exact_quote_missing(self):
        pool = _pool(current_price=110.0)
        orchestrator = self._orchestrator(pool)
        state = BotState(
            active_position=ActivePositionState(
                pool_address=pool.address,
                lower_price=95.0,
                upper_price=105.0,
                width_pct=10.0,
                executor="dry-run",
            )
        )
        forecasts = [
            SynthLpBoundForecast(
                width_pct=20.0,
                lower_bound=90.0,
                upper_bound=110.0,
                probability_to_stay_in_interval=0.9,
                expected_time_in_interval_minutes=60.0,
                expected_impermanent_loss=0.002,
            )
        ]
        lp_probabilities = SynthLpProbabilitiesSnapshot(
            asset="SOL",
            horizon="1h",
            points=[
                SynthLpProbabilityPoint(price=90.0, probability_below=0.2),
                SynthLpProbabilityPoint(price=100.0, probability_below=0.5),
                SynthLpProbabilityPoint(price=110.0, probability_below=0.8),
            ],
        )

        scored = orchestrator._compute_current_hold_ev(
            state=state,
            forecasts=forecasts,
            lp_probabilities=lp_probabilities,
            prediction_percentiles=None,
            width_ref_pct=20.0,
            bin_step_ref_bps=10.0,
            pools_by_address={pool.address: pool},
            pre_score_by_addr={},
            scoring_objective="odds_15m_exact",
            hold_oor_cycles=4,
        )

        self.assertIsNotNone(scored)
        assert scored is not None
        self.assertAlmostEqual(scored.forecast.lower_bound, 95.0, places=9)
        self.assertAlmostEqual(scored.forecast.upper_bound, 105.0, places=9)
        self.assertEqual(scored.ev_components.baseline_mode, "current_hold_hybrid_bounds")
        self.assertEqual(scored.ev_components.out_of_range_cycles, 4)

    def test_oor_hold_cycles_increment_and_reset(self):
        pool = _pool(current_price=110.0)
        orchestrator = self._orchestrator(pool)
        state = BotState(
            active_position=ActivePositionState(
                pool_address=pool.address,
                lower_price=95.0,
                upper_price=105.0,
                width_pct=10.0,
                executor="dry-run",
            ),
            strategy_state={"oor_hold_cycles": 2},
        )

        cycles = orchestrator._update_oor_hold_state(
            state=state,
            pools_by_address={pool.address: pool},
        )
        self.assertEqual(cycles, 3)
        self.assertEqual(state.strategy_state.get("oor_hold_cycles"), 3)
        self.assertFalse(state.strategy_state.get("last_active_in_range"))

        pool.current_price = 100.0
        cycles = orchestrator._update_oor_hold_state(
            state=state,
            pools_by_address={pool.address: pool},
        )
        self.assertEqual(cycles, 0)
        self.assertEqual(state.strategy_state.get("oor_hold_cycles"), 0)
        self.assertTrue(state.strategy_state.get("last_active_in_range"))


if __name__ == "__main__":
    unittest.main()
