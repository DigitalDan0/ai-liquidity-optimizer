from __future__ import annotations

import unittest
from types import SimpleNamespace

from ai_liquidity_optimizer.execution.base import PositionExecutor
from ai_liquidity_optimizer.models import (
    ActivePositionState,
    BinWeightingDiagnostics,
    BotState,
    EvComponentBreakdown,
    EvScoredCandidate,
    ExecutionApplyRequest,
    ExecutionApplyResult,
    MeteoraPoolSnapshot,
    SynthLpBoundForecast,
    WeightedBinPlan,
)
from ai_liquidity_optimizer.orchestrator import OptimizerOrchestrator
from ai_liquidity_optimizer.strategy.ev import EvLpScorer
from ai_liquidity_optimizer.strategy.scoring import StrategyScorer


class _StubExecutor(PositionExecutor):
    def apply_target_range(self, request: ExecutionApplyRequest) -> ExecutionApplyResult:  # pragma: no cover - not used
        raise RuntimeError("not used")


class _StubStateStore:
    def load(self):  # pragma: no cover - not used
        return BotState()

    def save(self, state):  # pragma: no cover - not used
        return None


def _pool() -> MeteoraPoolSnapshot:
    return MeteoraPoolSnapshot(
        address="pool-a",
        name="SOL-USDC",
        mint_x="x",
        mint_y="y",
        symbol_x="SOL",
        symbol_y="USDC",
        decimals_x=9,
        decimals_y=6,
        current_price=85.0,
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


def _candidate(
    *,
    ev_usd: float,
    action_type: str,
    structural_change: bool,
    il_15m_fraction: float = 0.0004,
    occ: float = 0.9,
    one_sided: float = 0.2,
    directional: float = 0.2,
) -> EvScoredCandidate:
    forecast = SynthLpBoundForecast(
        width_pct=1.2,
        lower_bound=84.0,
        upper_bound=86.0,
        probability_to_stay_in_interval=0.9,
        expected_time_in_interval_minutes=55.0,
        expected_impermanent_loss=0.002,
    )
    plan = WeightedBinPlan(
        range_lower=84.0,
        range_upper=86.0,
        bin_edges=[84.0, 85.0, 86.0],
        weights=[0.4, 0.6],
        diagnostics=BinWeightingDiagnostics(
            mass_in_range=occ,
            used_prediction_percentiles=True,
            fallback_reason=None,
            confidence_factor=1.0,
            t_frac=1.0,
            entropy=0.0,
            terminal_cdf_points=0,
            num_bins=2,
            binning_mode="test",
        ),
    )
    components = EvComponentBreakdown(
        expected_fees_usd=0.05,
        expected_il_usd=0.01,
        rebalance_cost_usd=0.0012,
        pool_switch_extra_cost_usd=0.0,
        active_occupancy_15m=occ,
        concentration_factor=1.2,
        fee_rate_15m_fraction=0.002,
        capital_usd=100.0,
        il_15m_fraction=il_15m_fraction,
        range_active_occupancy_15m=occ,
        one_sided_break_prob=one_sided,
        directional_confidence=directional,
    )
    return EvScoredCandidate(
        pool_address="pool-a",
        pool_name="SOL-USDC",
        pool_symbol_pair="SOL/USDC",
        pool_current_price_sol_usdc=85.0,
        forecast=forecast,
        weighted_bin_plan=plan,
        ev_15m_usd=ev_usd,
        ev_components=components,
        rebalance_structural_change=structural_change,
        pool_switch=False,
        range_change_bps_vs_active=50.0 if structural_change else 0.0,
        action_type=action_type,
    )


class LifecyclePolicyDecisionTests(unittest.TestCase):
    def _orchestrator(self) -> OptimizerOrchestrator:
        settings = SimpleNamespace(
            min_ev_improvement_usd=0.01,
            ev_utilization_floor=0.40,
            ev_min_utilization_gain=0.15,
            ev_max_protective_ev_slip_usd=0.002,
            ev_protective_breach_cycles=2,
            ev_idle_enabled=True,
            ev_idle_entry_threshold_usd=0.02,
            ev_idle_exit_threshold_usd=0.02,
            ev_idle_confirm_cycles=2,
            ev_trend_stop_enabled=False,
            ev_trend_stop_oor_cycles=3,
            ev_trend_stop_onesided_prob=0.65,
            ev_action_cooldown_cycles=0,
            ev_policy_lifecycle_enabled=True,
            ev_profit_take_pct=0.015,
            ev_profit_rebalance_max_slip_usd=0.002,
            ev_loss_recovery_trigger_pct=0.020,
            ev_high_il_15m_fraction=0.0006,
            ev_recovery_min_prob=0.55,
            ev_trend_continue_exit_prob=0.72,
            ev_trend_exit_persistence_cycles=2,
            ev_idle_preference_edge_usd=0.01,
        )
        return OptimizerOrchestrator(
            settings=settings,
            synth_client=SimpleNamespace(),
            meteora_client=SimpleNamespace(),
            scorer=StrategyScorer(),
            ev_scorer=EvLpScorer(),
            executor=_StubExecutor(),
            state_store=_StubStateStore(),
        )

    def _active_state(self, *, loss_exit_breach_count: int = 0) -> BotState:
        return BotState(
            active_position=ActivePositionState(
                pool_address="pool-a",
                lower_price=84.0,
                upper_price=86.0,
                width_pct=2.0,
                executor="dry-run",
                position_pubkey="pos-1",
            ),
            strategy_state={"loss_exit_breach_count": loss_exit_breach_count},
        )

    def test_profit_lock_upgrades_hold_to_rebalance(self):
        orchestrator = self._orchestrator()
        state = self._active_state()
        best = _candidate(ev_usd=0.049, action_type="rebalance", structural_change=True)
        hold = _candidate(ev_usd=0.050, action_type="hold", structural_change=False)
        idle = _candidate(ev_usd=-0.001, action_type="idle", structural_change=True)

        selected_action, should_rebalance, should_close_idle, reason, gate, _protective, _delta, _scores = (
            orchestrator._decide_ev_action(
                state=state,
                best=best,
                ev_current_hold=hold,
                idle_candidate=idle,
                prior_protective_breach_count=0,
                hold_oor_cycles=0,
                action_cooldown_remaining=0,
                lifecycle_metrics={"lifecycle_pnl_usd": 2.0, "lifecycle_pnl_pct": 0.02},
            )
        )

        self.assertEqual(selected_action, "rebalance")
        self.assertTrue(should_rebalance)
        self.assertFalse(should_close_idle)
        self.assertEqual(reason, "profit_lock_rotate")
        self.assertEqual(gate.get("policy_selected_override"), "profit_lock")
        self.assertTrue(bool(gate.get("policy_profit_triggered")))

    def test_loss_recovery_holds_instead_of_rebalance(self):
        orchestrator = self._orchestrator()
        state = self._active_state()
        best = _candidate(ev_usd=0.080, action_type="rebalance", structural_change=True)
        hold = _candidate(
            ev_usd=0.050,
            action_type="hold",
            structural_change=False,
            il_15m_fraction=0.0010,
            occ=0.95,
            one_sided=0.10,
            directional=0.10,
        )
        idle = _candidate(ev_usd=0.0, action_type="idle", structural_change=True)

        selected_action, should_rebalance, should_close_idle, reason, gate, _protective, _delta, _scores = (
            orchestrator._decide_ev_action(
                state=state,
                best=best,
                ev_current_hold=hold,
                idle_candidate=idle,
                prior_protective_breach_count=0,
                hold_oor_cycles=0,
                action_cooldown_remaining=0,
                lifecycle_metrics={"lifecycle_pnl_usd": -3.0, "lifecycle_pnl_pct": -0.03},
            )
        )

        self.assertEqual(selected_action, "hold")
        self.assertFalse(should_rebalance)
        self.assertFalse(should_close_idle)
        self.assertEqual(reason, "loss_recovery_hold")
        self.assertEqual(gate.get("policy_selected_override"), "loss_hold")
        self.assertTrue(bool(gate.get("policy_loss_recovery_hold_triggered")))

    def test_loss_trend_cut_rotates_after_persistence(self):
        orchestrator = self._orchestrator()
        state = self._active_state(loss_exit_breach_count=1)
        best = _candidate(ev_usd=0.031, action_type="rebalance", structural_change=True)
        hold = _candidate(
            ev_usd=0.030,
            action_type="hold",
            structural_change=False,
            il_15m_fraction=0.0012,
            occ=0.20,
            one_sided=0.95,
            directional=0.90,
        )
        idle = _candidate(ev_usd=0.0, action_type="idle", structural_change=True)

        selected_action, should_rebalance, should_close_idle, reason, gate, _protective, _delta, _scores = (
            orchestrator._decide_ev_action(
                state=state,
                best=best,
                ev_current_hold=hold,
                idle_candidate=idle,
                prior_protective_breach_count=0,
                hold_oor_cycles=0,
                action_cooldown_remaining=0,
                lifecycle_metrics={"lifecycle_pnl_usd": -4.0, "lifecycle_pnl_pct": -0.04},
            )
        )

        self.assertEqual(selected_action, "rebalance")
        self.assertTrue(should_rebalance)
        self.assertFalse(should_close_idle)
        self.assertEqual(reason, "loss_trend_cut_rotate")
        self.assertEqual(gate.get("policy_selected_override"), "loss_cut_rotate")

    def test_dynamic_gate_reason_when_effective_threshold_is_higher(self):
        orchestrator = self._orchestrator()
        state = self._active_state()
        best = _candidate(ev_usd=0.025, action_type="rebalance", structural_change=True)
        hold = _candidate(ev_usd=0.020, action_type="hold", structural_change=False)
        idle = _candidate(ev_usd=-0.001, action_type="idle", structural_change=True)
        best.ev_components.adjusted_ev_15m_usd = 0.025
        hold.ev_components.adjusted_ev_15m_usd = 0.020
        idle.ev_components.adjusted_ev_15m_usd = -0.001

        selected_action, should_rebalance, _should_close_idle, reason, gate, _protective, _delta, _scores = (
            orchestrator._decide_ev_action(
                state=state,
                best=best,
                ev_current_hold=hold,
                idle_candidate=idle,
                prior_protective_breach_count=0,
                hold_oor_cycles=0,
                action_cooldown_remaining=0,
                lifecycle_metrics={"lifecycle_pnl_usd": 0.0, "lifecycle_pnl_pct": 0.0},
                use_adjusted_ev=True,
                effective_min_delta_usd=0.02,
            )
        )

        self.assertEqual(selected_action, "hold")
        self.assertFalse(should_rebalance)
        self.assertTrue(str(reason).startswith("delta_below_dynamic_gate_"))
        self.assertTrue(bool(gate.get("dynamic_gate_blocked")))

    def test_adjusted_ev_nonpositive_blocks_rebalance(self):
        orchestrator = self._orchestrator()
        state = self._active_state()
        best = _candidate(ev_usd=0.050, action_type="rebalance", structural_change=True)
        hold = _candidate(ev_usd=0.010, action_type="hold", structural_change=False)
        idle = _candidate(ev_usd=-0.001, action_type="idle", structural_change=True)
        best.ev_components.adjusted_ev_15m_usd = -0.002
        hold.ev_components.adjusted_ev_15m_usd = -0.020
        idle.ev_components.adjusted_ev_15m_usd = -0.001

        selected_action, should_rebalance, should_close_idle, reason, gate, _protective, _delta, _scores = (
            orchestrator._decide_ev_action(
                state=state,
                best=best,
                ev_current_hold=hold,
                idle_candidate=idle,
                prior_protective_breach_count=0,
                hold_oor_cycles=0,
                action_cooldown_remaining=0,
                lifecycle_metrics={"lifecycle_pnl_usd": 0.0, "lifecycle_pnl_pct": 0.0},
                use_adjusted_ev=True,
                effective_min_delta_usd=0.0,
            )
        )

        self.assertEqual(selected_action, "hold")
        self.assertFalse(should_rebalance)
        self.assertFalse(should_close_idle)
        self.assertEqual(reason, "adjusted_ev_nonpositive")
        self.assertFalse(bool(gate.get("adjusted_ev_positive_passed")))

    def test_oor_grace_holds_loss_position_before_confirming_move(self):
        orchestrator = self._orchestrator()
        state = self._active_state()
        best = _candidate(ev_usd=0.090, action_type="rebalance", structural_change=True)
        hold = _candidate(ev_usd=0.010, action_type="hold", structural_change=False)
        idle = _candidate(ev_usd=-0.001, action_type="idle", structural_change=True)
        hold.ev_components.out_of_range_bps = 40.0
        hold.ev_components.out_of_range_prob_15m = 0.55

        selected_action, should_rebalance, should_close_idle, reason, gate, _protective, _delta, _scores = (
            orchestrator._decide_ev_action(
                state=state,
                best=best,
                ev_current_hold=hold,
                idle_candidate=idle,
                prior_protective_breach_count=0,
                hold_oor_cycles=1,
                action_cooldown_remaining=0,
                lifecycle_metrics={"lifecycle_pnl_usd": -0.10, "lifecycle_pnl_pct": -0.01},
            )
        )

        self.assertEqual(selected_action, "hold")
        self.assertFalse(should_rebalance)
        self.assertFalse(should_close_idle)
        self.assertTrue(str(reason).startswith("oor_grace_hold_"))
        self.assertTrue(bool(gate.get("policy_oor_grace_triggered")))
        self.assertEqual(gate.get("policy_selected_override"), "oor_grace_hold")

    def test_oor_reentry_prob_holds_loss_position_after_grace(self):
        orchestrator = self._orchestrator()
        state = self._active_state()
        best = _candidate(ev_usd=0.090, action_type="rebalance", structural_change=True)
        hold = _candidate(ev_usd=0.020, action_type="hold", structural_change=False)
        idle = _candidate(ev_usd=-0.001, action_type="idle", structural_change=True)
        hold.ev_components.out_of_range_bps = 45.0
        hold.ev_components.out_of_range_prob_15m = 0.20  # re-entry prob = 0.80

        selected_action, should_rebalance, should_close_idle, reason, gate, _protective, _delta, _scores = (
            orchestrator._decide_ev_action(
                state=state,
                best=best,
                ev_current_hold=hold,
                idle_candidate=idle,
                prior_protective_breach_count=0,
                hold_oor_cycles=3,
                action_cooldown_remaining=0,
                lifecycle_metrics={"lifecycle_pnl_usd": -0.12, "lifecycle_pnl_pct": -0.012},
            )
        )

        self.assertEqual(selected_action, "hold")
        self.assertFalse(should_rebalance)
        self.assertFalse(should_close_idle)
        self.assertTrue(str(reason).startswith("oor_reentry_hold_"))
        self.assertTrue(bool(gate.get("policy_oor_reentry_hold_triggered")))
        self.assertEqual(gate.get("policy_selected_override"), "oor_reentry_hold")


if __name__ == "__main__":
    unittest.main()
