import unittest

from ai_liquidity_optimizer.models import (
    ActivePositionState,
    BinWeightingDiagnostics,
    BotState,
    MeteoraPoolSnapshot,
    SynthLpBoundForecast,
    SynthPredictionPercentilesSnapshot,
    WeightedBinPlan,
)
from ai_liquidity_optimizer.orchestrator import _should_rebalance_ev
from ai_liquidity_optimizer.strategy.ev import EvLpScorer


def _pool(
    *,
    address: str,
    tvl: float,
    current_price: float = 100.0,
    bin_step_bps: float = 10.0,
    fee_1h: float | None = None,
    fee_24h: float | None = None,
    vol_1h: float = 0.0,
    vol_24h: float = 0.0,
    base_fee_pct: float = 0.2,
    dynamic_fee_pct: float = 0.05,
) -> MeteoraPoolSnapshot:
    fee_map = {}
    if fee_1h is not None:
        fee_map["1h"] = fee_1h
    if fee_24h is not None:
        fee_map["24h"] = fee_24h
    vol_map = {}
    if vol_1h:
        vol_map["1h"] = vol_1h
    if vol_24h:
        vol_map["24h"] = vol_24h
    return MeteoraPoolSnapshot(
        address=address,
        name=f"SOL-USDC-{address}",
        mint_x="x",
        mint_y="y",
        symbol_x="SOL",
        symbol_y="USDC",
        decimals_x=9,
        decimals_y=6,
        current_price=current_price,
        liquidity=tvl,
        volume_24h=vol_24h,
        fees_24h=0.0,
        fee_tvl_ratio_24h=fee_24h,
        tvl=tvl,
        bin_step_bps=bin_step_bps,
        base_fee_pct=base_fee_pct,
        dynamic_fee_pct=dynamic_fee_pct,
        fee_tvl_ratio_by_window=fee_map,
        volume_by_window=vol_map,
        fees_by_window={},
        raw={"pool_config": {"bin_step": bin_step_bps}},
    )


def _forecast(*, lower: float = 95.0, upper: float = 105.0, il: float = 0.002, t_in: float = 60.0) -> SynthLpBoundForecast:
    mid = (lower + upper) / 2.0
    return SynthLpBoundForecast(
        width_pct=((upper - lower) / mid) * 100.0,
        lower_bound=lower,
        upper_bound=upper,
        probability_to_stay_in_interval=0.9,
        expected_time_in_interval_minutes=t_in,
        expected_impermanent_loss=il,
    )


def _weighted_plan(*, lower: float = 95.0, upper: float = 105.0, weights: list[float] | None = None, mass_in_range: float = 1.0) -> WeightedBinPlan:
    w = weights or [0.5, 0.5]
    return WeightedBinPlan(
        range_lower=lower,
        range_upper=upper,
        bin_edges=[lower, (lower + upper) / 2.0, upper],
        weights=w,
        diagnostics=BinWeightingDiagnostics(
            mass_in_range=mass_in_range,
            used_prediction_percentiles=False,
            fallback_reason=None,
            confidence_factor=1.0,
            t_frac=1.0,
            entropy=0.0,
            terminal_cdf_points=11,
            num_bins=len(w),
            binning_mode="test",
        ),
        distribution_components={},
    )


class EvLpScorerTests(unittest.TestCase):
    def test_pool_pre_rank_uses_more_than_tvl(self):
        scorer = EvLpScorer()
        high_tvl_low_fee = _pool(
            address="A",
            tvl=5_000_000,
            bin_step_bps=25,
            fee_1h=0.001,
            fee_24h=0.01,
            vol_1h=20_000,
            vol_24h=200_000,
        )
        lower_tvl_better_yield = _pool(
            address="B",
            tvl=1_000_000,
            bin_step_bps=5,
            fee_1h=0.01,
            fee_24h=0.06,
            vol_1h=250_000,
            vol_24h=2_000_000,
        )
        ranked = scorer.pool_pre_rank([high_tvl_low_fee, lower_tvl_better_yield], limit=2)
        self.assertEqual([r.pool_address for r in ranked], ["B", "A"])
        self.assertIn("bin_step_fit_rank", ranked[0].components)
        self.assertIn("realized_fee_rate_15m_fraction", ranked[0].components)

    def test_same_pool_rebalance_cost_reduces_ev(self):
        scorer = EvLpScorer(rebalance_cost_usd=0.5, pool_switch_extra_cost_usd=1.0)
        pool = _pool(address="A", tvl=1_000_000, fee_1h=0.02, fee_24h=0.08, vol_1h=200_000, vol_24h=1_000_000)
        forecast = _forecast(lower=94, upper=106, il=0.0005, t_in=60.0)
        plan = _weighted_plan(lower=94, upper=106)
        active = ActivePositionState(pool_address="A", lower_price=90.0, upper_price=110.0, width_pct=20.0, executor="dry-run")

        with_cost = scorer.score_pool_range_ev_15m(
            pool=pool,
            forecast=forecast,
            weighted_bin_plan=plan,
            synth_horizon="1h",
            prediction_percentiles=None,
            deposit_sol_amount=1.0,
            deposit_usdc_amount=100.0,
            width_ref_pct=forecast.width_pct,
            bin_step_ref_bps=pool.bin_step_bps or 10.0,
            active_position=active,
            range_change_threshold_bps=1.0,
            apply_rebalance_costs=True,
        )
        no_cost = scorer.score_pool_range_ev_15m(
            pool=pool,
            forecast=forecast,
            weighted_bin_plan=plan,
            synth_horizon="1h",
            prediction_percentiles=None,
            deposit_sol_amount=1.0,
            deposit_usdc_amount=100.0,
            width_ref_pct=forecast.width_pct,
            bin_step_ref_bps=pool.bin_step_bps or 10.0,
            active_position=active,
            range_change_threshold_bps=1.0,
            apply_rebalance_costs=False,
        )
        self.assertAlmostEqual(no_cost.ev_15m_usd - with_cost.ev_15m_usd, 0.5, places=9)
        self.assertFalse(with_cost.pool_switch)

    def test_pool_switch_adds_extra_cost(self):
        scorer = EvLpScorer(rebalance_cost_usd=0.5, pool_switch_extra_cost_usd=1.0)
        pool = _pool(address="B", tvl=1_000_000, fee_1h=0.02, fee_24h=0.08)
        forecast = _forecast()
        plan = _weighted_plan()
        active = ActivePositionState(pool_address="A", lower_price=95.0, upper_price=105.0, width_pct=10.0, executor="dry-run")

        with_cost = scorer.score_pool_range_ev_15m(
            pool=pool,
            forecast=forecast,
            weighted_bin_plan=plan,
            synth_horizon="1h",
            prediction_percentiles=None,
            deposit_sol_amount=1.0,
            deposit_usdc_amount=100.0,
            width_ref_pct=forecast.width_pct,
            bin_step_ref_bps=pool.bin_step_bps or 10.0,
            active_position=active,
            range_change_threshold_bps=50.0,
            apply_rebalance_costs=True,
        )
        no_cost = scorer.score_pool_range_ev_15m(
            pool=pool,
            forecast=forecast,
            weighted_bin_plan=plan,
            synth_horizon="1h",
            prediction_percentiles=None,
            deposit_sol_amount=1.0,
            deposit_usdc_amount=100.0,
            width_ref_pct=forecast.width_pct,
            bin_step_ref_bps=pool.bin_step_bps or 10.0,
            active_position=active,
            range_change_threshold_bps=50.0,
            apply_rebalance_costs=False,
        )
        self.assertTrue(with_cost.pool_switch)
        self.assertAlmostEqual(no_cost.ev_15m_usd - with_cost.ev_15m_usd, 1.5, places=9)

    def test_active_occupancy_uses_first_15_minutes_of_percentiles(self):
        scorer = EvLpScorer(ev_horizon_minutes=15, ev_percentile_decay_half_life_minutes=15)
        forecast = _forecast(lower=90, upper=110)
        plan = _weighted_plan(lower=90, upper=110, weights=[0.1, 0.9])
        percentiles = SynthPredictionPercentilesSnapshot(
            asset="SOL",
            current_price=100.0,
            step_minutes=5,
            percentiles_by_step=[
                {0.0: 100.0, 100.0: 110.0},  # t=0 upper bin
                {0.0: 100.0, 100.0: 110.0},  # t=5 upper bin
                {0.0: 100.0, 100.0: 110.0},  # t=10 upper bin
                {0.0: 100.0, 100.0: 110.0},  # t=15 upper bin
                {0.0: 90.0, 100.0: 100.0},   # t=20 lower bin (should be ignored)
                {0.0: 90.0, 100.0: 100.0},   # t=25 lower bin (should be ignored)
            ],
        )
        occ, source = scorer.active_occupancy_15m(
            forecast=forecast,
            synth_horizon="1h",
            weighted_bin_plan=plan,
            prediction_percentiles=percentiles,
        )
        self.assertEqual(source, "prediction_percentiles_weighted")
        self.assertGreater(occ, 0.8)

    def test_ev_gate_blocks_small_delta(self):
        scorer = EvLpScorer()
        pool = _pool(address="A", tvl=1_000_000, fee_1h=0.02, fee_24h=0.08)
        candidate = scorer.score_pool_range_ev_15m(
            pool=pool,
            forecast=_forecast(),
            weighted_bin_plan=_weighted_plan(),
            synth_horizon="1h",
            prediction_percentiles=None,
            deposit_sol_amount=1.0,
            deposit_usdc_amount=100.0,
            width_ref_pct=10.0,
            bin_step_ref_bps=10.0,
            active_position=ActivePositionState(pool_address="A", lower_price=90, upper_price=110, width_pct=20, executor="dry-run"),
            range_change_threshold_bps=1.0,
            apply_rebalance_costs=True,
        )
        state = BotState(active_position=ActivePositionState(pool_address="A", lower_price=90, upper_price=110, width_pct=20, executor="dry-run"))
        should_rebalance, reason, gate = _should_rebalance_ev(
            state=state,
            best=candidate,
            ev_delta_usd=0.10,
            min_ev_improvement_usd=0.25,
        )
        self.assertFalse(should_rebalance)
        self.assertTrue(str(reason).startswith("ev_delta_below_threshold_"))
        self.assertFalse(bool(gate.get("ev_threshold_passed")))

    def test_ev_gate_requires_structural_change(self):
        scorer = EvLpScorer()
        pool = _pool(address="A", tvl=1_000_000, fee_1h=0.02, fee_24h=0.08)
        active = ActivePositionState(pool_address="A", lower_price=95, upper_price=105, width_pct=10, executor="dry-run")
        candidate = scorer.score_pool_range_ev_15m(
            pool=pool,
            forecast=_forecast(lower=95.0001, upper=105.0001),
            weighted_bin_plan=_weighted_plan(lower=95.0001, upper=105.0001),
            synth_horizon="1h",
            prediction_percentiles=None,
            deposit_sol_amount=1.0,
            deposit_usdc_amount=100.0,
            width_ref_pct=10.0,
            bin_step_ref_bps=10.0,
            active_position=active,
            range_change_threshold_bps=1_000.0,  # force "not structural"
            apply_rebalance_costs=True,
        )
        state = BotState(active_position=active)
        should_rebalance, _reason, gate = _should_rebalance_ev(
            state=state,
            best=candidate,
            ev_delta_usd=10.0,
            min_ev_improvement_usd=0.25,
        )
        self.assertFalse(should_rebalance)
        self.assertFalse(bool(gate.get("structural_change_passed")))


if __name__ == "__main__":
    unittest.main()
