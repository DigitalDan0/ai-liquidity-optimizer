from __future__ import annotations

import unittest

from ai_liquidity_optimizer.models import MeteoraPoolSnapshot, SynthLpBoundForecast, SynthPredictionPercentilesSnapshot
from ai_liquidity_optimizer.strategy.ev import EvLpScorer
from ai_liquidity_optimizer.strategy.scoring import StrategyScorer
from ai_liquidity_optimizer.strategy.synth_market import (
    SynthFusionConfig,
    build_candidate_ladder,
    build_synthetic_forecasts_from_prediction_percentiles,
    build_synth_market_state,
)


def _pool(spot: float = 100.0) -> MeteoraPoolSnapshot:
    return MeteoraPoolSnapshot(
        address="pool-a",
        name="SOL-USDC",
        mint_x="x",
        mint_y="y",
        symbol_x="SOL",
        symbol_y="USDC",
        decimals_x=9,
        decimals_y=6,
        current_price=spot,
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


def _forecast(*, lower: float, upper: float, p_stay: float, t_in: float, il: float) -> SynthLpBoundForecast:
    return SynthLpBoundForecast(
        width_pct=((upper / lower) ** 0.5 - 1.0) * 200.0,
        lower_bound=lower,
        upper_bound=upper,
        probability_to_stay_in_interval=p_stay,
        expected_time_in_interval_minutes=t_in,
        expected_impermanent_loss=il,
    )


def _config() -> SynthFusionConfig:
    return SynthFusionConfig(
        horizons=["15m", "1h", "4h", "24h"],
        range_max_center_drift_ratio=0.25,
        trend_min_center_drift_ratio=0.40,
        trend_min_onesided_prob=0.45,
        min_agreement_score=0.60,
        uncertain_width_expansion=0.35,
        entry_conflict_threshold=0.45,
        size_low_confidence=0.35,
        size_medium_confidence=0.55,
        size_full_confidence=0.75,
    )


class SynthMarketStateTests(unittest.TestCase):
    def test_builds_synthetic_15m_forecasts_from_prediction_percentiles(self):
        snapshot = SynthPredictionPercentilesSnapshot(
            asset="SOL",
            current_price=100.0,
            step_minutes=5,
            percentiles_by_step=[
                {10.0: 99.0, 50.0: 100.0, 90.0: 101.0},
                {10.0: 98.8, 50.0: 100.2, 90.0: 101.6},
                {10.0: 98.6, 50.0: 100.4, 90.0: 101.9},
                {10.0: 98.4, 50.0: 100.5, 90.0: 102.1},
            ],
        )

        forecasts = build_synthetic_forecasts_from_prediction_percentiles(
            prediction_percentiles=snapshot,
            horizon_minutes=15,
            current_price=100.0,
        )

        self.assertGreaterEqual(len(forecasts), 3)
        self.assertTrue(all(f.lower_bound < f.upper_bound for f in forecasts))
        self.assertTrue(all(0.0 < f.probability_to_stay_in_interval <= 1.0 for f in forecasts))
        self.assertTrue(all(f.expected_time_in_interval_minutes > 0.0 for f in forecasts))
        self.assertTrue(all(f.expected_impermanent_loss >= 0.0 for f in forecasts))
        self.assertEqual([round(f.width_pct, 6) for f in forecasts], sorted(round(f.width_pct, 6) for f in forecasts))

    def test_builds_range_regime_with_high_agreement(self):
        state = build_synth_market_state(
            representative_pool=_pool(),
            forecasts_by_horizon={
                "15m": [_forecast(lower=99.2, upper=100.8, p_stay=0.85, t_in=13.0, il=0.002)],
                "1h": [_forecast(lower=98.8, upper=101.2, p_stay=0.82, t_in=46.0, il=0.003)],
                "4h": [_forecast(lower=98.0, upper=102.0, p_stay=0.75, t_in=170.0, il=0.004)],
            },
            prediction_percentiles=None,
            scorer=StrategyScorer(),
            ev_scorer=EvLpScorer(),
            config=_config(),
        )

        self.assertIsNotNone(state)
        assert state is not None
        self.assertEqual(state.market_regime, "range")
        self.assertGreaterEqual(state.horizon_agreement_score, 0.60)
        self.assertIn(state.size_multiplier, {0.33, 0.66, 1.0})

    def test_builds_trend_up_regime_with_defensive_width(self):
        state = build_synth_market_state(
            representative_pool=_pool(),
            forecasts_by_horizon={
                "15m": [_forecast(lower=102.5, upper=103.5, p_stay=0.20, t_in=1.0, il=0.003)],
                "1h": [_forecast(lower=102.0, upper=104.0, p_stay=0.25, t_in=6.0, il=0.004)],
                "4h": [_forecast(lower=101.0, upper=105.0, p_stay=0.40, t_in=40.0, il=0.005)],
            },
            prediction_percentiles=None,
            scorer=StrategyScorer(),
            ev_scorer=EvLpScorer(),
            config=_config(),
        )

        self.assertIsNotNone(state)
        assert state is not None
        self.assertEqual(state.market_regime, "trend_up")
        self.assertGreaterEqual(state.horizon_agreement_score, 0.60)
        self.assertGreater(state.fused_width_pct, state.horizons["15m"].width_pct)
        self.assertIn(state.size_multiplier, {0.33, 0.66, 1.0})

    def test_candidate_ladder_includes_fused_trend_and_defensive_families(self):
        state = build_synth_market_state(
            representative_pool=_pool(),
            forecasts_by_horizon={
                "15m": [_forecast(lower=102.5, upper=103.5, p_stay=0.20, t_in=1.0, il=0.003)],
                "1h": [_forecast(lower=102.0, upper=104.0, p_stay=0.25, t_in=6.0, il=0.004)],
                "4h": [_forecast(lower=101.0, upper=105.0, p_stay=0.40, t_in=40.0, il=0.005)],
            },
            prediction_percentiles=None,
            scorer=StrategyScorer(),
            ev_scorer=EvLpScorer(),
            config=_config(),
        )
        assert state is not None

        ladder = build_candidate_ladder(state)
        families = {family for family, _forecast in ladder}

        self.assertIn("fused_symmetric", families)
        self.assertIn("trend_skewed", families)
        self.assertIn("defensive_wide", families)


if __name__ == "__main__":
    unittest.main()
