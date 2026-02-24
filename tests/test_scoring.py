import unittest

from ai_liquidity_optimizer.models import MeteoraPoolSnapshot, SynthLpBoundForecast
from ai_liquidity_optimizer.strategy.scoring import StrategyScorer, horizon_to_minutes, relative_range_change_bps


class ScoringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pool = MeteoraPoolSnapshot(
            address="pool",
            name="SOL/USDC",
            mint_x="x",
            mint_y="y",
            symbol_x="SOL",
            symbol_y="USDC",
            decimals_x=9,
            decimals_y=6,
            current_price=200.0,
            liquidity=1_000_000.0,
            volume_24h=200_000.0,
            fees_24h=2_000.0,
            fee_tvl_ratio_24h=0.002,  # 0.2%
            raw={},
        )

    def test_horizon_parser(self) -> None:
        self.assertEqual(horizon_to_minutes("24h"), 1440)
        self.assertEqual(horizon_to_minutes("7d"), 10080)

    def test_prefers_higher_net_expected_return(self) -> None:
        forecasts = [
            SynthLpBoundForecast(
                width_pct=5.0,
                lower_bound=190,
                upper_bound=210,
                probability_to_stay_in_interval=0.40,
                expected_time_in_interval_minutes=400,
                expected_impermanent_loss=0.0010,
            ),
            SynthLpBoundForecast(
                width_pct=10.0,
                lower_bound=180,
                upper_bound=220,
                probability_to_stay_in_interval=0.70,
                expected_time_in_interval_minutes=900,
                expected_impermanent_loss=0.0008,
            ),
        ]
        scorer = StrategyScorer(min_stay_probability=0.0)
        decision = scorer.rank_candidates(forecasts=forecasts, pool=self.pool, horizon="24h")
        self.assertEqual(decision.chosen.forecast.width_pct, 10.0)

    def test_range_change_bps(self) -> None:
        delta = relative_range_change_bps(190, 210, 191, 211)
        self.assertGreater(delta, 0)
        self.assertLess(delta, 100)


if __name__ == "__main__":
    unittest.main()

