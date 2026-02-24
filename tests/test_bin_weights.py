import math
import unittest

from ai_liquidity_optimizer.models import (
    BinWeightingConfig,
    MeteoraPoolSnapshot,
    SynthLpBoundForecast,
    SynthLpProbabilitiesSnapshot,
    SynthLpProbabilityPoint,
    SynthPredictionPercentilesSnapshot,
)
from ai_liquidity_optimizer.strategy.bin_weights import (
    build_terminal_cdf_from_lp_probabilities,
    compute_bin_weights_for_range,
    compute_time_decayed_occupancy_from_percentiles,
    derive_mvp_bin_edges_for_range,
)


def _lp_probs(points):
    return SynthLpProbabilitiesSnapshot(asset="SOL", horizon="24h", points=points)


def _forecast(lower=180.0, upper=220.0, p_stay=0.8, t_in=1000.0, il=0.001):
    mid = (lower + upper) / 2.0
    width_pct = ((upper - lower) / mid) * 100.0
    return SynthLpBoundForecast(
        width_pct=width_pct,
        lower_bound=lower,
        upper_bound=upper,
        probability_to_stay_in_interval=p_stay,
        expected_time_in_interval_minutes=t_in,
        expected_impermanent_loss=il,
    )


class TerminalCdfTests(unittest.TestCase):
    def test_monotonic_repair(self):
        snap = _lp_probs(
            [
                SynthLpProbabilityPoint(price=180, probability_below=0.30),
                SynthLpProbabilityPoint(price=200, probability_below=0.25),  # non-monotone
                SynthLpProbabilityPoint(price=220, probability_below=0.80),
            ]
        )
        cdf = build_terminal_cdf_from_lp_probabilities(snap)
        self.assertGreaterEqual(cdf.evaluate(200), cdf.evaluate(180))
        self.assertGreaterEqual(cdf.evaluate(220), cdf.evaluate(200))

    def test_blends_below_and_above(self):
        snap = _lp_probs([SynthLpProbabilityPoint(price=200, probability_below=0.52, probability_above=0.45)])
        cdf = build_terminal_cdf_from_lp_probabilities(snap)
        # average of below and (1-above) = average(0.52, 0.55) = 0.535
        self.assertAlmostEqual(cdf.evaluate(200), 0.535, places=6)

    def test_piecewise_constant_tails(self):
        snap = _lp_probs(
            [
                SynthLpProbabilityPoint(price=100, probability_below=0.1),
                SynthLpProbabilityPoint(price=200, probability_below=0.9),
            ]
        )
        cdf = build_terminal_cdf_from_lp_probabilities(snap)
        self.assertAlmostEqual(cdf.evaluate(50), 0.1)
        self.assertAlmostEqual(cdf.evaluate(300), 0.9)


class BinWeightingTests(unittest.TestCase):
    def test_mass_conservation_and_normalization(self):
        snap = _lp_probs(
            [
                SynthLpProbabilityPoint(price=180, probability_below=0.2),
                SynthLpProbabilityPoint(price=200, probability_below=0.5),
                SynthLpProbabilityPoint(price=220, probability_below=0.8),
            ]
        )
        edges = [180, 190, 200, 210, 220]
        plan = compute_bin_weights_for_range(
            forecast=_forecast(),
            horizon="24h",
            bin_edges=edges,
            current_price=200.0,
            lp_probabilities=snap,
        )
        self.assertAlmostEqual(sum(plan.weights), 1.0, places=9)
        self.assertTrue(all(w >= 0 for w in plan.weights))
        self.assertGreater(plan.diagnostics.mass_in_range, 0.0)

    def test_symmetric_distribution_yields_near_symmetric_weights(self):
        snap = _lp_probs(
            [
                SynthLpProbabilityPoint(price=180, probability_below=0.1),
                SynthLpProbabilityPoint(price=190, probability_below=0.3),
                SynthLpProbabilityPoint(price=200, probability_below=0.5),
                SynthLpProbabilityPoint(price=210, probability_below=0.7),
                SynthLpProbabilityPoint(price=220, probability_below=0.9),
            ]
        )
        edges = [180, 190, 200, 210, 220]
        plan = compute_bin_weights_for_range(
            forecast=_forecast(p_stay=0.95, t_in=1200),
            horizon="24h",
            bin_edges=edges,
            current_price=200.0,
            lp_probabilities=snap,
        )
        self.assertAlmostEqual(plan.weights[0], plan.weights[-1], delta=0.05)
        self.assertAlmostEqual(plan.weights[1], plan.weights[-2], delta=0.05)

    def test_upward_skew_allocates_more_to_upper_bins(self):
        snap = _lp_probs(
            [
                SynthLpProbabilityPoint(price=180, probability_below=0.02),
                SynthLpProbabilityPoint(price=190, probability_below=0.08),
                SynthLpProbabilityPoint(price=200, probability_below=0.20),
                SynthLpProbabilityPoint(price=210, probability_below=0.45),
                SynthLpProbabilityPoint(price=220, probability_below=0.85),
            ]
        )
        edges = [180, 190, 200, 210, 220]
        plan = compute_bin_weights_for_range(
            forecast=_forecast(),
            horizon="24h",
            bin_edges=edges,
            current_price=200.0,
            lp_probabilities=snap,
        )
        self.assertGreater(sum(plan.weights[2:]), sum(plan.weights[:2]))

    def test_low_confidence_flattens_distribution(self):
        snap = _lp_probs(
            [
                SynthLpProbabilityPoint(price=180, probability_below=0.05),
                SynthLpProbabilityPoint(price=190, probability_below=0.10),
                SynthLpProbabilityPoint(price=200, probability_below=0.20),
                SynthLpProbabilityPoint(price=210, probability_below=0.50),
                SynthLpProbabilityPoint(price=220, probability_below=0.90),
            ]
        )
        edges = [180, 190, 200, 210, 220]
        hi = compute_bin_weights_for_range(
            forecast=_forecast(p_stay=0.95, t_in=1300),
            horizon="24h",
            bin_edges=edges,
            current_price=200.0,
            lp_probabilities=snap,
        )
        lo = compute_bin_weights_for_range(
            forecast=_forecast(p_stay=0.03, t_in=120),
            horizon="24h",
            bin_edges=edges,
            current_price=200.0,
            lp_probabilities=snap,
        )
        uniform = [1.0 / len(hi.weights)] * len(hi.weights)
        dist_hi = sum(abs(a - b) for a, b in zip(hi.weights, uniform))
        dist_lo = sum(abs(a - b) for a, b in zip(lo.weights, uniform))
        self.assertLess(dist_lo, dist_hi)

    def test_low_mass_fallback_uses_proximity_prior(self):
        snap = _lp_probs(
            [
                SynthLpProbabilityPoint(price=10, probability_below=0.1),
                SynthLpProbabilityPoint(price=20, probability_below=0.9),
            ]
        )
        edges = [180, 190, 200, 210, 220]
        plan = compute_bin_weights_for_range(
            forecast=_forecast(),
            horizon="24h",
            bin_edges=edges,
            current_price=200.0,
            lp_probabilities=snap,
        )
        self.assertEqual(plan.diagnostics.fallback_reason, "low_mass_in_range")
        self.assertAlmostEqual(sum(plan.weights), 1.0, places=9)
        # Proximity prior should center near current price.
        self.assertGreater(plan.weights[1] + plan.weights[2], plan.weights[0] + plan.weights[3])

    def test_single_and_two_bin_edge_cases(self):
        snap = _lp_probs(
            [
                SynthLpProbabilityPoint(price=180, probability_below=0.2),
                SynthLpProbabilityPoint(price=220, probability_below=0.8),
            ]
        )
        one_bin = compute_bin_weights_for_range(
            forecast=_forecast(),
            horizon="24h",
            bin_edges=[180, 220],
            current_price=200.0,
            lp_probabilities=snap,
        )
        self.assertEqual(one_bin.weights, [1.0])

        two_bins = compute_bin_weights_for_range(
            forecast=_forecast(),
            horizon="24h",
            bin_edges=[180, 200, 220],
            current_price=200.0,
            lp_probabilities=snap,
        )
        self.assertAlmostEqual(sum(two_bins.weights), 1.0, places=9)
        self.assertEqual(len(two_bins.weights), 2)

    def test_prediction_percentiles_near_term_weighting(self):
        snap = _lp_probs(
            [
                SynthLpProbabilityPoint(price=90, probability_below=0.0),
                SynthLpProbabilityPoint(price=100, probability_below=0.5),
                SynthLpProbabilityPoint(price=110, probability_below=1.0),
            ]
        )
        percentiles = SynthPredictionPercentilesSnapshot(
            asset="SOL",
            current_price=100.0,
            step_minutes=5,
            percentiles_by_step=[
                {0.0: 100.0, 100.0: 110.0},  # near-term occupancy in upper bin
                {0.0: 90.0, 100.0: 100.0},   # later occupancy in lower bin
            ],
        )
        occupancy = compute_time_decayed_occupancy_from_percentiles(
            bin_edges=[90, 100, 110],
            prediction_percentiles=percentiles,
            tau_half_minutes=90,
        )
        self.assertIsNotNone(occupancy)
        assert occupancy is not None
        self.assertGreater(occupancy[1], occupancy[0])

        plan = compute_bin_weights_for_range(
            forecast=_forecast(lower=90, upper=110, p_stay=0.8, t_in=900),
            horizon="24h",
            bin_edges=[90, 100, 110],
            current_price=100.0,
            lp_probabilities=snap,
            prediction_percentiles=percentiles,
        )
        self.assertTrue(plan.diagnostics.used_prediction_percentiles)
        self.assertGreater(plan.weights[1], plan.weights[0])

    def test_derive_mvp_bin_edges_uses_bin_step_when_present(self):
        pool = MeteoraPoolSnapshot(
            address="pool",
            name="SOL/USDC",
            mint_x="x",
            mint_y="y",
            symbol_x="SOL",
            symbol_y="USDC",
            decimals_x=9,
            decimals_y=6,
            current_price=200.0,
            liquidity=0.0,
            volume_24h=0.0,
            fees_24h=0.0,
            raw={"bin_step": 25},
        )
        edges, mode = derive_mvp_bin_edges_for_range(pool, 190.0, 210.0)
        self.assertEqual(mode, "meteora_bin_step")
        self.assertAlmostEqual(edges[0], 190.0)
        self.assertAlmostEqual(edges[-1], 210.0)
        self.assertGreaterEqual(len(edges), 2)


if __name__ == "__main__":
    unittest.main()

