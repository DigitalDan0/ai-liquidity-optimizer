import unittest

from ai_liquidity_optimizer.clients.synth import SynthInsightsClient


class _FakeHttp:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def get_json(self, url, params=None, headers=None):
        self.calls.append((url, params or {}, headers or {}))
        for key, payload in self.responses.items():
            if url.endswith(key):
                return payload
        raise AssertionError(f"Unexpected URL: {url}")


class SynthClientParsingTests(unittest.TestCase):
    def test_lp_bounds_parses_nested_docs_shape(self):
        http = _FakeHttp(
            {
                "/insights/lp-bounds": {
                    "data": [
                        {
                            "interval": {
                                "full_width": "10.0%",
                                "lower_bound": "180",
                                "upper_bound": "220",
                            },
                            "probability_to_stay_in_interval": {"24": 0.71},
                            "expected_time_in_interval": {"24": 920},
                            "expected_impermanent_loss": {"24": 0.0012},
                        }
                    ]
                }
            }
        )
        client = SynthInsightsClient(base_url="https://api.synthdata.co", api_key="x", http_client=http)

        rows = client.get_lp_bounds(asset="SOL", horizon="24h")
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0].width_pct, 10.0)
        self.assertAlmostEqual(rows[0].lower_bound, 180.0)
        self.assertAlmostEqual(rows[0].upper_bound, 220.0)
        self.assertAlmostEqual(rows[0].probability_to_stay_in_interval, 0.71)
        self.assertAlmostEqual(rows[0].expected_time_in_interval_minutes, 920.0)
        self.assertAlmostEqual(rows[0].expected_impermanent_loss, 0.0012)

    def test_lp_bounds_parses_flattened_shape(self):
        http = _FakeHttp(
            {
                "/insights/lp-bounds": {
                    "data": [
                        {
                            "width_pct": 5,
                            "lower_bound": 190,
                            "upper_bound": 210,
                            "probability_to_stay_in_interval": 0.42,
                            "expected_time_in_interval": 500,
                            "expected_impermanent_loss": 0.0009,
                        }
                    ]
                }
            }
        )
        client = SynthInsightsClient(base_url="https://api.synthdata.co", api_key="x", http_client=http)
        rows = client.get_lp_bounds(asset="SOL", horizon="24h")
        self.assertEqual(rows[0].width_pct, 5.0)
        self.assertEqual(rows[0].lower_bound, 190.0)

    def test_lp_probabilities_parses_horizon_keyed_maps(self):
        http = _FakeHttp(
            {
                "/insights/lp-probabilities": {
                    "data": {
                        "24h": {
                            "probability_below": {"180": 0.2, "200": 0.5, "220": 0.8},
                            "probability_above": {"180": 0.81, "200": 0.49, "220": 0.19},
                        }
                    }
                }
            }
        )
        client = SynthInsightsClient(base_url="https://api.synthdata.co", api_key="x", http_client=http)
        snapshot = client.get_lp_probabilities(asset="SOL", horizon="24h")
        self.assertEqual(snapshot.asset, "SOL")
        self.assertEqual(snapshot.horizon, "24h")
        self.assertEqual([p.price for p in snapshot.points], [180.0, 200.0, 220.0])
        self.assertAlmostEqual(snapshot.points[1].probability_below or 0, 0.5)
        self.assertAlmostEqual(snapshot.points[1].probability_above or 0, 0.49)

    def test_prediction_percentiles_parses_rows(self):
        http = _FakeHttp(
            {
                "/insights/prediction-percentiles": {
                    "current_price": 200.0,
                    "forecast_future": {
                        "percentiles": [
                            {"5": 180, "50": 200, "95": 220},
                            {"5": 182, "50": 201, "95": 223},
                        ]
                    },
                }
            }
        )
        client = SynthInsightsClient(base_url="https://api.synthdata.co", api_key="x", http_client=http)
        snap = client.get_prediction_percentiles(asset="SOL")
        self.assertEqual(snap.asset, "SOL")
        self.assertEqual(snap.step_minutes, 5)
        self.assertEqual(len(snap.percentiles_by_step), 2)
        self.assertAlmostEqual(snap.percentiles_by_step[0][50.0], 200.0)


if __name__ == "__main__":
    unittest.main()
