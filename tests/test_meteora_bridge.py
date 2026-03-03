from __future__ import annotations

import unittest
from pathlib import Path

from ai_liquidity_optimizer.execution.meteora_node_bridge import MeteoraNodeBridgeExecutor
from ai_liquidity_optimizer.models import ActivePositionState


class _StubBridge(MeteoraNodeBridgeExecutor):
    def __init__(self, response: dict):
        super().__init__(
            repo_root=Path("."),
            rpc_url="https://example-rpc.invalid",
            private_key_b58="stub",
        )
        self._response = response
        self.last_payload = None

    def _run_node(self, payload: dict) -> dict:
        self.last_payload = payload
        return self._response


class MeteoraBridgeTests(unittest.TestCase):
    def test_quote_target_bins_parses_executor_response(self):
        bridge = _StubBridge(
            {
                "ok": True,
                "pool_address": "Pool111",
                "active_bin_id": 123,
                "sdk_price_orientation": "same_as_api",
                "pool_price_orientation": "USDC_per_SOL",
                "lower_for_sdk": 80.0,
                "upper_for_sdk": 90.0,
                "bins": [
                    {"bin_id": 120, "price_sdk": 81.0, "price_sol_usdc": 81.0},
                    {"bin_id": 121, "price_sdk": 82.0, "price_sol_usdc": 82.0},
                ],
            }
        )
        quote = bridge.quote_target_bins(
            pool_address="Pool111",
            symbol_x="SOL",
            symbol_y="USDC",
            api_current_price=85.0,
            target_lower_price=80.0,
            target_upper_price=90.0,
        )
        self.assertIsNotNone(quote)
        assert quote is not None
        self.assertEqual(quote.pool_address, "Pool111")
        self.assertEqual(quote.active_bin_id, 123)
        self.assertEqual(quote.bin_ids, [120, 121])
        self.assertEqual(quote.bin_prices_sol_usdc, [81.0, 82.0])
        self.assertEqual(quote.bin_prices_sdk, [81.0, 82.0])

    def test_quote_target_bins_returns_none_for_invalid_bin_payload(self):
        bridge = _StubBridge({"ok": True, "bins": [{"bin_id": "x", "price_sol_usdc": "bad"}]})
        quote = bridge.quote_target_bins(
            pool_address="Pool111",
            symbol_x="SOL",
            symbol_y="USDC",
            api_current_price=85.0,
            target_lower_price=80.0,
            target_upper_price=90.0,
        )
        self.assertIsNone(quote)
        self.assertIsNotNone(bridge.last_quote_failure_reason())

    def test_quote_target_bins_fills_missing_prices_from_range(self):
        bridge = _StubBridge(
            {
                "ok": True,
                "pool_address": "Pool111",
                "active_bin_id": 123,
                "sdk_price_orientation": "same_as_api",
                "pool_price_orientation": "USDC_per_SOL",
                "lower_for_sdk": 80.0,
                "upper_for_sdk": 90.0,
                "bins": [
                    {"bin_id": 120, "price_sdk": None, "price_sol_usdc": None},
                    {"bin_id": 121, "price_sdk": None, "price_sol_usdc": None},
                    {"bin_id": 122, "price_sdk": None, "price_sol_usdc": None},
                ],
            }
        )
        quote = bridge.quote_target_bins(
            pool_address="Pool111",
            symbol_x="SOL",
            symbol_y="USDC",
            api_current_price=85.0,
            target_lower_price=80.0,
            target_upper_price=90.0,
        )
        self.assertIsNotNone(quote)
        assert quote is not None
        self.assertEqual(quote.bin_ids, [120, 121, 122])
        self.assertEqual(len(quote.bin_prices_sol_usdc), 3)
        self.assertGreater(quote.bin_prices_sol_usdc[0], 0.0)
        self.assertGreater(quote.bin_prices_sol_usdc[1], quote.bin_prices_sol_usdc[0])
        self.assertGreater(quote.bin_prices_sol_usdc[2], quote.bin_prices_sol_usdc[1])

    def test_quote_target_bins_uses_pool_orientation_price_fallback(self):
        bridge = _StubBridge(
            {
                "ok": True,
                "pool_address": "Pool111",
                "active_bin_id": 200,
                "sdk_price_orientation": "same_as_api",
                "pool_price_orientation": "USDC_per_SOL",
                "lower_for_sdk": 80.0,
                "upper_for_sdk": 90.0,
                "bins": [
                    {"bin_id": 199, "price_pool_orientation": 81.5, "price_sol_usdc": None},
                    {"bin_id": 200, "price_pool_orientation": 82.5, "price_sol_usdc": None},
                ],
            }
        )
        quote = bridge.quote_target_bins(
            pool_address="Pool111",
            symbol_x="SOL",
            symbol_y="USDC",
            api_current_price=82.0,
            target_lower_price=80.0,
            target_upper_price=90.0,
        )
        self.assertIsNotNone(quote)
        assert quote is not None
        self.assertEqual(quote.bin_ids, [199, 200])
        self.assertEqual(quote.bin_prices_sol_usdc, [81.5, 82.5])

    def test_close_position_calls_close_command(self):
        bridge = _StubBridge(
            {
                "ok": True,
                "changed": True,
                "tx_signatures": ["sig1"],
            }
        )
        result = bridge.close_position(
            ActivePositionState(
                pool_address="Pool111",
                lower_price=80.0,
                upper_price=90.0,
                width_pct=11.7,
                executor="meteora-node",
                position_pubkey="Pos111",
                lower_bin_id=1,
                upper_bin_id=2,
            )
        )
        self.assertTrue(result.changed)
        self.assertIsNone(result.active_position)
        self.assertEqual(result.tx_signatures, ["sig1"])
        self.assertIsNotNone(bridge.last_payload)
        self.assertEqual(bridge.last_payload.get("command"), "close-position")

    def test_get_onchain_snapshot_normalizes_fields(self):
        bridge = _StubBridge(
            {
                "ok": True,
                "snapshot_at": "2026-03-02T13:00:00+00:00",
                "wallet_pubkey": "Wallet111",
                "sol_balance": 0.1234,
                "usdc_balance": 9.87,
                "native_sol_balance": 0.0234,
                "wallet_sol_token_balance": 0.1,
                "wallet_sol_total_balance": 0.1234,
                "wallet_usdc_total_balance": 9.87,
                "wallet_total_usd_est": 19.11,
                "position_total_usd_est": 1.10,
                "total_usd_est": 20.21,
                "spot_price_sol_usdc": 84.0,
                "active_position_exists": True,
                "position_snapshot": {"position_pubkey": "Pos111", "total_usd_est": 1.10},
                "pool_balances": {"wallet_sol_total_ui": 0.1234, "wallet_usdc_ui": 9.87},
            }
        )
        snapshot = bridge.get_onchain_snapshot(pool=None, active_position=None)
        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(snapshot.get("wallet_pubkey"), "Wallet111")
        self.assertAlmostEqual(float(snapshot.get("sol_balance")), 0.1234, places=6)
        self.assertAlmostEqual(float(snapshot.get("usdc_balance")), 9.87, places=6)
        self.assertAlmostEqual(float(snapshot.get("total_usd_est")), 20.21, places=6)
        self.assertAlmostEqual(float(snapshot.get("wallet_total_usd_est")), 19.11, places=6)
        self.assertAlmostEqual(float(snapshot.get("position_total_usd_est")), 1.10, places=6)
        self.assertEqual(snapshot.get("position_snapshot", {}).get("position_pubkey"), "Pos111")
        self.assertEqual(snapshot.get("active_position_exists"), True)
        self.assertIsNotNone(bridge.last_payload)
        self.assertEqual(bridge.last_payload.get("command"), "wallet-snapshot")

    def test_get_onchain_snapshot_reconstructs_total_when_missing(self):
        bridge = _StubBridge(
            {
                "ok": True,
                "snapshot_at": "2026-03-02T13:00:00+00:00",
                "wallet_pubkey": "Wallet111",
                "wallet_total_usd_est": 11.5,
                "position_snapshot": {"total_usd_est": 2.25},
                "spot_price_sol_usdc": 84.0,
            }
        )
        snapshot = bridge.get_onchain_snapshot(pool=None, active_position=None)
        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertAlmostEqual(float(snapshot.get("wallet_total_usd_est")), 11.5, places=6)
        self.assertAlmostEqual(float(snapshot.get("position_total_usd_est")), 2.25, places=6)
        self.assertAlmostEqual(float(snapshot.get("total_usd_est")), 13.75, places=6)


if __name__ == "__main__":
    unittest.main()
