from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from ai_liquidity_optimizer.models import ActivePositionState, BotState
from ai_liquidity_optimizer.orchestrator import OptimizerOrchestrator


class _StubStateStore:
    def load(self):  # pragma: no cover - not used
        return BotState()

    def save(self, state):  # pragma: no cover - not used
        return None


class TradeJournalTests(unittest.TestCase):
    def _orchestrator(self, journal_path: Path) -> OptimizerOrchestrator:
        settings = SimpleNamespace(
            trade_journal_enabled=True,
            trade_journal_path=journal_path,
            ev_mode=True,
        )
        return OptimizerOrchestrator(
            settings=settings,
            synth_client=SimpleNamespace(),
            meteora_client=SimpleNamespace(),
            scorer=SimpleNamespace(),
            ev_scorer=SimpleNamespace(),
            executor=SimpleNamespace(),
            state_store=_StubStateStore(),
        )

    def test_appends_decision_and_error_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal_path = Path(tmp) / "trade_journal.jsonl"
            orchestrator = self._orchestrator(journal_path)

            result = {
                "timestamp": "2026-03-02T02:00:00+00:00",
                "ev_mode": True,
                "selected_action": "rebalance",
                "selected_pool": {
                    "address": "pool1",
                    "name": "SOL-USDC",
                    "current_price": 84.0,
                },
                "chosen": {"lower_bound": 83.5, "upper_bound": 84.5, "width_pct": 1.2},
                "ev_best_candidate": {
                    "ev_15m_usd": 0.01,
                    "ev_components": {
                        "expected_fees_usd": 0.02,
                        "expected_il_usd": 0.008,
                        "range_active_occupancy_15m": 0.9,
                    },
                },
                "ev_current_hold": {
                    "ev_15m_usd": 0.004,
                    "ev_components": {
                        "utilization_ratio": 0.11,
                        "out_of_range_bps": 0.0,
                        "out_of_range_cycles": 0,
                    },
                },
                "ev_idle_candidate": {"ev_15m_usd": -0.001},
                "ev_delta_usd": 0.006,
                "lifecycle_pnl_usd": 0.25,
                "lifecycle_pnl_pct": 0.02,
                "lifecycle_open_position_usd": 12.0,
                "policy_profit_triggered": True,
                "policy_loss_recovery_hold_triggered": False,
                "policy_loss_trend_cut_triggered": False,
                "policy_recovery_prob": 0.6,
                "policy_trend_prob": 0.4,
                "policy_loss_exit_breach_count": 0,
                "policy_selected_override": "profit_lock",
                "rebalance_gate": {"gate_mode": "ev_only"},
                "rebalance": {
                    "should_rebalance": True,
                    "reason": "range_ev_gain_0.0060",
                    "should_close_to_idle": False,
                    "tx_signatures": ["sig1"],
                },
                "active_position": {"lower_price": 83.2, "upper_price": 84.2},
            }
            orchestrator._append_trade_journal_decision_entry(result=result)

            state = BotState(
                active_position=ActivePositionState(
                    pool_address="pool1",
                    lower_price=83.2,
                    upper_price=84.2,
                    width_pct=1.2,
                    executor="dry-run",
                )
            )
            orchestrator._append_trade_journal_error_entry(exc=RuntimeError("429 Too Many Requests"), state=state)

            lines = journal_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 2)
            decision = json.loads(lines[0])
            error = json.loads(lines[1])

            self.assertEqual(decision.get("event_type"), "decision")
            self.assertEqual(decision.get("selected_action"), "rebalance")
            self.assertEqual(decision.get("execution_status"), "success")
            self.assertEqual(decision.get("execution_tx_count"), 1)
            self.assertEqual(decision.get("pool_address"), "pool1")
            self.assertAlmostEqual(float(decision.get("lifecycle_pnl_usd")), 0.25, places=9)
            self.assertEqual(decision.get("policy_selected_override"), "profit_lock")
            self.assertTrue(bool(decision.get("policy_profit_triggered")))

            self.assertEqual(error.get("event_type"), "cycle_error")
            self.assertEqual(error.get("error_category"), "rpc_rate_limit")
            self.assertEqual(error.get("active_pool_address"), "pool1")


if __name__ == "__main__":
    unittest.main()
