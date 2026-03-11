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
                "market_state_stage": "shadow",
                "market_regime": "range",
                "regime_confidence": 0.72,
                "horizon_agreement_score": 0.81,
                "size_multiplier": 0.66,
                "candidate_family": "fused_symmetric",
                "reentry_prob_15m": 0.63,
                "reentry_prob_1h": 0.59,
                "counterfactual_top_k": [
                    {
                        "candidate_family": "fused_symmetric",
                        "action_type": "rebalance",
                        "ev_15m_usd": 0.01,
                    }
                ],
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
            self.assertEqual(decision.get("market_regime"), "range")
            self.assertAlmostEqual(float(decision.get("size_multiplier")), 0.66, places=9)
            self.assertEqual(decision.get("candidate_family"), "fused_symmetric")
            self.assertEqual(decision.get("counterfactual_top_family"), "fused_symmetric")

            self.assertEqual(error.get("event_type"), "cycle_error")
            self.assertEqual(error.get("error_category"), "rpc_rate_limit")
            self.assertEqual(error.get("active_pool_address"), "pool1")

    def test_excludes_scale_outlier_execution_from_calibration(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal_path = Path(tmp) / "trade_journal.jsonl"
            orchestrator = self._orchestrator(journal_path)

            result = {
                "timestamp": "2026-03-11T14:10:55+00:00",
                "ev_mode": True,
                "selected_action": "rebalance",
                "selected_pool": {
                    "address": "pool1",
                    "name": "SOL-USDC",
                    "current_price": 86.0,
                },
                "chosen": {"lower_bound": 85.0, "upper_bound": 87.0, "width_pct": 2.0},
                "ev_best_candidate": {
                    "ev_15m_usd": 0.16,
                    "ev_components": {
                        "expected_fees_usd": 0.18,
                        "expected_il_usd": 0.015,
                        "range_active_occupancy_15m": 0.96,
                    },
                },
                "ev_current_hold": {"ev_15m_usd": 0.01, "ev_components": {}},
                "rebalance": {
                    "should_rebalance": True,
                    "reason": "profit_lock_rotate",
                    "should_close_to_idle": False,
                    "tx_signatures": ["sig1"],
                },
                "rebalance_gate": {"gate_mode": "policy_lifecycle"},
                "onchain_delta": {
                    "delta_total_usd_est": -4.95,
                    "minutes_elapsed": 5.0,
                },
                "lifecycle_open_position_usd": 12.0,
            }

            orchestrator._append_trade_journal_decision_entry(result=result)
            decision = json.loads(journal_path.read_text(encoding="utf-8").strip())

            self.assertFalse(bool(decision.get("calibration_enabled_row")))
            self.assertEqual(decision.get("calibration_excluded_reason"), "execution_outlier_vs_position_scale")
            self.assertIsNone(decision.get("calibration_realized_net_usd"))

    def test_excludes_model_scale_outlier_without_position_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal_path = Path(tmp) / "trade_journal.jsonl"
            orchestrator = self._orchestrator(journal_path)

            result = {
                "timestamp": "2026-03-10T22:12:57+00:00",
                "ev_mode": True,
                "selected_action": "idle",
                "selected_pool": {
                    "address": "pool1",
                    "name": "SOL-USDC",
                    "current_price": 86.0,
                },
                "chosen": {"lower_bound": 85.0, "upper_bound": 87.0, "width_pct": 2.0},
                "ev_best_candidate": {"ev_15m_usd": 0.01, "ev_components": {}},
                "ev_current_hold": {"ev_15m_usd": 0.0, "ev_components": {}},
                "ev_idle_candidate": {
                    "ev_15m_usd": -0.001,
                    "ev_components": {
                        "expected_fees_usd": 0.0,
                        "expected_il_usd": 0.0,
                        "rebalance_cost_usd": 0.0013,
                    },
                },
                "rebalance": {
                    "should_rebalance": False,
                    "reason": "idle_ev_gain_0.01",
                    "should_close_to_idle": True,
                    "tx_signatures": ["sig1"],
                },
                "rebalance_gate": {"gate_mode": "idle_exit"},
                "onchain_delta": {
                    "delta_total_usd_est": 4.90,
                    "minutes_elapsed": 5.0,
                },
            }

            orchestrator._append_trade_journal_decision_entry(result=result)
            decision = json.loads(journal_path.read_text(encoding="utf-8").strip())

            self.assertFalse(bool(decision.get("calibration_enabled_row")))
            self.assertEqual(decision.get("calibration_excluded_reason"), "execution_outlier_vs_model_scale")
            self.assertIsNone(decision.get("calibration_realized_net_usd"))


if __name__ == "__main__":
    unittest.main()
