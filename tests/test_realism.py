from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ai_liquidity_optimizer.strategy.realism import (
    blend_calibration_snapshots,
    build_calibration_snapshot_from_journal,
    default_calibration_snapshot,
)


def _write_journal(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def _decision_row(
    *,
    ts: datetime,
    expected_net: float,
    realized_net: float,
    expected_fees: float,
    expected_il: float,
    expected_cost: float,
    selected_action: str = "hold",
    rebalance_should: bool = False,
    execution_status: str = "success",
    market_regime: str | None = None,
) -> dict:
    return {
        "event_type": "decision",
        "decision_timestamp": ts.isoformat(),
        "execution_status": execution_status,
        "selected_expected_net_usd": expected_net,
        "selected_raw_expected_net_usd": expected_net,
        "calibration_model_net_usd": expected_net,
        "calibration_enabled_row": execution_status == "success",
        "calibration_realized_net_usd": (realized_net if execution_status == "success" else None),
        "execution_realized_delta_total_usd_est": (realized_net if execution_status == "success" else None),
        "onchain_delta_total_usd_est": realized_net,
        "selected_expected_fees_usd": expected_fees,
        "selected_expected_il_usd": expected_il,
        "selected_expected_total_cost_usd": expected_cost,
        "selected_action": selected_action,
        "rebalance_should": rebalance_should,
        "market_regime": market_regime,
    }


class RealismCalibrationTests(unittest.TestCase):
    def test_uses_priors_when_no_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "journal.jsonl"
            snap = build_calibration_snapshot_from_journal(
                journal_path=path,
                window_hours=168,
                min_samples=30,
                fee_realism_prior=0.35,
                fee_realism_min=0.10,
                fee_realism_max=0.85,
                rebalance_drag_prior_usd=0.010,
                rebalance_drag_min_usd=0.000,
                rebalance_drag_max_usd=0.050,
            )
        self.assertEqual(snap.mode, "warmup")
        self.assertEqual(snap.sample_count, 0)
        self.assertAlmostEqual(snap.fee_realism_multiplier, 0.35, places=9)
        self.assertAlmostEqual(snap.rebalance_drag_usd, 0.010, places=9)

    def test_computes_ready_snapshot_with_sufficient_samples(self):
        now = datetime.now(timezone.utc)
        rows = []
        for i in range(40):
            ts = now - timedelta(minutes=40 - i)
            rows.append(
                _decision_row(
                    ts=ts,
                    expected_net=0.03,
                    realized_net=0.01,
                    expected_fees=0.05,
                    expected_il=0.015,
                    expected_cost=0.005,
                    selected_action="rebalance" if i % 2 == 0 else "hold",
                    rebalance_should=(i % 2 == 0),
                )
            )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "journal.jsonl"
            _write_journal(path, rows)
            snap = build_calibration_snapshot_from_journal(
                journal_path=path,
                window_hours=168,
                min_samples=30,
                fee_realism_prior=0.35,
                fee_realism_min=0.10,
                fee_realism_max=0.85,
                rebalance_drag_prior_usd=0.010,
                rebalance_drag_min_usd=0.000,
                rebalance_drag_max_usd=0.050,
                now=now,
            )
        self.assertEqual(snap.mode, "ready")
        self.assertGreaterEqual(snap.sample_count, 30)
        self.assertGreater(snap.model_rmse_usd, 0.0)
        self.assertGreaterEqual(snap.fee_realism_multiplier, 0.10)
        self.assertLessEqual(snap.fee_realism_multiplier, 0.85)
        self.assertGreaterEqual(snap.rebalance_drag_usd, 0.0)
        self.assertLessEqual(snap.rebalance_drag_usd, 0.050)

    def test_ignores_non_success_execution_rows(self):
        now = datetime.now(timezone.utc)
        rows = []
        for i in range(30):
            ts = now - timedelta(minutes=31 - i)
            rows.append(
                _decision_row(
                    ts=ts,
                    expected_net=0.02,
                    realized_net=0.01,
                    expected_fees=0.04,
                    expected_il=0.01,
                    expected_cost=0.003,
                    execution_status="blocked_before_execute",
                )
            )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "journal.jsonl"
            _write_journal(path, rows)
            snap = build_calibration_snapshot_from_journal(
                journal_path=path,
                window_hours=168,
                min_samples=30,
                fee_realism_prior=0.35,
                fee_realism_min=0.10,
                fee_realism_max=0.85,
                rebalance_drag_prior_usd=0.010,
                rebalance_drag_min_usd=0.000,
                rebalance_drag_max_usd=0.050,
                now=now,
            )
        self.assertEqual(snap.sample_count, 0)
        self.assertEqual(snap.mode, "warmup")

    def test_winsorization_prevents_outlier_domination(self):
        now = datetime.now(timezone.utc)
        rows = []
        for i in range(29):
            ts = now - timedelta(minutes=30 - i)
            rows.append(
                _decision_row(
                    ts=ts,
                    expected_net=0.02,
                    realized_net=0.015,
                    expected_fees=0.04,
                    expected_il=0.01,
                    expected_cost=0.003,
                )
            )
        rows.append(
            _decision_row(
                ts=now - timedelta(minutes=1),
                expected_net=0.02,
                realized_net=-5.0,
                expected_fees=0.04,
                expected_il=0.01,
                expected_cost=0.003,
                selected_action="rebalance",
                rebalance_should=True,
            )
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "journal.jsonl"
            _write_journal(path, rows)
            snap = build_calibration_snapshot_from_journal(
                journal_path=path,
                window_hours=168,
                min_samples=30,
                fee_realism_prior=0.35,
                fee_realism_min=0.10,
                fee_realism_max=0.85,
                rebalance_drag_prior_usd=0.010,
                rebalance_drag_min_usd=0.000,
                rebalance_drag_max_usd=0.050,
                now=now,
            )
        # Outlier should not blow past configured safety caps.
        self.assertLessEqual(snap.rebalance_drag_usd, 0.050)
        self.assertLessEqual(snap.fee_realism_multiplier, 0.85)

    def test_filters_rows_by_regime_label(self):
        now = datetime.now(timezone.utc)
        rows = []
        for i in range(8):
            rows.append(
                _decision_row(
                    ts=now - timedelta(minutes=20 - i),
                    expected_net=0.03,
                    realized_net=0.01,
                    expected_fees=0.05,
                    expected_il=0.01,
                    expected_cost=0.003,
                    market_regime="range",
                )
            )
        for i in range(4):
            rows.append(
                _decision_row(
                    ts=now - timedelta(minutes=10 - i),
                    expected_net=0.04,
                    realized_net=-0.01,
                    expected_fees=0.06,
                    expected_il=0.01,
                    expected_cost=0.003,
                    selected_action="rebalance",
                    rebalance_should=True,
                    market_regime="trend_up",
                )
            )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "journal.jsonl"
            _write_journal(path, rows)
            snap = build_calibration_snapshot_from_journal(
                journal_path=path,
                window_hours=168,
                min_samples=4,
                fee_realism_prior=0.35,
                fee_realism_min=0.10,
                fee_realism_max=0.85,
                rebalance_drag_prior_usd=0.010,
                rebalance_drag_min_usd=0.000,
                rebalance_drag_max_usd=0.050,
                regime_label="trend_up",
                now=now,
            )
        self.assertEqual(snap.sample_count, 4)
        self.assertEqual(snap.regime_label, "trend_up")

    def test_blends_global_and_regime_snapshots_conservatively_during_warmup(self):
        prior = default_calibration_snapshot(
            fee_realism_prior=0.35,
            fee_realism_min=0.10,
            fee_realism_max=0.85,
            rebalance_drag_prior_usd=0.010,
            rebalance_drag_min_usd=0.000,
            rebalance_drag_max_usd=0.050,
        )
        global_snapshot = build_calibration_snapshot_from_journal(
            journal_path=Path("/tmp/does-not-exist.jsonl"),
            window_hours=168,
            min_samples=30,
            fee_realism_prior=0.35,
            fee_realism_min=0.10,
            fee_realism_max=0.85,
            rebalance_drag_prior_usd=0.020,
            rebalance_drag_min_usd=0.000,
            rebalance_drag_max_usd=0.050,
        )
        global_snapshot.fee_realism_multiplier = 0.25
        global_snapshot.rebalance_drag_usd = 0.018
        global_snapshot.model_rmse_usd = 0.030
        global_snapshot.sample_count = 40
        global_snapshot.mode = "ready"

        regime_snapshot = default_calibration_snapshot(
            fee_realism_prior=0.35,
            fee_realism_min=0.10,
            fee_realism_max=0.85,
            rebalance_drag_prior_usd=0.010,
            rebalance_drag_min_usd=0.000,
            rebalance_drag_max_usd=0.050,
        )
        regime_snapshot.fee_realism_multiplier = 0.15
        regime_snapshot.rebalance_drag_usd = 0.040
        regime_snapshot.model_rmse_usd = 0.050
        regime_snapshot.sample_count = 3
        regime_snapshot.regime_label = "trend_up"

        blended = blend_calibration_snapshots(
            prior_snapshot=prior,
            global_snapshot=global_snapshot,
            regime_snapshot=regime_snapshot,
            regime_min_samples=8,
            regime_blend_max=0.60,
        )

        self.assertEqual(blended.mode, "regime_warmup")
        self.assertEqual(blended.regime_label, "trend_up")
        self.assertEqual(blended.global_sample_count, 40)
        self.assertAlmostEqual(blended.rebalance_drag_usd, 0.012, places=9)


if __name__ == "__main__":
    unittest.main()
