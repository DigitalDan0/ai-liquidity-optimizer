from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_liquidity_optimizer.config import load_settings


class SettingsTests(unittest.TestCase):
    def _write_env(self, root: Path, lines: list[str]) -> Path:
        env_file = root / ".env.test"
        env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return env_file

    def test_loads_valid_synth_weight_objective_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env_file = self._write_env(
                root,
                [
                    "SYNTH_API_KEY=test-key",
                    "RANGE_CHANGE_THRESHOLD_BPS=10",
                    "SYNTH_WEIGHT_OBJECTIVE=odds_15m_exact",
                    "SYNTH_WEIGHT_ODDS_BETA=0.85",
                    "SYNTH_WEIGHT_ODDS_EPS=0.000001",
                ],
            )
            with patch.dict(os.environ, {}, clear=True):
                settings = load_settings(repo_root=root, env_file=env_file)
        self.assertEqual(settings.synth_weight_objective, "odds_15m_exact")
        self.assertAlmostEqual(settings.synth_weight_odds_beta, 0.85, places=9)
        self.assertAlmostEqual(settings.synth_weight_odds_eps, 1e-6, places=12)
        self.assertEqual(settings.ev_exact_rescoring_top_n, 6)
        self.assertAlmostEqual(settings.ev_capture_min, 0.60, places=9)
        self.assertAlmostEqual(settings.synth_weight_max_single_bin, 0.30, places=9)
        self.assertTrue(settings.ev_oor_penalty_enabled)
        self.assertTrue(settings.ev_oor_penalize_hold_only)
        self.assertAlmostEqual(settings.ev_oor_deadband_bps, 5.0, places=9)
        self.assertAlmostEqual(settings.ev_oor_ref_bps, 50.0, places=9)
        self.assertAlmostEqual(settings.ev_oor_base_penalty_fraction_15m, 0.00025, places=12)
        self.assertAlmostEqual(settings.ev_oor_max_penalty_fraction_15m, 0.0015, places=12)
        self.assertAlmostEqual(settings.ev_oor_persistence_step, 0.20, places=9)
        self.assertEqual(settings.ev_oor_persistence_cap_cycles, 12)
        self.assertTrue(settings.ev_oor_grace_enabled)
        self.assertEqual(settings.ev_oor_grace_cycles, 2)
        self.assertAlmostEqual(settings.ev_oor_reentry_min_prob, 0.60, places=9)
        self.assertAlmostEqual(settings.ev_oor_loss_trigger_pct, 0.0, places=9)
        self.assertTrue(settings.ev_idle_enabled)
        self.assertAlmostEqual(settings.ev_idle_entry_threshold_usd, 0.0, places=9)
        self.assertAlmostEqual(settings.ev_idle_exit_threshold_usd, 0.01, places=9)
        self.assertEqual(settings.ev_idle_confirm_cycles, 2)
        self.assertAlmostEqual(settings.ev_il_drift_alpha, 0.60, places=9)
        self.assertAlmostEqual(settings.ev_il_oor_beta, 1.00, places=9)
        self.assertAlmostEqual(settings.ev_il_onesided_gamma, 0.75, places=9)
        self.assertAlmostEqual(settings.ev_il_persistence_delta, 0.50, places=9)
        self.assertAlmostEqual(settings.ev_il_mult_min, 1.0, places=9)
        self.assertAlmostEqual(settings.ev_il_mult_max, 3.0, places=9)
        self.assertAlmostEqual(settings.ev_il_drift_ref_bps, 50.0, places=9)
        self.assertEqual(settings.ev_il_drift_horizon_minutes, 30)
        self.assertTrue(settings.ev_trend_stop_enabled)
        self.assertEqual(settings.ev_trend_stop_oor_cycles, 3)
        self.assertAlmostEqual(settings.ev_trend_stop_onesided_prob, 0.65, places=9)
        self.assertEqual(settings.ev_action_cooldown_cycles, 1)
        self.assertTrue(settings.ev_policy_lifecycle_enabled)
        self.assertAlmostEqual(settings.ev_profit_take_pct, 0.015, places=9)
        self.assertAlmostEqual(settings.ev_profit_rebalance_max_slip_usd, 0.002, places=9)
        self.assertAlmostEqual(settings.ev_loss_recovery_trigger_pct, 0.020, places=9)
        self.assertAlmostEqual(settings.ev_high_il_15m_fraction, 0.0006, places=9)
        self.assertAlmostEqual(settings.ev_recovery_min_prob, 0.55, places=9)
        self.assertAlmostEqual(settings.ev_trend_continue_exit_prob, 0.72, places=9)
        self.assertEqual(settings.ev_trend_exit_persistence_cycles, 2)
        self.assertAlmostEqual(settings.ev_idle_preference_edge_usd, 0.01, places=9)
        self.assertTrue(settings.ev_realism_enabled)
        self.assertEqual(settings.ev_realism_window_hours, 168)
        self.assertEqual(settings.ev_realism_min_samples, 30)
        self.assertEqual(settings.synth_fusion_horizons, ["15m", "1h", "4h", "24h"])
        self.assertEqual(settings.synth_market_state_stage, "shadow")
        self.assertAlmostEqual(settings.synth_regime_range_max_center_drift_ratio, 0.25, places=9)
        self.assertAlmostEqual(settings.synth_regime_trend_min_center_drift_ratio, 0.40, places=9)
        self.assertAlmostEqual(settings.synth_regime_trend_min_onesided_prob, 0.45, places=9)
        self.assertAlmostEqual(settings.synth_regime_min_agreement_score, 0.60, places=9)
        self.assertAlmostEqual(settings.synth_regime_uncertain_width_expansion, 0.35, places=9)
        self.assertAlmostEqual(settings.synth_entry_conflict_threshold, 0.45, places=9)
        self.assertAlmostEqual(settings.synth_size_low_confidence, 0.35, places=9)
        self.assertAlmostEqual(settings.synth_size_medium_confidence, 0.55, places=9)
        self.assertAlmostEqual(settings.synth_size_full_confidence, 0.75, places=9)
        self.assertEqual(settings.ev_realism_regime_min_samples, 8)
        self.assertAlmostEqual(settings.ev_realism_regime_blend_max, 0.60, places=9)
        self.assertAlmostEqual(settings.ev_fee_realism_prior, 0.35, places=9)
        self.assertAlmostEqual(settings.ev_fee_realism_min, 0.10, places=9)
        self.assertAlmostEqual(settings.ev_fee_realism_max, 0.85, places=9)
        self.assertAlmostEqual(settings.ev_rebalance_drag_prior_usd, 0.010, places=9)
        self.assertAlmostEqual(settings.ev_rebalance_drag_min_usd, 0.000, places=9)
        self.assertAlmostEqual(settings.ev_rebalance_drag_max_usd, 0.050, places=9)
        self.assertAlmostEqual(settings.ev_uncertainty_k, 1.00, places=9)
        self.assertAlmostEqual(settings.ev_calibration_max_realized_position_fraction, 0.30, places=9)
        self.assertAlmostEqual(settings.ev_calibration_max_error_position_fraction, 0.30, places=9)
        self.assertAlmostEqual(settings.ev_calibration_max_error_model_scale_multiple, 25.0, places=9)
        self.assertTrue(settings.ev_dynamic_gate_enabled)
        self.assertAlmostEqual(settings.ev_dynamic_gate_base_margin_usd, 0.003, places=9)
        self.assertAlmostEqual(settings.ev_dynamic_gate_sigma_mult, 1.00, places=9)
        self.assertTrue(settings.ev_adjusted_ev_require_positive)
        self.assertFalse(settings.ev_realism_shadow_mode)
        self.assertTrue(settings.trade_journal_enabled)
        self.assertEqual(
            settings.trade_journal_path,
            settings.state_path.parent / "trade_journal.jsonl",
        )

    def test_rejects_invalid_synth_weight_objective(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env_file = self._write_env(
                root,
                [
                    "SYNTH_API_KEY=test-key",
                    "RANGE_CHANGE_THRESHOLD_BPS=10",
                    "SYNTH_WEIGHT_OBJECTIVE=not-a-mode",
                ],
            )
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(ValueError):
                    load_settings(repo_root=root, env_file=env_file)

    def test_rejects_non_positive_odds_beta(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env_file = self._write_env(
                root,
                [
                    "SYNTH_API_KEY=test-key",
                    "RANGE_CHANGE_THRESHOLD_BPS=10",
                    "SYNTH_WEIGHT_ODDS_BETA=0",
                ],
            )
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(ValueError):
                    load_settings(repo_root=root, env_file=env_file)

    def test_rejects_invalid_weight_concentration_caps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env_file = self._write_env(
                root,
                [
                    "SYNTH_API_KEY=test-key",
                    "RANGE_CHANGE_THRESHOLD_BPS=10",
                    "SYNTH_WEIGHT_MAX_SINGLE_BIN=0.8",
                    "SYNTH_WEIGHT_MAX_TOP3=0.5",
                ],
            )
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(ValueError):
                    load_settings(repo_root=root, env_file=env_file)

    def test_rejects_non_positive_rescoring_top_n(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env_file = self._write_env(
                root,
                [
                    "SYNTH_API_KEY=test-key",
                    "RANGE_CHANGE_THRESHOLD_BPS=10",
                    "EV_EXACT_RESCORING_TOP_N=0",
                ],
            )
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(ValueError):
                    load_settings(repo_root=root, env_file=env_file)

    def test_rejects_non_positive_oor_ref_bps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env_file = self._write_env(
                root,
                [
                    "SYNTH_API_KEY=test-key",
                    "RANGE_CHANGE_THRESHOLD_BPS=10",
                    "EV_OOR_REF_BPS=0",
                ],
            )
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(ValueError):
                    load_settings(repo_root=root, env_file=env_file)

    def test_rejects_invalid_oor_reentry_prob(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env_file = self._write_env(
                root,
                [
                    "SYNTH_API_KEY=test-key",
                    "RANGE_CHANGE_THRESHOLD_BPS=10",
                    "EV_OOR_REENTRY_MIN_PROB=1.5",
                ],
            )
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(ValueError):
                    load_settings(repo_root=root, env_file=env_file)

    def test_rejects_invalid_profit_take_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env_file = self._write_env(
                root,
                [
                    "SYNTH_API_KEY=test-key",
                    "RANGE_CHANGE_THRESHOLD_BPS=10",
                    "EV_PROFIT_TAKE_PCT=1.1",
                ],
            )
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(ValueError):
                    load_settings(repo_root=root, env_file=env_file)

    def test_rejects_invalid_il_multiplier_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env_file = self._write_env(
                root,
                [
                    "SYNTH_API_KEY=test-key",
                    "RANGE_CHANGE_THRESHOLD_BPS=10",
                    "EV_IL_MULT_MIN=2.0",
                    "EV_IL_MULT_MAX=1.5",
                ],
            )
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(ValueError):
                    load_settings(repo_root=root, env_file=env_file)

    def test_rejects_invalid_realism_multiplier_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env_file = self._write_env(
                root,
                [
                    "SYNTH_API_KEY=test-key",
                    "RANGE_CHANGE_THRESHOLD_BPS=10",
                    "EV_FEE_REALISM_MIN=0.7",
                    "EV_FEE_REALISM_MAX=0.5",
                ],
            )
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(ValueError):
                    load_settings(repo_root=root, env_file=env_file)

    def test_rejects_invalid_calibration_scale_thresholds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env_file = self._write_env(
                root,
                [
                    "SYNTH_API_KEY=test-key",
                    "RANGE_CHANGE_THRESHOLD_BPS=10",
                    "EV_CALIBRATION_MAX_ERROR_MODEL_SCALE_MULTIPLE=0",
                ],
            )
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(ValueError):
                    load_settings(repo_root=root, env_file=env_file)

    def test_rejects_invalid_synth_market_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env_file = self._write_env(
                root,
                [
                    "SYNTH_API_KEY=test-key",
                    "RANGE_CHANGE_THRESHOLD_BPS=10",
                    "SYNTH_MARKET_STATE_STAGE=bad-stage",
                ],
            )
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(ValueError):
                    load_settings(repo_root=root, env_file=env_file)

    def test_env_file_overrides_shell_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env_file = self._write_env(
                root,
                [
                    "SYNTH_API_KEY=from-env-file",
                    "RANGE_CHANGE_THRESHOLD_BPS=10",
                ],
            )
            with patch.dict(
                os.environ,
                {
                    "SYNTH_API_KEY": "from-shell",
                    "RANGE_CHANGE_THRESHOLD_BPS": "25",
                },
                clear=True,
            ):
                settings = load_settings(repo_root=root, env_file=env_file)
        self.assertEqual(settings.synth_api_key, "from-env-file")
        self.assertAlmostEqual(settings.range_change_threshold_bps, 10.0, places=9)

    def test_trade_journal_path_can_be_overridden(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env_file = self._write_env(
                root,
                [
                    "SYNTH_API_KEY=test-key",
                    "RANGE_CHANGE_THRESHOLD_BPS=10",
                    "TRADE_JOURNAL_ENABLED=true",
                    "TRADE_JOURNAL_PATH=state/custom_journal.jsonl",
                ],
            )
            with patch.dict(os.environ, {}, clear=True):
                settings = load_settings(repo_root=root, env_file=env_file)
        self.assertEqual(settings.trade_journal_path, root / "state" / "custom_journal.jsonl")


if __name__ == "__main__":
    unittest.main()
