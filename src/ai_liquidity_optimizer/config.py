from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

from ai_liquidity_optimizer.compat import dataclass


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_int(value: str | None, default: int) -> int:
    if value is None or value == "":
        return default
    return int(value)


def _parse_float(value: str | None, default: float) -> float:
    if value is None or value == "":
        return default
    return float(value)


def _require_env_value(env: Mapping[str, str], key: str) -> str:
    value = env.get(key)
    if value is None or value == "":
        raise ValueError(f"{key} is required (set it in your .env file).")
    return value


def _load_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    parsed: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        parsed[key.strip()] = value.strip().strip('"').strip("'")
    return parsed


def _merged_env(env_file: Path | None) -> Mapping[str, str]:
    # Explicit .env settings should win over inherited shell vars to keep
    # runtime behavior deterministic per project configuration.
    merged = dict(os.environ)
    if env_file:
        merged.update(_load_env_file(env_file))
    return merged


@dataclass(slots=True)
class Settings:
    ev_mode: bool
    ev_horizon_minutes: int
    synth_api_key: str | None
    synth_base_url: str
    synth_asset: str
    synth_horizon: str
    synth_days: int
    synth_limit: int
    meteora_api_base_url: str
    meteora_pool_query: str
    meteora_pool_address: str | None
    rebalance_interval_minutes: int
    min_stay_probability: float
    range_change_threshold_bps: float
    max_candidates: int
    fee_window: str
    rebalance_cost_usd: float
    tx_cost_mode: str
    rebalance_cost_lamports: int
    pool_switch_extra_cost_usd: float
    min_ev_improvement_usd: float
    pool_candidate_limit: int
    min_pool_tvl_usd: float
    top_k_ranges_per_pool: int
    ev_percentile_decay_half_life_minutes: int
    ev_concentration_gamma: float
    ev_concentration_min: float
    ev_concentration_max: float
    ev_exact_rescoring_top_n: int
    ev_capture_kappa: float
    ev_capture_min: float
    ev_capture_max: float
    ev_capture_eps: float
    ev_utilization_floor: float
    ev_min_utilization_gain: float
    ev_max_protective_ev_slip_usd: float
    ev_protective_breach_cycles: int
    ev_oor_penalty_enabled: bool
    ev_oor_penalize_hold_only: bool
    ev_oor_deadband_bps: float
    ev_oor_ref_bps: float
    ev_oor_base_penalty_fraction_15m: float
    ev_oor_max_penalty_fraction_15m: float
    ev_oor_persistence_step: float
    ev_oor_persistence_cap_cycles: int
    ev_idle_enabled: bool
    ev_idle_entry_threshold_usd: float
    ev_idle_exit_threshold_usd: float
    ev_idle_confirm_cycles: int
    ev_il_drift_alpha: float
    ev_il_oor_beta: float
    ev_il_onesided_gamma: float
    ev_il_persistence_delta: float
    ev_il_mult_min: float
    ev_il_mult_max: float
    ev_il_drift_ref_bps: float
    ev_il_drift_horizon_minutes: int
    ev_trend_stop_enabled: bool
    ev_trend_stop_oor_cycles: int
    ev_trend_stop_onesided_prob: float
    ev_action_cooldown_cycles: int
    executor: str
    meteora_liquidity_mode: str
    max_custom_weight_position_bins: int
    synth_weight_active_bin_floor_bps: int
    synth_weight_max_bin_bps_per_side: int
    synth_weight_max_single_bin: float
    synth_weight_max_top3: float
    synth_weight_objective: str
    synth_weight_odds_beta: float
    synth_weight_odds_eps: float
    deposit_sol_amount: float
    deposit_usdc_amount: float
    solana_rpc_url: str | None
    solana_private_key_b58: str | None
    state_path: Path
    log_level: str
    repo_root: Path

    def validate(self) -> None:
        if self.synth_asset.upper() != "SOL":
            raise ValueError("This MVP only supports SYNTH_ASSET=SOL.")
        if self.executor not in {"dry-run", "meteora-node"}:
            raise ValueError("EXECUTOR must be 'dry-run' or 'meteora-node'.")
        if self.meteora_liquidity_mode not in {"spot", "synth_weights"}:
            raise ValueError("METEORA_LIQUIDITY_MODE must be 'spot' or 'synth_weights'.")
        if self.max_custom_weight_position_bins <= 0:
            raise ValueError("MAX_CUSTOM_WEIGHT_POSITION_BINS must be > 0.")
        if self.synth_weight_active_bin_floor_bps < 0 or self.synth_weight_active_bin_floor_bps > 10_000:
            raise ValueError("SYNTH_WEIGHT_ACTIVE_BIN_FLOOR_BPS must be between 0 and 10000.")
        if self.synth_weight_max_bin_bps_per_side < 0 or self.synth_weight_max_bin_bps_per_side > 10_000:
            raise ValueError("SYNTH_WEIGHT_MAX_BIN_BPS_PER_SIDE must be between 0 and 10000.")
        if self.synth_weight_objective not in {"hybrid", "odds_15m_exact"}:
            raise ValueError("SYNTH_WEIGHT_OBJECTIVE must be 'hybrid' or 'odds_15m_exact'.")
        if self.synth_weight_odds_beta <= 0:
            raise ValueError("SYNTH_WEIGHT_ODDS_BETA must be > 0.")
        if self.synth_weight_odds_eps < 0:
            raise ValueError("SYNTH_WEIGHT_ODDS_EPS must be >= 0.")
        if self.synth_weight_max_single_bin <= 0 or self.synth_weight_max_single_bin > 1:
            raise ValueError("SYNTH_WEIGHT_MAX_SINGLE_BIN must be in (0, 1].")
        if self.synth_weight_max_top3 <= 0 or self.synth_weight_max_top3 > 1:
            raise ValueError("SYNTH_WEIGHT_MAX_TOP3 must be in (0, 1].")
        if self.synth_weight_max_top3 < self.synth_weight_max_single_bin:
            raise ValueError("SYNTH_WEIGHT_MAX_TOP3 must be >= SYNTH_WEIGHT_MAX_SINGLE_BIN.")
        if self.rebalance_interval_minutes <= 0:
            raise ValueError("REBALANCE_INTERVAL_MINUTES must be > 0.")
        if self.ev_horizon_minutes <= 0:
            raise ValueError("EV_HORIZON_MINUTES must be > 0.")
        if self.ev_exact_rescoring_top_n <= 0:
            raise ValueError("EV_EXACT_RESCORING_TOP_N must be > 0.")
        if self.ev_capture_kappa <= 0:
            raise ValueError("EV_CAPTURE_KAPPA must be > 0.")
        if self.ev_capture_eps <= 0:
            raise ValueError("EV_CAPTURE_EPS must be > 0.")
        if self.ev_capture_min <= 0 or self.ev_capture_max <= 0:
            raise ValueError("EV capture bounds must be > 0.")
        if self.ev_capture_min > self.ev_capture_max:
            raise ValueError("EV_CAPTURE_MIN cannot exceed EV_CAPTURE_MAX.")
        if self.ev_utilization_floor < 0 or self.ev_utilization_floor > 1:
            raise ValueError("EV_UTILIZATION_FLOOR must be in [0, 1].")
        if self.ev_min_utilization_gain < 0 or self.ev_min_utilization_gain > 1:
            raise ValueError("EV_MIN_UTILIZATION_GAIN must be in [0, 1].")
        if self.ev_protective_breach_cycles <= 0:
            raise ValueError("EV_PROTECTIVE_BREACH_CYCLES must be > 0.")
        if self.ev_oor_deadband_bps < 0:
            raise ValueError("EV_OOR_DEADBAND_BPS must be >= 0.")
        if self.ev_oor_ref_bps <= 0:
            raise ValueError("EV_OOR_REF_BPS must be > 0.")
        if self.ev_oor_base_penalty_fraction_15m < 0:
            raise ValueError("EV_OOR_BASE_PENALTY_FRACTION_15M must be >= 0.")
        if self.ev_oor_max_penalty_fraction_15m < 0:
            raise ValueError("EV_OOR_MAX_PENALTY_FRACTION_15M must be >= 0.")
        if self.ev_oor_base_penalty_fraction_15m > self.ev_oor_max_penalty_fraction_15m:
            raise ValueError("EV_OOR_BASE_PENALTY_FRACTION_15M cannot exceed EV_OOR_MAX_PENALTY_FRACTION_15M.")
        if self.ev_oor_persistence_step < 0:
            raise ValueError("EV_OOR_PERSISTENCE_STEP must be >= 0.")
        if self.ev_oor_persistence_cap_cycles < 0:
            raise ValueError("EV_OOR_PERSISTENCE_CAP_CYCLES must be >= 0.")
        if self.ev_idle_entry_threshold_usd < 0:
            raise ValueError("EV_IDLE_ENTRY_THRESHOLD_USD must be >= 0.")
        if self.ev_idle_exit_threshold_usd < 0:
            raise ValueError("EV_IDLE_EXIT_THRESHOLD_USD must be >= 0.")
        if self.ev_idle_confirm_cycles <= 0:
            raise ValueError("EV_IDLE_CONFIRM_CYCLES must be > 0.")
        if self.ev_il_drift_alpha < 0:
            raise ValueError("EV_IL_DRIFT_ALPHA must be >= 0.")
        if self.ev_il_oor_beta < 0:
            raise ValueError("EV_IL_OOR_BETA must be >= 0.")
        if self.ev_il_onesided_gamma < 0:
            raise ValueError("EV_IL_ONESIDED_GAMMA must be >= 0.")
        if self.ev_il_persistence_delta < 0:
            raise ValueError("EV_IL_PERSISTENCE_DELTA must be >= 0.")
        if self.ev_il_mult_min <= 0 or self.ev_il_mult_max <= 0:
            raise ValueError("EV_IL_MULT_MIN and EV_IL_MULT_MAX must be > 0.")
        if self.ev_il_mult_min > self.ev_il_mult_max:
            raise ValueError("EV_IL_MULT_MIN cannot exceed EV_IL_MULT_MAX.")
        if self.ev_il_drift_ref_bps <= 0:
            raise ValueError("EV_IL_DRIFT_REF_BPS must be > 0.")
        if self.ev_il_drift_horizon_minutes <= 0:
            raise ValueError("EV_IL_DRIFT_HORIZON_MINUTES must be > 0.")
        if self.ev_trend_stop_oor_cycles < 0:
            raise ValueError("EV_TREND_STOP_OOR_CYCLES must be >= 0.")
        if self.ev_trend_stop_onesided_prob < 0 or self.ev_trend_stop_onesided_prob > 1:
            raise ValueError("EV_TREND_STOP_ONESIDED_PROB must be in [0, 1].")
        if self.ev_action_cooldown_cycles < 0:
            raise ValueError("EV_ACTION_COOLDOWN_CYCLES must be >= 0.")
        if self.pool_candidate_limit <= 0:
            raise ValueError("POOL_CANDIDATE_LIMIT must be > 0.")
        if self.top_k_ranges_per_pool <= 0:
            raise ValueError("TOP_K_RANGES_PER_POOL must be > 0.")
        if self.ev_concentration_min <= 0 or self.ev_concentration_max <= 0:
            raise ValueError("EV concentration bounds must be > 0.")
        if self.ev_concentration_min > self.ev_concentration_max:
            raise ValueError("EV_CONCENTRATION_MIN cannot exceed EV_CONCENTRATION_MAX.")
        if self.tx_cost_mode not in {"fixed", "fixed_lamports", "observed"}:
            raise ValueError("TX_COST_MODE must be 'fixed', 'fixed_lamports', or 'observed' (alias).")
        if self.rebalance_cost_lamports < 0:
            raise ValueError("REBALANCE_COST_LAMPORTS must be >= 0.")
        if self.executor == "meteora-node":
            missing = []
            if not self.solana_rpc_url:
                missing.append("SOLANA_RPC_URL")
            if not self.solana_private_key_b58:
                missing.append("SOLANA_PRIVATE_KEY_B58")
            if missing:
                raise ValueError(f"Missing required env for EXECUTOR=meteora-node: {', '.join(missing)}")
        if self.synth_api_key is None:
            raise ValueError("SYNTH_API_KEY is required for live Synth forecast calls.")


def load_settings(repo_root: Path, env_file: Path | None = None) -> Settings:
    env = _merged_env(env_file)
    range_change_threshold_raw = _require_env_value(env, "RANGE_CHANGE_THRESHOLD_BPS")
    state_path = Path(env.get("STATE_PATH", "state/optimizer_state.json"))
    if not state_path.is_absolute():
        state_path = repo_root / state_path

    settings = Settings(
        ev_mode=_parse_bool(env.get("EV_MODE"), True),
        ev_horizon_minutes=_parse_int(env.get("EV_HORIZON_MINUTES"), 15),
        synth_api_key=env.get("SYNTH_API_KEY"),
        synth_base_url=env.get("SYNTH_BASE_URL", "https://api.synthdata.co").rstrip("/"),
        synth_asset=env.get("SYNTH_ASSET", "SOL").upper(),
        synth_horizon=env.get("SYNTH_HORIZON", "1h" if _parse_bool(env.get("EV_MODE"), True) else "24h"),
        synth_days=_parse_int(env.get("SYNTH_DAYS"), 30),
        synth_limit=_parse_int(env.get("SYNTH_LIMIT"), 20),
        meteora_api_base_url=env.get("METEORA_API_BASE_URL", "https://dlmm.datapi.meteora.ag").rstrip("/"),
        meteora_pool_query=env.get("METEORA_POOL_QUERY", "SOL/USDC"),
        meteora_pool_address=env.get("METEORA_POOL_ADDRESS"),
        rebalance_interval_minutes=_parse_int(env.get("REBALANCE_INTERVAL_MINUTES"), 10),
        min_stay_probability=_parse_float(env.get("MIN_STAY_PROBABILITY"), 0.02),
        range_change_threshold_bps=_parse_float(range_change_threshold_raw, 0.0),
        max_candidates=_parse_int(env.get("MAX_CANDIDATES"), 12),
        fee_window=env.get("FEE_WINDOW", "24h"),
        rebalance_cost_usd=_parse_float(env.get("REBALANCE_COST_USD"), 0.50),
        tx_cost_mode=env.get("TX_COST_MODE", "fixed_lamports").strip().lower(),
        rebalance_cost_lamports=_parse_int(
            env.get("REBALANCE_COST_LAMPORTS", env.get("REBALANCE_COST_LAMPORTS_FALLBACK")),
            15000,
        ),
        pool_switch_extra_cost_usd=_parse_float(env.get("POOL_SWITCH_EXTRA_COST_USD"), 1.00),
        min_ev_improvement_usd=_parse_float(env.get("MIN_EV_IMPROVEMENT_USD"), 0.25),
        pool_candidate_limit=_parse_int(env.get("POOL_CANDIDATE_LIMIT"), 12),
        min_pool_tvl_usd=_parse_float(env.get("MIN_POOL_TVL_USD"), 100000.0),
        top_k_ranges_per_pool=_parse_int(env.get("TOP_K_RANGES_PER_POOL"), 3),
        ev_percentile_decay_half_life_minutes=_parse_int(env.get("EV_PERCENTILE_DECAY_HALF_LIFE_MINUTES"), 15),
        ev_concentration_gamma=_parse_float(env.get("EV_CONCENTRATION_GAMMA"), 0.6),
        ev_concentration_min=_parse_float(env.get("EV_CONCENTRATION_MIN"), 0.70),
        ev_concentration_max=_parse_float(env.get("EV_CONCENTRATION_MAX"), 2.25),
        ev_exact_rescoring_top_n=_parse_int(env.get("EV_EXACT_RESCORING_TOP_N"), 6),
        ev_capture_kappa=_parse_float(env.get("EV_CAPTURE_KAPPA"), 0.7),
        ev_capture_min=_parse_float(env.get("EV_CAPTURE_MIN"), 0.60),
        ev_capture_max=_parse_float(env.get("EV_CAPTURE_MAX"), 1.05),
        ev_capture_eps=_parse_float(env.get("EV_CAPTURE_EPS"), 1e-9),
        ev_utilization_floor=_parse_float(env.get("EV_UTILIZATION_FLOOR"), 0.40),
        ev_min_utilization_gain=_parse_float(env.get("EV_MIN_UTILIZATION_GAIN"), 0.15),
        ev_max_protective_ev_slip_usd=_parse_float(env.get("EV_MAX_PROTECTIVE_EV_SLIP_USD"), 0.002),
        ev_protective_breach_cycles=_parse_int(env.get("EV_PROTECTIVE_BREACH_CYCLES"), 2),
        ev_oor_penalty_enabled=_parse_bool(env.get("EV_OOR_PENALTY_ENABLED"), True),
        ev_oor_penalize_hold_only=_parse_bool(env.get("EV_OOR_PENALIZE_HOLD_ONLY"), True),
        ev_oor_deadband_bps=_parse_float(env.get("EV_OOR_DEADBAND_BPS"), 5.0),
        ev_oor_ref_bps=_parse_float(env.get("EV_OOR_REF_BPS"), 50.0),
        ev_oor_base_penalty_fraction_15m=_parse_float(env.get("EV_OOR_BASE_PENALTY_FRACTION_15M"), 0.00025),
        ev_oor_max_penalty_fraction_15m=_parse_float(env.get("EV_OOR_MAX_PENALTY_FRACTION_15M"), 0.0015),
        ev_oor_persistence_step=_parse_float(env.get("EV_OOR_PERSISTENCE_STEP"), 0.20),
        ev_oor_persistence_cap_cycles=_parse_int(env.get("EV_OOR_PERSISTENCE_CAP_CYCLES"), 12),
        ev_idle_enabled=_parse_bool(env.get("EV_IDLE_ENABLED"), True),
        ev_idle_entry_threshold_usd=_parse_float(env.get("EV_IDLE_ENTRY_THRESHOLD_USD"), 0.0),
        ev_idle_exit_threshold_usd=_parse_float(env.get("EV_IDLE_EXIT_THRESHOLD_USD"), 0.01),
        ev_idle_confirm_cycles=_parse_int(env.get("EV_IDLE_CONFIRM_CYCLES"), 2),
        ev_il_drift_alpha=_parse_float(env.get("EV_IL_DRIFT_ALPHA"), 0.60),
        ev_il_oor_beta=_parse_float(env.get("EV_IL_OOR_BETA"), 1.00),
        ev_il_onesided_gamma=_parse_float(env.get("EV_IL_ONESIDED_GAMMA"), 0.75),
        ev_il_persistence_delta=_parse_float(env.get("EV_IL_PERSISTENCE_DELTA"), 0.50),
        ev_il_mult_min=_parse_float(env.get("EV_IL_MULT_MIN"), 1.0),
        ev_il_mult_max=_parse_float(env.get("EV_IL_MULT_MAX"), 3.0),
        ev_il_drift_ref_bps=_parse_float(env.get("EV_IL_DRIFT_REF_BPS"), 50.0),
        ev_il_drift_horizon_minutes=_parse_int(env.get("EV_IL_DRIFT_HORIZON_MINUTES"), 30),
        ev_trend_stop_enabled=_parse_bool(env.get("EV_TREND_STOP_ENABLED"), True),
        ev_trend_stop_oor_cycles=_parse_int(env.get("EV_TREND_STOP_OOR_CYCLES"), 3),
        ev_trend_stop_onesided_prob=_parse_float(env.get("EV_TREND_STOP_ONESIDED_PROB"), 0.65),
        ev_action_cooldown_cycles=_parse_int(env.get("EV_ACTION_COOLDOWN_CYCLES"), 1),
        executor=env.get("EXECUTOR", "dry-run"),
        meteora_liquidity_mode=env.get("METEORA_LIQUIDITY_MODE", "spot").strip().lower(),
        max_custom_weight_position_bins=_parse_int(env.get("MAX_CUSTOM_WEIGHT_POSITION_BINS"), 70),
        synth_weight_active_bin_floor_bps=_parse_int(env.get("SYNTH_WEIGHT_ACTIVE_BIN_FLOOR_BPS"), 1000),
        synth_weight_max_bin_bps_per_side=_parse_int(env.get("SYNTH_WEIGHT_MAX_BIN_BPS_PER_SIDE"), 7000),
        synth_weight_max_single_bin=_parse_float(env.get("SYNTH_WEIGHT_MAX_SINGLE_BIN"), 0.30),
        synth_weight_max_top3=_parse_float(env.get("SYNTH_WEIGHT_MAX_TOP3"), 0.65),
        synth_weight_objective=env.get("SYNTH_WEIGHT_OBJECTIVE", "hybrid").strip().lower(),
        synth_weight_odds_beta=_parse_float(env.get("SYNTH_WEIGHT_ODDS_BETA"), 0.9),
        synth_weight_odds_eps=_parse_float(env.get("SYNTH_WEIGHT_ODDS_EPS"), 1e-6),
        deposit_sol_amount=_parse_float(env.get("DEPOSIT_SOL_AMOUNT"), 0.10),
        deposit_usdc_amount=_parse_float(env.get("DEPOSIT_USDC_AMOUNT"), 20.0),
        solana_rpc_url=env.get("SOLANA_RPC_URL"),
        solana_private_key_b58=env.get("SOLANA_PRIVATE_KEY_B58"),
        state_path=state_path,
        log_level=env.get("LOG_LEVEL", "INFO").upper(),
        repo_root=repo_root,
    )
    settings.validate()
    return settings
