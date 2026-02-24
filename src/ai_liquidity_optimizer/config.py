from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


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
    merged = {}
    if env_file:
        merged.update(_load_env_file(env_file))
    for k, v in os.environ.items():
        merged[k] = v
    return merged


@dataclass(slots=True)
class Settings:
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
    executor: str
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
        if self.rebalance_interval_minutes <= 0:
            raise ValueError("REBALANCE_INTERVAL_MINUTES must be > 0.")
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
    state_path = Path(env.get("STATE_PATH", "state/optimizer_state.json"))
    if not state_path.is_absolute():
        state_path = repo_root / state_path

    settings = Settings(
        synth_api_key=env.get("SYNTH_API_KEY"),
        synth_base_url=env.get("SYNTH_BASE_URL", "https://api.synthdata.co").rstrip("/"),
        synth_asset=env.get("SYNTH_ASSET", "SOL").upper(),
        synth_horizon=env.get("SYNTH_HORIZON", "24h"),
        synth_days=_parse_int(env.get("SYNTH_DAYS"), 30),
        synth_limit=_parse_int(env.get("SYNTH_LIMIT"), 20),
        meteora_api_base_url=env.get("METEORA_API_BASE_URL", "https://dlmm.datapi.meteora.ag").rstrip("/"),
        meteora_pool_query=env.get("METEORA_POOL_QUERY", "SOL/USDC"),
        meteora_pool_address=env.get("METEORA_POOL_ADDRESS"),
        rebalance_interval_minutes=_parse_int(env.get("REBALANCE_INTERVAL_MINUTES"), 10),
        min_stay_probability=_parse_float(env.get("MIN_STAY_PROBABILITY"), 0.02),
        range_change_threshold_bps=_parse_float(env.get("RANGE_CHANGE_THRESHOLD_BPS"), 25.0),
        max_candidates=_parse_int(env.get("MAX_CANDIDATES"), 12),
        fee_window=env.get("FEE_WINDOW", "24h"),
        executor=env.get("EXECUTOR", "dry-run"),
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

