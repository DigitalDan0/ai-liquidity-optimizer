from __future__ import annotations

import json
from pathlib import Path

from ai_liquidity_optimizer.models import ActivePositionState, BotState, utc_now_iso


class JsonStateStore:
    def __init__(self, path: Path):
        self.path = path

    def load(self) -> BotState:
        if not self.path.exists():
            return BotState()
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        active = None
        active_raw = raw.get("active_position")
        if isinstance(active_raw, dict):
            active = ActivePositionState(
                pool_address=str(active_raw.get("pool_address") or ""),
                lower_price=float(active_raw.get("lower_price") or 0.0),
                upper_price=float(active_raw.get("upper_price") or 0.0),
                width_pct=float(active_raw.get("width_pct") or 0.0),
                executor=str(active_raw.get("executor") or "unknown"),
                position_pubkey=str(active_raw["position_pubkey"]) if active_raw.get("position_pubkey") else None,
                lower_bin_id=int(active_raw["lower_bin_id"]) if active_raw.get("lower_bin_id") is not None else None,
                upper_bin_id=int(active_raw["upper_bin_id"]) if active_raw.get("upper_bin_id") is not None else None,
                tx_signatures=[str(x) for x in active_raw.get("tx_signatures", []) if isinstance(x, str)],
                updated_at=str(active_raw.get("updated_at") or utc_now_iso()),
                meta=dict(active_raw.get("meta") or {}),
            )

        return BotState(
            active_position=active,
            last_decision=raw.get("last_decision") if isinstance(raw.get("last_decision"), dict) else None,
            updated_at=str(raw.get("updated_at") or utc_now_iso()),
        )

    def save(self, state: BotState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        payload = state.to_dict()
        payload["updated_at"] = utc_now_iso()
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(self.path)

