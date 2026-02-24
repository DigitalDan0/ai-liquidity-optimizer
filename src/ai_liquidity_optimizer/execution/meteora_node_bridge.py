from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ai_liquidity_optimizer.execution.base import PositionExecutor
from ai_liquidity_optimizer.models import ActivePositionState, ExecutionApplyRequest, ExecutionApplyResult


class MeteoraNodeBridgeExecutor(PositionExecutor):
    """Calls an optional Node.js helper that uses the official Meteora DLMM TypeScript SDK."""

    def __init__(
        self,
        repo_root: Path,
        rpc_url: str,
        private_key_b58: str,
        node_bin: str = "node",
    ):
        self.repo_root = repo_root
        self.rpc_url = rpc_url
        self.private_key_b58 = private_key_b58
        self.node_bin = node_bin
        self.script_path = repo_root / "executors" / "meteora_ts" / "dlmm_executor.mjs"

    def apply_target_range(self, request: ExecutionApplyRequest) -> ExecutionApplyResult:
        payload = {
            "command": "apply-range",
            "rpc_url": self.rpc_url,
            "private_key_b58": self.private_key_b58,
            "pool": {
                "address": request.pool.address,
                "name": request.pool.name,
                "symbol_x": request.pool.symbol_x,
                "symbol_y": request.pool.symbol_y,
                "decimals_x": request.pool.decimals_x,
                "decimals_y": request.pool.decimals_y,
                "api_current_price": request.pool.current_price,
            },
            "target_range_sol_usdc": {
                "lower": request.target_lower_price,
                "upper": request.target_upper_price,
            },
            "deposit": {
                "sol": request.deposit_sol_amount,
                "usdc": request.deposit_usdc_amount,
            },
            "existing_position": request.existing_position.to_dict() if request.existing_position else None,
        }
        proc = subprocess.run(
            [self.node_bin, str(self.script_path)],
            input=json.dumps(payload).encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(self.script_path.parent),
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "Meteora node executor failed. "
                f"exit={proc.returncode}, stderr={proc.stderr.decode('utf-8', errors='replace')}"
            )
        try:
            result = json.loads(proc.stdout.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON from Meteora node executor: {proc.stdout!r}") from exc

        if not result.get("ok", False):
            raise RuntimeError(f"Meteora node executor returned error: {result}")

        pos = result.get("active_position")
        active_position = None
        if isinstance(pos, dict):
            active_position = ActivePositionState(
                pool_address=str(pos.get("pool_address") or request.pool.address),
                lower_price=float(pos.get("lower_price") or request.target_lower_price),
                upper_price=float(pos.get("upper_price") or request.target_upper_price),
                width_pct=float(pos.get("width_pct") or request.target_forecast.width_pct),
                executor="meteora-node",
                position_pubkey=str(pos.get("position_pubkey")) if pos.get("position_pubkey") else None,
                lower_bin_id=int(pos["lower_bin_id"]) if pos.get("lower_bin_id") is not None else None,
                upper_bin_id=int(pos["upper_bin_id"]) if pos.get("upper_bin_id") is not None else None,
                tx_signatures=[str(x) for x in result.get("tx_signatures", []) if isinstance(x, str)],
                meta={k: v for k, v in pos.items() if k not in {"pool_address", "lower_price", "upper_price", "width_pct", "position_pubkey", "lower_bin_id", "upper_bin_id"}},
            )

        return ExecutionApplyResult(
            changed=bool(result.get("changed", True)),
            active_position=active_position,
            tx_signatures=[str(x) for x in result.get("tx_signatures", []) if isinstance(x, str)],
            details={k: v for k, v in result.items() if k not in {"ok", "changed", "active_position", "tx_signatures"}},
        )

