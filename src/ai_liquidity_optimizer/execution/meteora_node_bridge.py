from __future__ import annotations

import json
import logging
import math
import subprocess
from pathlib import Path
from typing import Any

from ai_liquidity_optimizer.execution.base import PositionExecutor
from ai_liquidity_optimizer.models import ActivePositionState, ExecutionApplyRequest, ExecutionApplyResult, ExecutorRangeBinQuote

LOGGER = logging.getLogger(__name__)


class MeteoraNodeBridgeExecutor(PositionExecutor):
    """Calls an optional Node.js helper that uses the official Meteora DLMM TypeScript SDK."""

    def __init__(
        self,
        repo_root: Path,
        rpc_url: str,
        private_key_b58: str,
        node_bin: str = "node",
        liquidity_mode: str = "spot",
        max_custom_weight_position_bins: int = 70,
        synth_weight_active_bin_floor_bps: int = 1000,
        synth_weight_max_bin_bps_per_side: int = 7000,
    ):
        self.repo_root = repo_root
        self.rpc_url = rpc_url
        self.private_key_b58 = private_key_b58
        self.node_bin = node_bin
        self.liquidity_mode = liquidity_mode
        self.max_custom_weight_position_bins = max_custom_weight_position_bins
        self.synth_weight_active_bin_floor_bps = synth_weight_active_bin_floor_bps
        self.synth_weight_max_bin_bps_per_side = synth_weight_max_bin_bps_per_side
        self.script_path = repo_root / "executors" / "meteora_ts" / "dlmm_executor.mjs"
        self._last_quote_failure_reason: str | None = None

    def reconcile_active_position(self, active_position: ActivePositionState | None) -> ActivePositionState | None:
        if active_position is None:
            return None
        if not active_position.position_pubkey:
            return active_position

        payload = {
            "command": "check-position",
            "rpc_url": self.rpc_url,
            "position_pubkey": active_position.position_pubkey,
        }
        try:
            result = self._run_node(payload)
        except Exception:
            # Best-effort only: avoid blocking the strategy loop if the health check itself fails.
            return active_position

        if not bool(result.get("exists", True)):
            return None
        return active_position

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
            "liquidity_mode": self.liquidity_mode,
            "max_custom_weight_position_bins": self.max_custom_weight_position_bins,
            "synth_weight_active_bin_floor_bps": self.synth_weight_active_bin_floor_bps,
            "synth_weight_max_bin_bps_per_side": self.synth_weight_max_bin_bps_per_side,
            "target_bin_ids": request.target_bin_ids,
            "target_bin_edges": request.target_bin_edges,
            "target_bin_weights": request.target_bin_weights,
        }
        result = self._run_node(payload)

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

    def close_position(self, active_position: ActivePositionState | None) -> ExecutionApplyResult:
        if active_position is None:
            return ExecutionApplyResult(
                changed=False,
                active_position=None,
                tx_signatures=[],
                details={"message": "No active position to close"},
            )
        payload = {
            "command": "close-position",
            "rpc_url": self.rpc_url,
            "private_key_b58": self.private_key_b58,
            "existing_position": active_position.to_dict(),
        }
        result = self._run_node(payload)
        return ExecutionApplyResult(
            changed=bool(result.get("changed", True)),
            active_position=None,
            tx_signatures=[str(x) for x in result.get("tx_signatures", []) if isinstance(x, str)],
            details={k: v for k, v in result.items() if k not in {"ok", "changed", "active_position", "tx_signatures"}},
        )

    def get_onchain_snapshot(
        self,
        *,
        pool: Any | None = None,
        active_position: ActivePositionState | None = None,
    ) -> dict[str, Any] | None:
        payload: dict[str, Any] = {
            "command": "wallet-snapshot",
            "rpc_url": self.rpc_url,
            "private_key_b58": self.private_key_b58,
        }
        if pool is not None:
            payload["pool"] = {
                "address": getattr(pool, "address", None),
                "name": getattr(pool, "name", None),
                "mint_x": getattr(pool, "mint_x", None),
                "mint_y": getattr(pool, "mint_y", None),
                "symbol_x": getattr(pool, "symbol_x", None),
                "symbol_y": getattr(pool, "symbol_y", None),
                "decimals_x": getattr(pool, "decimals_x", None),
                "decimals_y": getattr(pool, "decimals_y", None),
                "api_current_price": getattr(pool, "current_price", None),
            }
        if active_position is not None:
            payload["active_position"] = active_position.to_dict()
        result = self._run_node(payload)
        pool_balances = result.get("pool_balances") if isinstance(result.get("pool_balances"), dict) else None
        position_snapshot = result.get("position_snapshot") if isinstance(result.get("position_snapshot"), dict) else None

        spot_price = self._first_valid_float(
            result.get("spot_price_sol_usdc"),
            self._dict_get(pool_balances, "spot_sol_usdc"),
        )
        sol_balance = self._first_valid_float(
            result.get("sol_balance"),
            result.get("wallet_sol_total_balance"),
            result.get("wallet_sol_total_ui"),
            self._dict_get(pool_balances, "wallet_sol_total_ui"),
            self._dict_get(pool_balances, "sol_ui"),
        )
        usdc_balance = self._first_valid_float(
            result.get("usdc_balance"),
            result.get("wallet_usdc_total_balance"),
            result.get("wallet_usdc_total_ui"),
            self._dict_get(pool_balances, "wallet_usdc_ui"),
            self._dict_get(pool_balances, "usdc_ui"),
        )
        wallet_total_usd = self._first_valid_float(
            result.get("wallet_total_usd_est"),
            self._dict_get(pool_balances, "wallet_total_usd_est"),
            self._dict_get(pool_balances, "total_usd_est"),
        )
        position_total_usd = self._first_valid_float(
            result.get("position_total_usd_est"),
            self._dict_get(position_snapshot, "total_usd_est"),
        )
        total_usd = self._first_valid_float(result.get("total_usd_est"))
        if total_usd is None and (wallet_total_usd is not None or position_total_usd is not None):
            total_usd = float(wallet_total_usd or 0.0) + float(position_total_usd or 0.0)

        # Normalize to a stable shape used by orchestrator/journal.
        snapshot: dict[str, Any] = {
            "source": "meteora-node",
            "snapshot_at": str(result.get("snapshot_at") or ""),
            "wallet_pubkey": result.get("wallet_pubkey"),
            "sol_balance": sol_balance,
            "usdc_balance": usdc_balance,
            "total_usd_est": total_usd,
            "spot_price_sol_usdc": spot_price,
            "active_position_exists": bool(result.get("active_position_exists")) if result.get("active_position_exists") is not None else None,
            "native_sol_balance": self._first_valid_float(result.get("native_sol_balance")),
            "wallet_sol_token_balance": self._first_valid_float(
                result.get("wallet_sol_token_balance"),
                result.get("wallet_sol_token_ui"),
                self._dict_get(pool_balances, "wallet_sol_token_ui"),
            ),
            "wallet_sol_total_balance": self._first_valid_float(
                result.get("wallet_sol_total_balance"),
                result.get("wallet_sol_total_ui"),
                self._dict_get(pool_balances, "wallet_sol_total_ui"),
                self._dict_get(pool_balances, "sol_ui"),
            ),
            "wallet_usdc_total_balance": self._first_valid_float(
                result.get("wallet_usdc_total_balance"),
                result.get("wallet_usdc_total_ui"),
                self._dict_get(pool_balances, "wallet_usdc_ui"),
                self._dict_get(pool_balances, "usdc_ui"),
            ),
            "wallet_total_usd_est": wallet_total_usd,
            "position_total_usd_est": position_total_usd,
            "position_snapshot": position_snapshot,
            "pool_balances": pool_balances,
        }
        return snapshot

    def quote_target_bins(
        self,
        *,
        pool_address: str,
        symbol_x: str,
        symbol_y: str,
        api_current_price: float,
        target_lower_price: float,
        target_upper_price: float,
    ) -> ExecutorRangeBinQuote | None:
        self._last_quote_failure_reason = None
        payload = {
            "command": "quote-range-bins",
            "rpc_url": self.rpc_url,
            "pool": {
                "address": pool_address,
                "symbol_x": symbol_x,
                "symbol_y": symbol_y,
                "api_current_price": api_current_price,
            },
            "target_range_sol_usdc": {
                "lower": target_lower_price,
                "upper": target_upper_price,
            },
        }
        result = self._run_node(payload)
        bins_raw = result.get("bins")
        if not isinstance(bins_raw, list) or not bins_raw:
            self._record_quote_failure(
                "quote-range-bins returned no bins "
                f"(pool={pool_address} range={target_lower_price:.4f}-{target_upper_price:.4f} "
                f"keys={sorted(result.keys()) if isinstance(result, dict) else 'n/a'})"
            )
            return None
        sdk_price_orientation = (
            str(result.get("sdk_price_orientation")) if result.get("sdk_price_orientation") is not None else None
        )
        parsed_rows: list[tuple[int, float | None, float | None]] = []
        for item in bins_raw:
            if not isinstance(item, dict):
                continue
            try:
                bin_id = int(item.get("bin_id"))
            except (TypeError, ValueError):
                continue
            price_sdk = self._coerce_finite_positive(item.get("price_sdk"))
            price_sol_usdc = self._coerce_finite_positive(item.get("price_sol_usdc"))
            if price_sol_usdc is None:
                price_pool = self._coerce_finite_positive(item.get("price_pool_orientation"))
                if price_pool is None and price_sdk is not None:
                    price_pool = self._sdk_price_to_pool_price(
                        sdk_price=price_sdk,
                        sdk_price_orientation=sdk_price_orientation,
                    )
                if price_pool is not None:
                    price_sol_usdc = self._pool_price_to_sol_usdc(
                        pool_price=price_pool,
                        symbol_x=symbol_x,
                        symbol_y=symbol_y,
                    )
            parsed_rows.append((bin_id, price_sol_usdc, price_sdk))

        if not parsed_rows:
            sample = bins_raw[0] if bins_raw else None
            self._record_quote_failure(
                "quote-range-bins parsing produced zero valid rows "
                f"(pool={pool_address} range={target_lower_price:.4f}-{target_upper_price:.4f} sample={sample})"
            )
            return None
        parsed_rows.sort(key=lambda row: row[0])

        bin_ids: list[int] = [row[0] for row in parsed_rows]
        maybe_prices_sol_usdc: list[float | None] = [row[1] for row in parsed_rows]
        bin_prices_sdk: list[float] = [row[2] for row in parsed_rows if row[2] is not None]
        if any(price is None for price in maybe_prices_sol_usdc):
            maybe_prices_sol_usdc = self._fill_missing_prices_from_range(
                prices=maybe_prices_sol_usdc,
                range_lower=target_lower_price,
                range_upper=target_upper_price,
            )
        if any(price is None for price in maybe_prices_sol_usdc):
            missing_count = sum(1 for price in maybe_prices_sol_usdc if price is None)
            self._record_quote_failure(
                "quote-range-bins could not resolve prices "
                f"(pool={pool_address} range={target_lower_price:.4f}-{target_upper_price:.4f} missing={missing_count})"
            )
            return None
        bin_prices_sol_usdc = [float(price) for price in maybe_prices_sol_usdc if price is not None]
        if len(bin_prices_sol_usdc) != len(bin_ids):
            self._record_quote_failure(
                "quote-range-bins length mismatch after parsing "
                f"(pool={pool_address} ids={len(bin_ids)} prices={len(bin_prices_sol_usdc)})"
            )
            return None

        return ExecutorRangeBinQuote(
            pool_address=str(result.get("pool_address") or pool_address),
            active_bin_id=int(result["active_bin_id"]) if result.get("active_bin_id") is not None else None,
            sdk_price_orientation=sdk_price_orientation,
            pool_price_orientation=str(result.get("pool_price_orientation")) if result.get("pool_price_orientation") is not None else None,
            lower_for_sdk=float(result["lower_for_sdk"]) if result.get("lower_for_sdk") is not None else None,
            upper_for_sdk=float(result["upper_for_sdk"]) if result.get("upper_for_sdk") is not None else None,
            bin_ids=bin_ids,
            bin_prices_sol_usdc=bin_prices_sol_usdc,
            bin_prices_sdk=bin_prices_sdk,
            raw=result,
        )

    def last_quote_failure_reason(self) -> str | None:
        return self._last_quote_failure_reason

    @staticmethod
    def _coerce_finite_positive(value: object) -> float | None:
        try:
            val = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        if not math.isfinite(val) or val <= 0:
            return None
        return val

    @staticmethod
    def _sdk_price_to_pool_price(*, sdk_price: float, sdk_price_orientation: str | None) -> float | None:
        if not math.isfinite(sdk_price) or sdk_price <= 0:
            return None
        if sdk_price_orientation == "inverted_vs_api":
            if sdk_price == 0:
                return None
            return 1.0 / sdk_price
        return sdk_price

    @staticmethod
    def _pool_price_to_sol_usdc(*, pool_price: float, symbol_x: str, symbol_y: str) -> float | None:
        if not math.isfinite(pool_price) or pool_price <= 0:
            return None
        sx = symbol_x.upper()
        sy = symbol_y.upper()
        if sx == "SOL" and sy == "USDC":
            return pool_price
        if sx == "USDC" and sy == "SOL":
            if pool_price == 0:
                return None
            return 1.0 / pool_price
        return None

    @staticmethod
    def _fill_missing_prices_from_range(
        *,
        prices: list[float | None],
        range_lower: float,
        range_upper: float,
    ) -> list[float | None]:
        n = len(prices)
        if n == 0:
            return prices
        lo = float(min(range_lower, range_upper))
        hi = float(max(range_lower, range_upper))
        if not math.isfinite(lo) or not math.isfinite(hi) or lo <= 0 or hi <= 0:
            return prices
        if hi <= lo:
            hi = lo * (1.0 + 1e-9)

        if n == 1:
            default_prices = [math.sqrt(lo * hi)]
        else:
            ratio = math.exp(math.log(hi / lo) / max(n - 1, 1))
            default_prices = [lo * (ratio**i) for i in range(n)]

        filled = [prices[i] if prices[i] is not None else default_prices[i] for i in range(n)]
        # Enforce strictly increasing centers for downstream edge derivation.
        for i in range(1, n):
            prev = float(filled[i - 1])
            cur = float(filled[i])
            if cur <= prev:
                cur = prev * (1.0 + 1e-9)
                filled[i] = cur
        return filled

    @staticmethod
    def _coerce_optional_float(value: object) -> float | None:
        try:
            if value is None:
                return None
            val = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        if not math.isfinite(val):
            return None
        return val

    @classmethod
    def _first_valid_float(cls, *values: object) -> float | None:
        for value in values:
            parsed = cls._coerce_optional_float(value)
            if parsed is not None:
                return parsed
        return None

    @staticmethod
    def _dict_get(payload: dict[str, Any] | None, key: str) -> object:
        if not isinstance(payload, dict):
            return None
        return payload.get(key)

    def _run_node(self, payload: dict) -> dict:
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
        return result

    def _record_quote_failure(self, reason: str) -> None:
        self._last_quote_failure_reason = reason
        LOGGER.warning("%s", reason)
