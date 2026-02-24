from __future__ import annotations

from typing import Any

from ai_liquidity_optimizer.http import JsonHttpClient
from ai_liquidity_optimizer.models import MeteoraPoolSnapshot


class MeteoraDlmmApiClient:
    """Meteora DLMM Data API client for pool discovery and metrics."""

    def __init__(self, base_url: str, http_client: JsonHttpClient | None = None):
        self.base_url = base_url.rstrip("/")
        self.http = http_client or JsonHttpClient()

    def list_pools(
        self,
        query: str | None = None,
        page: int = 1,
        per_page: int = 100,
        sort_by: str = "tvl",
        order: str = "desc",
    ) -> list[MeteoraPoolSnapshot]:
        sort_param = sort_by if ":" in sort_by else f"{sort_by}:{order}"
        payload = self.http.get_json(
            f"{self.base_url}/pools",
            params={
                "query": query,
                "page": page,
                "per_page": per_page,
                "sort_by": sort_param,
                "hide_low_tvl": "true",
            },
            headers=_meteora_headers(),
        )
        return [self._parse_pool(p) for p in _extract_pools(payload)]

    def get_pool(self, pool_address: str) -> MeteoraPoolSnapshot:
        payload = self.http.get_json(f"{self.base_url}/pools/{pool_address}", headers=_meteora_headers())
        if isinstance(payload, dict) and isinstance(payload.get("data"), dict):
            return self._parse_pool(payload["data"])
        if isinstance(payload, dict):
            return self._parse_pool(payload)
        raise RuntimeError("Unexpected Meteora pool response format")

    def find_sol_usdc_pool(self, pool_address: str | None = None, query: str = "SOL/USDC") -> MeteoraPoolSnapshot:
        if pool_address:
            pool = self.get_pool(pool_address)
            if not _is_sol_usdc_pair(pool):
                raise RuntimeError(f"Configured pool {pool_address} is not SOL/USDC according to Meteora API")
            return pool

        pools = self.list_pools(query=query, per_page=50)
        candidates = [p for p in pools if _is_sol_usdc_pair(p)]
        if not candidates:
            # fallback: fetch more generic search results and local-filter
            pools = self.list_pools(query="SOL", per_page=200)
            candidates = [p for p in pools if _is_sol_usdc_pair(p)]
        if not candidates:
            raise RuntimeError("No SOL/USDC DLMM pool found via Meteora API")

        candidates.sort(key=lambda p: (p.liquidity, p.volume_24h), reverse=True)
        return candidates[0]

    def _parse_pool(self, raw: dict[str, Any]) -> MeteoraPoolSnapshot:
        token_x = raw.get("mint_x") or raw.get("token_x") or {}
        token_y = raw.get("mint_y") or raw.get("token_y") or {}
        fee_tvl_ratio = raw.get("fee_tvl_ratio")
        fee_tvl_ratio_24h = None
        if isinstance(fee_tvl_ratio, dict) and "24h" in fee_tvl_ratio:
            try:
                fee_tvl_ratio_24h = float(fee_tvl_ratio["24h"])
            except (TypeError, ValueError):
                fee_tvl_ratio_24h = None

        volume = raw.get("volume") if isinstance(raw.get("volume"), dict) else {}
        fees = raw.get("fees") if isinstance(raw.get("fees"), dict) else {}
        pool_config = raw.get("pool_config") if isinstance(raw.get("pool_config"), dict) else None

        liquidity = raw.get("liquidity")
        if liquidity is None:
            liquidity = raw.get("tvl")
        volume_24h = raw.get("trade_volume_24h")
        if volume_24h is None:
            volume_24h = raw.get("volume_24h")
        if volume_24h is None and volume:
            volume_24h = volume.get("24h")
        fees_24h = raw.get("fees_24h")
        if fees_24h is None and fees:
            fees_24h = fees.get("24h")

        return MeteoraPoolSnapshot(
            address=str(raw.get("address") or raw.get("pool_address") or ""),
            name=str(raw.get("name") or ""),
            mint_x=str(token_x.get("address") or raw.get("mint_x_address") or ""),
            mint_y=str(token_y.get("address") or raw.get("mint_y_address") or ""),
            symbol_x=str(token_x.get("symbol") or raw.get("mint_x_symbol") or "").upper(),
            symbol_y=str(token_y.get("symbol") or raw.get("mint_y_symbol") or "").upper(),
            decimals_x=int(token_x.get("decimals") or raw.get("mint_x_decimals") or 0),
            decimals_y=int(token_y.get("decimals") or raw.get("mint_y_decimals") or 0),
            current_price=float(raw.get("current_price") or 0.0),
            liquidity=float(liquidity or 0.0),
            volume_24h=float(volume_24h or 0.0),
            fees_24h=float(fees_24h or 0.0),
            fee_tvl_ratio_24h=fee_tvl_ratio_24h,
            raw={**raw, "_pool_config": pool_config} if pool_config else raw,
        )


def _extract_pools(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        if isinstance(data, dict):
            pools = data.get("pools")
            if isinstance(pools, list):
                return [x for x in pools if isinstance(x, dict)]
        pools = payload.get("pools")
        if isinstance(pools, list):
            return [x for x in pools if isinstance(x, dict)]
    raise RuntimeError("Unexpected Meteora pools response format")


def _is_sol_usdc_pair(pool: MeteoraPoolSnapshot) -> bool:
    symbols = {pool.symbol_x.upper(), pool.symbol_y.upper()}
    return symbols == {"SOL", "USDC"}


def _meteora_headers() -> dict[str, str]:
    return {
        "Accept": "application/json, text/plain, */*",
        # These help some edge gateways treat the request like a normal web client.
        "Origin": "https://app.meteora.ag",
        "Referer": "https://app.meteora.ag/",
    }
