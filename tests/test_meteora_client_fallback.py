from __future__ import annotations

from ai_liquidity_optimizer.clients.meteora import MeteoraDlmmApiClient


def _pool_payload(address: str) -> dict:
    return {
        "address": address,
        "name": "SOL-USDC",
        "mint_x": {"address": "So11111111111111111111111111111111111111112", "symbol": "SOL", "decimals": 9},
        "mint_y": {"address": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v", "symbol": "USDC", "decimals": 6},
        "current_price": 90.0,
        "liquidity": 1_000_000,
        "trade_volume_24h": 500_000,
        "fees_24h": 1_000,
        "tvl": 1_000_000,
    }


class _SequenceHttpClient:
    def __init__(self, responses):
        self._responses = list(responses)

    def get_json(self, url, params=None, headers=None):
        if not self._responses:
            raise AssertionError("Unexpected get_json call")
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_find_pool_uses_cached_snapshot_on_direct_fetch_timeout():
    address = "BGm1tav58oGcsQJehL9WXBFXF7D27vZsKefj4xJKD5Y"
    http = _SequenceHttpClient(
        responses=[
            {"data": _pool_payload(address)},
            RuntimeError("The read operation timed out"),
        ]
    )
    client = MeteoraDlmmApiClient(base_url="https://example.invalid", http_client=http)

    first = client.find_sol_usdc_pool(pool_address=address)
    second = client.find_sol_usdc_pool(pool_address=address)

    assert first.address == address
    assert second.address == address
    assert second.current_price == first.current_price


def test_find_pool_uses_listing_fallback_for_non_timeout_fetch_failure_without_cache():
    address = "BGm1tav58oGcsQJehL9WXBFXF7D27vZsKefj4xJKD5Y"
    http = _SequenceHttpClient(
        responses=[
            RuntimeError("upstream 503"),
            {"data": [_pool_payload(address)]},
        ]
    )
    client = MeteoraDlmmApiClient(base_url="https://example.invalid", http_client=http)

    pool = client.find_sol_usdc_pool(pool_address=address, query="SOL/USDC")
    assert pool.address == address
    assert pool.symbol_x == "SOL"
    assert pool.symbol_y == "USDC"


def test_find_pool_does_not_attempt_listing_fallback_for_timeout_without_cache():
    address = "BGm1tav58oGcsQJehL9WXBFXF7D27vZsKefj4xJKD5Y"
    http = _SequenceHttpClient(
        responses=[
            RuntimeError("The read operation timed out"),
        ]
    )
    client = MeteoraDlmmApiClient(base_url="https://example.invalid", http_client=http)

    try:
        client.find_sol_usdc_pool(pool_address=address, query="SOL/USDC")
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert "timed out" in str(exc).lower()
