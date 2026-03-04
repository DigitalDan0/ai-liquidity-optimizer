from __future__ import annotations

import unittest
from types import SimpleNamespace

from ai_liquidity_optimizer.execution.base import PositionExecutor
from ai_liquidity_optimizer.models import BotState, ExecutionApplyRequest, ExecutionApplyResult
from ai_liquidity_optimizer.orchestrator import OptimizerOrchestrator
from ai_liquidity_optimizer.strategy.ev import EvLpScorer
from ai_liquidity_optimizer.strategy.scoring import StrategyScorer


class _StubExecutor(PositionExecutor):
    def apply_target_range(self, request: ExecutionApplyRequest) -> ExecutionApplyResult:  # pragma: no cover - unused
        raise RuntimeError("unused")

    def quote_target_bins(  # pragma: no cover - unused
        self,
        *,
        pool_address: str,
        symbol_x: str,
        symbol_y: str,
        api_current_price: float,
        target_lower_price: float,
        target_upper_price: float,
    ):
        return None


class _StubStateStore:
    def load(self):  # pragma: no cover - unused
        return BotState()

    def save(self, state):  # pragma: no cover - unused
        return None


class _FailingMeteoraClient:
    def find_sol_usdc_pool(self, pool_address: str | None = None, query: str = "SOL/USDC"):
        raise RuntimeError("timed out")

    def list_sol_usdc_pool_candidates(self, *, query: str, per_page: int, min_tvl_usd: float):
        raise RuntimeError("timed out")


class OrchestratorPoolFallbackTests(unittest.TestCase):
    def _orchestrator(self) -> OptimizerOrchestrator:
        settings = SimpleNamespace(
            meteora_pool_address="BGm1tav58oGcsQJehL9WXBFXF7D27vZsKefj4xJKD5Y",
            meteora_pool_query="SOL/USDC",
            pool_candidate_limit=12,
            min_pool_tvl_usd=100_000.0,
        )
        return OptimizerOrchestrator(
            settings=settings,
            synth_client=SimpleNamespace(),
            meteora_client=_FailingMeteoraClient(),
            scorer=StrategyScorer(),
            ev_scorer=EvLpScorer(),
            executor=_StubExecutor(),
            state_store=_StubStateStore(),
        )

    def test_load_pool_candidates_uses_stale_state_selected_pool_snapshot(self):
        orchestrator = self._orchestrator()
        state = BotState(
            last_decision={
                "selected_pool": {
                    "address": "BGm1tav58oGcsQJehL9WXBFXF7D27vZsKefj4xJKD5Y",
                    "name": "SOL-USDC",
                    "current_price": 90.0,
                    "symbol_x": "SOL",
                    "symbol_y": "USDC",
                    "tvl": 1234567.0,
                    "bin_step_bps": 10.0,
                    "base_fee_pct": 0.2,
                    "dynamic_fee_pct": 0.05,
                },
                "ev_best_candidate": {
                    "ev_components": {
                        "fee_rate_15m_fraction": 0.001,
                    }
                },
            }
        )

        pools = orchestrator._load_pool_candidates(state=state)
        self.assertEqual(len(pools), 1)
        pool = pools[0]
        self.assertEqual(pool.address, "BGm1tav58oGcsQJehL9WXBFXF7D27vZsKefj4xJKD5Y")
        self.assertEqual(pool.symbol_x, "SOL")
        self.assertEqual(pool.symbol_y, "USDC")
        self.assertAlmostEqual(pool.current_price, 90.0, places=9)
        self.assertAlmostEqual(pool.tvl_usd(), 1234567.0, places=9)
        self.assertIsNotNone(pool.fee_tvl_ratio_24h)
        assert pool.fee_tvl_ratio_24h is not None
        self.assertGreater(pool.fee_tvl_ratio_24h, 0.0)

    def test_load_pool_candidates_raises_when_no_state_fallback_available(self):
        orchestrator = self._orchestrator()
        with self.assertRaises(RuntimeError):
            orchestrator._load_pool_candidates(state=BotState())


if __name__ == "__main__":
    unittest.main()
