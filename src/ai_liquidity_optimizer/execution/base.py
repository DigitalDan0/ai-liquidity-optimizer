from __future__ import annotations

from abc import ABC, abstractmethod

from ai_liquidity_optimizer.models import ActivePositionState, ExecutionApplyRequest, ExecutionApplyResult, ExecutorRangeBinQuote


class PositionExecutor(ABC):
    def reconcile_active_position(self, active_position: ActivePositionState | None) -> ActivePositionState | None:
        """Best-effort hook to reconcile local state with chain/executor state.

        Default behavior is a no-op so dry-run and simple executors remain compatible.
        """
        return active_position

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
        """Optional hook for returning exact SDK bins for a target range.

        Default behavior is no-op for executors that cannot provide quotes.
        """
        return None

    def close_position(self, active_position: ActivePositionState | None) -> ExecutionApplyResult:
        """Optional hook to close an existing active position and stay idle."""
        return ExecutionApplyResult(
            changed=False,
            active_position=active_position,
            tx_signatures=[],
            details={"message": "close_position is not supported by this executor"},
        )

    @abstractmethod
    def apply_target_range(self, request: ExecutionApplyRequest) -> ExecutionApplyResult:
        raise NotImplementedError
