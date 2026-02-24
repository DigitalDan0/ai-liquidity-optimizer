from __future__ import annotations

from abc import ABC, abstractmethod

from ai_liquidity_optimizer.models import ExecutionApplyRequest, ExecutionApplyResult


class PositionExecutor(ABC):
    @abstractmethod
    def apply_target_range(self, request: ExecutionApplyRequest) -> ExecutionApplyResult:
        raise NotImplementedError

