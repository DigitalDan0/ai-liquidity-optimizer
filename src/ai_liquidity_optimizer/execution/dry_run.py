from __future__ import annotations

from ai_liquidity_optimizer.execution.base import PositionExecutor
from ai_liquidity_optimizer.models import ActivePositionState, ExecutionApplyRequest, ExecutionApplyResult


class DryRunExecutor(PositionExecutor):
    def apply_target_range(self, request: ExecutionApplyRequest) -> ExecutionApplyResult:
        existing = request.existing_position
        simulated_pubkey = existing.position_pubkey if existing and existing.position_pubkey else "SIMULATED_POSITION"
        active = ActivePositionState(
            pool_address=request.pool.address,
            lower_price=request.target_lower_price,
            upper_price=request.target_upper_price,
            width_pct=request.target_forecast.width_pct,
            executor="dry-run",
            position_pubkey=simulated_pubkey,
            lower_bin_id=existing.lower_bin_id if existing else None,
            upper_bin_id=existing.upper_bin_id if existing else None,
            tx_signatures=[],
            meta={
                "mode": "dry-run",
                "deposit_sol_amount": request.deposit_sol_amount,
                "deposit_usdc_amount": request.deposit_usdc_amount,
            },
        )
        return ExecutionApplyResult(
            changed=True,
            active_position=active,
            tx_signatures=[],
            details={"mode": "dry-run", "message": "No on-chain transactions submitted"},
        )

