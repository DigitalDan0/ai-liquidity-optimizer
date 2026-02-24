from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class SynthLpBoundForecast:
    width_pct: float
    lower_bound: float
    upper_bound: float
    probability_to_stay_in_interval: float
    expected_time_in_interval_minutes: float
    expected_impermanent_loss: float

    @property
    def mid_price(self) -> float:
        return (self.lower_bound + self.upper_bound) / 2.0


@dataclass(slots=True)
class MeteoraPoolSnapshot:
    address: str
    name: str
    mint_x: str
    mint_y: str
    symbol_x: str
    symbol_y: str
    decimals_x: int
    decimals_y: int
    current_price: float
    liquidity: float
    volume_24h: float
    fees_24h: float
    fee_tvl_ratio_24h: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def fee_return_fraction_24h(self) -> float:
        if self.fee_tvl_ratio_24h is not None:
            return normalize_fraction(self.fee_tvl_ratio_24h)
        if self.liquidity > 0 and self.fees_24h >= 0:
            return self.fees_24h / self.liquidity
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data


@dataclass(slots=True)
class ScoredCandidate:
    forecast: SynthLpBoundForecast
    expected_active_fraction: float
    expected_fee_return_fraction: float
    confidence_multiplier: float
    expected_net_return_fraction: float
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "width_pct": self.forecast.width_pct,
            "lower_bound": self.forecast.lower_bound,
            "upper_bound": self.forecast.upper_bound,
            "probability_to_stay_in_interval": self.forecast.probability_to_stay_in_interval,
            "expected_time_in_interval_minutes": self.forecast.expected_time_in_interval_minutes,
            "expected_impermanent_loss": self.forecast.expected_impermanent_loss,
            "expected_active_fraction": self.expected_active_fraction,
            "expected_fee_return_fraction": self.expected_fee_return_fraction,
            "confidence_multiplier": self.confidence_multiplier,
            "expected_net_return_fraction": self.expected_net_return_fraction,
            "score": self.score,
        }


@dataclass(slots=True)
class StrategyDecision:
    chosen: ScoredCandidate
    ranked: list[ScoredCandidate]
    horizon: str
    generated_at: str = field(default_factory=utc_now_iso)


@dataclass(slots=True)
class ActivePositionState:
    pool_address: str
    lower_price: float
    upper_price: float
    width_pct: float
    executor: str
    position_pubkey: str | None = None
    lower_bin_id: int | None = None
    upper_bin_id: int | None = None
    tx_signatures: list[str] = field(default_factory=list)
    updated_at: str = field(default_factory=utc_now_iso)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class BotState:
    active_position: ActivePositionState | None = None
    last_decision: dict[str, Any] | None = None
    updated_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_position": self.active_position.to_dict() if self.active_position else None,
            "last_decision": self.last_decision,
            "updated_at": self.updated_at,
        }


@dataclass(slots=True)
class ExecutionApplyRequest:
    pool: MeteoraPoolSnapshot
    target_forecast: SynthLpBoundForecast
    target_lower_price: float
    target_upper_price: float
    deposit_sol_amount: float
    deposit_usdc_amount: float
    existing_position: ActivePositionState | None


@dataclass(slots=True)
class ExecutionApplyResult:
    changed: bool
    active_position: ActivePositionState | None
    tx_signatures: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


def normalize_fraction(value: float) -> float:
    if value < 0:
        return 0.0
    if value > 1.0:
        # Heuristic: some APIs expose percentage values like 2.5 for 2.5%.
        if value <= 100.0:
            return value / 100.0
    return value

