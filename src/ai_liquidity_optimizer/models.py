from __future__ import annotations

from dataclasses import asdict, field
from datetime import datetime, timezone
from typing import Any

from ai_liquidity_optimizer.compat import dataclass


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
class SynthLpProbabilityPoint:
    price: float
    probability_below: float | None = None
    probability_above: float | None = None


@dataclass(slots=True)
class SynthLpProbabilitiesSnapshot:
    asset: str
    horizon: str
    points: list[SynthLpProbabilityPoint]
    current_price: float | None = None
    as_of: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset": self.asset,
            "horizon": self.horizon,
            "current_price": self.current_price,
            "as_of": self.as_of,
            "points": [
                {
                    "price": p.price,
                    "probability_below": p.probability_below,
                    "probability_above": p.probability_above,
                }
                for p in self.points
            ],
        }


@dataclass(slots=True)
class SynthPredictionPercentilesSnapshot:
    asset: str
    percentiles_by_step: list[dict[float, float]]
    current_price: float | None = None
    step_minutes: int = 5
    as_of: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset": self.asset,
            "current_price": self.current_price,
            "step_minutes": self.step_minutes,
            "as_of": self.as_of,
            "num_steps": len(self.percentiles_by_step),
        }


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

    def current_price_sol_usdc(self) -> float:
        sx = self.symbol_x.upper()
        sy = self.symbol_y.upper()
        if self.current_price <= 0:
            return 0.0
        if sx == "SOL" and sy == "USDC":
            return self.current_price
        if sx == "USDC" and sy == "SOL":
            return 1.0 / self.current_price
        return self.current_price


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
class BinWeightingConfig:
    tau_half_minutes: int = 90
    alpha: float = 1.15
    eps: float = 1e-6
    low_mass_threshold: float = 0.02
    terminal_mass_weight_no_path: float = 0.85
    proximity_weight_no_path: float = 0.15
    path_weight: float = 0.60
    terminal_mass_weight_with_path: float = 0.30
    proximity_weight_with_path: float = 0.10
    final_floor_blend: float = 0.08
    sigma_range_scale: float = 0.35
    sigma_min: float = 0.005


@dataclass(slots=True)
class BinWeightingDiagnostics:
    mass_in_range: float
    used_prediction_percentiles: bool
    fallback_reason: str | None
    confidence_factor: float
    t_frac: float
    entropy: float
    terminal_cdf_points: int
    num_bins: int
    binning_mode: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class WeightedBinPlan:
    range_lower: float
    range_upper: float
    bin_edges: list[float]
    weights: list[float]
    diagnostics: BinWeightingDiagnostics
    distribution_components: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "range_lower": self.range_lower,
            "range_upper": self.range_upper,
            "bin_edges": self.bin_edges,
            "weights": self.weights,
            "diagnostics": self.diagnostics.to_dict(),
            "distribution_components": self.distribution_components,
        }


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
    target_bin_edges: list[float] | None = None
    target_bin_weights: list[float] | None = None


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
