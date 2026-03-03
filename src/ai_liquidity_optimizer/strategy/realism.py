from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ai_liquidity_optimizer.compat import dataclass


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _parse_timestamp(raw: Any) -> datetime | None:
    if not isinstance(raw, str) or not raw:
        return None
    text = raw.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    q = _clamp(float(q), 0.0, 1.0)
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    weight_hi = pos - lo
    return sorted_values[lo] * (1.0 - weight_hi) + sorted_values[hi] * weight_hi


def _winsorize(values: list[float], lo_q: float = 0.05, hi_q: float = 0.95) -> list[float]:
    if not values:
        return []
    ordered = sorted(values)
    lo = _percentile(ordered, lo_q)
    hi = _percentile(ordered, hi_q)
    if lo > hi:
        lo, hi = hi, lo
    return [_clamp(v, lo, hi) for v in values]


def _ewma_weights(count: int, alpha: float = 0.20) -> list[float]:
    if count <= 0:
        return []
    alpha = _clamp(alpha, 1e-6, 1.0)
    # Oldest -> newest, with newest receiving the largest weight.
    raw = [((1.0 - alpha) ** (count - 1 - i)) for i in range(count)]
    total = sum(raw)
    if total <= 0:
        return [1.0 / float(count)] * count
    return [w / total for w in raw]


def _weighted_mean(values: list[float], weights: list[float]) -> float | None:
    if not values or not weights or len(values) != len(weights):
        return None
    total_w = sum(weights)
    if total_w <= 0:
        return None
    return sum(v * w for v, w in zip(values, weights)) / total_w


def _weighted_rmse(values: list[float], weights: list[float]) -> float | None:
    if not values or not weights or len(values) != len(weights):
        return None
    total_w = sum(weights)
    if total_w <= 0:
        return None
    mse = sum((v * v) * w for v, w in zip(values, weights)) / total_w
    if mse < 0:
        return None
    return math.sqrt(mse)


@dataclass(slots=True)
class CalibrationSnapshot:
    fee_realism_multiplier: float
    rebalance_drag_usd: float
    model_rmse_usd: float
    sample_count: int
    fee_sample_count: int
    rebalance_sample_count: int
    mode: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "fee_realism_multiplier": self.fee_realism_multiplier,
            "rebalance_drag_usd": self.rebalance_drag_usd,
            "model_rmse_usd": self.model_rmse_usd,
            "sample_count": self.sample_count,
            "fee_sample_count": self.fee_sample_count,
            "rebalance_sample_count": self.rebalance_sample_count,
            "mode": self.mode,
        }


def default_calibration_snapshot(
    *,
    fee_realism_prior: float,
    fee_realism_min: float,
    fee_realism_max: float,
    rebalance_drag_prior_usd: float,
    rebalance_drag_min_usd: float,
    rebalance_drag_max_usd: float,
) -> CalibrationSnapshot:
    fee_multiplier = _clamp(float(fee_realism_prior), float(fee_realism_min), float(fee_realism_max))
    rebalance_drag = _clamp(
        float(rebalance_drag_prior_usd),
        float(rebalance_drag_min_usd),
        float(rebalance_drag_max_usd),
    )
    rmse_prior = max(0.005, rebalance_drag)
    return CalibrationSnapshot(
        fee_realism_multiplier=fee_multiplier,
        rebalance_drag_usd=rebalance_drag,
        model_rmse_usd=rmse_prior,
        sample_count=0,
        fee_sample_count=0,
        rebalance_sample_count=0,
        mode="warmup",
    )


def build_calibration_snapshot_from_journal(
    *,
    journal_path: Path,
    window_hours: int,
    min_samples: int,
    fee_realism_prior: float,
    fee_realism_min: float,
    fee_realism_max: float,
    rebalance_drag_prior_usd: float,
    rebalance_drag_min_usd: float,
    rebalance_drag_max_usd: float,
    now: datetime | None = None,
) -> CalibrationSnapshot:
    snapshot = default_calibration_snapshot(
        fee_realism_prior=fee_realism_prior,
        fee_realism_min=fee_realism_min,
        fee_realism_max=fee_realism_max,
        rebalance_drag_prior_usd=rebalance_drag_prior_usd,
        rebalance_drag_min_usd=rebalance_drag_min_usd,
        rebalance_drag_max_usd=rebalance_drag_max_usd,
    )
    if not journal_path.exists():
        return snapshot

    cutoff = (now or datetime.now(timezone.utc)) - timedelta(hours=max(int(window_hours), 1))
    rows: list[tuple[datetime, dict[str, Any]]] = []
    try:
        with journal_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                try:
                    row = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                if str(row.get("event_type") or "") != "decision":
                    continue
                ts = _parse_timestamp(row.get("decision_timestamp")) or _parse_timestamp(row.get("recorded_at"))
                if ts is None or ts < cutoff:
                    continue
                rows.append((ts, row))
    except OSError:
        return snapshot

    if not rows:
        return snapshot

    rows.sort(key=lambda item: item[0])
    residual_rows: list[tuple[float, float, float, bool]] = []
    # (expected_net, realized_net, residual, is_rebalance_row)
    fee_ratios: list[float] = []
    rebalance_drags: list[float] = []

    for _ts, row in rows:
        expected_net = _float_or_none(row.get("selected_expected_net_usd"))
        realized_net = _float_or_none(row.get("onchain_delta_total_usd_est"))
        if expected_net is None or realized_net is None:
            continue
        selected_action = str(row.get("selected_action") or "")
        is_rebalance_row = bool(row.get("rebalance_should")) or selected_action == "rebalance"
        residual = expected_net - realized_net
        residual_rows.append((expected_net, realized_net, residual, is_rebalance_row))

        model_fees = _float_or_none(row.get("selected_expected_fees_usd"))
        model_il = _float_or_none(row.get("selected_expected_il_usd"))
        model_cost = _float_or_none(row.get("selected_expected_total_cost_usd"))
        if model_fees is not None and model_fees > 0 and model_il is not None and model_cost is not None:
            realized_fee_proxy = realized_net + model_il + model_cost
            ratio = realized_fee_proxy / model_fees
            if math.isfinite(ratio):
                fee_ratios.append(ratio)

        if is_rebalance_row:
            drag = max(0.0, residual)
            if math.isfinite(drag):
                rebalance_drags.append(drag)

    sample_count = len(residual_rows)
    if sample_count <= 0:
        return snapshot

    residual_values = [row[2] for row in residual_rows]
    residual_values = _winsorize(residual_values)
    weights = _ewma_weights(len(residual_values))
    model_rmse_observed = _weighted_rmse(residual_values, weights)
    warmup_blend = _clamp(sample_count / float(max(min_samples, 1)), 0.0, 1.0)
    rmse_prior = snapshot.model_rmse_usd
    model_rmse_usd = (
        rmse_prior if model_rmse_observed is None else (1.0 - warmup_blend) * rmse_prior + warmup_blend * model_rmse_observed
    )

    fee_sample_count = len(fee_ratios)
    fee_multiplier = snapshot.fee_realism_multiplier
    if fee_sample_count > 0:
        ratio_values = _winsorize(fee_ratios)
        ratio_weights = _ewma_weights(len(ratio_values))
        ratio_observed = _weighted_mean(ratio_values, ratio_weights)
        if ratio_observed is not None:
            ratio_blend = _clamp(fee_sample_count / float(max(min_samples, 1)), 0.0, 1.0)
            blended_ratio = (1.0 - ratio_blend) * snapshot.fee_realism_multiplier + ratio_blend * ratio_observed
            fee_multiplier = _clamp(blended_ratio, float(fee_realism_min), float(fee_realism_max))

    rebalance_sample_count = len(rebalance_drags)
    rebalance_drag = snapshot.rebalance_drag_usd
    if rebalance_sample_count > 0:
        drag_values = _winsorize(rebalance_drags)
        drag_weights = _ewma_weights(len(drag_values))
        drag_observed = _weighted_mean(drag_values, drag_weights)
        if drag_observed is not None:
            drag_blend = _clamp(rebalance_sample_count / float(max(min_samples, 1)), 0.0, 1.0)
            blended_drag = (1.0 - drag_blend) * snapshot.rebalance_drag_usd + drag_blend * drag_observed
            rebalance_drag = _clamp(blended_drag, float(rebalance_drag_min_usd), float(rebalance_drag_max_usd))

    return CalibrationSnapshot(
        fee_realism_multiplier=float(fee_multiplier),
        rebalance_drag_usd=float(rebalance_drag),
        model_rmse_usd=max(0.0, float(model_rmse_usd)),
        sample_count=sample_count,
        fee_sample_count=fee_sample_count,
        rebalance_sample_count=rebalance_sample_count,
        mode="ready" if sample_count >= int(min_samples) else "warmup",
    )
