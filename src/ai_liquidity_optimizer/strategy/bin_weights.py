from __future__ import annotations

import math

from ai_liquidity_optimizer.compat import dataclass
from ai_liquidity_optimizer.models import (
    BinWeightingConfig,
    BinWeightingDiagnostics,
    MeteoraPoolSnapshot,
    SynthLpBoundForecast,
    SynthLpProbabilitiesSnapshot,
    SynthPredictionPercentilesSnapshot,
    WeightedBinPlan,
)
from ai_liquidity_optimizer.strategy.scoring import horizon_to_minutes


@dataclass(slots=True)
class TerminalCdf:
    prices: list[float]
    cdf_values: list[float]

    def evaluate(self, price: float) -> float:
        if price <= 0 or not self.prices:
            return 0.0
        if price <= self.prices[0]:
            return self.cdf_values[0]
        if price >= self.prices[-1]:
            return self.cdf_values[-1]

        ln_x = math.log(price)
        for idx in range(len(self.prices) - 1):
            x0 = self.prices[idx]
            x1 = self.prices[idx + 1]
            if x0 <= price <= x1:
                if x0 <= 0 or x1 <= 0:
                    # Should not happen after filtering, but keep a safe fallback.
                    f0 = self.cdf_values[idx]
                    f1 = self.cdf_values[idx + 1]
                    t_linear = (price - x0) / max(x1 - x0, 1e-12)
                    return _clamp(f0 + t_linear * (f1 - f0), 0.0, 1.0)
                ln_x0 = math.log(x0)
                ln_x1 = math.log(x1)
                if abs(ln_x1 - ln_x0) < 1e-12:
                    return self.cdf_values[idx]
                t = (ln_x - ln_x0) / (ln_x1 - ln_x0)
                f0 = self.cdf_values[idx]
                f1 = self.cdf_values[idx + 1]
                return _clamp(f0 + t * (f1 - f0), 0.0, 1.0)
        return self.cdf_values[-1]


def build_terminal_cdf_from_lp_probabilities(snapshot: SynthLpProbabilitiesSnapshot) -> TerminalCdf:
    raw_pairs: list[tuple[float, float]] = []
    for point in snapshot.points:
        if point.price <= 0:
            continue
        below = point.probability_below
        above = point.probability_above
        cdf = None
        if below is not None and above is not None:
            cdf = 0.5 * (below + (1.0 - above))
        elif below is not None:
            cdf = below
        elif above is not None:
            cdf = 1.0 - above
        if cdf is None:
            continue
        raw_pairs.append((float(point.price), _clamp(float(cdf), 0.0, 1.0)))

    if not raw_pairs:
        raise RuntimeError("No valid lp-probabilities points available to build terminal CDF")

    raw_pairs.sort(key=lambda t: t[0])
    merged_prices: list[float] = []
    merged_cdf: list[float] = []
    for price, cdf in raw_pairs:
        if merged_prices and abs(price - merged_prices[-1]) <= 1e-12:
            merged_cdf[-1] = max(merged_cdf[-1], cdf)
        else:
            merged_prices.append(price)
            merged_cdf.append(cdf)

    monotone_cdf: list[float] = []
    running = 0.0
    for cdf in merged_cdf:
        running = max(running, _clamp(cdf, 0.0, 1.0))
        monotone_cdf.append(running)

    return TerminalCdf(prices=merged_prices, cdf_values=monotone_cdf)


def derive_mvp_bin_edges_for_range(
    pool: MeteoraPoolSnapshot,
    range_lower: float,
    range_upper: float,
    target_bin_count: int = 24,
) -> tuple[list[float], str]:
    """Derive price bin edges inside the selected range.

    Uses Meteora `bin_step` from the pool API when available. Falls back to equal log-spacing.
    """
    if range_lower <= 0 or range_upper <= range_lower:
        raise ValueError("Invalid range bounds for bin edge derivation")

    bin_step_bps = _extract_bin_step_bps(pool)
    if bin_step_bps is not None and bin_step_bps > 0:
        ratio = 1.0 + (bin_step_bps / 10_000.0)
        if ratio > 1.0:
            anchor = pool.current_price if pool.current_price > 0 else math.sqrt(range_lower * range_upper)
            if anchor <= 0:
                anchor = math.sqrt(range_lower * range_upper)
            edges = _derive_ratio_aligned_edges(range_lower, range_upper, anchor, ratio)
            if len(edges) >= 2:
                return edges, "meteora_bin_step"

    return _derive_log_spaced_edges(range_lower, range_upper, target_bin_count), "log_spaced_fallback"


def compute_bin_weights_for_range(
    *,
    forecast: SynthLpBoundForecast,
    horizon: str,
    bin_edges: list[float],
    current_price: float,
    lp_probabilities: SynthLpProbabilitiesSnapshot,
    prediction_percentiles: SynthPredictionPercentilesSnapshot | None = None,
    config: BinWeightingConfig | None = None,
    binning_mode: str = "unknown",
) -> WeightedBinPlan:
    cfg = config or BinWeightingConfig()
    edges = _validate_and_normalize_bin_edges(bin_edges)
    num_bins = len(edges) - 1
    if num_bins < 1:
        raise ValueError("compute_bin_weights_for_range requires at least one bin")

    cdf = build_terminal_cdf_from_lp_probabilities(lp_probabilities)
    terminal_mass = _terminal_mass_per_bin(cdf, edges)
    mass_in_range = sum(terminal_mass)
    fallback_reason: str | None = None

    if mass_in_range >= cfg.low_mass_threshold:
        m_norm = _normalize(terminal_mass)
    else:
        m_norm = None
        fallback_reason = "low_mass_in_range"

    proximity = _proximity_prior(edges, current_price=current_price, range_lower=forecast.lower_bound, range_upper=forecast.upper_bound, cfg=cfg)
    occupancy = None
    used_prediction_percentiles = False
    if prediction_percentiles is not None:
        occupancy = compute_time_decayed_occupancy_from_percentiles(
            bin_edges=edges,
            prediction_percentiles=prediction_percentiles,
            tau_half_minutes=cfg.tau_half_minutes,
        )
        if occupancy is not None:
            used_prediction_percentiles = True

    if fallback_reason is not None:
        base = proximity[:]
    elif used_prediction_percentiles and occupancy is not None and m_norm is not None:
        base = _normalize(
            [
                cfg.path_weight * occupancy[i]
                + cfg.terminal_mass_weight_with_path * m_norm[i]
                + cfg.proximity_weight_with_path * proximity[i]
                for i in range(num_bins)
            ]
        )
    else:
        if m_norm is None:
            base = proximity[:]
        else:
            base = _normalize(
                [
                    cfg.terminal_mass_weight_no_path * m_norm[i]
                    + cfg.proximity_weight_no_path * proximity[i]
                    for i in range(num_bins)
                ]
            )

    concentrated = _normalize([(x + cfg.eps) ** cfg.alpha for x in base])

    horizon_minutes = horizon_to_minutes(horizon)
    t_frac = _clamp(forecast.expected_time_in_interval_minutes / max(float(horizon_minutes), 1.0), 0.0, 1.0)
    conf = _clamp(0.25 + 0.75 * math.sqrt(max(0.0, forecast.probability_to_stay_in_interval * t_frac)), 0.25, 1.0)

    uniform = [1.0 / num_bins] * num_bins
    flattened = [conf * concentrated[i] + (1.0 - conf) * uniform[i] for i in range(num_bins)]
    w_raw = [(1.0 - cfg.final_floor_blend) * flattened[i] + cfg.final_floor_blend * uniform[i] for i in range(num_bins)]
    weights = _normalize(w_raw)

    diagnostics = BinWeightingDiagnostics(
        mass_in_range=mass_in_range,
        used_prediction_percentiles=used_prediction_percentiles,
        fallback_reason=fallback_reason,
        confidence_factor=conf,
        t_frac=t_frac,
        entropy=_entropy(weights),
        terminal_cdf_points=len(cdf.prices),
        num_bins=num_bins,
        binning_mode=binning_mode,
    )

    return WeightedBinPlan(
        range_lower=forecast.lower_bound,
        range_upper=forecast.upper_bound,
        bin_edges=edges,
        weights=weights,
        diagnostics=diagnostics,
        distribution_components={
            "terminal_mass": terminal_mass,
            "terminal_mass_normalized": m_norm,
            "proximity_prior": proximity,
            "time_decayed_occupancy": occupancy,
            "base_distribution": base,
            "concentrated_distribution": concentrated,
        },
    )


def compute_time_decayed_occupancy_from_percentiles(
    *,
    bin_edges: list[float],
    prediction_percentiles: SynthPredictionPercentilesSnapshot,
    tau_half_minutes: int = 90,
) -> list[float] | None:
    edges = _validate_and_normalize_bin_edges(bin_edges)
    num_bins = len(edges) - 1
    if num_bins < 1:
        return None
    if tau_half_minutes <= 0:
        tau_half_minutes = 90

    accum = [0.0] * num_bins
    total_decay = 0.0
    ln2 = math.log(2.0)

    for step_index, row in enumerate(prediction_percentiles.percentiles_by_step):
        if not isinstance(row, dict) or len(row) < 2:
            continue
        sorted_quantiles = sorted((float(p), float(v)) for p, v in row.items() if 0.0 <= float(p) <= 100.0 and float(v) > 0.0)
        if len(sorted_quantiles) < 2:
            continue

        step_occupancy = [0.0] * num_bins
        for j in range(len(sorted_quantiles) - 1):
            p_lo, q_lo = sorted_quantiles[j]
            p_hi, q_hi = sorted_quantiles[j + 1]
            if p_hi <= p_lo:
                continue
            mass = (p_hi - p_lo) / 100.0
            lo = min(q_lo, q_hi)
            hi = max(q_lo, q_hi)
            if hi <= 0:
                continue
            if hi - lo <= 1e-12:
                idx = _find_bin_index_for_price(edges, hi)
                if idx is not None:
                    step_occupancy[idx] += mass
                continue
            for i in range(num_bins):
                overlap = max(0.0, min(edges[i + 1], hi) - max(edges[i], lo))
                if overlap > 0:
                    step_occupancy[i] += mass * (overlap / (hi - lo))

        step_total = sum(step_occupancy)
        if step_total <= 0:
            continue
        t_minutes = step_index * prediction_percentiles.step_minutes
        decay = math.exp(-ln2 * (t_minutes / float(tau_half_minutes)))
        total_decay += decay
        for i in range(num_bins):
            accum[i] += decay * (step_occupancy[i] / step_total)

    if total_decay <= 0:
        return None
    return _normalize(accum)


def _terminal_mass_per_bin(cdf: TerminalCdf, edges: list[float]) -> list[float]:
    masses: list[float] = []
    for i in range(len(edges) - 1):
        lo = edges[i]
        hi = edges[i + 1]
        masses.append(max(0.0, cdf.evaluate(hi) - cdf.evaluate(lo)))
    return masses


def _proximity_prior(
    edges: list[float],
    *,
    current_price: float,
    range_lower: float,
    range_upper: float,
    cfg: BinWeightingConfig,
) -> list[float]:
    num_bins = len(edges) - 1
    if num_bins <= 0:
        return []
    if current_price <= 0:
        return [1.0 / num_bins] * num_bins
    if range_lower <= 0 or range_upper <= range_lower:
        sigma_price = cfg.sigma_min
    else:
        sigma_price = max(cfg.sigma_min, cfg.sigma_range_scale * math.log(range_upper / range_lower))

    values = []
    for i in range(num_bins):
        center = math.sqrt(max(edges[i], 1e-12) * max(edges[i + 1], 1e-12))
        values.append(math.exp(-abs(math.log(center / current_price)) / sigma_price))
    return _normalize(values)


def _validate_and_normalize_bin_edges(bin_edges: list[float]) -> list[float]:
    if len(bin_edges) < 2:
        raise ValueError("bin_edges must contain at least two values")
    edges = [float(x) for x in bin_edges]
    for i, x in enumerate(edges):
        if x <= 0:
            raise ValueError(f"bin_edges[{i}] must be > 0")
        if i > 0 and x <= edges[i - 1]:
            raise ValueError("bin_edges must be strictly increasing")
    return edges


def _extract_bin_step_bps(pool: MeteoraPoolSnapshot) -> float | None:
    raw = pool.raw if isinstance(pool.raw, dict) else {}
    pool_config = raw.get("pool_config") if isinstance(raw.get("pool_config"), dict) else {}
    candidates = [
        raw.get("bin_step"),
        raw.get("binStep"),
        raw.get("bin_step_bps"),
        raw.get("binStepBps"),
        pool_config.get("bin_step") if isinstance(pool_config, dict) else None,
        pool_config.get("binStep") if isinstance(pool_config, dict) else None,
    ]
    for candidate in candidates:
        try:
            if candidate is not None:
                value = float(candidate)
                if value > 0:
                    return value
        except (TypeError, ValueError):
            continue
    return None


def _derive_ratio_aligned_edges(range_lower: float, range_upper: float, anchor: float, ratio: float) -> list[float]:
    if ratio <= 1.0:
        return [range_lower, range_upper]
    log_r = math.log(ratio)
    if abs(log_r) < 1e-12 or anchor <= 0:
        return [range_lower, range_upper]

    n_lo = math.floor(math.log(range_lower / anchor) / log_r)
    n_hi = math.ceil(math.log(range_upper / anchor) / log_r)
    grid = [anchor * (ratio ** n) for n in range(n_lo, n_hi + 1)]
    grid = [x for x in grid if range_lower < x < range_upper and x > 0]
    edges = [range_lower, *grid, range_upper]
    return _unique_sorted(edges)


def _derive_log_spaced_edges(range_lower: float, range_upper: float, target_bin_count: int) -> list[float]:
    bins = max(1, min(int(target_bin_count), 256))
    if bins == 1:
        return [range_lower, range_upper]
    ln_lo = math.log(range_lower)
    ln_hi = math.log(range_upper)
    edges = [math.exp(ln_lo + (ln_hi - ln_lo) * (i / bins)) for i in range(bins + 1)]
    edges[0] = range_lower
    edges[-1] = range_upper
    return _unique_sorted(edges)


def _unique_sorted(values: list[float]) -> list[float]:
    out: list[float] = []
    for x in sorted(values):
        if not out or abs(x - out[-1]) > 1e-12:
            out.append(float(x))
    if len(out) < 2:
        raise ValueError("Need at least two unique edges")
    return out


def _find_bin_index_for_price(edges: list[float], price: float) -> int | None:
    if price < edges[0] or price > edges[-1]:
        return None
    for i in range(len(edges) - 1):
        if edges[i] <= price < edges[i + 1]:
            return i
    if abs(price - edges[-1]) <= 1e-12:
        return len(edges) - 2
    return None


def _normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    total = sum(max(0.0, v) for v in values)
    if total <= 0:
        return [1.0 / len(values)] * len(values)
    return [max(0.0, v) / total for v in values]


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _entropy(weights: list[float]) -> float:
    total = 0.0
    for w in weights:
        if w > 0:
            total -= w * math.log(w)
    return total
