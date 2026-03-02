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
    max_single_bin: float | None = None,
    max_top3: float | None = None,
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
    pre_guard_weights = _normalize(w_raw)
    weights, guard_lambda = _apply_soft_concentration_guard(
        pre_guard_weights,
        max_single_bin=max_single_bin,
        max_top3=max_top3,
    )

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
            "pre_guard_weights": pre_guard_weights,
            "concentration_guard_lambda": guard_lambda,
            "top1_share": max(weights) if weights else None,
            "top3_share": sum(sorted(weights, reverse=True)[:3]) if weights else None,
        },
    )


def compute_time_decayed_occupancy_from_percentiles(
    *,
    bin_edges: list[float],
    prediction_percentiles: SynthPredictionPercentilesSnapshot,
    tau_half_minutes: int = 90,
    max_horizon_minutes: int | None = None,
    normalize_steps: bool = True,
    normalize_output: bool = True,
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
        t_minutes = step_index * prediction_percentiles.step_minutes
        if max_horizon_minutes is not None and t_minutes > max_horizon_minutes:
            break
        parsed_quantiles: list[tuple[float, float]] = []
        for p_raw, v_raw in row.items():
            try:
                p = float(p_raw)
                q = float(v_raw)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(p) or not math.isfinite(q):
                continue
            if p < 0.0 or q <= 0.0:
                continue
            parsed_quantiles.append((p, q))

        if len(parsed_quantiles) < 2:
            continue
        parsed_quantiles.sort(key=lambda t: t[0])
        percentile_scale = _detect_percentile_key_scale(parsed_quantiles)
        if percentile_scale is None:
            continue
        sorted_quantiles = parsed_quantiles
        if len(sorted_quantiles) < 2:
            continue

        step_occupancy = [0.0] * num_bins
        for j in range(len(sorted_quantiles) - 1):
            p_lo, q_lo = sorted_quantiles[j]
            p_hi, q_hi = sorted_quantiles[j + 1]
            if p_hi <= p_lo:
                continue
            mass = (p_hi - p_lo) / percentile_scale
            if mass <= 0:
                continue
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
        decay = math.exp(-ln2 * (t_minutes / float(tau_half_minutes)))
        total_decay += decay
        for i in range(num_bins):
            if normalize_steps:
                accum[i] += decay * (step_occupancy[i] / step_total)
            else:
                accum[i] += decay * step_occupancy[i]

    if total_decay <= 0:
        return None
    if normalize_steps:
        return _normalize(accum) if normalize_output else [v / total_decay for v in accum]
    averaged = [v / total_decay for v in accum]
    return _normalize(averaged) if normalize_output else averaged


def compute_active_occupancy_metrics_from_percentiles(
    *,
    bin_edges: list[float],
    prediction_percentiles: SynthPredictionPercentilesSnapshot,
    tau_half_minutes: int,
    max_horizon_minutes: int,
    bin_weights: list[float] | None = None,
) -> dict[str, float | list[float] | None] | None:
    occupancy = compute_time_decayed_occupancy_from_percentiles(
        bin_edges=bin_edges,
        prediction_percentiles=prediction_percentiles,
        tau_half_minutes=tau_half_minutes,
        max_horizon_minutes=max_horizon_minutes,
        normalize_steps=False,
        normalize_output=False,
    )
    if occupancy is None:
        return None
    range_active_occupancy = _clamp(sum(max(0.0, occ) for occ in occupancy), 0.0, 1.0)
    weight_alignment_score = None
    if bin_weights is not None and len(bin_weights) == len(occupancy) and occupancy:
        weight_alignment_score = _clamp(
            sum(max(0.0, float(w)) * max(0.0, occ) for w, occ in zip(bin_weights, occupancy)),
            0.0,
            1.0,
        )
    return {
        "occupancy_by_bin": occupancy,
        "range_active_occupancy": range_active_occupancy,
        "weight_alignment_score": weight_alignment_score,
    }


def compute_weighted_active_occupancy_from_percentiles(
    *,
    bin_edges: list[float],
    bin_weights: list[float],
    prediction_percentiles: SynthPredictionPercentilesSnapshot,
    tau_half_minutes: int,
    max_horizon_minutes: int,
) -> float | None:
    metrics = compute_active_occupancy_metrics_from_percentiles(
        bin_edges=bin_edges,
        prediction_percentiles=prediction_percentiles,
        tau_half_minutes=tau_half_minutes,
        max_horizon_minutes=max_horizon_minutes,
        bin_weights=bin_weights,
    )
    if metrics is None:
        return None
    value = metrics.get("weight_alignment_score")
    return float(value) if value is not None else None


def compute_exact_sdk_bin_odds_weight_plan(
    *,
    range_lower: float,
    range_upper: float,
    sdk_bin_prices_sol_usdc: list[float],
    prediction_percentiles: SynthPredictionPercentilesSnapshot | None,
    lp_probabilities: SynthLpProbabilitiesSnapshot | None,
    ev_horizon_minutes: int = 15,
    tau_half_minutes: int = 15,
    beta: float = 0.9,
    eps: float = 1e-6,
    max_single_bin: float | None = None,
    max_top3: float | None = None,
) -> WeightedBinPlan:
    if not sdk_bin_prices_sol_usdc:
        raise ValueError("compute_exact_sdk_bin_odds_weight_plan requires sdk_bin_prices_sol_usdc")
    edges = _derive_edges_from_bin_centers(
        range_lower=range_lower,
        range_upper=range_upper,
        bin_centers=sdk_bin_prices_sol_usdc,
    )
    num_bins = len(edges) - 1
    terminal_cdf_points = 0
    fallback_reason: str | None = None
    mass_in_range = 0.0
    used_prediction_percentiles = False
    occupancy = None
    source = "uniform"

    base: list[float] | None = None
    if prediction_percentiles is not None:
        metrics = compute_active_occupancy_metrics_from_percentiles(
            bin_edges=edges,
            prediction_percentiles=prediction_percentiles,
            tau_half_minutes=max(1, tau_half_minutes),
            max_horizon_minutes=max(1, ev_horizon_minutes),
            bin_weights=None,
        )
        if metrics is not None:
            occ = metrics.get("occupancy_by_bin")
            if isinstance(occ, list) and len(occ) == num_bins:
                occupancy = [max(0.0, float(x)) for x in occ]
                range_occ = float(metrics.get("range_active_occupancy") or 0.0)
                if range_occ > 0:
                    base = occupancy
                    mass_in_range = _clamp(range_occ, 0.0, 1.0)
                    used_prediction_percentiles = True
                    source = "prediction_percentiles_exact_15m"

    terminal_mass = None
    if base is None and lp_probabilities is not None:
        cdf = build_terminal_cdf_from_lp_probabilities(lp_probabilities)
        terminal_cdf_points = len(cdf.prices)
        terminal_mass = _terminal_mass_per_bin(cdf, edges)
        terminal_sum = sum(max(0.0, v) for v in terminal_mass)
        if terminal_sum > 0:
            base = terminal_mass
            mass_in_range = _clamp(terminal_sum, 0.0, 1.0)
            source = "lp_probabilities_terminal_mass_exact_bins"
        else:
            fallback_reason = "zero_terminal_mass_exact_bins"

    if base is None:
        base = [1.0] * num_bins
        if fallback_reason is None:
            fallback_reason = "uniform_exact_bins_fallback"
        source = "uniform_exact_bins"

    smooth_beta = beta if beta > 0 else 1.0
    smooth_eps = eps if eps >= 0 else 0.0
    smoothed = [(max(0.0, v) + smooth_eps) ** smooth_beta for v in base]
    pre_guard_weights = _normalize(smoothed)
    weights, guard_lambda = _apply_soft_concentration_guard(
        pre_guard_weights,
        max_single_bin=max_single_bin,
        max_top3=max_top3,
    )

    diagnostics = BinWeightingDiagnostics(
        mass_in_range=mass_in_range,
        used_prediction_percentiles=used_prediction_percentiles,
        fallback_reason=fallback_reason,
        confidence_factor=1.0,
        t_frac=1.0,
        entropy=_entropy(weights),
        terminal_cdf_points=terminal_cdf_points,
        num_bins=num_bins,
        binning_mode="exact_sdk_bins",
    )
    return WeightedBinPlan(
        range_lower=range_lower,
        range_upper=range_upper,
        bin_edges=edges,
        weights=weights,
        diagnostics=diagnostics,
        distribution_components={
            "source": source,
            "smoothing_beta": smooth_beta,
            "smoothing_eps": smooth_eps,
            "time_decayed_occupancy": occupancy,
            "terminal_mass": terminal_mass,
            "base_distribution": base,
            "concentrated_distribution": weights,
            "pre_guard_weights": pre_guard_weights,
            "concentration_guard_lambda": guard_lambda,
            "top1_share": max(weights) if weights else None,
            "top3_share": sum(sorted(weights, reverse=True)[:3]) if weights else None,
            "execution_weight_objective": "odds_15m_exact",
        },
    )


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


def _derive_edges_from_bin_centers(*, range_lower: float, range_upper: float, bin_centers: list[float]) -> list[float]:
    if not bin_centers:
        raise ValueError("bin_centers cannot be empty")
    centers = [float(x) for x in bin_centers]
    for i, c in enumerate(centers):
        if c <= 0:
            raise ValueError(f"bin_centers[{i}] must be > 0")
        if i > 0 and c <= centers[i - 1]:
            raise ValueError("bin_centers must be strictly increasing")
    lo = float(min(range_lower, range_upper))
    hi = float(max(range_lower, range_upper))
    if lo <= 0 or hi <= lo:
        raise ValueError("Invalid range bounds for deriving edges from centers")

    if len(centers) == 1:
        return _validate_and_normalize_bin_edges([lo, hi])

    edges = [lo]
    for i in range(len(centers) - 1):
        c0 = centers[i]
        c1 = centers[i + 1]
        mid = math.sqrt(c0 * c1)
        edges.append(mid)
    edges.append(hi)

    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = max(edges[i - 1] * (1.0 + 1e-12), edges[i - 1] + 1e-12)
    return _validate_and_normalize_bin_edges(edges)


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


def _apply_soft_concentration_guard(
    weights: list[float],
    *,
    max_single_bin: float | None,
    max_top3: float | None,
) -> tuple[list[float], float]:
    if not weights:
        return [], 0.0
    if len(weights) <= 1:
        return _normalize(weights), 0.0
    if max_single_bin is None and max_top3 is None:
        return _normalize(weights), 0.0

    m1 = float(max_single_bin) if max_single_bin is not None else 1.0
    m3 = float(max_top3) if max_top3 is not None else 1.0
    m1 = _clamp(m1, 0.0, 1.0)
    m3 = _clamp(m3, 0.0, 1.0)
    if m3 < m1:
        m3 = m1

    base = _normalize(weights)
    if _concentration_ok(base, m1, m3):
        return base, 0.0

    n = len(base)
    uniform = [1.0 / n] * n
    lo = 0.0
    hi = 1.0
    best = uniform
    for _ in range(40):
        mid = (lo + hi) / 2.0
        mixed = _normalize([(1.0 - mid) * base[i] + mid * uniform[i] for i in range(n)])
        if _concentration_ok(mixed, m1, m3):
            best = mixed
            hi = mid
        else:
            lo = mid
    return best, hi


def _concentration_ok(weights: list[float], max_single_bin: float, max_top3: float) -> bool:
    top1 = max(weights) if weights else 0.0
    top3 = sum(sorted(weights, reverse=True)[:3]) if weights else 0.0
    return top1 <= max_single_bin + 1e-12 and top3 <= max_top3 + 1e-12


def _detect_percentile_key_scale(sorted_quantiles: list[tuple[float, float]], eps: float = 1e-9) -> float | None:
    if not sorted_quantiles:
        return None
    keys = [p for p, _ in sorted_quantiles]
    if any(keys[i] > keys[i + 1] + eps for i in range(len(keys) - 1)):
        return None
    max_key = max(keys)
    if max_key <= 1.0 + eps:
        return 1.0
    if max_key <= 100.0 + eps:
        return 100.0
    return None


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _entropy(weights: list[float]) -> float:
    total = 0.0
    for w in weights:
        if w > 0:
            total -= w * math.log(w)
    return total
