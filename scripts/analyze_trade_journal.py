#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze AI liquidity optimizer trade journal")
    parser.add_argument(
        "--path",
        default="state/trade_journal.jsonl",
        help="Path to trade journal JSONL (default: state/trade_journal.jsonl)",
    )
    parser.add_argument("--last", type=int, default=0, help="Only analyze the last N journal records")
    parser.add_argument("--since-hours", type=float, default=0.0, help="Only include records newer than N hours")
    parser.add_argument(
        "--csv-out",
        default="",
        help="Optional output path for per-cycle CSV (includes timing and modeled P/L columns)",
    )
    parser.add_argument("--show-errors", type=int, default=5, help="Show up to N recent errors")
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


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


def _entry_timestamp(row: dict[str, Any]) -> datetime | None:
    return _parse_timestamp(row.get("decision_timestamp")) or _parse_timestamp(row.get("recorded_at"))


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / float(len(values))


def _rmse(values: list[float]) -> float | None:
    if not values:
        return None
    return (sum(v * v for v in values) / float(len(values))) ** 0.5


def _sum(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values))


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: float | None, decimals: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{decimals}f}"


def _select_number(*values: Any) -> float | None:
    for value in values:
        parsed = _float_or_none(value)
        if parsed is not None:
            return parsed
    return None


def _bucket_minutes(minutes: float | None) -> str:
    if minutes is None:
        return "n/a"
    if minutes < 10.0:
        return "<10m"
    if minutes < 30.0:
        return "10-30m"
    if minutes < 60.0:
        return "30-60m"
    if minutes < 120.0:
        return "60-120m"
    return "120m+"


def _build_cycle_rows(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed: list[tuple[datetime | None, int, dict[str, Any]]] = [
        (_entry_timestamp(row), i, row) for i, row in enumerate(decisions)
    ]
    indexed.sort(key=lambda item: (item[0] or datetime.min.replace(tzinfo=timezone.utc), item[1]))

    out: list[dict[str, Any]] = []
    last_cycle_ts: datetime | None = None
    last_rebalance_signal_ts: datetime | None = None
    last_rebalance_success_ts: datetime | None = None
    cumulative_selected_net = 0.0
    cumulative_selected_fees = 0.0
    cumulative_selected_il = 0.0
    cumulative_selected_cost = 0.0
    cumulative_onchain_delta_total = 0.0

    for ts, _, row in indexed:
        selected_action = str(row.get("selected_action") or "unknown")
        execution_status = str(row.get("execution_status") or "unknown")
        rebalance_should = bool(row.get("rebalance_should"))

        selected_fees = _select_number(
            row.get("selected_expected_fees_usd"),
            row.get("expected_fees_usd"),
        )
        selected_il = _select_number(
            row.get("selected_expected_il_usd"),
            row.get("expected_il_usd"),
        )
        selected_cost = _select_number(
            row.get("selected_expected_total_cost_usd"),
            row.get("expected_total_cost_usd"),
        )
        onchain_delta_total = _float_or_none(row.get("onchain_delta_total_usd_est"))
        onchain_delta_sol = _float_or_none(row.get("onchain_delta_sol"))
        onchain_delta_usdc = _float_or_none(row.get("onchain_delta_usdc"))
        onchain_delta_wallet_total = _float_or_none(row.get("onchain_delta_wallet_total_usd_est"))
        onchain_delta_position_total = _float_or_none(row.get("onchain_delta_position_total_usd_est"))
        onchain_total_usd = _float_or_none(row.get("onchain_total_usd_est"))
        onchain_sol_balance = _float_or_none(row.get("onchain_sol_balance"))
        onchain_usdc_balance = _float_or_none(row.get("onchain_usdc_balance"))
        onchain_wallet_total_usd = _float_or_none(row.get("onchain_wallet_total_usd_est"))
        onchain_position_total_usd = _float_or_none(row.get("onchain_position_total_usd_est"))
        selected_net = _select_number(
            row.get("selected_expected_net_usd"),
            row.get("ev_selected_usd"),
        )
        selected_raw_net = _select_number(
            row.get("selected_raw_expected_net_usd"),
            row.get("raw_expected_net_usd"),
        )
        selected_adjusted_net = _select_number(
            row.get("selected_adjusted_expected_net_usd"),
            row.get("adjusted_expected_net_usd"),
        )
        selected_net_from_components = _select_number(row.get("selected_expected_net_from_components_usd"))
        if selected_net is None:
            selected_net = selected_net_from_components
        if selected_raw_net is None:
            selected_raw_net = selected_net_from_components
        if selected_adjusted_net is None:
            selected_adjusted_net = selected_net
        if selected_net is None and selected_fees is not None and selected_il is not None:
            selected_net = selected_fees - selected_il - (selected_cost or 0.0)
        if selected_raw_net is None and selected_net is not None:
            selected_raw_net = selected_net
        if selected_adjusted_net is None and selected_net is not None:
            selected_adjusted_net = selected_net
        model_onchain_error = None
        if selected_net is not None and onchain_delta_total is not None:
            model_onchain_error = selected_net - onchain_delta_total
        has_equity_snapshot_components = (onchain_wallet_total_usd is not None) or (onchain_position_total_usd is not None)
        has_equity_delta_components = (onchain_delta_wallet_total is not None) or (onchain_delta_position_total is not None)

        mins_since_prev_cycle: float | None = None
        mins_since_rebalance_signal: float | None = None
        mins_since_rebalance_success: float | None = None
        if ts is not None and last_cycle_ts is not None:
            mins_since_prev_cycle = (ts - last_cycle_ts).total_seconds() / 60.0
        if ts is not None and last_rebalance_signal_ts is not None:
            mins_since_rebalance_signal = (ts - last_rebalance_signal_ts).total_seconds() / 60.0
        if ts is not None and last_rebalance_success_ts is not None:
            mins_since_rebalance_success = (ts - last_rebalance_success_ts).total_seconds() / 60.0

        if selected_net is not None:
            cumulative_selected_net += selected_net
        if selected_fees is not None:
            cumulative_selected_fees += selected_fees
        if selected_il is not None:
            cumulative_selected_il += selected_il
        if selected_cost is not None:
            cumulative_selected_cost += selected_cost
        if onchain_delta_total is not None:
            cumulative_onchain_delta_total += onchain_delta_total

        out_row = {
            "timestamp": (ts.isoformat() if ts else str(row.get("decision_timestamp") or row.get("recorded_at") or "")),
            "selected_action": selected_action,
            "rebalance_should": rebalance_should,
            "execution_status": execution_status,
            "rebalance_reason": row.get("rebalance_reason"),
            "gate_mode": row.get("gate_mode"),
            "pool_address": row.get("pool_address"),
            "spot_price_sol_usdc": _float_or_none(row.get("spot_price_sol_usdc")),
            "target_range_lower": _float_or_none(row.get("target_range_lower")),
            "target_range_upper": _float_or_none(row.get("target_range_upper")),
            "target_width_pct": _float_or_none(row.get("target_width_pct")),
            "ev_best_usd": _float_or_none(row.get("ev_best_usd")),
            "ev_hold_usd": _float_or_none(row.get("ev_hold_usd")),
            "ev_idle_usd": _float_or_none(row.get("ev_idle_usd")),
            "ev_delta_usd": _float_or_none(row.get("ev_delta_usd")),
            "lifecycle_pnl_usd": _float_or_none(row.get("lifecycle_pnl_usd")),
            "lifecycle_pnl_pct": _float_or_none(row.get("lifecycle_pnl_pct")),
            "lifecycle_open_position_usd": _float_or_none(row.get("lifecycle_open_position_usd")),
            "lifecycle_current_position_usd": _float_or_none(row.get("lifecycle_current_position_usd")),
            "policy_profit_triggered": bool(row.get("policy_profit_triggered")),
            "policy_loss_recovery_hold_triggered": bool(row.get("policy_loss_recovery_hold_triggered")),
            "policy_loss_trend_cut_triggered": bool(row.get("policy_loss_trend_cut_triggered")),
            "policy_recovery_prob": _float_or_none(row.get("policy_recovery_prob")),
            "policy_trend_prob": _float_or_none(row.get("policy_trend_prob")),
            "policy_loss_exit_breach_count": _float_or_none(row.get("policy_loss_exit_breach_count")),
            "policy_selected_override": str(row.get("policy_selected_override") or "none"),
            "selected_expected_fees_usd": selected_fees,
            "selected_expected_il_usd": selected_il,
            "selected_expected_total_cost_usd": selected_cost,
            "selected_expected_net_usd": selected_net,
            "selected_raw_expected_net_usd": selected_raw_net,
            "selected_adjusted_expected_net_usd": selected_adjusted_net,
            "selected_fee_realism_multiplier": _float_or_none(row.get("selected_fee_realism_multiplier")),
            "selected_execution_drag_usd": _float_or_none(row.get("selected_execution_drag_usd")),
            "selected_uncertainty_penalty_usd": _float_or_none(row.get("selected_uncertainty_penalty_usd")),
            "effective_min_delta_usd": _float_or_none(row.get("effective_min_delta_usd")),
            "realism_model_rmse_usd": _float_or_none(row.get("realism_model_rmse_usd")),
            "realism_rebalance_drag_usd": _float_or_none(row.get("realism_rebalance_drag_usd")),
            "realism_fee_realism_multiplier": _float_or_none(row.get("realism_fee_realism_multiplier")),
            "il_state_penalty_usd": _select_number(row.get("selected_il_state_penalty_usd"), row.get("il_state_penalty_usd")),
            "il_baseline_usd": _select_number(row.get("selected_il_baseline_usd"), row.get("il_baseline_usd")),
            "hold_out_of_range_bps": _float_or_none(row.get("hold_out_of_range_bps")),
            "hold_out_of_range_cycles": _float_or_none(row.get("hold_out_of_range_cycles")),
            "utilization_ratio": _float_or_none(row.get("utilization_ratio")),
            "hold_utilization_ratio": _float_or_none(row.get("hold_utilization_ratio")),
            "mins_since_prev_cycle": mins_since_prev_cycle,
            "mins_since_rebalance_signal": mins_since_rebalance_signal,
            "mins_since_rebalance_success": mins_since_rebalance_success,
            "rebalance_timing_bucket": _bucket_minutes(mins_since_rebalance_success),
            "execution_tx_count": _float_or_none(row.get("execution_tx_count")),
            "onchain_total_usd_est": onchain_total_usd,
            "onchain_sol_balance": onchain_sol_balance,
            "onchain_usdc_balance": onchain_usdc_balance,
            "onchain_wallet_total_usd_est": onchain_wallet_total_usd,
            "onchain_position_total_usd_est": onchain_position_total_usd,
            "onchain_delta_total_usd_est": onchain_delta_total,
            "onchain_delta_sol": onchain_delta_sol,
            "onchain_delta_usdc": onchain_delta_usdc,
            "onchain_delta_wallet_total_usd_est": onchain_delta_wallet_total,
            "onchain_delta_position_total_usd_est": onchain_delta_position_total,
            "onchain_snapshot_has_equity_components": has_equity_snapshot_components,
            "onchain_delta_has_equity_components": has_equity_delta_components,
            "model_minus_onchain_delta_usd": model_onchain_error,
            "model_abs_error_vs_onchain_delta_usd": abs(model_onchain_error) if model_onchain_error is not None else None,
            "cumulative_selected_net_usd": cumulative_selected_net,
            "cumulative_selected_fees_usd": cumulative_selected_fees,
            "cumulative_selected_il_usd": cumulative_selected_il,
            "cumulative_selected_cost_usd": cumulative_selected_cost,
            "cumulative_onchain_delta_total_usd_est": cumulative_onchain_delta_total,
        }
        out.append(out_row)

        if rebalance_should and ts is not None:
            last_rebalance_signal_ts = ts
        if selected_action == "rebalance" and execution_status == "success" and ts is not None:
            last_rebalance_success_ts = ts
        if ts is not None:
            last_cycle_ts = ts
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    args = _parse_args()
    journal_path = Path(args.path)
    rows = _load_jsonl(journal_path)
    if not rows:
        print(f"No journal data found at {journal_path}")
        return 1

    if args.last and args.last > 0:
        rows = rows[-args.last :]

    if args.since_hours and args.since_hours > 0:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=float(args.since_hours))
        rows = [row for row in rows if (_entry_timestamp(row) and _entry_timestamp(row) >= cutoff)]

    if not rows:
        print("No journal entries in requested window.")
        return 1

    decisions = [row for row in rows if row.get("event_type") == "decision"]
    errors = [row for row in rows if row.get("event_type") == "cycle_error"]
    cycle_rows = _build_cycle_rows(decisions)

    timestamps = [ts for ts in (_entry_timestamp(row) for row in rows) if ts is not None]
    start_ts = min(timestamps).isoformat() if timestamps else "n/a"
    end_ts = max(timestamps).isoformat() if timestamps else "n/a"

    action_counts = Counter(str(row.get("selected_action") or "unknown") for row in decisions)
    reason_counts = Counter(str(row.get("rebalance_reason") or "none") for row in decisions)
    status_counts = Counter(str(row.get("execution_status") or "unknown") for row in decisions)
    error_categories = Counter(str(row.get("error_category") or "unknown") for row in errors)

    ev_best_values = [_float_or_none(row.get("ev_best_usd")) for row in decisions]
    ev_hold_values = [_float_or_none(row.get("ev_hold_usd")) for row in decisions]
    ev_delta_values = [_float_or_none(row.get("ev_delta_usd")) for row in decisions]
    fee_values = [_float_or_none(row.get("expected_fees_usd")) for row in decisions]
    il_values = [_float_or_none(row.get("expected_il_usd")) for row in decisions]
    il_pen_values = [_float_or_none(row.get("il_state_penalty_usd")) for row in decisions]

    ev_best_values = [v for v in ev_best_values if v is not None]
    ev_hold_values = [v for v in ev_hold_values if v is not None]
    ev_delta_values = [v for v in ev_delta_values if v is not None]
    fee_values = [v for v in fee_values if v is not None]
    il_values = [v for v in il_values if v is not None]
    il_pen_values = [v for v in il_pen_values if v is not None]

    rebalance_attempts = [row for row in decisions if bool(row.get("rebalance_should"))]
    successful_execs = [row for row in decisions if str(row.get("execution_status")) == "success"]
    blocked_execs = [row for row in decisions if str(row.get("execution_status")) == "blocked_before_execute"]
    attempted_no_tx = [row for row in decisions if str(row.get("execution_status")) == "attempted_no_tx"]

    out_of_range_values = [
        _float_or_none(row.get("hold_out_of_range_bps"))
        for row in decisions
        if _float_or_none(row.get("hold_out_of_range_bps")) is not None
    ]
    out_of_range_cycles = len([v for v in out_of_range_values if v is not None and v > 0.0])
    out_of_range_ratio = (out_of_range_cycles / float(len(out_of_range_values))) if out_of_range_values else None

    selected_net_values = [v for v in (_float_or_none(row.get("selected_expected_net_usd")) for row in cycle_rows) if v is not None]
    selected_raw_net_values = [v for v in (_float_or_none(row.get("selected_raw_expected_net_usd")) for row in cycle_rows) if v is not None]
    selected_adjusted_net_values = [v for v in (_float_or_none(row.get("selected_adjusted_expected_net_usd")) for row in cycle_rows) if v is not None]
    selected_fee_values = [v for v in (_float_or_none(row.get("selected_expected_fees_usd")) for row in cycle_rows) if v is not None]
    selected_il_values = [v for v in (_float_or_none(row.get("selected_expected_il_usd")) for row in cycle_rows) if v is not None]
    selected_cost_values = [v for v in (_float_or_none(row.get("selected_expected_total_cost_usd")) for row in cycle_rows) if v is not None]
    realism_fee_mult_values = [v for v in (_float_or_none(row.get("realism_fee_realism_multiplier")) for row in cycle_rows) if v is not None]
    realism_drag_values = [v for v in (_float_or_none(row.get("realism_rebalance_drag_usd")) for row in cycle_rows) if v is not None]
    realism_rmse_values = [v for v in (_float_or_none(row.get("realism_model_rmse_usd")) for row in cycle_rows) if v is not None]
    effective_min_delta_values = [v for v in (_float_or_none(row.get("effective_min_delta_usd")) for row in cycle_rows) if v is not None]
    onchain_delta_total_values = [v for v in (_float_or_none(row.get("onchain_delta_total_usd_est")) for row in cycle_rows) if v is not None]
    onchain_delta_sol_values = [v for v in (_float_or_none(row.get("onchain_delta_sol")) for row in cycle_rows) if v is not None]
    onchain_delta_usdc_values = [v for v in (_float_or_none(row.get("onchain_delta_usdc")) for row in cycle_rows) if v is not None]
    onchain_delta_wallet_total_values = [v for v in (_float_or_none(row.get("onchain_delta_wallet_total_usd_est")) for row in cycle_rows) if v is not None]
    onchain_delta_position_total_values = [v for v in (_float_or_none(row.get("onchain_delta_position_total_usd_est")) for row in cycle_rows) if v is not None]
    cumulative_selected_net = None
    cumulative_onchain_delta_total = None
    if cycle_rows:
        last_cycle = cycle_rows[-1]
        cumulative_selected_net = _float_or_none(last_cycle.get("cumulative_selected_net_usd"))
        cumulative_onchain_delta_total = _float_or_none(last_cycle.get("cumulative_onchain_delta_total_usd_est"))

    calibration_pairs_all: list[tuple[float, float, str]] = []
    calibration_pairs_equity: list[tuple[float, float, str]] = []
    for row in cycle_rows:
        expected = _float_or_none(row.get("selected_expected_net_usd"))
        realized = _float_or_none(row.get("onchain_delta_total_usd_est"))
        if expected is None or realized is None:
            continue
        pair = (expected, realized, str(row.get("selected_action") or "unknown"))
        calibration_pairs_all.append(pair)
        if bool(row.get("onchain_delta_has_equity_components")):
            calibration_pairs_equity.append(pair)
    calibration_pairs = calibration_pairs_equity if calibration_pairs_equity else calibration_pairs_all
    calibration_label = "equity_components" if calibration_pairs_equity else "legacy_total_only"
    calibration_errors = [exp - real for exp, real, _ in calibration_pairs]
    calibration_abs_errors = [abs(v) for v in calibration_errors]
    calibration_sign_matches = [
        1.0 if ((exp >= 0 and real >= 0) or (exp < 0 and real < 0)) else 0.0
        for exp, real, _ in calibration_pairs
    ]
    by_action: dict[str, list[float]] = {}
    for exp, real, action in calibration_pairs:
        by_action.setdefault(action, []).append(exp - real)

    # Policy trigger slices and override calibration
    policy_trigger_slices = {
        "profit_triggered": "policy_profit_triggered",
        "loss_recovery_hold_triggered": "policy_loss_recovery_hold_triggered",
        "loss_trend_cut_triggered": "policy_loss_trend_cut_triggered",
    }
    policy_slice_stats: dict[str, dict[str, float | int | None]] = {}
    for label, key in policy_trigger_slices.items():
        flagged = [row for row in cycle_rows if bool(row.get(key))]
        unflagged = [row for row in cycle_rows if not bool(row.get(key))]
        flagged_onchain = [_float_or_none(row.get("onchain_delta_total_usd_est")) for row in flagged]
        unflagged_onchain = [_float_or_none(row.get("onchain_delta_total_usd_est")) for row in unflagged]
        flagged_onchain = [v for v in flagged_onchain if v is not None]
        unflagged_onchain = [v for v in unflagged_onchain if v is not None]
        policy_slice_stats[label] = {
            "flagged_n": len(flagged),
            "unflagged_n": len(unflagged),
            "flagged_avg_onchain_delta": _mean(flagged_onchain),
            "unflagged_avg_onchain_delta": _mean(unflagged_onchain),
        }

    override_calibration: dict[str, dict[str, float | int | None]] = {}
    for override in sorted({str(row.get("policy_selected_override") or "none") for row in cycle_rows}):
        rows_for_override = [row for row in cycle_rows if str(row.get("policy_selected_override") or "none") == override]
        pairs = []
        for row in rows_for_override:
            expected = _float_or_none(row.get("selected_expected_net_usd"))
            realized = _float_or_none(row.get("onchain_delta_total_usd_est"))
            if expected is None or realized is None:
                continue
            pairs.append((expected, realized))
        errs = [exp - real for exp, real in pairs]
        override_calibration[override] = {
            "n": len(rows_for_override),
            "cal_n": len(pairs),
            "mean_model": _mean([exp for exp, _ in pairs]) if pairs else None,
            "mean_onchain": _mean([real for _, real in pairs]) if pairs else None,
            "mean_err": _mean(errs) if errs else None,
            "mae": _mean([abs(v) for v in errs]) if errs else None,
        }

    forced_loss_cut_rows = [
        row
        for row in cycle_rows
        if str(row.get("policy_selected_override") or "none") in {"loss_cut_rotate", "loss_cut_idle"}
    ]
    forced_loss_cut_onchain = [
        _float_or_none(row.get("onchain_delta_total_usd_est"))
        for row in forced_loss_cut_rows
    ]
    forced_loss_cut_onchain = [v for v in forced_loss_cut_onchain if v is not None]

    adjusted_pairs = []
    for row in cycle_rows:
        adj = _float_or_none(row.get("selected_adjusted_expected_net_usd"))
        realized = _float_or_none(row.get("onchain_delta_total_usd_est"))
        if adj is None or realized is None:
            continue
        adjusted_pairs.append((adj, realized))
    adjusted_pairs.sort(key=lambda pair: pair[0])

    adjusted_quantile_stats: list[tuple[str, int, float | None, float | None]] = []
    if adjusted_pairs:
        n = len(adjusted_pairs)
        splits = [0, n // 4, n // 2, (3 * n) // 4, n]
        labels = ["Q1", "Q2", "Q3", "Q4"]
        for idx, label in enumerate(labels):
            lo = splits[idx]
            hi = splits[idx + 1]
            chunk = adjusted_pairs[lo:hi]
            if not chunk:
                adjusted_quantile_stats.append((label, 0, None, None))
                continue
            adjusted_quantile_stats.append(
                (
                    label,
                    len(chunk),
                    _mean([v for v, _ in chunk]),
                    _mean([v for _, v in chunk]),
                )
            )

    adjusted_positive = [pair for pair in adjusted_pairs if pair[0] > 0.0]
    adjusted_false_positive_count = len([1 for _adj, realized in adjusted_positive if realized < 0.0])
    adjusted_false_positive_rate = (
        adjusted_false_positive_count / float(len(adjusted_positive))
        if adjusted_positive
        else None
    )

    raw_pairs = []
    for row in cycle_rows:
        raw_net = _float_or_none(row.get("selected_raw_expected_net_usd"))
        realized = _float_or_none(row.get("onchain_delta_total_usd_est"))
        if raw_net is None or realized is None:
            continue
        raw_pairs.append((raw_net, realized))
    raw_positive = [pair for pair in raw_pairs if pair[0] > 0.0]
    raw_false_positive_count = len([1 for _raw, realized in raw_positive if realized < 0.0])
    raw_false_positive_rate = (raw_false_positive_count / float(len(raw_positive))) if raw_positive else None

    pre_realism_bucket = []
    post_realism_bucket = []
    for row in cycle_rows:
        realized = _float_or_none(row.get("onchain_delta_total_usd_est"))
        raw_net = _float_or_none(row.get("selected_raw_expected_net_usd"))
        adj_net = _float_or_none(row.get("selected_adjusted_expected_net_usd"))
        eff_min = _float_or_none(row.get("effective_min_delta_usd"))
        if realized is None:
            continue
        if raw_net is not None and raw_net > 0.0:
            pre_realism_bucket.append(realized)
        if adj_net is not None and eff_min is not None and adj_net > 0.0 and adj_net >= eff_min:
            post_realism_bucket.append(realized)

    # Timing-vs-fee/IL breakdown by time since last successful rebalance
    timing_groups: dict[str, dict[str, list[float]]] = {}
    for row in cycle_rows:
        bucket = str(row.get("rebalance_timing_bucket") or "n/a")
        metrics = timing_groups.setdefault(
            bucket,
            {
                "fees": [],
                "il": [],
                "net": [],
                "onchain_delta_total": [],
            },
        )
        fee_v = _float_or_none(row.get("selected_expected_fees_usd"))
        il_v = _float_or_none(row.get("selected_expected_il_usd"))
        net_v = _float_or_none(row.get("selected_expected_net_usd"))
        if fee_v is not None:
            metrics["fees"].append(fee_v)
        if il_v is not None:
            metrics["il"].append(il_v)
        if net_v is not None:
            metrics["net"].append(net_v)
        onchain_v = _float_or_none(row.get("onchain_delta_total_usd_est"))
        if onchain_v is not None:
            metrics["onchain_delta_total"].append(onchain_v)

    print("Trade Journal Summary")
    print(f"path: {journal_path}")
    print(f"window: {start_ts} -> {end_ts}")
    print(f"records: total={len(rows)} decisions={len(decisions)} errors={len(errors)}")
    print(
        "actions: "
        + ", ".join(f"{k}={v}" for k, v in sorted(action_counts.items()))
        if action_counts
        else "actions: none"
    )
    print(
        "execution: "
        f"rebalance_signals={len(rebalance_attempts)} "
        f"success={len(successful_execs)} "
        f"blocked={len(blocked_execs)} "
        f"attempted_no_tx={len(attempted_no_tx)}"
    )
    if status_counts:
        print("execution_status: " + ", ".join(f"{k}={v}" for k, v in sorted(status_counts.items())))
    print(
        "ev: "
        f"mean_best={_fmt(_mean(ev_best_values), 4)} "
        f"mean_hold={_fmt(_mean(ev_hold_values), 4)} "
        f"mean_delta={_fmt(_mean(ev_delta_values), 4)}"
    )
    print(
        "selected_model_pnl: "
        f"sum_net={_fmt(_sum(selected_net_values), 4)} "
        f"sum_raw_net={_fmt(_sum(selected_raw_net_values), 4)} "
        f"sum_adjusted_net={_fmt(_sum(selected_adjusted_net_values), 4)} "
        f"sum_fees={_fmt(_sum(selected_fee_values), 4)} "
        f"sum_il={_fmt(_sum(selected_il_values), 4)} "
        f"sum_cost={_fmt(_sum(selected_cost_values), 4)} "
        f"cum_net_last={_fmt(cumulative_selected_net, 4)}"
    )
    if realism_fee_mult_values or realism_drag_values or realism_rmse_values:
        print(
            "realism_calibration_drift: "
            f"fee_mult_avg={_fmt(_mean(realism_fee_mult_values), 4)} "
            f"fee_mult_min={_fmt(min(realism_fee_mult_values) if realism_fee_mult_values else None, 4)} "
            f"fee_mult_max={_fmt(max(realism_fee_mult_values) if realism_fee_mult_values else None, 4)} "
            f"drag_avg={_fmt(_mean(realism_drag_values), 4)} "
            f"drag_max={_fmt(max(realism_drag_values) if realism_drag_values else None, 4)} "
            f"rmse_avg={_fmt(_mean(realism_rmse_values), 4)} "
            f"rmse_max={_fmt(max(realism_rmse_values) if realism_rmse_values else None, 4)} "
            f"effective_min_delta_avg={_fmt(_mean(effective_min_delta_values), 4)}"
        )
    print(
        "onchain_balance_delta: "
        f"sum_total_usd_est={_fmt(_sum(onchain_delta_total_values), 4)} "
        f"sum_wallet_total_usd_est={_fmt(_sum(onchain_delta_wallet_total_values), 4)} "
        f"sum_position_total_usd_est={_fmt(_sum(onchain_delta_position_total_values), 4)} "
        f"sum_sol={_fmt(_sum(onchain_delta_sol_values), 6)} "
        f"sum_usdc={_fmt(_sum(onchain_delta_usdc_values), 4)} "
        f"cum_total_usd_est_last={_fmt(cumulative_onchain_delta_total, 4)}"
    )
    print(
        "model_vs_onchain_calibration: "
        f"mode={calibration_label} "
        f"n={len(calibration_pairs)} "
        f"mean_model={_fmt(_mean([exp for exp, _, _ in calibration_pairs]), 4)} "
        f"mean_onchain={_fmt(_mean([real for _, real, _ in calibration_pairs]), 4)} "
        f"mean_error_model_minus_onchain={_fmt(_mean(calibration_errors), 4)} "
        f"mae={_fmt(_mean(calibration_abs_errors), 4)} "
        f"rmse={_fmt(_rmse(calibration_errors), 4)} "
        f"sign_hit_rate={_fmt(_mean(calibration_sign_matches), 3)}"
    )
    if by_action:
        print(
            "model_vs_onchain_by_action: "
            + ", ".join(
                f"{action}:n={len(errs)} mean_err={_fmt(_mean(errs), 4)} mae={_fmt(_mean([abs(v) for v in errs]), 4)}"
                for action, errs in sorted(by_action.items())
            )
        )
    if calibration_pairs_equity and calibration_pairs_all and len(calibration_pairs_equity) != len(calibration_pairs_all):
        print(
            "model_vs_onchain_calibration_coverage: "
            f"equity_rows={len(calibration_pairs_equity)} total_rows_with_onchain={len(calibration_pairs_all)}"
        )
    if policy_slice_stats:
        print(
            "policy_trigger_slices: "
            + ", ".join(
                (
                    f"{label}:flagged_n={stats['flagged_n']} "
                    f"flagged_avg_onchain={_fmt(stats['flagged_avg_onchain_delta'], 4)} "
                    f"unflagged_avg_onchain={_fmt(stats['unflagged_avg_onchain_delta'], 4)}"
                )
                for label, stats in sorted(policy_slice_stats.items())
            )
        )
    if override_calibration:
        print(
            "policy_override_calibration: "
            + ", ".join(
                (
                    f"{override}:n={int(stats['n'] or 0)} cal_n={int(stats['cal_n'] or 0)} "
                    f"model={_fmt(stats['mean_model'], 4)} onchain={_fmt(stats['mean_onchain'], 4)} "
                    f"mean_err={_fmt(stats['mean_err'], 4)} mae={_fmt(stats['mae'], 4)}"
                )
                for override, stats in sorted(override_calibration.items())
            )
        )
    print(
        "forced_loss_cut_outcomes: "
        f"n={len(forced_loss_cut_rows)} "
        f"avg_onchain_delta_total={_fmt(_mean(forced_loss_cut_onchain), 4)}"
    )
    if adjusted_quantile_stats:
        print(
            "adjusted_ev_quantiles: "
            + ", ".join(
                f"{label}:n={count} avg_adj={_fmt(avg_adj, 4)} avg_realized={_fmt(avg_realized, 4)}"
                for label, count, avg_adj, avg_realized in adjusted_quantile_stats
            )
        )
    print(
        "adjusted_ev_false_positive_rate: "
        f"adjusted_pos_n={len(adjusted_positive)} adjusted_fp_rate={_fmt(adjusted_false_positive_rate, 4)} "
        f"raw_pos_n={len(raw_positive)} raw_fp_rate={_fmt(raw_false_positive_rate, 4)}"
    )
    print(
        "rebalance_outcome_pre_post_realism: "
        f"pre_filter_n={len(pre_realism_bucket)} pre_filter_avg_realized={_fmt(_mean(pre_realism_bucket), 4)} "
        f"post_filter_n={len(post_realism_bucket)} post_filter_avg_realized={_fmt(_mean(post_realism_bucket), 4)}"
    )
    print(
        "fees_vs_il: "
        f"sum_fees={_fmt(_sum(fee_values), 4)} "
        f"sum_il={_fmt(_sum(il_values), 4)} "
        f"sum_il_pen={_fmt(_sum(il_pen_values), 4)}"
    )
    print(
        "hold_range: "
        f"oor_ratio={_fmt(out_of_range_ratio, 3)} "
        f"avg_oor_bps={_fmt(_mean([v for v in out_of_range_values if v is not None]), 2)}"
    )
    if reason_counts:
        top_reasons = ", ".join(f"{k}={v}" for k, v in reason_counts.most_common(6))
        print(f"rebalance_reasons: {top_reasons}")
    if error_categories:
        top_err = ", ".join(f"{k}={v}" for k, v in error_categories.most_common(6))
        print(f"error_categories: {top_err}")

    if timing_groups:
        print("timing_vs_fee_il:")
        for bucket in ["<10m", "10-30m", "30-60m", "60-120m", "120m+", "n/a"]:
            group = timing_groups.get(bucket)
            if not group:
                continue
            count = max(len(group["net"]), len(group["fees"]), len(group["il"]), len(group["onchain_delta_total"]))
            print(
                f"  {bucket}: n={count} "
                f"avg_fee={_fmt(_mean(group['fees']), 4)} "
                f"avg_il={_fmt(_mean(group['il']), 4)} "
                f"avg_net={_fmt(_mean(group['net']), 4)} "
                f"avg_onchain_delta_total={_fmt(_mean(group['onchain_delta_total']), 4)}"
            )

    if args.show_errors > 0 and errors:
        print("\nRecent errors:")
        for row in errors[-args.show_errors :]:
            ts = row.get("recorded_at")
            cat = row.get("error_category")
            msg = str(row.get("error_message") or "")
            if len(msg) > 180:
                msg = msg[:177] + "..."
            print(f"- {ts} [{cat}] {msg}")

    if args.csv_out:
        csv_path = Path(args.csv_out)
        _write_csv(csv_path, cycle_rows)
        print(f"\nWrote cycle CSV: {csv_path} (rows={len(cycle_rows)})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
