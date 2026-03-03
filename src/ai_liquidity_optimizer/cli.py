from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from ai_liquidity_optimizer.clients.meteora import MeteoraDlmmApiClient
from ai_liquidity_optimizer.clients.synth import SynthInsightsClient
from ai_liquidity_optimizer.config import load_settings
from ai_liquidity_optimizer.execution.dry_run import DryRunExecutor
from ai_liquidity_optimizer.execution.meteora_node_bridge import MeteoraNodeBridgeExecutor
from ai_liquidity_optimizer.orchestrator import OptimizerOrchestrator
from ai_liquidity_optimizer.state_store import JsonStateStore
from ai_liquidity_optimizer.strategy.ev import EvLpScorer
from ai_liquidity_optimizer.strategy.scoring import StrategyScorer


LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI Liquidity Optimizer (Synth + Meteora DLMM MVP)")
    parser.add_argument("command", choices=["run"], nargs="?", default="run")
    parser.add_argument("--once", action="store_true", help="Run a single cycle and exit")
    parser.add_argument("--env-file", default=".env", help="Path to .env file (default: .env)")
    parser.add_argument("--log-level", default=None, help="Override LOG_LEVEL")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    env_file = Path(args.env_file)
    if not env_file.is_absolute():
        env_file = repo_root / env_file
    settings = load_settings(repo_root=repo_root, env_file=env_file)
    if args.log_level:
        settings.log_level = args.log_level.upper()

    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    synth_client = SynthInsightsClient(settings.synth_base_url, settings.synth_api_key or "")
    meteora_client = MeteoraDlmmApiClient(settings.meteora_api_base_url)
    scorer = StrategyScorer(min_stay_probability=settings.min_stay_probability)
    ev_scorer = EvLpScorer(
        ev_horizon_minutes=settings.ev_horizon_minutes,
        rebalance_cost_usd=settings.rebalance_cost_usd,
        pool_switch_extra_cost_usd=settings.pool_switch_extra_cost_usd,
        min_ev_improvement_usd=settings.min_ev_improvement_usd,
        ev_percentile_decay_half_life_minutes=settings.ev_percentile_decay_half_life_minutes,
        ev_concentration_gamma=settings.ev_concentration_gamma,
        ev_concentration_min=settings.ev_concentration_min,
        ev_concentration_max=settings.ev_concentration_max,
        ev_capture_kappa=settings.ev_capture_kappa,
        ev_capture_min=settings.ev_capture_min,
        ev_capture_max=settings.ev_capture_max,
        ev_capture_eps=settings.ev_capture_eps,
        ev_oor_penalty_enabled=settings.ev_oor_penalty_enabled,
        ev_oor_penalize_hold_only=settings.ev_oor_penalize_hold_only,
        ev_oor_deadband_bps=settings.ev_oor_deadband_bps,
        ev_oor_ref_bps=settings.ev_oor_ref_bps,
        ev_oor_base_penalty_fraction_15m=settings.ev_oor_base_penalty_fraction_15m,
        ev_oor_max_penalty_fraction_15m=settings.ev_oor_max_penalty_fraction_15m,
        ev_oor_persistence_step=settings.ev_oor_persistence_step,
        ev_oor_persistence_cap_cycles=settings.ev_oor_persistence_cap_cycles,
        ev_il_drift_alpha=settings.ev_il_drift_alpha,
        ev_il_oor_beta=settings.ev_il_oor_beta,
        ev_il_onesided_gamma=settings.ev_il_onesided_gamma,
        ev_il_persistence_delta=settings.ev_il_persistence_delta,
        ev_il_mult_min=settings.ev_il_mult_min,
        ev_il_mult_max=settings.ev_il_mult_max,
        ev_il_drift_ref_bps=settings.ev_il_drift_ref_bps,
        ev_il_drift_horizon_minutes=settings.ev_il_drift_horizon_minutes,
        min_stay_probability=settings.min_stay_probability,
    )
    state_store = JsonStateStore(settings.state_path)

    if settings.executor == "dry-run":
        executor = DryRunExecutor()
    else:
        executor = MeteoraNodeBridgeExecutor(
            repo_root=repo_root,
            rpc_url=settings.solana_rpc_url or "",
            private_key_b58=settings.solana_private_key_b58 or "",
            liquidity_mode=settings.meteora_liquidity_mode,
            max_custom_weight_position_bins=settings.max_custom_weight_position_bins,
            synth_weight_active_bin_floor_bps=settings.synth_weight_active_bin_floor_bps,
            synth_weight_max_bin_bps_per_side=settings.synth_weight_max_bin_bps_per_side,
        )

    orchestrator = OptimizerOrchestrator(
        settings=settings,
        synth_client=synth_client,
        meteora_client=meteora_client,
        scorer=scorer,
        ev_scorer=ev_scorer,
        executor=executor,
        state_store=state_store,
    )

    if args.once:
        result = orchestrator.run_once()
        save_paths = _persist_once_result(result=result, state_path=settings.state_path)
        LOGGER.info(_summarize_once_result(result))
        LOGGER.info(
            "Saved full run result for later review: latest=%s history=%s",
            save_paths["latest"],
            save_paths["history"],
        )
        return 0

    orchestrator.run_forever()
    return 0

def _persist_once_result(result: dict[str, Any], state_path: Path) -> dict[str, str]:
    state_dir = state_path.parent
    runs_dir = state_dir / "run_results"
    runs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = str(result.get("timestamp") or "unknown")
    safe_ts = (
        timestamp.replace(":", "-")
        .replace("/", "-")
        .replace(" ", "_")
        .replace("+", "_plus_")
    )

    latest_path = state_dir / "last_once_result.json"
    history_path = runs_dir / f"once_{safe_ts}.json"

    serialized = json.dumps(result, indent=2, sort_keys=True)
    latest_path.write_text(serialized, encoding="utf-8")
    history_path.write_text(serialized, encoding="utf-8")

    return {"latest": str(latest_path), "history": str(history_path)}


def _summarize_once_result(result: dict[str, Any]) -> str:
    if bool(result.get("ev_mode")) or isinstance(result.get("ev_best_candidate"), dict):
        return _summarize_once_result_ev(result)
    return _summarize_once_result_proxy(result)


def _summarize_once_result_ev(result: dict[str, Any]) -> str:
    pool = result.get("selected_pool") if isinstance(result.get("selected_pool"), dict) else {}
    if not pool:
        pool = result.get("pool") if isinstance(result.get("pool"), dict) else {}
    chosen = result.get("chosen") if isinstance(result.get("chosen"), dict) else {}
    rebalance = result.get("rebalance") if isinstance(result.get("rebalance"), dict) else {}
    gate = result.get("rebalance_gate") if isinstance(result.get("rebalance_gate"), dict) else {}
    best = result.get("ev_best_candidate") if isinstance(result.get("ev_best_candidate"), dict) else {}
    best_components = best.get("ev_components") if isinstance(best.get("ev_components"), dict) else {}
    hold = result.get("ev_current_hold") if isinstance(result.get("ev_current_hold"), dict) else {}
    active_position = result.get("active_position") if isinstance(result.get("active_position"), dict) else {}
    cost_model = result.get("cost_model") if isinstance(result.get("cost_model"), dict) else {}
    execution_config = result.get("execution_config") if isinstance(result.get("execution_config"), dict) else {}
    execution_plan = result.get("execution_bin_weight_plan") if isinstance(result.get("execution_bin_weight_plan"), dict) else {}
    bin_plan = execution_plan if execution_plan else (result.get("bin_weight_plan") if isinstance(result.get("bin_weight_plan"), dict) else {})
    diagnostics = bin_plan.get("diagnostics") if isinstance(bin_plan.get("diagnostics"), dict) else {}
    top_bins = _format_top_bins(bin_plan=bin_plan, limit=5)

    ev_best = best.get("ev_15m_usd", chosen.get("ev_15m_usd", chosen.get("score")))
    ev_best_raw = result.get("ev_best_raw_usd", chosen.get("raw_ev_15m_usd"))
    ev_best_adjusted = result.get("ev_best_adjusted_usd", chosen.get("adjusted_ev_15m_usd"))
    ev_hold = hold.get("ev_15m_usd")
    ev_hold_raw = result.get("ev_hold_raw_usd")
    ev_hold_adjusted = result.get("ev_hold_adjusted_usd")
    ev_delta = result.get("ev_delta_usd")
    hold_components = hold.get("ev_components") if isinstance(hold.get("ev_components"), dict) else {}
    lifecycle_pnl = result.get("lifecycle_pnl_usd")
    lifecycle_pct = result.get("lifecycle_pnl_pct")
    policy_override = result.get("policy_selected_override")
    realism = result.get("realism") if isinstance(result.get("realism"), dict) else {}
    realism_snapshot = realism.get("snapshot") if isinstance(realism.get("snapshot"), dict) else {}

    return (
        "RUN_ONCE_EV "
        f"ts={result.get('timestamp')} "
        f"h={result.get('horizon')} "
        f"pool={pool.get('name')}[{str(pool.get('address') or '')[:8]}]@{_fmt_num(pool.get('current_price'), 4)} "
        f"spot={_fmt_num(pool.get('current_price'), 4)} "
        f"active_range=[{_fmt_num(active_position.get('lower_price'), 4)},{_fmt_num(active_position.get('upper_price'), 4)}] "
        f"range=[{_fmt_num(chosen.get('lower_bound'), 4)},{_fmt_num(chosen.get('upper_bound'), 4)}] "
        f"w={_fmt_num(chosen.get('width_pct'), 2)}% "
        f"EVraw={_fmt_num(ev_best_raw, 4)} "
        f"EVadj={_fmt_num(ev_best_adjusted, 4)} "
        f"EV_used={_fmt_num(ev_best, 4)} "
        f"hold_raw={_fmt_num(ev_hold_raw, 4)} "
        f"hold_adj={_fmt_num(ev_hold_adjusted, 4)} "
        f"hold_used={_fmt_num(ev_hold, 4)} "
        f"delta={_fmt_num(ev_delta, 4)} "
        f"min_delta_eff={_fmt_num(result.get('effective_min_delta_usd'), 4)} "
        f"action={result.get('selected_action')} "
        f"lifecycle_pnl={_fmt_num(lifecycle_pnl, 4)} "
        f"lifecycle_pct={_fmt_num((float(lifecycle_pct) * 100.0) if lifecycle_pct is not None else None, 2)}% "
        f"policy={policy_override} "
        f"fees={_fmt_num(best_components.get('expected_fees_usd'), 4)} "
        f"fees_raw={_fmt_num(best_components.get('raw_expected_fees_usd'), 4)} "
        f"fees_adj={_fmt_num(best_components.get('adjusted_expected_fees_usd'), 4)} "
        f"il={_fmt_num(best_components.get('expected_il_usd'), 4)} "
        f"il_base={_fmt_num(best_components.get('il_baseline_usd'), 4)} "
        f"il_pen={_fmt_num(best_components.get('il_state_penalty_usd'), 4)} "
        f"drag={_fmt_num(best_components.get('execution_drag_usd'), 4)} "
        f"unc={_fmt_num(best_components.get('uncertainty_penalty_usd'), 4)} "
        f"il_mult={_fmt_num(best_components.get('il_multiplier'), 3)} "
        f"drift_bps={_fmt_num(best_components.get('p50_drift_bps'), 1)} "
        f"oor_p={_fmt_num(best_components.get('out_of_range_prob_15m'), 3)} "
        f"one_side={_fmt_num(best_components.get('one_sided_break_prob'), 3)} "
        f"cost={_fmt_num(best_components.get('rebalance_cost_usd'), 4)}+{_fmt_num(best_components.get('pool_switch_extra_cost_usd'), 4)} "
        f"occ_range={_fmt_num(best_components.get('range_active_occupancy_15m', best_components.get('active_occupancy_15m')), 3)} "
        f"align={_fmt_num(best_components.get('weight_alignment_score'), 3)} "
        f"util={_fmt_num(best_components.get('utilization_ratio'), 3)} "
        f"util_hold={_fmt_num(hold_components.get('utilization_ratio'), 3)} "
        f"hold_out_bps={_fmt_num(hold_components.get('out_of_range_bps'), 2)} "
        f"hold_oor_cycles={hold_components.get('out_of_range_cycles')} "
        f"hold_exact_bounds={str(str(hold_components.get('baseline_mode') or '').startswith('current_hold_exact_bounds')).lower()} "
        f"capture={_fmt_num(best_components.get('fee_capture_factor'), 3)} "
        f"conc={_fmt_num(best_components.get('concentration_factor'), 3)} "
        f"fee_rate15m={_fmt_num(best_components.get('fee_rate_15m_fraction'), 6)} "
        f"cost_model={cost_model.get('source')}:{cost_model.get('applied_total_fee_lamports')} "
        f"weight_obj={execution_config.get('synth_weight_objective')} "
        f"score_obj={result.get('scoring_objective_used')} "
        f"rescored={result.get('rescored_candidate_count')} "
        f"bins={diagnostics.get('num_bins', '?')} "
        f"path={diagnostics.get('used_prediction_percentiles', False)} "
        f"fallback={diagnostics.get('fallback_reason')} "
        f"mass={_fmt_num(diagnostics.get('mass_in_range'), 4)} "
        f"top1={_fmt_num(result.get('top1_weight_share'), 3)} "
        f"top3={_fmt_num(result.get('top3_weight_share'), 3)} "
        f"switch={best.get('pool_switch')} "
        f"gate={gate.get('gate_mode')} ev:{gate.get('ev_threshold_passed')} structural:{gate.get('structural_change_passed')} "
        f"protective_breach={gate.get('protective_breach_count')} "
        f"rebalance={rebalance.get('should_rebalance')} close_idle={rebalance.get('should_close_to_idle')}({rebalance.get('reason')}) "
        f"realism={realism.get('use_adjusted_for_decisions')}:{realism_snapshot.get('mode')}/{realism_snapshot.get('sample_count')} "
        f"fee_mult={_fmt_num(realism_snapshot.get('fee_realism_multiplier'), 3)} "
        f"drag_cal={_fmt_num(realism_snapshot.get('rebalance_drag_usd'), 4)} "
        f"rmse={_fmt_num(realism_snapshot.get('model_rmse_usd'), 4)} "
        f"top_bins={top_bins}"
    )


def _summarize_once_result_proxy(result: dict[str, Any]) -> str:
    pool = result.get("pool") if isinstance(result.get("pool"), dict) else {}
    chosen = result.get("chosen") if isinstance(result.get("chosen"), dict) else {}
    rebalance = result.get("rebalance") if isinstance(result.get("rebalance"), dict) else {}
    bin_plan = result.get("bin_weight_plan") if isinstance(result.get("bin_weight_plan"), dict) else {}
    diagnostics = bin_plan.get("diagnostics") if isinstance(bin_plan.get("diagnostics"), dict) else {}

    top_bins = _format_top_bins(bin_plan=bin_plan, limit=5)

    return (
        "RUN_ONCE "
        f"ts={result.get('timestamp')} "
        f"h={result.get('horizon')} "
        f"pool={pool.get('name')}@{_fmt_num(pool.get('current_price'), 4)} "
        f"range=[{_fmt_num(chosen.get('lower_bound'), 4)},{_fmt_num(chosen.get('upper_bound'), 4)}] "
        f"w={_fmt_num(chosen.get('width_pct'), 2)}% "
        f"score={_fmt_num(chosen.get('score'), 6)} "
        f"p_stay={_fmt_num(chosen.get('probability_to_stay_in_interval'), 4)} "
        f"t_in={_fmt_num(chosen.get('expected_time_in_interval_minutes'), 1)}m "
        f"IL={_fmt_num(chosen.get('expected_impermanent_loss'), 6)} "
        f"bins={diagnostics.get('num_bins', '?')} "
        f"path={diagnostics.get('used_prediction_percentiles', False)} "
        f"fallback={diagnostics.get('fallback_reason')} "
        f"mass={_fmt_num(diagnostics.get('mass_in_range'), 4)} "
        f"rebalance={rebalance.get('should_rebalance')}({rebalance.get('reason')}) "
        f"top_bins={top_bins}"
    )


def _format_top_bins(bin_plan: dict[str, Any], limit: int = 5) -> str:
    edges = bin_plan.get("bin_edges")
    weights = bin_plan.get("weights")
    if not isinstance(edges, list) or not isinstance(weights, list):
        return "n/a"
    if len(edges) != len(weights) + 1 or not weights:
        return "n/a"

    ranked: list[tuple[int, float, float]] = []
    for i, w in enumerate(weights):
        try:
            lo = float(edges[i])
            hi = float(edges[i + 1])
            weight = float(w)
        except (TypeError, ValueError, IndexError):
            continue
        center = (lo * hi) ** 0.5 if lo > 0 and hi > 0 else (lo + hi) / 2.0
        ranked.append((i, center, weight))
    if not ranked:
        return "n/a"

    ranked.sort(key=lambda t: t[2], reverse=True)
    top = ranked[: max(1, limit)]
    return ";".join(f"b{i}@{center:.3f}={weight*100:.2f}%" for i, center, weight in top)


def _fmt_num(value: Any, decimals: int) -> str:
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return "n/a"


if __name__ == "__main__":
    raise SystemExit(main())
