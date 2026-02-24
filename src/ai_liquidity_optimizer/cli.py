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
    state_store = JsonStateStore(settings.state_path)

    if settings.executor == "dry-run":
        executor = DryRunExecutor()
    else:
        executor = MeteoraNodeBridgeExecutor(
            repo_root=repo_root,
            rpc_url=settings.solana_rpc_url or "",
            private_key_b58=settings.solana_private_key_b58 or "",
        )

    orchestrator = OptimizerOrchestrator(
        settings=settings,
        synth_client=synth_client,
        meteora_client=meteora_client,
        scorer=scorer,
        executor=executor,
        state_store=state_store,
    )

    if args.once:
        result = orchestrator.run_once()
        save_paths = _persist_once_result(result=result, state_path=settings.state_path)
        LOGGER.info(_summarize_once_result(result))
        LOGGER.info(
            "Saved full dry-run result for later review: latest=%s history=%s",
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
