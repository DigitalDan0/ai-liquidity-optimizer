from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from ai_liquidity_optimizer.clients.meteora import MeteoraDlmmApiClient
from ai_liquidity_optimizer.clients.synth import SynthInsightsClient
from ai_liquidity_optimizer.config import load_settings
from ai_liquidity_optimizer.execution.dry_run import DryRunExecutor
from ai_liquidity_optimizer.execution.meteora_node_bridge import MeteoraNodeBridgeExecutor
from ai_liquidity_optimizer.orchestrator import OptimizerOrchestrator
from ai_liquidity_optimizer.state_store import JsonStateStore
from ai_liquidity_optimizer.strategy.scoring import StrategyScorer


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
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    orchestrator.run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

