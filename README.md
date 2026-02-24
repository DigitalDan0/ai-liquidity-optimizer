# AI Liquidity Optimizer (Hackathon MVP)

A minimal Python-first bot that uses Synth probabilistic LP forecasts to choose a SOL/USDC price range and manage a single Meteora DLMM position.

Scope (intentionally strict):
- Asset: `SOL`
- Pool: `SOL/USDC`
- One active position only
- Periodic rebalance loop (default `10` minutes, configurable)
- No UI
- Chain-agnostic strategy core with execution adapter separation

## What It Does

Every cycle, the bot:
1. Fetches Synth LP bounds forecasts (probability to stay in range, expected time in range, expected impermanent loss)
2. Fetches Meteora DLMM pool metadata/metrics for SOL/USDC
3. Scores candidate ranges using a simple expected net return proxy
4. Rebalances the active position when the optimal range materially changes

## Doc-Derived Integration Notes

This implementation was based on the official docs you linked:

- Synth trading/LP insights docs: [docs.synthdata.co/insights/trading](https://docs.synthdata.co/insights/trading)
  - Uses `GET /insights/lp-bounds` on `https://api.synthdata.co`
  - Auth header format: `Authorization: Apikey <key>`
  - Forecast fields used in this MVP:
    - `width_pct`
    - `lower_bound`
    - `upper_bound`
    - `probability_to_stay_in_interval`
    - `expected_time_in_interval`
    - `expected_impermanent_loss`

- Meteora DLMM overview and docs:
  - Overview / DLMM concept: [docs.meteora.ag](https://docs.meteora.ag/overview/products/dlmm/what-is-dlmm)
  - DLMM API reference (pool discovery/metrics): [docs.meteora.ag/api-reference/dlmm/overview](https://docs.meteora.ag/api-reference/dlmm/overview)
  - DLMM SDK functions: [docs.meteora.ag/developer-guide/guides/dlmm/typescript-sdk/sdk-functions](https://docs.meteora.ag/developer-guide/guides/dlmm/typescript-sdk/sdk-functions)

Important behavior note:
- Meteora DLMM position ranges are not edited in place; rebalancing is implemented as remove/close old position + create new one (via the SDK flow).

## Strategy Scoring (MVP)

The bot scores each Synth-provided candidate range using a simple risk-adjusted proxy:

- Base fee return proxy from Meteora pool metrics (`fee_tvl_ratio_24h` when available, else `fees_24h / liquidity`)
- Scaled by Synth `expected_time_in_interval / horizon`
- Further confidence-adjusted by Synth `probability_to_stay_in_interval`
- Minus Synth `expected_impermanent_loss`

This is intentionally simple for hackathon speed and can be upgraded later.

## Repository Layout

```text
ai-liquidity-optimizer/
├── src/ai_liquidity_optimizer/      # Python core (strategy + orchestration + adapters)
├── executors/meteora_ts/            # Optional Node bridge for Meteora DLMM SDK execution
├── tests/                           # Unit tests for strategy logic
├── .env.example
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Quick Start (Dry Run)

1. Create a virtualenv (optional but recommended)
2. Copy `.env.example` to `.env`
3. Set `SYNTH_API_KEY`
4. Keep `EXECUTOR=dry-run`

Run one cycle:

```bash
PYTHONPATH=src python -m ai_liquidity_optimizer --once
```

Run the periodic loop:

```bash
PYTHONPATH=src python -m ai_liquidity_optimizer
```

State is written to `state/optimizer_state.json` by default.

## Enable Real Meteora Execution (Optional)

The Python app is the primary project; Meteora execution is delegated to the official TypeScript SDK via a small bridge.

1. Install the bridge dependencies:

```bash
cd executors/meteora_ts
npm install
```

2. Configure in `.env`:
- `EXECUTOR=meteora-node`
- `SOLANA_RPC_URL`
- `SOLANA_PRIVATE_KEY_B58`
- `DEPOSIT_SOL_AMOUNT`
- `DEPOSIT_USDC_AMOUNT`

3. Run one cycle:

```bash
PYTHONPATH=src python -m ai_liquidity_optimizer --once
```

## Config Highlights

- `REBALANCE_INTERVAL_MINUTES` (default `10`)
- `RANGE_CHANGE_THRESHOLD_BPS` (rebalance trigger sensitivity)
- `METEORA_POOL_ADDRESS` (optional pool pinning)
- `METEORA_POOL_QUERY` (used for discovery, default `SOL/USDC`)
- `SYNTH_HORIZON` (default `24h`)

## Limitations (MVP)

- SOL/USDC only
- One position only
- Uses a simple net-return proxy, not a full transaction-cost model
- Real executor assumes the bot manages the position lifecycle and stores the active position metadata locally
- Not production-hardened (no secret manager, no alerting, no robust tx retry policy)

## GitHub-Ready Notes

- Includes `.gitignore`, `pyproject.toml`, `requirements.txt`, unit tests, and a basic GitHub Actions CI workflow.
- Designed to run in dry-run mode by default for safe demo iteration.

