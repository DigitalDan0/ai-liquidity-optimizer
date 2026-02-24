# Meteora DLMM Node Bridge (Optional)

This folder contains a small Node.js bridge used by the Python MVP to execute DLMM rebalances with the official Meteora TypeScript SDK.

Why this exists:
- Meteora DLMM developer tooling is officially documented around the TypeScript SDK.
- The Python app keeps strategy/orchestration chain-agnostic and delegates chain execution to this bridge.

Install:

```bash
cd executors/meteora_ts
npm install
```

The Python app calls `node dlmm_executor.mjs` directly when `EXECUTOR=meteora-node`.

Notes:
- The script uses the SDK flow documented by Meteora: `DLMM.create(...)`, `getBinsBetweenMinAndMaxPrice(...)`, `initializePositionAndAddLiquidityByStrategy(...)`, and `removeLiquidity(...)`.
- It assumes this bot created the currently active position and uses the local state file for the position pubkey/bin range.
- It supports SOL/USDC only (in either pool token order).

