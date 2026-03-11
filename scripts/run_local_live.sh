#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NODE20_BIN="${NODE20_BIN:-/opt/homebrew/opt/node@20/bin}"

cd "$ROOT"

if [[ ! -f ".env" ]]; then
  echo "Missing $ROOT/.env" >&2
  exit 1
fi

PATH="$NODE20_BIN:$PATH" \
PYTHONPATH=src \
python3 -m ai_liquidity_optimizer --env-file .env "$@"
