#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VPS_HOST="${VPS_HOST:-hetzner}"
REMOTE_ROOT="${REMOTE_ROOT:-/srv/apps/ai-liquidity-optimizer}"
SERVICE_NAME="${SERVICE_NAME:-ai-liquidity-optimizer}"
SSH_BIN="${SSH_BIN:-ssh}"
SSH_OPTS=(-o ClearAllForwardings=yes)

ssh_remote() {
  "$SSH_BIN" "${SSH_OPTS[@]}" "$VPS_HOST" "$@"
}

ssh_remote_tty() {
  "$SSH_BIN" -tt "${SSH_OPTS[@]}" "$VPS_HOST" "$@"
}

rsync_remote() {
  rsync -e "$SSH_BIN -o ClearAllForwardings=yes" "$@"
}

if [[ ! -f "$ROOT/.env" ]]; then
  echo "Missing $ROOT/.env" >&2
  exit 1
fi

env_value() {
  local key="$1"
  awk -F= -v key="$key" '$1 == key {print $2}' "$ROOT/.env" | tail -n 1
}

resolve_local_path() {
  local raw="$1"
  if [[ -z "$raw" ]]; then
    return 1
  fi
  if [[ "$raw" = /* ]]; then
    printf '%s\n' "$raw"
  else
    printf '%s/%s\n' "$ROOT" "${raw#./}"
  fi
}

STATE_PATH_RAW="$(env_value STATE_PATH)"
TRADE_JOURNAL_RAW="$(env_value TRADE_JOURNAL_PATH)"
if [[ -z "$STATE_PATH_RAW" ]]; then
  STATE_PATH_RAW="state/optimizer_state.json"
fi
if [[ -z "$TRADE_JOURNAL_RAW" ]]; then
  STATE_DIR="$(dirname "$STATE_PATH_RAW")"
  TRADE_JOURNAL_RAW="$STATE_DIR/trade_journal.jsonl"
fi

LOCAL_STATE_PATH="$(resolve_local_path "$STATE_PATH_RAW")"
LOCAL_TRADE_JOURNAL_PATH="$(resolve_local_path "$TRADE_JOURNAL_RAW")"

echo "Stopping remote service on ${VPS_HOST}..."
ssh_remote_tty "sudo systemctl stop '$SERVICE_NAME' || true"

mkdir -p "$(dirname "$LOCAL_STATE_PATH")"
mkdir -p "$(dirname "$LOCAL_TRADE_JOURNAL_PATH")"

echo "Pulling runtime state from ${VPS_HOST}..."
rsync_remote -az \
  "$VPS_HOST:$REMOTE_ROOT/${STATE_PATH_RAW#./}" \
  "$LOCAL_STATE_PATH"
rsync_remote -az \
  "$VPS_HOST:$REMOTE_ROOT/${TRADE_JOURNAL_RAW#./}" \
  "$LOCAL_TRADE_JOURNAL_PATH"

echo "Local runtime state refreshed."
