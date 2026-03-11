#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VPS_HOST="${VPS_HOST:-hetzner}"
REMOTE_ROOT="${REMOTE_ROOT:-/srv/apps/ai-liquidity-optimizer}"
SERVICE_NAME="${SERVICE_NAME:-ai-liquidity-optimizer}"
SSH_BIN="${SSH_BIN:-ssh}"
SSH_OPTS=(-o ClearAllForwardings=yes)
REMOTE_NODE_VERSION="${REMOTE_NODE_VERSION:-20}"

ssh_remote() {
  "$SSH_BIN" "${SSH_OPTS[@]}" "$VPS_HOST" "$@"
}

ssh_remote_tty() {
  "$SSH_BIN" -tt "${SSH_OPTS[@]}" "$VPS_HOST" "$@"
}

rsync_remote() {
  rsync -e "$SSH_BIN -o ClearAllForwardings=yes" "$@"
}

remote_bash() {
  local script="$1"
  ssh_remote "bash -lc $(printf '%q' "$script")"
}

if [[ ! -f "$ROOT/.env" ]]; then
  echo "Missing $ROOT/.env" >&2
  exit 1
fi

echo "Stopping remote service on ${VPS_HOST}..."
ssh_remote_tty "sudo systemctl stop '$SERVICE_NAME' || true"

echo "Syncing repo to ${VPS_HOST}:${REMOTE_ROOT}..."
ssh_remote "mkdir -p '$REMOTE_ROOT'"
rsync_remote -az --delete \
  --exclude '.git' \
  --exclude '.venv' \
  --exclude '.pytest_cache' \
  --exclude '__pycache__' \
  --exclude '.DS_Store' \
  --exclude 'state/run_results' \
  --exclude 'scripts/__pycache__' \
  "$ROOT/" "$VPS_HOST:$REMOTE_ROOT/"

echo "Refreshing remote dependencies..."
remote_bash "
set -euo pipefail
export NVM_DIR=\"\$HOME/.nvm\"
if [[ -s \"\$NVM_DIR/nvm.sh\" ]]; then
  . \"\$NVM_DIR/nvm.sh\"
  nvm use ${REMOTE_NODE_VERSION} >/dev/null
fi
cd '$REMOTE_ROOT'
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt >/tmp/ai_liquidity_optimizer_pip.log
cd executors/meteora_ts
npm ci --omit=dev >/tmp/ai_liquidity_optimizer_npm.log
"

echo "Starting remote service..."
ssh_remote_tty "sudo systemctl start '$SERVICE_NAME' && sudo systemctl status '$SERVICE_NAME' --no-pager -l"

echo "Handoff complete."
