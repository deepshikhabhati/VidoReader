#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! render whoami --output text >/dev/null 2>&1; then
  echo "Render login required."
  render login --confirm --output text
fi

WORKSPACE_ID="$(render workspace list --output json | python3 -c "import sys,json; data=json.load(sys.stdin); print(data[0]['owner']['id'] if data else '')")"
if [[ -z "$WORKSPACE_ID" ]]; then
  echo "Could not determine Render workspace. Run: render workspace set <id>"
  exit 1
fi

echo "Using workspace $WORKSPACE_ID"

if render services list --output json 2>/dev/null | python3 -c "import sys,json; data=json.load(sys.stdin); print(any(s.get('service',{}).get('name')=='vidoreader' for s in data))" | rg -q True; then
  echo "Service vidoreader already exists."
else
  render services create \
    --name vidoreader \
    --type web_service \
    --repo https://github.com/deepshikhabhati/VidoReader \
    --branch main \
    --runtime docker \
    --plan free \
    --health-check-path / \
    --env-var "OPENAI_COMPARISON_MODEL=gpt-4o" \
    --env-var "DATA_DIR=/app/deploy/data" \
    --env-var "TOKENIZERS_PARALLELISM=false" \
    --confirm \
    --output json
fi

echo "Done. Set OPENAI_API_KEY in the Render dashboard if not already configured."
