#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

# Replace CHANGE_ME for a local-only full-test configuration.
OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-CHANGE_ME}"
export OPENROUTER_API_KEY

if [[ "$OPENROUTER_API_KEY" == "CHANGE_ME" ]]; then
    echo "Set OPENROUTER_API_KEY before starting the full Jev experiment." >&2
    exit 2
fi

if [[ -n "${PYTHON:-}" ]]; then
    PYTHON_BIN="$PYTHON"
elif [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
else
    PYTHON_BIN="python3"
fi

cd "$PROJECT_ROOT"
exec "$PYTHON_BIN" scripts/run_jev_full_test.py "$@"
