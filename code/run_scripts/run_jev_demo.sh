#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="${JEV_ENV_FILE:-$PROJECT_ROOT/.env}"

# Replace CHANGE_ME for a local-only launcher configuration.
OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-CHANGE_ME}"

if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi
export OPENROUTER_API_KEY

ARGS=("$@")
if [[ ${#ARGS[@]} -eq 0 ]]; then
    ARGS=(
        --live
        --limit "${JEV_LIMIT:-1}"
        --model "${JEV_MODEL:-typesafe/jev-1.13}"
        --timeout "${JEV_TIMEOUT:-30}"
    )
fi

NEEDS_KEY=true
for arg in "${ARGS[@]}"; do
    if [[ "$arg" == "--dry-run" || "$arg" == "--replay" ]]; then
        NEEDS_KEY=false
        break
    fi
done

if [[ "$NEEDS_KEY" == true && "$OPENROUTER_API_KEY" == "CHANGE_ME" ]]; then
    if [[ ! -t 0 ]]; then
        echo "OPENROUTER_API_KEY is not set and no interactive terminal is available." >&2
        exit 2
    fi
    read -r -s -p "OpenRouter API key: " OPENROUTER_API_KEY
    echo
    export OPENROUTER_API_KEY
fi

if [[ -n "${PYTHON:-}" ]]; then
    PYTHON_BIN="$PYTHON"
elif [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
else
    PYTHON_BIN="python3"
fi

cd "$PROJECT_ROOT"
exec "$PYTHON_BIN" scripts/run_jev_demo.py "${ARGS[@]}"
