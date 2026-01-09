#!/usr/bin/env bash
set -euo pipefail

# Dev Containers runs postCreateCommand in the workspace folder already.
# So no hard-coded cd needed.

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

uv sync --group dev || uv sync --extra dev

uv run pre-commit install --install-hooks
