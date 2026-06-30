#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m ruff check src tests app
python -m black --check src tests app
python -m pytest
