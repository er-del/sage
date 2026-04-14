#!/usr/bin/env bash
set -euo pipefail

python -m serve.start --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}" "$@"
