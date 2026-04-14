#!/usr/bin/env bash
set -euo pipefail

python -m serve.start --cpu --host "${HOST:-0.0.0.0}" --port "${PORT:-8001}" "$@"
