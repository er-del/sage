#!/usr/bin/env bash
set -euo pipefail

uvicorn serve.server:app --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}" "$@"
