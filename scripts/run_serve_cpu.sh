#!/usr/bin/env bash
set -euo pipefail

uvicorn serve.server_cpu:app --host "${HOST:-0.0.0.0}" --port "${PORT:-8001}" "$@"
