#!/usr/bin/env bash
set -euo pipefail

python -m tokenizer.train_tokenizer "$@"
