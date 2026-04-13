#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
from eval.benchmarks import run_registered_benchmarks
from model.model import SageTransformer
from model.config import ModelConfig

model = SageTransformer(ModelConfig())
for result in run_registered_benchmarks(model):
    print(result)
PY
