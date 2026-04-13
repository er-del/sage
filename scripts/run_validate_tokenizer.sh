#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-tokenizer/tokenizer.model}"

python - "$MODEL_PATH" <<'PY'
import sys
from tokenizer.validate_tokenizer import validate_model_file

validate_model_file(sys.argv[1])
print("tokenizer ok")
PY
