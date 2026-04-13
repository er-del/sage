# SAGE Commands

This file is the short command-only reference for the repo.

## Install

```bash
pip install -r requirements.txt
```

## Run tests

```bash
pytest -q
```

## Train tokenizer

```bash
python -m tokenizer.train_tokenizer \
  --input data/raw/general_web.txt data/raw/code.txt \
  --model-prefix tokenizer/tokenizer \
  --model-prefix tokenizer/tokenizer \
```

## Validate tokenizer

```bash
bash scripts/run_validate_tokenizer.sh tokenizer/tokenizer.model
```

## Start a short training smoke run

```bash
python -m train.trainer \
  --train-shards data/processed/shard-00000.parquet \
  --validation-shards data/processed/shard-00001.parquet \
  --output-dir runs/smoke \
  --steps 20 \
  --disable-wandb
```

## Start full training

```bash
python -m train.trainer \
  --model-config configs/model/1b.yaml \
  --schedule-config configs/train/schedule.yaml \
  --train-shards data/processed/shard-00000.parquet data/processed/shard-00001.parquet \
  --validation-shards data/processed/shard-00002.parquet \
  --output-dir runs/sage-1b
```

## Run eval harness

```bash
bash scripts/run_eval.sh
```

## Start GPU server

```bash
bash scripts/run_serve.sh
```

## Start CPU server

```bash
bash scripts/run_serve_cpu.sh
```

## Check server health

```bash
curl http://127.0.0.1:8000/health
```

## Generate tokens from the API

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d "{\"input_ids\": [1, 42, 99], \"max_new_tokens\": 8}"
```
