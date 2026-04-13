# SAGE 1B

SAGE is a root-level rewrite of this repository into a production-style dense language model project. The current baseline is a 1B-class decoder-only transformer with RMSNorm, RoPE, grouped-query attention, SwiGLU, SentencePiece, resumable training, Parquet-backed datasets, and FastAPI serving.

This README is written as a practical operator guide. It tells you:

- what the project contains
- what is already implemented
- what commands to run
- what files are inputs and outputs
- what parts are scaffolding versus fully wired

## What SAGE Is

SAGE is organized into these layers:

1. `tokenizer/`
   Trains and validates a SentencePiece tokenizer.
2. `data/`
   Handles raw corpus ingest, filtering, deduplication, sharding, and packed datasets.
3. `model/`
   Implements the dense decoder-only transformer.
4. `train/`
   Handles optimizer setup, scheduler, hardware detection, checkpoints, and the training loop.
5. `eval/`
   Provides perplexity evaluation and benchmark harness registration.
6. `serve/`
   Exposes FastAPI servers and quantization helpers.

## Current Baseline

| Component | Value |
| --- | --- |
| Layers | 24 |
| d_model | 2048 |
| Attention heads | 16 |
| KV heads | 8 |
| Head dim | 128 |
| FFN dim | 5632 |
| Context length | 4096 |
| Vocab size | 50000 |
| Norm | RMSNorm |
| Positional encoding | RoPE |
| Attention | GQA + SDPA |
| Activation | SwiGLU |
| Weight tying | Enabled |

## Repository Layout

```text
configs/
  model/         model YAMLs for 1B, 3B, 7B
  data/          corpus mix and shard config
  train/         LR, checkpoint, and logging schedule
data/
  ingest.py      raw source registry and streaming helpers
  filter.py      license/lang/PII/safety/quality filtering
  dedup.py       exact and near-duplicate removal
  shard.py       tokenization + parquet shard writing + manifest
  dataset.py     packed iterable dataset with resume skip()
tokenizer/
  train_tokenizer.py
  validate_tokenizer.py
model/
  config.py
  rmsnorm.py
  rope.py
  attention.py
  mlp.py
  block.py
  model.py
train/
  loss.py
  optimizer.py
  checkpoint.py
  distributed.py
  hardware.py
  trainer.py
eval/
  perplexity.py
  benchmarks.py
  long_context.py
  regression.py
serve/
  kv_cache.py
  quantize.py
  server.py
  server_cpu.py
scripts/
  run_data_pipeline.sh
  run_training.sh
  run_eval.sh
  run_serve.sh
  run_serve_cpu.sh
  run_validate_tokenizer.sh
tests/
```

## What Is Fully Working vs. What Is Scaffolded

### Working now

- tokenizer training
- tokenizer validation
- data filtering and dedup helpers
- packed dataset logic
- dense transformer forward pass
- checkpoint save and resume
- hardware detection
- trainer entrypoint
- FastAPI health and basic generate endpoint
- unit and smoke tests

### Scaffolded but not yet a full production runner

- benchmark execution against downloaded external datasets
- a single end-to-end corpus build command that downloads and preprocesses public corpora automatically
- production-grade multi-node launch tooling
- real llama.cpp server wiring beyond availability checks

That means the core codebase is real, but you still need to provide your own corpus files and Parquet shards before running a training job.

## Install

Create and activate a virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Recommended optional extras:

- `sentencepiece` is required for tokenizer training and validation
- `bitsandbytes` is useful for 8-bit experiments
- `llama.cpp` or `llama-cpp-python` is needed for the CPU serving path

## Quick Start

If you want the shortest path to verifying the repo:

1. Install dependencies.
2. Run tests.
3. Start the FastAPI server.

```bash
pytest -q
uvicorn serve.server:app --host 127.0.0.1 --port 8000
```

Then check:

```bash
curl http://127.0.0.1:8000/health
```

## Command Reference

The detailed command guide is in [docs/COMMANDS.md](C:/Users/Lenovo/OneDrive/Desktop/Documents/LLM_MOdel/docs/COMMANDS.md:1). The most important commands are below.

### 1. Train tokenizer

Cross-platform Python command:

```bash
python -m tokenizer.train_tokenizer \
  --input data/raw/general_web.txt data/raw/code.txt \
  --model-prefix tokenizer/tokenizer \
  --vocab-size 50000
```

Linux/macOS/WSL wrapper:

```bash
bash scripts/run_data_pipeline.sh \
  --input data/raw/general_web.txt data/raw/code.txt \
  --model-prefix tokenizer/tokenizer \
  --vocab-size 50000
```

Outputs:

- `tokenizer/tokenizer.model`
- `tokenizer/tokenizer.vocab`
- `tokenizer/training_corpus.txt`

### 2. Validate tokenizer

```bash
python - <<'PY'
from tokenizer.validate_tokenizer import validate_model_file
validate_model_file("tokenizer/tokenizer.model")
print("tokenizer ok")
PY
```

Or:

```bash
bash scripts/run_validate_tokenizer.sh tokenizer/tokenizer.model
```

### 3. Train the model

Training expects existing Parquet shards. Example:

```bash
python -m train.trainer \
  --model-config configs/model/1b.yaml \
  --schedule-config configs/train/schedule.yaml \
  --train-shards data/processed/shard-00000.parquet data/processed/shard-00001.parquet \
  --validation-shards data/processed/shard-00002.parquet \
  --output-dir runs/sage-1b
```

Useful options:

- `--steps 100` for a short smoke run
- `--disable-wandb` to disable offline W&B logging

Example smoke run:

```bash
python -m train.trainer \
  --train-shards data/processed/shard-00000.parquet \
  --validation-shards data/processed/shard-00001.parquet \
  --output-dir runs/smoke \
  --steps 20 \
  --disable-wandb
```

### 4. Run evaluation harness

```bash
bash scripts/run_eval.sh
```

This currently prints the registered benchmark surfaces. It is a harness check, not a full benchmark download-and-run pipeline.

### 5. Start the GPU server

```bash
uvicorn serve.server:app --host 0.0.0.0 --port 8000
```

Or:

```bash
bash scripts/run_serve.sh
```

### 6. Start the CPU server

```bash
uvicorn serve.server_cpu:app --host 0.0.0.0 --port 8001
```

Or:

```bash
bash scripts/run_serve_cpu.sh
```

### 7. Call the generate endpoint

The current server takes token IDs directly, not raw text strings.

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d "{\"input_ids\": [1, 42, 99], \"max_new_tokens\": 8}"
```

Response shape:

```json
{
  "tokens": [1, 42, 99, 123, 456]
}
```

## How Training Works

The training flow is:

1. load model config from `configs/model/*.yaml`
2. load schedule config from `configs/train/schedule.yaml`
3. detect hardware in `train/hardware.py`
4. build optimizer and cosine scheduler
5. load latest checkpoint if one exists
6. call `PackedDataset.skip()` so resume does not replay already-trained batches
7. run forward/backward with autocast on CUDA or MPS
8. clip gradients
9. log metrics to `metrics.jsonl` and optionally offline W&B
10. run validation perplexity at eval intervals
11. save checkpoint every configured interval

Important output files during training:

- `runs/<run-name>/metrics.jsonl`
- `runs/<run-name>/ckpt_step_0001000.pt`
- later checkpoints in the same folder

## How Data Is Expected to Look

### Raw text files for tokenizer training

Simple UTF-8 text files are enough:

```text
This is a training document.
This is another one.
```

### Raw JSONL records for ingest/filter work

The ingest layer assumes records like:

```json
{"text": "example text"}
```

### Processed Parquet shards for training

The trainer expects Parquet rows with at least:

- `tokens`
- `split`

The sharding helper writes:

- `id`
- `text`
- `tokens`
- `domain_tag`
- `quality_tier`
- `lang`
- `token_count`
- `split`

## Main Config Files

### [configs/model/1b.yaml](C:/Users/Lenovo/OneDrive/Desktop/Documents/LLM_MOdel/configs/model/1b.yaml:1)

Controls the model shape:

- layers
- hidden size
- heads
- KV heads
- FFN size
- vocab size
- context length

### [configs/data/mix.yaml](C:/Users/Lenovo/OneDrive/Desktop/Documents/LLM_MOdel/configs/data/mix.yaml:1)

Controls corpus weights and split ratios.

### [configs/train/schedule.yaml](C:/Users/Lenovo/OneDrive/Desktop/Documents/LLM_MOdel/configs/train/schedule.yaml:1)

Controls:

- total token target
- LR schedule
- warmup
- checkpoint interval
- log interval
- eval interval

## Common Workflows

### Workflow A: verify the repo

```bash
pip install -r requirements.txt
pytest -q
```

### Workflow B: train tokenizer only

```bash
python -m tokenizer.train_tokenizer --input data/raw/general_web.txt --model-prefix tokenizer/tokenizer
python - <<'PY'
from tokenizer.validate_tokenizer import validate_model_file
validate_model_file("tokenizer/tokenizer.model")
print("ok")
PY
```

### Workflow C: smoke-train on local shards

```bash
python -m train.trainer \
  --train-shards data/processed/shard-00000.parquet \
  --validation-shards data/processed/shard-00001.parquet \
  --output-dir runs/smoke \
  --steps 20 \
  --disable-wandb
```

### Workflow D: serve locally

```bash
uvicorn serve.server:app --host 127.0.0.1 --port 8000
curl http://127.0.0.1:8000/health
```

## Troubleshooting

### `No training shards provided`

You launched the trainer without `--train-shards`. The trainer is working as designed, but it needs Parquet shard paths.

### `ModuleNotFoundError: sentencepiece`

Install dependencies:

```bash
pip install -r requirements.txt
```

### FastAPI starts but generate is not useful

That is expected right now if you have not trained or loaded a checkpoint. The server instantiates the model architecture, but it does not yet load a trained checkpoint automatically.

### CPU server says llama.cpp is unavailable

Install `llama.cpp` or `llama-cpp-python`. The current CPU server is a readiness surface, not a bundled llama.cpp runtime.

## Tests

Run the full suite:

```bash
pytest -q
```

Coverage areas:

- tokenizer roundtrip validation
- model shapes
- attention math
- data filtering and packing
- checkpoint roundtrip
- hardware summaries
- FastAPI health endpoints

## Next Practical Step

If you want the fastest real progress from here, the next step is:

1. prepare a small local corpus
2. train the tokenizer
3. write Parquet shards with `data/shard.py`
4. run a `--steps 20` smoke training job
5. only then start extending benchmark or serving behavior
