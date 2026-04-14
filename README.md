# SAGE — Self-Adaptive General Engine

> A production-grade, from-scratch Large Language Model built with Python and PyTorch.

**SAGE** is a 1B-parameter dense decoder-only transformer trained, served, and controlled entirely from this repository. It implements modern architecture choices — Grouped-Query Attention (GQA), Rotary Positional Embeddings (RoPE), SwiGLU feed-forward layers, and RMSNorm — and ships with a complete pipeline from raw text to live HTTP inference.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1%2B-orange)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/fastapi-0.110%2B-green)](https://fastapi.tiangolo.com/)
[![Hugging Face](https://img.shields.io/badge/🤗-sage002%2Fsage-yellow)](https://huggingface.co/sage002/sage)

---

## Table of Contents

1. [Architecture](#architecture)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [End-to-End Pipeline](#end-to-end-pipeline)
6. [Model Configuration](#model-configuration)
7. [Training Configuration](#training-configuration)
8. [Serving & API](#serving--api)
9. [Browser IDE](#browser-ide)
10. [Evaluation](#evaluation)
11. [Hugging Face](#hugging-face)

---

## Architecture

SAGE implements a **Llama-style decoder-only transformer** with the following design choices:

| Component | Implementation |
|---|---|
| Attention | Grouped-Query Attention (GQA) — 16 query heads, 8 KV heads |
| Positional Encoding | Rotary Positional Embeddings (RoPE) with base freq 500,000 |
| Feed-Forward | SwiGLU gated MLP (hidden dim 5,632) |
| Normalization | Pre-norm RMSNorm (eps 1e-5) |
| Inference | Flash-SDPA with KV-cache for O(1) token generation |
| Precision | bfloat16 on CUDA, float32 on CPU |
| Parallelism | DDP (single-node multi-GPU) or FSDP (multi-node) auto-selected |

### Model Sizes

| Model | Layers | d_model | Heads (Q/KV) | FFN Dim | Context | Parameters |
|---|---|---|---|---|---|---|
| **sage-1b** | 24 | 2048 | 16 / 8 | 5,632 | 4,096 | ~1B |
| **sage-3b** | 32 | 2560 | 20 / 10 | 7,040 | 4,096 | ~3B |
| **sage-7b** | 32 | 4096 | 32 / 8 | 11,008 | 4,096 | ~7B |

---

## Project Structure

```
LLM_MOdel/
│
├── configs/                    # YAML configuration files
│   ├── model/
│   │   ├── 1b.yaml             # 1B parameter model config
│   │   ├── 3b.yaml             # 3B parameter model config
│   │   └── 7b.yaml             # 7B parameter model config
│   └── train/
│       └── schedule.yaml       # Learning rate, warmup, checkpointing schedule
│
├── model/                      # Core model architecture
│   ├── config.py               # ModelConfig dataclass (all hyperparameters)
│   ├── model.py                # SageTransformer — top-level nn.Module
│   ├── block.py                # TransformerBlock (pre-norm, attn + MLP)
│   ├── attention.py            # GQAAttention with fused QKV + SDPA
│   ├── mlp.py                  # SwiGLU feed-forward network
│   ├── rope.py                 # RoPE cache builder + apply_rope
│   └── rmsnorm.py              # RMSNorm with float32 accumulation
│
├── tokenizer/                  # SentencePiece tokenizer
│   ├── train_tokenizer.py      # BPE tokenizer trainer
│   └── validate_tokenizer.py   # Roundtrip + edge-case validation
│
├── data/                       # Data pipeline
│   ├── bootstrap.py            # Generate starter JSONL corpora (5 sources)
│   ├── ingest.py               # Source registry + raw JSONL streaming
│   ├── filter.py               # Quality, language, PII, safety filtering
│   ├── dedup.py                # Exact + near-duplicate removal
│   ├── shard.py                # Tokenize → Parquet shards + manifest
│   ├── dataset.py              # PackedDataset for efficient training
│   └── pipeline.py             # End-to-end CLI: filter → dedup → shard
│
├── train/                      # Training infrastructure
│   ├── trainer.py              # Main training loop (AMP, grad clip, logging)
│   ├── hardware.py             # Auto-detect device, VRAM, batch sizes
│   ├── distributed.py          # DDP / FSDP / single-GPU strategy router
│   ├── optimizer.py            # AdamW + cosine LR schedule
│   ├── loss.py                 # Masked next-token cross-entropy
│   └── checkpoint.py           # Save, prune (keep N), and resume checkpoints
│
├── eval/                       # Evaluation suite
│   ├── perplexity.py           # Validation loss + perplexity computation
│   ├── benchmarks.py           # Benchmark harness registry
│   ├── long_context.py         # Needle-in-haystack probes
│   ├── regression.py           # Metric comparison across checkpoints
│   └── run_benchmarks.py       # CLI entry point for all evals
│
├── serve/                      # HTTP inference servers
│   ├── server.py               # GPU FastAPI server (/health, /generate, /chat)
│   ├── server_cpu.py           # CPU control-plane server
│   ├── control_plane.py        # Job manager, preset runner, SSE log stream
│   ├── kv_cache.py             # KV-cache container for generation
│   ├── quantize.py             # INT8 export + GGUF conversion command
│   └── static/
│       └── index.html          # SAGE IDE — full browser control interface
│
├── scripts/                    # Shell helper scripts
│   ├── run_data_pipeline.sh
│   ├── run_training.sh
│   ├── run_eval.sh
│   ├── run_serve.sh
│   ├── run_serve_cpu.sh
│   └── run_validate_tokenizer.sh
│
├── docs/                       # Documentation
│   ├── COMMANDS.md             # Full command reference
│   ├── flow_llm.mmd            # Operational flow diagram (Mermaid)
│   └── llm_Arch.mmd            # System architecture diagram (Mermaid)
│
├── tests/                      # Test suite (pytest)
├── hf_push.py                  # Hugging Face sync script
├── requirements.txt            # Python dependencies
└── test.ipynb                  # Colab/ngrok notebook
```

---

## Installation

### Requirements

- Python 3.10 or newer
- CUDA-capable GPU recommended (NVIDIA T4 16GB or better)
- 16+ GB RAM for CPU-only mode

```bash
# Clone from GitHub
git clone https://github.com/er-del/sage.git
cd sage

# Install all dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Version | Purpose |
|---|---|---|
| `torch` | ≥ 2.1.0 | Core deep learning |
| `fastapi` | ≥ 0.110.0 | HTTP inference server |
| `uvicorn` | ≥ 0.29.0 | ASGI web server |
| `sentencepiece` | ≥ 0.2.0 | BPE tokenizer |
| `pyarrow` | ≥ 16.0.0 | Parquet shard I/O |
| `pyyaml` | ≥ 6.0.1 | Config file parsing |
| `wandb` | ≥ 0.17.0 | Experiment tracking |
| `psutil` | ≥ 5.9.8 | Hardware detection |
| `bitsandbytes` | ≥ 0.43.0 | INT8 quantization |
| `pytest` | ≥ 8.2.0 | Test suite |

---

## Quick Start

### Run tests first

```bash
pytest -q
```

### Full pipeline in one go (smoke run)

```bash
# 1. Generate starter data
python -m data.bootstrap --output-dir data/raw --overwrite

# 2. Train tokenizer
python -m tokenizer.train_tokenizer \
  --input data/raw/general_web.jsonl data/raw/code.jsonl data/raw/math_science.jsonl \
           data/raw/multilingual.jsonl data/raw/synthetic.jsonl \
  --model-prefix tokenizer/tokenizer \
  --vocab-size 4096

# 3. Build shards
python -m data.pipeline \
  --tokenizer-model tokenizer/tokenizer.model \
  --output-dir data/processed \
  --shard-size 32 --limit-per-source 4

# 4. Train (20-step smoke)
python -m train.trainer \
  --model-config configs/model/1b.yaml \
  --schedule-config configs/train/schedule.yaml \
  --train-shards data/processed/shard-00000.parquet \
  --output-dir runs/smoke \
  --steps 20 --disable-wandb

# 5. Serve
python -m uvicorn serve.server:app --host 0.0.0.0 --port 8000
```

---

## End-to-End Pipeline

See **[docs/COMMANDS.md](docs/COMMANDS.md)** for the full command reference with all options.

### Stage 1 — Bootstrap Data

```bash
python -m data.bootstrap --output-dir data/raw
```

Writes five JSONL corpus files across domains: `general_web`, `code`, `math_science`, `multilingual`, `synthetic`. Each record has `{ "id", "text", "source_name" }`.

Bring your own data by placing JSONL files with a `"text"` field in `data/raw/`.

### Stage 2 — Train Tokenizer

```bash
python -m tokenizer.train_tokenizer \
  --input data/raw/*.jsonl \
  --model-prefix tokenizer/tokenizer \
  --vocab-size 50000
```

Outputs `tokenizer/tokenizer.model` and `tokenizer/tokenizer.vocab`.

### Stage 3 — Validate Tokenizer

```bash
python -m tokenizer.validate_tokenizer tokenizer/tokenizer.model
```

### Stage 4 — Build Parquet Shards

```bash
python -m data.pipeline \
  --tokenizer-model tokenizer/tokenizer.model \
  --output-dir data/processed \
  --shard-size 2048
```

The pipeline runs: **ingest → filter → dedup → tokenize → shard**. Each shard is a Parquet file with `input_ids`, `labels`, and `loss_mask` columns.

### Stage 5 — Train

```bash
python -m train.trainer \
  --model-config configs/model/1b.yaml \
  --schedule-config configs/train/schedule.yaml \
  --train-shards data/processed/shard-00000.parquet \
  --validation-shards data/processed/shard-00001.parquet \
  --output-dir runs/sage-1b
```

Checkpoints are saved every 1,000 steps to `runs/sage-1b/ckpt_step_XXXXXXX.pt`. Metrics are logged to `runs/sage-1b/metrics.jsonl` and optionally to Weights & Biases.

### Stage 6 — Evaluate

```bash
python -m eval.run_benchmarks
```

---

## Model Configuration

All model hyperparameters live in `configs/model/*.yaml`. The default configuration:

```yaml
# configs/model/1b.yaml
name: sage-1b
num_layers: 24
d_model: 2048
num_attn_heads: 16        # Query heads
num_kv_heads: 8           # KV heads (GQA ratio: 2:1)
head_dim: 128
ffn_hidden_dim: 5632      # SwiGLU hidden dimension
vocab_size: 50000
context_length: 4096
rope_base_frequency: 500000
rope_scaling_factor: 1.0
dropout: 0.0
tie_word_embeddings: true
```

> **Constraint:** `num_attn_heads × head_dim` must equal `d_model`, and `num_attn_heads` must be divisible by `num_kv_heads`.

---

## Training Configuration

```yaml
# configs/train/schedule.yaml
run_name: sage-1b-pretrain
total_tokens: 50_000_000_000   # 50B token target
effective_batch_tokens: 2_000_000
peak_learning_rate: 3.0e-4
min_learning_rate: 3.0e-5
warmup_steps: 2000             # Linear warmup
weight_decay: 0.1
betas: [0.9, 0.95]
adam_eps: 1.0e-8
gradient_clip_norm: 1.0
checkpoint_interval: 1000
log_interval: 10
eval_interval: 1000
seed: 42
```

Hardware is auto-detected. The system selects the appropriate strategy:

| Setup | Strategy | ZeRO Stage |
|---|---|---|
| CPU-only | `cpu` | — |
| Single GPU < 40 GB | `single` | ZeRO-3 |
| Single GPU ≥ 40 GB | `single` | ZeRO-0 |
| Multi-GPU (≤ 1B) | `ddp` | ZeRO-1 |
| Multi-GPU (> 1B) | `fsdp` | ZeRO-2 |

---

## Serving & API

### GPU Server (PyTorch)

```bash
python -m serve.start --host 0.0.0.0 --port 8000
```

### CPU Server (Control-plane only)

```bash
python -m serve.start --cpu --host 0.0.0.0 --port 8001
```

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Server health + hardware summary |
| `GET` | `/chat/status` | Tokenizer + checkpoint readiness |
| `POST` | `/chat` | Text generation from a prompt |
| `POST` | `/generate` | Raw token-id generation |
| `GET` | `/api/commands/presets` | List all UI presets |
| `POST` | `/api/commands/run` | Run a preset or shell command |
| `GET` | `/api/jobs` | List all background jobs |
| `GET` | `/api/jobs/{id}` | Job details + last 200 log lines |
| `GET` | `/api/jobs/{id}/stream` | SSE live log stream |
| `POST` | `/api/jobs/{id}/stop` | Terminate a running job |

#### Example — Health Check

```bash
curl http://localhost:8000/health
```

#### Example — Chat

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is a transformer?", "max_new_tokens": 128}'
```

#### Example — Raw Token Generation

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"input_ids": [1, 42, 99], "max_new_tokens": 32}'
```

---

## Browser IDE

Open `http://localhost:8000/` in any modern browser after starting the server.

The **SAGE IDE** is a full-featured control interface built into `serve/static/index.html`:

- **Chat panel** — conversational interface to `/chat` with Markdown rendering and code syntax highlighting
- **Preset launcher** — one-click forms for every pipeline operation (bootstrap, shard, train, eval, serve, git)
- **Job monitor** — live job list with SSE log streaming
- **CLI terminal** — embedded shell with command history and remote execution
- **Function inspector** — full documentation for every module and API endpoint
- **Settings panel** — all config options persisted to localStorage
- **Command palette** — `Ctrl+K` quick-action search

---

## Evaluation

```bash
python -m eval.run_benchmarks
```

The eval suite covers:

| Module | What it measures |
|---|---|
| `eval/perplexity.py` | Validation loss & perplexity on held-out shards |
| `eval/benchmarks.py` | Pluggable benchmark task harness |
| `eval/long_context.py` | Needle-in-haystack retrieval across long sequences |
| `eval/regression.py` | Metric comparison across saved checkpoints |

---

## Hugging Face

Pre-trained checkpoints, tokenizer, and datasets are published at:

🔗 **[huggingface.co/sage002/sage](https://huggingface.co/sage002/sage)**

To sync your local runs to Hugging Face:

```bash
python hf_push.py
```

---

## Diagrams

Architecture and flow diagrams are in `docs/` as Mermaid files (`.mmd`). Render them with any Mermaid-compatible viewer, or paste into [mermaid.live](https://mermaid.live).

| File | Content |
|---|---|
| `docs/llm_Arch.mmd` | Full system architecture — all modules and data flows |
| `docs/flow_llm.mmd` | Simplified operational flow — end-to-end stages |

---

## Disclaimer

SAGE is an experimental research engine. Response quality scales with training data volume and compute steps. The architecture is production-grade; checkpoints are experimental.

**Developed by Antigravity AI Systems.**
