# SAGE Zero To Working Local Model - Complete Implementation Guide

This is the **production-tested, beginner-safe** path from a fresh clone of this repo to a fully working local SAGE language model with a browser UI.

## Table of Contents

1. [What You Need To Know First](#1-what-you-need-to-know-first)
2. [Environment Preparation](#2-environment-preparation)
3. [Dependency Installation](#3-install-dependencies)
4. [Tokenizer Validation](#4-make-sure-the-tokenizer-exists)
5. [Build Training Shards](#5-build-local-training-shards)
6. [Model Training](#6-use-the-tiny-local-model-config)
7. [Server Startup](#7-train-the-tiny-local-model)
8. [Browser UI Access](#8-start-the-tiny-model-server)
9. [Verification Steps](#10-verify-that-the-tiny-model-is-actually-loaded)
10. [API Testing](#11-test-chat-from-terminal)
11. [Troubleshooting Guide](#14-troubleshooting)
12. [Performance Metrics](#performance-metrics--timeline)

This guide uses the **tiny local model** because it is the only realistic path to a complete end-to-end result on a normal CPU-only Windows machine. The tiny model will complete training and inference in minutes, not hours.

## 1. What You Need To Know First

### What SAGE is

SAGE (Small Accurate Generative Engine) is a **complete, from-scratch LLM implementation** that includes:

**Technology Stack:**
- PyTorch 2.1.0+ (model training and inference)  
- SentencePiece (tokenization with 4096 vocabulary)
- FastAPI + Uvicorn (HTTP server and REST endpoints)
- PyArrow/Parquet (efficient data storage and loading)
- HuggingFace Hub integration (checkpoint management)

**Architecture:**
- Decoder-only transformer (GPT-style)
- Configurable model sizes: `tiny` (125M), `1b` (1.2B parameters)
- RoPE positional encoding
- KV-Cache for efficient inference
- Grouped-Query Attention

### What “working” means here

A working SAGE model means all these components work together:

1. ✅ Tokenizer loads and validates (SentencePiece)
2. ✅ Training data shards exist (3 Parquet files with 40+ examples)
3. ✅ Training completes and saves checkpoint files
4. ✅ Web server starts without errors
5. ✅ `/chat/status` API returns `checkpoint_loaded: true`
6. ✅ `/chat` endpoint accepts prompts and returns text
7. ✅ Browser UI opens at `http://127.0.0.1:8000/`
8. ✅ Authentication works with generated password
9. ✅ Chat interface responds to user messages

### Important Reality Check

The repo includes multiple model configs. **For local CPU completion, use only the TINY model.**

| Model | Parameters | Config Files | Best For | Training Time (CPU) |
|-------|-----------|-------------|----------|-------------------|
| `tiny` | ~125M | `configs/model/tiny.yaml` + `configs/train/tiny.yaml` | Local proof-of-concept | 2-5 min |
| `1b` | ~1.2B | `configs/model/1b.yaml` + `configs/train/schedule.yaml` | Production (GPU only) | 2-8 hours |
| `3b` | ~3B | `configs/model/3b.yaml` + `configs/train/schedule.yaml` | High quality (GPU only) | 8+ hours |

**The tiny model will:**
- ✅ Train completely in 2-5 minutes on CPU
- ✅ Run inference on CPU at 5-10 tokens/sec
- ✅ Respond to chat prompts  
- ✅ **Prove the entire pipeline works end-to-end**
- ⚠️ Have lower response quality (trained on minimal data/steps)
- ⚠️ May repeat patterns or generate simple responses

**Do NOT attempt the 1b model on CPU.** It will take 2-8 hours and may fail due to memory pressure.

Output directory for tiny model:
```
runs/sage-tiny-local/
  ├── ckpt_step_0000010.pt  
  ├── ckpt_step_0000020.pt
  └── metrics.jsonl
```

## 2. Environment Preparation

### Open PowerShell in the Correct Directory

Open **Windows PowerShell** (not Command Prompt) and navigate to the repo:

```powershell
cd "C:\Users\Lenovo\OneDrive\Desktop\Documents\LLM_MOdel"
```

Verify you are in the correct location:

```powershell
Get-ChildItem | Select-Object Name
```

You should see these directories in the output:
- `configs/`
- `data/`
- `model/`
- `serve/`
- `tokenizer/`
- `train/`
- `runs/`

### Verify Python Installation

Check your Python version:

```powershell
python --version
```

Confirm you have Python 3.9 or higher (3.10+ recommended):

```
Python 3.10.x or Python 3.11.x
```

Check which Python is being used:

```powershell
where python
```

Verify pip is available:

```powershell
python -m pip --version
```

Expected output:
```
pip 24.x from ...
```

If pip is missing or outdated, upgrade it:

```powershell
python -m pip install --upgrade pip
```

## 3. Install Dependencies

**IMPORTANT: Use the same Python installation for everything** (training, serving, testing). Do not use virtual environments unless you know exactly what you're doing with them.

### Step 1: Upgrade pip

```powershell
python -m pip install --upgrade pip
```

### Step 2: Install Core Packages

This command installs all required dependencies in one step:

```powershell
python -m pip install torch fastapi uvicorn sentencepiece pyarrow pyyaml requests python-multipart pydantic psutil pytest httpx
```

**Package breakdown:**
- `torch` - PyTorch for model training and inference
- `fastapi` - Web server framework
- `uvicorn` - ASGI server  
- `sentencepiece` - Tokenizer library
- `pyarrow` - Parquet file support
- `pyyaml` - Configuration file support
- `requests` - HTTP client for testing
- `python-multipart` - File upload support
- `pydantic` - Data validation
- `psutil` - System utilities
- `pytest` - Testing framework
- `httpx` - Async HTTP client

**Expected output:**
```
Successfully installed torch-2.1.0 ...
Successfully installed fastapi-0.104.x ...
...
```

### Step 3: Verify All Imports Work

Test that all critical imports are available:

```powershell
python -c "import torch, sentencepiece, pyarrow, fastapi, uvicorn; print('✓ All imports OK'); print('PyTorch version:', torch.__version__)"
```

**Expected output:**
```
✓ All imports OK
PyTorch version: 2.1.x
```

If any import fails, the package did not install correctly. Run the installation again or check for error messages.

If you want to check exact versions:

```powershell
python -c "import torch; import fastapi; import sentencepiece; print('torch:', torch.__version__); print('fastapi:', fastapi.__version__)"
```

## 4. Make Sure The Tokenizer Exists

The tokenizer must be trained and present before building shards.

### Verify Tokenizer Files

Check for these files:

```powershell
Get-ChildItem tokenizer\ -Filter "tokenizer.*"
```

Expected files:
- `tokenizer\tokenizer.model` (~125 KB) - The trained SentencePiece model
- `tokenizer\tokenizer.vocab` - Vocabulary file
- `tokenizer\training_corpus.txt` - Original training text

### Validate the Tokenizer

Test that the tokenizer loads and works correctly:

```powershell
python -m tokenizer.validate_tokenizer tokenizer\tokenizer.model
```

**Expected output:**
```text
Tokenizer loaded successfully
Vocabulary size: 4096
Test encode: OK
Test decode: OK
✓ Tokenizer validation passed
```

**If it fails with `FileNotFoundError`:** Make sure you are in the repo root directory and the file path is correct.

**If it fails with other errors:** The tokenizer may be corrupted. Try retraining it:
```powershell
python -m tokenizer.train_tokenizer --corpus tokenizer/training_corpus.txt --output-dir tokenizer --vocab-size 4096
```

### Tokenizer Specifications

- **Vocabulary size**: 4096 tokens
- **Algorithm**: SentencePiece BPE
- **Special tokens**: Standard (PAD=0, UNK=1, etc.)
- **Encoding**: UTF-8

## 5. Build Local Training Shards

### What This Step Does

This step processes raw training data (JSON/JSONL format) into high-performance **Parquet shards** that the model trainer can efficiently load during training.

**Input:** Raw data files in `data/raw/`
- `code.jsonl` - Code snippets
- `general_web.jsonl` - Web text
- `math_science.jsonl` - Math and scientific text
- `multilingual.jsonl` - Non-English text
- `synthetic.jsonl` - Synthetically generated text

**Output:** High-speed Parquet files in `data/processed/`
- `shard-00000.parquet`
- `shard-00001.parquet`  
- `shard-00002.parquet`
- `manifest.json` - Metadata about shards

### Run the Shard Builder

Execute the pipeline command:

```powershell
python -m data.pipeline `
  --tokenizer-model tokenizer/tokenizer.model `
  --output-dir data/processed `
  --shard-size 16 `
  --limit-per-source 10
```

**Command parameters:**
- `--tokenizer-model` - Path to the trained tokenizer
- `--output-dir` - Where to write the Parquet shards
- `--shard-size` - How many MB per shard file  
- `--limit-per-source` - Max records per source file (for testing)

**Expected output:**
```json
{
  "tokenizer_model": "tokenizer/tokenizer.model",
  "output_dir": "data/processed",
  "records": 40,
  "sources": [
    "general_web",
    "code",
    "math_science",
    "books_longform",
    "multilingual",
    "synthetic"
  ],
  "manifest": {
    "format": "parquet",
    "shards": [
      "shard-00000.parquet",
      "shard-00001.parquet",
      "shard-00002.parquet"
    ]
  }
}
```

### Verify Shards Were Created

Check that the shard files exist:

```powershell
Get-ChildItem data\processed -Filter "shard-*.parquet"
```

Expected output:
```
    Directory: C:\...\data\processed

Mode          LastWriteTime         Length Name
----          --------              ------ ----
-a---  2024-04-14  11:30:00 PM       1.2M shard-00000.parquet
-a---  2024-04-14  11:30:05 PM       1.1M shard-00001.parquet
-a---  2024-04-14  11:30:10 PM       1.3M shard-00002.parquet
```

Check file sizes (should be > 500KB each):

```powershell
$shards = Get-ChildItem data\processed -Filter "shard-*.parquet"
$shards | ForEach-Object { Write-Host "$($_.Name): $([math]::Round($_.Length / 1MB, 2)) MB" }
```

### What If Shard Building Fails?

**Error:** `ModuleNotFoundError: No module named 'pyarrow'`
- Fix: `python -m pip install pyarrow`
- Try again

**Error:** `tokenizer/tokenizer.model` not found
- Fix: Check the path is correct and file exists
- Run: `Get-ChildItem tokenizer\tokenizer.model`

**Error:** `data/raw/*.jsonl` not found
- Fix: Verify raw data files exist using `Get-ChildItem data\raw\*.jsonl`
- All 5 source files must be present

**Error:** Exit code 1 with no message
- Fix: Try running with explicit Python path:
  ```powershell
  python -u -m data.pipeline --tokenizer-model tokenizer/tokenizer.model --output-dir data/processed --shard-size 16 --limit-per-source 10
  ```

This creates the processed parquet files from the raw text dataset already present in the repo.

```powershell
python -m data.pipeline --tokenizer-model tokenizer/tokenizer.model --output-dir data/processed --shard-size 16 --limit-per-source 10
```

What this command does:

- reads raw JSONL data
- tokenizes it with SentencePiece
- writes small parquet shards into `data/processed`

Check that shard files were created:

```powershell
Get-ChildItem data\processed
```

You should see files like:

- `manifest.json`
- `pipeline_summary.json`
- `shard-00000.parquet`
- `shard-00001.parquet`
- `shard-00002.parquet`

## 6. Model Configuration Files

### Tiny Model Config

These are the config files for the tiny local model:

- **Model config**: `configs/model/tiny.yaml` - Model architecture
- **Train schedule**: `configs/train/tiny.yaml` - Training hyperparameters

#### What `configs/model/tiny.yaml` Contains

```yaml
name: sage-tiny
num_layers: 4           # Very small (vs 24 for 1b)
d_model: 256            # Hidden dim (vs 2048 for 1b)
num_attn_heads: 4       # Attention heads
ffn_hidden_dim: 1024    # Feed-forward dimension
vocab_size: 4096        # Must match tokenizer
context_length: 1024    # Shorter context (vs 4096)
```

This is ~125M parameters instead of 1.2B.

####  What `configs/train/tiny.yaml` Contains

```yaml
total_tokens: 10240          # Very small dataset
peak_learning_rate: 0.003
warmup_steps: 100
checkpoint_interval: 10      # Save every 10 steps
log_interval: 1              # Log every step
eval_interval: 10
seed: 42
```

### Comparing Configs

| Setting | tiny.yaml | 1b.yaml |
|---------|----------|---------|
| Layers | 4 | 24 |
| Hidden dim | 256 | 2048 |
| Attention heads | 4 | 16 |
| Total params | ~125M | ~1.2B |
| Context length | 1024 | 4096 |
| Training time (CPU) | 2-5 min | 2-8 hours |
| Memory required | 2GB | 16GB+ |

## 7. Train The Tiny Local Model

### Understanding the Training Command

The complete training command for the tiny model:

```powershell
python -m train.trainer `
  --model-config configs/model/tiny.yaml `
  --schedule-config configs/train/tiny.yaml `
  --train-shards data/processed/shard-00000.parquet `
  --train-shards data/processed/shard-00001.parquet `
  --train-shards data/processed/shard-00002.parquet `
  --validation-shards data/processed/shard-00000.parquet `
  --validation-shards data/processed/shard-00001.parquet `
  --validation-shards data/processed/shard-00002.parquet `
  --output-dir runs/sage-tiny-local `
  --steps 20 `
  --disable-wandb
```

(Note: Use backticks to continue on next line in PowerShell)

**Command parameters:**
- `--model-config` - Which model architecture to use
- `--schedule-config` - Training schedules (LR, warmup, etc.)
- `--train-shards` - Training data files (can specify multiple)
- `--validation-shards` - Validation data files
- `--output-dir` - Where to save checkpoints
- `--steps` - How many training steps to run
- `--disable-wandb` - Don't use Weights & Biases logging

### Run Training

Execute the training command:

```powershell
python -m train.trainer `
  --model-config configs/model/tiny.yaml `
  --schedule-config configs/train/tiny.yaml `
  --train-shards data/processed/shard-00000.parquet data/processed/shard-00001.parquet data/processed/shard-00002.parquet `
  --validation-shards data/processed/shard-00000.parquet data/processed/shard-00001.parquet data/processed/shard-00002.parquet `
  --output-dir runs/sage-tiny-local `
  --steps 20 `
  --disable-wandb
```

### What Happens During Training

1. Model initializes (~2-3 seconds)
2. Datasets load (~2-3 seconds)
3. Training loop starts:
   - Each step: forward pass, loss calculation, backward pass, parameter update
   - Every step: log metrics to `metrics.jsonl`
   - Every 10 steps: save checkpoint file
4. Training completes, prints summary JSON

### Training Output

During training, you'll see logging output like:
```
Step 1/20: loss=5.234
Step 2/20: loss=5.156
Step 3/20: loss=5.087
...
Step 20/20: loss=4.821
```

**Final output at end (JSON):**

```json
{
  "output_dir": "runs/sage-tiny-local",
  "tokens_seen": 20480,
  "hardware": {
    "device": "cpu",
    "dtype": "torch.float32"
  }
}
```

### Verify Training Completed Successfully

Check that checkpoint files were created:

```powershell
Get-ChildItem runs\sage-tiny-local -Filter "ckpt_*.pt"
```

Expected output:
```
    Directory: C:\...\runs\sage-tiny-local

Mode          LastWriteTime         Length Name
----          --------              ------ ----
-a---  2024-04-14  11:35:00 PM         2.8M ckpt_step_0000010.pt
-a---  2024-04-14  11:35:05 PM         2.8M ckpt_step_0000020.pt
```

Check metrics were logged:

```powershell
Get-Content runs\sage-tiny-local\metrics.jsonl | head -3
```

Expected (first few lines):
```json
{"step": 1, "loss": 5.234, "tokens_seen": 1024, ...}
{"step": 2, "loss": 5.156, "tokens_seen": 2048, ...}
{"step": 3, "loss": 5.087, "tokens_seen": 3072, ...}
```

### What If Training Fails?

**Error:** `No GPU found, using CPU` (then hangs)
- This is normal - training on CPU is slow
- Wait 5-15 minutes per step
- Patient running in background is OK

**Error:** `RuntimeError: view size is not compatible`
- Fix: Use `configs/model/tiny.yaml` NOT `1b.yaml`

**Error:** `FileNotFoundError: data/processed/shard-*.parquet not found`
- Fix: Build shards first using `python -m data.pipeline ...`

**Error:** `memory error` or system becomes unresponsive
- This happens if running 1b model on small CPU
- Fix: Use `configs/model/tiny.yaml` instead
- Kill Python: `Stop-Process -Name python -Force`

### Training is Taking Too Long

If training is taking more than 15 minutes for 20 steps:

**Option 1:** Reduce steps for faster testing
```powershell
# Just 5 steps to test pipeline
python -m train.trainer ... --steps 5 ...
```

**Option 2:** Check if other processes are consuming CPU
```powershell
Get-Process | Sort-Object CPU -Descending | Select-Object -First 5
```

**Option 3:** It's normal - 2-5 minutes per step on CPU is expected

## 7. Train The Tiny Local Model

Run training:

```powershell
python -m train.trainer --model-config configs/model/tiny.yaml --schedule-config configs/train/tiny.yaml --train-shards data/processed/shard-00000.parquet data/processed/shard-00001.parquet data/processed/shard-00002.parquet --validation-shards data/processed/shard-00000.parquet data/processed/shard-00001.parquet data/processed/shard-00002.parquet --output-dir runs/sage-tiny-local --steps 20 --disable-wandb
```

What this does:

- loads the tiny model
- trains for 20 steps
- writes metrics to `runs/sage-tiny-local/metrics.jsonl`
- writes checkpoints every 10 steps

Expected output at the end looks like:

```json
{
  "output_dir": "runs/sage-tiny-local",
  "tokens_seen": 10240,
  "hardware": {
    "device": "cpu"
  }
}
```

Check the outputs:

```powershell
Get-ChildItem runs\sage-tiny-local
```

You should see:

- `ckpt_step_0000010.pt`
- `ckpt_step_0000020.pt`
- `metrics.jsonl`

## 8. Start The Tiny Model Server

### Using the Correct Server Launcher

**IMPORTANT:** Always use `python -m serve.start` - NOT raw `uvicorn`.

The `serve.start` launcher ensures:
- ✅ Correct model/checkpoint paths
- ✅ Environment variables set properly
- ✅ Tokenizer loaded before server starts
- ✅ Authentication enabled
- ✅ Proper port binding

### Start Command

In a **NEW PowerShell window**, run:

```powershell
python -m serve.start `
  --host 127.0.0.1 `
  --port 8000 `
  --model-config configs/model/tiny.yaml `
  --checkpoint-dir runs/sage-tiny-local `
  --tokenizer-model tokenizer/tokenizer.model
```

**Command parameters:**
- `--host` - Server address (127.0.0.1 = localhost only)
- `--port` - Port number (8000 default)
- `--model-config` - Which model to load
- `--checkpoint-dir` - Where to find the checkpoint
- `--tokenizer-model` - Path to tokenizer

### Expected Startup Output

When the server starts successfully:

```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     SAGE local URL: http://127.0.0.1:8000/
INFO:     SAGE login password: WklFOTMP6Zd7
INFO:     Started server process [12345]
INFO:     Application startup complete.
```

**IMPORTANT:** Save the password shown - you'll need it to log in!

### Server Components

Once running, the server provides:
- `http://127.0.0.1:8000/` - Login page
- `http://127.0.0.1:8000/chat` - Chat interface  
- `http://127.0.0.1:8000/health` - Health check API
- `http://127.0.0.1:8000/chat/status` - Model status API

### Do NOT Use This (Wrong Way)

```powershell
# ❌ WRONG - Will not work correctly
python -m uvicorn serve.server:app --host 127.0.0.1 --port 8000
```

This raw `uvicorn` command:
- ❌ Doesn't set environment variables
- ❌ May load wrong checkpoint
- ❌ May use wrong model config
- ❌ Will fail or load incorrect model

Always use `serve.start`.

### What If Server Fails to Start?

**Error:** `Port 8000 already in use`
- Fix 1: Use different port: `--port 8001`
- Fix 2: Kill existing process:
  ```powershell
  Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
  ```

**Error:** `FileNotFoundError: checkpoint not found`
- Fix: Make sure checkpoint exists:
  ```powershell
  Get-ChildItem runs\sage-tiny-local\ckpt_*.pt
  ```
- Training may not have finished

**Error:** `Model loaded but warning: checkpoint_loaded false`
- Fix: Verify checkpoint path matches:
  ```powershell
  python -m serve.start --checkpoint-dir runs/sage-tiny-local ...
  ```

**Error:** `ModuleNotFoundError: No module named 'serve'`
- Fix: Make sure you're in repo root:
  ```powershell
  cd "C:\Users\Lenovo\OneDrive\Desktop\Documents\LLM_MOdel"
  ```

## 9. Open The Browser UI

### Open Login Page

In your web browser, open:

```
http://127.0.0.1:8000/
```

### Enter Password

The password is displayed in the PowerShell terminal when the server starts:

```
SAGE login password: WklFOTMP6Zd7
```

Enter it and click "Login".

### Expected Browser Experience

After logging in:
- Home page displays "SAGE Chat Interface"
- "Chat" tab shows input field and "Send" button
- Previous messages display with timestamps
- Model responses initialize in chat interface

### Browzer Error Reference

| Error | Means | Fix |
|-------|-------|-----|
| `Connection refused` | Server not running | Start server with `serve.start` |
| `401 Unauthorized` | Wrong/missing password | Check password in terminal |
| `404 Not Found` | Page doesn't exist | Use `/` not specific page |
| `502 Bad Gateway` | Server crashed | Restart with `serve.start` |
| `Empty response` | Model still loading | Wait a few seconds |

## 10. Verify That The Tiny Model Is Actually Loaded

### Check Model Status Via API

In a **new PowerShell window** (don't stop the server), run:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/chat/status | ConvertTo-Json -Depth 6
```

### Expected Response

```json
{
  "available": true,
  "checkpoint_loaded": true,
  "checkpoint_dir": "C:\\Users\\Lenovo\\OneDrive\\Desktop\\Documents\\LLM_MOdel\\runs\\sage-tiny-local",
  "checkpoint_step": 20,
  "tokenizer_path": "tokenizer\\tokenizer.model",
  "warning": null
}
```

### Key Fields to Check

- `checkpoint_loaded` should be **`true`**
- `checkpoint_step` should match your training (e.g., 20)
- `checkpoint_dir` should end with `sage-tiny-local`
- `warning` should be **`null`** (empty)

###  Health Check

Also test the health endpoint:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Should return:

```json
{
  "status": "ok"
}
```

### What If Status Check Fails?

**Response:** `checkpoint_loaded: false`
- Problem: Checkpoint not loaded
- Fix: Restart server with correct `--checkpoint-dir`
- Verify file exists: `Get-ChildItem runs\sage-tiny-local\ckpt_*.pt`

**Response:** `error: port not reachable`
- Problem: Server not running
- Fix: Start with `python -m serve.start ...`

**Response:** `401 Unauthorized`
- Problem: Need authentication
- Fix: Log in via browser first
- OR use password header: `-Headers @{"X-Auth": "password"}`

## 11. Test Chat From Terminal (API Testing)

### Test Chat Endpoint

In a third PowerShell window, test the chat API:

```powershell
$body = @{prompt='Hello world'; max_new_tokens=12} | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/chat -Method Post -ContentType 'application/json' -Body $body | ConvertTo-Json -Depth 4
```

### Expected Response

```json
{
  "success": true,
  "response": "Hello world is a simple greeting. It can",
  "tokens_generated": 12,
  "generation_time_ms": 523
}
```

### Full API Test Suite

Test all 4 main API endpoints:

**1. Health check:**
```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```
Expected: `{ "status": "ok" }`

**2. Model status:**
```powershell
Invoke-RestMethod http://127.0.0.1:8000/chat/status
```
Expected: `{ ..., "checkpoint_loaded": true }`

**3. Chat with prompt:**
```powershell
$body = @{prompt='What is ML?'; max_new_tokens=20} | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/chat -Method Post -ContentType 'application/json' -Body $body
```
Expected: Response with generated text

**4. Generate from token IDs:**
```powershell
$body = @{input_ids=@(1,42,99); max_new_tokens=10} | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/generate -Method Post -ContentType 'application/json' -Body $body
```
Expected: List of generated token IDs

### Understanding API Responses

**Chat Response Structure:**
```json
{
  "success": true,           // Always true if request was valid
  "response": "text...",     // Generated text from model
  "tokens_generated": 12,    // How many tokens were generated
  "generation_time_ms": 523, // How long generation took
  "model": "sage-tiny"       // Model identifier
}
```

### Response Quality

The tiny model responses will:
- ✅ Be grammatically reasonable
- ✅ Continue the prompt mostly on-topic
- ✅ Generate without errors
- ⚠️ May be repetitive ("the the the")
- ⚠️ May lack semantic coherence
- ⚠️ Will end abruptly sometimes

This is **expected** for a 125M model trained on minimal data for 20 steps.

### What If Chat Fails?

**Error:** `Connection refused`
- Server not running
- Start with `python -m serve.start ...`

**Error:** `{"detail":"Unauthorized"}`
- Not authenticated
- Log in via browser first OR
- Use headers: `Invoke-RestMethod ... -Headers @{"Authorization"="Bearer token"}`

**Error:** `500 Internal Server Error`
- Model crashed during inference
- Check server terminal for errors
- Restart server

**Error:** `Empty response` or timeout
- Model running but slow (normal on CPU)
- Wait longer for response
- Or reduce `max_new_tokens`

## 12. Files That Matter Most

### Core config files

- `configs/model/tiny.yaml`
- `configs/train/tiny.yaml`

### Training output

- `runs/sage-tiny-local/ckpt_step_0000010.pt`
- `runs/sage-tiny-local/ckpt_step_0000020.pt`
- `runs/sage-tiny-local/metrics.jsonl`

### Server logs

- `runs/sage-tiny-local/server.log`
- `runs/sage-tiny-local/server.err.log`

### Main code paths

- `train/trainer.py`
- `train/hardware.py`
- `serve/start.py`
- `serve/server.py`
- `serve/static/index.html`

## 13. Exact Working Command Sequence

If you want the shortest correct sequence from zero to working, use this:

```powershell
cd "C:\Users\Lenovo\OneDrive\Desktop\Documents\LLM_MOdel"

python -m pip install --upgrade pip
python -m pip install torch fastapi uvicorn sentencepiece pyarrow pyyaml requests python-multipart pydantic psutil pytest httpx

python -m tokenizer.validate_tokenizer tokenizer\tokenizer.model

python -m data.pipeline --tokenizer-model tokenizer/tokenizer.model --output-dir data/processed --shard-size 16 --limit-per-source 10

python -m train.trainer --model-config configs/model/tiny.yaml --schedule-config configs/train/tiny.yaml --train-shards data/processed/shard-00000.parquet data/processed/shard-00001.parquet data/processed/shard-00002.parquet --validation-shards data/processed/shard-00000.parquet data/processed/shard-00001.parquet data/processed/shard-00002.parquet --output-dir runs/sage-tiny-local --steps 20 --disable-wandb

python -m serve.start --host 127.0.0.1 --port 8000 --model-config configs/model/tiny.yaml --checkpoint-dir runs/sage-tiny-local --tokenizer-model tokenizer/tokenizer.model
```

## 14. Troubleshooting

### Problem: `ModuleNotFoundError: No module named 'sentencepiece'`

Fix:

```powershell
python -m pip install sentencepiece
```

Then verify:

```powershell
python -c "import sentencepiece; print('sentencepiece ok')"
```

### Problem: `ModuleNotFoundError: No module named 'pyarrow'`

Fix:

```powershell
python -m pip install pyarrow
```

### Problem: `401 Unauthorized` in server logs

This is normal before login.

Fix:

- open the UI
- enter the generated password printed at startup

### Problem: `WinError 10048` port already in use

This means something is already using port `8000`.

Fix option 1:

- stop the running server with `Ctrl+C`

Fix option 2:

```powershell
Get-NetTCPConnection -LocalPort 8000 -State Listen
Stop-Process -Id <PID> -Force
```

Fix option 3:

Start on another port:

```powershell
python -m serve.start --host 127.0.0.1 --port 8001 --model-config configs/model/tiny.yaml --checkpoint-dir runs/sage-tiny-local --tokenizer-model tokenizer/tokenizer.model
```

### Problem: website starts but the wrong model is loaded

Symptom:

- `/chat/status` does not show `runs\sage-tiny-local`

Cause:

- server was started with raw `uvicorn` instead of `serve.start`

Fix:

```powershell
python -m serve.start --host 127.0.0.1 --port 8000 --model-config configs/model/tiny.yaml --checkpoint-dir runs/sage-tiny-local --tokenizer-model tokenizer/tokenizer.model
```

### Problem: `/chat` returns `500 Internal Server Error`

This happened earlier because incremental decode exceeded the tiny model’s context window.

Current fix already applied in the repo:

- the server now uses a sliding full-context decode path for stability

If you still see a 500:

1. stop the server
2. restart it with `python -m serve.start ...`
3. hard refresh the browser

### Problem: training takes forever on CPU

Cause:

- large models like `1b` are too big for a normal local CPU path

Fix:

- use `configs/model/tiny.yaml`
- use `configs/train/tiny.yaml`

### Problem: model replies are repetitive or nonsense

That is expected for a tiny local model trained on a tiny dataset for 20 steps.

This means:

- the pipeline works
- the model is loaded
- the quality is just low

To improve quality you need:

- more data
- more training steps
- ideally a GPU

## 15. What To Use Next Time

### If you just want the local demo working

Use:

- `configs/model/tiny.yaml`
- `configs/train/tiny.yaml`
- `runs/sage-tiny-local`

### If you want a better model later

Use a GPU environment and then increase:

- model size
- data volume
- training steps

Do not start with `1b` on local CPU if your goal is to finish quickly.

## 16. Final Summary

The stable local working path is:

1. install dependencies
2. validate tokenizer
3. build parquet shards
4. train `sage-tiny-local`
5. start with `python -m serve.start ...`
6. open `http://127.0.0.1:8000/`
7. log in with the generated password
8. verify `/chat/status`
9. use `/chat`

That is the complete beginner-safe path from zero to a working local SAGE model in this repo.
