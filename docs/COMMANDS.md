# SAGE Commands

This is the repo's current command reference for data preparation, tokenizer training, model training, serving, browser control, and validation.

## Install

```bash
pip install -r requirements.txt
```

## Run tests

```bash
pytest -q
```

## 1. Create a starter dataset

This repo does not ship a large training corpus. The fastest way to unblock the pipeline is to generate the built-in smoke dataset first:

```bash
python -m data.bootstrap --output-dir data/raw --overwrite
```

That writes JSONL files like:

```text
data/raw/general_web.jsonl
data/raw/code.jsonl
data/raw/math_science.jsonl
data/raw/multilingual.jsonl
data/raw/synthetic.jsonl
```

If you want to use your own corpus, put JSONL records in the same folder with at least a `text` field:

```json
{ "text": "your training sample here" }
```

## 2. Train the tokenizer

The tokenizer trainer now accepts plain text files or JSONL files.

```bash
python -m tokenizer.train_tokenizer \
  --input data/raw/general_web.jsonl data/raw/code.jsonl data/raw/math_science.jsonl data/raw/multilingual.jsonl data/raw/synthetic.jsonl \
  --model-prefix tokenizer/tokenizer \
  --vocab-size 4096 \
  --training-text tokenizer/training_corpus.txt
```

## 3. Validate the tokenizer

```bash
python -m tokenizer.validate_tokenizer tokenizer/tokenizer.model
```

## 4. Build parquet shards

```bash
python -m data.pipeline \
  --tokenizer-model tokenizer/tokenizer.model \
  --output-dir data/processed \
  --shard-size 128
```

For a short smoke run:

```bash
python -m data.pipeline \
  --tokenizer-model tokenizer/tokenizer.model \
  --output-dir data/processed \
  --shard-size 32 \
  --limit-per-source 4
```

The shell helper now points to the real data pipeline:

```bash
bash scripts/run_data_pipeline.sh --tokenizer-model tokenizer/tokenizer.model --output-dir data/processed
```

## 5. Start training

Smoke run:

```bash
python -m train.trainer \
  --model-config configs/model/1b.yaml \
  --schedule-config configs/train/schedule.yaml \
  --train-shards data/processed/shard-00000.parquet \
  --validation-shards data/processed/shard-00001.parquet \
  --output-dir runs/smoke \
  --steps 20 \
  --disable-wandb
```

Longer run:

```bash
python -m train.trainer \
  --model-config configs/model/1b.yaml \
  --schedule-config configs/train/schedule.yaml \
  --train-shards data/processed/shard-00000.parquet data/processed/shard-00001.parquet \
  --validation-shards data/processed/shard-00002.parquet \
  --output-dir runs/sage-1b
```

## 6. Serve the model

GPU/PyTorch server:

```bash
python -m serve.start --host 0.0.0.0 --port 8000
```

CPU control-plane server:

```bash
python -m serve.start --cpu --host 0.0.0.0 --port 8001
```

Helper scripts:

```bash
bash scripts/run_serve.sh
bash scripts/run_serve_cpu.sh
```

## 7. Browser control panel

Open the server root:

```text
http://127.0.0.1:8000/
```

The browser UI now supports:

- login with the random 12-character password printed in the terminal at server startup
- dataset bootstrap preset
- shard-building preset
- tokenizer/train/eval/server presets
- raw shell commands
- live job logs
- direct model chat through `/chat`

## 8. API commands

Health:

```bash
curl http://127.0.0.1:8000/health
```

Generate from token ids:

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d "{\"input_ids\": [1, 42, 99], \"max_new_tokens\": 8}"
```

Chat from text:

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d "{\"prompt\": \"Explain the training flow in this repo.\", \"max_new_tokens\": 64}"
```

Chat status:

```bash
curl http://127.0.0.1:8000/chat/status
```

## 9. Evaluation

```bash
python -m eval.run_benchmarks
```

Or use the helper:

```bash
bash scripts/run_eval.sh
```

## 10. Hugging Face sync

```bash
python hf_push.py
```
