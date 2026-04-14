# SAGE — 5 Billion Token Dataset Downloader

Automatically downloads ~5B tokens from free, public Hugging Face datasets and saves them as JSONL files in your `data/raw/` directory, fully compatible with the SAGE training pipeline.

---

## Token Budget

| File                 | Source                                            | Tokens           |
| -------------------- | ------------------------------------------------- | ---------------- |
| `general_web.jsonl`  | FineWeb                                           | 2.5B             |
| `code.jsonl`         | The Stack v2 (Python, JS, Rust, Go, C++ and more) | 1.0B             |
| `math_science.jsonl` | OpenWebMath                                       | 0.5B             |
| `multilingual.jsonl` | Wikipedia (20+ languages)                         | 0.5B             |
| `synthetic.jsonl`    | OpenHermes 2.5 (instruction data)                 | 0.5B             |
| **Total**            |                                                   | **~5.0B tokens** |

**Estimated disk space:** ~20–25 GB  
**Estimated download time:** 2–8 hours depending on your connection  
**Cost:** 100% free, no account required

---

## Requirements

### System

- Python 3.9+
- 25 GB free disk space
- Stable internet connection

### Python packages

```bash
pip install datasets huggingface_hub tqdm
```

---

## Usage

### Basic — download everything

```bash
python download_5b_tokens.py --output-dir data/raw
```

### Test run — 1% of data to verify everything works

```bash
python download_5b_tokens.py --output-dir data/raw --scale 0.01
```

### Resume — continue after an internet cutout

```bash
python download_5b_tokens.py --output-dir data/raw --resume
```

### Download only one specific file

```bash
python download_5b_tokens.py --output-dir data/raw --only code.jsonl
```

### Download multiple specific files

```bash
python download_5b_tokens.py --output-dir data/raw --only code.jsonl math_science.jsonl
```

---

## All Flags

| Flag           | Default    | Description                                                    |
| -------------- | ---------- | -------------------------------------------------------------- |
| `--output-dir` | `data/raw` | Directory where JSONL files are saved                          |
| `--resume`     | off        | Skip files that already hit their token target                 |
| `--only`       | all files  | Download only the specified file(s)                            |
| `--scale`      | `1.0`      | Scale all token targets (e.g. `0.1` = 10% of 5B = 500M tokens) |

---

## Output Format

Every record written to disk follows this structure with at minimum a `text` field, making it directly compatible with the SAGE pipeline:

```json
{ "text": "your training sample here", "source": "fineweb", "language": "en" }
```

---

## Data Sources

### 1. FineWeb — `general_web.jsonl`

- **Dataset:** `HuggingFaceFW/fineweb` (sample-10BT subset)
- **What it is:** A pre-shuffled, deduplicated 10B-token slice of web-crawl text, one of the cleanest freely available web datasets
- **Why it's used:** Broad general language coverage, essential for fluent text generation

### 2. The Stack v2 — `code.jsonl`

- **Dataset:** `bigcode/the-stack-v2-train-smol-ids`
- **What it is:** Source code across 10 programming languages: Python, JavaScript, TypeScript, Rust, Go, C++, Java, Bash, SQL, and HTML
- **Why it's used:** Teaches the model programming syntax, logic, and structure

### 3. OpenWebMath — `math_science.jsonl`

- **Dataset:** `open-web-math/open-web-math`
- **What it is:** 14.7B tokens of mathematical content extracted from the web, including LaTeX, proofs, and problem sets
- **Why it's used:** Improves numerical reasoning and scientific language understanding

### 4. Wikipedia — `multilingual.jsonl`

- **Dataset:** `wikimedia/wikipedia` (20231101 dumps)
- **Languages:** English, Spanish, French, German, Chinese, Japanese, Portuguese, Arabic, Russian, Hindi, Italian, Korean, Dutch, Polish, Swedish, Turkish, Vietnamese, Indonesian, Ukrainian, Persian
- **Why it's used:** Clean, factual, encyclopedic text across 20 languages

### 5. OpenHermes 2.5 — `synthetic.jsonl`

- **Dataset:** `teknium/OpenHermes-2.5`
- **What it is:** ~1M high-quality instruction-following pairs formatted as `### Instruction` / `### Response` conversations
- **Why it's used:** Teaches the model to follow instructions and produce structured, helpful responses

---

## What Happens After Download

Once all files are ready, continue with the standard SAGE pipeline:

### Train the tokenizer

```bash
python -m tokenizer.train_tokenizer \
  --input data/raw/general_web.jsonl \
          data/raw/code.jsonl \
          data/raw/math_science.jsonl \
          data/raw/multilingual.jsonl \
          data/raw/synthetic.jsonl \
  --model-prefix tokenizer/tokenizer \
  --vocab-size 32000
```

### Build parquet shards

```bash
python -m data.pipeline \
  --tokenizer-model tokenizer/tokenizer.model \
  --output-dir data/processed \
  --shard-size 128
```

### Start training

```bash
python -m train.trainer \
  --model-config configs/model/1b.yaml \
  --schedule-config configs/train/schedule.yaml \
  --train-shards data/processed/shard-00000.parquet \
  --validation-shards data/processed/shard-00001.parquet \
  --output-dir runs/sage-1b
```

---

## Troubleshooting

**Download stalls or disconnects**  
Run with `--resume` to pick up exactly where you left off. The writer appends to existing files and counts already-written tokens before continuing.

**A specific language or dataset fails**  
The downloader catches errors per-source and logs a warning, then moves on. The other files are unaffected. Re-run with `--only <filename> --resume` to retry just that file.

**Running out of disk space mid-download**  
Use `--scale 0.5` to target 2.5B tokens total (~10–12 GB) instead of the full 5B. The model will be slightly less capable but the pipeline will still work end to end.

**Slow download speed**  
All datasets are streamed — data is downloaded and written record by record, so you never need to load the entire dataset at once. If speed is consistently low, try running overnight or on a cloud VM closer to Hugging Face's CDN.

---

## Full Script

```python
"""
SAGE — 5 Billion Token Dataset Downloader
==========================================
Downloads ~5B tokens from free Hugging Face datasets and saves them
as JSONL files in your data/raw/ directory, ready for the SAGE pipeline.

Token budget breakdown:
  general_web.jsonl    →  2.5B tokens  (FineWeb)
  code.jsonl           →  1.0B tokens  (The Stack v2 - Python, JS, Rust, Go, C++)
  math_science.jsonl   →  0.5B tokens  (OpenWebMath)
  multilingual.jsonl   →  0.5B tokens  (Wikipedia 20+ languages)
  synthetic.jsonl      →  0.5B tokens  (OpenHermes instruction data)
  ─────────────────────────────────────
  TOTAL                →  ~5.0B tokens

Usage:
  pip install datasets huggingface_hub tqdm
  python download_5b_tokens.py --output-dir data/raw
  python download_5b_tokens.py --output-dir data/raw --resume
"""

import argparse
import json
import sys
import time
from pathlib import Path

missing = []
try:
    from datasets import load_dataset
except ImportError:
    missing.append("datasets")
try:
    from tqdm import tqdm
except ImportError:
    missing.append("tqdm")

if missing:
    print(f"[ERROR] Missing packages: {', '.join(missing)}")
    print(f"  Run:  pip install {' '.join(missing)}")
    sys.exit(1)


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)

def human_tokens(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n/1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    return f"{n:,}"

def human_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


class JSONLWriter:
    def __init__(self, path: Path, target_tokens: int, resume: bool = False):
        self.path = path
        self.target_tokens = target_tokens
        self.tokens_written = 0
        self.records_written = 0

        if resume and path.exists():
            print(f"  [resume] Counting existing tokens in {path.name}...")
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        self.tokens_written += estimate_tokens(rec.get("text", ""))
                        self.records_written += 1
                    except json.JSONDecodeError:
                        pass
            print(f"  [resume] Already have {human_tokens(self.tokens_written)} / {human_tokens(target_tokens)}")
            self._file = open(path, "a", encoding="utf-8", buffering=1024 * 1024)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(path, "w", encoding="utf-8", buffering=1024 * 1024)

    @property
    def done(self) -> bool:
        return self.tokens_written >= self.target_tokens

    def write(self, record: dict) -> int:
        text = record.get("text", "")
        if not text or len(text.strip()) < 50:
            return 0
        toks = estimate_tokens(text)
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.tokens_written += toks
        self.records_written += 1
        return toks

    def close(self):
        self._file.flush()
        self._file.close()

    def __enter__(self): return self
    def __exit__(self, *_): self.close()


def download_general_web(writer):
    print("\n[1/5] general_web.jsonl — FineWeb")
    bar = tqdm(total=writer.target_tokens, initial=writer.tokens_written,
               unit="tok", unit_scale=True, desc="  web tokens")
    ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT",
                      split="train", streaming=True)
    for sample in ds:
        if writer.done: break
        bar.update(writer.write({"text": sample["text"], "source": "fineweb",
                                  "url": sample.get("url", ""), "language": "en"}))
    bar.close()
    print(f"  ✓ {human_tokens(writer.tokens_written)} tokens | {writer.records_written:,} records")


def download_code(writer):
    print("\n[2/5] code.jsonl — The Stack v2")
    LANGUAGES = [("python","Python"),("javascript","JavaScript"),("typescript","TypeScript"),
                 ("rust","Rust"),("go","Go"),("cpp","C++"),("java","Java"),
                 ("bash","Bash"),("sql","SQL"),("html","HTML")]
    bar = tqdm(total=writer.target_tokens, initial=writer.tokens_written,
               unit="tok", unit_scale=True, desc="  code tokens")
    tokens_per_lang = writer.target_tokens // len(LANGUAGES)
    for lang_id, lang_name in LANGUAGES:
        if writer.done: break
        lang_tokens = 0
        print(f"    → {lang_name}...")
        try:
            ds = load_dataset("bigcode/the-stack-v2-train-smol-ids",
                              data_dir=f"data/{lang_id}", split="train",
                              streaming=True, trust_remote_code=True)
            for sample in ds:
                if writer.done or lang_tokens >= tokens_per_lang: break
                content = sample.get("content", "") or sample.get("text", "")
                if not content: continue
                t = writer.write({"text": content, "source": "the_stack_v2",
                                   "language": lang_id})
                bar.update(t); lang_tokens += t
        except Exception as e:
            print(f"    [warn] {lang_name} failed ({e}), skipping.")
    bar.close()
    print(f"  ✓ {human_tokens(writer.tokens_written)} tokens | {writer.records_written:,} records")


def download_math(writer):
    print("\n[3/5] math_science.jsonl — OpenWebMath")
    bar = tqdm(total=writer.target_tokens, initial=writer.tokens_written,
               unit="tok", unit_scale=True, desc="  math tokens")
    ds = load_dataset("open-web-math/open-web-math", split="train", streaming=True)
    for sample in ds:
        if writer.done: break
        bar.update(writer.write({"text": sample["text"], "source": "open_web_math",
                                  "url": sample.get("url", "")}))
    bar.close()
    print(f"  ✓ {human_tokens(writer.tokens_written)} tokens | {writer.records_written:,} records")


def download_multilingual(writer):
    print("\n[4/5] multilingual.jsonl — Wikipedia (20 languages)")
    LANGUAGES = [("en","English"),("es","Spanish"),("fr","French"),("de","German"),
                 ("zh","Chinese"),("ja","Japanese"),("pt","Portuguese"),("ar","Arabic"),
                 ("ru","Russian"),("hi","Hindi"),("it","Italian"),("ko","Korean"),
                 ("nl","Dutch"),("pl","Polish"),("sv","Swedish"),("tr","Turkish"),
                 ("vi","Vietnamese"),("id","Indonesian"),("uk","Ukrainian"),("fa","Persian")]
    bar = tqdm(total=writer.target_tokens, initial=writer.tokens_written,
               unit="tok", unit_scale=True, desc="  multilingual tokens")
    tokens_per_lang = writer.target_tokens // len(LANGUAGES)
    for lang_code, lang_name in LANGUAGES:
        if writer.done: break
        lang_tokens = 0
        try:
            ds = load_dataset("wikimedia/wikipedia", f"20231101.{lang_code}",
                              split="train", streaming=True, trust_remote_code=True)
            for sample in ds:
                if writer.done or lang_tokens >= tokens_per_lang: break
                text = sample.get("text", "")
                if not text: continue
                t = writer.write({"text": text, "source": "wikipedia",
                                   "language": lang_code, "title": sample.get("title","")})
                bar.update(t); lang_tokens += t
        except Exception as e:
            print(f"\n    [warn] Wikipedia {lang_name} failed: {e}")
    bar.close()
    print(f"  ✓ {human_tokens(writer.tokens_written)} tokens | {writer.records_written:,} records")


def download_synthetic(writer):
    print("\n[5/5] synthetic.jsonl — OpenHermes 2.5")
    bar = tqdm(total=writer.target_tokens, initial=writer.tokens_written,
               unit="tok", unit_scale=True, desc="  synthetic tokens")
    ds = load_dataset("teknium/OpenHermes-2.5", split="train", streaming=True)
    rounds = 0
    while not writer.done and rounds < 10:
        for sample in ds:
            if writer.done: break
            convs = sample.get("conversations", [])
            parts = []
            for turn in convs:
                role, value = turn.get("from",""), turn.get("value","")
                if role == "human":   parts.append(f"### Instruction\n{value}")
                elif role == "gpt":   parts.append(f"### Response\n{value}")
            text = "\n\n".join(parts) or sample.get("text","")
            if not text: continue
            bar.update(writer.write({"text": text, "source": "openhermes_2.5",
                                      "task": "instruction_following"}))
        rounds += 1
    bar.close()
    print(f"  ✓ {human_tokens(writer.tokens_written)} tokens | {writer.records_written:,} records")


TARGETS = {
    "general_web.jsonl":  2_500_000_000,
    "code.jsonl":         1_000_000_000,
    "math_science.jsonl":   500_000_000,
    "multilingual.jsonl":   500_000_000,
    "synthetic.jsonl":      500_000_000,
}
DOWNLOADERS = {
    "general_web.jsonl":  download_general_web,
    "code.jsonl":         download_code,
    "math_science.jsonl": download_math,
    "multilingual.jsonl": download_multilingual,
    "synthetic.jsonl":    download_synthetic,
}


def main():
    parser = argparse.ArgumentParser(description="Download ~5B tokens for SAGE training.")
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--only", nargs="+", choices=list(TARGETS.keys()))
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files_to_run = args.only or list(TARGETS.keys())
    total_target = sum(int(TARGETS[f] * args.scale) for f in files_to_run)

    print("=" * 60)
    print("  SAGE — 5 Billion Token Downloader")
    print("=" * 60)
    print(f"  Output dir : {out_dir.resolve()}")
    print(f"  Resume     : {args.resume}")
    print(f"  Scale      : {args.scale}x")
    print(f"  Target     : {human_tokens(total_target)} tokens")
    print(f"  Est. disk  : ~{total_target // 40_000_000} GB")
    print("=" * 60)

    grand_start = time.time()
    grand_tokens = 0

    for filename in files_to_run:
        target = int(TARGETS[filename] * args.scale)
        with JSONLWriter(out_dir / filename, target, resume=args.resume) as writer:
            if writer.done:
                print(f"\n[skip] {filename} already complete ({human_tokens(writer.tokens_written)} tokens)")
                grand_tokens += writer.tokens_written
                continue
            t0 = time.time()
            DOWNLOADERS[filename](writer)
            elapsed = time.time() - t0
            grand_tokens += writer.tokens_written
            size = (out_dir / filename).stat().st_size
            print(f"  Time: {elapsed/60:.1f} min  |  Size: {human_bytes(size)}")

    elapsed_total = time.time() - grand_start
    print("\n" + "=" * 60)
    print(f"  DONE — {human_tokens(grand_tokens)} tokens downloaded")
    print(f"  Total time: {elapsed_total/3600:.2f} hours")
    print(f"  Files: {out_dir.resolve()}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
```
