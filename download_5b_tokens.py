"""
SAGE — 5 Billion Token Dataset Downloader
==========================================
Downloads ~5B tokens from free, public Hugging Face datasets into
data/raw/ as JSONL files ready for the SAGE training pipeline.

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
    print(f"Missing packages: {', '.join(missing)}")
    print(f"  Run:  pip install {' '.join(missing)}")
    sys.exit(1)


def estimate_tokens(text: str) -> int:
    """Fast approximation: 1 token ≈ 4 characters."""
    return max(1, len(text) // 4)


def human_tokens(n: int) -> str:
    if n < 1_000_000:
        return f"{n:,}"
    if n < 1_000_000_000:
        return f"{n / 1_000_000:.1f}M"
    return f"{n / 1_000_000_000:.2f}B"


def human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


class JSONLWriter:
    """Append-friendly JSONL writer that tracks token counts and can resume."""

    def __init__(self, path: Path, target_tokens: int, resume: bool = False):
        self.path = path
        self.target_tokens = target_tokens
        self.tokens_written = 0
        self.records_written = 0

        if resume and path.exists():
            print(f"  [resume] Counting existing tokens in {path.name}...")
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        self.tokens_written += estimate_tokens(record.get("text", ""))
                        self.records_written += 1
                    except json.JSONDecodeError:
                        pass
            print(f"  [resume] Found {human_tokens(self.tokens_written)} existing tokens")

        self._file = open(path, "a" if resume else "w", encoding="utf-8")

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
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ── Download jobs ──────────────────────────────────────────────────────────


def download_general_web(writer):
    """FineWeb — 2.5B tokens of clean deduplicated web text."""
    print("\n[1/5] general_web.jsonl — FineWeb (HuggingFaceFW/fineweb)")
    print("      Streaming web-crawl text...")
    bar = tqdm(total=writer.target_tokens, initial=writer.tokens_written,
               unit="tok", unit_scale=True, desc="  web tokens")
    ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT",
                      split="train", streaming=True)
    for sample in ds:
        if writer.done:
            break
        bar.update(writer.write({
            "text": sample["text"],
            "source": "fineweb",
            "url": sample.get("url", ""),
            "language": "en",
        }))
    bar.close()
    print(f"  ✓ {human_tokens(writer.tokens_written)} tokens | {writer.records_written:,} records")


def download_code(writer):
    """The Stack v2 — 1B tokens across 10 programming languages."""
    print("\n[2/5] code.jsonl — The Stack v2 (bigcode/the-stack-v2-train-smol-ids)")
    LANGUAGES = [
        ("python", "Python"), ("javascript", "JavaScript"), ("typescript", "TypeScript"),
        ("rust", "Rust"), ("go", "Go"), ("cpp", "C++"), ("java", "Java"),
        ("bash", "Bash"), ("sql", "SQL"), ("html", "HTML"),
    ]
    bar = tqdm(total=writer.target_tokens, initial=writer.tokens_written,
               unit="tok", unit_scale=True, desc="  code tokens")
    tokens_per_lang = writer.target_tokens // len(LANGUAGES)
    for lang_id, lang_name in LANGUAGES:
        if writer.done:
            break
        lang_tokens = 0
        print(f"    → {lang_name}...")
        try:
            ds = load_dataset("bigcode/the-stack-v2-train-smol-ids",
                              data_dir=f"data/{lang_id}", split="train",
                              streaming=True, trust_remote_code=True)
            for sample in ds:
                if writer.done or lang_tokens >= tokens_per_lang:
                    break
                content = sample.get("content", "") or sample.get("text", "")
                if not content:
                    continue
                t = writer.write({
                    "text": content,
                    "source": "the_stack_v2",
                    "language": lang_id,
                })
                bar.update(t)
                lang_tokens += t
        except Exception as e:
            print(f"    [warn] {lang_name} failed ({e}), skipping.")
    bar.close()
    print(f"  ✓ {human_tokens(writer.tokens_written)} tokens | {writer.records_written:,} records")


def download_math(writer):
    """OpenWebMath — 0.5B tokens of mathematical/scientific text."""
    print("\n[3/5] math_science.jsonl — OpenWebMath (open-web-math/open-web-math)")
    bar = tqdm(total=writer.target_tokens, initial=writer.tokens_written,
               unit="tok", unit_scale=True, desc="  math tokens")
    ds = load_dataset("open-web-math/open-web-math", split="train", streaming=True)
    for sample in ds:
        if writer.done:
            break
        bar.update(writer.write({
            "text": sample["text"],
            "source": "open_web_math",
            "url": sample.get("url", ""),
        }))
    bar.close()
    print(f"  ✓ {human_tokens(writer.tokens_written)} tokens | {writer.records_written:,} records")


def download_multilingual(writer):
    """Wikipedia dumps in 20 languages — 0.5B tokens."""
    print("\n[4/5] multilingual.jsonl — Wikipedia (wikimedia/wikipedia)")
    LANGUAGES = [
        ("en", "English"), ("es", "Spanish"), ("fr", "French"), ("de", "German"),
        ("zh", "Chinese"), ("ja", "Japanese"), ("pt", "Portuguese"), ("ar", "Arabic"),
        ("ru", "Russian"), ("hi", "Hindi"), ("it", "Italian"), ("ko", "Korean"),
        ("nl", "Dutch"), ("pl", "Polish"), ("sv", "Swedish"), ("tr", "Turkish"),
        ("vi", "Vietnamese"), ("id", "Indonesian"), ("uk", "Ukrainian"), ("fa", "Persian"),
    ]
    bar = tqdm(total=writer.target_tokens, initial=writer.tokens_written,
               unit="tok", unit_scale=True, desc="  multilingual tokens")
    tokens_per_lang = writer.target_tokens // len(LANGUAGES)
    for lang_code, lang_name in LANGUAGES:
        if writer.done:
            break
        lang_tokens = 0
        try:
            ds = load_dataset("wikimedia/wikipedia", f"20231101.{lang_code}",
                              split="train", streaming=True, trust_remote_code=True)
            for sample in ds:
                if writer.done or lang_tokens >= tokens_per_lang:
                    break
                text = sample.get("text", "")
                if not text:
                    continue
                t = writer.write({
                    "text": text,
                    "source": "wikipedia",
                    "language": lang_code,
                    "title": sample.get("title", ""),
                })
                bar.update(t)
                lang_tokens += t
        except Exception as e:
            print(f"\n    [warn] Wikipedia {lang_name} failed: {e}")
    bar.close()
    print(f"  ✓ {human_tokens(writer.tokens_written)} tokens | {writer.records_written:,} records")


def download_synthetic(writer):
    """OpenHermes 2.5 — 0.5B tokens of instruction-following data."""
    print("\n[5/5] synthetic.jsonl — OpenHermes 2.5 (teknium/OpenHermes-2.5)")
    bar = tqdm(total=writer.target_tokens, initial=writer.tokens_written,
               unit="tok", unit_scale=True, desc="  synthetic tokens")
    ds = load_dataset("teknium/OpenHermes-2.5", split="train", streaming=True)
    rounds = 0
    while not writer.done and rounds < 10:
        for sample in ds:
            if writer.done:
                break
            convs = sample.get("conversations", [])
            parts = []
            for turn in convs:
                role, value = turn.get("from", ""), turn.get("value", "")
                if role == "human":
                    parts.append(f"### Instruction\n{value}")
                elif role == "gpt":
                    parts.append(f"### Response\n{value}")
            text = "\n\n".join(parts) or sample.get("text", "")
            if not text:
                continue
            bar.update(writer.write({
                "text": text,
                "source": "openhermes_2.5",
                "task": "instruction_following",
            }))
        rounds += 1
    bar.close()
    print(f"  ✓ {human_tokens(writer.tokens_written)} tokens | {writer.records_written:,} records")


# ── Main ──────────────────────────────────────────────────────────────────

TARGETS = {
    "general_web.jsonl": 2_500_000_000,
    "code.jsonl": 1_000_000_000,
    "math_science.jsonl": 500_000_000,
    "multilingual.jsonl": 500_000_000,
    "synthetic.jsonl": 500_000_000,
}

DOWNLOADERS = {
    "general_web.jsonl": download_general_web,
    "code.jsonl": download_code,
    "math_science.jsonl": download_math,
    "multilingual.jsonl": download_multilingual,
    "synthetic.jsonl": download_synthetic,
}


def main():
    parser = argparse.ArgumentParser(description="Download ~5B tokens for SAGE training.")
    parser.add_argument("--output-dir", default="data/raw",
                        help="Where to write JSONL files (default: data/raw)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip files already at target token count")
    parser.add_argument("--only", nargs="+", choices=list(TARGETS.keys()),
                        help="Download only specific files")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Scale all targets (e.g. 0.01 for a quick 1%% test)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files_to_run = args.only or list(TARGETS.keys())
    total_target = sum(int(TARGETS[f] * args.scale) for f in files_to_run)

    print("\n" + "=" * 60)
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
            print(f"  Time: {elapsed / 60:.1f} min  |  Size: {human_bytes(size)}")

    elapsed_total = time.time() - grand_start
    print("\n" + "=" * 60)
    print(f"  Done! {human_tokens(grand_tokens)} tokens total")
    print(f"  Total time: {elapsed_total / 3600:.2f} hours")
    print(f"  Files: {out_dir.resolve()}/")
    print("=" * 60)
    print("\nNext steps:")
    print("  python -m tokenizer.train_tokenizer \\")
    print("    --input data/raw/general_web.jsonl data/raw/code.jsonl \\")
    print("           data/raw/math_science.jsonl data/raw/multilingual.jsonl \\")
    print("           data/raw/synthetic.jsonl \\")
    print("    --model-prefix tokenizer/tokenizer --vocab-size 32000")


if __name__ == "__main__":
    main()
