"""
SAGE Data Pipeline
==================
Handles tokenization (tiktoken), streaming dataset loading from HuggingFace,
text cleaning, chunking into fixed-length sequences, and batched DataLoader
construction with shuffle buffering.
"""

import re
import random
import tiktoken
import torch
from torch.utils.data import IterableDataset, DataLoader
from typing import Iterator, List, Optional
from .config import SageConfig
from .utils import setup_logger

logger = setup_logger("sage.data")

# ---------------------------------------------------------------------------
# Tokenizer wrapper
# ---------------------------------------------------------------------------

class SageTokenizer:
    """Thin wrapper around tiktoken providing encode/decode and special tokens."""

    def __init__(self, encoding_name: str = "cl100k_base"):
        self.enc = tiktoken.get_encoding(encoding_name)
        self.encoding_name = encoding_name

        # Use the last token in the vocabulary as the EOS sentinel.
        # tiktoken doesn't expose a dedicated EOS, so we pick one that
        # won't collide with real text.
        self.eos_token_id: int = self.enc.n_vocab - 1
        self.pad_token_id: int = self.enc.n_vocab - 2
        self.vocab_size: int = self.enc.n_vocab

    def encode(self, text: str, add_eos: bool = False) -> List[int]:
        """Encode text to token IDs."""
        tokens = self.enc.encode(text, allowed_special="all")
        if add_eos:
            tokens.append(self.eos_token_id)
        return tokens

    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs back to text, filtering out special sentinel IDs."""
        # Filter out our custom pad/eos sentinels before decoding
        filtered = [t for t in tokens if t not in (self.eos_token_id, self.pad_token_id)]
        return self.enc.decode(filtered)

# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Strip HTML tags, collapse whitespace, and trim to reasonable length."""
    text = _HTML_TAG_RE.sub("", text)        # remove HTML tags
    text = _MULTI_SPACE_RE.sub(" ", text)     # collapse horizontal whitespace
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)  # collapse vertical whitespace
    return text.strip()

# ---------------------------------------------------------------------------
# Streaming iterable dataset
# ---------------------------------------------------------------------------

class StreamingTextDataset(IterableDataset):
    """
    An IterableDataset that streams data from HuggingFace ``datasets``,
    tokenizes on the fly, and yields fixed-length chunks.

    It maintains an internal shuffle buffer so that consecutive chunks are
    not always from the same document.
    """

    def __init__(
        self,
        dataset_name: str = "roneneldan/TinyStories",
        split: str = "train",
        seq_len: int = 512,
        tokenizer: Optional[SageTokenizer] = None,
        shuffle_buffer_size: int = 1000,
        text_field: str = "text",
        min_doc_len: int = 50,
        max_doc_len: int = 50000,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.split = split
        self.seq_len = seq_len
        self.tokenizer = tokenizer or SageTokenizer()
        self.shuffle_buffer_size = shuffle_buffer_size
        self.text_field = text_field
        self.min_doc_len = min_doc_len
        self.max_doc_len = max_doc_len

    def _stream_tokens(self) -> Iterator[int]:
        """Yields individual token IDs from the HuggingFace dataset stream."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "The 'datasets' library is required.  Install it with: "
                "pip install datasets"
            )

        logger.info(
            f"Streaming dataset '{self.dataset_name}' (split={self.split}) …"
        )
        ds = load_dataset(
            self.dataset_name,
            split=self.split,
            streaming=True,
            trust_remote_code=True,
        )

        for sample in ds:
            raw = sample.get(self.text_field, "")
            if not raw:
                continue

            text = clean_text(raw)

            # Filter documents that are too short or too long
            if len(text) < self.min_doc_len or len(text) > self.max_doc_len:
                continue

            tokens = self.tokenizer.encode(text, add_eos=True)
            yield from tokens

    def _chunk_tokens(self) -> Iterator[torch.Tensor]:
        """Groups raw token stream into fixed-length chunks of (seq_len + 1).

        The extra token is needed so that input = chunk[:-1] and
        target = chunk[1:] for next-token-prediction.
        """
        chunk: List[int] = []
        for tok in self._stream_tokens():
            chunk.append(tok)
            if len(chunk) == self.seq_len + 1:
                yield torch.tensor(chunk, dtype=torch.long)
                chunk = []
        # Discard any trailing partial chunk

    def __iter__(self) -> Iterator[torch.Tensor]:
        """Yields shuffled chunks from an internal buffer."""
        buffer: List[torch.Tensor] = []
        for chunk in self._chunk_tokens():
            buffer.append(chunk)
            if len(buffer) >= self.shuffle_buffer_size:
                random.shuffle(buffer)
                while len(buffer) > self.shuffle_buffer_size // 2:
                    yield buffer.pop()
        # Flush remaining items
        random.shuffle(buffer)
        yield from buffer

# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def create_dataloader(
    config: SageConfig,
    dataset_name: str = "roneneldan/TinyStories",
    split: str = "train",
    tokenizer: Optional[SageTokenizer] = None,
) -> DataLoader:
    """Creates a streaming DataLoader ready for the training loop."""
    tok = tokenizer or SageTokenizer()
    ds = StreamingTextDataset(
        dataset_name=dataset_name,
        split=split,
        seq_len=config.max_seq_len,
        tokenizer=tok,
    )
    return DataLoader(
        ds,
        batch_size=config.batch_size,
        num_workers=0,   # streaming datasets work best with 0 workers
        pin_memory=True,
        drop_last=True,
    )

# ---------------------------------------------------------------------------
# Instruction-tuning data helpers
# ---------------------------------------------------------------------------

INSTRUCTION_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n### Response:\n{response}"
)


def format_instruction_sample(instruction: str, response: str) -> str:
    """Formats an instruction/response pair into the chat template."""
    return INSTRUCTION_TEMPLATE.format(
        instruction=instruction.strip(),
        response=response.strip(),
    )


def create_instruction_batch(
    samples: List[dict],
    tokenizer: SageTokenizer,
    max_len: int = 512,
) -> dict:
    """
    Tokenize a list of {instruction, response} dicts and produce input_ids,
    labels, and a loss_mask that zeros out the instruction portion.

    Returns a dict with keys: input_ids, labels, loss_mask — all as tensors.
    """
    all_input_ids: List[List[int]] = []
    all_labels: List[List[int]] = []
    all_masks: List[List[int]] = []

    for sample in samples:
        instruction_text = f"### Instruction:\n{sample['instruction'].strip()}\n\n### Response:\n"
        response_text = sample["response"].strip()
        full_text = instruction_text + response_text

        instruction_tokens = tokenizer.encode(instruction_text)
        full_tokens = tokenizer.encode(full_text, add_eos=True)

        # Truncate to max_len
        full_tokens = full_tokens[:max_len]
        n_instruction = min(len(instruction_tokens), len(full_tokens))

        # Labels are the same as input shifted by 1 (handled by caller),
        # but we need a mask to zero out loss on instruction tokens.
        mask = [0] * n_instruction + [1] * (len(full_tokens) - n_instruction)

        # Pad to max_len
        pad_len = max_len - len(full_tokens)
        full_tokens += [tokenizer.pad_token_id] * pad_len
        mask += [0] * pad_len

        all_input_ids.append(full_tokens)
        all_labels.append(full_tokens)  # shift will be done in the loss fn
        all_masks.append(mask)

    return {
        "input_ids": torch.tensor(all_input_ids, dtype=torch.long),
        "labels": torch.tensor(all_labels, dtype=torch.long),
        "loss_mask": torch.tensor(all_masks, dtype=torch.float32),
    }
