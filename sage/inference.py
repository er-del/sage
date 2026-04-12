"""
SAGE Inference Engine
=====================
Text generation with greedy, temperature, top-k, and nucleus (top-p) sampling.
Supports KV-cache for O(1)-per-token generation and streaming output.
"""

import sys
import torch
import torch.nn.functional as F
from typing import Optional, List

from .config import SageConfig
from .model import SageModel
from .data import SageTokenizer
from .utils import setup_logger

logger = setup_logger("sage.inference")


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def _top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Zero out all logits outside the top-k highest values."""
    if k <= 0 or k >= logits.size(-1):
        return logits
    values, _ = torch.topk(logits, k)
    min_val = values[:, -1].unsqueeze(-1)
    return torch.where(logits < min_val, torch.full_like(logits, float("-inf")), logits)


def _top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus sampling: keep the smallest set of tokens whose cumulative
    probability exceeds *p*."""
    if p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Identify tokens to remove (cumulative prob exceeds p)
    sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= p
    sorted_logits[sorted_mask] = float("-inf")

    # Scatter back to original order
    logits = logits.scatter(1, sorted_idx, sorted_logits)
    return logits


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    greedy: bool = False,
) -> torch.Tensor:
    """
    Given raw logits for the last position, sample or greedily select the
    next token.

    Parameters
    ----------
    logits : Tensor  [batch, vocab]
    temperature : float
    top_k : int
    top_p : float
    greedy : bool — if True, ignore temperature/top-k/top-p and pick argmax.

    Returns
    -------
    Tensor  [batch, 1]
    """
    if greedy:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / max(temperature, 1e-8)
    logits = _top_k_filter(logits, top_k)
    logits = _top_p_filter(logits, top_p)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(
    model: SageModel,
    tokenizer: SageTokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    greedy: bool = False,
    stream: bool = True,
    device: Optional[torch.device] = None,
) -> str:
    """
    Generate text from *prompt* using the SAGE model.

    Parameters
    ----------
    model : SageModel
    tokenizer : SageTokenizer
    prompt : str
    max_new_tokens : int
    temperature, top_k, top_p : sampling hyper-parameters
    greedy : bool — use argmax decoding
    stream : bool — print tokens as they are generated
    device : torch.device

    Returns
    -------
    str — the complete generated text (prompt + new tokens).
    """
    if device is None:
        device = next(model.parameters()).device

    base_model = getattr(model, "module", model)
    base_model.eval()

    # Encode prompt
    prompt_tokens = tokenizer.encode(prompt)
    if not prompt_tokens:
        prompt_tokens = [tokenizer.eos_token_id]

    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

    generated_tokens: List[int] = list(prompt_tokens)
    kv_caches = None

    # --- Prefill: run the full prompt through the model once ---
    logits, kv_caches = base_model(input_ids)
    next_logits = logits[:, -1, :]

    for _ in range(max_new_tokens):
        next_id = sample_next_token(
            next_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            greedy=greedy,
        )

        token_id = next_id.item()

        # Stop on EOS
        if token_id == tokenizer.eos_token_id:
            break

        generated_tokens.append(token_id)

        # Stream output: decode and print only the new token
        if stream:
            token_str = tokenizer.decode([token_id])
            print(token_str, end="", flush=True)

        # --- Decode step: feed only the new token, reuse KV-cache ---
        next_input = next_id.view(1, 1)
        logits, kv_caches = base_model(next_input, kv_caches=kv_caches)
        next_logits = logits[:, -1, :]

    if stream:
        print()  # newline after streaming completes

    base_model.train()
    return tokenizer.decode(generated_tokens)
