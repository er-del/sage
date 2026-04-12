#!/usr/bin/env python3
"""
SAGE — Self-Adaptive General Engine (Single-File Edition)
=========================================================
A complete mini-LLM in one file.  Run with:

    python sage_single.py

All architecture, data, training, inference, fine-tuning, quantization,
RAG, and CLI components are included below.
"""

import os
import re
import sys
import math
import copy
import time
import random
import logging
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from torch.amp import GradScaler, autocast
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm
import tiktoken
import wandb

__version__ = "1.0.0"


# ===================================================================
# Section 1 — Configuration
# ===================================================================

@dataclass
class SageConfig:
    d_model: int = 512
    n_heads: int = 8
    n_kv_heads: int = 4
    n_layers: int = 6
    d_ff: int = 2048
    n_experts: int = 4
    num_experts_per_tok: int = 2
    vocab_size: int = 100277
    max_seq_len: int = 1024
    dropout: float = 0.1
    batch_size: int = 4
    gradient_accumulation_steps: int = 16
    learning_rate: float = 3e-4
    min_learning_rate: float = 1e-5
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    checkpoint_dir: str = "checkpoints"
    project_name: str = "sage-v2"

    @property
    def device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")


# ===================================================================
# Section 2 — Logging & Checkpoint Utilities
# ===================================================================

def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(h)
    logger.propagate = False
    return logger

logger = setup_logger("sage")

def save_checkpoint(model, optimizer, step, checkpoint_dir, filename="sage_latest.pt"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)
    base = getattr(model, "module", model)
    ckpt = {"step": step, "model_state_dict": base.state_dict()}
    if optimizer is not None:
        ckpt["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(ckpt, path)
    return path

def load_checkpoint(model, optimizer, checkpoint_dir, filename="sage_latest.pt", device="cpu"):
    path = os.path.join(checkpoint_dir, filename)
    if not os.path.exists(path):
        logger.warning(f"No checkpoint at {path}, starting fresh.")
        return model, optimizer, 0
    ckpt = torch.load(path, map_location=device)
    base = getattr(model, "module", model)
    base.load_state_dict(ckpt["model_state_dict"], strict=False)
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    step = ckpt.get("step", 0)
    logger.info(f"Loaded checkpoint from {path} (step {step})")
    return model, optimizer, step


# ===================================================================
# Section 3 — Tokenizer
# ===================================================================

class SageTokenizer:
    def __init__(self, encoding_name="cl100k_base"):
        self.enc = tiktoken.get_encoding(encoding_name)
        self.eos_token_id = self.enc.n_vocab - 1
        self.pad_token_id = self.enc.n_vocab - 2
        self.vocab_size = self.enc.n_vocab

    def encode(self, text, add_eos=False):
        tokens = self.enc.encode(text, allowed_special="all")
        if add_eos:
            tokens.append(self.eos_token_id)
        return tokens

    def decode(self, tokens):
        filtered = [t for t in tokens if t not in (self.eos_token_id, self.pad_token_id)]
        return self.enc.decode(filtered)


# ===================================================================
# Section 4 — Model Architecture (RoPE, Attention, MoE, Transformer)
# ===================================================================

def precompute_freqs_cis(dim, end, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
    t = torch.arange(end, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)

def apply_rotary_emb(xq, xk, freqs_cis):
    # Ensure freqs_cis is complex (DataParallel can sometimes replicate it as real)
    if not torch.is_complex(freqs_cis) and freqs_cis.shape[-1] == 2:
        freqs_cis = torch.view_as_complex(freqs_cis)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    fc = freqs_cis.unsqueeze(0).unsqueeze(2)
    xq_out = torch.view_as_real(xq_ * fc).flatten(3)
    xk_out = torch.view_as_real(xk_ * fc).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def repeat_kv(x, n_rep):
    if n_rep == 1: return x
    B, T, n_kv_heads, head_dim = x.size()
    return x[:, :, :, None, :].expand(B, T, n_kv_heads, n_rep, head_dim).reshape(B, T, n_kv_heads * n_rep, head_dim)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.n_heads
        self.wq = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.d_model, config.d_model, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x, freqs_cis, kv_cache=None):
        B, T, C = x.size()
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        q = q.view(B, T, self.n_heads, self.head_dim)
        k = k.view(B, T, self.n_kv_heads, self.head_dim)
        v = v.view(B, T, self.n_kv_heads, self.head_dim)
        q, k = apply_rotary_emb(q, k, freqs_cis)
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=1)
            v = torch.cat([kv_cache[1], v], dim=1)
            new_kv = (k, v)
        else:
            new_kv = None
        k, v = repeat_kv(k, self.n_rep), repeat_kv(v, self.n_rep)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        is_causal = kv_cache is None and T > 1
        try:
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0 if not self.training else 0.1, is_causal=is_causal)
        except Exception:
            attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
            if is_causal:
                mask = torch.tril(torch.ones(T, T, device=q.device)).view(1, 1, T, T)
                attn = attn.masked_fill(mask == 0, float('-inf'))
            attn = F.softmax(attn, dim=-1)
            if self.training: attn = F.dropout(attn, p=0.1)
            y = attn @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.wo(y)), new_kv

class ExpertFFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w2 = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.w3 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class MoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_experts = config.n_experts
        self.top_k = config.num_experts_per_tok
        self.router = nn.Linear(config.d_model, config.n_experts, bias=False)
        self.experts = nn.ModuleList([ExpertFFN(config) for _ in range(config.n_experts)])

    def forward(self, x):
        B, T, C = x.size()
        flat = x.view(-1, C)
        weights = F.softmax(self.router(flat), dim=-1)
        weights, indices = torch.topk(weights, self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        out = torch.zeros_like(flat)
        for i, expert in enumerate(self.experts):
            mask = (indices == i)
            tok_idx, kth = torch.where(mask)
            if tok_idx.shape[0] > 0:
                out[tok_idx] += expert(flat[tok_idx]) * weights[tok_idx, kth].unsqueeze(-1)
        return out.view(B, T, C)

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.moe = MoE(config)

    def forward(self, x, freqs_cis, kv_cache=None):
        h, new_kv = self.attn(self.norm1(x), freqs_cis, kv_cache)
        x = x + h
        x = x + self.moe(self.norm2(x))
        return x, new_kv

class SageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)  # tied
        self.wte.weight = self.lm_head.weight
        self.register_buffer("freqs_cis", precompute_freqs_cis(config.d_model // config.n_heads, config.max_seq_len * 2), persistent=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, idx, kv_caches=None):
        B, T = idx.size()
        start = kv_caches[0][0].shape[1] if kv_caches else 0
        fc = self.freqs_cis[start:start + T]
        x = self.drop(self.wte(idx))
        new_kvs = []
        for i, layer in enumerate(self.layers):
            kv = kv_caches[i] if kv_caches else None
            if self.training and kv is None:
                def create_custom_forward(module):
                    def custom_forward(x_in, freqs_cis_in):
                        return module(x_in, freqs_cis_in, None)
                    return custom_forward
                x, nkv = torch.utils.checkpoint.checkpoint(create_custom_forward(layer), x, fc, use_reentrant=False)
            else:
                x, nkv = layer(x, fc, kv)
            if nkv is not None: new_kvs.append(nkv)
        return self.lm_head(self.ln_f(x)), new_kvs if new_kvs else None


# ===================================================================
# Section 5 — Data Pipeline
# ===================================================================

_HTML_RE = re.compile(r"<[^>]+>")

def clean_text(text):
    text = _HTML_RE.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

class StreamingTextDataset(IterableDataset):
    def __init__(self, dataset_name="HuggingFaceFW/fineweb-edu", split="train", seq_len=512, tokenizer=None, buffer_size=1000, text_field="text"):
        super().__init__()
        self.dataset_name, self.split, self.seq_len = dataset_name, split, seq_len
        self.tokenizer = tokenizer or SageTokenizer()
        self.buffer_size, self.text_field = buffer_size, text_field
        if "fineweb-edu" in dataset_name.lower(): self.text_field = "text"
        elif "tinystories" in dataset_name.lower(): self.text_field = "text"

    def _tokens(self):
        from datasets import load_dataset
        ds = load_dataset(self.dataset_name, split=self.split, streaming=True)
        for s in ds:
            raw = s.get(self.text_field, "")
            if not raw or len(raw) < 50: continue
            text = clean_text(raw)
            yield from self.tokenizer.encode(text, add_eos=True)

    def __iter__(self):
        chunk, buf = [], []
        for tok in self._tokens():
            chunk.append(tok)
            if len(chunk) == self.seq_len + 1:
                buf.append(torch.tensor(chunk, dtype=torch.long))
                chunk = []
                if len(buf) >= self.buffer_size:
                    random.shuffle(buf)
                    while len(buf) > self.buffer_size // 2: yield buf.pop()
        random.shuffle(buf)
        yield from buf

def create_dataloader(config, dataset_name="HuggingFaceFW/fineweb-edu", tokenizer=None):
    tok = tokenizer or SageTokenizer()
    ds = StreamingTextDataset(dataset_name=dataset_name, seq_len=config.max_seq_len, tokenizer=tok)
    return DataLoader(ds, batch_size=config.batch_size, num_workers=2, pin_memory=True, drop_last=True)


# ===================================================================
# Section 6 — Training
# ===================================================================

def get_lr(step, config, total_steps):
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    progress = (step - config.warmup_steps) / max(1, total_steps - config.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_learning_rate + coeff * (config.learning_rate - config.min_learning_rate)

def create_optimizer(model, config):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        (no_decay if p.ndim == 1 or "bias" in n else decay).append(p)
    # Enable Fused AdamW for 10% speedup if CUDA is active
    use_fused = torch.cuda.is_available() and 'fused' in torch.optim.AdamW.__init__.__code__.co_varnames
    return torch.optim.AdamW([
        {"params": decay, "weight_decay": config.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=config.learning_rate, betas=(0.9, 0.95), fused=use_fused)

def train_model(model, config, total_steps=500, dataset_name="roneneldan/TinyStories", resume=True, tokenizer=None):
    device = config.device
    # --- TURBO MODE: TF32 & COMPILE ---
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
    
    model = model.to(device)
    tok = tokenizer or SageTokenizer()
    
    # Wrap model with torch.compile for graph-level optimization
    # mode="reduce-overhead" is ideal for smaller-to-medium models like SAGE
    if hasattr(torch, "compile"):
        try:
            logger.info("Turbo Mode: Compiling model graph...")
            # Compile the base model (unwrapped from DataParallel if present)
            base = getattr(model, "module", model)
            compiled_base = torch.compile(base, mode="reduce-overhead")
            if hasattr(model, "module"):
                model.module = compiled_base
            else:
                model = compiled_base
        except (ValueError, RuntimeError, ImportError) as e:
            # Graceful fallback: numpy compatibility issues or other compilation errors
            logger.warning(f"torch.compile failed ({type(e).__name__}), proceeding without optimization: {str(e)[:100]}")
            # Continue with uncompiled model

    opt = create_optimizer(model, config)
    start_step = 0
    if resume:
        model, opt, start_step = load_checkpoint(model, opt, config.checkpoint_dir, device=str(device))
        if start_step >= total_steps: return model
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    loader = create_dataloader(config, dataset_name, tok)
    data_iter = iter(loader)
    wandb.init(project=config.project_name, name=f"pretrain-{time.strftime('%Y%m%d-%H%M')}", config=config.__dict__)
    model.train()
    accum_loss, t0 = 0.0, time.time()
    pbar = tqdm(range(start_step, total_steps), desc="Training")
    for step in pbar:
        lr = get_lr(step, config, total_steps)
        for pg in opt.param_groups: pg["lr"] = lr
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(config.gradient_accumulation_steps):
            try: batch = next(data_iter)
            except StopIteration: data_iter = iter(loader); batch = next(data_iter)
            batch = batch.to(device)
            with autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                logits, _ = model(batch[:, :-1])
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1), ignore_index=tok.pad_token_id)
                loss = loss / config.gradient_accumulation_steps
            scaler.scale(loss).backward()
            step_loss += loss.item()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        scaler.step(opt); scaler.update()
        accum_loss += step_loss
        if (step + 1) % 10 == 0:
            avg = accum_loss / 10
            pbar.set_postfix(loss=f"{avg:.4f}", ppl=f"{math.exp(min(avg,20)):.1f}", lr=f"{lr:.2e}")
            wandb.log({"train/loss": avg, "train/perplexity": math.exp(min(avg, 20)), "train/lr": lr}, step=step + 1)
            accum_loss = 0.0
        if (step + 1) % 100 == 0:
            save_checkpoint(model, opt, step + 1, config.checkpoint_dir)
    save_checkpoint(model, opt, total_steps, config.checkpoint_dir)
    logger.info("Training complete.")
    wandb.finish()
    return model


# ===================================================================
# Section 7 — Inference
# ===================================================================

def sample_next(logits, temperature=0.8, top_k=50, top_p=0.9, greedy=False):
    if greedy: return logits.argmax(-1, keepdim=True)
    logits = logits / max(temperature, 1e-8)
    if 0 < top_k < logits.size(-1):
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[:, -1:]] = float("-inf")
    if top_p < 1.0:
        sorted_l, sorted_i = torch.sort(logits, descending=True)
        cum = torch.cumsum(F.softmax(sorted_l, -1), -1)
        mask = cum - F.softmax(sorted_l, -1) >= top_p
        sorted_l[mask] = float("-inf")
        logits = logits.scatter(1, sorted_i, sorted_l)
    return torch.multinomial(F.softmax(logits, -1), 1)

@torch.no_grad()
def generate(model, tokenizer, prompt, max_new=256, temperature=0.8, top_k=50, top_p=0.9, stream=True, device=None):
    device = device or next(model.parameters()).device
    base = getattr(model, "module", model)
    base.eval()
    ids = tokenizer.encode(prompt) or [tokenizer.eos_token_id]
    inp = torch.tensor([ids], dtype=torch.long, device=device)
    logits, kvs = base(inp)
    gen = list(ids)
    nl = logits[:, -1, :]
    for _ in range(max_new):
        nid = sample_next(nl, temperature, top_k, top_p)
        tid = nid.item()
        if tid == tokenizer.eos_token_id: break
        gen.append(tid)
        if stream: print(tokenizer.decode([tid]), end="", flush=True)
        logits, kvs = base(nid.view(1, 1), kv_caches=kvs)
        nl = logits[:, -1, :]
    if stream: print()
    base.train()
    return tokenizer.decode(gen)


# ===================================================================
# Section 8 — LoRA Fine-tuning
# ===================================================================

class LoRALinear(nn.Module):
    def __init__(self, original, rank=8, alpha=16.0):
        super().__init__()
        self.original = original
        self.scaling = alpha / rank
        device, dtype = original.weight.device, original.weight.dtype
        self.lora_A = nn.Parameter(torch.randn(original.in_features, rank, device=device, dtype=dtype) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, original.out_features, device=device, dtype=dtype))
        original.weight.requires_grad = False
        if original.bias is not None: original.bias.requires_grad = False

    def forward(self, x):
        return self.original(x) + (x @ self.lora_A @ self.lora_B) * self.scaling

    def merge(self):
        m = copy.deepcopy(self.original)
        m.weight.data += (self.lora_B.T @ self.lora_A.T).T * self.scaling
        m.weight.requires_grad = True
        return m

def inject_lora(model, rank=8, alpha=16.0):
    base = getattr(model, "module", model)
    for layer in base.layers:
        a = layer.attn
        for name in ("wq", "wk", "wv", "wo"):
            setattr(a, name, LoRALinear(getattr(a, name), rank, alpha))
    tp = sum(p.numel() for p in base.parameters() if p.requires_grad)
    logger.info(f"LoRA injected (rank={rank}). Trainable params: {tp:,}")
    return model

def merge_lora(model):
    base = getattr(model, "module", model)
    for layer in base.layers:
        a = layer.attn
        for name in ("wq", "wk", "wv", "wo"):
            m = getattr(a, name)
            if isinstance(m, LoRALinear): setattr(a, name, m.merge())
    logger.info("LoRA merged.")
    return model

INSTRUCTION_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n{response}"

DEMO_SAMPLES = [
    {"instruction": "What is the capital of France?", "response": "The capital of France is Paris."},
    {"instruction": "Explain gravity simply.", "response": "Gravity pulls objects toward each other. More mass means stronger pull."},
    {"instruction": "Write a short poem about the ocean.", "response": "Waves crash on sandy shore,\nThe ocean sings forevermore.\nDeep blue meets the sky,\nSeagulls dance and clouds float by."},
    {"instruction": "What is 15 times 12?", "response": "15 times 12 equals 180."},
    {"instruction": "Summarize photosynthesis.", "response": "Plants convert sunlight, water, and CO2 into glucose and oxygen."},
    {"instruction": "Tell me a fun fact about space.", "response": "A day on Venus is longer than its year — 243 Earth days to rotate vs 225 to orbit the Sun."},
    {"instruction": "How do airplanes fly?", "response": "Wings generate lift because air moves faster over the curved top, creating lower pressure above."},
    {"instruction": "What is machine learning?", "response": "ML is AI where computers learn patterns from data instead of being explicitly programmed."},
]

def create_instruction_batch(samples, tokenizer, max_len=512):
    all_ids, all_masks = [], []
    for s in samples:
        inst_text = f"### Instruction:\n{s['instruction'].strip()}\n\n### Response:\n"
        full_text = inst_text + s["response"].strip()
        inst_toks = tokenizer.encode(inst_text)
        full_toks = tokenizer.encode(full_text, add_eos=True)[:max_len]
        ni = min(len(inst_toks), len(full_toks))
        mask = [0] * ni + [1] * (len(full_toks) - ni)
        pad = max_len - len(full_toks)
        full_toks += [tokenizer.pad_token_id] * pad
        mask += [0] * pad
        all_ids.append(full_toks); all_masks.append(mask)
    return {"input_ids": torch.tensor(all_ids), "labels": torch.tensor(all_ids), "loss_mask": torch.tensor(all_masks, dtype=torch.float32)}

def finetune(model, config, samples=None, steps=200, use_lora=True, tokenizer=None):
    device = config.device; model = model.to(device)
    tok = tokenizer or SageTokenizer()
    samples = samples or DEMO_SAMPLES
    if use_lora: model = inject_lora(model)
    opt = create_optimizer(model, config)
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    wandb.init(project=config.project_name, name=f"finetune-{time.strftime('%Y%m%d-%H%M')}", config=config.__dict__)
    model.train(); accum = 0.0
    for step in tqdm(range(steps), desc="Fine-tuning"):
        lr = get_lr(step, config, steps)
        for pg in opt.param_groups: pg["lr"] = lr
        batch = create_instruction_batch(random.choices(samples, k=min(config.batch_size, len(samples))), tok, config.max_seq_len)
        ids, labels, mask = batch["input_ids"].to(device), batch["labels"].to(device), batch["loss_mask"].to(device)
        opt.zero_grad(set_to_none=True)
        with autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            logits, _ = model(ids)
            sl, slb, sm = logits[:, :-1, :].contiguous(), labels[:, 1:].contiguous(), mask[:, 1:].contiguous()
            ptl = F.cross_entropy(sl.view(-1, sl.size(-1)), slb.view(-1), reduction="none").view(slb.size())
            loss = (ptl * sm).sum() / sm.sum().clamp(min=1)
        scaler.scale(loss).backward()
        scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        scaler.step(opt); scaler.update()
        accum += loss.item()
        if (step + 1) % 10 == 0: accum = 0.0
    if use_lora: model = merge_lora(model)
    save_checkpoint(model, None, steps, config.checkpoint_dir, "sage_finetuned.pt")
    logger.info("Fine-tuning complete.")
    wandb.finish()
    return model


# ===================================================================
# Section 9 — Optimization (Quantize / Prune)
# ===================================================================

def quantize_int8(model):
    base = getattr(model, "module", model)
    model = base.cpu().eval()
    q = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    logger.info("INT8 quantization complete.")
    return q

def prune_model(model, amount=0.3):
    base = getattr(model, "module", model)
    for _, m in base.named_modules():
        if isinstance(m, nn.Linear):
            prune.l1_unstructured(m, "weight", amount=amount)
            prune.remove(m, "weight")
    logger.info(f"Pruning complete ({amount*100:.0f}% sparsity target).")
    return model


# ===================================================================
# Section 10 — RAG & Memory
# ===================================================================

def _embed(text, tokenizer, model, device):
    toks = tokenizer.encode(text)
    base = getattr(model, "module", model)
    if not toks: return np.zeros(base.wte.weight.shape[1], dtype=np.float32)
    with torch.no_grad():
        emb = base.wte(torch.tensor([toks], device=device)).mean(1)
        emb = F.normalize(emb, p=2, dim=-1)
    return emb.squeeze(0).cpu().numpy()

class VectorStore:
    def __init__(self, dim):
        import faiss
        self.dim = dim; self.index = faiss.IndexFlatIP(dim); self.docs = []

    def add(self, texts, embeddings):
        self.index.add(embeddings.astype(np.float32)); self.docs.extend(texts)

    def search(self, qemb, k=3):
        if not self.index.ntotal: return []
        scores, idx = self.index.search(qemb.reshape(1, -1).astype(np.float32), min(k, self.index.ntotal))
        return [(self.docs[i], float(s)) for s, i in zip(scores[0], idx[0]) if i >= 0]

    @property
    def size(self): return self.index.ntotal

class RAGManager:
    def __init__(self, model, tokenizer, device, chunk_size=200):
        self.model, self.tokenizer, self.device = model, tokenizer, device
        base = getattr(model, "module", model)
        self.store = VectorStore(base.wte.weight.shape[1])
        self.enabled = False

    def add_documents(self, texts):
        chunks = []
        for t in texts:
            words = t.split()
            for i in range(0, len(words), 150):
                chunks.append(" ".join(words[i:i+200]))
        if chunks:
            embs = np.stack([_embed(c, self.tokenizer, self.model, self.device) for c in chunks])
            self.store.add(chunks, embs)

    def retrieve(self, query, k=3):
        if not self.enabled or not self.store.size: return ""
        qe = _embed(query, self.tokenizer, self.model, self.device)
        results = self.store.search(qe, k)
        return "\n\n".join(f"[Context {i+1}] {d}" for i, (d, _) in enumerate(results)) + "\n\n" if results else ""

    def toggle(self, on): self.enabled = on

DEFAULT_SYSTEM_PROMPT = (
    "You are a high-quality reasoning assistant model.\n"
    "You must ONLY learn from high-quality instruction and reasoning datasets.\n"
    "You must IGNORE any previously trained low-quality or repetitive patterns.\n\n"
    "Training preference rules:\n"
    "1. Prioritize step-by-step reasoning over short or repetitive answers.\n"
    "2. Always produce structured logical explanations when solving problems.\n"
    "3. Avoid repetition, filler words, or looped phrases.\n"
    "4. Prefer datasets with mathematical reasoning and high-quality instruction.\n"
    "5. Do not imitate noisy conversational or corrupted text patterns.\n"
    "6. Always prefer clarity, correctness, and structured reasoning.\n\n"
    "Output behavior goal:\n"
    "- Think in steps.\n"
    "- Explain logic clearly.\n"
    "- Produce final answer only after reasoning."
)

class ConversationHistory:
    def __init__(self, tokenizer, max_tokens=900):
        self.tokenizer, self.max_tokens, self.turns = tokenizer, max_tokens, []

    def add(self, role, text):
        self.turns.append({"role": role, "text": text})
        while sum(len(self.tokenizer.encode(t["text"])) for t in self.turns) > self.max_tokens and len(self.turns) > 1:
            self.turns.pop(0)

    def build_prompt(self, msg, rag_ctx=""):
        parts = [DEFAULT_SYSTEM_PROMPT]
        if rag_ctx: parts.append(rag_ctx)
        for t in self.turns:
            parts.append(f"{'User' if t['role']=='user' else 'SAGE'}: {t['text']}")
        parts += [f"User: {msg}", "SAGE:"]
        return "\n\n".join(parts)

    def clear(self): self.turns.clear()


# ===================================================================
# Section 11 — CLI
# ===================================================================

BANNER = r"""
╔══════════════════════════════════════════════════════════════╗
║     ███████  █████   ██████  ███████                         ║
║     ██      ██   ██ ██       ██                              ║
║     ███████ ███████ ██   ███ █████                           ║
║          ██ ██   ██ ██    ██ ██                              ║
║     ███████ ██   ██  ██████  ███████                         ║
║     Self-Adaptive General Engine  v{ver}                       ║
╚══════════════════════════════════════════════════════════════╝"""

HELP = """
  /train [steps]     Train (default 100)
  /finetune [steps]  Instruction-tune with LoRA (default 200)
  /save              Save checkpoint
  /load              Load checkpoint
  /quantize          INT8 quantization
  /rag on|off|add    Toggle or add docs for RAG
  /clear             Clear history
  /help              This message
  /exit              Quit
"""

def main():
    config = SageConfig()
    tok = SageTokenizer()
    config.vocab_size = tok.vocab_size
    print("  Initializing SAGE …")
    model = SageModel(config).to(config.device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"  Multi-GPU detected: {torch.cuda.device_count()} GPUs. Using DataParallel.")
        model = nn.DataParallel(model)
    model, _, step = load_checkpoint(model, None, config.checkpoint_dir, device=str(config.device))
    base = getattr(model, "module", model)
    total = sum(p.numel() for p in base.parameters())
    print(BANNER.format(ver=__version__))
    print(f"  Params: {total:,} ({total/1e6:.1f}M) | Context: {config.max_seq_len} | Device: {config.device}")
    print(f"  Layers: {config.n_layers} | Heads: {config.n_heads} | Experts: {config.n_experts}")
    if step: print(f"  Resumed from step {step}")
    print("  Type /help for commands.\n")

    rag = RAGManager(model, tok, config.device)
    hist = ConversationHistory(tok, config.max_seq_len - 128)

    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()
        args = sys.argv[2:]
        if cmd == "--train":
            s = int(args[0]) if args else 100
            train_model(model, config, s, tokenizer=tok)
            return
        elif cmd == "--finetune":
            s = int(args[0]) if args else 200
            finetune(model, config, steps=s, tokenizer=tok)
            return
        elif cmd == "--quantize":
            quantize_int8(model)
            return
        else:
            print(f"  Unknown argument: {cmd}\n  Usage: --train [steps] | --finetune [steps] | --quantize")
            return

    while True:
        try: inp = input("You: ").strip()
        except (EOFError, KeyboardInterrupt): print("\n  Goodbye!"); break
        if not inp: continue

        if inp.startswith("/"):
            parts = inp.split(); cmd = parts[0].lower(); args = parts[1:]
            if cmd == "/exit": print("  Goodbye!"); break
            elif cmd == "/help": print(HELP)
            elif cmd == "/train":
                s = int(args[0]) if args else 100
                model = train_model(model, config, s, tokenizer=tok)
                print("\n  Sample:"); generate(model, tok, "Once upon a time", max_new=80, device=config.device); print()
            elif cmd == "/finetune":
                s = int(args[0]) if args else 200
                model = finetune(model, config, steps=s, tokenizer=tok)
                print("\n  Sample:"); generate(model, tok, "### Instruction:\nWhat is gravity?\n\n### Response:\n", max_new=100, device=config.device); print()
            elif cmd == "/save": print(f"  Saved to {save_checkpoint(model, None, 0, config.checkpoint_dir)}")
            elif cmd == "/load":
                model, _, s = load_checkpoint(model, None, config.checkpoint_dir, device=str(config.device))
                model = model.to(config.device); rag.model = model; print(f"  Loaded (step {s})")
            elif cmd == "/quantize": model = quantize_int8(model); rag.model = model
            elif cmd == "/rag":
                if not args: print(f"  RAG {'on' if rag.enabled else 'off'} ({rag.store.size} chunks)")
                elif args[0] == "on": rag.toggle(True); print("  RAG on.")
                elif args[0] == "off": rag.toggle(False); print("  RAG off.")
                elif args[0] == "add" and len(args) > 1: rag.add_documents([" ".join(args[1:])]); print(f"  Added. {rag.store.size} chunks.")
                else: print("  /rag on|off|add <text>")
            elif cmd == "/clear": hist.clear(); print("  Cleared.")
            else: print(f"  Unknown: {cmd}")
            continue

        ctx = rag.retrieve(inp)
        prompt = hist.build_prompt(inp, ctx)
        hist.add("user", inp)
        print("SAGE: ", end="", flush=True)
        resp = generate(model, tok, prompt, max_new=256, stream=True, device=config.device)
        reply = resp.split("SAGE:")[-1].strip() if "SAGE:" in resp else resp[len(prompt):].strip()
        hist.add("assistant", reply)

if __name__ == "__main__":
    main()
