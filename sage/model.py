import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from .config import SageConfig

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """Precomputes rotary positional embedding frequencies."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies rotary positional embeddings to queries and keys."""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # Reshape freqs_cis to broadcast with xq_ and xk_
    # xq_, xk_ shape: [batch, seq_len, n_heads, dim_head//2]
    # freqs_cis shape: [seq_len, dim_head//2]
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)
    
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)

class CausalSelfAttention(nn.Module):
    def __init__(self, config: SageConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        assert self.d_model % self.n_heads == 0
        self.head_dim = self.d_model // self.n_heads
        
        self.wq = nn.Linear(self.d_model, self.d_model, bias=False)
        self.wk = nn.Linear(self.d_model, self.d_model, bias=False)
        self.wv = nn.Linear(self.d_model, self.d_model, bias=False)
        self.wo = nn.Linear(self.d_model, self.d_model, bias=False)
        
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Flash attention handles causality via is_causal flag if seq_len > 1
        
    def forward(
        self, 
        x: torch.Tensor, 
        freqs_cis: torch.Tensor, 
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = x.size() # batch, seq_len, d_model
        
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        
        q = q.view(B, T, self.n_heads, self.head_dim)
        k = k.view(B, T, self.n_heads, self.head_dim)
        v = v.view(B, T, self.n_heads, self.head_dim)
        
        q, k = apply_rotary_emb(q, k, freqs_cis)
        
        if kv_cache is not None:
            # We are generating token by token
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=1)
            v = torch.cat([v_cache, v], dim=1)
            new_kv_cache = (k, v)
        else:
            new_kv_cache = None
            
        # Move heads to correct dimension: (B, n_heads, T, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Flash attention natively supported via scaled_dot_product_attention
        is_causal = (kv_cache is None and T > 1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0 if not self.training else 0.1, is_causal=is_causal)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.wo(y))
        
        return y, new_kv_cache

class ExpertFFN(nn.Module):
    def __init__(self, config: SageConfig):
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w2 = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.w3 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU activation structure
        hidden = F.silu(self.w1(x)) * self.w3(x)
        return self.dropout(self.w2(hidden))

class MoE(nn.Module):
    def __init__(self, config: SageConfig):
        super().__init__()
        self.n_experts = config.n_experts
        self.top_k = config.num_experts_per_tok
        self.d_model = config.d_model
        
        self.router = nn.Linear(self.d_model, self.n_experts, bias=False)
        self.experts = nn.ModuleList([ExpertFFN(config) for _ in range(self.n_experts)])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        x_flat = x.view(-1, C) # [B*T, C]
        
        router_logits = self.router(x_flat) # [B*T, n_experts]
        routing_weights = F.softmax(router_logits, dim=-1)
        
        # Select Top K experts
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1) # [B*T, top_k]
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True) # re-normalize
        
        final_out = torch.zeros_like(x_flat)
        
        # Iterate over experts and compute their outputs
        for i, expert in enumerate(self.experts):
            # Find which tokens chose this expert
            expert_mask = (selected_experts == i)
            token_idx, kth_expert = torch.where(expert_mask)
            
            if token_idx.shape[0] > 0:
                expert_inputs = x_flat[token_idx]
                expert_outputs = expert(expert_inputs)
                
                # Apply router weight
                weights = routing_weights[token_idx, kth_expert].unsqueeze(-1)
                final_out[token_idx] += expert_outputs * weights
                
        return final_out.view(B, T, C)

class TransformerBlock(nn.Module):
    def __init__(self, config: SageConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.moe = MoE(config)
        
    def forward(
        self, 
        x: torch.Tensor, 
        freqs_cis: torch.Tensor, 
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Pre-LayerNorm architecture
        h, new_kv_cache = self.attn(self.norm1(x), freqs_cis, kv_cache)
        x = x + h
        x = x + self.moe(self.norm2(x))
        return x, new_kv_cache

class SageModel(nn.Module):
    def __init__(self, config: SageConfig):
        super().__init__()
        self.config = config
        
        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Weight tying
        self.wte.weight = self.lm_head.weight
        
        # Precompute RoPE frequencies
        self.register_buffer("freqs_cis", precompute_freqs_cis(config.d_model // config.n_heads, config.max_seq_len * 2), persistent=False)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self, 
        idx: torch.Tensor, 
        kv_caches: Optional[list] = None
    ) -> Tuple[torch.Tensor, Optional[list]]:
        B, T = idx.size()
        
        if kv_caches is not None:
            # generating context, token is at specific position
            start_pos = kv_caches[0][0].shape[1]
        else:
            start_pos = 0
            
        freqs_cis = self.freqs_cis[start_pos : start_pos + T]
        
        x = self.drop(self.wte(idx))
        
        new_kv_caches = []
        for i, layer in enumerate(self.layers):
            kv_cache = kv_caches[i] if kv_caches else None
            
            # Use gradient checkpointing during training
            if self.training and kv_cache is None:
                def create_custom_forward(module):
                    def custom_forward(x_in, freqs_cis_in):
                        return module(x_in, freqs_cis_in, None)
                    return custom_forward
                
                x, new_kv_cache = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer), 
                    x, freqs_cis, 
                    use_reentrant=False
                )
            else:
                x, new_kv_cache = layer(x, freqs_cis, kv_cache)
                
            if new_kv_cache is not None:
                new_kv_caches.append(new_kv_cache)
                
        x = self.ln_f(x)
        logits = self.lm_head(x) # [B, T, vocab_size]
        
        return logits, new_kv_caches if len(new_kv_caches) > 0 else None
