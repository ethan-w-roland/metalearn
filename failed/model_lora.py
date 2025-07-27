"""
Minimal GPT-like transformer with built-in LoRA adapters.
`model.base_parameters()`      → iterator over W matrices only
`model.lora_parameters()`      → iterator over all LoRA A,B matrices
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator
import torch, torch.nn as nn
from torch.nn import functional as F

@dataclass
class Config:
    vocab_size: int = 50257
    block_size: int = 512
    n_layer: int = 8
    n_head: int = 6
    n_embd: int = 512
    r: int = 8

# ---------- LoRA wrapper ---------------------------------------------------- #
class LoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, r: int = 16, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(in_features)) if bias else None

        # LoRA: B (out×r) @ A (r×in)
        self.r = r
        self.A = nn.Parameter(torch.empty(in_features, r))
        self.B = nn.Parameter(torch.empty(r, out_features))

    def forward(self, x: torch.Tensor, use_lora: bool):
        out = x @ self.weight.T + self.bias
        if use_lora:
            lora_out = x @ self.B @ self.A
            out += lora_out
        return out

def rope_cache(seq_len: int, dim: int, device, dtype, theta: float = 10000.0):
    """Return cos/sin caches of shape [seq_len, dim//2]."""
    pos = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(1)
    idx = torch.arange(0, dim, 2, device=device, dtype=dtype)
    freqs = 1.0 / (theta ** (idx / dim))
    ang = pos * freqs                                    # [T, dim//2]
    return torch.cos(ang), torch.sin(ang)                # each [T, dim//2]

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Rotate last-dim pairs (… d0 d1 d2 d3 …) in-place."""
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    y  = torch.stack((x1 * cos - x2 * sin,
                      x1 * sin + x2 * cos), dim=-1)
    return y.flatten(-2)

class Attention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.n_head  = config.n_head
        self.head_dim= config.n_embd // config.n_head

        self.key   = LoRALinear(config.n_embd, config.n_embd, r=config.r)
        self.query = LoRALinear(config.n_embd, config.n_embd, r=config.r)
        self.value = LoRALinear(config.n_embd, config.n_embd, r=config.r)
        self.proj  = LoRALinear(config.n_embd, config.n_embd, r=config.r)

        # RoPE caches – initialised lazily on first forward pass
        self.register_buffer("rope_cos", torch.empty(0), persistent=False)
        self.register_buffer("rope_sin", torch.empty(0), persistent=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, use_lora: bool):

        B, T, C = x.size()

        # --- projections --------------------------------------------------- #
        q = self.query(x, use_lora).view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # [B,H,T,D]
        k = self.key  (x, use_lora).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.value(x, use_lora).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # --- rotary position embedding ------------------------------------- #
        if self.rope_cos.numel() < T * (self.head_dim // 2):
            cos, sin = rope_cache(T, self.head_dim, x.device, x.dtype)
            self.rope_cos, self.rope_sin = cos, sin
        cos = self.rope_cos[:T];  sin = self.rope_sin[:T]                 # [T, D/2]

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, gqa=True)

        y = attn_out.transpose(1, 2).contiguous().view(B, T, C)            # [B,T,C]
        y = self.proj(y, use_lora)

        return y

class MLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.fc1 = LoRALinear(config.n_embd, 4*config.n_embd)
        self.fc2 = LoRALinear(4*config.n_embd, config.n_embd)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, use_lora: bool):
        x = self.fc1(x, use_lora)
        x = self.act(x)
        x = self.fc2(x, use_lora)
        return x

class Block(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.norm1 = nn.RMSNorm(config.n_embd)
        self.attn= Attention(config)
        self.norm2 = nn.RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, use_lora: bool):
        x = x + self.attn(self.norm1(x), mask, use_lora)
        x = x + self.mlp (self.norm2(x), use_lora)
        return x

# ---------- full Transformer ----------------------------------------------- #
class Transformer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.inp_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.norm = nn.RMSNorm(config.n_embd)
        self.out_emb = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    # weight init identical to nanoGPT
    def _init_weights(self, module): 
        if isinstance(module, (nn.Linear, LoRALinear)):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        if isinstance(module, LoRALinear):
            nn.init.kaiming_normal_(module.A)
            nn.init.zeros_(module.B)

    # helpers to select param groups
    def base_parameters(self) -> Iterator[nn.Parameter]:
        for n,p in self.named_parameters():
            if ".A" in n or ".B" in n:  # LoRA params
                continue
            yield p

    def meta_parameters(self) -> Iterator[nn.Parameter]:
        for n,p in self.named_parameters():
            if ".A" in n or ".B" in n:
                yield p

    def token_embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.inp_emb(x)
    
    def logit_embed(self, x: torch.Tensor) -> torch.Tensor:
        return x.T @ self.inp_emb.weight #needed b/c logits can be "soft tokens"

    def forward(self, embed: torch.Tensor, mask: torch.Tensor, meta: bool = False):
        x = embed
        for blk in self.blocks:
            x = blk(x, mask, use_lora=meta)
        x = self.norm(x)
        x = self.out_emb(x)
        return x
