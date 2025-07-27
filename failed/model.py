"""
Minimal GPT-like transformer with built-in LoRA adapters.
`model.base_parameters()`      → iterator over W matrices only
`model.lora_parameters()`      → iterator over all LoRA A,B matrices
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch, torch.nn as nn
from torch.nn import functional as F

@dataclass
class Config:
    vocab_size: int = 50257
    block_size: int = 512
    embed_dim: int = 512
    mlp_dim: int = 512 * 4
    n_layer: int = 8
    n_head: int = 8
    n_kv: int = 2

class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 65536) -> None:
        super().__init__()

        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim // 4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim // 4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.cos = nn.Buffer(theta.cos(), persistent=False)
        self.sin = nn.Buffer(theta.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cos = self.cos[None, : x.size(-3), None, :]
        sin = self.sin[None, : x.size(-3), None, :]
        x1, x2 = x.float().chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x)


class Attention(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()

        self.n_head = config.n_head
        self.head_dim = config.embed_dim // config.n_head
        self.n_kv = config.n_kv

        assert config.embed_dim % config.n_head == 0
        assert config.n_head % config.n_kv == 0

        # attention projection
        self.c_attn_q = nn.Linear(config.embed_dim, config.embed_dim, bias=True)

        # key and value projection
        self.c_attn_kv = nn.Linear(config.embed_dim, 2 * config.n_kv * self.head_dim, bias=True)

        # rotary embedding
        self.rotary = Rotary(config.embed_dim // config.n_head)

        # output projection (same as attention projection)
        self.c_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        B, T, C = x.size()

        q = self.c_attn_q(x)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        k = self.c_attn_kv(x)
        k, v = k.split(self.n_kv * self.head_dim, dim=2)

        k = k.view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        q, k = self.rotary(q), self.rotary(k)

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)

        return y

class MLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.fc1 = nn.Linear(config.embed_dim, config.mlp_dim)
        self.fc2 = nn.Linear(config.mlp_dim, config.embed_dim)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class Block(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.norm1 = nn.RMSNorm(config.embed_dim)
        self.attn= Attention(config)
        self.norm2 = nn.RMSNorm(config.embed_dim)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp (self.norm2(x))
        return x

# ---------- full Transformer ----------------------------------------------- #
class Transformer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.inp_emb = nn.Embedding(config.vocab_size, config.embed_dim)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.norm = nn.RMSNorm(config.embed_dim)
        self.out_emb = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        self.inp_emb.weight = self.out_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module): 
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def embed_in(self, x: torch.Tensor) -> torch.Tensor:
        return self.inp_emb(x)

    def embed_out(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_emb(x)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # If no attention mask is supplied, create a standard causal mask so that
        # each position can only attend to positions at or before its own index.
        if mask is None:
            # x has shape (B, T, C) where T is the sequence length.
            T = x.size(1)
            mask = torch.triu(
                torch.full((T, T), float("-inf"), device=x.device, dtype=x.dtype),
                diagonal=1,
            )

        for blk in self.blocks:
            x = blk(x, mask)
        x = self.norm(x)
        return x