"""A compact Llama-shape decoder (RMSNorm, RoPE, SwiGLU, tied embeddings).

Kept architecture-identical to Llama so the same config scales from the 5M sanity
model to nl2jq-40m, and so HF/llama.cpp conversion needs no custom code.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    n_layer: int = 6
    n_head: int = 6
    n_kv_head: int = 6
    d_model: int = 288
    d_ff: int = 768          # SwiGLU hidden (multiple of 32)
    max_seq_len: int = 512
    rope_theta: float = 10000.0
    dropout: float = 0.0

    def n_params(self):
        # rough, ignoring norms/bias: embed(tied) + layers
        emb = self.vocab_size * self.d_model
        attn = self.n_layer * (self.d_model * self.d_model +
                               2 * self.d_model * (self.d_model // self.n_head) * self.n_kv_head +
                               self.d_model * self.d_model)
        ffn = self.n_layer * (3 * self.d_model * self.d_ff)
        return emb + attn + ffn


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * self.weight


def build_rope(seq_len, dim, theta, device):
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    # x: (B, H, T, D). Split into even/odd halves (rotate_half convention).
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    rot = torch.cat((-x2, x1), dim=-1)
    cos2 = torch.cat((cos, cos), dim=-1)
    sin2 = torch.cat((sin, sin), dim=-1)
    return x * cos2 + rot * sin2


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.nh, self.nkv = cfg.n_head, cfg.n_kv_head
        self.hd = cfg.d_model // cfg.n_head
        self.q = nn.Linear(cfg.d_model, self.nh * self.hd, bias=False)
        self.k = nn.Linear(cfg.d_model, self.nkv * self.hd, bias=False)
        self.v = nn.Linear(cfg.d_model, self.nkv * self.hd, bias=False)
        self.o = nn.Linear(self.nh * self.hd, cfg.d_model, bias=False)
        self.drop = cfg.dropout

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.nh, self.hd).transpose(1, 2)
        k = self.k(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        v = self.v(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if self.nkv != self.nh:
            rep = self.nh // self.nkv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.drop if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o(y)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w3 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.n2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)
        x = x + self.mlp(self.n2(x))
        return x


class NL2JQModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # tied
        self._cos, self._sin = None, None
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _rope(self, T, device):
        if self._cos is None or self._cos.shape[0] < T or self._cos.device != device:
            cos, sin = build_rope(self.cfg.max_seq_len, self.cfg.d_model // self.cfg.n_head,
                                  self.cfg.rope_theta, device)
            self._cos, self._sin = cos, sin
        return self._cos[:T], self._sin[:T]

    def forward(self, idx, targets=None, loss_mask=None):
        B, T = idx.shape
        cos, sin = self._rope(T, idx.device)
        x = self.tok_emb(idx)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            ls = logits.view(-1, logits.size(-1))
            tg = targets.reshape(-1)
            if loss_mask is not None:
                per = F.cross_entropy(ls, tg, reduction="none", ignore_index=-100)
                m = loss_mask.reshape(-1).float()
                loss = (per * m).sum() / m.sum().clamp(min=1)
            else:
                loss = F.cross_entropy(ls, tg, ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, eos_id, temperature=0.0, top_k=None):
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.cfg.max_seq_len:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :]
            if temperature <= 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = -float("inf")
                probs = F.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, 1)
            idx = torch.cat([idx, nxt], dim=1)
            if eos_id is not None and (nxt == eos_id).all():
                break
        return idx


CONFIGS = {
    # ~5M params for the M0 sanity gate
    "sanity": ModelConfig(n_layer=6, n_head=6, n_kv_head=6, d_model=288, d_ff=768,
                          max_seq_len=512),
    # ~40M params for nl2jq-40m (SPEC §5.1)
    "40m": ModelConfig(n_layer=10, n_head=8, n_kv_head=8, d_model=512, d_ff=1376,
                       max_seq_len=2048),
}
