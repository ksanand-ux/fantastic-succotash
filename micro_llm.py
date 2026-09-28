"""Micro LLM in pure Python + PyTorch.

This is a minimal, educational character-level language model:
- Tiny GPT-style Transformer
- Trains on local text
- Generates continuations from a prompt

Usage:
    python micro_llm.py --text "hello world" --steps 300
    python micro_llm.py --file data.txt --steps 1000 --generate "Once upon"
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    block_size: int = 64
    batch_size: int = 32
    n_embed: int = 96
    n_head: int = 4
    n_layer: int = 3
    dropout: float = 0.1
    lr: float = 3e-4
    steps: int = 800
    eval_every: int = 100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embed: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        assert n_embed % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embed // n_head

        self.qkv = nn.Linear(n_embed, 3 * n_embed)
        self.proj = nn.Linear(n_embed, n_embed)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask.view(1, 1, block_size, block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(c, dim=-1)

        q = q.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :t, :t] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.resid_drop(self.proj(y))


class MLP(nn.Module):
    def __init__(self, n_embed: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.GELU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embed)
        self.attn = CausalSelfAttention(cfg.n_embed, cfg.n_head, cfg.block_size, cfg.dropout)
        self.ln2 = nn.LayerNorm(cfg.n_embed)
        self.mlp = MLP(cfg.n_embed, cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(vocab_size, cfg.n_embed)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embed)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embed)
        self.head = nn.Linear(cfg.n_embed, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        b, t = idx.shape
        pos = torch.arange(0, t, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx


def build_dataset(text: str):
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}

    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

    def encode(s: str):
        return [stoi[c] for c in s if c in stoi]

    def decode(tokens: list[int]):
        return "".join(itos[t] for t in tokens)

    return data, encode, decode, len(chars)


def get_batch(data: torch.Tensor, cfg: Config):
    ix = torch.randint(len(data) - cfg.block_size - 1, (cfg.batch_size,))
    x = torch.stack([data[i : i + cfg.block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + cfg.block_size + 1] for i in ix])
    return x.to(cfg.device), y.to(cfg.device)


def train(model: TinyGPT, data: torch.Tensor, cfg: Config):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    model.train()
    for step in range(1, cfg.steps + 1):
        xb, yb = get_batch(data, cfg)
        _, loss = model(xb, yb)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % cfg.eval_every == 0 or step == 1:
            print(f"step {step:4d}/{cfg.steps} | loss {loss.item():.4f}")


def main():
    p = argparse.ArgumentParser(description="Train a micro GPT on local text")
    p.add_argument("--text", type=str, help="inline training text")
    p.add_argument("--file", type=str, help="path to training text file")
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--generate", type=str, default="The ")
    p.add_argument("--tokens", type=int, default=200)
    args = p.parse_args()

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        text = (
            "Machine learning is fun. "
            "A tiny model can still learn local text patterns. "
            "This is only an educational micro LLM. "
        )

    cfg = Config(steps=args.steps)

    data, encode, decode, vocab_size = build_dataset(text)
    if len(data) < cfg.block_size + 2:
        raise ValueError(
            f"Need at least {cfg.block_size + 2} chars of training text, got {len(data)}"
        )

    model = TinyGPT(vocab_size, cfg).to(cfg.device)
    train(model, data, cfg)

    prompt_tokens = encode(args.generate) or [0]
    start = torch.tensor([prompt_tokens], dtype=torch.long, device=cfg.device)
    out = model.generate(start, max_new_tokens=args.tokens)[0].tolist()
    print("\n=== Generated ===")
    print(decode(out))


if __name__ == "__main__":
    main()
