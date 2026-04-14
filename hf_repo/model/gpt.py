import torch
import torch.nn.functional as F
import torch.nn as nn
from utils.helpers import apply_rotary_emb
from utils.optimizers import Muon, DualOptimizer
import math
import inspect


class Attention(nn.Module):
    def __init__(self, n_embd, n_heads, n_kv_heads=None, use_qk_norm=True):
        super().__init__()
        assert n_embd % n_heads == 0
        self.n_embd = n_embd
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        assert n_heads % self.n_kv_heads == 0
        self.H = n_embd // n_heads
        self.use_qk_norm = use_qk_norm
        self.attn = nn.Linear(n_embd, self.n_heads * self.H + 2 * self.n_kv_heads * self.H, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        if use_qk_norm:
            self.q_norm = nn.RMSNorm(self.H)
            self.k_norm = nn.RMSNorm(self.H)
        self.proj.RESIDUAL_SCALE_INIT_FACTOR = True

    def forward(self, x, sin, cos):
        B, T, C = x.shape
        q, k, v = self.attn(x).split(
            [self.n_heads * self.H, self.n_kv_heads * self.H, self.n_kv_heads * self.H], dim=-1
        )
        q = q.view(B, T, self.n_heads, self.H).transpose(1, 2)
        k = k.view(B, T, self.n_kv_heads, self.H).transpose(1, 2)
        v = v.view(B, T, self.n_kv_heads, self.H).transpose(1, 2)
        q = apply_rotary_emb(q, sin, cos)
        k = apply_rotary_emb(k, sin, cos)
        if self.use_qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        if self.n_kv_heads != self.n_heads:
            reps = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(reps, dim=1)
            v = v.repeat_interleave(reps, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.n_embd = n_embd
        hidden_dim = int(8 * n_embd // 3)
        self.hidden_dim = (hidden_dim + 255) // 256 * 256
        self.gate_proj = nn.Linear(n_embd, self.hidden_dim * 2, bias=False)
        self.down_proj = nn.Linear(self.hidden_dim, n_embd, bias=False)
        self.down_proj.RESIDUAL_SCALE_INIT_FACTOR = True

    def forward(self, x):
        y, gate = torch.chunk(self.gate_proj(x), 2, dim=-1)
        gate = F.silu(gate)
        y = gate * y
        return self.down_proj(y)


class Block(nn.Module):
    def __init__(self, n_embd, n_heads, n_kv_heads=None, use_qk_norm=True):
        super().__init__()
        self.ln1 = nn.RMSNorm(n_embd)
        self.sa = Attention(n_embd, n_heads, n_kv_heads=n_kv_heads, use_qk_norm=use_qk_norm)
        self.ln2 = nn.RMSNorm(n_embd)
        self.mlp = MLP(n_embd)

    def forward(self, x, sin, cos):
        x = x + self.sa(self.ln1(x), sin, cos)
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, n_embd, vocab_size, block_size, n_heads, head_size, rope_head_size,
                 n_layers, n_kv_heads=None, use_qk_norm=True):
        super().__init__()
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_layers = n_layers
        self.wte = nn.Embedding(vocab_size, n_embd)
        sin, cos = self._precompute_rotary_embeddings(block_size, (n_embd // n_heads))
        self.register_buffer("sin", sin)
        self.register_buffer("cos", cos)
        self.transformer = nn.ModuleList(
            [Block(n_embd, n_heads, n_kv_heads=n_kv_heads, use_qk_norm=use_qk_norm) for _ in range(n_layers)]
        )
        self.ln = nn.RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.01
            if hasattr(module, "RESIDUAL_SCALE_INIT_FACTOR"):
                std *= 1 / (math.sqrt(2 * self.n_layers))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.block_size
        x = self.wte(idx)
        sin = self.sin[:, :, :T, :]
        cos = self.cos[:, :, :T, :]
        for block in self.transformer:
            x = block(x, sin, cos)
        x = self.ln(x)
        if targets is not None:
            logits = self.lm_head(x)
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
            return None, loss
        else:
            logits = self.lm_head(x)
            return logits, None

    def generate(self, idx, num_sequences=5, max_tokens=200, topk=50, temperature=1.0,
                 repetition_penalty=1.0, chat_mode=False, eos_token=50256):
        idx = torch.repeat_interleave(idx.unsqueeze(0), num_sequences, dim=0)
        for _ in range(max_tokens):
            logits, _ = self.forward(idx)
            logits = logits[:, -1, :]
            if repetition_penalty != 1.0:
                for i in range(logits.shape[0]):
                    for token_id in idx[i].tolist():
                        logits[i, token_id] /= repetition_penalty
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, k=topk)
            idx_next = torch.multinomial(topk_probs, num_samples=1)
            idx_next = torch.gather(topk_indices, -1, idx_next)
            if chat_mode and (idx_next == eos_token).all():
                break
            idx = torch.cat([idx, idx_next], dim=-1)
        return idx

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)
        sin, cos = freqs.sin(), freqs.cos()
        sin, cos = sin.bfloat16(), cos.bfloat16()
        sin, cos = sin[None, None, :, :], cos[None, None, :, :]
        return sin, cos
