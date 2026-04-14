import torch


def apply_rotary_emb(x, sin, cos):
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    return torch.cat((-x2, x1), dim=-1) * sin + x * cos
