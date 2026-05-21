from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    mask = mask.to(dtype=x.dtype)
    while mask.dim() < x.dim():
        mask = mask.unsqueeze(-1)
    numerator = (x * mask).sum(dim=dim)
    denominator = mask.sum(dim=dim).clamp_min(eps)
    out = numerator / denominator
    if not torch.isfinite(out).all():
        raise FloatingPointError("masked_mean produced non-finite values.")
    return out


def masked_max(x: torch.Tensor, mask: torch.Tensor, dim: int = 1) -> torch.Tensor:
    bool_mask = mask.bool()
    while bool_mask.dim() < x.dim():
        bool_mask = bool_mask.unsqueeze(-1)
    masked = x.masked_fill(~bool_mask, -1e9)
    out = masked.max(dim=dim).values
    if not torch.isfinite(out).all():
        raise FloatingPointError("masked_max produced non-finite values.")
    return out


class MultiScaleCNNBlock(nn.Module):
    def __init__(self, dim: int, kernels: list[int], dropout: float) -> None:
        super().__init__()
        if not kernels:
            raise ValueError("At least one CNN kernel size is required.")
        for kernel in kernels:
            if kernel % 2 == 0:
                raise ValueError(f"Only odd kernel sizes support exact same padding, got {kernel}")
        self.out_dim = dim
        self.kernels = list(kernels)
        self.branch_dim = max(1, dim // len(kernels))
        self.concat_dim = self.branch_dim * len(kernels)
        self.norm = nn.LayerNorm(dim)
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(dim, self.branch_dim, kernel_size=kernel, padding=kernel // 2)
                for kernel in kernels
            ]
        )
        self.project = nn.Conv1d(self.concat_dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        residual = x
        y = self.norm(x).transpose(1, 2)
        y = torch.cat([branch(y) for branch in self.branches], dim=1)
        if y.shape[1] != self.concat_dim:
            raise RuntimeError(f"Expected CNN concat channels {self.concat_dim}, got {y.shape[1]}")
        y = self.project(y)
        y = F.gelu(y).transpose(1, 2)
        if y.shape[1] != residual.shape[1]:
            raise RuntimeError(f"Expected CNN output length {residual.shape[1]}, got {y.shape[1]}")
        if y.shape[-1] != self.out_dim:
            raise RuntimeError(f"Expected CNN output dim {self.out_dim}, got {y.shape[-1]}")
        y = self.dropout(y)
        y = residual + y
        if mask is not None:
            y = y * mask.to(dtype=y.dtype).unsqueeze(-1)
        return y


class SwiGLUFusion(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float, gate_init: float) -> None:
        super().__init__()
        self.base = nn.Linear(2 * dim, dim)
        self.a = nn.Linear(2 * dim, hidden_dim)
        self.v = nn.Linear(2 * dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, semantic: torch.Tensor, local: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        u = torch.cat([semantic, local], dim=-1)
        base = self.base(u)
        glu = F.silu(self.a(u)) * self.v(u)
        out = self.out(glu)
        alpha = torch.sigmoid(self.gate)
        fused = self.norm(base + alpha * self.dropout(out))
        return fused * mask.to(dtype=fused.dtype).unsqueeze(-1)


class ScaleDropout(nn.Module):
    def __init__(self, p: float) -> None:
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"ScaleDropout probability must be in [0, 1), got {p}")
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0.0:
            return x
        keep_prob = 1.0 - self.p
        mask = torch.empty(x.shape[0], 1, x.shape[-1], device=x.device, dtype=x.dtype).bernoulli_(keep_prob)
        return x * mask / keep_prob


class GatedAddFusion(nn.Module):
    def __init__(self, dim: int, dropout: float, branch_dropout: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= branch_dropout < 1.0:
            raise ValueError(f"branch_dropout must be in [0, 1), got {branch_dropout}")
        self.gate = nn.Linear(2 * dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
        self.branch_dropout = float(branch_dropout)

    def _drop_branch(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.branch_dropout <= 0.0:
            return x
        keep_prob = 1.0 - self.branch_dropout
        mask = torch.empty(x.shape[0], 1, 1, device=x.device, dtype=x.dtype).bernoulli_(keep_prob)
        return x * mask / keep_prob

    def forward(self, semantic: torch.Tensor, local: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        semantic = self._drop_branch(semantic)
        local = self._drop_branch(local)
        gate = torch.sigmoid(self.gate(torch.cat([semantic, local], dim=-1)))
        fused = gate * semantic + (1.0 - gate) * local
        fused = self.norm(self.dropout(fused))
        return fused * mask.to(dtype=fused.dtype).unsqueeze(-1)


class AddFusion(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, semantic: torch.Tensor, local: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        fused = self.norm(self.dropout(semantic + local))
        return fused * mask.to(dtype=fused.dtype).unsqueeze(-1)


class ConcatMLPFusion(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        hidden_dim = max(int(hidden_dim), dim)
        self.net = nn.Sequential(
            nn.Linear(2 * dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, semantic: torch.Tensor, local: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        fused = self.net(torch.cat([semantic, local], dim=-1))
        fused = self.norm(self.dropout(fused))
        return fused * mask.to(dtype=fused.dtype).unsqueeze(-1)


class AttentionWeightedMeanPooling(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        bool_mask = mask.bool()
        scores = self.score(x).squeeze(-1)
        scores = scores.masked_fill(~bool_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        weights = weights * bool_mask.to(dtype=weights.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)
        if not torch.isfinite(pooled).all():
            raise FloatingPointError("AttentionWeightedMeanPooling produced non-finite values.")
        return pooled
