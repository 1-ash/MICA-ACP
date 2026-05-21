from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch

def _ecdf_percentile(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    sorted_values = np.sort(values)
    return np.searchsorted(sorted_values, values, side="right").astype(np.float64) / float(values.size)


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(x, p=2, dim=-1, eps=1e-8)


def compute_embedding_hardness(
    sample_indices: torch.Tensor,
    labels: torch.Tensor,
    representations: torch.Tensor,
    topk_pos: int,
) -> dict[int, float]:
    labels = labels.detach().cpu().float()
    sample_indices = sample_indices.detach().cpu().long()
    representations = _normalize(representations.detach().cpu().float())
    pos_rows = torch.where(labels > 0.5)[0]
    neg_rows = torch.where(labels <= 0.5)[0]
    if pos_rows.numel() == 0 or neg_rows.numel() == 0:
        raise ValueError("Training set must contain both positive and negative samples for H_embed.")

    k = min(int(topk_pos), int(pos_rows.numel()))
    sim = representations[neg_rows] @ representations[pos_rows].T
    scores = sim.topk(k=k, dim=1).values.mean(dim=1).cpu().numpy()
    h_embed = _ecdf_percentile(scores)
    result: dict[int, float] = {}
    for row_tensor, score in zip(neg_rows, h_embed):
        sample_index = int(sample_indices[int(row_tensor.item())].item())
        result[sample_index] = float(score)
    return result


def nhl_warmup_gate(epoch: int, start_epoch: int, warmup_epochs: int) -> float:
    if epoch < start_epoch:
        return 0.0
    return float(min(max((epoch - start_epoch + 1) / float(max(warmup_epochs, 1)), 0.0), 1.0))


class HardNegativeMiner:
    def __init__(
        self,
        negative_indices: Sequence[int],
        h_sim: dict[int, float],
        rho: float,
        tau_loss: float,
        alpha: float,
        beta: float,
    ) -> None:
        self.negative_indices = [int(idx) for idx in negative_indices]
        self.h_sim = {int(k): float(v) for k, v in h_sim.items()}
        self.rho = float(rho)
        self.tau_loss = float(tau_loss)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.m_loss = {idx: 0.0 for idx in self.negative_indices}
        self.q_loss = {idx: 0.0 for idx in self.negative_indices}
        self.seen: set[int] = set()

    def update_loss_memory(
        self,
        sample_indices: torch.Tensor,
        labels: torch.Tensor,
        bce_losses: torch.Tensor,
    ) -> None:
        indices = sample_indices.detach().cpu().tolist()
        label_values = labels.detach().cpu().tolist()
        losses = bce_losses.detach().cpu().tolist()
        for idx, label, loss in zip(indices, label_values, losses):
            idx = int(idx)
            if float(label) > 0.5 or idx not in self.m_loss:
                continue
            loss_value = float(loss)
            if not math.isfinite(loss_value):
                continue
            if idx not in self.seen:
                self.m_loss[idx] = loss_value
                self.seen.add(idx)
            else:
                self.m_loss[idx] = self.rho * self.m_loss[idx] + (1.0 - self.rho) * loss_value

    def finalize_epoch(self) -> None:
        values = np.asarray([self.m_loss[idx] for idx in self.negative_indices], dtype=np.float64)
        q_raw = _ecdf_percentile(values)
        q_clip = np.minimum(q_raw, self.tau_loss)
        q_loss = q_clip / max(self.tau_loss, 1e-8)
        self.q_loss = {
            idx: float(value) for idx, value in zip(self.negative_indices, q_loss)
        }

    def combined_hardness(self, sample_index: int) -> float:
        idx = int(sample_index)
        h_sim = self.h_sim.get(idx, 0.0)
        q_loss = self.q_loss.get(idx, 0.0)
        return float((self.alpha * h_sim + (1.0 - self.alpha) * q_loss + self.beta * h_sim * q_loss) / (1.0 + self.beta))

    def set_h_sim(self, h_sim: dict[int, float]) -> None:
        self.h_sim = {int(k): float(v) for k, v in h_sim.items()}

    def update_h_sim(self, new_h_sim: dict[int, float], decay: float) -> None:
        """以 EMA 动量混合方式更新 H_embed/H_sim。

        ``decay`` 取值范围为 ``[0.0, 1.0)``：
        - ``decay == 0``：等价于 ``set_h_sim``，直接用 ``new_h_sim`` 覆盖（适用于 NHL 冷启动）。
        - ``0 < decay < 1``：``h_sim[i] = decay * h_sim[i] + (1 - decay) * new_h_sim[i]``，
          对所有 ``negative_indices`` 中存在的样本做逐样本混合；``new_h_sim`` 中缺失的索引保留旧值。
        """
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"update_h_sim decay must be in [0, 1), got {decay}")
        new_values = {int(k): float(v) for k, v in new_h_sim.items()}
        if decay == 0.0:
            self.h_sim = new_values
            return
        merged: dict[int, float] = {}
        for idx in self.negative_indices:
            new_val = new_values.get(int(idx))
            old_val = float(self.h_sim.get(int(idx), 0.0))
            if new_val is None:
                merged[int(idx)] = old_val
            else:
                merged[int(idx)] = float(decay) * old_val + (1.0 - float(decay)) * float(new_val)
        self.h_sim = merged

    def sample_weights(
        self,
        sample_indices: torch.Tensor,
        labels: torch.Tensor,
        gate: float,
        lambda_hn: float,
        w_hn_max: float,
    ) -> torch.Tensor:
        weights = torch.ones_like(labels, dtype=torch.float32)
        indices = sample_indices.detach().cpu().tolist()
        label_values = labels.detach().cpu().tolist()
        for row, (idx, label) in enumerate(zip(indices, label_values)):
            if float(label) <= 0.5:
                hardness = self.combined_hardness(int(idx))
                weights[row] = 1.0 + float(lambda_hn) * float(gate) * hardness
        weights = weights.clamp(min=1.0, max=float(w_hn_max))
        return weights.to(device=labels.device)

    def state_dict(self) -> dict[str, object]:
        return {
            "negative_indices": self.negative_indices,
            "h_sim": self.h_sim,
            "rho": self.rho,
            "tau_loss": self.tau_loss,
            "alpha": self.alpha,
            "beta": self.beta,
            "m_loss": self.m_loss,
            "q_loss": self.q_loss,
            "seen": sorted(self.seen),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.negative_indices = [int(idx) for idx in state.get("negative_indices", self.negative_indices)]
        self.h_sim = {int(k): float(v) for k, v in dict(state.get("h_sim", self.h_sim)).items()}
        self.rho = float(state.get("rho", self.rho))
        self.tau_loss = float(state.get("tau_loss", self.tau_loss))
        self.alpha = float(state.get("alpha", self.alpha))
        self.beta = float(state.get("beta", self.beta))
        self.m_loss = {int(k): float(v) for k, v in dict(state.get("m_loss", {})).items()}
        self.q_loss = {int(k): float(v) for k, v in dict(state.get("q_loss", {})).items()}
        self.seen = {int(idx) for idx in state.get("seen", [])}
