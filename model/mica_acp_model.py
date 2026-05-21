from __future__ import annotations

import torch
from torch import nn

from config import MICAConfig
from model.modules import (
    AddFusion,
    AttentionWeightedMeanPooling,
    ConcatMLPFusion,
    GatedAddFusion,
    MultiScaleCNNBlock,
    ScaleDropout,
    SwiGLUFusion,
)


class MICAACPModel(nn.Module):
    def __init__(self, cfg: MICAConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.max_len = cfg.Lmax
        self.p0_dim = cfg.p0_dim
        self.esm_dim = cfg.esm_dim
        self.dim = cfg.d

        self.use_esm = not bool(cfg.disable_esm)
        self.use_p0 = not bool(cfg.disable_p0) and cfg.p0_dim > 0
        self.use_local = not bool(cfg.disable_token_cnn)
        if not (self.use_esm or self.use_p0 or self.use_local):
            raise ValueError("MICAACPModel requires at least one enabled branch.")

        if self.use_esm:
            self.esm_proj = nn.Linear(cfg.esm_dim, cfg.d)
            self.esm_norm = nn.LayerNorm(cfg.d)
        if self.use_p0:
            if cfg.use_prior_bottleneck:
                self.p0_branch = nn.Sequential(
                    nn.LayerNorm(cfg.p0_dim),
                    ScaleDropout(cfg.scale_dropout),
                    nn.Linear(cfg.p0_dim, cfg.prior_bottleneck_dim),
                    nn.GELU(),
                    nn.Dropout(cfg.prior_dropout),
                    nn.Linear(cfg.prior_bottleneck_dim, cfg.d),
                    nn.LayerNorm(cfg.d),
                )
            else:
                self.p0_branch = nn.Sequential(
                    nn.LayerNorm(cfg.p0_dim),
                    ScaleDropout(cfg.scale_dropout),
                    nn.Linear(cfg.p0_dim, cfg.d),
                    nn.Dropout(cfg.prior_dropout),
                    nn.LayerNorm(cfg.d),
                )

        self.use_cross_attention = self.use_esm and self.use_p0 and not bool(cfg.disable_cross_attention)
        if self.use_cross_attention:
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=cfg.d,
                num_heads=cfg.num_heads,
                dropout=cfg.dropout,
                batch_first=True,
            )
            self.cross_attn_gate = nn.Parameter(torch.tensor(float(cfg.gate_init)))
            self.cross_attn_dropout = nn.Dropout(cfg.dropout)
            self.cross_attn_norm = nn.LayerNorm(cfg.d)
        elif self.use_esm and self.use_p0:
            self.semantic_prior_add_norm = nn.LayerNorm(cfg.d)

        if self.use_local:
            self.embedding = nn.Embedding(21, cfg.d, padding_idx=0)
            self.embedding_norm = nn.LayerNorm(cfg.d)
            self.cnn_blocks = nn.ModuleList(
                [MultiScaleCNNBlock(cfg.d, cfg.cnn_kernels, cfg.dropout) for _ in range(cfg.num_cnn_layers)]
            )

        if self.use_local and (self.use_esm or self.use_p0):
            self.fusion = self._build_fusion(cfg)
        self.post_sequence_model = cfg.post_sequence_model
        self._init_post_sequence_model(cfg)
        self.pooling = AttentionWeightedMeanPooling(cfg.d)
        self.classifier = nn.Sequential(
            nn.Linear(cfg.d, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(64, 1),
        )

    def _build_fusion(self, cfg: MICAConfig) -> nn.Module:
        if cfg.fusion_type == "gated_add":
            return GatedAddFusion(cfg.d, cfg.dropout, cfg.branch_dropout)
        if cfg.fusion_type == "concat":
            return ConcatMLPFusion(cfg.d, cfg.d_ff, cfg.dropout)
        if cfg.fusion_type == "add":
            return AddFusion(cfg.d, cfg.dropout)
        if cfg.fusion_type == "swiglu":
            return SwiGLUFusion(cfg.d, cfg.d_ff, cfg.dropout, cfg.gate_init)
        raise ValueError(f"Unknown fusion_type: {cfg.fusion_type!r}")

    def _init_post_sequence_model(self, cfg: MICAConfig) -> None:
        if cfg.post_sequence_model == "none":
            return
        if cfg.post_sequence_model == "bigru":
            self.post_bigru = nn.GRU(
                input_size=cfg.d,
                hidden_size=cfg.bigru_hidden,
                num_layers=cfg.bigru_layers,
                dropout=cfg.bigru_dropout if cfg.bigru_layers > 1 else 0.0,
                batch_first=True,
                bidirectional=True,
            )
            self.post_bigru_proj = nn.Linear(2 * cfg.bigru_hidden, cfg.d)
            self.post_bigru_norm = nn.LayerNorm(cfg.d)
            return
        if cfg.post_sequence_model in {"gru", "lstm"}:
            rnn_cls = nn.LSTM if cfg.post_sequence_model == "lstm" else nn.GRU
            self.post_rnn = rnn_cls(
                input_size=cfg.d,
                hidden_size=cfg.bigru_hidden,
                num_layers=cfg.bigru_layers,
                dropout=cfg.bigru_dropout if cfg.bigru_layers > 1 else 0.0,
                batch_first=True,
                bidirectional=False,
            )
            self.post_rnn_proj = nn.Linear(cfg.bigru_hidden, cfg.d)
            self.post_rnn_norm = nn.LayerNorm(cfg.d)
            return
        if cfg.post_sequence_model == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=cfg.d,
                nhead=cfg.num_heads,
                dim_feedforward=cfg.d_ff,
                dropout=cfg.dropout,
                batch_first=True,
                norm_first=True,
            )
            self.post_transformer = nn.TransformerEncoder(layer, num_layers=max(1, cfg.bigru_layers))
            return
        if cfg.post_sequence_model == "cnn":
            self.post_cnn_blocks = nn.ModuleList(
                [MultiScaleCNNBlock(cfg.d, cfg.cnn_kernels, cfg.dropout) for _ in range(max(1, cfg.bigru_layers))]
            )
            return
        if cfg.post_sequence_model == "attention":
            self.post_attn = nn.MultiheadAttention(
                embed_dim=cfg.d,
                num_heads=cfg.num_heads,
                dropout=cfg.dropout,
                batch_first=True,
            )
            self.post_attn_dropout = nn.Dropout(cfg.dropout)
            self.post_attn_norm = nn.LayerNorm(cfg.d)
            return
        raise ValueError(f"Unknown post_sequence_model: {cfg.post_sequence_model!r}")

    def encode(
        self,
        token_ids: torch.Tensor,
        p0: torch.Tensor,
        esm_h: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        self._assert_inputs(token_ids, p0, esm_h, valid_mask)

        mask = valid_mask.bool()
        padding_key_mask = ~mask
        float_mask = mask.to(dtype=esm_h.dtype)

        semantic = None
        if self.use_esm:
            semantic = self.esm_norm(self.esm_proj(esm_h.float()))
            semantic = semantic * float_mask.unsqueeze(-1)

        prior = None
        if self.use_p0:
            prior = self.p0_branch(p0.float())
            prior = prior * float_mask.unsqueeze(-1)

        semantic_prior = self._combine_semantic_prior(semantic, prior, mask, padding_key_mask)

        local = None
        if self.use_local:
            local = self.embedding_norm(self.embedding(token_ids.long()))
            local = local * float_mask.unsqueeze(-1)
            for block in self.cnn_blocks:
                local = block(local, mask)

        if semantic_prior is not None and local is not None:
            fused = self.fusion(semantic_prior, local, mask)
        elif semantic_prior is not None:
            fused = semantic_prior
        elif local is not None:
            fused = local
        else:
            raise RuntimeError("No active representation branch produced features.")

        fused = self._apply_post_sequence(fused, mask, padding_key_mask)
        return self.pooling(fused, mask)

    def _combine_semantic_prior(
        self,
        semantic: torch.Tensor | None,
        prior: torch.Tensor | None,
        mask: torch.Tensor,
        padding_key_mask: torch.Tensor,
    ) -> torch.Tensor | None:
        if semantic is None and prior is None:
            return None
        if semantic is None:
            return prior
        if prior is None:
            return semantic
        if not self.use_cross_attention:
            combined = self.semantic_prior_add_norm(semantic + prior)
            return combined * mask.to(dtype=combined.dtype).unsqueeze(-1)

        if self.cfg.cross_attention_direction == "semantic_to_prior":
            query = semantic
            key = prior
            value = prior
            residual = semantic
        else:
            query = prior
            key = semantic
            value = semantic
            residual = semantic
        context, _ = self.cross_attn(
            query=query,
            key=key,
            value=value,
            key_padding_mask=padding_key_mask,
            need_weights=False,
        )
        alpha_ca = torch.sigmoid(self.cross_attn_gate)
        out = self.cross_attn_norm(residual + alpha_ca * self.cross_attn_dropout(context))
        return out * mask.to(dtype=out.dtype).unsqueeze(-1)

    def _apply_post_sequence(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        padding_key_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.post_sequence_model == "none":
            return x
        float_mask = mask.to(dtype=x.dtype).unsqueeze(-1)
        if self.post_sequence_model in {"bigru", "gru", "lstm"}:
            lengths = mask.sum(dim=1).detach().cpu().to(torch.long)
            packed = nn.utils.rnn.pack_padded_sequence(
                x,
                lengths=lengths,
                batch_first=True,
                enforce_sorted=False,
            )
            rnn = self.post_bigru if self.post_sequence_model == "bigru" else self.post_rnn
            packed_out, _ = rnn(packed)
            interacted, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out,
                batch_first=True,
                total_length=self.max_len,
            )
            if self.post_sequence_model == "bigru":
                out = self.post_bigru_norm(self.post_bigru_proj(interacted))
            else:
                out = self.post_rnn_norm(self.post_rnn_proj(interacted))
            return out * float_mask
        if self.post_sequence_model == "transformer":
            out = self.post_transformer(x, src_key_padding_mask=padding_key_mask)
            return out * float_mask
        if self.post_sequence_model == "cnn":
            out = x
            for block in self.post_cnn_blocks:
                out = block(out, mask)
            return out * float_mask
        if self.post_sequence_model == "attention":
            context, _ = self.post_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=padding_key_mask,
                need_weights=False,
            )
            out = self.post_attn_norm(x + self.post_attn_dropout(context))
            return out * float_mask
        raise RuntimeError(f"Unhandled post_sequence_model: {self.post_sequence_model!r}")

    def forward(
        self,
        token_ids: torch.Tensor,
        p0: torch.Tensor,
        esm_h: torch.Tensor,
        valid_mask: torch.Tensor,
        return_repr: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch_size = token_ids.shape[0]
        pooled = self.encode(token_ids=token_ids, p0=p0, esm_h=esm_h, valid_mask=valid_mask)
        logits = self.classifier(pooled).squeeze(-1)
        assert logits.shape == (batch_size,)
        if return_repr:
            return logits, pooled
        return logits

    def _assert_inputs(
        self,
        token_ids: torch.Tensor,
        p0: torch.Tensor,
        esm_h: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> None:
        if token_ids.dim() != 2:
            raise ValueError(f"token_ids must have shape [B, Lmax], got {tuple(token_ids.shape)}")
        batch_size = token_ids.shape[0]
        expected_tokens = (batch_size, self.max_len)
        expected_p0 = (batch_size, self.max_len, self.p0_dim)
        expected_esm = (batch_size, self.max_len, self.esm_dim)
        assert token_ids.shape == expected_tokens, (token_ids.shape, expected_tokens)
        assert valid_mask.shape == expected_tokens, (valid_mask.shape, expected_tokens)
        assert p0.shape == expected_p0, (p0.shape, expected_p0)
        assert esm_h.shape == expected_esm, (esm_h.shape, expected_esm)
        if token_ids.dtype != torch.long:
            raise TypeError(f"token_ids must be torch.long, got {token_ids.dtype}")
        if int(token_ids.min().item()) < 0 or int(token_ids.max().item()) > 20:
            raise ValueError("token_ids must stay in the inclusive range [0, 20].")
        mask = valid_mask.bool()
        if not mask.any(dim=1).all():
            raise ValueError("Every sample must contain at least one valid residue.")
        if not torch.equal(token_ids.eq(0), ~mask):
            raise ValueError("token_ids padding positions must match valid_mask.")
        if p0.numel():
            padding_p0 = p0.masked_select((~mask).unsqueeze(-1))
            if padding_p0.numel() and not torch.allclose(padding_p0, torch.zeros_like(padding_p0)):
                raise ValueError("P0 padding positions must be zero.")


def build_mica_acp_model(cfg: MICAConfig) -> MICAACPModel:
    return MICAACPModel(cfg)
