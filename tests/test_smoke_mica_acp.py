from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from config import MICAConfig, build_arg_parser, config_from_args
from dataprocess.dataset import PeptideDataset, PeptideSample, collate_peptide_batch, tokenize_sequence
from dataprocess.hard_negative import HardNegativeMiner, compute_embedding_hardness
from model.mica_acp_model import MICAACPModel
from train import build_optimizer_parameter_groups


def _toy_features(samples, cfg):
    p0_features = {}
    esm_features = {}
    for sample in samples:
        p0 = torch.randn(cfg.Lmax, cfg.p0_dim)
        esm = torch.randn(cfg.Lmax, cfg.esm_dim)
        p0[len(sample.sequence) :] = 0.0
        esm[len(sample.sequence) :] = 0.0
        p0_features[sample.sample_index] = p0
        esm_features[sample.sample_index] = esm
    return p0_features, esm_features


def _toy_batch(cfg):
    samples = [
        PeptideSample(0, "0", 1.0, "ACDEFGHIK"),
        PeptideSample(1, "1", 1.0, "KLMNPQRST"),
        PeptideSample(2, "2", 0.0, "VVVVVVVV"),
        PeptideSample(3, "3", 0.0, "WYACDEFG"),
    ]
    p0_features, esm_features = _toy_features(samples, cfg)
    dataset = PeptideDataset(
        samples,
        cfg.Lmax,
        cfg.p0_dim,
        cfg.esm_dim,
        p0_features=p0_features,
        esm_features=esm_features,
        truncate_strategy=cfg.truncate_strategy,
    )
    return collate_peptide_batch([dataset[i] for i in range(len(dataset))])


def test_mica_acp_smoke_step_and_hard_negative_logic():
    cfg = MICAConfig(
        d=32,
        d_ff=64,
        num_heads=4,
        batch_size=4,
        dropout=0.0,
        aaindex_dim=3,
        cnn_kernels=[3, 5, 7],
    )
    assert cfg.p0_dim == 43
    samples = [
        PeptideSample(0, "0", 1.0, "ACDEFGHIK"),
        PeptideSample(1, "1", 1.0, "KLMNPQRST"),
        PeptideSample(2, "2", 0.0, "VVVVVVVV"),
        PeptideSample(3, "3", 0.0, "WYACDEFG"),
    ]
    p0_features, esm_features = _toy_features(samples, cfg)
    dataset = PeptideDataset(
        samples,
        cfg.Lmax,
        cfg.p0_dim,
        cfg.esm_dim,
        p0_features=p0_features,
        esm_features=esm_features,
    )
    batch = collate_peptide_batch([dataset[i] for i in range(len(dataset))])

    model = MICAACPModel(cfg)
    optimizer = torch.optim.AdamW(build_optimizer_parameter_groups(model, cfg.weight_decay), lr=cfg.lr)
    optimizer_param_ids = {id(param) for group in optimizer.param_groups for param in group["params"]}
    assert optimizer_param_ids <= {id(param) for param in model.parameters() if param.requires_grad}

    logits = model(
        token_ids=batch["token_ids"],
        p0=batch["p0"],
        esm_h=batch["esm_h"],
        valid_mask=batch["valid_mask"],
    )
    assert logits.shape == (4,)
    criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
    bce = criterion(logits, batch["labels"])
    loss = bce.mean()
    assert torch.isfinite(loss)

    h_sim = compute_embedding_hardness(
        batch["sample_indices"],
        batch["labels"],
        torch.randn(len(samples), cfg.d),
        topk_pos=2,
    )
    miner = HardNegativeMiner(
        negative_indices=[2, 3],
        h_sim=h_sim,
        rho=cfg.rho,
        tau_loss=cfg.tau_loss,
        alpha=cfg.alpha,
        beta=cfg.beta,
    )
    weights = miner.sample_weights(
        batch["sample_indices"],
        batch["labels"],
        gate=0.5,
        lambda_hn=cfg.lambda_hn,
        w_hn_max=cfg.w_hn_max,
    )
    assert weights.shape == batch["labels"].shape
    assert torch.isfinite(weights).all()
    miner.update_loss_memory(batch["sample_indices"], batch["labels"], bce.detach())
    miner.finalize_epoch()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def test_ablation_cli_switches_are_wired_into_config():
    args = build_arg_parser().parse_args(
        [
            "--Lmax",
            "30",
            "--truncate_strategy",
            "tail",
            "--disable_blosum",
            "--disable_cross_attention",
            "--fusion_type",
            "swiglu",
            "--post_sequence_model",
            "gru",
            "--lambda_hn",
            "1.5",
            "--topk_pos",
            "5",
            "--rho",
            "0.8",
            "--tau_loss",
            "0.9",
            "--alpha",
            "0.7",
            "--beta",
            "0.2",
        ]
    )
    cfg = config_from_args(args)
    assert cfg.Lmax == 30
    assert cfg.truncate_strategy == "tail"
    assert cfg.disable_blosum is True
    assert cfg.blosum_dim == 0
    assert cfg.p0_dim == cfg.aaindex_dim + cfg.aac_dim
    assert cfg.disable_cross_attention is True
    assert cfg.fusion_type == "swiglu"
    assert cfg.post_sequence_model == "gru"
    assert cfg.use_post_fusion_bigru is False
    assert cfg.lambda_hn == 1.5
    assert cfg.topk_pos == 5
    assert cfg.rho == 0.8
    assert cfg.tau_loss == 0.9
    assert cfg.alpha == 0.7
    assert cfg.beta == 0.2


def test_truncate_strategy_tail_changes_tokenized_region():
    token_ids, mask = tokenize_sequence("ACDEFGHIK", max_len=4, truncate_strategy="tail")
    assert mask.tolist() == [1.0, 1.0, 1.0, 1.0]
    assert token_ids.tolist() == [6, 7, 8, 9]


@pytest.mark.parametrize(
    "overrides",
    [
        {"disable_p0": True, "disable_token_cnn": True},
        {"disable_esm": True, "disable_token_cnn": True},
        {"disable_esm": True, "disable_p0": True},
        {"disable_cross_attention": True, "fusion_type": "add", "post_sequence_model": "none"},
        {"fusion_type": "concat", "post_sequence_model": "gru"},
        {"fusion_type": "swiglu", "post_sequence_model": "attention"},
        {"fusion_type": "gated_add", "post_sequence_model": "transformer", "bigru_layers": 1},
        {"fusion_type": "gated_add", "post_sequence_model": "cnn", "bigru_layers": 1},
    ],
)
def test_ablation_model_variants_forward(overrides):
    cfg = MICAConfig(
        d=32,
        d_ff=64,
        num_heads=4,
        dropout=0.0,
        aaindex_dim=3,
        cnn_kernels=[3, 5],
        esm_dim=16,
        Lmax=12,
        **overrides,
    )
    batch = _toy_batch(cfg)
    model = MICAACPModel(cfg)
    logits = model(
        token_ids=batch["token_ids"],
        p0=batch["p0"],
        esm_h=batch["esm_h"],
        valid_mask=batch["valid_mask"],
    )
    assert logits.shape == (4,)
    assert torch.isfinite(logits).all()
