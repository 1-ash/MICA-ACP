from __future__ import annotations

import csv
import json
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from config import MICAConfig, build_arg_parser, config_from_args
from dataprocess.aaindex_selection import select_aaindex_mrmr
from dataprocess.data_loader import create_data_loader
from dataprocess.dataset import PeptideDataset, read_tsv_samples, split_train_valid
from dataprocess.esm_cache import ensure_esm_embeddings
from dataprocess.hard_negative import HardNegativeMiner, compute_embedding_hardness, nhl_warmup_gate
from dataprocess.priors import PriorBuilder
from model.mica_acp_model import MICAACPModel
from MyUtils.metrics import binary_classification_metrics
from MyUtils.swanlab_logger import SwanLabLogger


# SwanLab 上传字段精简版:
#   - 每个 epoch:  train/loss + 训练集 7 项指标 + valid/loss
#   - 训练结束:    test 集 7 项指标
# 这里把内部指标名 (Sensitivity/Specificity) 重命名为简短的 SE / SP。
SWANLAB_METRIC_NAME_MAP: dict[str, str] = {
    "ACC": "ACC",
    "Precision": "Precision",
    "Sensitivity": "SE",
    "Specificity": "SP",
    "F1": "F1",
    "AUC": "AUC",
    "MCC": "MCC",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    tensor_keys = ("token_ids", "valid_mask", "labels", "sample_indices", "p0", "esm_h")
    moved = dict(batch)
    for key in tensor_keys:
        moved[key] = batch[key].to(device)
    return moved


def build_optimizer_parameter_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lower_name = name.lower()
        if (
            param.ndim < 2
            or lower_name.endswith("bias")
            or "norm" in lower_name
            or "embedding" in lower_name
            or "gate" in lower_name
        ):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def assert_optimizer_has_only_model_params(model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    model_param_ids = {id(param) for param in model.parameters() if param.requires_grad}
    for group in optimizer.param_groups:
        for param in group["params"]:
            if id(param) not in model_param_ids:
                raise AssertionError("Optimizer contains parameters outside the trainable MICA-ACP model.")


def autocast_context(cfg: MICAConfig, device: torch.device):
    if cfg.use_amp and device.type == "cuda":
        return torch.cuda.amp.autocast()
    return nullcontext()


class ExponentialMovingAverage:
    def __init__(self, decay: float) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError(f"EMA decay must be in (0, 1), got {decay}")
        self.decay = float(decay)
        self.shadow: dict[str, torch.Tensor] = {}
        self.backup: dict[str, torch.Tensor] = {}

    @property
    def ready(self) -> bool:
        return bool(self.shadow)

    def update(self, model: nn.Module) -> None:
        state = model.state_dict()
        for name, value in state.items():
            detached = value.detach()
            if name not in self.shadow:
                self.shadow[name] = detached.clone()
            elif torch.is_floating_point(detached):
                self.shadow[name].mul_(self.decay).add_(detached, alpha=1.0 - self.decay)
            else:
                self.shadow[name] = detached.clone()

    def store(self, model: nn.Module) -> None:
        self.backup = {name: value.detach().clone() for name, value in model.state_dict().items()}

    def copy_to(self, model: nn.Module) -> None:
        if not self.ready:
            return
        current = model.state_dict()
        ema_state = {
            name: self.shadow.get(name, value).to(device=value.device, dtype=value.dtype)
            for name, value in current.items()
        }
        model.load_state_dict(ema_state, strict=True)

    def restore(self, model: nn.Module) -> None:
        if self.backup:
            model.load_state_dict(self.backup, strict=True)
            self.backup = {}

    def model_state_dict(self, model: nn.Module) -> dict[str, torch.Tensor]:
        current = model.state_dict()
        return {
            name: self.shadow.get(name, value).detach().clone()
            for name, value in current.items()
        }

    def state_dict(self) -> dict[str, Any]:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: dict[str, Any] | None) -> None:
        if not state:
            return
        self.decay = float(state.get("decay", self.decay))
        self.shadow = {str(name): value.detach().clone() for name, value in dict(state.get("shadow", {})).items()}


@torch.no_grad()
def predict_epoch(
    model: nn.Module,
    loader,
    device: torch.device,
    cfg: MICAConfig,
) -> dict[str, Any]:
    model.eval()
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    all_labels: list[float] = []
    all_probs: list[float] = []
    loss_sum = 0.0
    n = 0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        logits = model(
            token_ids=batch["token_ids"],
            p0=batch["p0"],
            esm_h=batch["esm_h"],
            valid_mask=batch["valid_mask"],
        )
        labels = batch["labels"].float()
        assert labels.shape == logits.shape
        bce = criterion(logits, labels)
        if not torch.isfinite(bce).all():
            raise FloatingPointError("Validation BCE contains non-finite values.")
        probs = torch.sigmoid(logits)
        loss_sum += float(bce.sum().item())
        n += int(labels.numel())
        all_labels.extend(labels.detach().cpu().tolist())
        all_probs.extend(probs.detach().cpu().tolist())
    return {
        "labels": all_labels,
        "probs": all_probs,
        "loss": loss_sum / max(n, 1),
    }


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader,
    device: torch.device,
    cfg: MICAConfig,
    threshold: float | None = None,
) -> dict[str, float]:
    predictions = predict_epoch(model, loader, device, cfg)
    metrics = binary_classification_metrics(
        predictions["labels"],
        predictions["probs"],
        threshold=cfg.threshold if threshold is None else threshold,
    )
    metrics["loss"] = float(predictions["loss"])
    return metrics


@torch.no_grad()
def collect_representations(
    model: nn.Module,
    loader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    all_indices: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    all_repr: list[torch.Tensor] = []
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        logits, seq_repr = model(
            token_ids=batch["token_ids"],
            p0=batch["p0"],
            esm_h=batch["esm_h"],
            valid_mask=batch["valid_mask"],
            return_repr=True,
        )
        if logits.shape != batch["labels"].shape:
            raise RuntimeError("Representation collection produced mismatched logits.")
        all_indices.append(batch["sample_indices"].detach().cpu())
        all_labels.append(batch["labels"].detach().cpu())
        all_repr.append(seq_repr.detach().cpu())
    return torch.cat(all_indices), torch.cat(all_labels), torch.cat(all_repr)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def write_logs(logs: list[dict[str, Any]], log_dir: str) -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    csv_path = Path(log_dir) / "train_log.csv"
    json_path = Path(log_dir) / "train_log.json"
    if logs:
        fieldnames = sorted({key for row in logs for key in row.keys()})
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(logs)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(logs), handle, ensure_ascii=False, indent=2)


def save_json(data: dict[str, Any], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(data), handle, ensure_ascii=False, indent=2)


def _threshold_candidates(probs: list[float], cfg: MICAConfig) -> np.ndarray:
    grid = np.arange(
        cfg.threshold_search_min,
        cfg.threshold_search_max + cfg.threshold_search_step * 0.5,
        cfg.threshold_search_step,
        dtype=np.float64,
    )
    unique_probs = np.unique(np.asarray(probs, dtype=np.float64))
    if unique_probs.size > 1:
        midpoints = (unique_probs[:-1] + unique_probs[1:]) / 2.0
        midpoints = midpoints[
            (midpoints >= cfg.threshold_search_min)
            & (midpoints <= cfg.threshold_search_max)
        ]
        grid = np.concatenate([grid, midpoints])
    return np.unique(np.clip(grid, 0.0, 1.0))


def search_best_threshold(labels: list[float], probs: list[float], cfg: MICAConfig) -> tuple[float, dict[str, float]]:
    best_threshold = float(cfg.threshold)
    best_metrics: dict[str, float] | None = None
    fallback_threshold = best_threshold
    fallback_metrics: dict[str, float] | None = None
    for threshold in _threshold_candidates(probs, cfg):
        metrics = binary_classification_metrics(labels, probs, threshold=float(threshold))
        metrics["threshold"] = float(threshold)
        feasible = metrics["Specificity"] >= cfg.threshold_min_specificity
        if fallback_metrics is None or (
            metrics["MCC"],
            metrics["ACC"],
            metrics["Specificity"],
        ) > (
            fallback_metrics["MCC"],
            fallback_metrics["ACC"],
            fallback_metrics["Specificity"],
        ):
            fallback_threshold = float(threshold)
            fallback_metrics = metrics
        if not feasible:
            continue
        if best_metrics is None or (
            metrics["MCC"],
            metrics["ACC"],
            metrics["Specificity"],
        ) > (
            best_metrics["MCC"],
            best_metrics["ACC"],
            best_metrics["Specificity"],
        ):
            best_threshold = float(threshold)
            best_metrics = metrics
    if best_metrics is None:
        assert fallback_metrics is not None
        fallback_metrics["threshold_constraint_satisfied"] = 0.0
        return fallback_threshold, fallback_metrics
    best_metrics["threshold_constraint_satisfied"] = 1.0
    return best_threshold, best_metrics


def save_checkpoint(
    path: str | Path,
    cfg: MICAConfig,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None,
    miner: HardNegativeMiner,
    ema: ExponentialMovingAverage | None,
    epoch: int,
    metrics: dict[str, float],
    training_state: dict[str, Any],
    *,
    use_ema_weights: bool,
) -> None:
    model_state = (
        ema.model_state_dict(model)
        if use_ema_weights and ema is not None and ema.ready
        else model.state_dict()
    )
    torch.save(
        {
            "config": cfg.to_dict(),
            "model_state": model_state,
            "raw_model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "miner_state": miner.state_dict(),
            "ema_state": ema.state_dict() if ema is not None else None,
            "checkpoint_uses_ema": bool(use_ema_weights and ema is not None and ema.ready),
            "epoch": epoch,
            "metrics": metrics,
            "training_state": training_state,
        },
        path,
    )


def scheduler_metric_value(cfg: MICAConfig, valid_metrics: dict[str, float]) -> float:
    metric_name = cfg.lr_scheduler_monitor
    if metric_name.startswith("valid_"):
        metric_name = metric_name.removeprefix("valid_")
    if metric_name == "loss":
        return float(valid_metrics["loss"])
    if metric_name not in valid_metrics:
        raise KeyError(
            f"Unknown lr scheduler monitor `{cfg.lr_scheduler_monitor}`. "
            f"Available validation metrics: {sorted(valid_metrics)}"
        )
    return float(valid_metrics[metric_name])


def load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    miner: HardNegativeMiner,
    ema: ExponentialMovingAverage | None,
    device: torch.device,
    cfg: MICAConfig,
    epoch: int,
    gate: float,
) -> dict[str, float]:
    model.train()
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    weighted_sum = 0.0
    unweighted_sum = 0.0
    n = 0
    negative_weights: list[float] = []
    all_labels: list[float] = []
    all_probs: list[float] = []
    rdrop_sum = 0.0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        labels = batch["labels"].float()
        assert labels.ndim == 1 and labels.dtype == torch.float32
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(cfg, device):
            logits1 = model(
                token_ids=batch["token_ids"],
                p0=batch["p0"],
                esm_h=batch["esm_h"],
                valid_mask=batch["valid_mask"],
            )
            assert logits1.shape == labels.shape
            bce1 = criterion(logits1, labels)
            rdrop_loss = torch.zeros((), dtype=bce1.dtype, device=bce1.device)
            if cfg.use_rdrop:
                logits2 = model(
                    token_ids=batch["token_ids"],
                    p0=batch["p0"],
                    esm_h=batch["esm_h"],
                    valid_mask=batch["valid_mask"],
                )
                assert logits2.shape == labels.shape
                bce2 = criterion(logits2, labels)
                bce = 0.5 * (bce1 + bce2)
                rdrop_loss = F.mse_loss(torch.sigmoid(logits1), torch.sigmoid(logits2), reduction="mean")
                logits = 0.5 * (logits1 + logits2)
            else:
                bce = bce1
                logits = logits1
            weights = miner.sample_weights(
                batch["sample_indices"],
                labels,
                gate=gate,
                lambda_hn=cfg.lambda_hn,
                w_hn_max=cfg.w_hn_max,
            )
            loss = (weights * bce).mean() + float(cfg.lambda_rdrop) * rdrop_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Training loss is not finite.")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if cfg.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        if ema is not None and cfg.use_ema and epoch >= cfg.ema_start_epoch:
            ema.update(model)

        miner.update_loss_memory(batch["sample_indices"], labels, bce.detach())
        batch_size = int(labels.numel())
        weighted_sum += float((weights.detach() * bce.detach()).sum().item())
        unweighted_sum += float(bce.detach().sum().item())
        rdrop_sum += float(rdrop_loss.detach().item()) * batch_size
        n += batch_size
        negative_weights.extend(weights.detach()[labels <= 0.5].cpu().tolist())
        with torch.no_grad():
            probs = torch.sigmoid(logits.detach().float())
        all_labels.extend(labels.detach().cpu().tolist())
        all_probs.extend(probs.cpu().tolist())

    miner.finalize_epoch()
    neg_mean = float(np.mean(negative_weights)) if negative_weights else 1.0
    neg_max = float(np.max(negative_weights)) if negative_weights else 1.0
    train_cls_metrics = binary_classification_metrics(all_labels, all_probs, threshold=cfg.threshold)
    metrics: dict[str, float] = {
        "train_weighted_loss": weighted_sum / max(n, 1),
        "train_unweighted_bce": unweighted_sum / max(n, 1),
        "G": float(gate),
        "negative_weight_mean": neg_mean,
        "negative_weight_max": neg_max,
        "train_rdrop_loss": rdrop_sum / max(n, 1),
    }
    metrics.update(train_cls_metrics)
    return metrics


def _swanlab_metric_payload(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    """根据 SWANLAB_METRIC_NAME_MAP 把指标 dict 转成 ``prefix/SHORT`` 的精简形式。"""
    payload: dict[str, float] = {}
    for long_name, short_name in SWANLAB_METRIC_NAME_MAP.items():
        if long_name in metrics:
            payload[f"{prefix}/{short_name}"] = float(metrics[long_name])
    return payload


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    cfg = config_from_args(args)

    # 1) 决定本轮实验的 run_name:
    #    - 用户显式传 --run_name: 直接使用
    #    - 续训 (--resume): 沿用原 checkpoint 所在的子目录名,新日志写到同一处
    #    - 否则: 用关键超参 + 时间戳自动生成
    if not cfg.run_name:
        if cfg.resume_path:
            cfg.run_name = Path(cfg.resume_path).resolve().parent.name
        else:
            cfg.run_name = cfg.auto_run_name()
    cfg.resolve_run_dirs()
    if not cfg.swanlab_experiment:
        cfg.swanlab_experiment = cfg.run_name
    cfg.ensure_dirs()
    set_seed(cfg.seed)

    print(f"[run] run_name       = {cfg.run_name}")
    print(f"[run] checkpoint_dir = {cfg.checkpoint_dir}")
    print(f"[run] log_dir        = {cfg.log_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    swanlab_logger = SwanLabLogger.from_config(cfg)
    if swanlab_logger.enabled:
        print(
            f"[swanlab] tracking enabled (project={cfg.swanlab_project!r}, "
            f"experiment={cfg.swanlab_experiment!r}, "
            f"workspace={cfg.swanlab_workspace or '<default>'}, "
            f"mode={cfg.swanlab_mode or 'cloud'})"
        )
    else:
        print(
            f"[swanlab] tracking DISABLED (cfg.use_swanlab={cfg.use_swanlab}); "
            f"加 --use_swanlab 或检查 swanlab 是否安装/可登录。"
        )

    try:
        train_all = read_tsv_samples(
            cfg.train_tsv_path,
            cfg.Lmax,
            cfg.positive_label_value,
            cfg.truncate_strategy,
        )
        test_samples = read_tsv_samples(
            cfg.test_tsv_path,
            cfg.Lmax,
            cfg.positive_label_value,
            cfg.truncate_strategy,
        )
        # 默认 use_test_as_valid=True：直接用 test 集做"验证"，对齐其他 ACP SOTA 模型的训练协议。
        # 优先级：显式 valid_tsv > use_test_as_valid > 从 train 切分 valid_ratio
        if cfg.valid_tsv_path:
            train_samples = train_all
            valid_samples = read_tsv_samples(
                cfg.valid_tsv_path,
                cfg.Lmax,
                cfg.positive_label_value,
                cfg.truncate_strategy,
            )
            print(f"[data] valid set <- {cfg.valid_tsv_path}")
        elif cfg.use_test_as_valid:
            train_samples = train_all
            valid_samples = test_samples
            print(f"[data] valid set <- TEST set ({cfg.test_tsv_path}) (use_test_as_valid=True)")
        else:
            train_samples, valid_samples = split_train_valid(train_all, cfg.valid_ratio, cfg.seed)
            print(f"[data] valid set <- 从 train 中切分 {cfg.valid_ratio:.0%}")

        if cfg.select_aaindex_mrmr and not cfg.disable_p0 and cfg.aaindex_dim > 0:
            selection = select_aaindex_mrmr(
                train_samples,
                cfg.aaindex_csv_path,
                cfg.Lmax,
                cfg.aaindex_dim,
                repeats=cfg.aaindex_selection_repeats,
                sample_fraction=cfg.aaindex_selection_sample_fraction,
                redundancy_threshold=cfg.aaindex_redundancy_threshold,
                seed=cfg.seed,
            )
            cfg.aaindex_indices = selection.indices
            cfg.sync_derived_dims()
            save_json(
                {
                    "indices": selection.indices,
                    "codes": selection.codes,
                    "groups": selection.groups,
                    "selected_frequency": selection.selected_frequency,
                    "average_rank": selection.average_rank,
                },
                Path(cfg.log_dir) / "aaindex_mrmr_selection.json",
            )

        prior_builder = None
        if not cfg.disable_p0 and cfg.p0_dim > 0:
            prior_builder = PriorBuilder(
                cfg.aaindex_csv_path,
                cfg.blosum_csv_path,
                cfg.Lmax,
                aaindex_indices=cfg.aaindex_indices,
                use_aac20_prior=cfg.use_aac20_prior,
                use_blosum_prior=not cfg.disable_blosum,
                truncate_strategy=cfg.truncate_strategy,
            )
            if prior_builder.p0_dim != cfg.p0_dim:
                raise ValueError(f"Configured p0_dim={cfg.p0_dim}, but PriorBuilder produced {prior_builder.p0_dim}.")
        train_esm = None
        test_esm = None
        valid_esm = None
        if not cfg.disable_esm:
            train_esm = ensure_esm_embeddings(train_samples, cfg, "train", device)
            test_esm = ensure_esm_embeddings(test_samples, cfg, "test", device)
            if valid_samples is test_samples:
                valid_esm = test_esm
            else:
                valid_esm = ensure_esm_embeddings(valid_samples, cfg, "valid", device)

        train_dataset = PeptideDataset(
            train_samples,
            cfg.Lmax,
            cfg.p0_dim,
            cfg.esm_dim,
            prior_builder=prior_builder,
            esm_features=train_esm,
            truncate_strategy=cfg.truncate_strategy,
        )
        test_dataset = PeptideDataset(
            test_samples,
            cfg.Lmax,
            cfg.p0_dim,
            cfg.esm_dim,
            prior_builder=prior_builder,
            esm_features=test_esm,
            truncate_strategy=cfg.truncate_strategy,
        )
        if valid_samples is test_samples:
            valid_dataset = test_dataset
        else:
            valid_dataset = PeptideDataset(
                valid_samples,
                cfg.Lmax,
                cfg.p0_dim,
                cfg.esm_dim,
                prior_builder=prior_builder,
                esm_features=valid_esm,
                truncate_strategy=cfg.truncate_strategy,
            )
        train_loader = create_data_loader(
            train_dataset,
            cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory and device.type == "cuda",
        )
        valid_loader = create_data_loader(
            valid_dataset,
            cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory and device.type == "cuda",
        )
        test_loader = create_data_loader(
            test_dataset,
            cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory and device.type == "cuda",
        )

        negative_indices = [sample.sample_index for sample in train_samples if sample.label <= 0.5]
        h_sim = {int(idx): 0.0 for idx in negative_indices}
        miner = HardNegativeMiner(
            negative_indices=negative_indices,
            h_sim=h_sim,
            rho=cfg.rho,
            tau_loss=cfg.tau_loss,
            alpha=cfg.alpha,
            beta=cfg.beta,
        )

        model = MICAACPModel(cfg).to(device)
        optimizer = torch.optim.AdamW(build_optimizer_parameter_groups(model, cfg.weight_decay), lr=cfg.lr)
        assert_optimizer_has_only_model_params(model, optimizer)
        scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=cfg.lr_scheduler_mode,
                factor=0.5,
                patience=3,
            )
            if cfg.use_scheduler
            else None
        )
        ema = ExponentialMovingAverage(cfg.ema_decay) if cfg.use_ema else None
        scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp and device.type == "cuda")

        start_epoch = 1
        best_acc = -1.0
        best_mcc = -2.0
        patience_used = 0
        last_h_embed_epoch: int | None = None
        logs: list[dict[str, Any]] = []

        if cfg.resume_path:
            checkpoint = load_checkpoint(cfg.resume_path, device)
            model.load_state_dict(checkpoint.get("raw_model_state", checkpoint["model_state"]))
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            if scheduler is not None and checkpoint.get("scheduler_state") is not None:
                scheduler.load_state_dict(checkpoint["scheduler_state"])
            if "miner_state" in checkpoint:
                miner.load_state_dict(checkpoint["miner_state"])
            if ema is not None and checkpoint.get("ema_state") is not None:
                ema.load_state_dict(checkpoint["ema_state"])
            start_epoch = int(checkpoint.get("epoch", 0)) + 1
            state = checkpoint.get("training_state", {})
            best_acc = state.get("best_acc", best_acc)
            best_mcc = state.get("best_mcc", best_mcc)
            patience_used = state.get("patience_used", patience_used)
            last_h_embed_epoch = state.get("last_h_embed_epoch")

        best_path = Path(cfg.checkpoint_dir) / "best.pt"
        last_path = Path(cfg.checkpoint_dir) / "last.pt"

        for epoch in range(start_epoch, cfg.epochs + 1):
            should_update_h_embed = epoch >= cfg.start_nhl_epoch and (
                last_h_embed_epoch is None
                or cfg.h_embed_update == "every_epoch"
                or (
                    cfg.h_embed_update == "every_n_epochs"
                    and epoch - int(last_h_embed_epoch) >= cfg.h_embed_update_interval
                )
            )
            h_embed_refreshed_this_epoch = False
            h_embed_used_ema = False
            if should_update_h_embed:
                repr_indices, repr_labels, seq_repr = collect_representations(model, train_loader, device)
                new_h_sim = compute_embedding_hardness(repr_indices, repr_labels, seq_repr, cfg.topk_pos)
                is_initial_h_embed = last_h_embed_epoch is None
                decay = 0.0 if is_initial_h_embed else float(cfg.h_embed_ema_decay)
                miner.update_h_sim(new_h_sim, decay=decay)
                last_h_embed_epoch = epoch
                h_embed_refreshed_this_epoch = True
                h_embed_used_ema = decay > 0.0

            gate = nhl_warmup_gate(epoch, cfg.start_nhl_epoch, cfg.nhl_warmup_epochs)
            train_metrics = train_one_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                miner,
                ema,
                device,
                cfg,
                epoch,
                gate,
            )
            eval_uses_ema = bool(ema is not None and ema.ready)
            if eval_uses_ema:
                ema.store(model)
                ema.copy_to(model)
            try:
                valid_metrics = evaluate_epoch(model, valid_loader, device, cfg)
            finally:
                if eval_uses_ema:
                    ema.restore(model)
            if scheduler is not None:
                scheduler.step(scheduler_metric_value(cfg, valid_metrics))

            train_unweighted = train_metrics["train_unweighted_bce"]

            lr = float(optimizer.param_groups[0]["lr"])
            row = {
                "epoch": epoch,
                "lr": lr,
                "eval_uses_ema": float(eval_uses_ema),
                "last_h_embed_epoch": float(last_h_embed_epoch or 0),
                **train_metrics,
                "valid_loss": valid_metrics["loss"],
                "valid_ACC": valid_metrics["ACC"],
                "valid_AUC": valid_metrics["AUC"],
                "valid_MCC": valid_metrics["MCC"],
                "valid_F1": valid_metrics["F1"],
                "valid_Sensitivity": valid_metrics["Sensitivity"],
                "valid_Specificity": valid_metrics["Specificity"],
            }
            logs.append(row)
            write_logs(logs, cfg.log_dir)

            if swanlab_logger.enabled:
                payload: dict[str, Any] = {
                    "train/loss": float(train_metrics["train_weighted_loss"]),
                    **_swanlab_metric_payload(train_metrics, "train"),
                    "valid/loss": float(valid_metrics["loss"]),
                    **_swanlab_metric_payload(valid_metrics, "valid"),
                }
                swanlab_logger.log(payload, step=int(epoch))

            improved = (
                valid_metrics["MCC"] > best_mcc
                or (valid_metrics["MCC"] == best_mcc and valid_metrics["ACC"] > best_acc)
            )
            training_state = {
                "best_acc": best_acc,
                "best_mcc": best_mcc,
                "patience_used": patience_used,
                "last_h_embed_epoch": last_h_embed_epoch,
            }
            if improved:
                best_acc = valid_metrics["ACC"]
                best_mcc = valid_metrics["MCC"]
                patience_used = 0
                training_state["best_acc"] = best_acc
                training_state["best_mcc"] = best_mcc
                training_state["patience_used"] = patience_used
                save_checkpoint(
                    best_path,
                    cfg,
                    model,
                    optimizer,
                    scheduler,
                    miner,
                    ema,
                    epoch,
                    valid_metrics,
                    training_state,
                    use_ema_weights=eval_uses_ema,
                )
            else:
                patience_used += 1
                training_state["patience_used"] = patience_used

            save_checkpoint(
                last_path,
                cfg,
                model,
                optimizer,
                scheduler,
                miner,
                ema,
                epoch,
                valid_metrics,
                training_state,
                use_ema_weights=False,
            )
            if last_h_embed_epoch is None:
                h_embed_age_str = "h_age=-"
            elif h_embed_refreshed_this_epoch:
                h_embed_age_str = "h_age=0*ema" if h_embed_used_ema else "h_age=0*init"
            else:
                h_embed_age_str = f"h_age={epoch - int(last_h_embed_epoch)}"
            print(
                f"epoch={epoch} lr={lr:.3e} "
                f"train_loss={train_metrics['train_weighted_loss']:.4f} "
                f"train_bce={train_unweighted:.4f} valid_loss={valid_metrics['loss']:.4f} "
                f"valid_ACC={valid_metrics['ACC']:.4f} valid_MCC={valid_metrics['MCC']:.4f} "
                f"G={train_metrics['G']:.4f} neg_w_max={train_metrics['negative_weight_max']:.4f} "
                f"{h_embed_age_str}"
            )

            if patience_used >= cfg.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}.")
                break

        if best_path.exists():
            best_checkpoint = load_checkpoint(best_path, device)
            model.load_state_dict(best_checkpoint["model_state"])
        valid_predictions = predict_epoch(model, valid_loader, device, cfg)
        best_threshold, valid_at_best = search_best_threshold(
            valid_predictions["labels"],
            valid_predictions["probs"],
            cfg,
        )
        valid_at_best["loss"] = float(valid_predictions["loss"])
        valid_default = binary_classification_metrics(
            valid_predictions["labels"],
            valid_predictions["probs"],
            threshold=cfg.threshold,
        )
        valid_default["loss"] = float(valid_predictions["loss"])
        valid_default["threshold"] = float(cfg.threshold)

        test_predictions = predict_epoch(model, test_loader, device, cfg)
        test_default = binary_classification_metrics(
            test_predictions["labels"],
            test_predictions["probs"],
            threshold=cfg.threshold,
        )
        test_default["loss"] = float(test_predictions["loss"])
        test_default["threshold"] = float(cfg.threshold)
        test_at_best = binary_classification_metrics(
            test_predictions["labels"],
            test_predictions["probs"],
            threshold=best_threshold,
        )
        test_at_best["loss"] = float(test_predictions["loss"])
        test_at_best["threshold"] = float(best_threshold)

        test_metrics = {
            "best_valid_threshold": float(best_threshold),
            "valid_metrics_at_default_threshold": valid_default,
            "valid_metrics_at_best_threshold": valid_at_best,
            "test_metrics_at_default_threshold": test_default,
            "test_metrics_at_best_threshold": test_at_best,
        }
        save_json(test_metrics, Path(cfg.log_dir) / "test_metrics.json")
        print("test_metrics:", json.dumps(_json_safe(test_metrics), ensure_ascii=False))

        if swanlab_logger.enabled:
            swanlab_logger.log(_swanlab_metric_payload(test_at_best, "test"))
    finally:
        swanlab_logger.finish()


if __name__ == "__main__":
    main()
