from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import torch

from config import MICAConfig
from dataprocess.data_loader import create_data_loader
from dataprocess.dataset import PeptideDataset, PeptideSample, clean_sequence, samples_from_sequences
from dataprocess.esm_cache import ensure_esm_embeddings
from dataprocess.priors import PriorBuilder
from model.mica_acp_model import MICAACPModel


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MICA-ACP inference.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sequence")
    parser.add_argument("--input_tsv")
    parser.add_argument("--input_csv")
    parser.add_argument("--output_csv")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--aaindex_csv", dest="aaindex_csv_path")
    parser.add_argument("--blosum_csv", dest="blosum_csv_path")
    parser.add_argument("--esm_model_dir")
    parser.add_argument("--cache_dir")
    parser.add_argument("--Lmax", type=int)
    parser.add_argument("--truncate_strategy", choices=["head", "tail", "center", "head_tail"])
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--force_rebuild_cache", action="store_true")
    return parser


def load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def read_inference_file(path: str | Path, delimiter: str, cfg: MICAConfig) -> list[PeptideSample]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Inference input file does not exist: {path}")
    samples: list[PeptideSample] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if "text" not in (reader.fieldnames or []):
            raise ValueError(f"Inference file must contain a `text` column: {path}")
        for row_idx, row in enumerate(reader):
            seq = clean_sequence(row["text"], cfg.Lmax, cfg.truncate_strategy)
            if seq is None:
                continue
            source_index = str(row.get("index", row_idx))
            samples.append(PeptideSample(row_idx, source_index, 0.0, seq))
    if not samples:
        raise ValueError(f"No valid peptide sequences remain after cleaning: {path}")
    return samples


@torch.no_grad()
def run_inference(cfg: MICAConfig, checkpoint: dict[str, Any], samples: list[PeptideSample], device: torch.device):
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
    esm_features = None if cfg.disable_esm else ensure_esm_embeddings(samples, cfg, "infer", device)
    dataset = PeptideDataset(
        samples,
        cfg.Lmax,
        cfg.p0_dim,
        cfg.esm_dim,
        prior_builder=prior_builder,
        esm_features=esm_features,
        truncate_strategy=cfg.truncate_strategy,
    )
    loader = create_data_loader(
        dataset,
        cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    model = MICAACPModel(cfg).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    rows: list[dict[str, object]] = []
    for batch in loader:
        token_ids = batch["token_ids"].to(device)
        p0 = batch["p0"].to(device)
        esm_h = batch["esm_h"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        logits = model(token_ids=token_ids, p0=p0, esm_h=esm_h, valid_mask=valid_mask)
        probs = torch.sigmoid(logits).detach().cpu().tolist()
        for sequence, prob in zip(batch["sequences"], probs):
            rows.append(
                {
                    "sequence": sequence,
                    "prob_acp": float(prob),
                    "pred_label": int(float(prob) >= cfg.threshold),
                }
            )
    return rows


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(args.checkpoint, device)
    cfg = MICAConfig.from_dict(checkpoint.get("config", {}))
    for key in (
        "aaindex_csv_path",
        "blosum_csv_path",
        "esm_model_dir",
        "cache_dir",
        "Lmax",
        "truncate_strategy",
        "batch_size",
        "threshold",
    ):
        value = getattr(args, key, None)
        if value is not None:
            setattr(cfg, key, value)
    if args.force_rebuild_cache:
        cfg.force_rebuild_cache = True
    cfg.sync_derived_dims()
    cfg.ensure_dirs()

    if args.sequence:
        samples = samples_from_sequences([args.sequence], cfg.Lmax, cfg.truncate_strategy)
    elif args.input_tsv:
        samples = read_inference_file(args.input_tsv, "\t", cfg)
    elif args.input_csv:
        samples = read_inference_file(args.input_csv, ",", cfg)
    else:
        raise ValueError("Provide one of --sequence, --input_tsv, or --input_csv.")

    rows = run_inference(cfg, checkpoint, samples, device)
    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["sequence", "prob_acp", "pred_label"])
            writer.writeheader()
            writer.writerows(rows)
    else:
        for row in rows:
            print(f"{row['sequence']}\t{row['prob_acp']:.6f}\t{row['pred_label']}")


if __name__ == "__main__":
    main()
