from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_ID = {aa: i + 1 for i, aa in enumerate(AMINO_ACIDS)}
ID_TO_AA = {i: aa for aa, i in AA_TO_ID.items()}
PAD_ID = 0
REQUIRED_TSV_COLUMNS = {"index", "label", "text"}
TRUNCATE_STRATEGIES = {"head", "tail", "center", "head_tail"}


@dataclass(frozen=True)
class PeptideSample:
    sample_index: int
    source_index: str
    label: float
    sequence: str


def truncate_sequence(sequence: str, max_len: int, strategy: str = "head") -> str:
    if max_len <= 0:
        raise ValueError(f"max_len must be positive, got {max_len}")
    if strategy not in TRUNCATE_STRATEGIES:
        raise ValueError(f"Unknown truncate_strategy {strategy!r}; expected one of {sorted(TRUNCATE_STRATEGIES)}")
    if len(sequence) <= max_len:
        return sequence
    if strategy == "head":
        return sequence[:max_len]
    if strategy == "tail":
        return sequence[-max_len:]
    if strategy == "center":
        start = max((len(sequence) - max_len) // 2, 0)
        return sequence[start : start + max_len]
    head_len = (max_len + 1) // 2
    tail_len = max_len - head_len
    if tail_len <= 0:
        return sequence[:head_len]
    return sequence[:head_len] + sequence[-tail_len:]


def clean_sequence(sequence: str, max_len: int, truncate_strategy: str = "head") -> str | None:
    seq = str(sequence).strip().upper()
    if not seq:
        return None
    if any(aa not in AA_TO_ID for aa in seq):
        return None
    return truncate_sequence(seq, max_len, truncate_strategy)


def tokenize_sequence(sequence: str, max_len: int, truncate_strategy: str = "head") -> tuple[np.ndarray, np.ndarray]:
    seq = clean_sequence(sequence, max_len, truncate_strategy)
    if seq is None:
        raise ValueError(f"Invalid peptide sequence: {sequence!r}")
    token_ids = np.zeros(max_len, dtype=np.int64)
    mask = np.zeros(max_len, dtype=np.float32)
    encoded = [AA_TO_ID[aa] for aa in seq]
    token_ids[: len(encoded)] = encoded
    mask[: len(encoded)] = 1.0
    return token_ids, mask


def _parse_label(value: str, positive_label_value: int) -> float:
    if positive_label_value not in (0, 1):
        raise ValueError(f"positive_label_value must be 0 or 1, got {positive_label_value}")
    try:
        raw = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Label must be 0 or 1, got {value!r}") from exc
    if raw not in (0, 1):
        raise ValueError(f"Label must be 0 or 1, got {value!r}")
    return 1.0 if raw == positive_label_value else 0.0


def read_tsv_samples(
    path: str | Path,
    max_len: int,
    positive_label_value: int = 1,
    truncate_strategy: str = "head",
) -> list[PeptideSample]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"TSV file does not exist: {path}")

    cleaned_rows: list[tuple[str, float, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_TSV_COLUMNS - columns
        if missing:
            raise ValueError(f"TSV file {path} is missing required columns: {sorted(missing)}")
        for row in reader:
            seq = clean_sequence(row["text"], max_len, truncate_strategy)
            if seq is None:
                continue
            label = _parse_label(row["label"], positive_label_value)
            cleaned_rows.append((str(row["index"]), label, seq))

    if not cleaned_rows:
        raise ValueError(f"No valid peptide samples remain after cleaning: {path}")

    raw_indices: list[int] = []
    can_use_raw_index = True
    for source_index, _, _ in cleaned_rows:
        try:
            raw_indices.append(int(source_index))
        except ValueError:
            can_use_raw_index = False
            break
    if can_use_raw_index and len(set(raw_indices)) != len(raw_indices):
        can_use_raw_index = False

    samples: list[PeptideSample] = []
    for row_pos, (source_index, label, seq) in enumerate(cleaned_rows):
        sample_index = raw_indices[row_pos] if can_use_raw_index else row_pos
        samples.append(
            PeptideSample(
                sample_index=sample_index,
                source_index=source_index,
                label=label,
                sequence=seq,
            )
        )
    return samples


def samples_from_sequences(
    sequences: Sequence[str],
    max_len: int,
    truncate_strategy: str = "head",
) -> list[PeptideSample]:
    samples: list[PeptideSample] = []
    for idx, seq in enumerate(sequences):
        cleaned = clean_sequence(seq, max_len, truncate_strategy)
        if cleaned is None:
            continue
        samples.append(PeptideSample(idx, str(idx), 0.0, cleaned))
    if not samples:
        raise ValueError("No valid peptide sequences were provided.")
    return samples


def split_train_valid(
    samples: Sequence[PeptideSample],
    valid_ratio: float,
    seed: int,
) -> tuple[list[PeptideSample], list[PeptideSample]]:
    if not 0.0 < valid_ratio < 1.0:
        raise ValueError(f"valid_ratio must be in (0, 1), got {valid_ratio}")
    rng = random.Random(seed)
    by_label = {0.0: [], 1.0: []}
    for sample in samples:
        by_label[float(sample.label)].append(sample)
    train: list[PeptideSample] = []
    valid: list[PeptideSample] = []
    for label_samples in by_label.values():
        items = list(label_samples)
        rng.shuffle(items)
        n_valid = max(1, int(round(len(items) * valid_ratio))) if len(items) > 1 else 0
        valid.extend(items[:n_valid])
        train.extend(items[n_valid:])
    if not train or not valid:
        raise ValueError("Unable to split train/valid with non-empty partitions.")
    rng.shuffle(train)
    rng.shuffle(valid)
    return train, valid


class PeptideDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[PeptideSample],
        max_len: int,
        p0_dim: int,
        esm_dim: int,
        prior_builder=None,
        p0_features: dict[int, torch.Tensor] | None = None,
        esm_features: dict[int, torch.Tensor] | None = None,
        truncate_strategy: str = "head",
    ) -> None:
        if not samples:
            raise ValueError("PeptideDataset requires at least one sample.")
        sample_indices = [int(sample.sample_index) for sample in samples]
        if len(set(sample_indices)) != len(sample_indices):
            raise ValueError("PeptideDataset sample_index values must be unique.")
        self.samples = list(samples)
        self.max_len = max_len
        self.p0_dim = p0_dim
        self.esm_dim = esm_dim
        self.prior_builder = prior_builder
        self.p0_features = p0_features
        self.esm_features = esm_features
        self.truncate_strategy = truncate_strategy

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        sample = self.samples[idx]
        token_ids, valid_mask = tokenize_sequence(sample.sequence, self.max_len, self.truncate_strategy)
        p0 = self._get_p0(sample)
        esm_h = self._get_esm(sample)

        assert token_ids.shape == (self.max_len,)
        assert valid_mask.shape == (self.max_len,)
        assert p0.shape == (self.max_len, self.p0_dim)
        assert esm_h.shape == (self.max_len, self.esm_dim)
        assert token_ids.min() >= 0 and token_ids.max() <= 20
        assert np.all(token_ids[valid_mask == 0] == PAD_ID)
        assert np.allclose(p0[valid_mask == 0], 0.0)
        assert np.allclose(esm_h[valid_mask == 0], 0.0)

        return {
            "token_ids": torch.from_numpy(token_ids).long(),
            "valid_mask": torch.from_numpy(valid_mask).float(),
            "labels": torch.tensor(sample.label, dtype=torch.float32),
            "sample_indices": torch.tensor(sample.sample_index, dtype=torch.long),
            "p0": torch.as_tensor(p0, dtype=torch.float32),
            "esm_h": torch.as_tensor(esm_h, dtype=torch.float32),
            "sequences": sample.sequence,
            "source_indices": sample.source_index,
        }

    def _get_p0(self, sample: PeptideSample) -> np.ndarray:
        if self.p0_features is not None:
            value = self.p0_features[sample.sample_index]
            return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
        if self.prior_builder is None:
            return np.zeros((self.max_len, self.p0_dim), dtype=np.float32)
        return self.prior_builder.encode_sequence(sample.sequence)

    def _get_esm(self, sample: PeptideSample) -> np.ndarray:
        if self.esm_features is None:
            return np.zeros((self.max_len, self.esm_dim), dtype=np.float32)
        value = self.esm_features[sample.sample_index]
        return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)

    def p0_feature_map(self) -> dict[int, torch.Tensor]:
        return {
            sample.sample_index: torch.as_tensor(self._get_p0(sample), dtype=torch.float32)
            for sample in self.samples
        }


def collate_peptide_batch(batch: Iterable[dict[str, object]]) -> dict[str, object]:
    items = list(batch)
    return {
        "token_ids": torch.stack([item["token_ids"] for item in items]),
        "valid_mask": torch.stack([item["valid_mask"] for item in items]),
        "labels": torch.stack([item["labels"] for item in items]),
        "sample_indices": torch.stack([item["sample_indices"] for item in items]),
        "p0": torch.stack([item["p0"] for item in items]),
        "esm_h": torch.stack([item["esm_h"] for item in items]),
        "sequences": [item["sequences"] for item in items],
        "source_indices": [item["source_indices"] for item in items],
    }
