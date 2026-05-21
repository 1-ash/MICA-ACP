from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from dataprocess.dataset import AMINO_ACIDS, PeptideSample, clean_sequence


AAINDEX_MECHANISM_GROUPS = ("charge", "hydrophobic", "amphipathic", "structure")

_PROXY_VECTORS = {
    "charge": np.asarray([0.0, 1.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "hydrophobic": np.asarray([1.8, -4.5, -3.5, -3.5, 2.5, -3.5, -3.5, -0.4, -3.2, 4.5, 3.8, -3.9, 1.9, 2.8, -1.6, -0.8, -0.7, -0.9, -1.3, 4.2]),
    "amphipathic": np.asarray([0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0]),
    "structure": np.asarray([1.45, 0.79, 0.73, 0.98, 0.77, 1.17, 1.53, 0.53, 1.24, 1.00, 1.34, 1.07, 1.20, 1.12, 0.59, 0.79, 0.82, 1.14, 0.61, 1.14]),
}


@dataclass(frozen=True)
class AAIndexSelectionResult:
    indices: list[int]
    codes: list[str]
    groups: dict[str, list[str]]
    selected_frequency: dict[str, float]
    average_rank: dict[str, float | None]


def _standardize(values: np.ndarray) -> np.ndarray | None:
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        return None
    std = float(values.std())
    if std < 1e-8:
        return None
    return (values - float(values.mean())) / std


def _abs_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size != b.size or a.size < 2:
        return 0.0
    a_std = float(a.std())
    b_std = float(b.std())
    if a_std < 1e-8 or b_std < 1e-8:
        return 0.0
    return float(abs(np.corrcoef(a, b)[0, 1]))


def _load_clean_scales(path: str | Path) -> tuple[list[int], list[str], np.ndarray]:
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if not fieldnames:
            raise ValueError(f"AAindex CSV has no header: {path}")
        code_field = fieldnames[0]
        missing_columns = set(AMINO_ACIDS) - set(fieldnames)
        if missing_columns:
            raise ValueError(f"AAindex CSV is missing amino-acid columns: {sorted(missing_columns)}")

        row_indices: list[int] = []
        codes: list[str] = []
        vectors: list[np.ndarray] = []
        for row_index, row in enumerate(reader):
            try:
                raw = np.asarray([float(row[aa]) for aa in AMINO_ACIDS], dtype=np.float64)
            except (TypeError, ValueError):
                continue
            values = _standardize(raw)
            if values is None:
                continue
            row_indices.append(row_index)
            codes.append(str(row.get(code_field, row_index)).strip() or str(row_index))
            vectors.append(values)

    if not vectors:
        raise ValueError(f"No usable AAindex scales remain after cleaning: {path}")
    return row_indices, codes, np.stack(vectors, axis=0)


def _sequence_stat_features(samples: Sequence[PeptideSample], scale_values: np.ndarray, max_len: int) -> np.ndarray:
    aa_to_col = {aa: idx for idx, aa in enumerate(AMINO_ACIDS)}
    features = np.zeros((len(samples), scale_values.shape[0], 4), dtype=np.float64)
    for sample_idx, sample in enumerate(samples):
        seq = clean_sequence(sample.sequence, max_len)
        if seq is None:
            raise ValueError(f"Invalid sample sequence during AAindex selection: {sample.sequence!r}")
        cols = [aa_to_col[aa] for aa in seq]
        values = scale_values[:, cols]
        features[sample_idx, :, 0] = values.mean(axis=1)
        features[sample_idx, :, 1] = values.std(axis=1)
        features[sample_idx, :, 2] = values.max(axis=1)
        features[sample_idx, :, 3] = values.min(axis=1)
    return features


def _scale_relevance(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    relevance = np.zeros(features.shape[1], dtype=np.float64)
    for scale_idx in range(features.shape[1]):
        relevance[scale_idx] = max(_abs_corr(features[:, scale_idx, stat_idx], labels) for stat_idx in range(4))
    return relevance


def _drop_correlated(scale_values: np.ndarray, relevance: np.ndarray, threshold: float) -> np.ndarray:
    keep: list[int] = []
    for idx in np.argsort(-relevance):
        if all(_abs_corr(scale_values[idx], scale_values[kept]) <= threshold for kept in keep):
            keep.append(int(idx))
    return np.asarray(keep, dtype=np.int64)


def _assign_groups(scale_values: np.ndarray) -> np.ndarray:
    proxies = {name: _standardize(values) for name, values in _PROXY_VECTORS.items()}
    groups: list[str] = []
    for values in scale_values:
        best_group = max(AAINDEX_MECHANISM_GROUPS, key=lambda name: _abs_corr(values, proxies[name]))
        groups.append(best_group)
    return np.asarray(groups, dtype=object)


def _subsample_rows(labels: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    rows: list[np.ndarray] = []
    for label in (0.0, 1.0):
        label_rows = np.where(labels == label)[0]
        if label_rows.size == 0:
            continue
        take = max(1, int(round(label_rows.size * fraction)))
        take = min(take, int(label_rows.size))
        rows.append(rng.choice(label_rows, size=take, replace=False))
    if not rows:
        raise ValueError("Cannot subsample AAindex rows without labels.")
    selected = np.concatenate(rows)
    rng.shuffle(selected)
    return selected


def _group_quotas(groups: np.ndarray, relevance: np.ndarray, target_k: int) -> dict[str, int]:
    quotas = {group: target_k // len(AAINDEX_MECHANISM_GROUPS) for group in AAINDEX_MECHANISM_GROUPS}
    remainder = target_k - sum(quotas.values())
    group_priority = sorted(
        AAINDEX_MECHANISM_GROUPS,
        key=lambda group: float(relevance[groups == group].max()) if np.any(groups == group) else -1.0,
        reverse=True,
    )
    for group in group_priority[:remainder]:
        quotas[group] += 1
    return quotas


def _mrmr_select(candidates: np.ndarray, relevance: np.ndarray, scale_values: np.ndarray, k: int) -> list[int]:
    remaining = [int(idx) for idx in candidates]
    selected: list[int] = []
    while remaining and len(selected) < k:
        best_idx = max(
            remaining,
            key=lambda idx: (
                float(relevance[idx])
                - (float(np.mean([_abs_corr(scale_values[idx], scale_values[chosen]) for chosen in selected])) if selected else 0.0),
                float(relevance[idx]),
                -idx,
            ),
        )
        selected.append(best_idx)
        remaining.remove(best_idx)
    return selected


def select_aaindex_mrmr(
    samples: Sequence[PeptideSample],
    aaindex_csv_path: str | Path,
    max_len: int,
    target_k: int,
    *,
    repeats: int = 20,
    sample_fraction: float = 0.8,
    redundancy_threshold: float = 0.95,
    seed: int = 42,
) -> AAIndexSelectionResult:
    if target_k <= 0:
        return AAIndexSelectionResult([], [], {group: [] for group in AAINDEX_MECHANISM_GROUPS}, {}, {})
    if not samples:
        raise ValueError("AAindex mRMR selection requires training samples.")
    labels = np.asarray([float(sample.label) for sample in samples], dtype=np.float64)
    if len(set(labels.tolist())) < 2:
        raise ValueError("AAindex mRMR selection requires both positive and negative samples.")

    row_indices, codes, scale_values = _load_clean_scales(aaindex_csv_path)
    full_features = _sequence_stat_features(samples, scale_values, max_len)
    global_relevance = _scale_relevance(full_features, labels)
    keep = _drop_correlated(scale_values, global_relevance, redundancy_threshold)
    if keep.size < target_k:
        raise ValueError(
            f"Only {keep.size} AAindex scales remain after cleaning/redundancy filtering; "
            f"cannot select target_k={target_k}."
        )

    row_indices = [row_indices[int(i)] for i in keep]
    codes = [codes[int(i)] for i in keep]
    scale_values = scale_values[keep]
    features = full_features[:, keep, :]
    global_relevance = global_relevance[keep]
    groups = _assign_groups(scale_values)
    quotas = _group_quotas(groups, global_relevance, target_k)

    rng = np.random.default_rng(seed)
    repeats = max(1, int(repeats))
    sample_fraction = min(max(float(sample_fraction), 0.1), 1.0)
    frequency = np.zeros(scale_values.shape[0], dtype=np.float64)
    rank_sum = np.zeros(scale_values.shape[0], dtype=np.float64)

    for _ in range(repeats):
        rows = _subsample_rows(labels, sample_fraction, rng)
        relevance = _scale_relevance(features[rows], labels[rows])
        for group in AAINDEX_MECHANISM_GROUPS:
            candidates = np.where(groups == group)[0]
            selected = _mrmr_select(candidates, relevance, scale_values, quotas[group])
            for rank, idx in enumerate(selected, start=1):
                frequency[idx] += 1.0
                rank_sum[idx] += float(rank)

    avg_rank = np.full(scale_values.shape[0], np.inf, dtype=np.float64)
    seen = frequency > 0
    avg_rank[seen] = rank_sum[seen] / frequency[seen]

    final: list[int] = []
    for group in AAINDEX_MECHANISM_GROUPS:
        candidates = [int(idx) for idx in np.where(groups == group)[0] if int(idx) not in final]
        candidates.sort(key=lambda idx: (-frequency[idx], avg_rank[idx], -global_relevance[idx], idx))
        final.extend(candidates[: quotas[group]])
    if len(final) < target_k:
        remaining = [idx for idx in range(scale_values.shape[0]) if idx not in set(final)]
        remaining.sort(key=lambda idx: (-frequency[idx], avg_rank[idx], -global_relevance[idx], idx))
        final.extend(remaining[: target_k - len(final)])
    final = final[:target_k]

    grouped_codes = {
        group: [codes[idx] for idx in final if groups[idx] == group]
        for group in AAINDEX_MECHANISM_GROUPS
    }
    return AAIndexSelectionResult(
        indices=[int(row_indices[idx]) for idx in final],
        codes=[codes[idx] for idx in final],
        groups=grouped_codes,
        selected_frequency={codes[idx]: float(frequency[idx] / repeats) for idx in final},
        average_rank={codes[idx]: (None if not np.isfinite(avg_rank[idx]) else float(avg_rank[idx])) for idx in final},
    )
