from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from config import DEFAULT_AAINDEX_INDICES
from dataprocess.dataset import AMINO_ACIDS, clean_sequence


SELECTED_AAINDEX_INDICES = np.array(DEFAULT_AAINDEX_INDICES, dtype=np.int64)


class PriorBuilder:
    def __init__(
        self,
        aaindex_csv_path: str | Path,
        blosum_csv_path: str | Path,
        max_len: int,
        aaindex_indices: list[int] | tuple[int, ...] | np.ndarray | None = None,
        use_aac20_prior: bool = True,
        use_blosum_prior: bool = True,
        truncate_strategy: str = "head",
    ) -> None:
        self.max_len = max_len
        self.use_aac20_prior = bool(use_aac20_prior)
        self.use_blosum_prior = bool(use_blosum_prior)
        self.truncate_strategy = truncate_strategy
        self.aaindex_indices = np.asarray(
            SELECTED_AAINDEX_INDICES if aaindex_indices is None else aaindex_indices,
            dtype=np.int64,
        )
        if self.aaindex_indices.ndim != 1:
            raise ValueError("aaindex_indices must be a one-dimensional list of row indices.")
        if (self.aaindex_indices < 0).any():
            raise ValueError("aaindex_indices must be non-negative row indices.")
        self.aaindex_dim = int(self.aaindex_indices.size)
        self.blosum_dim = 20 if self.use_blosum_prior else 0
        self.aac_dim = 20 if self.use_aac20_prior else 0
        self.p0_dim = self.blosum_dim + self.aaindex_dim + self.aac_dim
        self.blosum = (
            self._load_blosum(blosum_csv_path)
            if self.use_blosum_prior
            else {aa: np.zeros(0, dtype=np.float32) for aa in AMINO_ACIDS}
        )
        self.aaindex = self._load_aaindex(aaindex_csv_path)
        self.residue_features = {
            aa: np.concatenate([self.blosum[aa], self.aaindex[aa]], axis=0).astype(np.float32)
            for aa in AMINO_ACIDS
        }
        residue_dim = self.blosum_dim + self.aaindex_dim
        for aa, vector in self.residue_features.items():
            if vector.shape != (residue_dim,):
                raise ValueError(f"Residue prior for {aa} must have dimension {residue_dim}, got {vector.shape}")

    def encode_sequence(self, sequence: str) -> np.ndarray:
        seq = clean_sequence(sequence, self.max_len, self.truncate_strategy)
        if seq is None:
            raise ValueError(f"Invalid peptide sequence for P0 construction: {sequence!r}")
        p0 = np.zeros((self.max_len, self.p0_dim), dtype=np.float32)
        aac20 = self._aac20(seq) if self.use_aac20_prior else None
        for pos, aa in enumerate(seq):
            if aac20 is None:
                p0[pos] = self.residue_features[aa]
            else:
                p0[pos] = np.concatenate([self.residue_features[aa], aac20], axis=0)
        return p0

    def _aac20(self, sequence: str) -> np.ndarray:
        counts = np.zeros(20, dtype=np.float32)
        for aa in sequence:
            counts[AMINO_ACIDS.index(aa)] += 1.0
        denom = max(float(len(sequence)), 1.0)
        return counts / denom

    def _load_blosum(self, path: str | Path) -> dict[str, np.ndarray]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"BLOSUM CSV file does not exist: {path}")
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or [])
            missing_columns = set(AMINO_ACIDS) - columns
            if missing_columns:
                raise ValueError(f"BLOSUM CSV is missing amino-acid columns: {sorted(missing_columns)}")
            table: dict[str, np.ndarray] = {}
            for row in reader:
                row_name = (row.get("") or row.get("AA") or row.get("amino_acid") or "").strip()
                if row_name in AMINO_ACIDS:
                    table[row_name] = np.asarray([float(row[aa]) for aa in AMINO_ACIDS], dtype=np.float32)
        missing_rows = set(AMINO_ACIDS) - set(table)
        if missing_rows:
            raise ValueError(f"BLOSUM CSV is missing amino-acid rows: {sorted(missing_rows)}")
        for aa, vector in table.items():
            if vector.shape != (20,):
                raise ValueError(f"BLOSUM vector for {aa} must have dimension 20, got {vector.shape}")
        return table

    def _load_aaindex(self, path: str | Path) -> dict[str, np.ndarray]:
        if self.aaindex_dim == 0:
            return {aa: np.zeros(0, dtype=np.float32) for aa in AMINO_ACIDS}
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"AAindex CSV file does not exist: {path}")
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"AAindex CSV has no data rows: {path}")
        if self.aaindex_indices.max() >= len(rows):
            raise ValueError(
                "AAindex CSV does not contain enough rows for selected indices: "
                f"need index {int(self.aaindex_indices.max())}, got {len(rows)} rows"
            )
        missing_columns = set(AMINO_ACIDS) - set(rows[0].keys())
        if missing_columns:
            raise ValueError(f"AAindex CSV is missing amino-acid columns: {sorted(missing_columns)}")
        selected_rows = [rows[int(i)] for i in self.aaindex_indices]
        row_values = np.asarray(
            [[float(row[aa]) for aa in AMINO_ACIDS] for row in selected_rows],
            dtype=np.float32,
        )
        if not np.isfinite(row_values).all():
            raise ValueError("Selected AAindex rows contain non-finite values.")
        row_mean = row_values.mean(axis=1, keepdims=True)
        row_std = row_values.std(axis=1, keepdims=True)
        if (row_std < 1e-8).any():
            bad = [int(self.aaindex_indices[i]) for i in np.where((row_std[:, 0] < 1e-8))[0]]
            raise ValueError(f"Selected AAindex rows are constant or near-constant: {bad}")
        row_values = (row_values - row_mean) / row_std
        table: dict[str, np.ndarray] = {}
        for aa_idx, aa in enumerate(AMINO_ACIDS):
            values = row_values[:, aa_idx].astype(np.float32)
            if values.shape != (self.aaindex_dim,):
                raise ValueError(
                    f"AAindex vector for {aa} must have dimension {self.aaindex_dim}, got {values.shape}"
                )
            if not np.isfinite(values).all():
                raise ValueError(f"AAindex vector for {aa} contains non-finite values.")
            table[aa] = values
        return table
