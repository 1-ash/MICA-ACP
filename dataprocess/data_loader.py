from __future__ import annotations

from torch.utils.data import DataLoader

from dataprocess.dataset import PeptideDataset, collate_peptide_batch


def create_data_loader(
    dataset: PeptideDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_peptide_batch,
        drop_last=False,
    )
