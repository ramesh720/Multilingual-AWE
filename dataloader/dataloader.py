import torch
from torch.utils.data import DataLoader

from dataloader.dataset import UniqueWordBatchSampler, collate_fn


def create_data_loader(
    dataset,
    batch_size,
    shuffle=True,
    num_workers=1,
    verification=True,
    drop_last=None,
):
    """
    Builds a DataLoader that guarantees every WORD within a batch is
    DIFFERENT (via UniqueWordBatchSampler, `batch_size` = N unique words),
    while each word contributes M same-word audio views (dataset.M),
    grouped by collate_fn into (N, M, ...) tensors — needed for both the
    CLAP loss (per-view, then averaged over M per Eq. 14) and the DWD
    loss (across the full N*M audio embeddings, per Eq. 10-13).

    `drop_last`:
      - None (default) -> follows `shuffle` (True for shuffled/training use,
        so OneCycleLR gets a stable steps_per_epoch; False otherwise).
      - Pass explicitly (e.g. drop_last=False for validation) if you want
        every word seen at least once, even in a smaller final batch.
    """
    if drop_last is None:
        drop_last = shuffle

    batch_sampler = UniqueWordBatchSampler(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
    )

    def _collate(batch):
        result = collate_fn(batch)
        if verification and result is not None:
            mels, anc_sequences, mel_lengths, word_labels, anc_lengths = result
            print("=" * 60)
            print("N (unique words):", mels.size(0))
            print("M (views/word)  :", mels.size(1))
            print("Unique words    :", len(torch.unique(word_labels)))
        return result

    return DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=_collate,
        num_workers=num_workers,
    )