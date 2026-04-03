import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from collections import Counter

class SequenceDataset(Dataset):
    def __init__(self, X, y, seq_len, df_idx=None, custom_targets=None, safe=True,
                 trade_outcomes=None, seq_idx_filter=None):
        """
        seq_idx_filter : list[int] | None
            If provided, only these sequence-start indices are included in the
            dataset (anchor-time / session gating).  Each value ``i`` in the
            list must satisfy ``0 <= i < len(X) - seq_len``; the prediction
            anchor is at row ``i + seq_len - 1``.
            When ``None`` (default) all valid sequences are included.
        """
        self.X = X
        self.y = y
        self.seq_len = seq_len
        self.num_samples = len(self.X) - self.seq_len
        self.trade_outcomes = trade_outcomes  # NEW: Trade outcomes (can be 1D or 2D for buy/sell)
        if df_idx is None:
            self.df_idx = list(range(len(self.y)))
        else:
            self.df_idx = df_idx
        
        # Build sequences and targets
        self.y_seqs = np.array([self.y[i+self.seq_len - 1] for i in range(self.num_samples)])
        self.indices = np.arange(len(self.y_seqs))

        # Anchor-time / session gating: restrict to caller-supplied sequence indices
        if seq_idx_filter is not None:
            self.indices = np.array(seq_idx_filter, dtype=np.int64)

        if custom_targets is not None:
            # Initialize with original indices
            new_indices = []
            class_counts = Counter(self.y_seqs)
            
            # For each class
            for cls in sorted(class_counts.keys()):
                cls_idx = np.where(self.y_seqs == cls)[0]
                target_count = custom_targets.get(cls, len(cls_idx))
                
                if target_count > len(cls_idx):
                    # Oversample
                    resampled = np.random.choice(cls_idx, target_count, replace=True)
                else:
                    # Keep original or undersample
                    resampled = np.random.choice(cls_idx, target_count, replace=False)
                new_indices.extend(resampled)
            
            # Shuffle the indices
            self.indices = np.array(new_indices)
            np.random.shuffle(self.indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        seq_idx = self.indices[idx]
        X_seq = self.X[seq_idx:seq_idx+self.seq_len]
        y_seq = self.y[seq_idx+self.seq_len - 1]
        df_idx = self.df_idx[seq_idx+self.seq_len - 1]
        
        # Include trade_outcome if available
        if self.trade_outcomes is not None:
            outcome = self.trade_outcomes[seq_idx+self.seq_len - 1]
            # outcome can be 1D (single value) or 2D (buy/sell pair)
            # Convert to tensor, preserving dimensionality
            if isinstance(outcome, np.ndarray) and outcome.ndim > 0:
                outcome_tensor = torch.tensor(outcome, dtype=torch.float32)
            else:
                outcome_tensor = torch.tensor(outcome, dtype=torch.float32)
            return torch.tensor(X_seq, dtype=torch.float32), torch.tensor(y_seq, dtype=torch.long), outcome_tensor
        else:
            return torch.tensor(X_seq, dtype=torch.float32), torch.tensor(y_seq, dtype=torch.long), df_idx


class IndexedDataLoader:
    """A thin wrapper around `torch.utils.data.DataLoader` that records the
    original `SequenceDataset` indices for each yielded batch.

    Behaviour:
    - Yields exactly the same batches as a normal `DataLoader` (so existing
      training loops don't need changes).
    - During iteration it stores the mapped original indices (from
      `SequenceDataset.indices`) for each batch in `batch_indices_history` and
      makes the most-recent batch available via `last_batch_indices`.

    Notes:
    - The wrapper expects the underlying dataset to expose an `indices`
      attribute (as `SequenceDataset` does). If not available, the raw dataset
      indices from the sampler are stored instead.
    """

    def __init__(self, *dataloader_args, **dataloader_kwargs):
        # Construct an internal DataLoader with the provided args/kwargs
        self.loader = DataLoader(*dataloader_args, **dataloader_kwargs)

        # Determine dataset reference for index mapping
        self.dataset = None
        if len(dataloader_args) > 0:
            # DataLoader(dataset, ...)
            self.dataset = dataloader_args[0]
        else:
            self.dataset = dataloader_kwargs.get("dataset")

        self.batch_indices_history = []
        self.last_batch_indices = None

    def __iter__(self):
        # Reset history at the start of each epoch/iteration
        self.batch_indices_history = []
        self.last_batch_indices = None

        batch_idx_iter = iter(self.loader.batch_sampler)
        data_iter = iter(self.loader)

        for batch_indices in batch_idx_iter:
            # Retrieve the actual batch from the underlying DataLoader
            batch = next(data_iter)

            # Map sampler dataset indices to original SequenceDataset indices
            try:
                mapped = [int(self.dataset.indices[i]) for i in batch_indices]
            except Exception:
                # Fall back to raw sampler indices if mapping unavailable
                mapped = [int(i) for i in batch_indices]

            self.last_batch_indices = mapped
            self.batch_indices_history.append(mapped)

            # Extract data from batch (3-tuple from SequenceDataset)
            if len(batch) == 3:
                X, y, third = batch  # third can be df_idx or trade_outcomes
                yield X, y, third
            else:
                yield batch

    def __len__(self):
        return len(self.loader)

    def __getattr__(self, name):
        # Delegate attribute access to the underlying DataLoader
        return getattr(self.loader, name)