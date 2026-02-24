from __future__ import annotations

from typing import Iterator

import numpy as np
import torch


class NumpyDataset:
    """An :class:`~rectokens.core.dataset.ItemDataset` backed by a NumPy array.

    Args:
        embeddings: Array of shape ``(N, D)``.  Converted to ``float32``.
    """

    def __init__(self, embeddings: np.ndarray) -> None:
        self._data = torch.from_numpy(embeddings.astype(np.float32))

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self._data[idx]

    def iter_batches(self, batch_size: int = 256) -> Iterator[torch.Tensor]:
        for start in range(0, len(self._data), batch_size):
            yield self._data[start : start + batch_size]


class TensorDataset:
    """An :class:`~rectokens.core.dataset.ItemDataset` backed by a PyTorch tensor.

    Args:
        embeddings: Tensor of shape ``(N, D)``.  Cast to ``float32``.
    """

    def __init__(self, embeddings: torch.Tensor) -> None:
        self._data = embeddings.float()

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self._data[idx]

    def iter_batches(self, batch_size: int = 256) -> Iterator[torch.Tensor]:
        for start in range(0, len(self._data), batch_size):
            yield self._data[start : start + batch_size]
