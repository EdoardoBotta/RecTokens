from __future__ import annotations

from typing import Iterator, Protocol, runtime_checkable

import torch


@runtime_checkable
class ItemDataset(Protocol):
    """Structural protocol for item feature datasets.

    Any object implementing ``__len__``, ``__getitem__``, and ``iter_batches``
    satisfies this protocol without needing to inherit from it.  This keeps
    concrete dataset classes decoupled from the library.
    """

    def __len__(self) -> int:
        """Total number of items."""
        ...

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Return the feature vector for item ``idx``.  Shape: ``(D,)``."""
        ...

    def iter_batches(self, batch_size: int = 256) -> Iterator[torch.Tensor]:
        """Yield batches of item features.  Each batch has shape ``(B, D)``."""
        ...
