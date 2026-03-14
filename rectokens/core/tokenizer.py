from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass
class TokenSequence:
    """A batch of discrete token sequences produced by a :class:`Tokenizer`.

    Attributes:
        codes: Long tensor of shape ``(B, num_levels)`` for a batch of items,
               or ``(num_levels,)`` for a single item.
    """

    codes: torch.Tensor  # (..., num_levels) long

    @property
    def num_levels(self) -> int:
        """Number of token levels (codebook depth)."""
        return self.codes.shape[-1]

    @property
    def batch_size(self) -> int | None:
        """Batch size, or ``None`` when a single item was encoded."""
        return self.codes.shape[0] if self.codes.ndim == 2 else None

    def to_tuple_ids(self) -> list[tuple[int, ...]]:
        """Convert to a list of ``(c_0, c_1, ..., c_L)`` tuples, one per item.

        Useful as dictionary keys or composite item identifiers.
        """
        codes = self.codes if self.codes.ndim == 2 else self.codes.unsqueeze(0)
        return [tuple(row.tolist()) for row in codes]

    def to_flat_ids(self, base: int | None = None) -> torch.Tensor:
        """Flatten multi-level codes into a single integer per item.

        Interprets the tuple ``(c_0, c_1, ..., c_{L-1})`` as a mixed-radix
        number in base ``base`` (defaults to ``max(codes) + 1``).  Useful
        when a single unique integer ID is needed per item.

        Args:
            base: Radix for each level.  Defaults to the observed maximum
                  code value plus one.

        Returns:
            Long tensor of shape ``(B,)`` (or scalar for single items).
        """
        codes = self.codes if self.codes.ndim == 2 else self.codes.unsqueeze(0)
        if base is None:
            base = int(codes.max().item()) + 1
        multipliers = torch.tensor(
            [base**i for i in range(self.num_levels - 1, -1, -1)],
            dtype=torch.long,
            device=codes.device,
        )
        flat = (codes * multipliers).sum(dim=-1)
        return flat if self.codes.ndim == 2 else flat.squeeze(0)


class Tokenizer(ABC):
    """Abstract base class for item tokenizers.

    A tokenizer converts item feature vectors (embeddings) into sequences
    of discrete token IDs.  The three-step lifecycle is:

    1. **Fit** — learn codebooks from a dataset (``fit``).
    2. **Encode** — map features to :class:`TokenSequence` (``encode``).
    3. **Decode** — reconstruct approximate features from codes (``decode``).

    Subclasses that use learnable parameters should also inherit from
    ``torch.nn.Module``.
    """

    @abstractmethod
    def fit_step(self, batch: torch.Tensor) -> Tokenizer:
        """Update the tokenizer with a single batch of item features.

        Args:
            batch: Float tensor of shape ``(B, D)``.

        Returns:
            ``self``, for method chaining.
        """
        ...

    @abstractmethod
    def encode(self, features: torch.Tensor) -> TokenSequence:
        """Encode item features to discrete token sequences.

        Args:
            features: Float tensor of shape ``(B, D)`` or ``(D,)`` for a
                      single item.

        Returns:
            :class:`TokenSequence` with codes of shape ``(B, L)`` or ``(L,)``.
        """
        ...

    @abstractmethod
    def decode(self, tokens: TokenSequence) -> torch.Tensor:
        """Reconstruct approximate item features from token codes.

        Args:
            tokens: :class:`TokenSequence` produced by :meth:`encode`.

        Returns:
            Float tensor of shape ``(B, D)`` or ``(D,)`` matching the
            original input shape.
        """
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """Serialize the tokenizer state to ``path``."""
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> Tokenizer:
        """Deserialize and return a tokenizer from ``path``."""
        ...
