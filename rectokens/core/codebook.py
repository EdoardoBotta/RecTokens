from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class SearchResult:
    """Nearest-neighbor search result from a codebook query."""

    codes: torch.Tensor      # (B,) long — index of nearest code per query


class Codebook(nn.Module):
    """Abstract base class for discrete codebooks.

    A codebook stores a table of ``size`` embedding vectors of dimension
    ``dim``.  Subclasses must implement nearest-neighbor search and entry
    lookup; they may optionally support in-place updates for EMA-style
    training or K-means centroid replacement.

    Inherits from ``nn.Module`` so that learnable codebooks (e.g. in RQVAE)
    participate naturally in PyTorch's parameter/state-dict machinery.
    Non-learned codebooks (e.g. K-means) simply register their table as a
    buffer instead of a parameter.
    """

    @property
    @abstractmethod
    def size(self) -> int:
        """Number of discrete codes in the codebook."""
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        """Embedding dimension of each code."""
        ...

    @abstractmethod
    def lookup(self, codes: torch.Tensor) -> torch.Tensor:
        """Return embeddings for the given code indices.

        Args:
            codes: Long tensor of shape ``(B,)``.

        Returns:
            Float tensor of shape ``(B, D)``.
        """
        ...

    @abstractmethod
    def find_nearest(self, query: torch.Tensor) -> SearchResult:
        """Find the nearest codebook entry for each query vector.

        Args:
            query: Float tensor of shape ``(B, D)``.

        Returns:
            :class:`SearchResult` with ``codes`` ``(B,)`` and
            ``distances`` ``(B,)``.
        """
        ...

    @abstractmethod
    def update(self, codes: torch.Tensor, embeddings: torch.Tensor) -> None:
        """Overwrite specific codebook entries.

        Used for K-means centroid replacement and EMA codebook updates.

        Args:
            codes: Long tensor of shape ``(K,)`` — indices to update.
            embeddings: Float tensor of shape ``(K, D)`` — new values.
        """
        ...
