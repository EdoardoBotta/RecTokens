from __future__ import annotations

import torch
import torch.nn as nn

from rectokens.core.codebook import Codebook, SearchResult


class EuclideanCodebook(Codebook):
    """A flat codebook that uses exhaustive L2 nearest-neighbor search.

    The code table is stored as a non-learnable buffer by default, making it
    suitable for K-means centroids.  Pass ``learnable=True`` to store it as an
    ``nn.Parameter`` instead (useful for VQ-style end-to-end training).

    Nearest-neighbor search is fully vectorized:

    .. math::
        \\|q - e_k\\|^2 = \\|q\\|^2 + \\|e_k\\|^2 - 2\\, q \\cdot e_k^\\top

    This avoids Python loops and runs efficiently on CPU or GPU.

    For very large codebooks (>100k codes) consider swapping in a
    FAISS-backed codebook that performs approximate nearest-neighbor search.

    Args:
        size: Number of codes.
        dim: Embedding dimension.
        learnable: If ``True``, store the code table as a trainable parameter
                   rather than a buffer.
    """

    def __init__(self, size: int, dim: int, *, learnable: bool = False) -> None:
        super().__init__()
        self._size = size
        self._dim = dim
        embeddings = torch.zeros(size, dim)
        if learnable:
            self.embeddings = nn.Parameter(embeddings)
        else:
            self.register_buffer("embeddings", embeddings)

    # ------------------------------------------------------------------
    # Class-method constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_tensor(
        cls, embeddings: torch.Tensor, *, learnable: bool = False
    ) -> EuclideanCodebook:
        """Build a codebook pre-populated from an existing tensor."""
        cb = cls(size=embeddings.shape[0], dim=embeddings.shape[1], learnable=learnable)
        with torch.no_grad():
            cb.embeddings.copy_(embeddings.float())
        return cb

    # ------------------------------------------------------------------
    # Codebook interface
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        return self._size

    @property
    def dim(self) -> int:
        return self._dim

    def lookup(self, codes: torch.Tensor) -> torch.Tensor:
        """Return embeddings for the given code indices.

        Args:
            codes: Long tensor of shape ``(B,)``.

        Returns:
            Float tensor of shape ``(B, D)``.
        """
        return self.embeddings[codes]

    def find_nearest(self, query: torch.Tensor) -> SearchResult:
        """Find the nearest codebook entry for each query vector.

        Uses the identity ``‖q - e‖² = ‖q‖² + ‖e‖² - 2qeᵀ`` for a
        single-pass, loop-free computation.

        Args:
            query: Float tensor of shape ``(B, D)``.

        Returns:
            :class:`~rectokens.core.codebook.SearchResult` with ``codes``
            ``(B,)`` and ``distances`` ``(B,)``.
        """
        # query: (B, D),  embeddings: (K, D)
        q_sq = (query ** 2).sum(dim=1, keepdim=True)    # (B, 1)
        e_sq = (self.embeddings ** 2).sum(dim=1)         # (K,)
        dots = query @ self.embeddings.t()               # (B, K)
        dist2 = q_sq + e_sq - 2.0 * dots                # (B, K)  [squared L2]
        codes = dist2.argmin(dim=1)                      # (B,)
        distances = dist2[torch.arange(len(codes), device=codes.device), codes]
        return SearchResult(codes=codes, distances=distances)

    def update(self, codes: torch.Tensor, embeddings: torch.Tensor) -> None:
        """Overwrite specific entries in the codebook.

        Args:
            codes: Long tensor of shape ``(K,)`` — indices to overwrite.
            embeddings: Float tensor of shape ``(K, D)`` — new values.
        """
        with torch.no_grad():
            self.embeddings[codes] = embeddings.float()
