from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch

from .codebook import Codebook


@dataclass
class QuantizerOutput:
    """Output of a single-level quantizer."""

    codes: torch.Tensor  # (B,) long — discrete code indices
    quantized: torch.Tensor  # (B, D) float — selected codebook entries
    residuals: torch.Tensor  # (B, D) float — input minus quantized
    commitment_loss: torch.Tensor | None = None  # scalar; set during VQ training


@dataclass
class ResidualQuantizerOutput:
    """Output of a multi-level residual quantizer."""

    codes: torch.Tensor  # (B, num_levels) long
    quantized: torch.Tensor  # (B, D) float — sum of all levels
    level_outputs: list[QuantizerOutput] = field(default_factory=list)

    @property
    def commitment_loss(self) -> torch.Tensor:
        """Sum of commitment losses across all levels (0 if none set)."""
        losses = [
            o.commitment_loss
            for o in self.level_outputs
            if o.commitment_loss is not None
        ]
        if not losses:
            return torch.tensor(0.0)
        return torch.stack(losses).sum()


class Quantizer(ABC):
    """Abstract single-level quantizer.

    Maps continuous vectors to their nearest entry in a :class:`~rectokens.core.codebook.Codebook`
    and returns the code index, the quantized vector, and the residual.

    Subclasses are *not* required to be ``nn.Module``s — clustering-based
    quantizers (e.g. K-means) have no learnable parameters.  Neural
    quantizers (e.g. VQ) should multiply-inherit from both ``Quantizer``
    and ``nn.Module``.
    """

    @property
    @abstractmethod
    def codebook(self) -> Codebook:
        """The codebook backing this quantizer."""
        ...

    @abstractmethod
    def quantize(self, x: torch.Tensor) -> QuantizerOutput:
        """Quantize input vectors.

        Args:
            x: Float tensor of shape ``(B, D)``.

        Returns:
            :class:`QuantizerOutput` with per-item codes, quantized vectors,
            and residuals.
        """
        ...

    def fit_step(self, batch: torch.Tensor) -> Quantizer:
        """Update the quantizer with a single batch of training data.

        The default implementation is a no-op (suitable for learned
        quantizers trained end-to-end).  Override in clustering-based
        subclasses.

        Args:
            batch: Float tensor of shape ``(B, D)``.

        Returns:
            ``self``, for method chaining.
        """
        return self
