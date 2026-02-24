from __future__ import annotations

from typing import Generator, TYPE_CHECKING

import torch
import torch.nn as nn

from rectokens.core.quantizer import Quantizer, ResidualQuantizerOutput

if TYPE_CHECKING:
    from rectokens.core.dataset import ItemDataset


class ResidualQuantizer(nn.Module):
    """Chains ``N`` quantizers to produce ``N`` codes per input vector.

    At each level ``l``, the quantizer for that level encodes the *residual*
    from the previous level:

    .. code-block:: text

        r_0 = x
        code_l, q_l = quantizer_l.quantize(r_{l-1})
        r_l = r_{l-1} - q_l

    The final reconstruction is the sum of all ``q_l``.  This hierarchical
    decomposition lets a small codebook at each level represent a large
    effective vocabulary: ``K`` codes per level → ``K^L`` unique items.

    Fitting is done greedily level by level: each quantizer is fit on the
    residual left by all previously fit levels.

    Args:
        quantizers: Ordered list of :class:`~rectokens.core.quantizer.Quantizer`
                    objects, one per level.  Quantizers that are also
                    ``nn.Module``s are registered in a ``ModuleList`` so their
                    parameters appear in ``state_dict``; plain quantizers are
                    held in a regular list.
    """

    def __init__(self, quantizers: list[Quantizer]) -> None:
        super().__init__()
        if not quantizers:
            raise ValueError("ResidualQuantizer requires at least one quantizer.")

        # Separate Module quantizers (register them) from plain ones
        module_quantizers = [q for q in quantizers if isinstance(q, nn.Module)]
        self._module_levels = nn.ModuleList(module_quantizers)
        self._levels: list[Quantizer] = quantizers

    @property
    def levels(self) -> list[Quantizer]:
        """Ordered list of per-level quantizers."""
        return self._levels

    @property
    def num_levels(self) -> int:
        """Number of quantization levels."""
        return len(self._levels)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, dataset: ItemDataset, batch_size: int = 256) -> ResidualQuantizer:
        """Fit each level sequentially on the residual of all previous levels.

        The dataset is iterated **once per level**.  At each level ``l``, a
        lazy generator applies the already-fitted quantizers ``0..l-1`` to
        every batch from the dataset and yields the resulting residuals.
        Only one batch is held in memory at a time, so the dataset never
        needs to fit in memory.

        Args:
            dataset: Any object satisfying the
                     :class:`~rectokens.core.dataset.ItemDataset` protocol.
            batch_size: Number of items per batch passed to each quantizer.

        Returns:
            ``self``.
        """
        for level_idx, quantizer in enumerate(self._levels):
            quantizer.fit(self._residual_batches(dataset, batch_size, level_idx))
        return self

    def _residual_batches(
        self,
        dataset: ItemDataset,
        batch_size: int,
        level_idx: int,
    ) -> Generator[torch.Tensor, None, None]:
        """Yield residual batches for fitting level ``level_idx``.

        Each batch is produced by taking raw item features from ``dataset``
        and computing the residual after applying levels ``0..level_idx-1``.
        """
        with torch.no_grad():
            for batch in dataset.iter_batches(batch_size):
                residual = batch.float()
                for q in self._levels[:level_idx]:
                    residual = q.quantize(residual).residuals
                yield residual

    # ------------------------------------------------------------------
    # Quantize
    # ------------------------------------------------------------------

    def quantize(self, x: torch.Tensor) -> ResidualQuantizerOutput:
        """Quantize ``x`` across all levels, encoding residuals.

        Args:
            x: Float tensor of shape ``(B, D)``.

        Returns:
            :class:`~rectokens.core.quantizer.ResidualQuantizerOutput` with
            stacked codes of shape ``(B, num_levels)``.
        """
        all_codes: list[torch.Tensor] = []
        level_outputs = []
        total_quantized = torch.zeros_like(x)
        residual = x

        for quantizer in self._levels:
            out = quantizer.quantize(residual)
            all_codes.append(out.codes)       # (B,)
            level_outputs.append(out)
            total_quantized = total_quantized + out.quantized
            residual = out.residuals

        return ResidualQuantizerOutput(
            codes=torch.stack(all_codes, dim=1),  # (B, num_levels)
            quantized=total_quantized,
            level_outputs=level_outputs,
        )

    # forward is an alias so ResidualQuantizer can be used directly as a Module
    def forward(self, x: torch.Tensor) -> ResidualQuantizerOutput:
        return self.quantize(x)
