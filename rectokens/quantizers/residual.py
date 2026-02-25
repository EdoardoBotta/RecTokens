from __future__ import annotations

import torch
import torch.nn as nn

from rectokens.core.quantizer import Quantizer, ResidualQuantizerOutput


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

    def fit_step(self, batch: torch.Tensor) -> ResidualQuantizer:
        """Update each level with a single batch, propagating residuals.

        Level 0 receives the raw batch.  Each subsequent level receives the
        residual left by all previous levels.

        Args:
            batch: Float tensor of shape ``(B, D)``.

        Returns:
            ``self``.
        """
        with torch.no_grad():
            residual = batch.float()
            for quantizer in self._levels:
                quantizer.fit_step(residual)
                residual = quantizer.quantize(residual).residuals
        return self

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
