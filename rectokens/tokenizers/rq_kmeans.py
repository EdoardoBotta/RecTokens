from __future__ import annotations

from pathlib import Path

import torch

from rectokens.core.dataset import ItemDataset
from rectokens.core.tokenizer import TokenSequence, Tokenizer
from rectokens.quantizers.kmeans import KMeansQuantizer
from rectokens.quantizers.residual import ResidualQuantizer


class RQKMeansTokenizer(Tokenizer):
    """Tokenizer using Residual Quantization with K-means codebooks.

    Each item is assigned ``num_levels`` code indices, one per residual
    quantization level.  This produces a ``(num_levels,)`` token tuple per
    item that can be used as a discrete item ID in recommendation models
    (e.g. as input to an auto-regressive Transformer).

    Example::

        from rectokens import RQKMeansTokenizer, NumpyDataset
        import numpy as np

        data = np.random.randn(10_000, 128).astype("float32")
        tok = RQKMeansTokenizer(num_levels=3, codebook_size=256, dim=128)
        tok.fit(NumpyDataset(data))

        features = torch.randn(8, 128)
        tokens = tok.encode(features)   # TokenSequence  codes: (8, 3)
        recon  = tok.decode(tokens)     # Tensor (8, 128)

    Args:
        num_levels: Number of residual quantization levels ``L``.
        codebook_size: Codebook size at each level ``K``.
        dim: Item feature dimensionality ``D``.
        init_size: Number of samples buffered for K-means++ seeding at each
                   level.  Defaults to ``10 * codebook_size`` per level.
                   Increase for better initialisation quality on large datasets.
        fit_batch_size: Batch size used when streaming the dataset during
                        fitting.  Does not affect encode/decode.
        seed: Base random seed; level ``i`` uses ``seed + i``.
    """

    def __init__(
        self,
        num_levels: int = 3,
        codebook_size: int = 256,
        dim: int = 64,
        *,
        init_size: int = 0,
        fit_batch_size: int = 4096,
        seed: int = 42,
    ) -> None:
        self.num_levels = num_levels
        self.codebook_size = codebook_size
        self.dim = dim
        self._fit_batch_size = fit_batch_size

        quantizers = [
            KMeansQuantizer(
                codebook_size=codebook_size,
                dim=dim,
                init_size=init_size,
                seed=seed + i,
            )
            for i in range(num_levels)
        ]
        self._rq = ResidualQuantizer(quantizers)
        self._fitted = False

    # ------------------------------------------------------------------
    # Tokenizer interface
    # ------------------------------------------------------------------

    def fit(self, dataset: ItemDataset) -> RQKMeansTokenizer:
        """Fit all K-means codebooks by streaming through ``dataset``.

        The dataset is iterated ``num_levels`` times (once per residual level)
        and only ``fit_batch_size`` items are held in memory per batch, so
        datasets of arbitrary size are supported.

        Args:
            dataset: Any object satisfying the :class:`~rectokens.core.dataset.ItemDataset`
                     protocol.

        Returns:
            ``self``.
        """
        self._rq.fit(dataset, batch_size=self._fit_batch_size)
        self._fitted = True
        return self

    def encode(self, features: torch.Tensor) -> TokenSequence:
        """Encode item features to RQ token sequences.

        Args:
            features: Float tensor of shape ``(B, D)`` or ``(D,)``.

        Returns:
            :class:`~rectokens.core.tokenizer.TokenSequence` with codes of
            shape ``(B, num_levels)`` or ``(num_levels,)``.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if not self._fitted:
            raise RuntimeError("RQKMeansTokenizer must be fit before encoding.")
        single = features.ndim == 1
        if single:
            features = features.unsqueeze(0)
        out = self._rq.quantize(features)
        codes = out.codes  # (B, num_levels)
        return TokenSequence(codes=codes.squeeze(0) if single else codes)

    def decode(self, tokens: TokenSequence) -> torch.Tensor:
        """Reconstruct item features by summing codebook entries.

        Args:
            tokens: :class:`~rectokens.core.tokenizer.TokenSequence`.

        Returns:
            Float tensor of shape ``(B, D)`` or ``(D,)``.
        """
        codes = tokens.codes
        single = codes.ndim == 1
        if single:
            codes = codes.unsqueeze(0)

        recon = torch.zeros(len(codes), self.dim, dtype=torch.float32)
        for level_idx, quantizer in enumerate(self._rq.levels):
            level_codes = codes[:, level_idx]
            recon = recon + quantizer.codebook.lookup(level_codes)

        return recon.squeeze(0) if single else recon

    def save(self, path: str) -> None:
        """Save tokenizer state via ``torch.save``.

        Args:
            path: Destination file path (e.g. ``"tok.pt"``).
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self, path)

    @classmethod
    def load(cls, path: str) -> RQKMeansTokenizer:
        """Load a previously saved tokenizer.

        Args:
            path: Path produced by :meth:`save`.

        Returns:
            Deserialized :class:`RQKMeansTokenizer`.
        """
        return torch.load(path, weights_only=False)
