from __future__ import annotations

import torch

from rectokens.codebooks.euclidean import EuclideanCodebook
from rectokens.core.codebook import Codebook
from rectokens.core.quantizer import Quantizer, QuantizerOutput


class KMeansQuantizer(Quantizer):
    """Single-level quantizer whose codebook is fit via mini-batch K-means.

    This is *not* an ``nn.Module`` — the codebook contains no trainable
    parameters and is fit unsupervised via :meth:`fit`.

    Unlike full-batch K-means, mini-batch K-means processes the training data
    one batch at a time, making it suitable for datasets that do not fit in
    memory.

    Algorithm
    ---------
    1. **Initialisation** — K-means++ seeding on the first ``init_size``
       samples seen in the stream.  Only this buffer needs to be held in
       memory simultaneously; the rest of the data is processed online.
    2. **Mini-batch update** — for each incoming batch, assign every point to
       its nearest current centroid, then update centroids via a running
       average::

           n_k  ← n_k + |batch_k|
           c_k  ← (n_k_old · c_k + sum(batch_k)) / n_k

       This is an unbiased online estimate of the centroid for every cluster
       that has received at least one point.
    3. **Empty clusters** — centroids that never receive a point retain their
       initialised values, which is safe and avoids re-seeding complexity.

    Args:
        codebook_size: Number of centroids / discrete codes ``K``.
        dim: Feature dimensionality ``D``.
        init_size: Number of samples buffered from the start of the stream for
                   K-means++ initialisation.  Defaults to ``10 * codebook_size``,
                   which gives reliable seeding without loading the full dataset.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        codebook_size: int,
        dim: int,
        *,
        seed: int = 42,
    ) -> None:
        self._codebook = EuclideanCodebook(codebook_size, dim)
        self._seed = seed
        self._counts: torch.Tensor | None = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Quantizer interface
    # ------------------------------------------------------------------

    @property
    def codebook(self) -> Codebook:
        return self._codebook

    def fit_step(self, batch: torch.Tensor) -> KMeansQuantizer:
        """Update centroids with one batch of data.

        On the first call the codebook is seeded with K-means++ on ``batch``.
        Every subsequent call performs a mini-batch running-average update.

        Args:
            batch: Float tensor of shape ``(B, D)``.

        Returns:
            ``self``.
        """
        batch = batch.float()
        k = self._codebook.size
        device = batch.device

        if self._counts is None:
            # First call: K-means++ initialisation on this batch
            n = len(batch)
            if n == 0:
                raise ValueError("fit_step() received an empty batch.")
            actual_k = min(k, n)
            generator = torch.Generator(device=device).manual_seed(self._seed)
            centroids = self._kmeans_plus_plus_init(batch, actual_k, generator)
            if actual_k < k:
                extra_idx = torch.randint(n, (k - actual_k,), generator=generator, device=device)
                centroids = torch.cat([centroids, batch[extra_idx]])
            self._codebook.update(torch.arange(k, device=device), centroids)
            self._counts = torch.zeros(k, dtype=torch.long, device=device)

        centroids = self._codebook.embeddings.detach().clone()
        centroids, self._counts = _minibatch_update(batch, centroids, self._counts)
        self._codebook.update(torch.arange(k, device=device), centroids)
        self._fitted = True
        return self

    def quantize(self, x: torch.Tensor) -> QuantizerOutput:
        """Assign each vector in ``x`` to its nearest centroid.

        Args:
            x: Float tensor of shape ``(B, D)``.

        Returns:
            :class:`~rectokens.core.quantizer.QuantizerOutput` (no gradients,
            ``commitment_loss`` is ``None``).

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if not self._fitted:
            raise RuntimeError("KMeansQuantizer must be fit before quantizing.")
        with torch.no_grad():
            result = self._codebook.find_nearest(x)
            quantized = self._codebook.lookup(result.codes)
            residuals = x - quantized
        return QuantizerOutput(
            codes=result.codes,
            quantized=quantized,
            residuals=residuals,
            commitment_loss=None,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _kmeans_plus_plus_init(
        data: torch.Tensor,
        k: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """K-means++ centroid seeding on ``data``."""
        n = len(data)
        first_idx = int(torch.randint(n, (1,), generator=generator, device=data.device).item())
        centroids = [data[first_idx]]

        for _ in range(k - 1):
            stacked = torch.stack(centroids)                          # (c, D)
            dists = torch.cdist(data, stacked).min(dim=1).values ** 2  # (N,)
            total = dists.sum()
            if total == 0:
                # All points coincide with existing centroids; pick uniformly
                idx = int(torch.randint(n, (1,), generator=generator, device=data.device).item())
            else:
                idx = int(torch.multinomial(dists / total, 1, generator=generator).item())
            centroids.append(data[idx])

        return torch.stack(centroids)


def _minibatch_update(
    batch: torch.Tensor,
    centroids: torch.Tensor,
    counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update ``centroids`` with one mini-batch using a running average.

    For each centroid ``k`` that receives at least one point in ``batch``:

        n_k  ← n_k + |batch_k|
        c_k  ← (n_k_old · c_k + sum(batch_k)) / n_k

    Centroids that receive no points are left unchanged.

    Args:
        batch: Float tensor ``(B, D)``.
        centroids: Current centroid matrix ``(K, D)``.
        counts: Running count of points assigned to each centroid ``(K,)``.

    Returns:
        Updated ``(centroids, counts)``.
    """
    k, d = centroids.shape
    device = batch.device

    tmp_cb = EuclideanCodebook.from_tensor(centroids)
    assignments = tmp_cb.find_nearest(batch).codes  # (B,)

    batch_counts = torch.zeros(k, dtype=torch.long, device=device)
    batch_sums = torch.zeros(k, d, dtype=torch.float32, device=device)
    ones = torch.ones(len(batch), dtype=torch.long, device=device)

    batch_counts.scatter_add_(0, assignments, ones)
    batch_sums.scatter_add_(0, assignments.unsqueeze(1).expand(-1, d), batch)

    new_counts = counts + batch_counts
    new_centroids = centroids.clone()

    active = batch_counts > 0
    if active.any():
        n_old = counts[active].float().unsqueeze(1)
        n_new = new_counts[active].float().unsqueeze(1)
        new_centroids[active] = (n_old * centroids[active] + batch_sums[active]) / n_new

    return new_centroids, new_counts
