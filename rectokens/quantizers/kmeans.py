from __future__ import annotations

from typing import Iterable

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
        init_size: int = 0,
        seed: int = 42,
    ) -> None:
        self._codebook = EuclideanCodebook(codebook_size, dim)
        self._init_size = init_size if init_size > 0 else 10 * codebook_size
        self._seed = seed
        self._fitted = False

    # ------------------------------------------------------------------
    # Quantizer interface
    # ------------------------------------------------------------------

    @property
    def codebook(self) -> Codebook:
        return self._codebook

    def fit(self, batches: Iterable[torch.Tensor]) -> KMeansQuantizer:
        """Fit centroids via streaming mini-batch K-means.

        Only ``init_size`` samples are held in memory at once for the
        K-means++ initialisation step; all remaining data is processed a
        batch at a time.

        Args:
            batches: Iterable of float tensors of shape ``(B, D)``.  May be
                     a lazy generator — it is consumed exactly once.

        Returns:
            ``self``.
        """
        k = self._codebook.size
        batch_iter = iter(batches)

        # ------------------------------------------------------------------
        # Step 1: buffer init_size samples for K-means++ seeding
        # ------------------------------------------------------------------
        init_data, leftover, batch_iter = self._collect_init(batch_iter, self._init_size)

        if len(init_data) == 0:
            raise ValueError("fit() received no data.")

        device = init_data.device
        generator = torch.Generator(device=device).manual_seed(self._seed)

        # ------------------------------------------------------------------
        # Step 2: K-means++ initialisation on the buffered samples
        # ------------------------------------------------------------------
        n_init = len(init_data)
        actual_k = min(k, n_init)  # can't have more centroids than data points
        centroids = self._kmeans_plus_plus_init(init_data, actual_k, generator)

        # Pad to k if we had fewer data points than codebook entries
        if actual_k < k:
            extra_idx = torch.randint(n_init, (k - actual_k,), generator=generator, device=device)
            centroids = torch.cat([centroids, init_data[extra_idx]])

        # ------------------------------------------------------------------
        # Step 3: mini-batch updates — running-average centroid estimates
        # ------------------------------------------------------------------
        counts = torch.zeros(k, dtype=torch.long, device=device)

        # Process the buffered init data first (so it is counted in the update)
        centroids, counts = _minibatch_update(init_data, centroids, counts)

        # Process the partial batch that was split off during buffering
        if leftover is not None and len(leftover) > 0:
            centroids, counts = _minibatch_update(leftover, centroids, counts)

        # Process the remainder of the stream
        for batch in batch_iter:
            centroids, counts = _minibatch_update(batch.float(), centroids, counts)

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
    def _collect_init(
        batch_iter: Iterable[torch.Tensor],
        init_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None, Iterable[torch.Tensor]]:
        """Buffer up to ``init_size`` samples from the front of the stream.

        Returns:
            ``(init_data, leftover, remaining_iter)`` where ``leftover`` is the
            portion of the last consumed batch that was not needed for init, and
            ``remaining_iter`` is the iterator to continue from.
        """
        buffer: list[torch.Tensor] = []
        collected = 0
        leftover: torch.Tensor | None = None

        for batch in batch_iter:
            batch = batch.float()
            need = init_size - collected
            if len(batch) <= need:
                buffer.append(batch)
                collected += len(batch)
            else:
                buffer.append(batch[:need])
                leftover = batch[need:]
                collected = init_size
                break
            if collected >= init_size:
                break

        if not buffer:
            return torch.empty(0), None, iter([])

        return torch.cat(buffer), leftover, batch_iter

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
            probs = dists / dists.sum()
            idx = int(torch.multinomial(probs, 1, generator=generator).item())
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
