from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from rectokens.codebooks.euclidean import EuclideanCodebook
from rectokens.core.quantizer import Quantizer, QuantizerOutput
from rectokens.core.tokenizer import TokenSequence, Tokenizer
from rectokens.quantizers.residual import ResidualQuantizer


# ---------------------------------------------------------------------------
# VQ building block — straight-through vector quantizer
# ---------------------------------------------------------------------------


class VQQuantizer(Quantizer, nn.Module):
    """Single-level vector quantizer with straight-through gradient estimator.

    During the forward pass (``quantize``):
    1. Find the nearest codebook entry ``e_k`` for each input ``z``.
    2. Return ``z + (e_k - z).detach()`` as the "quantized" output so that
       gradients flow directly through to the encoder (straight-through
       estimator).
    3. Compute the VQ commitment loss:
       ``‖sg[z] - e_k‖² + β·‖z - sg[e_k]‖²``
       where ``sg[·]`` is stop-gradient and ``β`` is the commitment weight.

    The codebook embeddings are updated via Exponential Moving Average (EMA)
    rather than gradient descent, which is more stable in practice.

    Args:
        codebook_size: Number of discrete codes.
        dim: Embedding dimension.
        commitment_weight: Weight ``β`` on the encoder commitment term.
        ema_decay: EMA decay factor for codebook updates (``γ`` in the paper).
        restart_after_steps: Replace a code with a random batch sample once it
            has gone this many consecutive training steps without any assignment.
            Covers both codes that were never hit *and* codes that became stranded
            as the encoder drifted.  Set relative to ``batch_size / codebook_size``:
            a value ≈20 works well when each code is expected to receive 2–3
            assignments per batch.  Ignored when ``learnable_codebook=True``.
        learnable_codebook: If ``True``, store codebook entries as an
            ``nn.Parameter`` and train them with the optimiser instead of EMA.
            The full VQ loss ``‖sg[z] - e_k‖² + β·‖z - sg[e_k]‖²`` is used so
            gradients flow into both the encoder and the codebook.
            EMA updates and dead-code restarts are disabled in this mode.
    """

    def __init__(
        self,
        codebook_size: int,
        dim: int,
        *,
        commitment_weight: float = 0.25,
        ema_decay: float = 0.99,
        restart_after_steps: int = 20,
        learnable_codebook: bool = False,
    ) -> None:
        # Both Quantizer (ABC) and nn.Module need their __init__ called
        nn.Module.__init__(self)
        self._codebook = EuclideanCodebook(
            codebook_size, dim, learnable=learnable_codebook
        )
        self.commitment_weight = commitment_weight
        self.ema_decay = ema_decay
        self.restart_after_steps = restart_after_steps
        self.learnable_codebook = learnable_codebook

        # EMA statistics buffers (not parameters — updated manually)
        self.register_buffer("_ema_cluster_size", torch.zeros(codebook_size))
        self.register_buffer("_ema_embed_sum", torch.zeros(codebook_size, dim))
        # Persisted in state_dict so save/load preserves whether data-driven init ran
        self.register_buffer("_initialized", torch.zeros(1, dtype=torch.bool))
        # Steps since each code last received an assignment — used for dead-code restart
        self.register_buffer(
            "_steps_since_active", torch.zeros(codebook_size, dtype=torch.long)
        )

        # Placeholder init — overwritten on the first training batch by _init_from_batch,
        # which seeds entries directly from encoder outputs.
        nn.init.normal_(self._codebook.embeddings)

    @property
    def codebook(self) -> EuclideanCodebook:
        return self._codebook

    def fit_step(self, batch: torch.Tensor) -> VQQuantizer:  # noqa: ARG002
        """No-op: VQQuantizer is trained end-to-end via :meth:`quantize`."""
        return self

    def quantize(self, x: torch.Tensor) -> QuantizerOutput:
        """Quantize ``x`` with straight-through gradient and EMA codebook update.

        Args:
            x: Float tensor of shape ``(B, D)``.

        Returns:
            :class:`~rectokens.core.quantizer.QuantizerOutput` with
            commitment loss set.
        """
        # On the first training batch, seed the codebook from actual encoder outputs
        # so that every entry starts inside the data manifold.
        if self.training and not self._initialized.item():
            self._init_from_batch(x.detach())

        result = self._codebook.find_nearest(x)
        codes = result.codes  # (B,)
        e_k = self._codebook.lookup(codes)  # (B, D)  — no grad

        # Straight-through: gradients pass through as if no quantization
        quantized_st = x + (e_k - x).detach()

        residuals = (x - e_k).detach()

        if self.learnable_codebook:
            # Gradient mode: full VQ loss — both encoder and codebook receive
            # gradients.  e_k is an nn.Parameter so F.mse_loss(x.detach(), e_k)
            # pushes the codebook entry toward the encoder output; the commitment
            # term pushes the encoder output toward the (detached) codebook entry.
            # import pdb; pdb.set_trace()  # noqa: T100
            commitment_loss = (
                F.mse_loss(x.detach(), e_k, reduction="none").sum(dim=-1).mean()
                + self.commitment_weight
                * F.mse_loss(x, e_k.detach(), reduction="none").sum(dim=-1).mean()
            )
        else:
            # EMA mode: codebook updated externally; only encoder commitment term.
            commitment_loss = self.commitment_weight * F.mse_loss(x, e_k.detach())

            # EMA codebook update + dead-code restart (only during training)
            if self.training:
                self._ema_update(x.detach(), codes)

        return QuantizerOutput(
            codes=codes,
            quantized=quantized_st,
            residuals=residuals,
            commitment_loss=commitment_loss,
        )

    def _init_from_batch(self, x: torch.Tensor) -> None:
        """Seed codebook entries via K-means++ on the first training batch.

        K-means++ places the initial centroids far apart within the actual
        encoder-output distribution, giving each code a distinct region of
        latent space to own from step one.  Without data-driven init, random-
        normal codes sit far from the data manifold and the nearest-neighbour
        search always returns the same few codes.
        """
        k = self._codebook.size
        n = len(x)
        actual_k = min(k, n)

        # Seed first centroid uniformly at random
        first = int(torch.randint(n, (1,), device=x.device).item())
        centroids = [x[first]]

        for _ in range(actual_k - 1):
            stacked = torch.stack(centroids)  # (c, D)
            dists = torch.cdist(x, stacked).min(dim=1).values ** 2  # (N,)
            total = dists.sum()
            if total == 0:
                idx = int(torch.randint(n, (1,), device=x.device).item())
            else:
                idx = int(torch.multinomial(dists / total, 1).item())
            centroids.append(x[idx])

        init = torch.stack(centroids)  # (actual_k, D)
        if actual_k < k:
            # Batch smaller than codebook: fill remaining with random repeats
            extra = torch.randint(n, (k - actual_k,), device=x.device)
            init = torch.cat([init, x[extra]])

        self._codebook.update(torch.arange(k, device=x.device), init)
        self._initialized.fill_(True)

    def _ema_update(self, x: torch.Tensor, codes: torch.Tensor) -> None:
        """Update assigned codebook entries via EMA; restart dead codes.

        Only codes that received at least one assignment in this batch are
        EMA-updated.  Applying ``mul_(decay)`` to every code including those
        with zero assignments causes their accumulators to decay to 0 and their
        codebook entries to be overwritten with ``0 / ε = 0``.

        Dead-code restart: ``_steps_since_active`` is incremented for every
        code that receives no assignment in the current batch and reset to 0
        for codes that do.  Any code whose counter reaches
        ``restart_after_steps`` is replaced with a random encoder output from
        the current batch.  This handles two distinct failure modes:

        * **Never-used codes** — random-normal or K-means++ entries that the
          encoder never visits; the counter starts at 0 and climbs until
          restart.
        * **Abandoned codes** — entries that *were* active but became stranded
          as the encoder drifted.  Under the active-only EMA update their
          ``_ema_cluster_size`` stays positive forever, so a simple
          ``ema == 0`` check would never restart them.
        """
        k = self._codebook.size
        one_hot = F.one_hot(codes, num_classes=k).float()  # (B, K)
        cluster_size = one_hot.sum(dim=0)  # (K,)
        embed_sum = one_hot.t() @ x  # (K, D)

        active = cluster_size > 0
        if active.any():
            # EMA update restricted to active codes only
            self._ema_cluster_size[active] = (
                self.ema_decay * self._ema_cluster_size[active]
                + (1 - self.ema_decay) * cluster_size[active]
            )
            self._ema_embed_sum[active] = (
                self.ema_decay * self._ema_embed_sum[active]
                + (1 - self.ema_decay) * embed_sum[active]
            )

            n = self._ema_cluster_size[active].clamp(min=1e-5)
            new_embeddings = self._ema_embed_sum[active] / n.unsqueeze(1)
            self._codebook.update(torch.where(active)[0], new_embeddings)

        # Dead-code restart: track consecutive steps without assignment and
        # replace stranded codes with random batch samples.
        #
        # This handles two failure modes that the EMA-only check misses:
        #   1. Codes never seeded into the data manifold (steps_since_active
        #      increments from 0 until restart_after_steps is reached).
        #   2. Codes that WERE active but became stranded as the encoder drifted
        #      (ema_cluster_size stays > 0 forever under the active-only update,
        #      so a pure ema==0 check would never restart them).
        self._steps_since_active[active] = 0
        self._steps_since_active[~active] += 1

        dead = self._steps_since_active >= self.restart_after_steps
        n_dead = int(dead.sum().item())
        if n_dead > 0:
            rand_idx = torch.randint(len(x), (n_dead,), device=x.device)
            self._codebook.update(torch.where(dead)[0], x[rand_idx])
            self._ema_cluster_size[dead] = 0.0
            self._ema_embed_sum[dead] = 0.0
            self._steps_since_active[dead] = 0

    # Expose as nn.Module forward as well
    def forward(self, x: torch.Tensor) -> QuantizerOutput:
        return self.quantize(x)


# ---------------------------------------------------------------------------
# Shared MLP (used for both encoder and decoder)
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    """ReLU MLP with configurable depth.

    Args:
        in_dim: Input dimensionality.
        hidden_dim: Width of each hidden layer.
        out_dim: Output dimensionality.
        num_layers: Total number of linear layers (≥2).
    """

    def __init__(
        self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 3
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# RQVAETokenizer
# ---------------------------------------------------------------------------


class RQVAETokenizer(nn.Module, Tokenizer):
    """Tokenizer using a learned Residual-Quantization VAE.

    Architecture:

    .. code-block:: text

        item features (D)
              │
         Encoder (MLP)
              │
         latent z (latent_dim)
              │
        ResidualQuantizer
        ├── VQQuantizer level 0  →  code_0
        ├── VQQuantizer level 1  →  code_1
        └── VQQuantizer level L  →  code_L
              │
         quantized ẑ (latent_dim)
              │
         Decoder (MLP)
              │
        reconstruction x̂ (D)

    Training uses:
    * **Reconstruction loss** — ``MSE(x̂, x)`` (or cross-entropy for categorical features).
    * **VQ commitment loss** — summed across all levels, encourages encoder outputs
      to stay close to their assigned codebook entries.
    * **Straight-through estimator** — lets gradients flow through the
      non-differentiable argmin in each :class:`VQQuantizer`.

    The codebook embeddings are updated via Exponential Moving Average (EMA)
    rather than being parameters in the optimiser.

    Args:
        input_dim: Dimensionality of item feature vectors.
        latent_dim: Dimensionality of the encoder output / codebook entries.
        hidden_dim: Hidden layer width for the Encoder and Decoder MLPs.
        num_levels: Number of residual quantization levels.
        codebook_size: Codebook size at each level.
        num_encoder_layers: Depth of the Encoder MLP.
        num_decoder_layers: Depth of the Decoder MLP.
        commitment_weight: VQ commitment loss weight ``β``.
        ema_decay: EMA decay for codebook updates.  Ignored when
            ``learnable_codebook=True``.
        learnable_codebook: If ``True``, codebook entries are ``nn.Parameter``s
            trained by the optimiser rather than updated via EMA.  The full VQ
            loss is used, enabling gradient flow into both encoder and codebook.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 64,
        hidden_dim: int = 256,
        num_levels: int = 3,
        codebook_size: int = 256,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        commitment_weight: float = 0.25,
        ema_decay: float = 0.99,
        learnable_codebook: bool = False,
    ) -> None:
        nn.Module.__init__(self)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_levels = num_levels
        self.codebook_size = codebook_size
        self.learnable_codebook = learnable_codebook

        self.encoder = MLP(input_dim, hidden_dim, latent_dim, num_encoder_layers)
        self.decoder = MLP(latent_dim, hidden_dim, input_dim, num_decoder_layers)

        vq_levels = [
            VQQuantizer(
                codebook_size=codebook_size,
                dim=latent_dim,
                commitment_weight=commitment_weight,
                ema_decay=ema_decay,
                learnable_codebook=learnable_codebook,
            )
            for _ in range(num_levels)
        ]
        self.rq = ResidualQuantizer(vq_levels)
        self._fitted = False

    # ------------------------------------------------------------------
    # nn.Module forward (used during training)
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Full encode → quantize → decode forward pass.

        Args:
            x: Float tensor of shape ``(B, input_dim)``.

        Returns:
            Dict with keys:
            - ``"recon"`` — reconstructed features ``(B, input_dim)``.
            - ``"commitment_loss"`` — scalar VQ commitment loss.
            - ``"codes"`` — long tensor ``(B, num_levels)`` (detached).
        """
        z = self.encoder(x)
        rq_out = self.rq.quantize(z)
        recon = self.decoder(rq_out.quantized)

        with torch.no_grad():
            codes = rq_out.codes  # (B, L)
            # Pairwise full-tuple equality: (B, B)
            eq = (codes.unsqueeze(0) == codes.unsqueeze(1)).all(dim=-1)
            # Row i is True when no later row j (j > i) shares the same tuple,
            # i.e. i is the last occurrence of its tuple in the batch.
            # p_unique_ids = (# distinct tuples) / B
            p_unique_ids = (~torch.triu(eq, diagonal=1)).all(dim=1).float().mean()

        return {
            "recon": recon,
            "commitment_loss": rq_out.commitment_loss,
            "codes": rq_out.codes.detach(),
            "p_unique_ids": p_unique_ids,
        }

    # ------------------------------------------------------------------
    # Tokenizer interface
    # ------------------------------------------------------------------

    def fit_step(self, batch: torch.Tensor) -> RQVAETokenizer:  # type: ignore[override]
        """Train the RQVAE tokenizer on a single batch.

        .. note::
            This method is a **stub**.  A full training step requires:

            1. An optimiser (e.g. ``Adam``) over all *non-EMA* parameters
               (encoder + decoder weights).
            2. A call to ``forward(batch)`` to get the outputs.
            3. Computing the total loss::

                   loss = F.mse_loss(out["recon"], batch) + out["commitment_loss"]

               and calling ``loss.backward()`` and ``optimizer.step()``.

        Implement this method (or train with an external loop calling
        :meth:`forward`) to produce a fitted tokenizer.

        Raises:
            NotImplementedError: Always, until implemented.
        """
        raise NotImplementedError(
            "RQVAETokenizer.fit_step is not yet implemented.  "
            "Train the model end-to-end using the forward() method and an "
            "external optimisation loop, then call tok._fitted = True."
        )

    @torch.no_grad()
    def encode(self, features: torch.Tensor) -> TokenSequence:
        """Encode item features to RQ token sequences.

        Runs the encoder followed by residual quantization.

        Args:
            features: Float tensor of shape ``(B, input_dim)`` or ``(input_dim,)``.

        Returns:
            :class:`~rectokens.core.tokenizer.TokenSequence`.
        """
        single = features.ndim == 1
        if single:
            features = features.unsqueeze(0)
        z = self.encoder(features)
        rq_out = self.rq.quantize(z)
        codes = rq_out.codes
        return TokenSequence(codes=codes.squeeze(0) if single else codes)

    @torch.no_grad()
    def decode(self, tokens: TokenSequence) -> torch.Tensor:
        """Reconstruct item features from token codes.

        Looks up codebook entries for each level, sums them, then passes
        through the decoder MLP.

        Args:
            tokens: :class:`~rectokens.core.tokenizer.TokenSequence`.

        Returns:
            Float tensor of shape ``(B, input_dim)`` or ``(input_dim,)``.
        """
        codes = tokens.codes
        single = codes.ndim == 1
        if single:
            codes = codes.unsqueeze(0)

        latent = torch.zeros(
            len(codes), self.latent_dim, dtype=torch.float32, device=codes.device
        )
        for level_idx, quantizer in enumerate(self.rq.levels):
            level_codes = codes[:, level_idx]
            latent = latent + quantizer.codebook.lookup(level_codes)

        recon = self.decoder(latent)
        return recon.squeeze(0) if single else recon

    def save(self, path: str) -> None:
        """Save model weights and config via ``torch.save``.

        Args:
            path: Destination file path.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.state_dict(), "config": self._config()}, path)

    @classmethod
    def load(cls, path: str) -> RQVAETokenizer:
        """Load a saved RQVAE tokenizer.

        Args:
            path: Path produced by :meth:`save`.

        Returns:
            :class:`RQVAETokenizer` with restored weights.
        """
        checkpoint = torch.load(path, weights_only=True)
        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        model._fitted = True
        return model

    def _config(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "hidden_dim": self.encoder.net[0].out_features,
            "num_levels": self.num_levels,
            "codebook_size": self.codebook_size,
            "learnable_codebook": self.learnable_codebook,
        }
