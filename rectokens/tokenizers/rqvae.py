from __future__ import annotations

from pathlib import Path
from typing import Iterable, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from rectokens.codebooks.euclidean import EuclideanCodebook
from rectokens.core.quantizer import Quantizer, QuantizerOutput
from rectokens.core.tokenizer import TokenSequence, Tokenizer
from rectokens.quantizers.residual import ResidualQuantizer

if TYPE_CHECKING:
    from rectokens.core.dataset import ItemDataset


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
    """

    def __init__(
        self,
        codebook_size: int,
        dim: int,
        *,
        commitment_weight: float = 0.25,
        ema_decay: float = 0.99,
    ) -> None:
        # Both Quantizer (ABC) and nn.Module need their __init__ called
        nn.Module.__init__(self)
        self._codebook = EuclideanCodebook(codebook_size, dim, learnable=False)
        self.commitment_weight = commitment_weight
        self.ema_decay = ema_decay

        # EMA statistics buffers (not parameters — updated manually)
        self.register_buffer("_ema_cluster_size", torch.zeros(codebook_size))
        self.register_buffer("_ema_embed_sum", torch.zeros(codebook_size, dim))

        # Initialise codebook with normal random values
        nn.init.normal_(self._codebook.embeddings)

    @property
    def codebook(self) -> EuclideanCodebook:
        return self._codebook

    def fit(self, batches: Iterable[torch.Tensor]) -> VQQuantizer:
        """No-op: VQQuantizer is trained end-to-end via :meth:`quantize`."""
        del batches  # unused; training happens through forward() + an external optimiser
        return self

    def quantize(self, x: torch.Tensor) -> QuantizerOutput:
        """Quantize ``x`` with straight-through gradient and EMA codebook update.

        Args:
            x: Float tensor of shape ``(B, D)``.

        Returns:
            :class:`~rectokens.core.quantizer.QuantizerOutput` with
            commitment loss set.
        """
        result = self._codebook.find_nearest(x)
        codes = result.codes                           # (B,)
        e_k = self._codebook.lookup(codes)             # (B, D)  — no grad

        # Straight-through: gradients pass through as if no quantization
        quantized_st = x + (e_k - x).detach()

        residuals = (x - e_k).detach()

        # Commitment loss
        commitment_loss = (
            F.mse_loss(x.detach(), e_k)
            + self.commitment_weight * F.mse_loss(x, e_k.detach())
        )

        # EMA codebook update (only during training)
        if self.training:
            self._ema_update(x.detach(), codes)

        return QuantizerOutput(
            codes=codes,
            quantized=quantized_st,
            residuals=residuals,
            commitment_loss=commitment_loss,
        )

    def _ema_update(self, x: torch.Tensor, codes: torch.Tensor) -> None:
        """Update codebook via Exponential Moving Average."""
        k = self._codebook.size
        one_hot = F.one_hot(codes, num_classes=k).float()    # (B, K)
        cluster_size = one_hot.sum(dim=0)                    # (K,)
        embed_sum = one_hot.t() @ x                          # (K, D)

        self._ema_cluster_size.mul_(self.ema_decay).add_(cluster_size * (1 - self.ema_decay))
        self._ema_embed_sum.mul_(self.ema_decay).add_(embed_sum * (1 - self.ema_decay))

        n = self._ema_cluster_size.clamp(min=1e-5)
        new_embeddings = self._ema_embed_sum / n.unsqueeze(1)
        self._codebook.update(torch.arange(k, device=x.device), new_embeddings)

    # Expose as nn.Module forward as well
    def forward(self, x: torch.Tensor) -> QuantizerOutput:
        return self.quantize(x)


# ---------------------------------------------------------------------------
# Encoder / Decoder MLPs
# ---------------------------------------------------------------------------


class Encoder(nn.Module):
    """MLP that maps item features to a latent space.

    Args:
        input_dim: Dimensionality of input item features.
        hidden_dim: Width of the hidden layer.
        latent_dim: Dimensionality of the output latent space
                    (must match the codebook ``dim``).
        num_layers: Total number of linear layers (≥2).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    """MLP that reconstructs item features from a quantized latent vector.

    Args:
        latent_dim: Dimensionality of the quantized latent space.
        hidden_dim: Width of the hidden layer.
        output_dim: Dimensionality of the reconstructed output.
        num_layers: Total number of linear layers (≥2).
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(latent_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, output_dim))
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
        ema_decay: EMA decay for codebook updates.
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
    ) -> None:
        nn.Module.__init__(self)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_levels = num_levels
        self.codebook_size = codebook_size

        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, num_encoder_layers)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim, num_decoder_layers)

        vq_levels = [
            VQQuantizer(
                codebook_size=codebook_size,
                dim=latent_dim,
                commitment_weight=commitment_weight,
                ema_decay=ema_decay,
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
        return {
            "recon": recon,
            "commitment_loss": rq_out.commitment_loss,
            "codes": rq_out.codes.detach(),
        }

    # ------------------------------------------------------------------
    # Tokenizer interface
    # ------------------------------------------------------------------

    def fit(self, dataset: ItemDataset) -> RQVAETokenizer:  # type: ignore[override]
        """Train the RQVAE tokenizer on ``dataset``.

        .. note::
            This method is a **stub**.  A full training loop requires:

            1. An optimiser (e.g. ``Adam``) over all *non-EMA* parameters
               (encoder + decoder weights).
            2. A data loader that yields batches from ``dataset``.
            3. A training loop that calls ``forward`` each step, computes
               the total loss::

                   loss = F.mse_loss(out["recon"], x) + out["commitment_loss"]

               calls ``loss.backward()`` and ``optimizer.step()``.
            4. Optionally: a learning rate scheduler, validation loop,
               early stopping, and checkpoint saving.

        Implement this method (or train with an external loop calling
        :meth:`forward`) to produce a fitted tokenizer.

        Raises:
            NotImplementedError: Always, until implemented.
        """
        raise NotImplementedError(
            "RQVAETokenizer.fit is not yet implemented.  "
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

        latent = torch.zeros(len(codes), self.latent_dim, dtype=torch.float32)
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
        }
