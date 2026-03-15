from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from rectokens.datasets import NumpyDataset, TensorDataset
from rectokens.tokenizers.rq_kmeans import RQKMeansTokenizer
from rectokens.tokenizers.rqvae import RQVAETokenizer

N_ITEMS = 512
DIM = 32
NUM_LEVELS = 3
CODEBOOK_SIZE = 64
LATENT_DIM = 16
HIDDEN_DIM = 64
RQVAE_TRAIN_STEPS = 200
BATCH_SIZE = 64
LR = 1e-3


def _train_rqvae(tok: RQVAETokenizer, data: torch.Tensor, steps: int, lr: float) -> list[float]:
    optimizer = torch.optim.Adam([p for p in tok.parameters() if p.requires_grad], lr=lr)
    tok.train()
    losses: list[float] = []
    n = len(data)
    for _ in range(steps):
        idx = torch.randint(0, n, (BATCH_SIZE,))
        x = data[idx]
        out = tok(x)
        loss = F.mse_loss(out["recon"], x) + out["commitment_loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    tok.eval()
    tok._fitted = True
    return losses


class TestTokenizers(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        # RQKMeans setup
        rng = np.random.default_rng(0)
        cls.numpy_data = rng.standard_normal((N_ITEMS, DIM)).astype(np.float32)
        cls.numpy_dataset = NumpyDataset(cls.numpy_data)

        torch.manual_seed(1)
        cls.tensor_data = torch.randn(N_ITEMS, DIM)
        cls.tensor_dataset = TensorDataset(cls.tensor_data)

        cls.fitted_rqkmeans = RQKMeansTokenizer(
            num_levels=NUM_LEVELS, codebook_size=CODEBOOK_SIZE, dim=DIM
        )
        for batch in cls.numpy_dataset.iter_batches(batch_size=BATCH_SIZE):
            cls.fitted_rqkmeans.fit_step(batch)

        # RQVAETokenizer setup
        torch.manual_seed(42)
        cls.rqvae_data = torch.randn(N_ITEMS, DIM)
        cls.rqvae_tok = RQVAETokenizer(
            input_dim=DIM,
            latent_dim=LATENT_DIM,
            hidden_dim=HIDDEN_DIM,
            num_levels=NUM_LEVELS,
            codebook_size=CODEBOOK_SIZE,
        )
        cls.rqvae_losses = _train_rqvae(cls.rqvae_tok, cls.rqvae_data, steps=RQVAE_TRAIN_STEPS, lr=LR)

    # ---------------------------------------------------------------------------
    # RQKMeansTokenizer
    # ---------------------------------------------------------------------------

    def test_rqkmeans_fit_sets_fitted(self) -> None:
        assert self.fitted_rqkmeans._fitted

    def test_rqkmeans_encode_batch_shape(self) -> None:
        tokens = self.fitted_rqkmeans.encode(torch.from_numpy(self.numpy_data[:8]))
        assert tokens.codes.shape == (8, NUM_LEVELS)
        assert tokens.codes.dtype == torch.long

    def test_rqkmeans_codes_in_range(self) -> None:
        tokens = self.fitted_rqkmeans.encode(torch.from_numpy(self.numpy_data[:8]))
        assert (tokens.codes >= 0).all()
        assert (tokens.codes < CODEBOOK_SIZE).all()

    def test_rqkmeans_encode_single_shape(self) -> None:
        tokens = self.fitted_rqkmeans.encode(torch.from_numpy(self.numpy_data[0]))
        assert tokens.codes.shape == (NUM_LEVELS,)

    def test_rqkmeans_decode_batch_shape(self) -> None:
        tokens = self.fitted_rqkmeans.encode(torch.from_numpy(self.numpy_data[:8]))
        recon = self.fitted_rqkmeans.decode(tokens)
        assert recon.shape == (8, DIM)

    def test_rqkmeans_decode_single_shape(self) -> None:
        tokens = self.fitted_rqkmeans.encode(torch.from_numpy(self.numpy_data[0]))
        recon = self.fitted_rqkmeans.decode(tokens)
        assert recon.shape == (DIM,)

    def test_rqkmeans_reconstruction_quality(self) -> None:
        tokens = self.fitted_rqkmeans.encode(torch.from_numpy(self.numpy_data[:8]))
        recon = self.fitted_rqkmeans.decode(tokens)
        assert torch.isfinite(recon).all()
        assert recon.abs().max() > 1e-6

    def test_rqkmeans_to_tuple_ids(self) -> None:
        tokens = self.fitted_rqkmeans.encode(torch.from_numpy(self.numpy_data[:8]))
        ids = tokens.to_tuple_ids()
        assert len(ids) == 8
        assert all(len(t) == NUM_LEVELS for t in ids)

    def test_rqkmeans_to_flat_ids(self) -> None:
        tokens = self.fitted_rqkmeans.encode(torch.from_numpy(self.numpy_data[:8]))
        flat = tokens.to_flat_ids()
        assert flat.shape == (8,)
        assert flat.dtype == torch.long

    def test_rqkmeans_tensor_dataset(self) -> None:
        tok = RQKMeansTokenizer(num_levels=NUM_LEVELS, codebook_size=CODEBOOK_SIZE, dim=DIM)
        for batch in self.tensor_dataset.iter_batches(batch_size=BATCH_SIZE):
            tok.fit_step(batch)
        tokens = tok.encode(self.tensor_data[:4])
        assert tokens.codes.shape == (4, NUM_LEVELS)

    def test_rqkmeans_save_load_roundtrip(self) -> None:
        features = torch.from_numpy(self.numpy_data[:8])
        tokens = self.fitted_rqkmeans.encode(features)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "rqkmeans.pt")
            self.fitted_rqkmeans.save(path)
            loaded = RQKMeansTokenizer.load(path)
            assert (loaded.encode(features).codes == tokens.codes).all()

    def test_rqkmeans_encode_before_fit_raises(self) -> None:
        tok = RQKMeansTokenizer(num_levels=NUM_LEVELS, codebook_size=CODEBOOK_SIZE, dim=DIM)
        with self.assertRaises(RuntimeError):
            tok.encode(torch.randn(4, DIM))

    # ---------------------------------------------------------------------------
    # RQVAETokenizer
    # ---------------------------------------------------------------------------

    def test_rqvae_forward_shapes(self) -> None:
        torch.manual_seed(42)
        tok = RQVAETokenizer(
            input_dim=DIM,
            latent_dim=LATENT_DIM,
            hidden_dim=HIDDEN_DIM,
            num_levels=NUM_LEVELS,
            codebook_size=CODEBOOK_SIZE,
        )
        out = tok(torch.randn(8, DIM))
        assert out["recon"].shape == (8, DIM)
        assert out["commitment_loss"].shape == ()
        assert out["codes"].shape == (8, NUM_LEVELS)

    def test_rqvae_loss_decreases(self) -> None:
        assert self.rqvae_losses[0] > self.rqvae_losses[-1]

    def test_rqvae_encode_batch_shape(self) -> None:
        tokens = self.rqvae_tok.encode(self.rqvae_data[:8])
        assert tokens.codes.shape == (8, NUM_LEVELS)
        assert tokens.codes.dtype == torch.long

    def test_rqvae_codes_in_range(self) -> None:
        tokens = self.rqvae_tok.encode(self.rqvae_data[:8])
        assert (tokens.codes >= 0).all()
        assert (tokens.codes < CODEBOOK_SIZE).all()

    def test_rqvae_encode_single_shape(self) -> None:
        tokens = self.rqvae_tok.encode(self.rqvae_data[0])
        assert tokens.codes.shape == (NUM_LEVELS,)

    def test_rqvae_decode_batch_shape(self) -> None:
        tokens = self.rqvae_tok.encode(self.rqvae_data[:8])
        recon = self.rqvae_tok.decode(tokens)
        assert recon.shape == (8, DIM)
        assert torch.isfinite(recon).all()

    def test_rqvae_decode_single_shape(self) -> None:
        tokens = self.rqvae_tok.encode(self.rqvae_data[0])
        recon = self.rqvae_tok.decode(tokens)
        assert recon.shape == (DIM,)

    def test_rqvae_fit_step_raises(self) -> None:
        with self.assertRaises(NotImplementedError):
            self.rqvae_tok.fit_step(torch.zeros(4, DIM))

    def test_rqvae_save_load_roundtrip(self) -> None:
        features = self.rqvae_data[:8]
        tokens = self.rqvae_tok.encode(features)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "rqvae.pt")
            self.rqvae_tok.save(path)
            loaded = RQVAETokenizer.load(path)
            loaded.eval()
            assert (loaded.encode(features).codes == tokens.codes).all()
