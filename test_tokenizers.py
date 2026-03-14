#!/usr/bin/env /opt/miniconda3/envs/rqkmeans/bin/python
"""Quick smoke tests for RQKMeansTokenizer and RQVAETokenizer."""

from __future__ import annotations

import sys

import numpy as np
import torch
import torch.nn.functional as F

from rectokens.datasets import NumpyDataset, TensorDataset
from rectokens.tokenizers.rq_kmeans import RQKMeansTokenizer
from rectokens.tokenizers.rqvae import RQVAETokenizer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N_ITEMS = 512
DIM = 32
NUM_LEVELS = 3
CODEBOOK_SIZE = 64
LATENT_DIM = 16
HIDDEN_DIM = 64
RQVAE_TRAIN_STEPS = 200
BATCH_SIZE = 64
LR = 1e-3

PASS = "PASS"
FAIL = "FAIL"


def section(title: str) -> None:
    width = 60
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def check(label: str, condition: bool, detail: str = "") -> bool:
    status = PASS if condition else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{status}] {label}{suffix}")
    return condition


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_numpy_dataset(
    n: int = N_ITEMS, dim: int = DIM, seed: int = 0
) -> tuple[np.ndarray, NumpyDataset]:
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n, dim)).astype(np.float32)
    return data, NumpyDataset(data)


def make_tensor_dataset(
    n: int = N_ITEMS, dim: int = DIM, seed: int = 1
) -> tuple[torch.Tensor, TensorDataset]:
    torch.manual_seed(seed)
    data = torch.randn(n, dim)
    return data, TensorDataset(data)


# ---------------------------------------------------------------------------
# RQKMeansTokenizer tests
# ---------------------------------------------------------------------------


def test_rqkmeans() -> bool:
    section("RQKMeansTokenizer")
    results: list[bool] = []

    # Build dataset
    data_np, dataset = make_numpy_dataset()
    tok = RQKMeansTokenizer(
        num_levels=NUM_LEVELS,
        codebook_size=CODEBOOK_SIZE,
        dim=DIM,
    )

    # 1. Fit
    for batch in dataset.iter_batches(batch_size=64):
        tok.fit_step(batch)
    results.append(check("fit_step() sets _fitted = True", tok._fitted))

    # 2. Batch encode
    features = torch.from_numpy(data_np[:8])
    tokens = tok.encode(features)
    results.append(
        check(
            "encode() returns codes of shape (B, num_levels)",
            tokens.codes.shape == (8, NUM_LEVELS),
            str(tokens.codes.shape),
        )
    )
    results.append(
        check(
            "codes are long integers",
            tokens.codes.dtype == torch.long,
        )
    )
    results.append(
        check(
            "codes are in [0, codebook_size)",
            bool((tokens.codes >= 0).all() and (tokens.codes < CODEBOOK_SIZE).all()),
        )
    )

    # 3. Single-item encode
    single_tokens = tok.encode(features[0])
    results.append(
        check(
            "single-item encode() returns codes of shape (num_levels,)",
            single_tokens.codes.shape == (NUM_LEVELS,),
            str(single_tokens.codes.shape),
        )
    )

    # 4. Batch decode
    recon = tok.decode(tokens)
    results.append(
        check(
            "decode() returns tensor of shape (B, D)",
            recon.shape == (8, DIM),
            str(recon.shape),
        )
    )

    # 5. Single-item decode
    single_recon = tok.decode(single_tokens)
    results.append(
        check(
            "single-item decode() returns tensor of shape (D,)",
            single_recon.shape == (DIM,),
            str(single_recon.shape),
        )
    )

    # 6. Reconstruction quality — should be finite, not all-zero
    results.append(check("reconstruction is finite", bool(torch.isfinite(recon).all())))
    results.append(
        check("reconstruction is non-trivial", bool(recon.abs().max() > 1e-6))
    )

    # 7. to_tuple_ids
    ids = tokens.to_tuple_ids()
    results.append(
        check(
            "to_tuple_ids() returns list of length B",
            len(ids) == 8,
        )
    )
    results.append(
        check(
            "each tuple has length num_levels",
            all(len(t) == NUM_LEVELS for t in ids),
        )
    )

    # 8. to_flat_ids
    flat = tokens.to_flat_ids()
    results.append(
        check(
            "to_flat_ids() returns long tensor of shape (B,)",
            flat.shape == (8,) and flat.dtype == torch.long,
            str(flat.shape),
        )
    )

    # 9. TensorDataset variant
    data_t, td = make_tensor_dataset()
    tok2 = RQKMeansTokenizer(
        num_levels=NUM_LEVELS, codebook_size=CODEBOOK_SIZE, dim=DIM
    )
    for batch in td.iter_batches(batch_size=64):
        tok2.fit_step(batch)
    tok2_tokens = tok2.encode(data_t[:4])
    results.append(
        check(
            "TensorDataset fit_step+encode works",
            tok2_tokens.codes.shape == (4, NUM_LEVELS),
        )
    )

    # 10. Save / load round-trip
    import tempfile, os

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "rqkmeans.pt")
        tok.save(path)
        tok_loaded = RQKMeansTokenizer.load(path)
        loaded_tokens = tok_loaded.encode(features)
        results.append(
            check(
                "save/load round-trip preserves codes",
                bool((loaded_tokens.codes == tokens.codes).all()),
            )
        )

    # 11. Encode before fit_step raises
    tok_unfitted = RQKMeansTokenizer(
        num_levels=NUM_LEVELS, codebook_size=CODEBOOK_SIZE, dim=DIM
    )
    try:
        tok_unfitted.encode(features)
        results.append(check("encode before fit_step raises RuntimeError", False))
    except RuntimeError:
        results.append(check("encode before fit_step raises RuntimeError", True))

    return all(results)


# ---------------------------------------------------------------------------
# RQVAETokenizer tests
# ---------------------------------------------------------------------------


def _train_rqvae(
    tok: RQVAETokenizer, data: torch.Tensor, steps: int, lr: float
) -> list[float]:
    """Minimal training loop for the RQVAE tokenizer."""
    optimizer = torch.optim.Adam(
        [p for p in tok.parameters() if p.requires_grad],
        lr=lr,
    )
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


def test_rqvae() -> bool:
    section("RQVAETokenizer")
    results: list[bool] = []

    torch.manual_seed(42)
    data = torch.randn(N_ITEMS, DIM)

    tok = RQVAETokenizer(
        input_dim=DIM,
        latent_dim=LATENT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_levels=NUM_LEVELS,
        codebook_size=CODEBOOK_SIZE,
    )

    # 1. forward() before training
    x_batch = data[:8]
    out = tok(x_batch)
    results.append(
        check(
            "forward() returns recon of shape (B, input_dim)",
            out["recon"].shape == (8, DIM),
            str(out["recon"].shape),
        )
    )
    results.append(
        check(
            "forward() returns scalar commitment_loss",
            out["commitment_loss"].shape == (),
        )
    )
    results.append(
        check(
            "forward() returns codes of shape (B, num_levels)",
            out["codes"].shape == (8, NUM_LEVELS),
            str(out["codes"].shape),
        )
    )

    # 2. Train
    losses = _train_rqvae(tok, data, steps=RQVAE_TRAIN_STEPS, lr=LR)
    loss_drop = losses[0] - losses[-1]
    results.append(
        check(
            f"loss decreases over {RQVAE_TRAIN_STEPS} steps",
            loss_drop > 0,
            f"Δloss={loss_drop:.4f}",
        )
    )

    # 3. Batch encode
    features = data[:8]
    tokens = tok.encode(features)
    results.append(
        check(
            "encode() returns codes of shape (B, num_levels)",
            tokens.codes.shape == (8, NUM_LEVELS),
            str(tokens.codes.shape),
        )
    )
    results.append(
        check(
            "codes are long integers",
            tokens.codes.dtype == torch.long,
        )
    )
    results.append(
        check(
            "codes are in [0, codebook_size)",
            bool((tokens.codes >= 0).all() and (tokens.codes < CODEBOOK_SIZE).all()),
        )
    )

    # 4. Single-item encode
    single_tokens = tok.encode(features[0])
    results.append(
        check(
            "single-item encode() returns codes of shape (num_levels,)",
            single_tokens.codes.shape == (NUM_LEVELS,),
            str(single_tokens.codes.shape),
        )
    )

    # 5. Batch decode
    recon = tok.decode(tokens)
    results.append(
        check(
            "decode() returns tensor of shape (B, D)",
            recon.shape == (8, DIM),
            str(recon.shape),
        )
    )
    results.append(check("reconstruction is finite", bool(torch.isfinite(recon).all())))

    # 6. Single-item decode
    single_recon = tok.decode(single_tokens)
    results.append(
        check(
            "single-item decode() returns tensor of shape (D,)",
            single_recon.shape == (DIM,),
            str(single_recon.shape),
        )
    )

    # 7. fit_step() stub raises NotImplementedError
    dummy_batch = torch.zeros(4, DIM)
    try:
        tok.fit_step(dummy_batch)
        results.append(check("fit_step() stub raises NotImplementedError", False))
    except NotImplementedError:
        results.append(check("fit_step() stub raises NotImplementedError", True))

    # 8. Save / load round-trip
    import tempfile, os

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "rqvae.pt")
        tok.save(path)
        tok_loaded = RQVAETokenizer.load(path)
        tok_loaded.eval()
        loaded_tokens = tok_loaded.encode(features)
        results.append(
            check(
                "save/load round-trip preserves codes",
                bool((loaded_tokens.codes == tokens.codes).all()),
            )
        )

    return all(results)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    print("\nRecTokens — tokenizer smoke tests")

    ok_kmeans = test_rqkmeans()
    ok_rqvae = test_rqvae()

    section("Summary")
    check("RQKMeansTokenizer", ok_kmeans)
    check("RQVAETokenizer", ok_rqvae)
    print()

    if not (ok_kmeans and ok_rqvae):
        sys.exit(1)


if __name__ == "__main__":
    main()
