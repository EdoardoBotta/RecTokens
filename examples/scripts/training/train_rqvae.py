"""Train an RQVAETokenizer on Amazon item embeddings.

Usage:
    python examples/scripts/training/train_rqvae.py examples/configs/train_rqvae_beauty.gin
"""

from __future__ import annotations

import os

import gin
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import BatchSampler, DataLoader, RandomSampler

from examples.data.amazon import ItemData
from examples.utils import parse_config
from rectokens.tokenizers.rqvae import RQVAETokenizer


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@gin.configurable
def train(
    root: str = "data/amazon",
    split: str = "beauty",
    train_test_split: str = "train",
    latent_dim: int = 64,
    hidden_dim: int = 512,
    num_levels: int = 3,
    codebook_size: int = 256,
    num_epochs: int = 100,
    batch_size: int = 640,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    log_every: int = 1,
    save_every: int = 10,
    output_dir: str = "checkpoints/rqvae",
) -> RQVAETokenizer:
    device = get_device()
    print(f"Training on: {device}")

    dataset = ItemData(root=root, split=split, train_test_split=train_test_split)
    print(f"Dataset: {len(dataset)} items, dim={dataset[0].x.shape[0]}")

    input_dim = dataset[0].x.shape[0]
    model = RQVAETokenizer(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_levels=num_levels,
        codebook_size=codebook_size,
        learnable_codebook=False,
    ).to(device)

    # AdamW only sees encoder + decoder weights — codebook embeddings are
    # registered as non-trainable buffers and updated via EMA in forward().
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_sampler = BatchSampler(RandomSampler(dataset), batch_size, False)
    loader = DataLoader(
        dataset, sampler=train_sampler, batch_size=None, collate_fn=lambda batch: batch
    )

    os.makedirs(output_dir, exist_ok=True)

    model.train()
    for epoch in range(1, num_epochs + 1):
        total_recon = 0.0
        total_commit = 0.0
        total_unique = 0.0

        for batch in loader:
            x = batch.x.float().to(device)
            optimizer.zero_grad()

            out = model(x)
            recon_loss = F.mse_loss(out.recon, x, reduction="none").sum(dim=-1).mean()
            commit_loss = out.commitment_loss
            (recon_loss + commit_loss).backward()
            optimizer.step()

            total_recon += recon_loss.item()
            total_commit += commit_loss.item()
            total_unique += out.p_unique_ids.item()

        if epoch % log_every == 0:
            n = len(loader)
            print(
                f"epoch {epoch:3d}/{num_epochs}"
                f"  recon={total_recon / n:.4f}"
                f"  commit={total_commit / n:.4f}"
                f"  total={(total_recon + total_commit) / n:.4f}"
                f"  p_unique={total_unique / n:.3f}"
            )

        if save_every > 0 and epoch % save_every == 0:
            ckpt = os.path.join(output_dir, f"epoch_{epoch}.pt")
            model._fitted = True
            model.save(ckpt)
            print(f"Saved checkpoint → {ckpt}")

    model._fitted = True
    final_path = os.path.join(output_dir, "final.pt")
    model.save(final_path)
    print(f"Saved final model → {final_path}")

    # Sanity check
    model.eval()
    tokens = model.encode(dataset.item_data[:4].float().to(device))
    recon = model.decode(tokens)
    print(f"\nEncoded shape : {tokens.codes.shape}")
    print(f"Decoded shape : {recon.shape}")
    print(f"Token tuples  : {tokens.to_tuple_ids()}")

    return model


if __name__ == "__main__":
    torch.manual_seed(0)
    parse_config()
    train()
