"""Train an RQVAETokenizer on Amazon item embeddings.

Usage:
    python examples/training/train_rqvae.py \
        --root data/amazon --split beauty \
        --latent_dim 64 --hidden_dim 512 \
        --num_levels 3 --codebook_size 256 \
        --num_epochs 100 --batch_size 640 --lr 1e-3 \
        --save_every 10 --output_dir checkpoints/rqvae
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import BatchSampler, DataLoader, RandomSampler

from examples.data.amazon import ItemData
from rectokens.tokenizers.rqvae import RQVAETokenizer


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train RQVAETokenizer on Amazon item embeddings")
    p.add_argument("--root", type=str, default="data/amazon")
    p.add_argument("--split", type=str, default="beauty", choices=["beauty", "sports", "toys"])
    p.add_argument("--train_test_split", type=str, default="train", choices=["train", "eval", "all"])
    p.add_argument("--latent_dim", type=int, default=64)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--num_levels", type=int, default=3)
    p.add_argument("--codebook_size", type=int, default=256)
    p.add_argument("--num_epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=640)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--log_every", type=int, default=1)
    p.add_argument("--save_every", type=int, default=10,
                   help="Save a checkpoint every this many epochs (0 = only save at end)")
    p.add_argument("--output_dir", type=str, default="checkpoints/rqvae")
    return p.parse_args()


def train(args: argparse.Namespace) -> RQVAETokenizer:
    device = get_device()
    print(f"Training on: {device}")

    dataset = ItemData(root=args.root, split=args.split, train_test_split=args.train_test_split)
    print(f"Dataset: {len(dataset)} items, dim={dataset[0].x.shape[0]}")

    input_dim = dataset[0].x.shape[0]
    model = RQVAETokenizer(
        input_dim=input_dim,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_levels=args.num_levels,
        codebook_size=args.codebook_size,
        learnable_codebook=False,
    ).to(device)

    # AdamW only sees encoder + decoder weights — codebook embeddings are
    # registered as non-trainable buffers and updated via EMA in forward().
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_sampler = BatchSampler(RandomSampler(dataset), args.batch_size, False)
    loader = DataLoader(
        dataset, sampler=train_sampler, batch_size=None, collate_fn=lambda batch: batch
    )

    os.makedirs(args.output_dir, exist_ok=True)

    model.train()
    for epoch in range(1, args.num_epochs + 1):
        total_recon = 0.0
        total_commit = 0.0
        total_unique = 0.0

        for batch in loader:
            x = batch.x.float().to(device)
            optimizer.zero_grad()

            out = model(x)
            recon_loss = F.mse_loss(out["recon"], x, reduction="none").sum(dim=-1).mean()
            commit_loss = out["commitment_loss"]
            (recon_loss + commit_loss).backward()
            optimizer.step()

            total_recon += recon_loss.item()
            total_commit += commit_loss.item()
            total_unique += out["p_unique_ids"].item()

        if epoch % args.log_every == 0:
            n = len(loader)
            print(
                f"epoch {epoch:3d}/{args.num_epochs}"
                f"  recon={total_recon / n:.4f}"
                f"  commit={total_commit / n:.4f}"
                f"  total={(total_recon + total_commit) / n:.4f}"
                f"  p_unique={total_unique / n:.3f}"
            )

        if args.save_every > 0 and epoch % args.save_every == 0:
            ckpt = os.path.join(args.output_dir, f"epoch_{epoch}.pt")
            model._fitted = True
            model.save(ckpt)
            print(f"Saved checkpoint → {ckpt}")

    model._fitted = True
    final_path = os.path.join(args.output_dir, "final.pt")
    model.save(final_path)
    print(f"Saved final model → {final_path}")
    return model


if __name__ == "__main__":
    torch.manual_seed(0)
    args = parse_args()
    model = train(args)

    # Sanity check
    dataset = ItemData(root=args.root, split=args.split, train_test_split=args.train_test_split)
    model.eval()
    device = next(model.parameters()).device
    tokens = model.encode(dataset.item_data[:4].float().to(device))
    recon = model.decode(tokens)
    print(f"\nEncoded shape : {tokens.codes.shape}")
    print(f"Decoded shape : {recon.shape}")
    print(f"Token tuples  : {tokens.to_tuple_ids()}")
