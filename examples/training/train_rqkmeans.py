"""Train an RQKMeansTokenizer on Amazon item embeddings.

Usage:
    python examples/training/train_rqkmeans.py \
        --root data/amazon --split beauty \
        --num_levels 3 --codebook_size 256 \
        --num_epochs 20 --batch_size 640 \
        --save_every 5 --output_dir checkpoints/rqkmeans
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import BatchSampler, DataLoader, RandomSampler

from examples.data.amazon import ItemData
from rectokens.tokenizers.rq_kmeans import RQKMeansTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train RQKMeansTokenizer on Amazon item embeddings")
    p.add_argument("--root", type=str, default="data/amazon")
    p.add_argument("--split", type=str, default="beauty", choices=["beauty", "sports", "toys"])
    p.add_argument("--train_test_split", type=str, default="train", choices=["train", "eval", "all"])
    p.add_argument("--num_levels", type=int, default=3)
    p.add_argument("--codebook_size", type=int, default=256)
    p.add_argument("--num_epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=640)
    p.add_argument("--log_every", type=int, default=1)
    p.add_argument("--save_every", type=int, default=5,
                   help="Save a checkpoint every this many epochs (0 = only save at end)")
    p.add_argument("--output_dir", type=str, default="checkpoints/rqkmeans")
    return p.parse_args()


def train(args: argparse.Namespace) -> RQKMeansTokenizer:
    dataset = ItemData(root=args.root, split=args.split, train_test_split=args.train_test_split)
    print(f"Dataset: {len(dataset)} items, dim={dataset[0].x.shape[0]}")

    input_dim = dataset[0].x.shape[0]
    model = RQKMeansTokenizer(
        num_levels=args.num_levels,
        codebook_size=args.codebook_size,
        dim=input_dim,
    )

    train_sampler = BatchSampler(RandomSampler(dataset), args.batch_size, False)
    loader = DataLoader(
        dataset, sampler=train_sampler, batch_size=None, collate_fn=lambda batch: batch
    )

    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.num_epochs + 1):
        total_recon = 0.0
        total_unique = 0.0

        for batch in loader:
            x = batch.x.float()

            # Mini-batch K-means update (no gradients)
            model.fit_step(x)

            # Evaluate reconstruction and uniqueness on this batch
            with torch.no_grad():
                tokens = model.encode(x)
                recon = model.decode(tokens)
                total_recon += F.mse_loss(recon, x).item()

                codes = tokens.codes  # (B, num_levels)
                eq = (codes.unsqueeze(0) == codes.unsqueeze(1)).all(dim=-1)
                p_unique = (~torch.triu(eq, diagonal=1)).all(dim=1).float().mean()
                total_unique += p_unique.item()

        if epoch % args.log_every == 0:
            n = len(loader)
            print(
                f"epoch {epoch:3d}/{args.num_epochs}"
                f"  recon={total_recon / n:.4f}"
                f"  p_unique={total_unique / n:.3f}"
            )

        if args.save_every > 0 and epoch % args.save_every == 0:
            ckpt = os.path.join(args.output_dir, f"epoch_{epoch}.pt")
            model.save(ckpt)
            print(f"Saved checkpoint → {ckpt}")

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
    tokens = model.encode(dataset.item_data[:4].float())
    recon = model.decode(tokens)
    print(f"\nEncoded shape : {tokens.codes.shape}")
    print(f"Decoded shape : {recon.shape}")
    print(f"Token tuples  : {tokens.to_tuple_ids()}")
