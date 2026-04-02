"""Train an RQKMeansTokenizer on Amazon item embeddings.

Usage:
    python examples/scripts/training/train_rqkmeans.py examples/configs/train_rqkmeans_beauty.gin
"""

from __future__ import annotations

import os

import gin
import torch
import wandb
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

from examples.data.amazon import ItemData
from examples.utils import parse_config
from examples.scripts.training.utils import recon_loss, run_eval
from rectokens.tokenizers.rq_kmeans import RQKMeansTokenizer


@gin.configurable
def train(
    root: str = "data/amazon",
    split: str = "beauty",
    train_test_split: str = "train",
    num_levels: int = 3,
    codebook_size: int = 256,
    num_epochs: int = 20,
    batch_size: int = 640,
    log_every: int = 1,
    save_every: int = 5,
    eval_every_n: int = 0,
    output_dir: str = "checkpoints/rqkmeans",
    wandb_project: str | None = "rqkmeans-training",
) -> RQKMeansTokenizer:
    if wandb_project is not None:
        params = locals()

    dataset = ItemData(root=root, split=split, train_test_split=train_test_split)

    if wandb_project is not None:
        wandb.login()
        run = wandb.init(project=wandb_project, config=params)

    eval_loader = None
    if eval_every_n > 0:
        eval_dataset = ItemData(root=root, split=split, train_test_split="eval")
        eval_sampler = BatchSampler(SequentialSampler(eval_dataset), batch_size, False)
        eval_loader = DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=None,
            collate_fn=lambda batch: batch,
        )

    input_dim = dataset[0].x.shape[0]
    model = RQKMeansTokenizer(
        num_levels=num_levels,
        codebook_size=codebook_size,
        dim=input_dim,
    )

    train_sampler = BatchSampler(RandomSampler(dataset), batch_size, False)
    loader = DataLoader(
        dataset, sampler=train_sampler, batch_size=None, collate_fn=lambda batch: batch
    )

    os.makedirs(output_dir, exist_ok=True)

    pbar = tqdm(range(1, num_epochs + 1), desc="train")
    for epoch in pbar:
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
                total_recon += recon_loss(recon, x).item()

                codes = tokens.codes  # (B, num_levels)
                eq = (codes.unsqueeze(0) == codes.unsqueeze(1)).all(dim=-1)
                p_unique = (~torch.triu(eq, diagonal=1)).all(dim=1).float().mean()
                total_unique += p_unique.item()

        if epoch % log_every == 0:
            n = len(loader)
            avg_recon = total_recon / n
            avg_p_unique = total_unique / n
            pbar.set_postfix(recon=f"{avg_recon:.4f}", p_unique=f"{avg_p_unique:.3f}")
            if wandb_project is not None:
                wandb.log(
                    {"train/recon": avg_recon, "train/p_unique": avg_p_unique},
                    step=epoch,
                )

        if eval_every_n > 0 and epoch % eval_every_n == 0:

            def _step(x):
                tokens = model.encode(x)
                recon = model.decode(tokens)
                return recon_loss(recon, x), None, tokens.codes

            stats = run_eval(eval_loader, _step, num_levels, codebook_size)
            avg_recon = stats["avg_recon"]
            pbar.set_postfix(
                eval_recon=f"{avg_recon:.4f}",
                eval_p_unique=f"{stats['p_unique']:.3f}",
            )
            if wandb_project is not None:
                eval_log = {
                    "eval/recon": avg_recon,
                    "eval/p_unique": stats["p_unique"],
                }
                for lvl, e in enumerate(stats["entropies"]):
                    eval_log[f"eval/entropy_lvl{lvl}"] = e
                wandb.log(eval_log, step=epoch)

        if save_every > 0 and epoch % save_every == 0:
            ckpt = os.path.join(output_dir, f"epoch_{epoch}.pt")
            model.save(ckpt)

    final_path = os.path.join(output_dir, "final.pt")
    model.save(final_path)

    if wandb_project is not None:
        wandb.finish()

    return model


if __name__ == "__main__":
    torch.manual_seed(0)
    parse_config()
    train()
