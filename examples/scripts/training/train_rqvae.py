"""Train an RQVAETokenizer on Amazon item embeddings.

Usage:
    python examples/scripts/training/train_rqvae.py examples/configs/pretraining/train_rqvae_beauty.gin
"""

from __future__ import annotations

import os

import gin
import torch
import wandb
from torch.optim import AdamW
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

from examples.data.amazon import ItemData
from examples.utils import parse_config
from examples.scripts.training.utils import recon_loss, run_eval
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
    eval_every_n: int = 0,
    output_dir: str = "checkpoints/rqvae",
    learnable_codebook: bool = False,
    wandb_project: str | None = "rqvae-training",
    profile: bool = False,
) -> RQVAETokenizer:
    if wandb_project is not None:
        params = locals()

    device = get_device()

    if wandb_project is not None:
        wandb.login()
        run = wandb.init(project=wandb_project, config=params)

    dataset = ItemData(root=root, split=split, train_test_split=train_test_split)

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
    model = RQVAETokenizer(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_levels=num_levels,
        codebook_size=codebook_size,
        learnable_codebook=learnable_codebook,
    ).to(device)

    # AdamW only sees encoder + decoder weights — codebook embeddings are
    # registered as non-trainable buffers and updated via EMA in forward().
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_sampler = BatchSampler(RandomSampler(dataset), batch_size, False)
    loader = DataLoader(
        dataset, sampler=train_sampler, batch_size=None, collate_fn=lambda batch: batch
    )

    os.makedirs(output_dir, exist_ok=True)

    profiler = None
    if profile:
        profile_dir = os.path.join(output_dir, "profile")
        # wait=1 step, warmup=2 steps, active=3 steps — 6 steps total then stop.
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=2, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        profiler.start()
        print(f"[profile] Tracing enabled — trace will be written to {profile_dir}/")

    profile_step = 0
    _PROFILE_STEPS = 6  # wait(1) + warmup(2) + active(3)

    model.train()
    pbar = tqdm(range(1, num_epochs + 1), desc="train")
    for epoch in pbar:
        total_recon = torch.zeros(1, device=device)
        total_commit = torch.zeros(1, device=device)
        total_unique = torch.zeros(1, device=device)

        for batch in loader:
            x = batch.x.float().to(device)
            optimizer.zero_grad()

            out = model(x)

            r_loss = recon_loss(out.recon, x)
            commit_loss = out.commitment_loss
            (r_loss + commit_loss).backward()
            optimizer.step()

            total_recon += r_loss.detach()
            total_commit += commit_loss.detach()
            total_unique += out.p_unique_ids.detach()

            if profiler is not None:
                profiler.step()
                profile_step += 1
                if profile_step >= _PROFILE_STEPS:
                    profiler.stop()
                    print(
                        f"[profile] Done. Open trace with: tensorboard --logdir {profile_dir}"
                    )
                    return model

        if epoch % log_every == 0:
            n = len(loader)
            avg_recon = total_recon.item() / n
            avg_commit = total_commit.item() / n
            avg_p_unique = total_unique.item() / n
            pbar.set_postfix(
                recon=f"{avg_recon:.4f}",
                commit=f"{avg_commit:.4f}",
                p_unique=f"{avg_p_unique:.3f}",
            )
            if wandb_project is not None:
                wandb.log(
                    {
                        "train/recon": avg_recon,
                        "train/commit": avg_commit,
                        "train/total": avg_recon + avg_commit,
                        "train/p_unique": avg_p_unique,
                    },
                    step=epoch,
                )

        if eval_every_n > 0 and epoch % eval_every_n == 0:
            model.eval()

            def _step(x):
                x = x.to(device)
                out = model(x)
                return recon_loss(out.recon, x), out.commitment_loss, out.codes

            stats = run_eval(eval_loader, _step, num_levels, codebook_size)
            model.train()
            avg_recon, avg_commit = stats["avg_recon"], stats["avg_commit"]
            pbar.set_postfix(
                eval_recon=f"{avg_recon:.4f}",
                eval_commit=f"{avg_commit:.4f}",
                eval_p_unique=f"{stats['p_unique']:.3f}",
            )
            if wandb_project is not None:
                eval_log = {
                    "eval/recon": avg_recon,
                    "eval/commit": avg_commit,
                    "eval/total": avg_recon + avg_commit,
                    "eval/p_unique": stats["p_unique"],
                }
                for lvl, e in enumerate(stats["entropies"]):
                    eval_log[f"eval/entropy_lvl{lvl}"] = e
                wandb.log(eval_log, step=epoch)

        if save_every > 0 and epoch % save_every == 0:
            ckpt = os.path.join(output_dir, f"epoch_{epoch}.pt")
            model._fitted = True
            model.save(ckpt)

    model._fitted = True
    final_path = os.path.join(output_dir, "final.pt")
    model.save(final_path)

    if wandb_project is not None:
        wandb.finish()

    return model


if __name__ == "__main__":
    torch.manual_seed(0)
    parse_config()
    train()
