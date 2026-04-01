"""Shared training utilities for RQ tokenizer scripts."""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def recon_loss(recon: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Per-sample sum of squared errors, averaged over the batch."""
    return F.mse_loss(recon, x, reduction="none").sum(dim=-1).mean()


def run_eval(
    eval_loader: DataLoader,
    step_fn: Callable[
        [torch.Tensor], tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]
    ],
    num_levels: int,
    codebook_size: int,
) -> dict:
    """Run evaluation over eval_loader.

    Args:
        eval_loader: DataLoader yielding batches with a .x attribute.
        step_fn: Takes a float tensor x and returns
            ``(recon_loss, commit_loss_or_None, codes)``.
        num_levels: Number of RQ codebook levels.
        codebook_size: Size of each codebook.

    Returns:
        Dict with keys ``avg_recon``, ``avg_commit`` (None if unused),
        ``p_unique``, and ``entropies``.
    """
    all_codes = []
    total_recon = 0.0
    total_commit = 0.0
    has_commit = False
    n_batches = 0

    with torch.no_grad():
        for batch in eval_loader:
            x = batch.x.float()
            r_loss, c_loss, codes = step_fn(x)
            total_recon += r_loss.item()
            if c_loss is not None:
                total_commit += c_loss.item()
                has_commit = True
            all_codes.append(codes.cpu())
            n_batches += 1

    codes = torch.cat(all_codes, dim=0)  # (N_eval, num_levels)
    N = len(codes)

    n_unique = len(set(map(tuple, codes.tolist())))
    p_unique = n_unique / N

    entropies = []
    for lvl in range(num_levels):
        counts = torch.bincount(codes[:, lvl], minlength=codebook_size).float()
        p = counts / N
        log_p = torch.where(p > 0, p.log(), torch.zeros_like(p))
        entropies.append(-(p * log_p).sum().item())

    return {
        "avg_recon": total_recon / n_batches,
        "avg_commit": total_commit / n_batches if has_commit else None,
        "p_unique": p_unique,
        "entropies": entropies,
    }
