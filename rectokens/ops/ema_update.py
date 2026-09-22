"""Python dispatch layer for the fused EMA-update operation.

Selects the Triton kernel on CUDA and falls back to pure PyTorch on CPU.
Callers should use :func:`ema_update` exclusively; the ``_cuda_*`` helper is an
implementation detail.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

IS_GPU_AVAILABLE = torch.cuda.is_available()

if IS_GPU_AVAILABLE:
    from rectokens.kernels.ema_update import ema_update_kernel


def _cuda_ema_update(
    x: torch.Tensor,
    codes: torch.Tensor,
    ema_cluster_size: torch.Tensor,
    ema_embed_sum: torch.Tensor,
    codebook: torch.Tensor,
    steps_since_active: torch.Tensor,
    decay: float,
    restart_after_steps: int,
) -> None:
    """In-place EMA codebook update via the fused Triton kernel.

    All tensor arguments are mutated in-place.  ``x`` and ``codes`` are
    read-only inputs; the remaining four tensors are updated buffers.

    Args:
        x: Encoder outputs, shape ``(B, D)``, float32, CUDA, contiguous.
        codes: Nearest-code assignments, shape ``(B,)``, int64, CUDA.
        ema_cluster_size: EMA cluster-size buffer, shape ``(K,)``.
        ema_embed_sum: EMA embedding-sum buffer, shape ``(K, D)``.
        codebook: Codebook embedding matrix, shape ``(K, D)``.
        steps_since_active: Consecutive-inactive-step counter, shape ``(K,)``.
        decay: EMA decay factor γ.
        restart_after_steps: Dead-code restart threshold.
    """
    B, D = x.shape
    K = ema_cluster_size.shape[0]

    # Pre-draw one random batch index per codebook entry for dead-code
    # replacement.  Done on the host to keep the kernel deterministic and
    # avoid random-state complexity inside the Triton JIT.
    rand_idx = torch.randint(B, (K,), device=x.device, dtype=torch.int64)

    # Ensure contiguous layout (kernel assumes unit inner stride for the
    # D-dimension).
    x = x.contiguous()
    codes = codes.contiguous()

    grid = (K,)
    ema_update_kernel[grid](
        x_ptr=x,
        codes_ptr=codes,
        rand_idx_ptr=rand_idx,
        decay=decay,
        restart_after_steps=restart_after_steps,
        B=B,
        K=K,
        D=D,
        x_stride_B=x.stride(0),
        x_stride_D=x.stride(1),
        es_stride_K=ema_embed_sum.stride(0),
        es_stride_D=ema_embed_sum.stride(1),
        cb_stride_K=codebook.stride(0),
        cb_stride_D=codebook.stride(1),
        ema_cluster_size_ptr=ema_cluster_size,
        ema_embed_sum_ptr=ema_embed_sum,
        codebook_ptr=codebook,
        steps_since_active_ptr=steps_since_active,
    )


def _cpu_ema_update(
    x: torch.Tensor,
    codes: torch.Tensor,
    ema_cluster_size: torch.Tensor,
    ema_embed_sum: torch.Tensor,
    codebook: torch.Tensor,
    steps_since_active: torch.Tensor,
    decay: float,
    restart_after_steps: int,
) -> None:
    """Pure-PyTorch reference implementation (CPU fallback).

    Semantically equivalent to the Triton kernel; used when CUDA is not
    available.
    """
    k = codebook.shape[0]

    one_hot = F.one_hot(codes, num_classes=k).float()  # (B, K)
    cluster_size = one_hot.sum(dim=0)                  # (K,)
    embed_sum = one_hot.t() @ x                        # (K, D)

    active = cluster_size > 0  # (K,) bool

    new_ema_cs = decay * ema_cluster_size + (1 - decay) * cluster_size
    ema_cluster_size.copy_(
        torch.where(active, new_ema_cs, ema_cluster_size)
    )

    new_ema_es = decay * ema_embed_sum + (1 - decay) * embed_sum
    ema_embed_sum.copy_(
        torch.where(active.unsqueeze(1), new_ema_es, ema_embed_sum)
    )

    n = ema_cluster_size.clamp(min=1e-5)
    new_embeddings = ema_embed_sum / n.unsqueeze(1)
    codebook.copy_(
        torch.where(active.unsqueeze(1), new_embeddings, codebook)
    )

    new_steps = torch.where(
        active,
        torch.zeros_like(steps_since_active),
        steps_since_active + 1,
    )
    dead = new_steps >= restart_after_steps

    rand_idx = torch.randint(len(x), (k,), device=x.device)
    replacement = x[rand_idx]
    dead_exp = dead.unsqueeze(1)

    codebook.copy_(torch.where(dead_exp, replacement, codebook))
    ema_cluster_size.copy_(
        torch.where(dead, torch.zeros_like(ema_cluster_size), ema_cluster_size)
    )
    ema_embed_sum.copy_(
        torch.where(dead_exp, torch.zeros_like(ema_embed_sum), ema_embed_sum)
    )
    steps_since_active.copy_(
        torch.where(dead, torch.zeros_like(new_steps), new_steps)
    )


def ema_update(
    x: torch.Tensor,
    codes: torch.Tensor,
    ema_cluster_size: torch.Tensor,
    ema_embed_sum: torch.Tensor,
    codebook: torch.Tensor,
    steps_since_active: torch.Tensor,
    decay: float,
    restart_after_steps: int,
) -> None:
    """Fused EMA codebook update — dispatches to Triton (CUDA) or PyTorch (CPU).

    Mutates ``ema_cluster_size``, ``ema_embed_sum``, ``codebook``, and
    ``steps_since_active`` in-place.

    Args:
        x: Encoder outputs of shape ``(B, D)``.
        codes: Nearest-code indices of shape ``(B,)``, int64.
        ema_cluster_size: EMA cluster-size buffer of shape ``(K,)``.
        ema_embed_sum: EMA embedding-sum buffer of shape ``(K, D)``.
        codebook: Codebook embedding matrix of shape ``(K, D)``.
        steps_since_active: Consecutive inactive-step counter of shape ``(K,)``.
        decay: EMA decay factor γ ∈ (0, 1).
        restart_after_steps: Replace a code once it has gone this many
            consecutive steps without any assignment.
    """
    if x.is_cuda:
        _cuda_ema_update(
            x, codes,
            ema_cluster_size, ema_embed_sum,
            codebook, steps_since_active,
            decay, restart_after_steps,
        )
    else:
        _cpu_ema_update(
            x, codes,
            ema_cluster_size, ema_embed_sum,
            codebook, steps_since_active,
            decay, restart_after_steps,
        )
