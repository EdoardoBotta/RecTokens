import time
from typing import Optional

import torch

assert torch.cuda.is_available(), "CUDA is required to import ELL constrained node transition kernels."

import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

from rectokens.kernels.constrained_node_transition import (
    _select_branch,
    _FUSED_MIN_BLOCK_BRANCHES,
)


# ─────────────────────────────────────────────────────────────────────────────
# ELL-specific autotune configs
# ─────────────────────────────────────────────────────────────────────────────

# Separate from _FUSED_AUTOTUNE_CONFIGS: adds BLOCK_B=32 to prevent register
# spilling at large batch sizes, and explicit num_warps for occupancy tuning.
_ELL_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_B": 64,  "BLOCK_K": 64,  "BLOCK_BRANCHES": 4},  num_warps=4),
    triton.Config({"BLOCK_B": 128, "BLOCK_K": 64,  "BLOCK_BRANCHES": 4},  num_warps=4),
    triton.Config({"BLOCK_B": 256, "BLOCK_K": 64,  "BLOCK_BRANCHES": 4},  num_warps=4),
    triton.Config({"BLOCK_B": 64,  "BLOCK_K": 128, "BLOCK_BRANCHES": 8},  num_warps=4),
    triton.Config({"BLOCK_B": 128, "BLOCK_K": 128, "BLOCK_BRANCHES": 8},  num_warps=4),
    triton.Config({"BLOCK_B": 64,  "BLOCK_K": 64,  "BLOCK_BRANCHES": 16}, num_warps=4),
    triton.Config({"BLOCK_B": 128, "BLOCK_K": 64,  "BLOCK_BRANCHES": 16}, num_warps=4),
]


# ─────────────────────────────────────────────────────────────────────────────
# Shared Triton device-function helpers
# ─────────────────────────────────────────────────────────────────────────────


@triton.jit
def _compute_ell_branch_logits(
    offs_B,
    offs_BR,
    b_mask,
    cur_node,
    branch_cols,   # [BLOCK_B, BLOCK_BRANCHES] — pre-loaded by caller
    branch_valid,  # [BLOCK_B, BLOCK_BRANCHES] — pre-computed by caller
    a_ptr,
    b_ptr,
    bias_ptr,
    ell_cols_vals_ptr,
    a_stride_B,
    a_stride_K,
    b_stride_K,
    b_stride_N,
    ell_node_stride,  # ell_cols_vals.stride(0) = 2 * max_branches
    ell_cv_stride,    # ell_cols_vals.stride(1) = max_branches (cols→vals gap within a node)
    K: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_BRANCHES: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """Compute per-branch dot-product logits from ELL format.

    branch_cols and branch_valid are pre-loaded by the caller so validity can be
    checked before this function is called, enabling an early exit for empty tiles.

    ell_node_base [BLOCK_B] avoids the former ell_row_offsets [BLOCK_B, BLOCK_BRANCHES]
    intermediate, reducing register footprint by BLOCK_BRANCHES×.
    """
    ell_node_base = cur_node.to(tl.int64) * ell_node_stride  # [BLOCK_B]

    branch_vals = tl.load(
        ell_cols_vals_ptr + ell_cv_stride + ell_node_base[:, None] + offs_BR[None, :].to(tl.int64),
        mask=branch_valid,
        other=-1,
    )  # [BLOCK_B, BLOCK_BRANCHES]

    logits = tl.zeros((BLOCK_B, BLOCK_BRANCHES), dtype=tl.float32)

    for k_tile in range(0, tl.cdiv(K, BLOCK_K)):
        offs_K = k_tile * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = offs_K < K

        a_chunk = tl.load(
            a_ptr + offs_B[:, None] * a_stride_B + offs_K[None, :] * a_stride_K,
            mask=b_mask[:, None] & k_mask[None, :],
            other=0.0,
        )  # [BLOCK_B, BLOCK_K]

        for local_br in tl.static_range(BLOCK_BRANCHES):
            br_sel, col_k, c_mask = _select_branch(
                local_br, branch_cols, branch_valid, BLOCK_BRANCHES
            )
            b_chunk = tl.load(
                b_ptr + offs_K[None, :] * b_stride_K + col_k[:, None] * b_stride_N,
                mask=c_mask[:, None] & k_mask[None, :],
                other=0.0,
            )  # [BLOCK_B, BLOCK_K]
            dot = tl.sum(a_chunk * b_chunk, axis=1)  # [BLOCK_B]
            logits = tl.where(br_sel[None, :], logits + dot[:, None], logits)

    if HAS_BIAS:
        for local_br in tl.static_range(BLOCK_BRANCHES):
            br_sel, col_k, c_mask = _select_branch(
                local_br, branch_cols, branch_valid, BLOCK_BRANCHES
            )
            bias_k = tl.load(bias_ptr + col_k, mask=c_mask, other=0.0)
            logits = tl.where(br_sel[None, :], logits + bias_k[:, None], logits)

    return branch_vals, logits


@triton.jit
def _ell_fused_prologue(
    cur_node_ptr,
    ell_n_children_ptr,
    a_ptr,
    b_ptr,
    bias_ptr,
    ell_cols_vals_ptr,
    next_node_ptr,
    valid_idxs_ptr,
    a_stride_B,
    a_stride_K,
    b_stride_K,
    b_stride_N,
    ell_node_stride,
    ell_cv_stride,
    next_node_stride_B,
    next_node_stride_N,
    valid_idxs_stride_B,
    valid_idxs_stride_N,
    max_branches: tl.constexpr,
    B: tl.constexpr,
    K: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_BRANCHES: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """Shared ELL prologue: pid setup, branch-cols load, validity check, logit compute, stores.

    Uses n_children as a block-level gate: if no batch item has a child in
    [pid_BR*BLOCK_BRANCHES, ...), the ell_cols_vals load is skipped entirely, avoiding
    HBM traffic for padding blocks when max_branches is large. Within valid blocks,
    per-lane validity is derived from the -1 sentinel in branch_cols.

    Returns any_valid so the calling kernel can skip its own work (gumbel sampling /
    logit stores) when the block has no valid branches.
    """
    pid_B = tl.program_id(axis=0)
    pid_BR = tl.program_id(axis=1)

    offs_B = pid_B * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_BR = pid_BR * BLOCK_BRANCHES + tl.arange(0, BLOCK_BRANCHES)
    b_mask = offs_B < B

    cur_node = tl.load(cur_node_ptr + offs_B, mask=b_mask, other=-1)
    n_children = tl.load(ell_n_children_ptr + cur_node, mask=cur_node >= 0, other=0)

    # Block-level gate: skip ELL load if no batch item has a child in this branch slice.
    # n_children is a compact (num_nodes,) int tensor that fits in L2; checking it here
    # avoids reading the much larger ell_cols_vals for blocks that are entirely padding.
    block_br_start = pid_BR * BLOCK_BRANCHES
    n_in_range = tl.sum((n_children > block_br_start).to(tl.int32))
    if n_in_range > 0:
        ell_node_base = cur_node.to(tl.int64) * ell_node_stride
        load_mask = b_mask[:, None] & (offs_BR[None, :] < max_branches)
        branch_cols = tl.load(
            ell_cols_vals_ptr + ell_node_base[:, None] + offs_BR[None, :].to(tl.int64),
            mask=load_mask,
            other=-1,
        )  # [BLOCK_B, BLOCK_BRANCHES]
        branch_valid = b_mask[:, None] & (branch_cols >= 0)  # -1 sentinel marks padding

        any_valid = tl.sum(branch_valid.to(tl.int32)) > 0
        if any_valid:
            branch_vals, logits = _compute_ell_branch_logits(
                offs_B, offs_BR, b_mask, cur_node, branch_cols, branch_valid,
                a_ptr, b_ptr, bias_ptr, ell_cols_vals_ptr,
                a_stride_B, a_stride_K, b_stride_K, b_stride_N,
                ell_node_stride, ell_cv_stride,
                K, BLOCK_B, BLOCK_K, BLOCK_BRANCHES, HAS_BIAS,
            )
            store_mask = b_mask[:, None] & (offs_BR[None, :] < max_branches)
            tl.store(
                next_node_ptr
                + offs_B[:, None] * next_node_stride_B
                + offs_BR[None, :] * next_node_stride_N,
                branch_vals,
                mask=store_mask,
            )
            tl.store(
                valid_idxs_ptr
                + offs_B[:, None] * valid_idxs_stride_B
                + offs_BR[None, :] * valid_idxs_stride_N,
                branch_cols,
                mask=store_mask,
            )
        else:
            logits = tl.zeros([BLOCK_B, BLOCK_BRANCHES], dtype=tl.float32)
    else:
        branch_cols = tl.full([BLOCK_B, BLOCK_BRANCHES], -1, dtype=tl.int64)
        branch_valid = branch_cols >= 0  # all False
        logits = tl.zeros([BLOCK_B, BLOCK_BRANCHES], dtype=tl.float32)
        any_valid = n_in_range > 0  # False

    return pid_BR, offs_B, offs_BR, b_mask, branch_cols, branch_valid, logits, any_valid


# ─────────────────────────────────────────────────────────────────────────────
# Fused sparse linear + Gumbel-max sampling (ELL)
# ─────────────────────────────────────────────────────────────────────────────


@triton_op("vtnk_ell::_fused_linear_constrained_node_transition_sampling_op", mutates_args={})
def _ell_fused_linear_constrained_node_transition_sampling_op(
    a: torch.Tensor,
    b: torch.Tensor,
    bias_val: torch.Tensor,
    cur_node: torch.Tensor,
    ell_cols_vals: torch.Tensor,
    ell_n_children: torch.Tensor,
    max_branches: int,
    has_bias: bool,
    rng_seed: Optional[int] = None,
    temperature: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if rng_seed is None:
        rng_seed = time.time_ns() & 0x7FFFFFFF
    if temperature is None or temperature == 1.0:
        temperature = torch.ones(1, dtype=torch.float32, device=a.device)
    elif isinstance(temperature, float):
        temperature = torch.tensor(temperature, dtype=torch.float32, device=a.device)

    B, K = a.shape

    assert cur_node.shape == (B,), f"Expected cur_node shape ({B},), got {cur_node.shape}"

    a = a.contiguous()
    cur_node = cur_node.contiguous()
    ell_cols_vals = ell_cols_vals.contiguous()
    ell_n_children = ell_n_children.contiguous()
    bias_val = bias_val.contiguous()

    next_node = cur_node.new_full((B, max_branches), -1)
    valid_idxs = cur_node.new_full((B, max_branches), -1)
    max_br_blocks = triton.cdiv(max_branches, _FUSED_MIN_BLOCK_BRANCHES)
    gumbel_block_max = torch.full(
        (B, max_br_blocks), float("-inf"), dtype=torch.float32, device=a.device
    )
    block_sample_buf = torch.full(
        (B, max_br_blocks), -1.0, dtype=torch.float32, device=a.device
    )

    grid = lambda meta: (
        triton.cdiv(B, meta["BLOCK_B"]),
        triton.cdiv(max_branches, meta["BLOCK_BRANCHES"]),
    )
    wrap_triton(_ell_fused_sampling_kernel)[grid](
        a_ptr=a,
        b_ptr=b,
        bias_ptr=bias_val,
        cur_node_ptr=cur_node,
        ell_cols_vals_ptr=ell_cols_vals,
        ell_n_children_ptr=ell_n_children,
        temperature_ptr=temperature,
        a_stride_B=a.stride(0),
        a_stride_K=a.stride(1),
        b_stride_K=b.stride(0),
        b_stride_N=b.stride(1),
        ell_node_stride=ell_cols_vals.stride(0),
        ell_cv_stride=ell_cols_vals.stride(1),
        next_node_ptr=next_node,
        valid_idxs_ptr=valid_idxs,
        gumbel_block_max_ptr=gumbel_block_max,
        block_sample_ptr=block_sample_buf,
        next_node_stride_B=next_node.stride(0),
        next_node_stride_N=next_node.stride(1),
        valid_idxs_stride_B=valid_idxs.stride(0),
        valid_idxs_stride_N=valid_idxs.stride(1),
        max_br_blocks=max_br_blocks,
        rng_seed=rng_seed,
        B=B,
        K=K,
        max_branches=max_branches,
        HAS_BIAS=has_bias,
    )

    argmax = torch.max(gumbel_block_max, dim=1).indices  # [B]
    sample = block_sample_buf.gather(1, argmax.unsqueeze(1)).squeeze(1)
    return next_node, valid_idxs, sample


@triton.autotune(
    configs=_ELL_AUTOTUNE_CONFIGS,
    key=["B", "K", "max_branches"],
    restore_value=["next_node_ptr", "valid_idxs_ptr", "gumbel_block_max_ptr", "block_sample_ptr"],
)
@triton.jit
def _ell_fused_sampling_kernel(
    # Inputs
    a_ptr,
    b_ptr,
    bias_ptr,
    cur_node_ptr,
    ell_cols_vals_ptr,
    ell_n_children_ptr,
    temperature_ptr,
    a_stride_B,
    a_stride_K,
    b_stride_K,
    b_stride_N,
    ell_node_stride,
    ell_cv_stride,
    # Outputs
    next_node_ptr,
    valid_idxs_ptr,
    gumbel_block_max_ptr,
    block_sample_ptr,
    next_node_stride_B,
    next_node_stride_N,
    valid_idxs_stride_B,
    valid_idxs_stride_N,
    max_br_blocks,
    rng_seed,
    # Constants
    B: tl.constexpr,
    K: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_BRANCHES: tl.constexpr,
    max_branches: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid_BR, offs_B, offs_BR, b_mask, branch_cols, branch_valid, logits, any_valid = _ell_fused_prologue(
        cur_node_ptr, ell_n_children_ptr,
        a_ptr, b_ptr, bias_ptr, ell_cols_vals_ptr,
        next_node_ptr, valid_idxs_ptr,
        a_stride_B, a_stride_K, b_stride_K, b_stride_N,
        ell_node_stride, ell_cv_stride,
        next_node_stride_B, next_node_stride_N,
        valid_idxs_stride_B, valid_idxs_stride_N,
        max_branches, B, K, BLOCK_B, BLOCK_K, BLOCK_BRANCHES, HAS_BIAS,
    )
    if any_valid == 0:
        return

    temperature = tl.load(temperature_ptr)
    u = tl.rand(seed=rng_seed, offset=offs_B[:, None] * max_branches + offs_BR[None, :])
    gumbel = -tl.log(-tl.log(u + 1e-10) + 1e-10)
    g_vals = tl.where(branch_valid, logits / temperature + gumbel, float("-inf"))
    block_max_gumbel = tl.max(g_vals, axis=1)  # [BLOCK_B]

    winner_idx = tl.argmax(g_vals, axis=1)  # [BLOCK_B]
    br_sel = tl.arange(0, BLOCK_BRANCHES)[None, :] == winner_idx[:, None]
    block_sample = tl.sum(
        tl.where(br_sel & branch_valid, branch_cols.to(tl.float32), 0.0), axis=1
    )

    tl.store(
        gumbel_block_max_ptr + offs_B * max_br_blocks + pid_BR,
        block_max_gumbel,
        mask=b_mask,
    )
    tl.store(
        block_sample_ptr + offs_B * max_br_blocks + pid_BR,
        block_sample,
        mask=b_mask & (block_max_gumbel > float("-inf")),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Fused sparse linear + top-K (ELL)
# ─────────────────────────────────────────────────────────────────────────────


@triton_op("vtnk_ell::_fused_linear_constrained_node_transition_topk_op", mutates_args={})
def _ell_fused_linear_constrained_node_transition_topk_op(
    a: torch.Tensor,
    b: torch.Tensor,
    bias_val: torch.Tensor,
    cur_node: torch.Tensor,
    ell_cols_vals: torch.Tensor,
    ell_n_children: torch.Tensor,
    max_branches: int,
    has_bias: bool,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, K = a.shape

    assert cur_node.shape == (B,), f"Expected cur_node shape ({B},), got {cur_node.shape}"

    a = a.contiguous()
    cur_node = cur_node.contiguous()
    ell_cols_vals = ell_cols_vals.contiguous()
    ell_n_children = ell_n_children.contiguous()
    bias_val = bias_val.contiguous()

    next_node = cur_node.new_full((B, max_branches), -1)
    valid_idxs = cur_node.new_full((B, max_branches), -1)
    branch_logits = torch.full(
        (B, max_branches), float("-inf"), dtype=torch.float32, device=a.device
    )

    grid = lambda meta: (
        triton.cdiv(B, meta["BLOCK_B"]),
        triton.cdiv(max_branches, meta["BLOCK_BRANCHES"]),
    )
    wrap_triton(_ell_fused_compact_kernel)[grid](
        a_ptr=a,
        b_ptr=b,
        bias_ptr=bias_val,
        cur_node_ptr=cur_node,
        ell_cols_vals_ptr=ell_cols_vals,
        ell_n_children_ptr=ell_n_children,
        a_stride_B=a.stride(0),
        a_stride_K=a.stride(1),
        b_stride_K=b.stride(0),
        b_stride_N=b.stride(1),
        ell_node_stride=ell_cols_vals.stride(0),
        ell_cv_stride=ell_cols_vals.stride(1),
        next_node_ptr=next_node,
        valid_idxs_ptr=valid_idxs,
        branch_logits_ptr=branch_logits,
        next_node_stride_B=next_node.stride(0),
        next_node_stride_N=next_node.stride(1),
        valid_idxs_stride_B=valid_idxs.stride(0),
        valid_idxs_stride_N=valid_idxs.stride(1),
        B=B,
        K=K,
        max_branches=max_branches,
        HAS_BIAS=has_bias,
    )

    if k >= max_branches:
        return next_node, valid_idxs, branch_logits, valid_idxs.clone()
    topk_logits, topk_branch_idxs = torch.topk(branch_logits, k, dim=-1)
    topk_idxs = valid_idxs.gather(1, topk_branch_idxs)
    return next_node, valid_idxs, topk_logits, topk_idxs


@triton.autotune(
    configs=_ELL_AUTOTUNE_CONFIGS,
    key=["B", "K", "max_branches"],
    restore_value=["next_node_ptr", "valid_idxs_ptr", "branch_logits_ptr"],
)
@triton.jit
def _ell_fused_compact_kernel(
    # Inputs
    a_ptr,
    b_ptr,
    bias_ptr,
    cur_node_ptr,
    ell_cols_vals_ptr,
    ell_n_children_ptr,
    a_stride_B,
    a_stride_K,
    b_stride_K,
    b_stride_N,
    ell_node_stride,
    ell_cv_stride,
    # Outputs
    next_node_ptr,
    valid_idxs_ptr,
    branch_logits_ptr,
    next_node_stride_B,
    next_node_stride_N,
    valid_idxs_stride_B,
    valid_idxs_stride_N,
    # Constants
    B: tl.constexpr,
    K: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_BRANCHES: tl.constexpr,
    max_branches: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    _, offs_B, offs_BR, b_mask, _, branch_valid, logits, any_valid = _ell_fused_prologue(
        cur_node_ptr, ell_n_children_ptr,
        a_ptr, b_ptr, bias_ptr, ell_cols_vals_ptr,
        next_node_ptr, valid_idxs_ptr,
        a_stride_B, a_stride_K, b_stride_K, b_stride_N,
        ell_node_stride, ell_cv_stride,
        next_node_stride_B, next_node_stride_N,
        valid_idxs_stride_B, valid_idxs_stride_N,
        max_branches, B, K, BLOCK_B, BLOCK_K, BLOCK_BRANCHES, HAS_BIAS,
    )
    if any_valid == 0:
        return

    store_mask = b_mask[:, None] & (offs_BR[None, :] < max_branches)
    tl.store(
        branch_logits_ptr + offs_B[:, None] * max_branches + offs_BR[None, :],
        tl.where(branch_valid, logits, float("-inf")),
        mask=store_mask,
    )
