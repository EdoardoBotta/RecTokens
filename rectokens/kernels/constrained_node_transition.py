import time
from typing import Optional

import torch

assert torch.cuda.is_available(), "CUDA is required to import VTNK kernel."

import triton
import triton.language as tl

from torch.library import triton_op, wrap_triton


@triton_op("vtnk::_constrained_node_transition_op", mutates_args={})
def _constrained_node_transition_op(
    logits: torch.Tensor,
    cur_node: torch.Tensor,
    csr_row_ptrs: torch.Tensor,
    csr_cols_vals: torch.Tensor,
    max_branches: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, N = logits.shape

    assert cur_node.shape == (B,), (
        f"Expected cur_node shape ({B},), got {cur_node.shape}"
    )

    logits = logits.contiguous()
    cur_node = cur_node.contiguous()
    csr_cols_vals = csr_cols_vals.contiguous()

    corrected_logits = torch.empty_like(logits)
    next_node = cur_node.new_empty(B, max_branches)
    valid_idxs = cur_node.new_empty(B, max_branches)

    grid = lambda meta: (
        triton.cdiv(B, meta["BLOCK_B"]) * triton.cdiv(N, meta["BLOCK_N"]),
    )
    wrap_triton(_constrained_node_transition_kernel)[grid](
        logits_ptr=logits,
        cur_node_ptr=cur_node,
        csr_trie_row_ptr=csr_row_ptrs,
        csr_trie_cols_vals_ptr=csr_cols_vals,
        logits_stride_B=logits.stride(0),
        logits_stride_N=logits.stride(1),
        cols_vals_stride_0=csr_cols_vals.stride(0),
        corrected_logits_ptr=corrected_logits,
        next_node_ptr=next_node,
        valid_idxs_ptr=valid_idxs,
        corrected_logits_stride_B=corrected_logits.stride(0),
        corrected_logits_stride_N=corrected_logits.stride(1),
        next_node_stride_B=next_node.stride(0),
        next_node_stride_N=next_node.stride(1),
        valid_idxs_stride_B=valid_idxs.stride(0),
        valid_idxs_stride_N=valid_idxs.stride(1),
        B=B,
        N=N,
        max_branches=max_branches,
    )

    return next_node, valid_idxs, corrected_logits


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_B": 32, "BLOCK_N": 128, "GROUP_SIZE_M": 4}),
        triton.Config({"BLOCK_B": 64, "BLOCK_N": 64, "GROUP_SIZE_M": 4}),
        triton.Config({"BLOCK_B": 64, "BLOCK_N": 128, "GROUP_SIZE_M": 4}),
        triton.Config({"BLOCK_B": 128, "BLOCK_N": 64, "GROUP_SIZE_M": 4}),
        triton.Config({"BLOCK_B": 128, "BLOCK_N": 128, "GROUP_SIZE_M": 8}),
    ],
    key=["B", "N"],
    restore_value=["corrected_logits_ptr", "next_node_ptr", "valid_idxs_ptr"],
)
@triton.jit
def _constrained_node_transition_kernel(
    # Inputs
    logits_ptr,
    cur_node_ptr,
    csr_trie_row_ptr,
    csr_trie_cols_vals_ptr,
    logits_stride_B,
    logits_stride_N,
    cols_vals_stride_0,
    # Outputs
    corrected_logits_ptr,
    next_node_ptr,
    valid_idxs_ptr,
    corrected_logits_stride_B,
    corrected_logits_stride_N,
    next_node_stride_B,
    next_node_stride_N,
    valid_idxs_stride_B,
    valid_idxs_stride_N,
    # Constants
    B: tl.constexpr,
    N: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    max_branches,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(B, BLOCK_B)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_B = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_N = (pid % num_pid_in_group) // group_size_m

    offs_B = pid_B * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_N = pid_N * BLOCK_N + tl.arange(0, BLOCK_N)

    cur_node_ptrs = cur_node_ptr + offs_B
    logits_ptrs = (
        logits_ptr
        + offs_B[:, None] * logits_stride_B
        + offs_N[None, :] * logits_stride_N
    )

    logits_mask = (offs_B[:, None] < B) & (offs_N[None, :] < N)
    logits = tl.load(logits_ptrs, mask=logits_mask, other=float("-inf"))
    cur_node = tl.load(cur_node_ptrs, mask=offs_B < B, other=-1)

    csr_row_ptrs = tl.load(csr_trie_row_ptr + cur_node, mask=cur_node >= 0, other=0)
    csr_next_ptrs = tl.load(
        csr_trie_row_ptr + cur_node + 1, mask=cur_node >= 0, other=0
    )
    n_children = csr_next_ptrs - csr_row_ptrs

    b_valid = offs_B < B
    logits_correction_mask = tl.zeros([BLOCK_B, BLOCK_N], dtype=tl.int1)
    for k in range(max_branches):
        col_k = tl.load(
            csr_trie_cols_vals_ptr + csr_row_ptrs + k,
            mask=b_valid & (n_children > k),
            other=-1,
        )
        logits_correction_mask = logits_correction_mask | (
            tl.reshape(col_k, [BLOCK_B, 1]) == offs_N[None, :]
        )
    corrected_logits = tl.where(logits_correction_mask, logits, float("-inf"))
    tl.store(
        corrected_logits_ptr
        + offs_B[:, None] * corrected_logits_stride_B
        + offs_N[None, :] * corrected_logits_stride_N,
        corrected_logits,
        mask=logits_mask,
    )

    if pid_N == 0:
        b_valid_store = offs_B < B
        for k in range(max_branches):
            child_valid = b_valid_store & (n_children > k)
            col_k = tl.load(
                csr_trie_cols_vals_ptr + csr_row_ptrs + k,
                mask=child_valid,
                other=-1,
            )
            val_k = tl.load(
                csr_trie_cols_vals_ptr + csr_row_ptrs + k + cols_vals_stride_0,
                mask=child_valid,
                other=-1,
            )
            tl.store(
                next_node_ptr + offs_B * next_node_stride_B + k * next_node_stride_N,
                val_k,
                mask=b_valid_store,
            )
            tl.store(
                valid_idxs_ptr + offs_B * valid_idxs_stride_B + k * valid_idxs_stride_N,
                col_k,
                mask=b_valid_store,
            )


_FUSED_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_B": 64, "BLOCK_K": 64, "BLOCK_BRANCHES": 4}),
    triton.Config({"BLOCK_B": 128, "BLOCK_K": 64, "BLOCK_BRANCHES": 4}),
    triton.Config({"BLOCK_B": 256, "BLOCK_K": 64, "BLOCK_BRANCHES": 4}),
    triton.Config({"BLOCK_B": 64, "BLOCK_K": 128, "BLOCK_BRANCHES": 8}),
    triton.Config({"BLOCK_B": 128, "BLOCK_K": 128, "BLOCK_BRANCHES": 8}),
    triton.Config({"BLOCK_B": 64, "BLOCK_K": 64, "BLOCK_BRANCHES": 16}),
    triton.Config({"BLOCK_B": 128, "BLOCK_K": 64, "BLOCK_BRANCHES": 16}),
]


@triton.jit
def _sparse_branch_body(
    branch_idx,
    max_branches,
    offs_B,
    b_mask,
    csr_row_ptrs,
    n_children,
    a_ptr,
    b_ptr,
    bias_ptr,
    csr_trie_cols_vals_ptr,
    cols_vals_stride_0,
    corrected_logits_ptr,
    next_node_ptr,
    valid_idxs_ptr,
    a_stride_B,
    a_stride_K,
    b_stride_K,
    b_stride_N,
    corrected_logits_stride_B,
    corrected_logits_stride_N,
    next_node_stride_B,
    next_node_stride_N,
    valid_idxs_stride_B,
    valid_idxs_stride_N,
    K: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    # Shared inner body for both fused kernels: load one branch from the CSR trie,
    # compute its dot product against the weight matrix, and write the three outputs
    # (corrected_logits, next_node, valid_idxs). Returns (col_k, logit_k, child_mask)
    # so the sampling kernel can apply the Gumbel-max update without re-loading.
    in_range = branch_idx < max_branches
    child_mask = b_mask & (n_children > branch_idx) & in_range

    col_k = tl.load(
        csr_trie_cols_vals_ptr + csr_row_ptrs + branch_idx,
        mask=child_mask,
        other=-1,
    )
    val_k = tl.load(
        csr_trie_cols_vals_ptr + csr_row_ptrs + branch_idx + cols_vals_stride_0,
        mask=child_mask,
        other=-1,
    )

    logit_k = tl.zeros((BLOCK_B,), dtype=tl.float32)
    for k_tile in range(0, tl.cdiv(K, BLOCK_K)):
        offs_K = k_tile * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = offs_K < K
        a_ptrs = a_ptr + offs_B[:, None] * a_stride_B + offs_K[None, :] * a_stride_K
        a_chunk = tl.load(a_ptrs, mask=b_mask[:, None] & k_mask[None, :], other=0.0)
        b_ptrs = b_ptr + offs_K[None, :] * b_stride_K + col_k[:, None] * b_stride_N
        b_chunk = tl.load(
            b_ptrs, mask=child_mask[:, None] & k_mask[None, :], other=0.0
        )
        logit_k += tl.sum(a_chunk * b_chunk, axis=1)

    if HAS_BIAS:
        logit_k += tl.load(bias_ptr + col_k, mask=child_mask, other=0.0)

    tl.store(
        corrected_logits_ptr
        + offs_B * corrected_logits_stride_B
        + col_k * corrected_logits_stride_N,
        logit_k,
        mask=child_mask,
    )
    tl.store(
        next_node_ptr + offs_B * next_node_stride_B + branch_idx * next_node_stride_N,
        val_k,
        mask=b_mask & in_range,
    )
    tl.store(
        valid_idxs_ptr
        + offs_B * valid_idxs_stride_B
        + branch_idx * valid_idxs_stride_N,
        col_k,
        mask=b_mask & in_range,
    )

    return col_k, logit_k, child_mask


@triton.jit
def _gumbel_max_update(
    rng_seed,
    offs_B,
    max_branches,
    branch_idx,
    logit_k,
    temperature,
    child_mask,
    col_k,
    block_sample,
    block_max_gumbel,
):
    # See: https://arxiv.org/pdf/2603.15854
    u = tl.rand(seed=rng_seed, offset=offs_B * max_branches + branch_idx)
    gumbel = -tl.log(-tl.log(u + 1e-10) + 1e-10)
    g_k = tl.where(child_mask, logit_k / temperature + gumbel, float("-inf"))
    improved = g_k > block_max_gumbel
    return (
        tl.where(improved, col_k.to(tl.float32), block_sample),
        tl.where(improved, g_k, block_max_gumbel),
    )


@triton_op("vtnk::_fused_linear_constrained_node_transition_op", mutates_args={})
def _fused_linear_constrained_node_transition_op(
    a: torch.Tensor,
    b: torch.Tensor,
    bias_val: torch.Tensor,
    cur_node: torch.Tensor,
    csr_row_ptrs: torch.Tensor,
    csr_cols_vals: torch.Tensor,
    max_branches: int,
    has_bias: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, K = a.shape
    N = b.shape[1]

    assert cur_node.shape == (B,), f"Expected cur_node shape (B,), got {cur_node.shape}"

    a = a.contiguous()
    cur_node = cur_node.contiguous()
    csr_cols_vals = csr_cols_vals.contiguous()
    bias_val = bias_val.contiguous()

    corrected_logits = torch.full(
        (B, N), float("-inf"), dtype=torch.float32, device=a.device
    )
    next_node = cur_node.new_full((B, max_branches), -1)
    valid_idxs = cur_node.new_full((B, max_branches), -1)

    grid = lambda meta: (
        triton.cdiv(B, meta["BLOCK_B"]),
        triton.cdiv(max_branches, meta["BLOCK_BRANCHES"]),
    )
    wrap_triton(_fused_sparse_linear_constrained_node_transition_kernel)[grid](
        a_ptr=a,
        b_ptr=b,
        bias_ptr=bias_val,
        cur_node_ptr=cur_node,
        csr_trie_row_ptr=csr_row_ptrs,
        csr_trie_cols_vals_ptr=csr_cols_vals,
        a_stride_B=a.stride(0),
        a_stride_K=a.stride(1),
        b_stride_K=b.stride(0),
        b_stride_N=b.stride(1),
        cols_vals_stride_0=csr_cols_vals.stride(0),
        corrected_logits_ptr=corrected_logits,
        next_node_ptr=next_node,
        valid_idxs_ptr=valid_idxs,
        corrected_logits_stride_B=corrected_logits.stride(0),
        corrected_logits_stride_N=corrected_logits.stride(1),
        next_node_stride_B=next_node.stride(0),
        next_node_stride_N=next_node.stride(1),
        valid_idxs_stride_B=valid_idxs.stride(0),
        valid_idxs_stride_N=valid_idxs.stride(1),
        B=B,
        K=K,
        N=N,
        max_branches=max_branches,
        HAS_BIAS=has_bias,
    )

    return next_node, valid_idxs, corrected_logits


@triton.autotune(
    configs=_FUSED_AUTOTUNE_CONFIGS,
    key=["B", "K", "N"],
    restore_value=["corrected_logits_ptr", "next_node_ptr", "valid_idxs_ptr"],
)
@triton.jit
def _fused_sparse_linear_constrained_node_transition_kernel(
    # Inputs
    a_ptr,
    b_ptr,
    bias_ptr,
    cur_node_ptr,
    csr_trie_row_ptr,
    csr_trie_cols_vals_ptr,
    a_stride_B,
    a_stride_K,
    b_stride_K,
    b_stride_N,
    cols_vals_stride_0,
    # Outputs (corrected_logits pre-filled with -inf; kernel only writes valid positions)
    corrected_logits_ptr,
    next_node_ptr,
    valid_idxs_ptr,
    corrected_logits_stride_B,
    corrected_logits_stride_N,
    next_node_stride_B,
    next_node_stride_N,
    valid_idxs_stride_B,
    valid_idxs_stride_N,
    # Constants
    B: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_BRANCHES: tl.constexpr,
    max_branches,
    HAS_BIAS: tl.constexpr,
):
    """
    Sparse fused kernel: computes dot products only for the valid (constrained) tokens.

    Grid: (ceil(B/BLOCK_B), ceil(max_branches/BLOCK_BRANCHES)) programs.
    axis=0 tiles the batch; axis=1 tiles the branch slots.
    For each branch tile, tl.static_range(BLOCK_BRANCHES) unrolls BLOCK_BRANCHES iterations
    at compile time, each handling one absolute branch index abs_br = pid_BR*BLOCK_BRANCHES+local_br.
    Branches outside [0, max_branches) are fully masked and produce no stores.
    """
    pid_B = tl.program_id(axis=0)
    pid_BR = tl.program_id(axis=1)

    offs_B = pid_B * BLOCK_B + tl.arange(0, BLOCK_B)
    b_mask = offs_B < B

    cur_node = tl.load(cur_node_ptr + offs_B, mask=b_mask, other=-1)
    csr_row_ptrs = tl.load(csr_trie_row_ptr + cur_node, mask=cur_node >= 0, other=0)
    csr_next_ptrs = tl.load(
        csr_trie_row_ptr + cur_node + 1, mask=cur_node >= 0, other=0
    )
    n_children = csr_next_ptrs - csr_row_ptrs

    for local_br in tl.static_range(BLOCK_BRANCHES):
        branch_idx = pid_BR * BLOCK_BRANCHES + local_br
        _sparse_branch_body(
            branch_idx, max_branches, offs_B, b_mask, csr_row_ptrs, n_children,
            a_ptr, b_ptr, bias_ptr, csr_trie_cols_vals_ptr, cols_vals_stride_0,
            corrected_logits_ptr, next_node_ptr, valid_idxs_ptr,
            a_stride_B, a_stride_K, b_stride_K, b_stride_N,
            corrected_logits_stride_B, corrected_logits_stride_N,
            next_node_stride_B, next_node_stride_N,
            valid_idxs_stride_B, valid_idxs_stride_N,
            K, BLOCK_B, BLOCK_K, HAS_BIAS,
        )


@triton_op(
    "vtnk::_fused_linear_constrained_node_transition_sampling_op", mutates_args={}
)
def _fused_linear_constrained_node_transition_sampling_op(
    a: torch.Tensor,
    b: torch.Tensor,
    bias_val: torch.Tensor,
    cur_node: torch.Tensor,
    csr_row_ptrs: torch.Tensor,
    csr_cols_vals: torch.Tensor,
    max_branches: int,
    has_bias: bool,
    rng_seed: Optional[int] = None,
    temperature: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if rng_seed is None:
        rng_seed = time.time_ns() & 0x7FFFFFFF
    if temperature is None or temperature == 1.0:
        temperature = torch.ones(1, dtype=torch.float32, device=a.device)
    elif isinstance(temperature, float):
        temperature = torch.tensor(temperature, dtype=torch.float32, device=a.device)

    B, K = a.shape
    N = b.shape[1]

    assert cur_node.shape == (B,), (
        f"Expected cur_node shape ({B},), got {cur_node.shape}"
    )

    a = a.contiguous()
    cur_node = cur_node.contiguous()
    csr_cols_vals = csr_cols_vals.contiguous()
    bias_val = bias_val.contiguous()

    corrected_logits = torch.full(
        (B, N), float("-inf"), dtype=torch.float32, device=a.device
    )
    next_node = cur_node.new_full((B, max_branches), -1)
    valid_idxs = cur_node.new_full((B, max_branches), -1)
    sample = torch.full((B,), -1.0, dtype=torch.float32, device=a.device)
    gumbel_max = torch.full((B,), float("-inf"), dtype=torch.float32, device=a.device)
    num_locks = triton.cdiv(B, 16)
    locks = torch.zeros(num_locks, dtype=torch.int32, device=a.device)

    grid = lambda meta: (
        triton.cdiv(B, meta["BLOCK_B"]),
        triton.cdiv(max_branches, meta["BLOCK_BRANCHES"]),
    )
    wrap_triton(_fused_sparse_linear_constrained_node_transition_sampling_kernel)[grid](
        a_ptr=a,
        b_ptr=b,
        bias_ptr=bias_val,
        cur_node_ptr=cur_node,
        csr_trie_row_ptr=csr_row_ptrs,
        csr_trie_cols_vals_ptr=csr_cols_vals,
        temperature_ptr=temperature,
        gumbel_max_ptr=gumbel_max,
        locks_ptr=locks,
        num_locks=num_locks,
        a_stride_B=a.stride(0),
        a_stride_K=a.stride(1),
        b_stride_K=b.stride(0),
        b_stride_N=b.stride(1),
        cols_vals_stride_0=csr_cols_vals.stride(0),
        corrected_logits_ptr=corrected_logits,
        next_node_ptr=next_node,
        valid_idxs_ptr=valid_idxs,
        sample_ptr=sample,
        corrected_logits_stride_B=corrected_logits.stride(0),
        corrected_logits_stride_N=corrected_logits.stride(1),
        next_node_stride_B=next_node.stride(0),
        next_node_stride_N=next_node.stride(1),
        valid_idxs_stride_B=valid_idxs.stride(0),
        valid_idxs_stride_N=valid_idxs.stride(1),
        rng_seed=rng_seed,
        B=B,
        K=K,
        N=N,
        max_branches=max_branches,
        HAS_BIAS=has_bias,
    )
    return next_node, valid_idxs, corrected_logits, sample


@triton.autotune(
    configs=_FUSED_AUTOTUNE_CONFIGS,
    key=["B", "K", "N"],
    restore_value=[
        "corrected_logits_ptr",
        "next_node_ptr",
        "valid_idxs_ptr",
        "sample_ptr",
        "gumbel_max_ptr",
        "locks_ptr",
    ],
)
@triton.jit
def _fused_sparse_linear_constrained_node_transition_sampling_kernel(
    # Inputs
    a_ptr,
    b_ptr,
    bias_ptr,
    cur_node_ptr,
    csr_trie_row_ptr,
    csr_trie_cols_vals_ptr,
    temperature_ptr,
    gumbel_max_ptr,
    locks_ptr,
    num_locks,
    a_stride_B,
    a_stride_K,
    b_stride_K,
    b_stride_N,
    cols_vals_stride_0,
    # Outputs (corrected_logits pre-filled with -inf; kernel only writes valid positions)
    corrected_logits_ptr,
    next_node_ptr,
    valid_idxs_ptr,
    sample_ptr,
    corrected_logits_stride_B,
    corrected_logits_stride_N,
    next_node_stride_B,
    next_node_stride_N,
    valid_idxs_stride_B,
    valid_idxs_stride_N,
    rng_seed,
    # Constants
    B: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_BRANCHES: tl.constexpr,
    max_branches,
    HAS_BIAS: tl.constexpr,
):
    """
    Sparse fused kernel with Gumbel-max sampling.

    Grid: (ceil(B/BLOCK_B), ceil(max_branches/BLOCK_BRANCHES)).
    Each branch block accumulates a local Gumbel max over its BLOCK_BRANCHES branches,
    then merges into a global gumbel_max / sample buffer using a per-batch-block spinlock.
    """
    pid_B = tl.program_id(axis=0)
    pid_BR = tl.program_id(axis=1)

    offs_B = pid_B * BLOCK_B + tl.arange(0, BLOCK_B)
    b_mask = offs_B < B

    cur_node = tl.load(cur_node_ptr + offs_B, mask=b_mask, other=-1)
    csr_row_ptrs = tl.load(csr_trie_row_ptr + cur_node, mask=cur_node >= 0, other=0)
    csr_next_ptrs = tl.load(
        csr_trie_row_ptr + cur_node + 1, mask=cur_node >= 0, other=0
    )
    n_children = csr_next_ptrs - csr_row_ptrs

    temperature = tl.load(temperature_ptr)
    block_max_gumbel = tl.full((BLOCK_B,), float("-inf"), dtype=tl.float32)
    block_sample = tl.full((BLOCK_B,), -1.0, dtype=tl.float32)

    for local_br in tl.static_range(BLOCK_BRANCHES):
        branch_idx = pid_BR * BLOCK_BRANCHES + local_br
        col_k, logit_k, child_mask = _sparse_branch_body(
            branch_idx, max_branches, offs_B, b_mask, csr_row_ptrs, n_children,
            a_ptr, b_ptr, bias_ptr, csr_trie_cols_vals_ptr, cols_vals_stride_0,
            corrected_logits_ptr, next_node_ptr, valid_idxs_ptr,
            a_stride_B, a_stride_K, b_stride_K, b_stride_N,
            corrected_logits_stride_B, corrected_logits_stride_N,
            next_node_stride_B, next_node_stride_N,
            valid_idxs_stride_B, valid_idxs_stride_N,
            K, BLOCK_B, BLOCK_K, HAS_BIAS,
        )
        block_sample, block_max_gumbel = _gumbel_max_update(
            rng_seed, offs_B, max_branches, branch_idx,
            logit_k, temperature, child_mask, col_k,
            block_sample, block_max_gumbel,
        )

    # Cross-block Gumbel-max reduction: spinlock protects per-batch-block update.
    # tl.atomic_max on float32 is not correct for negative values, so we use a
    # compare-and-store pattern under the lock instead.
    lock_ptr = locks_ptr + pid_B // tl.cdiv(B, BLOCK_B * num_locks)
    while tl.atomic_cas(lock_ptr, 0, 1) == 1:
        pass

    cur_max = tl.load(gumbel_max_ptr + offs_B, mask=b_mask, other=float("-inf"))
    improved_global = block_max_gumbel > cur_max
    tl.store(
        gumbel_max_ptr + offs_B,
        tl.where(improved_global, block_max_gumbel, cur_max),
        mask=b_mask,
    )
    cur_sample = tl.load(sample_ptr + offs_B, mask=b_mask, other=-1.0)
    tl.store(
        sample_ptr + offs_B,
        tl.where(improved_global, block_sample, cur_sample),
        mask=b_mask,
    )

    tl.debug_barrier()
    tl.atomic_xchg(lock_ptr, 0)


# ─────────────────────────────────────────────────────────────────────────────
# Fused sparse linear + top-K kernel
# ─────────────────────────────────────────────────────────────────────────────


@triton.jit
def _bitonic_compare_swap(val_a, idx_a, val_b, idx_b):
    """Descending compare-and-swap Triton device function.

    Returns (larger_val, larger_idx, smaller_val, smaller_idx).
    Inlined by the Triton compiler when called from a @triton.jit kernel.
    val_a / idx_a and val_b / idx_b are (BLOCK_B,) vectors; the swap is
    applied element-wise across the batch dimension.
    """
    swap = val_a < val_b
    return (
        tl.where(swap, val_b, val_a),
        tl.where(swap, idx_b, idx_a),
        tl.where(swap, val_a, val_b),
        tl.where(swap, idx_a, idx_b),
    )


def _bitonic_topk(logits: list, idxs: list, n: int, k: int):
    """Bitonic sort on n elements (must be a power of 2); return first k descending.

    Plain Python — all loops run at trace time (n and k are Python ints derived
    from tl.constexpr kernel parameters).  Calls the @triton.jit
    _bitonic_compare_swap device function, which Triton inlines into the
    kernel IR.  The result is a flat sequence of tl.where ops with no runtime
    branching.

    Comparator counts (vs. incremental insertion sort, worst case K_TOP == n):
      n=4:  5 comparators  vs 16
      n=8:  19 comparators vs 64
      n=16: 63 comparators vs 256
    """
    step = 2
    while step <= n:
        half = step // 2
        while half >= 1:
            for i in range(n):
                j = i ^ half
                if j > i:
                    if (i & step) == 0:
                        # position i gets the larger value (descending sub-sequence)
                        logits[i], idxs[i], logits[j], idxs[j] = _bitonic_compare_swap(
                            logits[i], idxs[i], logits[j], idxs[j]
                        )
                    else:
                        # position j gets the larger value (ascending sub-sequence)
                        logits[j], idxs[j], logits[i], idxs[i] = _bitonic_compare_swap(
                            logits[j], idxs[j], logits[i], idxs[i]
                        )
            half //= 2
        step *= 2
    return logits[:k], idxs[:k]


@triton_op("vtnk::_fused_linear_constrained_node_transition_topk_op", mutates_args={})
def _fused_linear_constrained_node_transition_topk_op(
    a: torch.Tensor,
    b: torch.Tensor,
    bias_val: torch.Tensor,
    cur_node: torch.Tensor,
    csr_row_ptrs: torch.Tensor,
    csr_cols_vals: torch.Tensor,
    max_branches: int,
    has_bias: bool,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, K = a.shape
    N = b.shape[1]

    assert cur_node.shape == (B,), (
        f"Expected cur_node shape ({B},), got {cur_node.shape}"
    )

    a = a.contiguous()
    cur_node = cur_node.contiguous()
    csr_cols_vals = csr_cols_vals.contiguous()
    bias_val = bias_val.contiguous()

    corrected_logits = torch.full(
        (B, N), float("-inf"), dtype=torch.float32, device=a.device
    )
    next_node = cur_node.new_full((B, max_branches), -1)
    valid_idxs = cur_node.new_full((B, max_branches), -1)
    topk_logits = torch.full((B, k), float("-inf"), dtype=torch.float32, device=a.device)
    topk_idxs = torch.full((B, k), -1, dtype=torch.int64, device=a.device)
    num_locks = triton.cdiv(B, 16)
    locks = torch.zeros(num_locks, dtype=torch.int32, device=a.device)

    grid = lambda meta: (
        triton.cdiv(B, meta["BLOCK_B"]),
        triton.cdiv(max_branches, meta["BLOCK_BRANCHES"]),
    )
    wrap_triton(_fused_sparse_linear_constrained_node_transition_topk_kernel)[grid](
        a_ptr=a,
        b_ptr=b,
        bias_ptr=bias_val,
        cur_node_ptr=cur_node,
        csr_trie_row_ptr=csr_row_ptrs,
        csr_trie_cols_vals_ptr=csr_cols_vals,
        locks_ptr=locks,
        num_locks=num_locks,
        a_stride_B=a.stride(0),
        a_stride_K=a.stride(1),
        b_stride_K=b.stride(0),
        b_stride_N=b.stride(1),
        cols_vals_stride_0=csr_cols_vals.stride(0),
        corrected_logits_ptr=corrected_logits,
        next_node_ptr=next_node,
        valid_idxs_ptr=valid_idxs,
        topk_logits_ptr=topk_logits,
        topk_idxs_ptr=topk_idxs,
        corrected_logits_stride_B=corrected_logits.stride(0),
        corrected_logits_stride_N=corrected_logits.stride(1),
        next_node_stride_B=next_node.stride(0),
        next_node_stride_N=next_node.stride(1),
        valid_idxs_stride_B=valid_idxs.stride(0),
        valid_idxs_stride_N=valid_idxs.stride(1),
        B=B,
        K=K,
        N=N,
        K_TOP=k,
        max_branches=max_branches,
        HAS_BIAS=has_bias,
    )
    return next_node, valid_idxs, topk_logits, topk_idxs


@triton.autotune(
    configs=_FUSED_AUTOTUNE_CONFIGS,
    key=["B", "K", "N", "K_TOP"],
    restore_value=[
        "corrected_logits_ptr",
        "next_node_ptr",
        "valid_idxs_ptr",
        "topk_logits_ptr",
        "topk_idxs_ptr",
        "locks_ptr",
    ],
)
@triton.jit
def _fused_sparse_linear_constrained_node_transition_topk_kernel(
    # Inputs
    a_ptr,
    b_ptr,
    bias_ptr,
    cur_node_ptr,
    csr_trie_row_ptr,
    csr_trie_cols_vals_ptr,
    locks_ptr,
    num_locks,
    a_stride_B,
    a_stride_K,
    b_stride_K,
    b_stride_N,
    cols_vals_stride_0,
    # Outputs (corrected_logits pre-filled with -inf; kernel only writes valid positions)
    corrected_logits_ptr,
    next_node_ptr,
    valid_idxs_ptr,
    topk_logits_ptr,
    topk_idxs_ptr,
    corrected_logits_stride_B,
    corrected_logits_stride_N,
    next_node_stride_B,
    next_node_stride_N,
    valid_idxs_stride_B,
    valid_idxs_stride_N,
    # Constants
    B: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    K_TOP: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_BRANCHES: tl.constexpr,
    max_branches,
    HAS_BIAS: tl.constexpr,
):
    """
    Sparse fused linear + top-K kernel.

    Grid: (ceil(B/BLOCK_B), ceil(max_branches/BLOCK_BRANCHES)).

    For each branch in BLOCK_BRANCHES (unrolled via tl.static_range):
      1. Compute the sparse logit via _sparse_branch_body (writes corrected_logits,
         next_node, valid_idxs as a side effect).
      2. Acquire a per-batch-block spinlock (shared across all pid_BR blocks).
      3. Insert (logit, col) into the global (B, K_TOP) top-K buffers via insertion
         sort — each slot is updated with _bitonic_compare_swap, which ensures the
         larger value stays in the slot and the displaced value propagates forward.

    Note: Triton's AST code generator cannot lower Python list operations (append,
    indexed assignment) to GPU IR, so the "collect all branches, then sort once"
    pattern is not usable here.  The per-branch spinlock design avoids register
    arrays and remains correct: O(BLOCK_BRANCHES * K_TOP) store ops per block.
    """
    pid_B = tl.program_id(axis=0)
    pid_BR = tl.program_id(axis=1)

    offs_B = pid_B * BLOCK_B + tl.arange(0, BLOCK_B)
    b_mask = offs_B < B

    cur_node = tl.load(cur_node_ptr + offs_B, mask=b_mask, other=-1)
    csr_row_ptrs = tl.load(csr_trie_row_ptr + cur_node, mask=cur_node >= 0, other=0)
    csr_next_ptrs = tl.load(
        csr_trie_row_ptr + cur_node + 1, mask=cur_node >= 0, other=0
    )
    n_children = csr_next_ptrs - csr_row_ptrs

    # Spinlock pointer for this batch block — shared across all pid_BR blocks.
    lock_ptr = locks_ptr + pid_B // tl.cdiv(B, BLOCK_B * num_locks)

    # Per-branch processing: for each branch, compute the logit, then acquire the
    # spinlock once and insert into the global top-K buffer via insertion sort.
    # Using _bitonic_compare_swap (a @triton.jit device function) for each slot
    # keeps the logic compact: each call ensures the slot retains the larger value
    # and the displaced value propagates to the next slot.
    # Triton does not support Python list operations (e.g., list.append) inside
    # @triton.jit kernels — the AST code generator cannot lower them to IR.  The
    # per-branch spinlock pattern avoids needing register arrays entirely.
    for local_br in tl.static_range(BLOCK_BRANCHES):
        branch_idx = pid_BR * BLOCK_BRANCHES + local_br
        col_k, logit_k, child_mask = _sparse_branch_body(
            branch_idx, max_branches, offs_B, b_mask, csr_row_ptrs, n_children,
            a_ptr, b_ptr, bias_ptr, csr_trie_cols_vals_ptr, cols_vals_stride_0,
            corrected_logits_ptr, next_node_ptr, valid_idxs_ptr,
            a_stride_B, a_stride_K, b_stride_K, b_stride_N,
            corrected_logits_stride_B, corrected_logits_stride_N,
            next_node_stride_B, next_node_stride_N,
            valid_idxs_stride_B, valid_idxs_stride_N,
            K, BLOCK_B, BLOCK_K, HAS_BIAS,
        )
        cur_l = tl.where(child_mask, logit_k, float("-inf"))
        cur_i = col_k.to(tl.int64)

        while tl.atomic_cas(lock_ptr, 0, 1) == 1:
            pass

        # Insertion sort: each slot stores the larger of (cur, slot); the smaller
        # value propagates forward.  All K_TOP iterations unroll at trace time.
        for slot in range(K_TOP):
            global_l = tl.load(
                topk_logits_ptr + offs_B * K_TOP + slot,
                mask=b_mask,
                other=float("-inf"),
            )
            global_i = tl.load(
                topk_idxs_ptr + offs_B * K_TOP + slot,
                mask=b_mask,
                other=-1,
            )
            new_slot_l, new_slot_i, cur_l, cur_i = _bitonic_compare_swap(
                cur_l, cur_i, global_l, global_i
            )
            tl.store(topk_logits_ptr + offs_B * K_TOP + slot, new_slot_l, mask=b_mask)
            tl.store(topk_idxs_ptr + offs_B * K_TOP + slot, new_slot_i, mask=b_mask)

        tl.debug_barrier()
        tl.atomic_xchg(lock_ptr, 0)
