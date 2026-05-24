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
    key=["B", "N", "max_branches"],
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
    max_branches: tl.constexpr,
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
    # triton.Config({"BLOCK_B": 64, "BLOCK_K": 64, "BLOCK_BRANCHES": 64}),
    # triton.Config({"BLOCK_B": 128, "BLOCK_K": 64, "BLOCK_BRANCHES": 64}),
]
# Minimum BLOCK_BRANCHES across _FUSED_AUTOTUNE_CONFIGS; used to bound max_br_blocks at allocation time.
_FUSED_MIN_BLOCK_BRANCHES = 4


# ─────────────────────────────────────────────────────────────────────────────
# Shared Triton device-function helpers
# ─────────────────────────────────────────────────────────────────────────────


@triton.jit
def _load_csr_state(offs_B, b_mask, cur_node_ptr, csr_trie_row_ptr):
    """Load (csr_row_ptrs, n_children) for the current batch from the CSR trie."""
    cur_node = tl.load(cur_node_ptr + offs_B, mask=b_mask, other=-1)
    csr_row_ptrs = tl.load(csr_trie_row_ptr + cur_node, mask=cur_node >= 0, other=0)
    csr_next_ptrs = tl.load(
        csr_trie_row_ptr + cur_node + 1, mask=cur_node >= 0, other=0
    )
    return csr_row_ptrs, csr_next_ptrs - csr_row_ptrs


@triton.jit
def _select_branch(
    local_br: tl.constexpr,
    branch_cols,  # [BLOCK_B, BLOCK_BRANCHES]
    branch_valid,  # [BLOCK_B, BLOCK_BRANCHES]
    BLOCK_BRANCHES: tl.constexpr,
):
    """Return (br_sel, col_k, c_mask) for branch slot local_br.

    br_sel is the compile-time column selector; col_k and c_mask are [BLOCK_B] vectors.
    """
    br_sel = tl.arange(0, BLOCK_BRANCHES) == local_br  # [BLOCK_BRANCHES], compile-time
    col_k = tl.sum(tl.where(br_sel[None, :], branch_cols, 0), axis=1)
    c_mask = tl.sum(tl.where(br_sel[None, :], branch_valid.to(tl.int32), 0), axis=1).to(
        tl.int1
    )
    return br_sel, col_k, c_mask



@triton.jit
def _compute_branch_logits(
    offs_B,
    offs_BR,
    b_mask,
    csr_row_ptrs,
    n_children,
    a_ptr,
    b_ptr,
    bias_ptr,
    csr_trie_cols_vals_ptr,
    a_stride_B,
    a_stride_K,
    b_stride_K,
    b_stride_N,
    cols_vals_stride_0,
    max_branches,
    K: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_BRANCHES: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """Load all branch data and compute per-branch dot-product logits.

    Loads a_chunk once per K-tile and amortizes it across all BLOCK_BRANCHES branches,
    reducing query-vector HBM traffic by BLOCK_BRANCHES× vs computing each branch separately.

    Returns (branch_cols, branch_vals, branch_valid, logits) as [BLOCK_B, BLOCK_BRANCHES]
    tensors. Bias is folded into logits when HAS_BIAS=True.
    """
    branch_valid = (
        b_mask[:, None]
        & (n_children[:, None] > offs_BR[None, :].to(tl.int64))
        & (offs_BR[None, :] < max_branches)
    )  # [BLOCK_B, BLOCK_BRANCHES]
    branch_cols = tl.load(
        csr_trie_cols_vals_ptr + csr_row_ptrs[:, None] + offs_BR[None, :],
        mask=branch_valid,
        other=-1,
    )  # [BLOCK_B, BLOCK_BRANCHES]
    branch_vals = tl.load(
        csr_trie_cols_vals_ptr
        + csr_row_ptrs[:, None]
        + offs_BR[None, :]
        + cols_vals_stride_0,
        mask=branch_valid,
        other=-1,
    )  # [BLOCK_B, BLOCK_BRANCHES]

    logits = tl.zeros((BLOCK_B, BLOCK_BRANCHES), dtype=tl.float32)

    for k_tile in range(0, tl.cdiv(K, BLOCK_K)):
        offs_K = k_tile * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = offs_K < K

        # Load a_chunk once per tile; reused across all BLOCK_BRANCHES branches.
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

    return branch_cols, branch_vals, branch_valid, logits


@triton.jit
def _fused_prologue(
    cur_node_ptr,
    csr_trie_row_ptr,
    a_ptr,
    b_ptr,
    bias_ptr,
    csr_trie_cols_vals_ptr,
    next_node_ptr,
    valid_idxs_ptr,
    a_stride_B,
    a_stride_K,
    b_stride_K,
    b_stride_N,
    cols_vals_stride_0,
    next_node_stride_B,
    next_node_stride_N,
    valid_idxs_stride_B,
    valid_idxs_stride_N,
    max_branches,
    B: tl.constexpr,
    K: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_BRANCHES: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """Shared prologue: pid setup, CSR load, branch logit compute, and next_node/valid_idxs stores."""
    pid_B = tl.program_id(axis=0)
    pid_BR = tl.program_id(axis=1)

    offs_B = pid_B * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_BR = pid_BR * BLOCK_BRANCHES + tl.arange(0, BLOCK_BRANCHES)
    b_mask = offs_B < B

    csr_row_ptrs, n_children = _load_csr_state(
        offs_B, b_mask, cur_node_ptr, csr_trie_row_ptr
    )

    branch_cols, branch_vals, branch_valid, logits = _compute_branch_logits(
        offs_B, offs_BR, b_mask, csr_row_ptrs, n_children,
        a_ptr, b_ptr, bias_ptr, csr_trie_cols_vals_ptr,
        a_stride_B, a_stride_K, b_stride_K, b_stride_N, cols_vals_stride_0,
        max_branches, K, BLOCK_B, BLOCK_K, BLOCK_BRANCHES, HAS_BIAS,
    )

    store_mask = b_mask[:, None] & (offs_BR[None, :] < max_branches)
    tl.store(
        next_node_ptr + offs_B[:, None] * next_node_stride_B + offs_BR[None, :] * next_node_stride_N,
        branch_vals,
        mask=store_mask,
    )
    tl.store(
        valid_idxs_ptr + offs_B[:, None] * valid_idxs_stride_B + offs_BR[None, :] * valid_idxs_stride_N,
        branch_cols,
        mask=store_mask,
    )

    return pid_BR, offs_B, offs_BR, b_mask, branch_cols, branch_valid, logits


# ─────────────────────────────────────────────────────────────────────────────
# Fused sparse linear kernel
# ─────────────────────────────────────────────────────────────────────────────


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
        (B, N), float("-inf"), dtype=torch.bfloat16, device=a.device
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
    _, offs_B, _, _, branch_cols, branch_valid, logits = _fused_prologue(
        cur_node_ptr, csr_trie_row_ptr,
        a_ptr, b_ptr, bias_ptr, csr_trie_cols_vals_ptr,
        next_node_ptr, valid_idxs_ptr,
        a_stride_B, a_stride_K, b_stride_K, b_stride_N, cols_vals_stride_0,
        next_node_stride_B, next_node_stride_N, valid_idxs_stride_B, valid_idxs_stride_N,
        max_branches, B, K, BLOCK_B, BLOCK_K, BLOCK_BRANCHES, HAS_BIAS,
    )

    # corrected_logits is indexed by token column (scattered) — requires per-branch loop.
    for local_br in tl.static_range(BLOCK_BRANCHES):
        br_sel, col_k, c_mask = _select_branch(local_br, branch_cols, branch_valid, BLOCK_BRANCHES)
        logit_k = tl.sum(tl.where(br_sel[None, :], logits, 0.0), axis=1)
        tl.store(
            corrected_logits_ptr
            + offs_B * corrected_logits_stride_B
            + col_k * corrected_logits_stride_N,
            logit_k.to(tl.bfloat16),
            mask=c_mask,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Fused sparse linear + Gumbel-max sampling kernel
# ─────────────────────────────────────────────────────────────────────────────


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if rng_seed is None:
        rng_seed = time.time_ns() & 0x7FFFFFFF
    if temperature is None or temperature == 1.0:
        temperature = torch.ones(1, dtype=torch.float32, device=a.device)
    elif isinstance(temperature, float):
        temperature = torch.tensor(temperature, dtype=torch.float32, device=a.device)

    B, K = a.shape

    assert cur_node.shape == (B,), (
        f"Expected cur_node shape ({B},), got {cur_node.shape}"
    )

    a = a.contiguous()
    cur_node = cur_node.contiguous()
    csr_cols_vals = csr_cols_vals.contiguous()
    bias_val = bias_val.contiguous()

    next_node = cur_node.new_full((B, max_branches), -1)
    valid_idxs = cur_node.new_full((B, max_branches), -1)
    # Worst-case block count uses the minimum BLOCK_BRANCHES across all autotune configs.
    # Configs with larger BLOCK_BRANCHES write fewer entries; unused slots stay at -inf/-1.
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
    wrap_triton(_fused_sparse_linear_constrained_node_transition_sampling_kernel)[grid](
        a_ptr=a,
        b_ptr=b,
        bias_ptr=bias_val,
        cur_node_ptr=cur_node,
        csr_trie_row_ptr=csr_row_ptrs,
        csr_trie_cols_vals_ptr=csr_cols_vals,
        temperature_ptr=temperature,
        a_stride_B=a.stride(0),
        a_stride_K=a.stride(1),
        b_stride_K=b.stride(0),
        b_stride_N=b.stride(1),
        cols_vals_stride_0=csr_cols_vals.stride(0),
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
    configs=_FUSED_AUTOTUNE_CONFIGS,
    key=["B", "K", "max_branches"],
    restore_value=["next_node_ptr", "valid_idxs_ptr", "gumbel_block_max_ptr", "block_sample_ptr"],
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
    a_stride_B,
    a_stride_K,
    b_stride_K,
    b_stride_N,
    cols_vals_stride_0,
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
    max_branches,
    HAS_BIAS: tl.constexpr,
):
    pid_BR, offs_B, offs_BR, b_mask, branch_cols, branch_valid, logits = _fused_prologue(
        cur_node_ptr, csr_trie_row_ptr,
        a_ptr, b_ptr, bias_ptr, csr_trie_cols_vals_ptr,
        next_node_ptr, valid_idxs_ptr,
        a_stride_B, a_stride_K, b_stride_K, b_stride_N, cols_vals_stride_0,
        next_node_stride_B, next_node_stride_N, valid_idxs_stride_B, valid_idxs_stride_N,
        max_branches, B, K, BLOCK_B, BLOCK_K, BLOCK_BRANCHES, HAS_BIAS,
    )

    # Vectorized Gumbel-max over the full [BLOCK_B, BLOCK_BRANCHES] block — registers only, no sync.
    # See: https://arxiv.org/pdf/2603.15854
    temperature = tl.load(temperature_ptr)
    u = tl.rand(seed=rng_seed, offset=offs_B[:, None] * max_branches + offs_BR[None, :])
    gumbel = -tl.log(-tl.log(u + 1e-10) + 1e-10)
    g_vals = tl.where(branch_valid, logits / temperature + gumbel, float("-inf"))
    block_max_gumbel = tl.max(g_vals, axis=1)  # [BLOCK_B]

    # Recover winning token: exactly one winner per batch element (gumbel ties are negligible).
    is_winner = (g_vals == block_max_gumbel[:, None]) & branch_valid
    block_sample = tl.sum(tl.where(is_winner, branch_cols.to(tl.float32), 0.0), axis=1)

    # One write per (batch, branch-block) — unique addresses, no lock needed.
    # Skip block_sample write when no valid branch exists; buf retains its -1.0 init value.
    tl.store(gumbel_block_max_ptr + offs_B * max_br_blocks + pid_BR, block_max_gumbel, mask=b_mask)
    tl.store(
        block_sample_ptr + offs_B * max_br_blocks + pid_BR,
        block_sample,
        mask=b_mask & (block_max_gumbel > float("-inf")),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Fused sparse linear + top-K kernel
# ─────────────────────────────────────────────────────────────────────────────


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

    assert cur_node.shape == (B,), (
        f"Expected cur_node shape ({B},), got {cur_node.shape}"
    )

    a = a.contiguous()
    cur_node = cur_node.contiguous()
    csr_cols_vals = csr_cols_vals.contiguous()
    bias_val = bias_val.contiguous()

    next_node = cur_node.new_full((B, max_branches), -1)
    valid_idxs = cur_node.new_full((B, max_branches), -1)
    # Pass 1: compute per-branch logits into a [B, max_branches] scratch buffer.
    branch_logits = torch.full(
        (B, max_branches), float("-inf"), dtype=torch.bfloat16, device=a.device
    )

    grid = lambda meta: (
        triton.cdiv(B, meta["BLOCK_B"]),
        triton.cdiv(max_branches, meta["BLOCK_BRANCHES"]),
    )
    wrap_triton(_fused_sparse_linear_constrained_node_transition_compact_kernel)[grid](
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

    # Pass 2: topk on the small [B, max_branches] buffer — no serialization.
    if k >= max_branches:
        return next_node, valid_idxs, branch_logits, valid_idxs.clone()
    topk_logits, topk_branch_idxs = torch.topk(branch_logits, k, dim=-1)
    topk_idxs = valid_idxs.gather(1, topk_branch_idxs)
    return next_node, valid_idxs, topk_logits, topk_idxs


@triton.autotune(
    configs=_FUSED_AUTOTUNE_CONFIGS,
    key=["B", "K", "max_branches"],
    restore_value=["next_node_ptr", "valid_idxs_ptr", "branch_logits_ptr"],
)
@triton.jit
def _fused_sparse_linear_constrained_node_transition_compact_kernel(
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
    max_branches,
    HAS_BIAS: tl.constexpr,
):
    _, offs_B, offs_BR, b_mask, _, branch_valid, logits = _fused_prologue(
        cur_node_ptr, csr_trie_row_ptr,
        a_ptr, b_ptr, bias_ptr, csr_trie_cols_vals_ptr,
        next_node_ptr, valid_idxs_ptr,
        a_stride_B, a_stride_K, b_stride_K, b_stride_N, cols_vals_stride_0,
        next_node_stride_B, next_node_stride_N, valid_idxs_stride_B, valid_idxs_stride_N,
        max_branches, B, K, BLOCK_B, BLOCK_K, BLOCK_BRANCHES, HAS_BIAS,
    )

    store_mask = b_mask[:, None] & (offs_BR[None, :] < max_branches)
    tl.store(
        branch_logits_ptr + offs_B[:, None] * max_branches + offs_BR[None, :],
        tl.where(branch_valid, logits, float("-inf")).to(tl.bfloat16),
        mask=store_mask,
    )
