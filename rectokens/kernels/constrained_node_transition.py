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
                mask=child_valid, other=-1,
            )
            val_k = tl.load(
                csr_trie_cols_vals_ptr + csr_row_ptrs + k + cols_vals_stride_0,
                mask=child_valid, other=-1,
            )
            tl.store(
                next_node_ptr + offs_B * next_node_stride_B + k * next_node_stride_N,
                val_k, mask=b_valid_store,
            )
            tl.store(
                valid_idxs_ptr + offs_B * valid_idxs_stride_B + k * valid_idxs_stride_N,
                col_k, mask=b_valid_store,
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

    grid = lambda meta: (triton.cdiv(B, meta["BLOCK_B"]),)
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
    configs=[
        triton.Config({"BLOCK_B": 64, "BLOCK_K": 64}),
        triton.Config({"BLOCK_B": 128, "BLOCK_K": 64}),
        triton.Config({"BLOCK_B": 256, "BLOCK_K": 64}),
        triton.Config({"BLOCK_B": 64, "BLOCK_K": 128}),
        triton.Config({"BLOCK_B": 128, "BLOCK_K": 128}),
    ],
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
    max_branches,
    HAS_BIAS: tl.constexpr,
):
    """
    Sparse fused kernel: computes dot products only for the valid (constrained) tokens.

    Grid: ceil(B / BLOCK_B) programs, one per batch tile.
    For each branch slot k in [0, max_branches):
      - Gather valid column index col_k[b] from the CSR trie for each batch element.
      - Compute dot(a[b, :], weight[col_k[b], :]) via K-loop with element-wise multiply+sum.
        (b = weight.T with b_stride_K=1, b_stride_N=K, so weight[col, :] is contiguous.)
      - Scatter the result to corrected_logits[b, col_k[b]].
    Invalid branch slots (k >= n_children) are skipped via masking.
    """
    pid = tl.program_id(axis=0)
    offs_B = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    b_mask = offs_B < B

    cur_node = tl.load(cur_node_ptr + offs_B, mask=b_mask, other=-1)
    csr_row_ptrs = tl.load(csr_trie_row_ptr + cur_node, mask=cur_node >= 0, other=0)
    csr_next_ptrs = tl.load(
        csr_trie_row_ptr + cur_node + 1, mask=cur_node >= 0, other=0
    )
    n_children = csr_next_ptrs - csr_row_ptrs

    for branch_idx in range(max_branches):
        child_mask = b_mask & (n_children > branch_idx)

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

        # Compute dot(a[b], weight[col_k[b], :]) for each batch element.
        # b = weight.T (non-contiguous view): b_stride_K=1, b_stride_N=K
        # => b_ptr + col_k[b_i] * K + offs_K is weight[col_k[b_i], offs_K]: contiguous per row.
        logit_k = tl.zeros((BLOCK_B,), dtype=tl.float32)
        for k_tile in range(0, tl.cdiv(K, BLOCK_K)):
            offs_K = k_tile * BLOCK_K + tl.arange(0, BLOCK_K)
            k_mask = offs_K < K

            a_ptrs = a_ptr + offs_B[:, None] * a_stride_B + offs_K[None, :] * a_stride_K
            a_chunk = tl.load(a_ptrs, mask=b_mask[:, None] & k_mask[None, :], other=0.0)

            # Gather weight[col_k[b_i], offs_K] for each batch element b_i.
            b_ptrs = b_ptr + offs_K[None, :] * b_stride_K + col_k[:, None] * b_stride_N
            b_chunk = tl.load(
                b_ptrs, mask=child_mask[:, None] & k_mask[None, :], other=0.0
            )

            logit_k += tl.sum(a_chunk * b_chunk, axis=1)

        if HAS_BIAS:
            logit_k += tl.load(bias_ptr + col_k, mask=child_mask, other=0.0)

        # Scatter valid logits; corrected_logits was pre-filled with -inf.
        out_ptrs = (
            corrected_logits_ptr
            + offs_B * corrected_logits_stride_B
            + col_k * corrected_logits_stride_N
        )
        tl.store(out_ptrs, logit_k, mask=child_mask)

        # Store next_node and valid_idxs for this branch slot.
        next_node_ptrs = (
            next_node_ptr
            + offs_B * next_node_stride_B
            + branch_idx * next_node_stride_N
        )
        valid_idxs_ptrs = (
            valid_idxs_ptr
            + offs_B * valid_idxs_stride_B
            + branch_idx * valid_idxs_stride_N
        )
        tl.store(next_node_ptrs, val_k, mask=b_mask)
        tl.store(valid_idxs_ptrs, col_k, mask=b_mask)
