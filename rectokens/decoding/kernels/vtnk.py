import torch

assert torch.cuda.is_available(), "CUDA is required to import VTNK kernel."

import triton
import triton.language as tl

from rectokens.decoding.schemas.compact_csr_trie import CompactCSRTrie
from rectokens.decoding.vntk import vtnk_pytorch
from rectokens.decoding.kernels.utils import tl_fp32_to_tf32
from torch.library import triton_op
from torch.library import wrap_triton

IS_PTX_RNA_TF32_SUPPORTED = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 8

def fused_linear_constrained_node_transition(
    a: torch.Tensor,
    b: torch.Tensor,
    cur_node: torch.Tensor,
    constraint_transitions: CompactCSRTrie,
    step: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused linear projection + constrained node transition.

    Computes corrected_logits = mask(a @ b) without materializing unconstrained logits.

    Returns (next_node, valid_idxs, corrected_logits):
      next_node:        (B, max_branches) int64 — child BFS IDs, -1 for padding
      valid_idxs:       (B, max_branches) int64 — valid token indices, -1 for padding
      corrected_logits: (B, N) float32 — (a @ b) zeroed for invalid tokens
    """
    max_branches = constraint_transitions.layer_max_branches[step]
    return _fused_linear_constrained_node_transition_op(
        a,
        b,
        cur_node,
        constraint_transitions.row_ptrs,
        constraint_transitions.stacked_cols_vals,
        max_branches,
    )


@triton_op("vtnk::_fused_linear_constrained_node_transition_op", mutates_args={})
def _fused_linear_constrained_node_transition_op(
    a: torch.Tensor,
    b: torch.Tensor,
    cur_node: torch.Tensor,
    csr_row_ptrs: torch.Tensor,
    csr_cols_vals: torch.Tensor,
    max_branches: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, K = a.shape
    N = b.shape[1]

    assert cur_node.shape == (B,), f"Expected cur_node shape (B,), got {cur_node.shape}"

    a = a.contiguous()
    b = b.contiguous()
    cur_node = cur_node.contiguous()
    csr_cols_vals = csr_cols_vals.contiguous()

    corrected_logits = torch.empty(B, N, dtype=torch.float32, device=a.device)
    next_node = cur_node.new_empty(B, max_branches)
    valid_idxs = cur_node.new_empty(B, max_branches)

    grid = lambda meta: (triton.cdiv(B, meta["BLOCK_B"]) * triton.cdiv(N, meta["BLOCK_N"]),)
    wrap_triton(_fused_linear_constrained_node_transition_kernel)[grid](
        a_ptr=a,
        b_ptr=b,
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
        N=N,
        K=K,
        BLOCK_B=64,
        BLOCK_N=64,
        BLOCK_K=32,
        GROUP_SIZE_M=4,
        max_branches=max_branches,
        FP32_TO_TF32_MAX_PRECISION=IS_PTX_RNA_TF32_SUPPORTED,
    )

    return next_node, valid_idxs, corrected_logits
def constrained_node_transition(
    logits: torch.Tensor,
    cur_node: torch.Tensor,
    constraint_transitions: CompactCSRTrie,
    step: int,
    vocab_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Constrained node transition — GPU Triton kernel with CPU fallback.

    Returns (next_node, valid_idxs, corrected_logits):
      next_node:        (B, max_branches) int64 — child BFS IDs, -1 for padding
      valid_idxs:       (B, max_branches) int64 — valid token indices, -1 for padding
      corrected_logits: (B, vocab_size)  float  — logits zeroed for invalid tokens
    """
    max_branches = constraint_transitions.layer_max_branches[step]
    return _constrained_node_transition_op(
        logits,
        cur_node,
        constraint_transitions.row_ptrs,
        constraint_transitions.stacked_cols_vals,
        max_branches,
        vocab_size,
    )


@triton_op("vtnk::_constrained_node_transition_op", mutates_args={})
def _constrained_node_transition_op(
    logits: torch.Tensor,
    cur_node: torch.Tensor,
    csr_row_ptrs: torch.Tensor,
    csr_cols_vals: torch.Tensor,
    max_branches: int,
    vocab_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, N = logits.shape

    assert cur_node.shape == (B,), f"Expected cur_node shape (B,), got {cur_node.shape}"

    logits = logits.contiguous()
    cur_node = cur_node.contiguous()
    csr_cols_vals = csr_cols_vals.contiguous()

    corrected_logits = torch.empty_like(logits)
    next_node = cur_node.new_empty(B, max_branches)
    valid_idxs = cur_node.new_empty(B, max_branches)

    grid = lambda meta: (triton.cdiv(B, meta["BLOCK_B"]) * triton.cdiv(N, meta["BLOCK_N"]),)
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
        BLOCK_B=64,
        BLOCK_N=64,
        GROUP_SIZE_M=4,
        max_branches=max_branches,
    )

    return next_node, valid_idxs, corrected_logits


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
    logits_ptrs = logits_ptr + offs_B[:, None] * logits_stride_B + offs_N[None, :] * logits_stride_N

    logits_mask = (offs_B[:, None] < B) & (offs_N[None, :] < N)
    logits = tl.load(logits_ptrs, mask=logits_mask, other=float('-inf'))
    cur_node = tl.load(cur_node_ptrs, mask=offs_B < B, other=-1)

    csr_row_ptrs = tl.load(csr_trie_row_ptr + cur_node, mask=cur_node >= 0, other=0)
    csr_next_ptrs = tl.load(csr_trie_row_ptr + cur_node + 1, mask=cur_node >= 0, other=0)
    n_children = csr_next_ptrs - csr_row_ptrs

    b_valid = offs_B < B
    logits_correction_mask = tl.zeros([BLOCK_B, BLOCK_N], dtype=tl.int1)
    for k in tl.static_range(max_branches):
        col_k = tl.load(
            csr_trie_cols_vals_ptr + csr_row_ptrs + k,
            mask=b_valid & (n_children > k),
            other=-1,
        )
        logits_correction_mask = logits_correction_mask | (tl.reshape(col_k, [BLOCK_B, 1]) == offs_N[None, :])
    corrected_logits = tl.where(logits_correction_mask, logits, float('-inf'))
    tl.store(corrected_logits_ptr + offs_B[:, None] * corrected_logits_stride_B + offs_N[None, :] * corrected_logits_stride_N, corrected_logits, mask=logits_mask)

    if pid_N == 0:
        slice_range = tl.arange(0, max_branches)
        offs_cols_vals = csr_row_ptrs[:, None] + slice_range
        children_mask = n_children[:, None] > slice_range[None, :]
        cols = tl.load(csr_trie_cols_vals_ptr + offs_cols_vals, mask=children_mask, other=-1)
        next_node_vals = tl.load(csr_trie_cols_vals_ptr + offs_cols_vals + cols_vals_stride_0, mask=children_mask, other=-1)
        next_node_ptrs = next_node_ptr + offs_B[:, None] * next_node_stride_B + tl.arange(0, max_branches) * next_node_stride_N
        valid_idxs_ptrs = valid_idxs_ptr + offs_B[:, None] * valid_idxs_stride_B + tl.arange(0, max_branches) * valid_idxs_stride_N
        tl.store(next_node_ptrs, next_node_vals, mask=offs_B[:, None] < B)
        tl.store(valid_idxs_ptrs, cols, mask=offs_B[:, None] < B)


@triton.jit
def _fused_linear_constrained_node_transition_kernel(
    # Inputs
    a_ptr, b_ptr,
    cur_node_ptr,
    csr_trie_row_ptr,
    csr_trie_cols_vals_ptr,
    a_stride_B,
    a_stride_K,
    b_stride_K,
    b_stride_N,
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
    K: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    max_branches: tl.constexpr,
    FP32_TO_TF32_MAX_PRECISION: tl.constexpr,
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
    offs_K = tl.arange(0, BLOCK_K)

    cur_node_ptrs = cur_node_ptr + offs_B
    a_ptrs = a_ptr + offs_B[:, None] * a_stride_B + offs_K[None, :] * a_stride_K
    b_ptrs = b_ptr + offs_K[:, None] * b_stride_K + offs_N[None, :] * b_stride_N
    
    logits = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=(offs_B[:, None] < B) & (offs_K[None, :] < K - k * BLOCK_K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_K[:, None] < K - k * BLOCK_K) & (offs_N[None, :] < N), other=0.0)

        if FP32_TO_TF32_MAX_PRECISION:
            a = tl_fp32_to_tf32(a)
            b = tl_fp32_to_tf32(b)
        logits = tl.dot(a, b, logits)

        a_ptrs += BLOCK_K * a_stride_K
        b_ptrs += BLOCK_K * b_stride_K

    logits_mask = (offs_B[:, None] < B) & (offs_N[None, :] < N)

    cur_node = tl.load(cur_node_ptrs, mask=offs_B < B, other=-1)
    csr_row_ptrs = tl.load(csr_trie_row_ptr + cur_node, mask=cur_node >= 0, other=0)
    csr_next_ptrs = tl.load(csr_trie_row_ptr + cur_node + 1, mask=cur_node >= 0, other=0)
    n_children = csr_next_ptrs - csr_row_ptrs

    b_valid = offs_B < B
    logits_correction_mask = tl.zeros([BLOCK_B, BLOCK_N], dtype=tl.int1)
    for k in tl.static_range(max_branches):
        col_k = tl.load(
            csr_trie_cols_vals_ptr + csr_row_ptrs + k,
            mask=b_valid & (n_children > k),
            other=-1,
        )
        logits_correction_mask = logits_correction_mask | (col_k[:, None] == offs_N[None, :])
    corrected_logits = tl.where(logits_correction_mask, logits, float('-inf'))
    tl.store(corrected_logits_ptr + offs_B[:, None] * corrected_logits_stride_B + offs_N[None, :] * corrected_logits_stride_N, corrected_logits, mask=logits_mask)

    if pid_N == 0:
        slice_range = tl.arange(0, max_branches)
        offs_cols_vals = csr_row_ptrs[:, None] + slice_range
        children_mask = n_children[:, None] > slice_range[None, :]
        cols = tl.load(csr_trie_cols_vals_ptr + offs_cols_vals, mask=children_mask, other=-1)
        next_node_vals = tl.load(csr_trie_cols_vals_ptr + offs_cols_vals + cols_vals_stride_0, mask=children_mask, other=-1)
        next_node_ptrs = next_node_ptr + offs_B[:, None] * next_node_stride_B + tl.arange(0, max_branches) * next_node_stride_N
        valid_idxs_ptrs = valid_idxs_ptr + offs_B[:, None] * valid_idxs_stride_B + tl.arange(0, max_branches) * valid_idxs_stride_N
        tl.store(next_node_ptrs, next_node_vals, mask=offs_B[:, None] < B)
        tl.store(valid_idxs_ptrs, cols, mask=offs_B[:, None] < B)

