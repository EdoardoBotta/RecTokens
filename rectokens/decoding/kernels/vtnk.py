import torch

assert torch.cuda.is_available(), "CUDA is required to import VTNK kernel."

import triton
import triton.language as tl

from rectokens.decoding.schemas.compact_csr_trie import CompactCSRTrie
from rectokens.decoding.vntk import vtnk_pytorch
from torch.library import triton_op
from torch.library import wrap_triton

# TODO: Add merged linear + constrained_node_transition kernel to avoid materializing uncorrected logits to memory.
def constrained_node_transition(
    logits: torch.Tensor,
    cur_node: torch.Tensor,
    constraint_transitions: CompactCSRTrie,
    step: int,
    vocab_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Constrained node transition — GPU Triton kernel with CPU fallback.

    Returns (next_node, corrected_logits):
      next_node:        (B, max_branches) int64 — child BFS IDs, -1 for padding
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
) -> tuple[torch.Tensor, torch.Tensor]:
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

    return next_node, corrected_logits


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

    offs_cols_vals = csr_row_ptrs[:, None] + tl.arange(0, max_branches)
    children_mask = n_children[:, None] > tl.arange(0, max_branches)[None, :]

    cols = tl.load(csr_trie_cols_vals_ptr + offs_cols_vals, mask=children_mask, other=-1)
    next_node_vals = tl.load(csr_trie_cols_vals_ptr + offs_cols_vals + cols_vals_stride_0, mask=children_mask, other=-1)

    logits_correction_mask = tl.sum(cols[:,:,None] == offs_N[None, None, :], axis=1, dtype=tl.int1)
    corrected_logits = tl.where(logits_correction_mask, logits, float('-inf'))
    tl.store(corrected_logits_ptr + offs_B[:, None] * corrected_logits_stride_B + offs_N[None, :] * corrected_logits_stride_N, corrected_logits, mask=logits_mask)

    if pid_N == 0:
        next_node_ptrs = next_node_ptr + offs_B[:, None] * next_node_stride_B + tl.arange(0, max_branches) * next_node_stride_N
        valid_idxs_ptrs = valid_idxs_ptr + offs_B[:, None] * valid_idxs_stride_B + tl.arange(0, max_branches) * valid_idxs_stride_N
        tl.store(next_node_ptrs, next_node_vals, mask=offs_B[:, None] < B)
        tl.store(valid_idxs_ptrs, cols, mask=offs_B[:, None] < B)

