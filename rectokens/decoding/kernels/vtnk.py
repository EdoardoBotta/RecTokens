import torch

assert torch.cuda.is_available(), "CUDA is required to import VTNK kernel."

import triton
import triton.language as tl

from rectokens.decoding.schemas.compact_csr_trie import CompactCSRTrie
from rectokens.decoding.vntk import vtnk_pytorch
from torch.library import triton_op
from torch.library import wrap_triton


@triton_op("vtnk::constrained_node_transition", mutates_args={})
def constrained_node_transition(
    logits: torch.Tensor,
    cur_node: torch.Tensor,
    constraint_transtions: CompactCSRTrie,
    step: int,
    vocab_size: int):

    B, N = logits.shape

    assert cur_node.shape == (B,), f"Expected cur_node shape to be (B,), got {cur_node.shape}"
    assert step < len(constraint_transtions.layer_max_branches), f"Step {step} exceeds max depth of trie {len(constraint_transtions.layer_max_branches)}"
    assert vocab_size >= constraint_transtions.stacked_cols_vals[0].max() + 1, f"Vocab size {vocab_size} does not match trie"

    logits = logits.contiguous()
    cur_node = cur_node.contiguous()
    csr_trie_cols_vals = constraint_transtions.stacked_cols_vals.contiguous()
    csr_trie_rows = constraint_transtions.row_ptrs

    corrected_logits = torch.empty_like(logits)
    max_branches = constraint_transtions.layer_max_branches[step]
    next_node = torch.empty(B, max_branches, dtype=cur_node.dtype, device=cur_node.device)

    grid = lambda meta: (triton.cdiv(B, meta["BLOCK_B"])*triton.cdiv(N, meta["BLOCK_N"]),)
    wrap_triton(constrained_node_transition_kernel)[grid](
        logits_ptr=logits,
        cur_node_ptr=cur_node,
        csr_trie_row_ptr=csr_trie_rows,
        csr_trie_cols_vals_ptr=csr_trie_cols_vals,
        logits_stride_B=logits.stride(0),
        logits_stride_N=logits.stride(1),
        cols_vals_stride_0=csr_trie_cols_vals.stride(0),
        corrected_logits_ptr=corrected_logits,
        next_node_ptr=next_node,
        corrected_logits_stride_B=corrected_logits.stride(0),
        corrected_logits_stride_N=corrected_logits.stride(1),
        B=B,
        N=N,
        BLOCK_B=64,
        BLOCK_N=64,
        GROUP_SIZE_M=4,
        max_branches=max_branches,
    )

    return next_node, corrected_logits

@triton.jit
def constrained_node_transition_kernel(
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
    corrected_logits_stride_B,
    corrected_logits_stride_N,
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

    # Build valid_mask[BLOCK_B, BLOCK_N]: True if token offs_N[n] is a valid child of cur_node[b]
    valid_mask = tl.zeros([BLOCK_B, BLOCK_N], dtype=tl.int1)
    for k in tl.static_range(max_branches):
        col_k = tl.reshape(cols[:, k], [BLOCK_B, 1])       # (BLOCK_B, 1)
        child_ok = tl.reshape(children_mask[:, k], [BLOCK_B, 1])  # (BLOCK_B, 1)
        match = (col_k == offs_N[None, :]) & child_ok
        valid_mask = valid_mask | match

    corrected_logits = tl.where(valid_mask, logits, float('-inf'))
    cl_ptrs = corrected_logits_ptr + offs_B[:, None] * corrected_logits_stride_B + offs_N[None, :] * corrected_logits_stride_N
    tl.store(cl_ptrs, corrected_logits, mask=logits_mask)

    # Write next_node[b, k] — only the first N-tile writes to avoid redundant stores racing
    offs_k = tl.arange(0, max_branches)
    nn_ptrs = next_node_ptr + offs_B[:, None] * max_branches + offs_k[None, :]
    nn_mask = offs_B[:, None] < B
    if pid_N == 0:
        tl.store(nn_ptrs, next_node_vals, mask=nn_mask)


constrained_node_transition.register_kernel("cpu")
def constrained_node_transition_cpu(
    logits: torch.Tensor,
    cur_node: torch.Tensor,
    constraint_transtions: CompactCSRTrie,
    step: int,
    vocab_size: int
):
    next_node, _, corrected_logits = vtnk_pytorch(logits, cur_node, constraint_transtions, step, vocab_size)
    return next_node, corrected_logits
