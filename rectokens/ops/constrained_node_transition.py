import torch

from rectokens.kernels.constrained_node_transition import (
    _constrained_node_transition_op,
    _fused_linear_constrained_node_transition_op,
)
from rectokens.schemas.compact_csr_trie import CompactCSRTrie

def fused_linear_constrained_node_transition(
    a: torch.Tensor,
    b: torch.Tensor,
    cur_node: torch.Tensor,
    constraint_transitions: CompactCSRTrie,
    step: int,
    bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused linear projection + constrained node transition.

    Only computes dot products for valid (constrained) tokens — skips all others.
    Expects b = weight.T as a non-contiguous transposed view so that each column
    of b (= row of weight) is contiguous in memory.

    Returns (next_node, valid_idxs, corrected_logits):
      next_node:        (B, max_branches) int64 — child BFS IDs, -1 for padding
      valid_idxs:       (B, max_branches) int64 — valid token indices, -1 for padding
      corrected_logits: (B, N) float32 — logits for valid tokens, -inf elsewhere
    """
    # Use a dummy bias tensor when none is provided; has_bias gates all loads.
    bias_val = bias if bias is not None else a.new_empty(0)
    return _fused_linear_constrained_node_transition_op(
        a,
        b,
        bias_val,
        cur_node,
        constraint_transitions.row_ptrs,
        constraint_transitions.stacked_cols_vals,
        constraint_transitions.layer_max_branches[step],
        bias is not None,
    )


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
