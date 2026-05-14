"""Dispatch layer for the fused constrained-node-transition + top-k operation.

Selects the CuTe DSL Python kernel on CUDA when ``cuda-python`` and ``numba``
are available, otherwise falls back to the pure-PyTorch reference that chains
the existing Triton constrained-node-transition kernel with ``torch.topk``.

Public API
----------
The single entry point is :func:`fused_topk_constrained_node_transition`.

Example
-------
.. code-block:: python

    from rectokens.ops.fused_topk_constrained_node_transition import (
        fused_topk_constrained_node_transition,
    )
    from rectokens.schemas.state import ConstraintState

    next_node, valid_idxs, top_k_values, top_k_indices = (
        fused_topk_constrained_node_transition(logits, constraint_state, k=10)
    )
"""

from __future__ import annotations

import math

import torch

from rectokens.schemas.state import ConstraintState


def _ceil_pow2(n: int) -> int:
    """Smallest power of two that is ≥ n (minimum 1)."""
    return 1 if n <= 1 else 2 ** math.ceil(math.log2(n))


# ---------------------------------------------------------------------------
# CuTe DSL CUDA path
# ---------------------------------------------------------------------------

_IS_GPU_AVAILABLE = torch.cuda.is_available()

if _IS_GPU_AVAILABLE:
    from rectokens.kernels.fused_topk_constrained_node_transition_cute import (
        _CUTE_DSL_AVAILABLE,
        _MAX_BRANCHES_SMEM,
        _MAX_N,
        fused_topk_constrained_node_transition_cuda,
    )
else:
    _CUTE_DSL_AVAILABLE = False
    _MAX_N = 4096
    _MAX_BRANCHES_SMEM = 64


def _cuda_fused_topk_cst(
    logits: torch.Tensor,
    constraint_state: ConstraintState,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """CuTe DSL dispatch — requires cuda-python and numba."""
    step = constraint_state.step
    trie = constraint_state.trie
    max_branches = _ceil_pow2(trie.layer_max_branches[step])
    return fused_topk_constrained_node_transition_cuda(
        logits,
        constraint_state.cur_node,
        trie.row_ptrs,
        trie.stacked_cols_vals,
        max_branches,
        k,
    )


# ---------------------------------------------------------------------------
# PyTorch fallback path (always available)
# ---------------------------------------------------------------------------


def _pytorch_fused_topk_cst(
    logits: torch.Tensor,
    constraint_state: ConstraintState,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-PyTorch reference: chains the Triton kernel with ``torch.topk``.

    Used on CPU, when CUDA is available but ``cuda-python``/``numba`` are not
    installed, and as the ground-truth reference in tests.
    """
    from rectokens.ops.constrained_node_transition import constrained_node_transition

    next_node, valid_idxs, corrected_logits = constrained_node_transition(
        logits, constraint_state
    )
    top_k_values, top_k_indices = corrected_logits.topk(k, dim=-1, sorted=True)
    return next_node, valid_idxs, top_k_values, top_k_indices.to(torch.int32)


# ---------------------------------------------------------------------------
# Public dispatch function
# ---------------------------------------------------------------------------


def fused_topk_constrained_node_transition(
    logits: torch.Tensor,
    constraint_state: ConstraintState,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused constrained node transition and top-*k* selection.

    Applies the CSR-trie constraint mask to ``logits`` (setting invalid token
    logits to ``-∞``) and simultaneously returns the top-*k* valid logit values
    and their vocabulary indices, together with the standard trie-transition
    metadata.

    On CUDA with ``cuda-python ≥ 12.4`` and ``numba ≥ 0.57`` installed, the
    two operations are fused into a single CuTe-DSL kernel launch that avoids
    writing the full ``(B, N)`` corrected-logits tensor to global memory.
    Otherwise the function falls back to chaining the existing Triton
    ``constrained_node_transition`` kernel with ``torch.topk``.

    The CuTe path is bypassed automatically when:

    * The input is on CPU.
    * ``cuda-python`` or ``numba`` is not installed.
    * ``N > 4 096`` (vocab size exceeds the kernel's static tile).
    * ``max_branches > 64`` (fan-out exceeds the kernel's shared-memory budget).

    Args:
        logits:           ``(B, N)`` float logit tensor for the current step.
        constraint_state: Trie node and step information.
        k:                Number of top-*k* candidates to return.

    Returns:
        ``(next_node, valid_idxs, top_k_values, top_k_indices)`` where:

        * ``next_node``    — ``(B, max_branches)`` int64 child BFS node IDs.
        * ``valid_idxs``   — ``(B, max_branches)`` int64 valid vocab indices.
        * ``top_k_values`` — ``(B, k)`` float32 top-*k* logit values.
        * ``top_k_indices``— ``(B, k)`` int32  top-*k* vocabulary indices.
    """
    B, N = logits.shape
    step = constraint_state.step
    max_branches = _ceil_pow2(constraint_state.trie.layer_max_branches[step])

    use_cute = (
        logits.is_cuda
        and _CUTE_DSL_AVAILABLE
        and N <= _MAX_N
        and max_branches <= _MAX_BRANCHES_SMEM
    )

    if use_cute:
        return _cuda_fused_topk_cst(logits, constraint_state, k)
    return _pytorch_fused_topk_cst(logits, constraint_state, k)
