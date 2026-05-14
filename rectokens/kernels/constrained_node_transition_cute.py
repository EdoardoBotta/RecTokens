"""
CuTe DSL reimplementation of the fused top-k constrained decoding kernel.

Reimplements ``_fused_linear_constrained_node_transition_topk_op`` using
NVIDIA's CuTe DSL (CUTLASS 4.x Python DSL) instead of Triton.

The kernel fuses:
  1. Sparse linear projection — dot products computed only for valid
     (trie-constrained) token columns.
  2. CSR trie traversal — next-node and valid-index extraction.
  3. Per-branch logit materialization into a compact ``[B, max_branches]``
     buffer that is then fed to ``torch.topk`` on the host.

CuTe DSL compiles the ``@cute.kernel`` function to PTX via MLIR/LLVM,
giving full control over the thread ↔ data mapping while still accepting
PyTorch tensors at the call-site via DLPack.
"""

from __future__ import annotations

import cutlass.cute as cute
import torch


# ---------------------------------------------------------------------------
# Kernel class — compile-time ``has_bias`` specialisation via ``self``
# ---------------------------------------------------------------------------


class _FusedTopKKernel:
    """Encapsulates the CuTe DSL fused sparse-linear + constrained node
    transition + top-K kernel.

    ``has_bias`` is a class attribute and therefore a compile-time constant
    inside ``@cute.kernel`` methods (the preprocessor inlines it).
    """

    BLOCK_B: int = 128

    def __init__(self, has_bias: bool) -> None:
        self._has_bias = has_bias

    # -- host-side launcher (JIT-compiled) ----------------------------------

    @cute.jit
    def _launch(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        bias: cute.Tensor,
        cur_node: cute.Tensor,
        row_ptrs: cute.Tensor,
        cols: cute.Tensor,
        vals: cute.Tensor,
        next_node_out: cute.Tensor,
        valid_idxs_out: cute.Tensor,
        branch_logits_out: cute.Tensor,
        max_branches_val,
    ):
        B_val = a.shape[0]
        grid_x = (B_val + self.BLOCK_B - 1) // self.BLOCK_B
        grid_y = max_branches_val

        self._kernel(
            a,
            b,
            bias,
            cur_node,
            row_ptrs,
            cols,
            vals,
            next_node_out,
            valid_idxs_out,
            branch_logits_out,
        ).launch(
            grid=(grid_x, grid_y, 1),
            block=(self.BLOCK_B, 1, 1),
        )

    # -- device kernel -------------------------------------------------------

    @cute.kernel
    def _kernel(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        bias: cute.Tensor,
        cur_node: cute.Tensor,
        row_ptrs: cute.Tensor,
        cols: cute.Tensor,
        vals: cute.Tensor,
        next_node_out: cute.Tensor,
        valid_idxs_out: cute.Tensor,
        branch_logits_out: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx_x, bidx_y, _ = cute.arch.block_idx()
        bdim, _, _ = cute.arch.block_dim()

        B_val, K_val = a.shape
        batch_idx = bidx_x * bdim + tidx
        branch_idx = bidx_y

        if batch_idx < B_val:
            node = cur_node[batch_idx]
            if node >= 0:
                row_start = row_ptrs[node]
                row_end = row_ptrs[node + 1]
                n_children = row_end - row_start

                if branch_idx < n_children:
                    col = cols[row_start + branch_idx]
                    val = vals[row_start + branch_idx]

                    # Dot-product: a[batch, :] · b[:, col]
                    # Initialise from the first product to keep float32.
                    logit = a[batch_idx, 0] * b[0, col]
                    for k in range(1, K_val):
                        logit = logit + a[batch_idx, k] * b[k, col]

                    if self._has_bias:
                        logit = logit + bias[col]

                    next_node_out[batch_idx, branch_idx] = val
                    valid_idxs_out[batch_idx, branch_idx] = col
                    branch_logits_out[batch_idx, branch_idx] = logit


# ---------------------------------------------------------------------------
# Kernel instance cache
# ---------------------------------------------------------------------------

_kernel_cache: dict[bool, _FusedTopKKernel] = {}


def _get_kernel(has_bias: bool) -> _FusedTopKKernel:
    if has_bias not in _kernel_cache:
        _kernel_cache[has_bias] = _FusedTopKKernel(has_bias)
    return _kernel_cache[has_bias]


# ---------------------------------------------------------------------------
# Public op — drop-in replacement for the Triton ``_fused_linear_…_topk_op``
# ---------------------------------------------------------------------------


def _cute_fused_linear_constrained_node_transition_topk_op(
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
    """CuTe DSL fused linear + constrained-node-transition + top-K.

    Interface mirrors the Triton ``_fused_linear_constrained_node_transition_topk_op``.

    Returns ``(next_node, valid_idxs, topk_logits, topk_idxs)``.
    """
    B, K = a.shape

    assert cur_node.shape == (B,), (
        f"Expected cur_node shape ({B},), got {cur_node.shape}"
    )

    a = a.contiguous()
    cur_node = cur_node.contiguous()
    csr_cols_vals = csr_cols_vals.contiguous()
    bias_val = bias_val.contiguous()

    # Pre-fill outputs with sentinel values so the kernel only writes valid
    # entries — invalid slots keep -1 / -inf.
    next_node = cur_node.new_full((B, max_branches), -1)
    valid_idxs = cur_node.new_full((B, max_branches), -1)
    branch_logits = torch.full(
        (B, max_branches), float("-inf"), dtype=torch.float32, device=a.device
    )

    # ``stacked_cols_vals`` is [2, nnz+1]: row 0 = columns, row 1 = values.
    cols = csr_cols_vals[0]
    vals = csr_cols_vals[1]

    _get_kernel(has_bias)._launch(
        a,
        b,
        bias_val,
        cur_node,
        csr_row_ptrs,
        cols,
        vals,
        next_node,
        valid_idxs,
        branch_logits,
        max_branches,
    )

    # Pass 2: top-K on the compact [B, max_branches] buffer — no kernel needed.
    topk_logits, topk_branch_idxs = torch.topk(branch_logits, k, dim=-1)
    topk_idxs = valid_idxs.gather(1, topk_branch_idxs)

    return next_node, valid_idxs, topk_logits, topk_idxs
