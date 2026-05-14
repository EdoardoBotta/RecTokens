"""
CuTe DSL reimplementation of the fused top-K constrained decoding kernel.

Reimplements ``_fused_linear_constrained_node_transition_topk_op`` using
NVIDIA's CuTe DSL (CUTLASS 4.x Python DSL) instead of Triton.

The kernel fuses:
  1. Sparse linear projection — dot products computed only for valid
     (trie-constrained) token columns.
  2. CSR trie traversal — next-node and valid-index extraction.
  3. Per-branch logit materialization into a compact ``[B, max_branches]``
     buffer that is then fed to ``torch.topk`` on the host.

Parallelism strategy
--------------------
Block layout: ``(WARP_SIZE=32, BLOCK_B=4, 1)`` — 128 threads total.

- ``threadIdx.x = lane``    (0..31): parallelises the K dimension within one dot product.
- ``threadIdx.y = local_b`` (0..BLOCK_B-1): four independent (batch, branch) pairs per block.

Because ``blockDim.x == WARP_SIZE``, each y-stripe is exactly one CUDA warp and all
32 lanes share the same ``batch_idx``.  The three guarding conditions
(``batch_idx < B``, ``node >= 0``, ``branch_idx < n_children``) are therefore
*warp-uniform* — no divergence.

Memory access pattern
---------------------
``b`` is passed as weight (shape ``[N, K]``), NOT as ``weight.T``.  This makes
``b[col, k]`` stride-1 in ``k``.  With all 32 lanes holding the same ``col``
and consecutive ``lane`` values, the loads for ``a[batch, lane+i*32]`` and
``b[col, lane+i*32]`` are fully coalesced within each warp.

Warp reduction
--------------
After each lane accumulates its K/32 partial products, a butterfly
``shuffle_sync_down`` reduces the 32 partial sums into lane 0, which writes
the final logit.

MLIR trace cost
---------------
The serial-K loop in the original kernel traced 512 MLIR operations per call
(even on JIT cache hits).  The warp-parallel loop traces only K/WARP_SIZE = 16
operations, cutting per-call Python overhead ~32×.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch


# ---------------------------------------------------------------------------
# Kernel class — compile-time ``has_bias`` specialisation via ``self``
# ---------------------------------------------------------------------------


class _FusedTopKKernel:
    """Encapsulates the CuTe DSL fused sparse-linear + constrained node
    transition + top-K kernel.

    Block layout: ``(WARP_SIZE, BLOCK_B, 1)``.  Each warp (32 threads in x)
    cooperates on a single dot product; BLOCK_B warps per block handle
    independent batch items simultaneously.
    """

    BLOCK_B: int = 4            # warps (= batch items) per thread-block
    WARP_SIZE: int = 32         # threads per dot product; must divide K evenly

    def __init__(self, has_bias: bool) -> None:
        self._has_bias = has_bias

    # -- host-side launcher (JIT-compiled) ----------------------------------

    @cute.jit
    def _launch(
        self,
        a: cute.Tensor,              # [B, K] float32
        b: cute.Tensor,              # [N, K] float32  (un-transposed weight)
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
        K_val = a.shape[1]
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
            B_val,
            K_val,
        ).launch(
            grid=(grid_x, grid_y, 1),
            block=(self.WARP_SIZE, self.BLOCK_B, 1),
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
        B_val,    # compile-time int
        K_val,    # compile-time int — must be divisible by WARP_SIZE
    ):
        # lane = threadIdx.x (0..WARP_SIZE-1), local_b = threadIdx.y (0..BLOCK_B-1).
        # Each y-stripe is exactly one warp; all 32 lanes share the same batch_idx.
        lane, local_b, _ = cute.arch.thread_idx()
        bidx_x, bidx_y, _ = cute.arch.block_idx()

        batch_idx = bidx_x * self.BLOCK_B + local_b   # i32
        branch_idx = bidx_y                            # i32

        if batch_idx < B_val:
            node = cur_node[batch_idx]
            node_32 = node.to(cutlass.Int32)
            if node_32 >= 0:
                row_start = row_ptrs[node_32]
                row_end = row_ptrs[node_32 + 1]
                n_children = row_end - row_start

                if branch_idx < n_children.to(cutlass.Int32):
                    offset = row_start.to(cutlass.Int32) + branch_idx
                    col = cols[offset]    # i64 vocab index
                    val = vals[offset]    # i64 next-node id
                    col_32 = col.to(cutlass.Int32)

                    # --- Parallel dot product over K ---
                    #
                    # b is [N, K] (un-transposed weight), so b[col, k] has stride 1 in k.
                    # a[batch, k] also has stride 1 in k.
                    #
                    # Within a warp all lanes share col_32.  For iteration i:
                    #   lane 0..31 access a[batch, i*32+0 .. i*32+31]   → coalesced
                    #   lane 0..31 access b[col,   i*32+0 .. i*32+31]   → coalesced
                    #
                    # K_val // WARP_SIZE is compile-time → loop is fully unrolled (16 iters
                    # for K=512).  This also reduces MLIR trace cost from 512→16 ops/call.
                    logit = a[batch_idx, lane] * b[col_32, lane]
                    for i in range(1, K_val // self.WARP_SIZE):
                        k = lane + i * self.WARP_SIZE
                        logit = logit + a[batch_idx, k] * b[col_32, k]

                    # --- Warp butterfly reduction ---
                    # After 5 rounds, lane 0 holds sum of all 32 partial products.
                    for shfl_offset in [16, 8, 4, 2, 1]:
                        logit = logit + cute.arch.shuffle_sync_down(logit, shfl_offset)

                    # Only lane 0 writes; other lanes' values are intermediate sums.
                    if lane == 0:
                        if self._has_bias:
                            logit = logit + bias[col_32]

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
    ``b`` must be ``weight.T`` (shape ``[K, N]``); the kernel receives it
    transposed back to ``[N, K]`` for coalesced column access.

    Returns ``(next_node, valid_idxs, topk_logits, topk_idxs)``.
    """
    B, K = a.shape

    assert cur_node.shape == (B,), (
        f"Expected cur_node shape ({B},), got {cur_node.shape}"
    )
    assert K % _FusedTopKKernel.WARP_SIZE == 0, (
        f"K={K} must be divisible by WARP_SIZE={_FusedTopKKernel.WARP_SIZE}"
    )

    a = a.contiguous()
    cur_node = cur_node.contiguous()
    csr_cols_vals = csr_cols_vals.contiguous()
    bias_val = bias_val.contiguous()

    # b arrives as weight.T (shape [K, N]).  b.T = weight (shape [N, K]),
    # which is already C-contiguous when weight was originally C-contiguous —
    # so this is a free view, not a copy.  The kernel accesses b[col, k] with
    # stride 1 in k, giving fully coalesced loads within each warp.
    b_nk = b.T.contiguous()

    next_node = cur_node.new_full((B, max_branches), -1)
    valid_idxs = cur_node.new_full((B, max_branches), -1)
    branch_logits = torch.full(
        (B, max_branches), float("-inf"), dtype=torch.float32, device=a.device
    )

    cols = csr_cols_vals[0]
    vals = csr_cols_vals[1]

    _get_kernel(has_bias)._launch(
        from_dlpack(a),
        from_dlpack(b_nk),
        from_dlpack(bias_val),
        from_dlpack(cur_node),
        from_dlpack(csr_row_ptrs),
        from_dlpack(cols),
        from_dlpack(vals),
        from_dlpack(next_node),
        from_dlpack(valid_idxs),
        from_dlpack(branch_logits),
        max_branches,
    )

    topk_logits, topk_branch_idxs = torch.topk(branch_logits, k, dim=-1)
    topk_idxs = valid_idxs.gather(1, topk_branch_idxs)

    return next_node, valid_idxs, topk_logits, topk_idxs
