"""
CuTe DSL reimplementation of the fused top-K constrained decoding kernel.

Reimplements ``_fused_linear_constrained_node_transition_topk_op`` using
NVIDIA's CuTe DSL (CUTLASS 4.x Python DSL) instead of Triton.

The kernel fuses:
  1. Sparse linear projection — dot products for valid (trie-constrained) columns.
  2. CSR trie traversal — next-node and valid-index extraction.
  3. Per-branch logit materialization into a compact ``[B, max_branches]`` buffer.

Parallelism strategy
--------------------
Block layout: ``(WARP_SIZE=16, BLOCK_B=16, 1)`` — 256 threads total.
``threadIdx.x`` (lane 0–15) is the reduction dimension; ``threadIdx.y``
(local_b 0–15) selects the batch item within the block.

Because ``WARP_SIZE=16``, two consecutive batch items (local_b 2i and 2i+1)
share one physical CUDA warp (32 threads).  The warp-shuffle reduction therefore
uses offsets ``[8, 4, 2, 1]`` — communicating only within one half-warp — to
avoid mixing accumulators across batch items.

Register staging of ``a``
-------------------------
Each thread pre-loads its slice ``a[batch_idx, lane :: WARP_SIZE]`` into a
small register array before the branch loop, then reuses those registers
across all ``BRANCHES_PER_BLOCK`` dot products.  There is no cross-thread
sharing on ``a`` (lane ``L`` only ever reads positions ``L, L+16, L+32, …``),
so shared memory is unnecessary.

Eliminating per-call MLIR re-generation
----------------------------------------
``@cute.jit`` re-generates MLIR IR from the Python function body on every
call — even on JIT cache hits.  For typical configs this overhead is ~30 ms,
completely dominating the ~1.5 ms GPU kernel.

Fix: ``_FusedTopKKernel.launch()`` calls ``_launch(..., compile_only=True)``
on the first invocation to obtain the compiled ``JitExecutor`` and caches it.
Subsequent calls invoke the executor directly — no MLIR IR is regenerated.

Avoiding JIT recompilation
--------------------------
``@cute.jit`` bakes Python-int arguments into the MLIR module; a different
value forces a full recompile (~75 ms extra).  ``max_branches`` and
``BRANCHES_PER_BLOCK`` influence ``grid_y`` via ``self._max_branches``, a
stable instance attribute.

``_get_kernel`` rounds ``max_branches`` up to the next power of two (bucket)
so one compiled kernel covers all trie steps within that bound.

Residual compile-time dependencies
------------------------------------
``B_val`` and ``K_val`` are extracted from tensor shapes inside ``@cute.jit``
and baked per ``(has_bias, bucket, B, K)`` combination.  Each unique combination
compiles once and hits the executor cache on every subsequent step.
"""

from __future__ import annotations

import atexit
import logging
import warnings

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch

# CuTe DSL emits a WARNING-level log message (via logging.lastResort → stderr)
# and a UserWarning whenever compile_only=True is used.  Both are expected
# behaviour in our JitExecutor caching strategy — suppress them entirely.
#
# The NullHandler prevents logging.lastResort from printing to stderr when no
# other handler is configured for the CUTE_DSL logger.
logging.getLogger("CUTE_DSL").addHandler(logging.NullHandler())
warnings.filterwarnings(
    "ignore",
    message="Cache is disabled as user wants to compile only",
    category=UserWarning,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ceil_power_of_2(n: int) -> int:
    """Smallest power of two >= n."""
    p = 1
    while p < n:
        p <<= 1
    return p


# ---------------------------------------------------------------------------
# Kernel class
# ---------------------------------------------------------------------------


class _FusedTopKKernel:
    """CuTe DSL kernel, one instance per (has_bias, max_branches_bucket, B, K) tuple.

    After the first ``launch()`` call the compiled ``JitExecutor`` is cached;
    all subsequent calls bypass MLIR re-generation entirely.
    """

    BLOCK_B           = 16   # batch items per thread-block (= half-warps)
    WARP_SIZE         = 16   # threads per dot-product reduction (half-warp)
    BRANCHES_PER_BLOCK = 16   # branches handled per block; a[batch,:] is read once
                              # from gmem and reused across all BRANCHES_PER_BLOCK dot products

    def __init__(self, has_bias: bool, max_branches: int, B: int, K: int) -> None:
        self._has_bias     = has_bias
        self._max_branches = max_branches  # stable bucket; never changes per instance
        self._B            = B             # baked into compiled kernel as Python int
        self._K            = K             # baked into compiled kernel as Python int
        self._jit_executor = None          # populated on first launch(), None until then

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
    ):
        # B and K come from instance attributes (Python ints) — keeps alloc_smem
        # and shape arithmetic compile-time constant. The kernel cache is already
        # keyed on (has_bias, bucket, B, K), so each instance has fixed B/K.
        grid_x = (self._B + self.BLOCK_B - 1) // self.BLOCK_B
        grid_y = (self._max_branches + self.BRANCHES_PER_BLOCK - 1) // self.BRANCHES_PER_BLOCK

        self._kernel(
            a, b, bias,
            cur_node, row_ptrs, cols, vals,
            next_node_out, valid_idxs_out, branch_logits_out,
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
    ):
        lane, local_b, _ = cute.arch.thread_idx()
        bidx_x, bidx_y, _ = cute.arch.block_idx()

        batch_idx   = bidx_x * self.BLOCK_B + local_b
        branch_base = bidx_y * self.BRANCHES_PER_BLOCK

        K_PER_LANE = self._K // self.WARP_SIZE

        if batch_idx < self._B:
            # Pre-load a[batch_idx, lane :: WARP_SIZE] into registers.
            # range_constexpr forces a Python-time loop (preprocessor rewrites
            # it to a plain Python range), so a_cache stays a list of
            # K_PER_LANE distinct SSA values (register-allocated).
            # NB: must be a real `for` statement — the preprocessor doesn't
            # rewrite list-comprehensions.
            a_cache = []
            for i in cutlass.range_constexpr(K_PER_LANE):
                a_cache.append(a[batch_idx, lane + i * self.WARP_SIZE])

            node    = cur_node[batch_idx]
            node_32 = node.to(cutlass.Int32)
            if node_32 >= 0:
                row_start  = row_ptrs[node_32]
                row_end    = row_ptrs[node_32 + 1]
                n_children = row_end - row_start

                for j in cutlass.range_constexpr(self.BRANCHES_PER_BLOCK):
                    branch_idx = branch_base + j
                    if branch_idx < n_children.to(cutlass.Int32):
                        offset = row_start.to(cutlass.Int32) + branch_idx
                        col    = cols[offset]
                        val    = vals[offset]
                        col_32 = col.to(cutlass.Int32)

                        # Dot product using cached register values — a is read
                        # from gmem exactly once per (batch, thread).
                        logit = a_cache[0] * b[col_32, lane]
                        for i in cutlass.range_constexpr(1, K_PER_LANE):
                            k = lane + i * self.WARP_SIZE
                            logit = logit + a_cache[i] * b[col_32, k]

                        # Half-warp reduction: WARP_SIZE=16, so two batch items share
                        # one 32-thread physical warp (lanes 0–15 = local_b N,
                        # lanes 16–31 = local_b N+1).  Offset 16 would cross batch
                        # boundaries; use [8, 4, 2, 1] to stay within one half-warp.
                        for shfl_offset in [8, 4, 2, 1]:
                            logit = logit + cute.arch.shuffle_sync_down(logit, shfl_offset)

                        if lane == 0:
                            if self._has_bias:
                                logit = logit + bias[col_32]
                            next_node_out[batch_idx, branch_idx]     = val
                            valid_idxs_out[batch_idx, branch_idx]    = col
                            branch_logits_out[batch_idx, branch_idx] = logit

    # -- fast launcher: bypasses MLIR re-generation after first call --------

    def launch(
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
    ) -> None:
        """Launch without MLIR re-generation on calls after the first.

        First call: ``compile_only=True`` compiles the kernel and caches the
        ``JitExecutor``.  Subsequent calls invoke the executor directly,
        cutting per-call overhead from ~30 ms to ~1.5 ms.
        """
        args = (a, b, bias, cur_node, row_ptrs, cols, vals,
                next_node_out, valid_idxs_out, branch_logits_out)
        if self._jit_executor is None:
            self._jit_executor = self._launch(*args, compile_only=True)
        self._jit_executor(*args)


# ---------------------------------------------------------------------------
# Kernel instance cache
# ---------------------------------------------------------------------------

# Keyed on (has_bias, bucket, B, K) — @cute.jit bakes B and K into the MLIR module,
# so each unique (B, K) combination needs its own compiled JitExecutor.
_kernel_cache: dict[tuple, _FusedTopKKernel] = {}


def _clear_kernel_cache() -> None:
    """Drop all cached JitExecutors before Python tears down the CUDA context.

    JitExecutor.__del__ calls cuda_helpers.unload_cubin_module, but by the
    time Python's normal module-level atexit runs, cuda_helpers may already
    be None — producing a spurious TypeError.  Clearing the cache here
    (before the CUDA runtime is torn down) prevents that.
    """
    for k in _kernel_cache:
        _kernel_cache[k]._jit_executor = None
    _kernel_cache.clear()


atexit.register(_clear_kernel_cache)


def _get_kernel(has_bias: bool, max_branches: int, B: int, K: int) -> _FusedTopKKernel:
    bucket = max(_ceil_power_of_2(max_branches), _FusedTopKKernel.BLOCK_B)
    key = (has_bias, bucket, B, K)
    if key not in _kernel_cache:
        _kernel_cache[key] = _FusedTopKKernel(has_bias, bucket, B, K)
    return _kernel_cache[key]


# ---------------------------------------------------------------------------
# Public op
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

    ``b`` must be ``weight.T`` (shape ``[K, N]``); the kernel receives it
    transposed back to ``[N, K]`` for coalesced column access.

    Returns ``(next_node, valid_idxs, topk_logits, topk_idxs)``.
    """
    B, K = a.shape

    assert cur_node.shape == (B,), f"Expected cur_node shape ({B},), got {cur_node.shape}"
    assert K % _FusedTopKKernel.WARP_SIZE == 0, (
        f"K={K} must be divisible by WARP_SIZE={_FusedTopKKernel.WARP_SIZE}"
    )

    a        = a.contiguous()
    cur_node = cur_node.contiguous()
    bias_val = bias_val.contiguous()

    # b arrives as weight.T ([K, N]).  b.T = weight ([N, K]) is already
    # C-contiguous, so this is a free view — no device copy.
    b_nk = b.T.contiguous()

    next_node     = cur_node.new_full((B, max_branches), -1)
    valid_idxs    = cur_node.new_full((B, max_branches), -1)
    branch_logits = torch.full(
        (B, max_branches), float("-inf"), dtype=torch.float32, device=a.device
    )

    cols = csr_cols_vals[0]
    vals = csr_cols_vals[1]

    _get_kernel(has_bias, max_branches, B, K).launch(
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
    )

    topk_logits, topk_branch_idxs = torch.topk(branch_logits, k, dim=-1)
    topk_idxs = valid_idxs.gather(1, topk_branch_idxs)

    return next_node, valid_idxs, topk_logits, topk_idxs
