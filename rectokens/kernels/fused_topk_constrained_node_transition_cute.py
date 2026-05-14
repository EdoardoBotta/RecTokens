"""Fused constrained-node-transition + top-k kernel written with the
NVIDIA CuTe Python DSL (``cuda.cooperative.experimental``).

Background
----------
The existing Triton kernel (:mod:`rectokens.kernels.constrained_node_transition`)
applies a CSR-trie constraint mask to a ``(B, N)`` logits tensor and returns
corrected logits alongside the trie-transition metadata (``next_node``,
``valid_idxs``).  The downstream beam-search step then calls ``torch.topk`` on
the ``(B, N)`` corrected logits to obtain the top-*k* candidates.

Writing the full ``(B, N)`` corrected logits to global memory and immediately
reading them back for the ``topk`` launch is wasteful: the valid tokens per node
are bounded by ``max_branches`` (typically ≤ 32), so the result is almost
entirely ``-∞``.  This kernel fuses both operations into one pass:

1. Thread 0 in each block traverses the CSR trie to find the ``≤ max_branches``
   valid children of ``cur_node[b]``, loading their logit values from
   ``logits[b, col_k]``.
2. All threads cooperatively perform a **block-level descending sort** of a
   padded logits tile using the CuTe-backed ``merge_sort_pairs`` algorithm from
   ``cuda.cooperative.experimental``.  Masked (invalid) slots are given a large
   positive sentinel in the negated domain so they always sort to the tail.
3. The first ``k`` threads write the top-*k* logit values and vocab indices.
4. Thread 0 (or threads ``0..max_branches-1``) writes the trie metadata
   (``next_node``, ``valid_idxs``).

This eliminates the ``(B, N)`` global write-read round-trip and fuses two
kernel launches into one.

CuTe Python DSL
---------------
``cuda.cooperative.experimental`` exposes NVIDIA's CuTe / CUB block-level
algorithms as Python callables.  The ``block.merge_sort_pairs`` factory returns:

* ``.files``              — list of CUB/CuTe CUDA headers to link at JIT time.
* ``.temp_storage_bytes`` — shared-memory scratch space required by the sort.
* ``__call__(storage, keys, values)`` — cooperative sort invocable inside a
  ``numba.cuda.jit`` kernel; each thread contributes ``ITEMS_PER_THREAD``
  key-value pairs and, after the call, holds ``ITEMS_PER_THREAD`` globally
  sorted items.

Constraints
-----------
* ``N ≤ BLOCK_SIZE * ITEMS_PER_THREAD`` (default 256 × 16 = 4096).  For larger
  vocabs fall back to the Triton kernel + ``torch.topk``.
* ``max_branches ≤ _MAX_BRANCHES_SMEM`` (64).
* Requires ``cuda-python >= 12.4`` and ``numba >= 0.57``.
* CUDA device required.
"""

from __future__ import annotations

import math

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Availability guard — identical pattern to the existing Triton kernels
# ---------------------------------------------------------------------------

try:
    import cuda.cooperative.experimental as cudax  # type: ignore[import]
    import numba  # type: ignore[import]
    import numba.cuda  # type: ignore[import]

    _CUTE_DSL_AVAILABLE = True
except ImportError:
    _CUTE_DSL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Compile-time kernel constants
# ---------------------------------------------------------------------------

# One CUDA thread block per batch element.  Each thread handles ITEMS_PER_THREAD
# logit slots, giving BLOCK_SIZE × ITEMS_PER_THREAD = 4 096 slots per block.
BLOCK_SIZE: int = 256
ITEMS_PER_THREAD: int = 16
_MAX_N: int = BLOCK_SIZE * ITEMS_PER_THREAD  # 4 096

# Upper bound on max_branches stored in shared memory.  The trie fan-out in
# this codebase is at most a few dozen; 64 is a generous ceiling.
_MAX_BRANCHES_SMEM: int = 64

# Sentinel values (in the negated domain used for ascending sort)
_BIG: np.float32 = np.float32(1e30)   # marks a masked/invalid logit slot
_NEG_INF: np.float32 = np.float32(-1e30)  # written to output for unfilled top-k


# ---------------------------------------------------------------------------
# CuTe / CUB block sort algorithm instance
# ---------------------------------------------------------------------------

if _CUTE_DSL_AVAILABLE:
    # Build the cooperative block sort algorithm once at module import time.
    # The algorithm is parametric on (dtype, num_threads, items_per_thread);
    # the same instance is reused across all kernel launches.
    _block_sort = cudax.block.merge_sort_pairs(
        key_dtype=np.float32,
        value_dtype=np.int32,
        num_threads=BLOCK_SIZE,
        items_per_thread=ITEMS_PER_THREAD,
    )

    _SORT_SMEM_BYTES: int = int(_block_sort.temp_storage_bytes)

    # ------------------------------------------------------------------
    # CuTe DSL Python kernel
    # ------------------------------------------------------------------

    @numba.cuda.jit(link=_block_sort.files)  # type: ignore[misc]
    def _fused_topk_cst_cute_kernel(
        logits,           # (B, N)            float32  — input logits
        cur_node,         # (B,)              int64   — current trie node per beam
        csr_row_ptrs,     # (num_nodes + 1,)  int64   — CSR row pointers
        csr_cols_vals,    # (2, E)            int64   — stacked cols (row 0) and
        #                                               child-node values (row 1)
        top_k_values,     # (B, k)            float32  — output: top-k logit values
        top_k_indices,    # (B, k)            int32   — output: top-k vocab indices
        next_node,        # (B, max_branches) int64   — output: trie child IDs
        valid_idxs,       # (B, max_branches) int64   — output: valid vocab indices
        B,                # int — batch size
        N,                # int — vocab / logit dimension
        k,                # int — number of top-k results requested
        max_branches,     # int — padded fan-out (power of 2 ≤ _MAX_BRANCHES_SMEM)
    ):
        """One CUDA thread block per batch element *b*.

        Steps
        -----
        1. Thread 0 walks the CSR trie and fills shared-memory child buffers.
        2. All threads cooperatively initialise the shared masked-logits tile to
           *BIG* (masked sentinel), then thread 0 overwrites valid children with
           their negated logit values (negation turns descending into ascending).
        3. Each thread loads its ``ITEMS_PER_THREAD`` items from shared memory
           into register arrays.
        4. CuTe ``merge_sort_pairs`` sorts the register arrays ascending — the
           globally best (highest) logits emerge at the front of thread 0's
           registers.
        5. The first *k* threads (rank ``0 .. k-1``) scatter top-*k* results to
           global memory, un-negating the logit values.
        6. Threads ``0 .. max_branches-1`` write trie metadata outputs.
        """
        bid = numba.cuda.blockIdx.x
        tid = numba.cuda.threadIdx.x

        if bid >= B:
            return

        # ── Shared memory allocations ─────────────────────────────────────
        # Sort scratch space (required by CuTe merge_sort_pairs)
        smem_sort = numba.cuda.shared.array(
            shape=_SORT_SMEM_BYTES, dtype=numba.types.uint8
        )
        # CSR child data for this node (filled by thread 0, read by all)
        smem_cols = numba.cuda.shared.array(
            shape=_MAX_BRANCHES_SMEM, dtype=numba.types.int64
        )
        smem_vals = numba.cuda.shared.array(
            shape=_MAX_BRANCHES_SMEM, dtype=numba.types.int64
        )
        smem_n_children = numba.cuda.shared.array(shape=1, dtype=numba.types.int32)
        # Masked logits tile (one entry per logit slot, initialised to _BIG)
        smem_masked = numba.cuda.shared.array(shape=_MAX_N, dtype=numba.types.float32)

        # ── Step 1: Thread 0 traverses the CSR trie ──────────────────────
        if tid == 0:
            node = cur_node[bid]
            row_start = csr_row_ptrs[node]
            row_end = csr_row_ptrs[node + 1]
            n = numba.int32(row_end - row_start)
            smem_n_children[0] = n
            for c in range(_MAX_BRANCHES_SMEM):
                if c < n:
                    smem_cols[c] = csr_cols_vals[0, row_start + c]
                    smem_vals[c] = csr_cols_vals[1, row_start + c]
                else:
                    smem_cols[c] = numba.int64(-1)
                    smem_vals[c] = numba.int64(-1)

        numba.cuda.syncthreads()

        # ── Step 2: All threads cooperatively fill smem_masked with _BIG ─
        # (masked sentinel — sorts to the tail of the ascending order)
        for i in range(ITEMS_PER_THREAD):
            smem_masked[tid * ITEMS_PER_THREAD + i] = _BIG

        numba.cuda.syncthreads()

        # ── Step 3: Thread 0 overwrites valid children with negated logits
        # Negation converts "largest logit = best" into "smallest key = best"
        # so the CuTe ascending sort naturally surfaces the top-k candidates.
        if tid == 0:
            n = smem_n_children[0]
            for c in range(n):
                col = numba.int32(smem_cols[c])
                if 0 <= col < N:
                    smem_masked[col] = -logits[bid, col]

        numba.cuda.syncthreads()

        # ── Step 4: Each thread loads its ITEMS_PER_THREAD items ─────────
        thread_keys = numba.cuda.local.array(ITEMS_PER_THREAD, dtype=numba.types.float32)
        thread_values = numba.cuda.local.array(ITEMS_PER_THREAD, dtype=numba.types.int32)
        for i in range(ITEMS_PER_THREAD):
            idx = tid * ITEMS_PER_THREAD + i
            thread_keys[i] = smem_masked[idx]
            thread_values[i] = numba.int32(idx)

        # ── Step 5: CuTe cooperative block sort (ascending on negated keys) ─
        # After this call each thread holds its globally sorted slice:
        #   thread 0 → global ranks 0 .. ITEMS_PER_THREAD-1  (top candidates)
        #   thread 1 → global ranks IPT .. 2*IPT-1
        #   ...
        _block_sort(smem_sort, thread_keys, thread_values)

        # ── Step 6: Write top-k results ───────────────────────────────────
        # global_rank = tid * ITEMS_PER_THREAD + i; we write when rank < k.
        for i in range(ITEMS_PER_THREAD):
            global_rank = tid * ITEMS_PER_THREAD + i
            if global_rank < k:
                nk = thread_keys[i]
                vi = thread_values[i]
                if nk < _BIG * numba.float32(0.5):
                    # Valid constrained logit — un-negate to recover original value
                    top_k_values[bid, global_rank] = -nk
                    top_k_indices[bid, global_rank] = vi
                else:
                    # Fewer than k valid tokens; fill remaining slots with sentinels
                    top_k_values[bid, global_rank] = _NEG_INF
                    top_k_indices[bid, global_rank] = numba.int32(-1)

        # ── Step 7: Write trie-transition metadata ────────────────────────
        # Threads 0 .. max_branches-1 each write one branch slot.
        if tid < max_branches:
            next_node[bid, tid] = smem_vals[tid]
            valid_idxs[bid, tid] = smem_cols[tid]


def _ceil_pow2(n: int) -> int:
    """Smallest power of two that is ≥ n (minimum 1)."""
    return 1 if n <= 1 else 2 ** math.ceil(math.log2(n))


def fused_topk_constrained_node_transition_cuda(
    logits: torch.Tensor,
    cur_node: torch.Tensor,
    csr_row_ptrs: torch.Tensor,
    csr_cols_vals: torch.Tensor,
    max_branches: int,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch the CuTe DSL fused constrained-transition + top-k kernel.

    Args:
        logits:        ``(B, N)`` float32 CUDA tensor of per-step logit scores.
        cur_node:      ``(B,)``   int64  CUDA tensor of current trie node IDs.
        csr_row_ptrs:  ``(num_nodes+1,)`` int64 CSR row-pointer array.
        csr_cols_vals: ``(2, E)`` int64 stacked cols (row 0) and child-node
                       values (row 1).
        max_branches:  Padded fan-out for this trie layer (power of 2).
        k:             Number of top-k logit candidates to return.

    Returns:
        Tuple ``(next_node, valid_idxs, top_k_values, top_k_indices)`` where:

        * ``next_node``    — ``(B, max_branches)`` int64, child BFS IDs.
        * ``valid_idxs``   — ``(B, max_branches)`` int64, valid vocab indices.
        * ``top_k_values`` — ``(B, k)`` float32, top-k constrained logit values.
        * ``top_k_indices``— ``(B, k)`` int32, top-k vocab indices.

    Raises:
        AssertionError: If N > ``_MAX_N`` (4 096) or ``max_branches`` >
            ``_MAX_BRANCHES_SMEM`` (64).
    """
    assert _CUTE_DSL_AVAILABLE, (
        "CuTe DSL kernel requires cuda-python >= 12.4 and numba >= 0.57.  "
        "Install with: pip install 'cuda-python[cooperative]>=12.4' numba"
    )

    B, N = logits.shape
    assert N <= _MAX_N, (
        f"Vocab size N={N} exceeds the kernel maximum _MAX_N={_MAX_N}.  "
        "Increase ITEMS_PER_THREAD or fall back to the Triton kernel."
    )
    assert max_branches <= _MAX_BRANCHES_SMEM, (
        f"max_branches={max_branches} exceeds _MAX_BRANCHES_SMEM={_MAX_BRANCHES_SMEM}."
    )

    logits = logits.contiguous().float()
    cur_node = cur_node.contiguous()
    csr_row_ptrs = csr_row_ptrs.contiguous()
    csr_cols_vals = csr_cols_vals.contiguous()

    # Allocate outputs
    top_k_values = torch.empty(B, k, dtype=torch.float32, device=logits.device)
    top_k_indices = torch.empty(B, k, dtype=torch.int32, device=logits.device)
    next_node_out = torch.full(
        (B, max_branches), -1, dtype=torch.int64, device=logits.device
    )
    valid_idxs_out = torch.full(
        (B, max_branches), -1, dtype=torch.int64, device=logits.device
    )

    _fused_topk_cst_cute_kernel[B, BLOCK_SIZE](
        logits,
        cur_node,
        csr_row_ptrs,
        csr_cols_vals,
        top_k_values,
        top_k_indices,
        next_node_out,
        valid_idxs_out,
        B,
        N,
        k,
        max_branches,
    )

    return next_node_out, valid_idxs_out, top_k_values, top_k_indices
