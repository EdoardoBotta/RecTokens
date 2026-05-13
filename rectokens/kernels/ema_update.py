"""Fused Triton kernel for VQ-codebook EMA update.

Fuses all five steps of :meth:`VQQuantizer._ema_update` into a single GPU kernel pass:

1. **Scatter-accumulate** — for each codebook entry *k*, scan the batch and sum
   the encoder outputs ``x[i]`` whose assigned code equals *k*, accumulating
   ``cluster_size[k]`` and ``embed_sum[k]`` without materialising the
   ``(B, K)`` one-hot matrix.
2. **EMA update** — blend the new statistics into the running EMA buffers,
   restricted to codes that received at least one assignment (*active-only*
   update).
3. **Codebook refresh** — recompute each active codebook entry as
   ``ema_embed_sum[k] / max(ema_cluster_size[k], ε)``.
4. **Dead-code counter** — reset ``steps_since_active`` to 0 for active codes
   and increment by 1 for inactive ones.
5. **Dead-code restart** — replace stranded codes (those whose counter reaches
   ``restart_after_steps``) with a random encoder output drawn from the current
   batch, and zero out their EMA accumulators.

All five steps are executed in a single kernel launch per codebook level,
eliminating the intermediate ``(B, K)`` allocation of the PyTorch reference and
merging many element-wise passes into one.

Grid shape: ``(K,)`` — one thread block per codebook entry.  This avoids
inter-block atomics at the cost of reading the full ``(B,)`` codes array ``K``
times.  For the typical regime (K ≤ 4096, B ≤ 32768) the resulting memory
traffic is dominated by the ``x`` reads, which are L2-cached across blocks.

Constraints (matching the existing kernel style):
    * ``D`` must be a power of two (used as ``tl.constexpr``; Triton recompiles
      one kernel variant per unique ``D`` value).
    * All tensors must be contiguous and on CUDA (enforced by the Python
      wrapper in :mod:`rectokens.ops.ema_update`).
"""

from __future__ import annotations

import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_B": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_B": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_B": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_B": 256}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_B": 512}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_B": 1024}, num_warps=16, num_stages=4),
    ],
    key=["B", "K", "D"],
    restore_value=[
        "ema_cluster_size_ptr",
        "ema_embed_sum_ptr",
        "codebook_ptr",
        "steps_since_active_ptr",
    ],
)
@triton.jit
def ema_update_kernel(
    # ── inputs ──────────────────────────────────────────────────────────────
    x_ptr,                    # (B, D) fp32 — encoder outputs (read-only)
    codes_ptr,                # (B,)   int64 — nearest-code indices (read-only)
    rand_idx_ptr,             # (K,)   int64 — pre-drawn random batch indices for
    #                           dead-code replacement (read-only)
    decay,                    # scalar fp32 — EMA decay factor γ
    restart_after_steps,      # scalar int32 — dead-code restart threshold
    B,                        # int — batch size
    K,                        # int — codebook size (== grid dimension)
    D: tl.constexpr,          # constexpr int — embedding dimension
    x_stride_B,               # stride of x along the batch axis
    x_stride_D,               # stride of x along the feature axis
    es_stride_K,              # stride of ema_embed_sum along K axis
    es_stride_D,              # stride of ema_embed_sum along D axis
    cb_stride_K,              # stride of codebook along K axis
    cb_stride_D,              # stride of codebook along D axis
    # ── in-place buffers ────────────────────────────────────────────────────
    ema_cluster_size_ptr,     # (K,)    fp32 — EMA cluster-size statistics
    ema_embed_sum_ptr,        # (K, D)  fp32 — EMA embedding-sum statistics
    codebook_ptr,             # (K, D)  fp32 — codebook embeddings
    steps_since_active_ptr,   # (K,)   int64 — consecutive inactive steps per code
    BLOCK_B: tl.constexpr,    # autotuned batch-tile size
):
    """One program per codebook entry *k*.

    Scans the batch in ``BLOCK_B``-sized tiles, accumulates the statistics for
    entry *k*, then writes back the updated EMA buffers, codebook entry, and
    dead-code counter — all without communicating with other programs.
    """
    k = tl.program_id(0)

    offs_D = tl.arange(0, D)  # (D,) — D-dimension offsets (D is constexpr)

    # ── Step 1: scatter-accumulate cluster_size and embed_sum ────────────────
    # Scan every batch sample; accumulate those whose code equals k.
    cluster_size = 0.0                               # scalar fp32 accumulator
    embed_sum = tl.zeros((D,), dtype=tl.float32)    # (D,) fp32 accumulator

    for b_start in range(0, tl.cdiv(B, BLOCK_B)):
        offs_B = b_start * BLOCK_B + tl.arange(0, BLOCK_B)
        b_mask = offs_B < B

        # Load code assignments for this batch tile; use int32 for comparison
        # (codebook size K < 2^31 in all practical settings).
        batch_codes = tl.load(codes_ptr + offs_B, mask=b_mask, other=-1).to(tl.int32)
        matches = (batch_codes == k) & b_mask  # (BLOCK_B,) bool

        # Count assignments for code k in this tile.
        cluster_size += tl.sum(matches.to(tl.float32))

        # Load encoder outputs for the tile; zero-fill out-of-bounds rows.
        x_block = tl.load(
            x_ptr + offs_B[:, None] * x_stride_B + offs_D[None, :] * x_stride_D,
            mask=b_mask[:, None],
            other=0.0,
        ).to(tl.float32)  # (BLOCK_B, D)

        # Accumulate x[i] for every matched sample.
        embed_sum = embed_sum + tl.sum(
            tl.where(matches[:, None], x_block, tl.zeros_like(x_block)),
            axis=0,
        )  # (D,)

    # ── Step 2: active-only EMA update ──────────────────────────────────────
    ema_cs = tl.load(ema_cluster_size_ptr + k).to(tl.float32)  # scalar
    ema_es = tl.load(
        ema_embed_sum_ptr + k * es_stride_K + offs_D * es_stride_D
    ).to(tl.float32)  # (D,)

    active = cluster_size > 0.0  # scalar bool

    new_ema_cs = decay * ema_cs + (1.0 - decay) * cluster_size
    new_ema_es = decay * ema_es + (1.0 - decay) * embed_sum  # (D,)

    # Only overwrite statistics for codes that received an assignment;
    # inactive codes keep their existing EMA values.
    updated_ema_cs = tl.where(active, new_ema_cs, ema_cs)           # scalar
    updated_ema_es = tl.where(active, new_ema_es, ema_es)           # (D,)

    # ── Step 3: recompute codebook entry for active codes ───────────────────
    n = tl.maximum(updated_ema_cs, 1e-5)     # avoid division by zero
    new_embedding = updated_ema_es / n       # (D,)

    # ── Step 4: dead-code counter ────────────────────────────────────────────
    steps = tl.load(steps_since_active_ptr + k)  # int64 scalar
    # Reset to 0 for active codes; increment by 1 for inactive ones.
    # Multiply to preserve int64 type without a literal cast.
    new_steps = tl.where(active, steps * 0, steps + 1)  # int64 scalar
    dead = new_steps >= restart_after_steps              # bool scalar

    # ── Step 5: dead-code restart ────────────────────────────────────────────
    # Load the pre-drawn random replacement sample from the batch.
    rand_b = tl.load(rand_idx_ptr + k)
    replacement = tl.load(
        x_ptr + rand_b * x_stride_B + offs_D * x_stride_D
    ).to(tl.float32)  # (D,)

    # Current codebook entry — needed for inactive, non-dead codes.
    cur_embedding = tl.load(
        codebook_ptr + k * cb_stride_K + offs_D * cb_stride_D
    ).to(tl.float32)  # (D,)

    # Priority: dead overrides active; active overrides unchanged.
    final_embedding = tl.where(
        dead,
        replacement,
        tl.where(active, new_embedding, cur_embedding),
    )  # (D,)
    final_ema_cs = tl.where(dead, 0.0, updated_ema_cs)              # scalar
    final_ema_es = tl.where(
        dead, tl.zeros((D,), dtype=tl.float32), updated_ema_es
    )  # (D,)
    final_steps = tl.where(dead, steps * 0, new_steps)              # int64

    # ── Write back ───────────────────────────────────────────────────────────
    tl.store(ema_cluster_size_ptr + k, final_ema_cs)
    tl.store(
        ema_embed_sum_ptr + k * es_stride_K + offs_D * es_stride_D,
        final_ema_es,
    )
    tl.store(
        codebook_ptr + k * cb_stride_K + offs_D * cb_stride_D,
        final_embedding,
    )
    tl.store(steps_since_active_ptr + k, final_steps)
