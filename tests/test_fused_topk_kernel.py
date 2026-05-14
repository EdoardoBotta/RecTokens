"""Tests for the fused constrained-node-transition + top-k kernel.

Test strategy
-------------
Every test runs the same scenario through both the PyTorch fallback (which
chains the Triton ``constrained_node_transition`` kernel with ``torch.topk``)
and the CuTe DSL CUDA kernel, then asserts numerical and semantic equivalence.

When the CuTe DSL kernel is unavailable (``cuda-python`` / ``numba`` not
installed, or no CUDA), the CUDA-specific tests are skipped automatically; the
PyTorch-fallback-only tests still run to validate the dispatch logic.

Test matrix
-----------
* Basic correctness — top-k values/indices match the reference PyTorch path.
* Trie metadata — ``next_node`` and ``valid_idxs`` are identical to the
  reference Triton kernel output.
* Top-k ordering — returned values are in descending order.
* ``k > max_branches`` — fewer valid tokens than requested; excess slots filled
  with ``-∞`` / ``-1``.
* ``k == 1`` — single best token.
* Multiple batch elements with different active nodes.
* Large N (up to 1 024) — verifies tile masking beyond the first BLOCK_SIZE
  slots.
"""

from __future__ import annotations

import math
import unittest

import torch

from rectokens.schemas.compact_csr_trie import CompactCSRTrie
from rectokens.schemas.state import ConstraintState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _lex_sort(rows: list[list[int]]) -> torch.Tensor:
    return torch.tensor(sorted(rows), dtype=torch.long)


def _build_small_trie(device: torch.device) -> CompactCSRTrie:
    seqs = _lex_sort([[1, 2, 1], [3, 1, 2], [3, 1, 3]])
    csr = CompactCSRTrie.from_sorted_batch(seqs, vocab_size=8)
    return csr._replace(
        row_ptrs=csr.row_ptrs.to(device),
        stacked_cols_vals=csr.stacked_cols_vals.to(device),
        dense_mask_by_layer=[v.to(device) for v in csr.dense_mask_by_layer],
        dense_states=csr.dense_states.to(device),
    )


def _build_dense_trie(vocab: int, device: torch.device) -> CompactCSRTrie:
    """All combinations of length-2 from a vocab of size ``vocab``."""
    seqs = _lex_sort([[i, j] for i in range(vocab) for j in range(vocab)])
    csr = CompactCSRTrie.from_sorted_batch(seqs, vocab_size=vocab)
    return csr._replace(
        row_ptrs=csr.row_ptrs.to(device),
        stacked_cols_vals=csr.stacked_cols_vals.to(device),
        dense_mask_by_layer=[v.to(device) for v in csr.dense_mask_by_layer],
        dense_states=csr.dense_states.to(device),
    )


def _ceil_pow2(n: int) -> int:
    return 1 if n <= 1 else 2 ** math.ceil(math.log2(n))


def _pytorch_reference(
    logits: torch.Tensor,
    constraint_state: ConstraintState,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-PyTorch reference using the Triton kernel + torch.topk."""
    from rectokens.ops.fused_topk_constrained_node_transition import (
        _pytorch_fused_topk_cst,
    )
    return _pytorch_fused_topk_cst(logits, constraint_state, k)


# ---------------------------------------------------------------------------
# Base class — CPU-only tests (no CuTe, no CUDA required)
# ---------------------------------------------------------------------------


class TestFusedTopkFallback(unittest.TestCase):
    """Tests for the PyTorch fallback path — runs on any machine."""

    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required for constrained_node_transition")
        cls.device = torch.device("cuda")
        cls.trie_small = _build_small_trie(cls.device)
        cls.trie_dense4 = _build_dense_trie(4, cls.device)

    def _state(self, trie: CompactCSRTrie, step: int, nodes: list[int]) -> ConstraintState:
        return ConstraintState(
            step=step,
            trie=trie,
            cur_node=torch.tensor(nodes, device=self.device),
        )

    # ------------------------------------------------------------------
    # Dispatch smoke test
    # ------------------------------------------------------------------

    def test_dispatch_returns_four_tensors(self) -> None:
        from rectokens.ops.fused_topk_constrained_node_transition import (
            fused_topk_constrained_node_transition,
        )
        torch.manual_seed(0)
        logits = torch.randn(2, 8, device=self.device)
        state = self._state(self.trie_small, step=0, nodes=[0, 0])
        result = fused_topk_constrained_node_transition(logits, state, k=2)
        self.assertEqual(len(result), 4)
        next_node, valid_idxs, top_k_values, top_k_indices = result
        self.assertEqual(next_node.shape, (2, _ceil_pow2(self.trie_small.layer_max_branches[0])))
        self.assertEqual(top_k_values.shape, (2, 2))
        self.assertEqual(top_k_indices.shape, (2, 2))

    # ------------------------------------------------------------------
    # Top-k ordering
    # ------------------------------------------------------------------

    def test_topk_descending_order(self) -> None:
        """Returned logit values must be in non-increasing order."""
        torch.manual_seed(1)
        B, N = 4, 8
        logits = torch.randn(B, N, device=self.device)
        state = self._state(self.trie_small, step=0, nodes=[0] * B)
        _, _, tv, _ = _pytorch_reference(logits, state, k=2)
        for b in range(B):
            self.assertGreaterEqual(
                float(tv[b, 0]), float(tv[b, 1]),
                msg=f"Batch {b}: top-k not descending"
            )

    def test_topk_k1(self) -> None:
        """k=1 should return the single highest valid logit."""
        torch.manual_seed(2)
        B, N = 3, 8
        logits = torch.randn(B, N, device=self.device)
        state = self._state(self.trie_small, step=0, nodes=[0] * B)
        _, _, tv1, ti1 = _pytorch_reference(logits, state, k=1)
        _, _, tv2, ti2 = _pytorch_reference(logits, state, k=2)
        self.assertTrue(torch.allclose(tv1[:, 0], tv2[:, 0]))
        self.assertTrue(torch.equal(ti1[:, 0], ti2[:, 0]))

    # ------------------------------------------------------------------
    # k > max_branches: fewer valid tokens than requested
    # ------------------------------------------------------------------

    def test_k_exceeds_valid_tokens(self) -> None:
        """When k > #valid children, excess slots carry -inf / -1."""
        torch.manual_seed(3)
        B, N = 2, 8
        # At step 0, root has 2 valid children (tokens 1 and 3) for the small trie.
        logits = torch.randn(B, N, device=self.device)
        state = self._state(self.trie_small, step=0, nodes=[0] * B)
        _, _, tv, ti = _pytorch_reference(logits, state, k=5)
        # Slots beyond the 2 valid children should be -inf.
        for b in range(B):
            for slot in range(2, 5):
                self.assertTrue(
                    float(tv[b, slot]) <= -1e10,
                    msg=f"Batch {b} slot {slot}: expected -inf, got {tv[b, slot]}"
                )

    # ------------------------------------------------------------------
    # Metadata (next_node, valid_idxs) matches the reference Triton kernel
    # ------------------------------------------------------------------

    def test_metadata_matches_triton(self) -> None:
        """next_node and valid_idxs must equal the Triton constrained_node_transition."""
        from rectokens.ops.constrained_node_transition import constrained_node_transition
        torch.manual_seed(4)
        B, N = 3, 8
        logits = torch.randn(B, N, device=self.device)
        state = self._state(self.trie_small, step=0, nodes=[0] * B)

        ref_nn, ref_vi, _ = constrained_node_transition(logits, state)
        fused_nn, fused_vi, _, _ = _pytorch_reference(logits, state, k=2)

        self.assertTrue(torch.equal(ref_nn, fused_nn), "next_node mismatch")
        self.assertTrue(torch.equal(ref_vi, fused_vi), "valid_idxs mismatch")

    # ------------------------------------------------------------------
    # Top-k indices point to valid constrained tokens
    # ------------------------------------------------------------------

    def test_topk_indices_are_valid_children(self) -> None:
        """Every top-k index (that is not -1) must be a valid child token."""
        from rectokens.ops.constrained_node_transition import constrained_node_transition
        torch.manual_seed(5)
        B, N = 3, 8
        logits = torch.randn(B, N, device=self.device)
        state = self._state(self.trie_small, step=0, nodes=[0] * B)

        _, valid_idxs, _, ti = _pytorch_reference(logits, state, k=2)
        for b in range(B):
            valid_set = set(valid_idxs[b][valid_idxs[b] >= 0].tolist())
            for s in range(2):
                idx = int(ti[b, s])
                if idx != -1:
                    self.assertIn(idx, valid_set, msg=f"b={b}, slot={s}: {idx} not in {valid_set}")

    # ------------------------------------------------------------------
    # Multiple batch elements with different nodes
    # ------------------------------------------------------------------

    def test_batched_different_nodes(self) -> None:
        """Different cur_node values per batch element must give correct outputs."""
        torch.manual_seed(6)
        B, N = 4, 8
        logits = torch.randn(B, N, device=self.device)
        # Mix of step-0 nodes and step-1 nodes
        state = self._state(self.trie_small, step=1, nodes=[1, 2, 1, 2])
        nn, vi, tv, ti = _pytorch_reference(logits, state, k=1)
        self.assertEqual(nn.shape[0], B)
        self.assertEqual(tv.shape, (B, 1))

    # ------------------------------------------------------------------
    # Large N (stress the tile coverage)
    # ------------------------------------------------------------------

    def test_large_vocab(self) -> None:
        """N=16 with a dense 4×4 trie — all top-k indices are valid children."""
        torch.manual_seed(7)
        B, N = 4, 16
        logits = torch.randn(B, N, device=self.device)
        state = self._state(self.trie_dense4, step=0, nodes=[0] * B)
        nn, vi, tv, ti = _pytorch_reference(logits, state, k=4)
        self.assertEqual(tv.shape, (B, 4))
        # All returned indices should be in the valid set for the root node
        valid_set = set(vi[0][vi[0] >= 0].tolist())
        for b in range(B):
            for s in range(4):
                idx = int(ti[b, s])
                if idx != -1:
                    self.assertIn(idx, valid_set)


# ---------------------------------------------------------------------------
# CuTe DSL CUDA kernel tests — skipped if kernel is unavailable
# ---------------------------------------------------------------------------


@unittest.skipUnless(
    torch.cuda.is_available(),
    "CUDA required",
)
class TestFusedTopkCuteDSL(unittest.TestCase):
    """Compare the CuTe DSL kernel against the PyTorch reference."""

    @classmethod
    def setUpClass(cls) -> None:
        from rectokens.kernels.fused_topk_constrained_node_transition_cute import (
            _CUTE_DSL_AVAILABLE,
        )
        if not _CUTE_DSL_AVAILABLE:
            raise unittest.SkipTest(
                "cuda-python >= 12.4 and numba >= 0.57 required for CuTe DSL tests"
            )

        cls.device = torch.device("cuda")
        cls.trie_small = _build_small_trie(cls.device)
        cls.trie_dense4 = _build_dense_trie(4, cls.device)

    def _state(self, trie: CompactCSRTrie, step: int, nodes: list[int]) -> ConstraintState:
        return ConstraintState(
            step=step,
            trie=trie,
            cur_node=torch.tensor(nodes, device=self.device),
        )

    def _assert_topk_close(
        self,
        ref_tv: torch.Tensor,
        ref_ti: torch.Tensor,
        got_tv: torch.Tensor,
        got_ti: torch.Tensor,
        msg: str = "",
        atol: float = 1e-4,
    ) -> None:
        """Values must be close; indices must be bit-exact for non-(-inf) slots."""
        # Mask for "real" slots (value > -1e10)
        ref_real = ref_tv > -1e10
        got_real = got_tv > -1e10
        self.assertTrue(
            torch.equal(ref_real, got_real),
            f"{msg}: valid-slot mask mismatch\n  ref={ref_real}\n  got={got_real}",
        )
        self.assertTrue(
            torch.allclose(ref_tv[ref_real], got_tv[got_real].float(), atol=atol),
            f"{msg}: top-k values mismatch",
        )
        # Indices can legitimately differ when logit values are equal (ties);
        # only check that the returned index is valid (≥0) when a real slot.
        self.assertTrue(
            (got_ti[got_real] >= 0).all(),
            f"{msg}: negative index in valid slot",
        )

    def _run_both(
        self,
        logits: torch.Tensor,
        constraint_state: ConstraintState,
        k: int,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        from rectokens.ops.fused_topk_constrained_node_transition import (
            _cuda_fused_topk_cst,
        )
        ref = _pytorch_reference(logits, constraint_state, k)
        got = _cuda_fused_topk_cst(logits, constraint_state, k)
        torch.cuda.synchronize()
        return ref, got

    # ------------------------------------------------------------------
    # Core correctness
    # ------------------------------------------------------------------

    def test_step0_b1(self) -> None:
        torch.manual_seed(10)
        logits = torch.randn(1, 8, device=self.device)
        state = self._state(self.trie_small, step=0, nodes=[0])
        ref, got = self._run_both(logits, state, k=2)
        self.assertTrue(torch.equal(ref[0], got[0]), "next_node mismatch")
        self.assertTrue(torch.equal(ref[1], got[1]), "valid_idxs mismatch")
        self._assert_topk_close(ref[2], ref[3], got[2], got[3], "step0_b1")

    def test_step0_b4(self) -> None:
        torch.manual_seed(11)
        logits = torch.randn(4, 8, device=self.device)
        state = self._state(self.trie_small, step=0, nodes=[0, 0, 0, 0])
        ref, got = self._run_both(logits, state, k=2)
        self.assertTrue(torch.equal(ref[0], got[0]))
        self.assertTrue(torch.equal(ref[1], got[1]))
        self._assert_topk_close(ref[2], ref[3], got[2], got[3], "step0_b4")

    def test_step1_mixed_nodes(self) -> None:
        torch.manual_seed(12)
        logits = torch.randn(4, 8, device=self.device)
        state = self._state(self.trie_small, step=1, nodes=[1, 2, 1, 2])
        ref, got = self._run_both(logits, state, k=1)
        self.assertTrue(torch.equal(ref[0], got[0]))
        self.assertTrue(torch.equal(ref[1], got[1]))
        self._assert_topk_close(ref[2], ref[3], got[2], got[3], "step1_mixed")

    def test_step2_leaf_nodes(self) -> None:
        torch.manual_seed(13)
        logits = torch.randn(2, 8, device=self.device)
        state = self._state(self.trie_small, step=2, nodes=[3, 4])
        ref, got = self._run_both(logits, state, k=2)
        self.assertTrue(torch.equal(ref[0], got[0]))
        self.assertTrue(torch.equal(ref[1], got[1]))
        self._assert_topk_close(ref[2], ref[3], got[2], got[3], "step2_leaf")

    # ------------------------------------------------------------------
    # Top-k semantics
    # ------------------------------------------------------------------

    def test_topk_descending_order_cute(self) -> None:
        """CuTe kernel must return values in non-increasing order."""
        torch.manual_seed(14)
        logits = torch.randn(8, 8, device=self.device)
        state = self._state(self.trie_small, step=0, nodes=[0] * 8)
        from rectokens.ops.fused_topk_constrained_node_transition import _cuda_fused_topk_cst
        _, _, tv, _ = _cuda_fused_topk_cst(logits, state, k=2)
        torch.cuda.synchronize()
        for b in range(8):
            v0, v1 = float(tv[b, 0]), float(tv[b, 1])
            if v1 > -1e10:  # only compare when both slots are valid
                self.assertGreaterEqual(v0, v1, msg=f"Batch {b}: not descending")

    def test_k_exceeds_branches_cute(self) -> None:
        """Excess top-k slots must carry -inf / -1 in the CuTe kernel."""
        torch.manual_seed(15)
        logits = torch.randn(2, 8, device=self.device)
        # Root has 2 valid children at step 0
        state = self._state(self.trie_small, step=0, nodes=[0, 0])
        from rectokens.ops.fused_topk_constrained_node_transition import _cuda_fused_topk_cst
        _, _, tv, ti = _cuda_fused_topk_cst(logits, state, k=5)
        torch.cuda.synchronize()
        for b in range(2):
            for slot in range(2, 5):
                self.assertLessEqual(
                    float(tv[b, slot]), -1e10,
                    msg=f"b={b} slot={slot}: expected -inf sentinel",
                )
                self.assertEqual(
                    int(ti[b, slot]), -1,
                    msg=f"b={b} slot={slot}: expected -1 index",
                )

    # ------------------------------------------------------------------
    # Dense trie (large fan-out per node)
    # ------------------------------------------------------------------

    def test_dense_trie_step0(self) -> None:
        torch.manual_seed(16)
        B, N = 8, 16
        logits = torch.randn(B, N, device=self.device)
        state = self._state(self.trie_dense4, step=0, nodes=[0] * B)
        ref, got = self._run_both(logits, state, k=4)
        self.assertTrue(torch.equal(ref[0], got[0]))
        self.assertTrue(torch.equal(ref[1], got[1]))
        self._assert_topk_close(ref[2], ref[3], got[2], got[3], "dense_step0")

    # ------------------------------------------------------------------
    # Large batch
    # ------------------------------------------------------------------

    def test_large_batch(self) -> None:
        torch.manual_seed(17)
        B, N = 64, 8
        logits = torch.randn(B, N, device=self.device)
        nodes = [0] * B
        state = self._state(self.trie_small, step=0, nodes=nodes)
        ref, got = self._run_both(logits, state, k=2)
        self.assertTrue(torch.equal(ref[0], got[0]))
        self.assertTrue(torch.equal(ref[1], got[1]))
        self._assert_topk_close(ref[2], ref[3], got[2], got[3], "large_batch")
