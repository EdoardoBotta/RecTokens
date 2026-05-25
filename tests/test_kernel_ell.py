"""Tests that the ELL-based fused kernels produce identical outputs to the CSR-based kernels."""
from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

if not torch.cuda.is_available():
    raise unittest.SkipTest("CUDA required")

from rectokens.schemas.compact_csr_trie import CompactCSRTrie
from rectokens.schemas.compact_ell_trie import CompactELLTrie
from rectokens.kernels.constrained_node_transition import (
    _fused_linear_constrained_node_transition_sampling_op as csr_sampling_op,
    _fused_linear_constrained_node_transition_topk_op as csr_topk_op,
)
from rectokens.kernels.constrained_node_transition_ell import (
    _ell_fused_linear_constrained_node_transition_sampling_op as ell_sampling_op,
    _ell_fused_linear_constrained_node_transition_topk_op as ell_topk_op,
)
from rectokens.decoding.vntk import (
    sparse_linear_pytorch,
    sparse_linear_compact_pytorch,
    sparse_linear_ell_pytorch,
    sparse_linear_compact_ell_pytorch,
)


DEVICE = torch.device("cuda")
VOCAB_SIZE = 8


def lex_sort(rows: list[list[int]]) -> torch.Tensor:
    return torch.tensor(sorted(rows), dtype=torch.long)


def to_device(csr: CompactCSRTrie) -> CompactCSRTrie:
    return csr._replace(
        row_ptrs=csr.row_ptrs.to(DEVICE),
        stacked_cols_vals=csr.stacked_cols_vals.to(DEVICE),
        dense_mask_by_layer=[v.to(DEVICE) for v in csr.dense_mask_by_layer],
        dense_states=csr.dense_states.to(DEVICE),
    )


def ell_to_device(ell: CompactELLTrie) -> CompactELLTrie:
    return ell._replace(
        ell_cols_vals=ell.ell_cols_vals.to(DEVICE),
        n_children=ell.n_children.to(DEVICE),
        dense_mask_by_layer=[v.to(DEVICE) for v in ell.dense_mask_by_layer],
        dense_states=ell.dense_states.to(DEVICE),
    )


def make_tries(seqs: list[list[int]], vocab_size: int) -> tuple[CompactCSRTrie, CompactELLTrie]:
    csr = to_device(CompactCSRTrie.from_sorted_batch(lex_sort(seqs), vocab_size=vocab_size))
    ell = ell_to_device(CompactELLTrie.from_csr(csr))
    return csr, ell


class TestELLvCSRSampling(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        seqs_small = [[1, 2, 1], [3, 1, 2], [3, 1, 3]]
        cls.csr_small, cls.ell_small = make_tries(seqs_small, VOCAB_SIZE)

        seqs_dense = [[i, j, k] for i in range(4) for j in range(4) for k in range(4)]
        cls.csr_dense, cls.ell_dense = make_tries(seqs_dense, vocab_size=16)

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _run_both(
        self,
        B: int,
        K: int,
        step: int,
        cur_node_vals: list[int],
        seed: int,
        csr: CompactCSRTrie,
        ell: CompactELLTrie,
    ):
        torch.manual_seed(0)
        a = torch.randn(B, K, device=DEVICE)
        b = torch.randn(K, csr.vocab_size, device=DEVICE)
        cur_node = torch.tensor(cur_node_vals, device=DEVICE)
        bias_val = a.new_empty(0)
        max_branches = csr.layer_max_branches[step]

        csr_out = csr_sampling_op(
            a, b, bias_val, cur_node,
            csr.row_ptrs, csr.stacked_cols_vals,
            max_branches, False, rng_seed=seed,
        )
        ell_out = ell_sampling_op(
            a, b, bias_val, cur_node,
            ell.ell_cols_vals, ell.n_children,
            max_branches, False, rng_seed=seed,
        )
        return csr_out, ell_out

    def _assert_sampling_match(
        self, B, K, step, cur_node_vals, seed, csr=None, ell=None
    ):
        if csr is None:
            csr, ell = self.csr_small, self.ell_small
        (csr_nn, csr_vi, csr_s), (ell_nn, ell_vi, ell_s) = self._run_both(
            B, K, step, cur_node_vals, seed, csr, ell
        )
        self.assertTrue(torch.equal(ell_nn, csr_nn), "next_node mismatch")
        self.assertTrue(torch.equal(ell_vi, csr_vi), "valid_idxs mismatch")
        self.assertTrue(torch.equal(ell_s, csr_s), "sample mismatch")

    # -------------------------------------------------------------------------
    # next_node / valid_idxs / sample agree with CSR
    # -------------------------------------------------------------------------

    def test_b1_step0(self) -> None:
        self._assert_sampling_match(1, 16, 0, [0], seed=42)

    def test_b2_step0_same_node(self) -> None:
        self._assert_sampling_match(2, 16, 0, [0, 0], seed=7)

    def test_b2_step1_diff_nodes(self) -> None:
        self._assert_sampling_match(2, 16, 1, [1, 2], seed=99)

    def test_b3_step2(self) -> None:
        self._assert_sampling_match(3, 16, 2, [3, 4, 3], seed=123)

    def test_b8_large_k(self) -> None:
        self._assert_sampling_match(8, 128, 0, [0] * 8, seed=1)

    def test_b32_step0(self) -> None:
        self._assert_sampling_match(32, 64, 0, [0] * 32, seed=5)

    def test_dense_trie_step0(self) -> None:
        self._assert_sampling_match(
            8, 16, 0, [0] * 8, seed=42, csr=self.csr_dense, ell=self.ell_dense
        )

    def test_dense_trie_step1(self) -> None:
        self._assert_sampling_match(
            4, 16, 1, [1, 2, 3, 4], seed=7, csr=self.csr_dense, ell=self.ell_dense
        )

    # -------------------------------------------------------------------------
    # Determinism: same seed → same result (ELL-internal consistency)
    # -------------------------------------------------------------------------

    def test_deterministic_b1(self) -> None:
        _, (_, _, s1) = self._run_both(1, 16, 0, [0], 42, self.csr_small, self.ell_small)
        _, (_, _, s2) = self._run_both(1, 16, 0, [0], 42, self.csr_small, self.ell_small)
        self.assertTrue(torch.equal(s1, s2))

    def test_deterministic_b4(self) -> None:
        (_, _, s1), _ = self._run_both(4, 16, 0, [0, 0, 0, 0], 77, self.csr_small, self.ell_small)
        (_, _, s2), _ = self._run_both(4, 16, 0, [0, 0, 0, 0], 77, self.csr_small, self.ell_small)
        self.assertTrue(torch.equal(s1, s2))

    # -------------------------------------------------------------------------
    # Sample must be a valid child of the current node
    # -------------------------------------------------------------------------

    def test_sample_is_valid_child_step0(self) -> None:
        _, (_, vi, sample) = self._run_both(
            1, 16, 0, [0], 123, self.csr_small, self.ell_small
        )
        valid = vi[0][vi[0] >= 0].tolist()
        self.assertIn(int(sample[0].item()), valid)

    def test_sample_is_valid_child_step1(self) -> None:
        _, (_, vi, sample) = self._run_both(
            2, 16, 1, [1, 2], 456, self.csr_small, self.ell_small
        )
        for b in range(2):
            valid = vi[b][vi[b] >= 0].tolist()
            self.assertIn(int(sample[b].item()), valid)

    def test_sample_is_valid_child_step2(self) -> None:
        _, (_, vi, sample) = self._run_both(
            3, 16, 2, [3, 4, 3], 789, self.csr_small, self.ell_small
        )
        for b in range(3):
            valid = vi[b][vi[b] >= 0].tolist()
            self.assertIn(int(sample[b].item()), valid)

    # -------------------------------------------------------------------------
    # Bias path
    # -------------------------------------------------------------------------

    def test_with_bias(self) -> None:
        B, K, step = 2, 16, 0
        cur_node_vals = [0, 0]
        seed = 42
        torch.manual_seed(0)
        a = torch.randn(B, K, device=DEVICE)
        b = torch.randn(K, VOCAB_SIZE, device=DEVICE)
        bias = torch.randn(VOCAB_SIZE, device=DEVICE)
        cur_node = torch.tensor(cur_node_vals, device=DEVICE)
        max_branches = self.csr_small.layer_max_branches[step]

        csr_nn, csr_vi, csr_s = csr_sampling_op(
            a, b, bias, cur_node,
            self.csr_small.row_ptrs, self.csr_small.stacked_cols_vals,
            max_branches, True, rng_seed=seed,
        )
        ell_nn, ell_vi, ell_s = ell_sampling_op(
            a, b, bias, cur_node,
            self.ell_small.ell_cols_vals, self.ell_small.n_children,
            max_branches, True, rng_seed=seed,
        )
        self.assertTrue(torch.equal(ell_nn, csr_nn))
        self.assertTrue(torch.equal(ell_vi, csr_vi))
        self.assertTrue(torch.equal(ell_s, csr_s))


class TestELLvCSRTopK(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        seqs_small = [[1, 2, 1], [3, 1, 2], [3, 1, 3]]
        cls.csr_small, cls.ell_small = make_tries(seqs_small, VOCAB_SIZE)

        seqs_dense = [[i, j, k] for i in range(4) for j in range(4) for k in range(4)]
        cls.csr_dense, cls.ell_dense = make_tries(seqs_dense, vocab_size=16)

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _run_both(
        self,
        B: int,
        K: int,
        step: int,
        cur_node_vals: list[int],
        k: int,
        csr: CompactCSRTrie,
        ell: CompactELLTrie,
    ):
        torch.manual_seed(42)
        a = torch.randn(B, K, device=DEVICE)
        b = torch.randn(K, csr.vocab_size, device=DEVICE)
        cur_node = torch.tensor(cur_node_vals, device=DEVICE)
        bias_val = a.new_empty(0)
        max_branches = csr.layer_max_branches[step]

        csr_out = csr_topk_op(
            a, b, bias_val, cur_node,
            csr.row_ptrs, csr.stacked_cols_vals,
            max_branches, False, k,
        )
        ell_out = ell_topk_op(
            a, b, bias_val, cur_node,
            ell.ell_cols_vals, ell.n_children,
            max_branches, False, k,
        )
        return csr_out, ell_out

    def _assert_topk_match(self, B, K, step, cur_node_vals, k, csr=None, ell=None):
        if csr is None:
            csr, ell = self.csr_small, self.ell_small
        (csr_nn, csr_vi, csr_tl, csr_ti), (ell_nn, ell_vi, ell_tl, ell_ti) = self._run_both(
            B, K, step, cur_node_vals, k, csr, ell
        )

        self.assertTrue(torch.equal(ell_nn, csr_nn), "next_node mismatch")
        self.assertTrue(torch.equal(ell_vi, csr_vi), "valid_idxs mismatch")

        # Sort along k-dim to handle tie-breaking, then compare.
        self.assertTrue(
            torch.allclose(
                ell_tl.float().sort(dim=-1).values,
                csr_tl.float().sort(dim=-1).values,
                atol=1e-3,
                equal_nan=True,
            ),
            f"topk logits mismatch\nELL: {ell_tl}\nCSR: {csr_tl}",
        )
        self.assertTrue(
            torch.equal(ell_ti.sort(dim=-1).values, csr_ti.sort(dim=-1).values),
            f"topk idxs mismatch\nELL: {ell_ti}\nCSR: {csr_ti}",
        )

    # -------------------------------------------------------------------------
    # next_node / valid_idxs / topk_logits / topk_idxs agree with CSR
    # -------------------------------------------------------------------------

    def test_k1_b1_step0(self) -> None:
        self._assert_topk_match(1, 16, 0, [0], k=1)

    def test_k1_b2_step1(self) -> None:
        self._assert_topk_match(2, 16, 1, [1, 2], k=1)

    def test_k2_b1_step0(self) -> None:
        self._assert_topk_match(1, 16, 0, [0], k=2)

    def test_k2_b2_step1(self) -> None:
        self._assert_topk_match(2, 16, 1, [1, 2], k=2)

    def test_k1_b3_step2(self) -> None:
        self._assert_topk_match(3, 16, 2, [3, 4, 3], k=1)

    def test_k2_b3_step2(self) -> None:
        self._assert_topk_match(3, 16, 2, [3, 4, 3], k=2)

    def test_k2_b8_large_k(self) -> None:
        self._assert_topk_match(8, 128, 0, [0] * 8, k=2)

    def test_k_exceeds_branches(self) -> None:
        # When k >= max_branches the kernel returns all branches; both CSR and ELL do the same.
        self._assert_topk_match(2, 16, 1, [1, 2], k=10)

    def test_dense_trie_step0(self) -> None:
        self._assert_topk_match(
            8, 16, 0, [0] * 8, k=2, csr=self.csr_dense, ell=self.ell_dense
        )

    def test_dense_trie_step1_k3(self) -> None:
        self._assert_topk_match(
            4, 16, 1, [1, 2, 3, 4], k=3, csr=self.csr_dense, ell=self.ell_dense
        )

    def test_b32_step0(self) -> None:
        self._assert_topk_match(32, 64, 0, [0] * 32, k=2)

    # -------------------------------------------------------------------------
    # Bias path
    # -------------------------------------------------------------------------

    def test_with_bias(self) -> None:
        B, K, step, k = 2, 16, 0, 1
        cur_node_vals = [0, 0]
        torch.manual_seed(0)
        a = torch.randn(B, K, device=DEVICE)
        b = torch.randn(K, VOCAB_SIZE, device=DEVICE)
        bias = torch.randn(VOCAB_SIZE, device=DEVICE)
        cur_node = torch.tensor(cur_node_vals, device=DEVICE)
        max_branches = self.csr_small.layer_max_branches[step]

        csr_nn, csr_vi, csr_tl, csr_ti = csr_topk_op(
            a, b, bias, cur_node,
            self.csr_small.row_ptrs, self.csr_small.stacked_cols_vals,
            max_branches, True, k,
        )
        ell_nn, ell_vi, ell_tl, ell_ti = ell_topk_op(
            a, b, bias, cur_node,
            self.ell_small.ell_cols_vals, self.ell_small.n_children,
            max_branches, True, k,
        )
        self.assertTrue(torch.equal(ell_nn, csr_nn))
        self.assertTrue(torch.equal(ell_vi, csr_vi))
        self.assertTrue(
            torch.allclose(ell_tl.float(), csr_tl.float(), atol=1e-3, equal_nan=True)
        )
        self.assertTrue(torch.equal(ell_ti, csr_ti))


class TestELLPytorchvCSRPytorch(unittest.TestCase):
    """ELL sparse_linear_pytorch outputs match CSR sparse_linear_pytorch outputs."""

    @classmethod
    def setUpClass(cls) -> None:
        seqs_small = [[1, 2, 1], [3, 1, 2], [3, 1, 3]]
        cls.csr_small, cls.ell_small = make_tries(seqs_small, VOCAB_SIZE)

        seqs_dense = [[i, j, k] for i in range(4) for j in range(4) for k in range(4)]
        cls.csr_dense, cls.ell_dense = make_tries(seqs_dense, vocab_size=16)

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _run_both(self, B, K, step, cur_node_vals, csr, ell):
        torch.manual_seed(0)
        a = torch.randn(B, K, device=DEVICE)
        weight = torch.randn(csr.vocab_size, K, device=DEVICE)
        cur_node = torch.tensor(cur_node_vals, device=DEVICE)

        csr_out = sparse_linear_pytorch(a, weight, cur_node, csr, step)
        ell_out = sparse_linear_ell_pytorch(a, weight, cur_node, ell, step)
        return csr_out, ell_out

    def _run_both_compact(self, B, K, step, cur_node_vals, csr, ell):
        torch.manual_seed(0)
        a = torch.randn(B, K, device=DEVICE)
        weight = torch.randn(csr.vocab_size, K, device=DEVICE)
        cur_node = torch.tensor(cur_node_vals, device=DEVICE)

        csr_out = sparse_linear_compact_pytorch(a, weight, cur_node, csr, step)
        ell_out = sparse_linear_compact_ell_pytorch(a, weight, cur_node, ell, step)
        return csr_out, ell_out

    def _assert_full_match(self, B, K, step, cur_node_vals, csr=None, ell=None):
        if csr is None:
            csr, ell = self.csr_small, self.ell_small
        (csr_nn, csr_vi, csr_cl), (ell_nn, ell_vi, ell_cl) = self._run_both(
            B, K, step, cur_node_vals, csr, ell
        )
        self.assertTrue(torch.equal(ell_nn, csr_nn), "next_node mismatch")
        self.assertTrue(torch.equal(ell_vi, csr_vi), "valid_idxs mismatch")
        self.assertTrue(
            torch.allclose(ell_cl.float(), csr_cl.float(), equal_nan=True),
            "corrected_logits mismatch",
        )

    def _assert_compact_match(self, B, K, step, cur_node_vals, csr=None, ell=None):
        if csr is None:
            csr, ell = self.csr_small, self.ell_small
        (csr_nn, csr_vi, csr_bl), (ell_nn, ell_vi, ell_bl) = self._run_both_compact(
            B, K, step, cur_node_vals, csr, ell
        )
        self.assertTrue(torch.equal(ell_nn, csr_nn), "next_node mismatch")
        self.assertTrue(torch.equal(ell_vi, csr_vi), "valid_idxs mismatch")
        self.assertTrue(
            torch.allclose(ell_bl.float(), csr_bl.float(), equal_nan=True),
            "branch_logits mismatch",
        )

    # -------------------------------------------------------------------------
    # sparse_linear_ell_pytorch matches sparse_linear_pytorch (full scatter)
    # -------------------------------------------------------------------------

    def test_full_b1_step0(self) -> None:
        self._assert_full_match(1, 16, 0, [0])

    def test_full_b2_step0_same_node(self) -> None:
        self._assert_full_match(2, 16, 0, [0, 0])

    def test_full_b2_step1_diff_nodes(self) -> None:
        self._assert_full_match(2, 16, 1, [1, 2])

    def test_full_b3_step2(self) -> None:
        self._assert_full_match(3, 16, 2, [3, 4, 3])

    def test_full_b8_large_k(self) -> None:
        self._assert_full_match(8, 128, 0, [0] * 8)

    def test_full_dense_trie_step0(self) -> None:
        self._assert_full_match(
            8, 16, 0, [0] * 8, csr=self.csr_dense, ell=self.ell_dense
        )

    def test_full_dense_trie_step1(self) -> None:
        self._assert_full_match(
            4, 16, 1, [1, 2, 3, 4], csr=self.csr_dense, ell=self.ell_dense
        )

    # -------------------------------------------------------------------------
    # sparse_linear_compact_ell_pytorch matches sparse_linear_compact_pytorch
    # -------------------------------------------------------------------------

    def test_compact_b1_step0(self) -> None:
        self._assert_compact_match(1, 16, 0, [0])

    def test_compact_b2_step1_diff_nodes(self) -> None:
        self._assert_compact_match(2, 16, 1, [1, 2])

    def test_compact_b3_step2(self) -> None:
        self._assert_compact_match(3, 16, 2, [3, 4, 3])

    def test_compact_b8_large_k(self) -> None:
        self._assert_compact_match(8, 128, 0, [0] * 8)

    def test_compact_dense_trie_step1(self) -> None:
        self._assert_compact_match(
            4, 16, 1, [1, 2, 3, 4], csr=self.csr_dense, ell=self.ell_dense
        )

    # -------------------------------------------------------------------------
    # sample built on ELL full output matches sample built on CSR full output
    # -------------------------------------------------------------------------

    def test_sample_valid_child_step0(self) -> None:
        torch.manual_seed(7)
        a = torch.randn(2, 16, device=DEVICE)
        weight = torch.randn(VOCAB_SIZE, 16, device=DEVICE)
        cur_node = torch.tensor([0, 0], device=DEVICE)
        _, vi, corrected_logits = sparse_linear_ell_pytorch(
            a, weight, cur_node, self.ell_small, step=0
        )
        probs = F.softmax(corrected_logits, dim=-1)
        sample = torch.multinomial(probs, num_samples=1).squeeze(-1)
        for b in range(2):
            valid = vi[b][vi[b] >= 0].tolist()
            self.assertIn(int(sample[b].item()), valid)

    # -------------------------------------------------------------------------
    # top-k built on ELL compact output matches top-k built on CSR compact output
    # -------------------------------------------------------------------------

    def test_topk_compact_b2_k1_step1(self) -> None:
        torch.manual_seed(42)
        B, K, step, k = 2, 16, 1, 1
        a = torch.randn(B, K, device=DEVICE)
        weight = torch.randn(VOCAB_SIZE, K, device=DEVICE)
        cur_node = torch.tensor([1, 2], device=DEVICE)

        _, csr_vi, csr_bl = sparse_linear_compact_pytorch(a, weight, cur_node, self.csr_small, step)
        csr_topk_l, csr_topk_bi = torch.topk(csr_bl, k, dim=-1)
        csr_topk_i = csr_vi.gather(1, csr_topk_bi)

        _, ell_vi, ell_bl = sparse_linear_compact_ell_pytorch(a, weight, cur_node, self.ell_small, step)
        ell_topk_l, ell_topk_bi = torch.topk(ell_bl, k, dim=-1)
        ell_topk_i = ell_vi.gather(1, ell_topk_bi)

        self.assertTrue(torch.allclose(ell_topk_l, csr_topk_l, equal_nan=True))
        self.assertTrue(torch.equal(ell_topk_i, csr_topk_i))
