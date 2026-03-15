from __future__ import annotations

import unittest

import torch

if not torch.cuda.is_available():
    raise unittest.SkipTest("CUDA required")

from rectokens.schemas.compact_csr_trie import CompactCSRTrie
from rectokens.decoding.vntk import vtnk_pytorch
from rectokens.ops.constrained_node_transition import (
    constrained_node_transition,
    fused_linear_constrained_node_transition,
)
from rectokens.schemas.state import ConstraintState


DEVICE = torch.device("cuda")
VOCAB_SIZE = 8


def lex_sort(rows: list[list[int]]) -> torch.Tensor:
    return torch.tensor(sorted(rows), dtype=torch.long)


class TestKernel(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        seqs_small = [[1, 2, 1], [3, 1, 2], [3, 1, 3]]
        csr = CompactCSRTrie.from_sorted_batch(
            lex_sort(seqs_small), vocab_size=VOCAB_SIZE
        )
        cls.csr_small = csr._replace(
            row_ptrs=csr.row_ptrs.to(DEVICE),
            stacked_cols_vals=csr.stacked_cols_vals.to(DEVICE),
            dense_mask_by_layer=[v.to(DEVICE) for v in csr.dense_mask_by_layer],
            dense_states=csr.dense_states.to(DEVICE),
        )

        seqs_dense = [[i, j, k] for i in range(4) for j in range(4) for k in range(4)]
        csr2 = CompactCSRTrie.from_sorted_batch(lex_sort(seqs_dense), vocab_size=16)
        cls.csr_dense = csr2._replace(
            row_ptrs=csr2.row_ptrs.to(DEVICE),
            stacked_cols_vals=csr2.stacked_cols_vals.to(DEVICE),
            dense_mask_by_layer=[v.to(DEVICE) for v in csr2.dense_mask_by_layer],
            dense_states=csr2.dense_states.to(DEVICE),
        )

    # ---------------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------------

    def _assert_constrained_node_transition(
        self, B: int, step: int, cur_node_vals: list[int] | None
    ) -> None:
        torch.manual_seed(0)
        logits = torch.randn(B, VOCAB_SIZE, device=DEVICE)
        cur_node = (
            torch.zeros(B, dtype=torch.long, device=DEVICE)
            if cur_node_vals is None
            else torch.tensor(cur_node_vals, device=DEVICE)
        )
        ref_nn, ref_vi, ref_cl = vtnk_pytorch(logits, cur_node, self.csr_small, step)
        constraint_state = ConstraintState(
            step=step, trie=self.csr_small, cur_node=cur_node
        )
        ker_nn, ker_vi, ker_cl = constrained_node_transition(logits, constraint_state)
        assert ker_nn.shape == ref_nn.shape
        assert ker_vi.shape == ref_vi.shape
        assert ker_cl.shape == ref_cl.shape
        assert torch.equal(ker_nn, ref_nn)
        assert torch.equal(ker_vi, ref_vi)
        assert torch.allclose(ker_cl, ref_cl, equal_nan=True)

    def _assert_valid_idxs(
        self, cur_node_vals: list[int], step: int, expected_per_batch: list[list[int]]
    ) -> None:
        cur_node = torch.tensor(cur_node_vals, device=DEVICE)
        dummy_logits = torch.zeros(len(cur_node_vals), VOCAB_SIZE, device=DEVICE)
        constraint_state = ConstraintState(
            step=step, trie=self.csr_small, cur_node=cur_node
        )
        _, vi, _ = constrained_node_transition(dummy_logits, constraint_state)
        for b, expected in enumerate(expected_per_batch):
            actual = sorted(vi[b][vi[b] >= 0].tolist())
            assert actual == sorted(expected)
            assert (vi[b][vi[b] < 0] == -1).all()
            assert vi[b].shape[0] == self.csr_small.layer_max_branches[step]

    def _assert_fused_linear(
        self, B: int, K: int, step: int, cur_node_vals: list[int]
    ) -> None:
        torch.manual_seed(42)
        a = torch.randn(B, K, device=DEVICE)
        b = torch.randn(K, VOCAB_SIZE, device=DEVICE)
        cur_node = torch.tensor(cur_node_vals, device=DEVICE)
        ref_logits = (a @ b).float()
        ref_nn, ref_vi, ref_cl = vtnk_pytorch(
            ref_logits, cur_node, self.csr_small, step
        )
        constraint_state = ConstraintState(
            step=step, trie=self.csr_small, cur_node=cur_node
        )
        ker_nn, ker_vi, ker_cl = fused_linear_constrained_node_transition(
            a, b, constraint_state
        )
        assert ker_nn.shape == ref_nn.shape
        assert ker_vi.shape == ref_vi.shape
        assert ker_cl.shape == ref_cl.shape
        assert torch.equal(ker_nn, ref_nn)
        assert torch.equal(ker_vi, ref_vi)
        # atol for tf32 accumulation differences
        assert torch.allclose(ker_cl, ref_cl, atol=1e-2, equal_nan=True)

    # ---------------------------------------------------------------------------
    # constrained_node_transition — matches vtnk_pytorch reference
    # ---------------------------------------------------------------------------

    def test_constrained_node_transition_b1_step0(self) -> None:
        self._assert_constrained_node_transition(1, 0, [0])

    def test_constrained_node_transition_b2_step0(self) -> None:
        self._assert_constrained_node_transition(2, 0, [0, 0])

    def test_constrained_node_transition_b2_step1(self) -> None:
        self._assert_constrained_node_transition(2, 1, [1, 2])

    def test_constrained_node_transition_b3_step2(self) -> None:
        self._assert_constrained_node_transition(3, 2, [3, 4, 3])

    def test_constrained_node_transition_b4_step1(self) -> None:
        self._assert_constrained_node_transition(4, 1, [1, 2, 1, 2])

    def test_constrained_node_transition_b256_step0(self) -> None:
        self._assert_constrained_node_transition(256, 0, None)

    def test_constrained_node_transition_dense_trie_step0(self) -> None:
        B, vocab_size = 32, 16
        torch.manual_seed(0)
        logits = torch.randn(B, vocab_size, device=DEVICE)
        cur_node = torch.zeros(B, dtype=torch.long, device=DEVICE)
        ref_nn, ref_vi, ref_cl = vtnk_pytorch(logits, cur_node, self.csr_dense, step=0)
        constraint_state = ConstraintState(
            step=0, trie=self.csr_dense, cur_node=cur_node
        )
        ker_nn, ker_vi, ker_cl = constrained_node_transition(logits, constraint_state)
        assert torch.equal(ker_nn, ref_nn)
        assert torch.equal(ker_vi, ref_vi)
        assert torch.allclose(ker_cl, ref_cl, equal_nan=True)

    def test_constrained_node_transition_dense_trie_step1(self) -> None:
        B, vocab_size = 32, 16
        torch.manual_seed(0)
        ref_nn0, _, _ = vtnk_pytorch(
            torch.randn(B, vocab_size, device=DEVICE),
            torch.zeros(B, dtype=torch.long, device=DEVICE),
            self.csr_dense,
            step=0,
        )
        next_nodes = ref_nn0[:, 0]
        logits = torch.randn(B, vocab_size, device=DEVICE)
        ref_nn, ref_vi, ref_cl = vtnk_pytorch(
            logits, next_nodes, self.csr_dense, step=1
        )
        constraint_state = ConstraintState(
            step=1, trie=self.csr_dense, cur_node=next_nodes
        )
        ker_nn, ker_vi, ker_cl = constrained_node_transition(logits, constraint_state)
        assert torch.equal(ker_nn, ref_nn)
        assert torch.equal(ker_vi, ref_vi)
        assert torch.allclose(ker_cl, ref_cl, equal_nan=True)

    # ---------------------------------------------------------------------------
    # valid_idxs semantic correctness
    # ---------------------------------------------------------------------------

    def test_valid_idxs_root(self) -> None:
        self._assert_valid_idxs([0], 0, [[1, 3]])

    def test_valid_idxs_step1_nodes(self) -> None:
        self._assert_valid_idxs([1, 2], 1, [[2], [1]])

    def test_valid_idxs_node4(self) -> None:
        self._assert_valid_idxs([4], 2, [[2, 3]])

    def test_valid_idxs_nodes3_4(self) -> None:
        self._assert_valid_idxs([3, 4], 2, [[1], [2, 3]])

    # ---------------------------------------------------------------------------
    # fused_linear_constrained_node_transition — matches unfused reference
    # ---------------------------------------------------------------------------

    def test_fused_linear_b1_step0(self) -> None:
        self._assert_fused_linear(1, 16, 0, [0])

    def test_fused_linear_b2_step1(self) -> None:
        self._assert_fused_linear(2, 16, 1, [1, 2])

    def test_fused_linear_b3_step2(self) -> None:
        self._assert_fused_linear(3, 16, 2, [3, 4, 3])

    def test_fused_linear_b8_large_k(self) -> None:
        self._assert_fused_linear(8, 128, 0, [0] * 8)
