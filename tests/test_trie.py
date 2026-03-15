from __future__ import annotations

import unittest

import torch

from rectokens.schemas.compact_csr_trie import CompactCSRTrie
from rectokens.decoding.trie import Trie
from rectokens.decoding.vntk import vtnk_pytorch


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def make_semantic_ids(
    num_items: int, num_levels: int, codebook_size: int, seed: int = 0
) -> torch.Tensor:
    rng = torch.Generator()
    rng.manual_seed(seed)
    return torch.randint(0, codebook_size, (num_items, num_levels), generator=rng)


def build_trie(semantic_ids: torch.Tensor) -> Trie:
    trie = Trie()
    for row in semantic_ids.tolist():
        trie.insert(row)
    return trie


def children_of(csr: CompactCSRTrie, bfs_row: int) -> dict[int, int]:
    row_ptrs = csr.row_ptrs.tolist()
    if bfs_row >= len(row_ptrs):
        return {}
    cols = csr.stacked_cols_vals[0].tolist()
    vals = csr.stacked_cols_vals[1].tolist()
    start = row_ptrs[bfs_row]
    end = row_ptrs[bfs_row + 1] if bfs_row + 1 < len(row_ptrs) else len(cols) - 1
    if start == end:
        return {}
    return {cols[j]: vals[j] for j in range(start, end)}


def assert_same_edges(t: CompactCSRTrie, b: CompactCSRTrie, label: str) -> None:
    cols_t = t.stacked_cols_vals[0].tolist()
    cols_b = b.stacked_cols_vals[0].tolist()
    vals_t = t.stacked_cols_vals[1].tolist()
    vals_b = b.stacked_cols_vals[1].tolist()
    rows_t = t.row_ptrs.tolist()
    rows_b = b.row_ptrs.tolist()
    assert cols_t == cols_b, (
        f"{label} cols mismatch\n  trie : {cols_t}\n  batch: {cols_b}"
    )
    assert vals_t == vals_b, (
        f"{label} vals mismatch\n  trie : {vals_t}\n  batch: {vals_b}"
    )
    assert rows_t == rows_b, (
        f"{label} row_ptrs mismatch\n  trie : {rows_t}\n  batch: {rows_b}"
    )
    assert (t.dense_states == b.dense_states).all(), f"{label} dense_states mismatch"


def lex_sort(rows: list[list[int]]) -> torch.Tensor:
    return torch.tensor(sorted(rows), dtype=torch.long)


class TestTrie(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.semantic_ids = make_semantic_ids(12, 3, 8, seed=0)
        cls.semantic_trie = build_trie(cls.semantic_ids)
        cls.csr_three_seqs = CompactCSRTrie.from_sorted_batch(
            lex_sort([[1, 2, 1], [3, 1, 2], [3, 1, 3]]), vocab_size=8
        )

    # ---------------------------------------------------------------------------
    # Trie
    # ---------------------------------------------------------------------------

    def test_trie_full_tuple_lookup(self) -> None:
        for row in self.semantic_ids.tolist():
            node = self.semantic_trie.find_prefix(row)
            assert node is not None and node.is_end_of_token

    def test_trie_valid_next_tokens(self) -> None:
        first = self.semantic_ids[0].tolist()
        for depth in range(3):
            node = self.semantic_trie.find_prefix(first[:depth])
            assert node is not None

    def test_trie_oov_prefix_returns_none(self) -> None:
        assert self.semantic_trie.find_prefix([8 + 99]) is None

    def test_trie_duplicate_insertion_idempotent(self) -> None:
        trie = build_trie(self.semantic_ids)
        build_trie(self.semantic_ids)  # insert again
        for row in self.semantic_ids.tolist():
            node = trie.find_prefix(row)
            assert node is not None and node.is_end_of_token

    def test_trie_shared_prefix_branches(self) -> None:
        trie = Trie()
        trie.insert([1, 2, 3])
        trie.insert([1, 2, 7])
        node = trie.find_prefix([1, 2])
        assert node is not None
        assert sorted(node.children.keys()) == [3, 7]

    # ---------------------------------------------------------------------------
    # CSR from trie
    # ---------------------------------------------------------------------------

    def test_csr_single_token(self) -> None:
        t = Trie()
        t.insert([42])
        csr = CompactCSRTrie.from_trie(t, vocab_size=50, dense_lookup_layers=1)
        assert csr.row_ptrs.tolist() == [0, 1]
        assert csr.stacked_cols_vals[0].tolist()[0] == 42
        assert csr.stacked_cols_vals[1].tolist()[0] == 1
        assert children_of(csr, 0) == {42: 1}
        assert children_of(csr, 1) == {}

    def test_csr_linear_chain(self) -> None:
        t = Trie()
        t.insert([1, 2, 3])
        csr = CompactCSRTrie.from_trie(t, vocab_size=8)
        assert csr.row_ptrs.tolist() == [0, 1, 2, 3]
        assert children_of(csr, 0) == {1: 1}
        assert children_of(csr, 1) == {2: 2}
        assert children_of(csr, 2) == {3: 3}
        assert children_of(csr, 3) == {}

    def test_csr_branching(self) -> None:
        t = Trie()
        t.insert([1, 2, 3])
        t.insert([1, 2, 7])
        csr = CompactCSRTrie.from_trie(t, vocab_size=8)
        assert csr.row_ptrs.tolist() == [0, 1, 2, 4, 4]
        assert children_of(csr, 0) == {1: 1}
        assert children_of(csr, 1) == {2: 2}
        assert children_of(csr, 2) == {3: 3, 7: 4}
        assert children_of(csr, 3) == {}
        assert children_of(csr, 4) == {}

    def test_csr_walk(self) -> None:
        t = Trie()
        t.insert([1, 2, 3])
        t.insert([1, 2, 7])
        csr = CompactCSRTrie.from_trie(t, vocab_size=8)
        for target in ([1, 2, 3], [1, 2, 7]):
            bfs_row = 0
            for token in target:
                ch = children_of(csr, bfs_row)
                assert token in ch
                bfs_row = ch[token]

    def test_csr_random_batch(self) -> None:
        trie = build_trie(self.semantic_ids)
        csr = CompactCSRTrie.from_trie(trie, vocab_size=8)
        for row in self.semantic_ids.tolist():
            bfs_row = 0
            for token in row:
                ch = children_of(csr, bfs_row)
                assert token in ch
                bfs_row = ch[token]

    def test_csr_multi_branch(self) -> None:
        t = Trie()
        t.insert([1, 2, 1])
        t.insert([3, 1, 2])
        t.insert([3, 1, 3])
        csr = CompactCSRTrie.from_trie(t, vocab_size=8)
        assert csr.row_ptrs.tolist() == [0, 2, 3, 4, 5, 7, 7, 7]
        assert csr.stacked_cols_vals[0].tolist() == [1, 3, 2, 1, 1, 2, 3, -1]
        assert csr.stacked_cols_vals[1].tolist() == [1, 2, 3, 4, 5, 6, 7, -1]
        assert children_of(csr, 0) == {1: 1, 3: 2}
        assert children_of(csr, 1) == {2: 3}
        assert children_of(csr, 2) == {1: 4}
        assert children_of(csr, 3) == {1: 5}
        assert children_of(csr, 4) == {2: 6, 3: 7}
        assert children_of(csr, 5) == {}
        assert children_of(csr, 6) == {}
        assert children_of(csr, 7) == {}
        for target in ([1, 2, 1], [3, 1, 2], [3, 1, 3]):
            bfs_row = 0
            for token in target:
                ch = children_of(csr, bfs_row)
                assert token in ch
                bfs_row = ch[token]

    # ---------------------------------------------------------------------------
    # CSR from sorted batch
    # ---------------------------------------------------------------------------

    def test_csr_sorted_batch_single_token(self) -> None:
        t = Trie()
        t.insert([42])
        b = CompactCSRTrie.from_sorted_batch(
            lex_sort([[42]]), vocab_size=50, dense_lookup_layers=1
        )
        assert b.dense_mask_by_layer[0].shape == (50,) and b.dense_mask_by_layer[0][42]
        assert_same_edges(
            CompactCSRTrie.from_trie(t, vocab_size=50, dense_lookup_layers=1), b, "[42]"
        )
        assert children_of(b, 0) == {42: 1}
        assert children_of(b, 1) == {}
        assert b.dense_states.shape == (50,)
        assert b.dense_states[42] == 1
        assert b.dense_states.sum() == 1

    def test_csr_sorted_batch_linear_chain(self) -> None:
        t = Trie()
        t.insert([1, 2, 3])
        b = CompactCSRTrie.from_sorted_batch(lex_sort([[1, 2, 3]]), vocab_size=8)
        assert_same_edges(CompactCSRTrie.from_trie(t, vocab_size=8), b, "[1,2,3]")
        assert children_of(b, 0) == {1: 1}
        assert children_of(b, 1) == {2: 2}
        assert children_of(b, 2) == {3: 3}
        assert children_of(b, 3) == {}
        assert b.dense_mask_by_layer[1].shape == (8, 8)
        assert b.dense_mask_by_layer[1][1, 2] and b.dense_mask_by_layer[1].sum() == 1
        assert b.dense_states.shape == (8, 8)
        assert b.dense_states[1, 2] == children_of(b, children_of(b, 0)[1])[2]

    def test_csr_sorted_batch_branching(self) -> None:
        t = Trie()
        t.insert([1, 2, 3])
        t.insert([1, 2, 7])
        b = CompactCSRTrie.from_sorted_batch(
            lex_sort([[1, 2, 3], [1, 2, 7]]), vocab_size=8
        )
        assert_same_edges(
            CompactCSRTrie.from_trie(t, vocab_size=8), b, "[1,2,3]+[1,2,7]"
        )
        assert children_of(b, 2) == {3: 3, 7: 4}
        assert b.dense_mask_by_layer[1][1, 2] and b.dense_mask_by_layer[1].sum() == 1
        assert b.dense_states[1, 2] == children_of(b, children_of(b, 0)[1])[2]

    def test_csr_sorted_batch_multi_branch(self) -> None:
        seqs = [[1, 2, 1], [3, 1, 2], [3, 1, 3]]
        t = Trie()
        for s in seqs:
            t.insert(s)
        b = CompactCSRTrie.from_sorted_batch(lex_sort(seqs), vocab_size=8)
        assert_same_edges(
            CompactCSRTrie.from_trie(t, vocab_size=8), b, "[1,2,1]+[3,1,2]+[3,1,3]"
        )
        assert children_of(b, 0) == {1: 1, 3: 2}
        assert children_of(b, 4) == {2: 6, 3: 7}
        assert b.dense_mask_by_layer[1][1, 2] and b.dense_mask_by_layer[1][3, 1]
        assert b.dense_mask_by_layer[1].sum() == 2
        for target in seqs:
            bfs_row = 0
            for token in target:
                ch = children_of(b, bfs_row)
                assert token in ch
                bfs_row = ch[token]

    def test_csr_sorted_batch_zero_tokens(self) -> None:
        seqs = [[0, 1, 2], [0, 3, 0], [2, 0, 1]]
        t = Trie()
        for s in seqs:
            t.insert(s)
        b = CompactCSRTrie.from_sorted_batch(lex_sort(seqs), vocab_size=4)
        assert_same_edges(
            CompactCSRTrie.from_trie(t, vocab_size=4), b, "zero-token seqs"
        )
        assert 0 in children_of(b, 0)
        assert 2 in children_of(b, 0)
        assert (
            b.dense_mask_by_layer[1][0, 1]
            and b.dense_mask_by_layer[1][0, 3]
            and b.dense_mask_by_layer[1][2, 0]
        )
        assert b.dense_mask_by_layer[1].sum() == 3
        for target in seqs:
            bfs_row = 0
            for token in target:
                ch = children_of(b, bfs_row)
                assert token in ch
                bfs_row = ch[token]

    def test_csr_sorted_batch_matches_trie(self) -> None:
        ids_sorted = torch.tensor(sorted(self.semantic_ids.tolist()), dtype=torch.long)
        trie_rand = build_trie(ids_sorted)
        csr_t = CompactCSRTrie.from_trie(trie_rand, vocab_size=8)
        csr_b = CompactCSRTrie.from_sorted_batch(ids_sorted, vocab_size=8)
        assert csr_b.dense_mask_by_layer[1].dtype == torch.bool
        assert csr_b.dense_mask_by_layer[1][ids_sorted[:, 0], ids_sorted[:, 1]].all()
        assert (csr_b.dense_states[ids_sorted[:, 0], ids_sorted[:, 1]] > 0).all()
        assert_same_edges(csr_t, csr_b, "random batch")
        for row in ids_sorted.tolist():
            bfs_t, bfs_b = 0, 0
            for token in row:
                ch_t = children_of(csr_t, bfs_t)
                ch_b = children_of(csr_b, bfs_b)
                assert token in ch_t and token in ch_b
                assert ch_t[token] == ch_b[token]
                bfs_t = ch_t[token]
                bfs_b = ch_b[token]

    # ---------------------------------------------------------------------------
    # vtnk_pytorch
    # ---------------------------------------------------------------------------

    def test_vtnk_batch1_root(self) -> None:
        csr = self.csr_three_seqs
        logits = torch.zeros(1, 8)
        nn, vi, cl = vtnk_pytorch(logits, torch.tensor([0]), csr, step=0)
        assert nn.shape == vi.shape == (1, csr.layer_max_branches[0])
        assert cl.shape == (1, 8)
        assert sorted(nn[0][nn[0] >= 0].tolist()) == [1, 2]
        assert sorted(vi[0][vi[0] >= 0].tolist()) == [1, 3]
        assert cl[0, 1] == 0.0 and cl[0, 3] == 0.0
        assert cl[0, 0].item() == float("-inf")
        assert (cl[0] > float("-inf")).sum() == 2

    def test_vtnk_batch2_same_root(self) -> None:
        csr = self.csr_three_seqs
        logits = torch.zeros(2, 8)
        nn, vi, cl = vtnk_pytorch(logits, torch.tensor([0, 0]), csr, step=0)
        assert nn.shape == vi.shape == (2, csr.layer_max_branches[0])
        assert (
            (nn[0] == nn[1]).all() and (vi[0] == vi[1]).all() and (cl[0] == cl[1]).all()
        )
        assert sorted(vi[0][vi[0] >= 0].tolist()) == [1, 3]

    def test_vtnk_batch2_different_nodes(self) -> None:
        csr = self.csr_three_seqs
        logits = torch.zeros(2, 8)
        nn, vi, cl = vtnk_pytorch(logits, torch.tensor([1, 2]), csr, step=1)
        assert nn[0, 0] == 3 and nn[1, 0] == 4
        assert vi[0, 0] == 2 and vi[1, 0] == 1
        assert cl[0, 2] == 0.0 and cl[0, 1].item() == float("-inf")
        assert cl[1, 1] == 0.0 and cl[1, 2].item() == float("-inf")

    def test_vtnk_batch3_mixed_branching(self) -> None:
        csr = self.csr_three_seqs
        logits = torch.zeros(3, 8)
        nn, vi, cl = vtnk_pytorch(logits, torch.tensor([3, 3, 4]), csr, step=2)
        assert nn.shape == vi.shape == (3, csr.layer_max_branches[2])
        assert (nn[0] == nn[1]).all() and (vi[0] == vi[1]).all()
        assert sorted(nn[2][nn[2] >= 0].tolist()) == [6, 7]
        assert sorted(vi[2][vi[2] >= 0].tolist()) == [2, 3]
        assert (cl[0] > float("-inf")).sum() == 1
        assert (cl[2] > float("-inf")).sum() == 2
