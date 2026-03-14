# ---------------------------------------------------------------------------
# Trie smoke-test
#
# Builds a Trie from a batch of semantic ID tuples (e.g. from RQVAETokenizer),
# then exercises prefix lookup and constrained-decoding queries.
# ---------------------------------------------------------------------------

import torch
from rectokens.decoding.trie import Trie
from rectokens.decoding.csr import CompactCSRTrie


def make_semantic_ids(
    num_items: int, num_levels: int, codebook_size: int, seed: int = 0
) -> torch.Tensor:
    """Generate a random batch of semantic ID tuples."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    return torch.randint(0, codebook_size, (num_items, num_levels), generator=rng)


def build_trie(semantic_ids: torch.Tensor) -> Trie:
    """Insert every item's token tuple into a fresh Trie."""
    trie = Trie()
    for row in semantic_ids.tolist():
        trie.insert(row)
    return trie


def valid_next_tokens(trie: Trie, prefix: list[int]) -> list[int]:
    """Return all valid next token IDs given a prefix."""
    node = trie.find_prefix(prefix)
    if node is None:
        return []
    return list(node.children.keys())


def main() -> None:
    NUM_ITEMS = 12
    NUM_LEVELS = 3
    CODEBOOK_SIZE = 8  # small so collisions are visible

    ids = make_semantic_ids(NUM_ITEMS, NUM_LEVELS, CODEBOOK_SIZE)
    print("Semantic IDs (each row = one item):")
    for i, row in enumerate(ids.tolist()):
        print(f"  item {i:2d}: {row}")

    trie = build_trie(ids)

    # ------------------------------------------------------------------
    # 1. Prefix lookup for every item — must find the node
    # ------------------------------------------------------------------
    print("\n--- Full-tuple lookup (all must succeed) ---")
    for i, row in enumerate(ids.tolist()):
        node = trie.find_prefix(row)
        ok = node is not None and node.is_end_of_token
        print(f"  item {i:2d} {row}: found={ok}")

    # ------------------------------------------------------------------
    # 2. Valid next tokens at each level for the first item
    # ------------------------------------------------------------------
    first = ids[0].tolist()
    print(f"\n--- Constrained decoding for item 0 {first} ---")
    for depth in range(NUM_LEVELS):
        prefix = first[:depth]
        nexts = valid_next_tokens(trie, prefix)
        print(f"  prefix={prefix}  valid_next={sorted(nexts)}")

    # ------------------------------------------------------------------
    # 3. Non-existent prefix → None
    # ------------------------------------------------------------------
    bad_prefix = [CODEBOOK_SIZE + 99]  # guaranteed out-of-vocab
    node = trie.find_prefix(bad_prefix)
    print(f"\n--- OOV prefix {bad_prefix}: node={node} (expected None) ---")

    # ------------------------------------------------------------------
    # 4. Duplicate insertion is idempotent
    # ------------------------------------------------------------------
    trie2 = build_trie(ids)
    build_trie(ids)  # insert same IDs again
    for row in ids.tolist():
        node = trie2.find_prefix(row)
        assert node is not None and node.is_end_of_token
    print("\nDuplicate insertion: all lookups still valid ✓")

    # ------------------------------------------------------------------
    # 5. Shared prefix: two items that share levels 0–1 must branch at level 2
    # ------------------------------------------------------------------
    shared_trie = Trie()
    shared_trie.insert([1, 2, 3])
    shared_trie.insert([1, 2, 7])
    nexts_after_shared = valid_next_tokens(shared_trie, [1, 2])
    print(
        f"\nShared prefix [1,2]: valid_next={sorted(nexts_after_shared)}  (expected [3, 7])"
    )
    assert sorted(nexts_after_shared) == [3, 7], f"Got {nexts_after_shared}"

    print("\nAll checks passed.")


def test_csr() -> None:
    from rectokens.decoding.csr import csr_from_trie as trie_to_csr, CompactCSRTrie

    # ------------------------------------------------------------------
    # Helper: look up valid (token, child_bfs_row) pairs for a node
    # ------------------------------------------------------------------
    def children_of(result: CompactCSRTrie, bfs_row: int) -> dict[int, int]:
        row_ptrs = result.row_ptrs.tolist()
        if bfs_row >= len(row_ptrs):
            return {}
        cols = result.stacked_cols_vals[0].tolist()
        vals = result.stacked_cols_vals[1].tolist()
        start = row_ptrs[bfs_row]
        end = row_ptrs[bfs_row + 1] if bfs_row + 1 < len(row_ptrs) else len(cols) - 1
        if start == end:
            return {}
        return {cols[j]: vals[j] for j in range(start, end)}

    # ------------------------------------------------------------------
    # 1. Single token [42] — root has one child, child has none
    # ------------------------------------------------------------------
    print("\n--- CSR: single token [42] ---")
    t1 = Trie()
    t1.insert([42])
    csr1 = trie_to_csr(t1, vocab_size=50, dense_lookup_layers=1)
    rows1 = csr1.row_ptrs.tolist()
    cols1 = csr1.stacked_cols_vals[0].tolist()
    vals1 = csr1.stacked_cols_vals[1].tolist()
    print(f"  crow_indices : {rows1}")
    print(f"  col_indices  : {cols1}")
    print(f"  values       : {vals1}")
    assert rows1 == [0, 1], f"row_ptrs: {rows1}"
    assert cols1[0] == 42, f"root child token: {cols1[0]}"
    assert vals1[0] == 1, f"root child BFS index: {vals1[0]}"
    assert children_of(csr1, 0) == {42: 1}, "root → node_42 via token 42"
    assert children_of(csr1, 1) == {}, "node_42 is a leaf"
    print("  ✓")

    # ------------------------------------------------------------------
    # 2. Linear chain [1, 2, 3] — 4 BFS nodes, 3 edges
    # ------------------------------------------------------------------
    print("\n--- CSR: linear chain [1,2,3] ---")
    t2 = Trie()
    t2.insert([1, 2, 3])
    csr2 = trie_to_csr(t2, vocab_size=8)
    rows2 = csr2.row_ptrs.tolist()
    cols2 = csr2.stacked_cols_vals[0].tolist()
    vals2 = csr2.stacked_cols_vals[1].tolist()
    print(f"  crow_indices : {rows2}")
    print(f"  col_indices  : {cols2}")
    print(f"  values       : {vals2}")
    assert rows2 == [0, 1, 2, 3], f"row_ptrs: {rows2}"
    assert children_of(csr2, 0) == {1: 1}, "root → node@1 via token 1"
    assert children_of(csr2, 1) == {2: 2}, "node@1 → node@2 via token 2"
    assert children_of(csr2, 2) == {3: 3}, "node@2 → node@3 via token 3"
    assert children_of(csr2, 3) == {}, "node@3 leaf"
    print("  ✓")

    # ------------------------------------------------------------------
    # 3. Branching trie [1,2,3] + [1,2,7] — 5 BFS nodes, 4 edges
    # ------------------------------------------------------------------
    print("\n--- CSR: branching [1,2,3] + [1,2,7] ---")
    t3 = Trie()
    t3.insert([1, 2, 3])
    t3.insert([1, 2, 7])
    csr3 = trie_to_csr(t3, vocab_size=8)
    rows3 = csr3.row_ptrs.tolist()
    cols3 = csr3.stacked_cols_vals[0].tolist()
    vals3 = csr3.stacked_cols_vals[1].tolist()
    print(f"  crow_indices : {rows3}")
    print(f"  col_indices  : {cols3}")
    print(f"  values       : {vals3}")
    assert rows3 == [0, 1, 2, 4, 4], f"row_ptrs: {rows3}"
    assert children_of(csr3, 0) == {1: 1}, "root → node@1"
    assert children_of(csr3, 1) == {2: 2}, "node@1 → node@12"
    assert children_of(csr3, 2) == {3: 3, 7: 4}, "node@12 → node@123 and node@127"
    assert children_of(csr3, 3) == {}, "node@123 leaf"
    assert children_of(csr3, 4) == {}, "node@127 leaf"
    print("  ✓")

    # ------------------------------------------------------------------
    # 4. Constrained beam walk through the CSR mirrors trie prefix lookup
    # ------------------------------------------------------------------
    print("\n--- CSR constrained-decoding walk: [1,2,3] + [1,2,7] ---")
    for target in ([1, 2, 3], [1, 2, 7]):
        bfs_row = 0
        for token in target:
            ch = children_of(csr3, bfs_row)
            assert token in ch, (
                f"token {token} not valid at bfs_row={bfs_row}; children={ch}"
            )
            bfs_row = ch[token]
        print(f"  walk {target} → terminal BFS row {bfs_row}  ✓")

    # ------------------------------------------------------------------
    # 5. CSR from the random batch trie — num rows = num trie nodes
    # ------------------------------------------------------------------
    print("\n--- CSR: random-batch trie ---")
    ids = make_semantic_ids(12, 3, 8, seed=0)
    trie_batch = build_trie(ids)
    csr_batch = trie_to_csr(trie_batch, vocab_size=8)
    n_rows = len(csr_batch.row_ptrs)
    print(f"  CSR rows (= trie nodes): {n_rows}")
    for row in ids.tolist():
        bfs_row = 0
        for token in row:
            ch = children_of(csr_batch, bfs_row)
            assert token in ch, f"token {token} missing at bfs_row={bfs_row}"
            bfs_row = ch[token]
    print("  All batch items reachable via CSR walk  ✓")

    # ------------------------------------------------------------------
    # 6. Multi-branch: [1,2,1], [3,1,2], [3,1,3]
    # ------------------------------------------------------------------
    print("\n--- CSR: [1,2,1] + [3,1,2] + [3,1,3] ---")
    t6 = Trie()
    t6.insert([1, 2, 1])
    t6.insert([3, 1, 2])
    t6.insert([3, 1, 3])
    csr6 = trie_to_csr(t6, vocab_size=8)
    rows6 = csr6.row_ptrs.tolist()
    cols6 = csr6.stacked_cols_vals[0].tolist()
    vals6 = csr6.stacked_cols_vals[1].tolist()
    print(f"  crow_indices : {rows6}")
    print(f"  col_indices  : {cols6}")
    print(f"  values       : {vals6}")

    assert rows6 == [0, 2, 3, 4, 5, 7, 7, 7], f"row_ptrs: {rows6}"
    assert cols6 == [1, 3, 2, 1, 1, 2, 3, -1], f"cols: {cols6}"
    assert vals6 == [1, 2, 3, 4, 5, 6, 7, -1], f"vals: {vals6}"

    assert children_of(csr6, 0) == {1: 1, 3: 2}, "root has two children"
    assert children_of(csr6, 1) == {2: 3}, "node[1] → node[1,2] via token 2"
    assert children_of(csr6, 2) == {1: 4}, "node[3] → node[3,1] via token 1"
    assert children_of(csr6, 3) == {1: 5}, "node[1,2] → node[1,2,1] via token 1"
    assert children_of(csr6, 4) == {2: 6, 3: 7}, "node[3,1] → leaves via tokens 2 and 3"
    assert children_of(csr6, 5) == {}, "node[1,2,1] leaf"
    assert children_of(csr6, 6) == {}, "node[3,1,2] leaf"
    assert children_of(csr6, 7) == {}, "node[3,1,3] leaf"

    for target in ([1, 2, 1], [3, 1, 2], [3, 1, 3]):
        bfs_row = 0
        for token in target:
            ch = children_of(csr6, bfs_row)
            assert token in ch, (
                f"token {token} not in children of bfs_row={bfs_row}: {ch}"
            )
            bfs_row = ch[token]
        print(f"  walk {target} → terminal BFS row {bfs_row}  ✓")

    print("\nAll CSR checks passed.")


def test_csr_sorted_batch() -> None:
    from rectokens.decoding.csr import csr_from_trie, csr_from_sorted_batch

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def lex_sort(rows: list[list[int]]) -> torch.Tensor:
        return torch.tensor(sorted(rows), dtype=torch.long)

    def children_of_batch(result: CompactCSRTrie, bfs_row: int) -> dict[int, int]:
        row_ptrs = result.row_ptrs.tolist()
        if bfs_row >= len(row_ptrs):
            return {}
        cols = result.stacked_cols_vals[0].tolist()
        vals = result.stacked_cols_vals[1].tolist()
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
        print(f"  trie rows : {rows_t}")
        print(f"  batch rows: {rows_b}")
        assert cols_t == cols_b, (
            f"{label} cols mismatch\n  trie : {cols_t}\n  batch: {cols_b}"
        )
        assert vals_t == vals_b, (
            f"{label} vals mismatch\n  trie : {vals_t}\n  batch: {vals_b}"
        )
        assert rows_t == rows_b, (
            f"{label} row_ptrs mismatch\n  trie : {rows_t}\n  batch: {rows_b}"
        )
        assert (t.dense_states == b.dense_states).all(), (
            f"{label} dense_states mismatch"
        )

    # ------------------------------------------------------------------
    # 1. Single token [42]
    # ------------------------------------------------------------------
    print("\n--- sorted_batch vs trie: [42] ---")
    t1 = Trie()
    t1.insert([42])
    b1 = csr_from_sorted_batch(lex_sort([[42]]), vocab_size=50, dense_lookup_layers=1)
    assert b1.dense_mask_by_layer[0].shape == (50,) and b1.dense_mask_by_layer[0][42]
    assert_same_edges(
        csr_from_trie(t1, vocab_size=50, dense_lookup_layers=1), b1, "[42]"
    )
    assert children_of_batch(b1, 0) == {42: 1}
    assert children_of_batch(b1, 1) == {}
    assert b1.dense_states.shape == (50,)
    assert b1.dense_states[42] == children_of_batch(b1, 0)[42]  # == 1
    assert b1.dense_states.sum() == 1
    print("  ✓")

    # ------------------------------------------------------------------
    # 2. Linear chain [1, 2, 3]
    # ------------------------------------------------------------------
    print("\n--- sorted_batch vs trie: [1,2,3] ---")
    t2 = Trie()
    t2.insert([1, 2, 3])
    b2 = csr_from_sorted_batch(lex_sort([[1, 2, 3]]), vocab_size=8)
    assert_same_edges(csr_from_trie(t2, vocab_size=8), b2, "[1,2,3]")
    assert children_of_batch(b2, 0) == {1: 1}
    assert children_of_batch(b2, 1) == {2: 2}
    assert children_of_batch(b2, 2) == {3: 3}
    assert children_of_batch(b2, 3) == {}
    # default dense_lookup_layers=2: mask shape (8,8), indexed by (tok0, tok1)
    assert (
        b2.dense_mask_by_layer[1].shape == (8, 8)
        and b2.dense_mask_by_layer[1][1, 2]
        and b2.dense_mask_by_layer[1].sum() == 1
    )
    assert b2.dense_states.shape == (8, 8)
    # dense_states[1, 2] == BFS node ID reached after tokens (1, 2) == 2
    assert (
        b2.dense_states[1, 2] == children_of_batch(b2, children_of_batch(b2, 0)[1])[2]
    )
    print("  ✓")

    # ------------------------------------------------------------------
    # 3. Branching [1,2,3] + [1,2,7]
    # ------------------------------------------------------------------
    print("\n--- sorted_batch vs trie: [1,2,3]+[1,2,7] ---")
    t3 = Trie()
    t3.insert([1, 2, 3])
    t3.insert([1, 2, 7])
    b3 = csr_from_sorted_batch(lex_sort([[1, 2, 3], [1, 2, 7]]), vocab_size=8)
    assert_same_edges(csr_from_trie(t3, vocab_size=8), b3, "[1,2,3]+[1,2,7]")
    assert children_of_batch(b3, 0) == {1: 1}
    assert children_of_batch(b3, 1) == {2: 2}
    assert children_of_batch(b3, 2) == {3: 3, 7: 4}
    assert children_of_batch(b3, 3) == {}
    assert children_of_batch(b3, 4) == {}
    # Both sequences share prefix (1,2), so only one cell is set
    assert (
        b3.dense_mask_by_layer[1].shape == (8, 8)
        and b3.dense_mask_by_layer[1][1, 2]
        and b3.dense_mask_by_layer[1].sum() == 1
    )
    assert b3.dense_states.shape == (8, 8)
    assert (
        b3.dense_states[1, 2] == children_of_batch(b3, children_of_batch(b3, 0)[1])[2]
    )
    print("  ✓")

    # ------------------------------------------------------------------
    # 4. Multi-branch [1,2,1] + [3,1,2] + [3,1,3]
    # ------------------------------------------------------------------
    print("\n--- sorted_batch vs trie: [1,2,1]+[3,1,2]+[3,1,3] ---")
    seqs4 = [[1, 2, 1], [3, 1, 2], [3, 1, 3]]
    t4 = Trie()
    for s in seqs4:
        t4.insert(s)
    b4 = csr_from_sorted_batch(lex_sort(seqs4), vocab_size=8)
    assert_same_edges(csr_from_trie(t4, vocab_size=8), b4, "[1,2,1]+[3,1,2]+[3,1,3]")
    assert children_of_batch(b4, 0) == {1: 1, 3: 2}
    assert children_of_batch(b4, 1) == {2: 3}
    assert children_of_batch(b4, 2) == {1: 4}
    assert children_of_batch(b4, 3) == {1: 5}
    assert children_of_batch(b4, 4) == {2: 6, 3: 7}
    assert children_of_batch(b4, 5) == {}
    assert children_of_batch(b4, 6) == {}
    assert children_of_batch(b4, 7) == {}
    for target in seqs4:
        bfs_row = 0
        for token in target:
            ch = children_of_batch(b4, bfs_row)
            assert token in ch, f"token {token} missing at bfs_row={bfs_row}: {ch}"
            bfs_row = ch[token]
        print(f"  walk {target} → terminal BFS row {bfs_row}  ✓")
    # Prefixes: (1,2) and (3,1) are the unique 2-token starts
    assert (
        b4.dense_mask_by_layer[1].shape == (8, 8)
        and b4.dense_mask_by_layer[1][1, 2]
        and b4.dense_mask_by_layer[1][3, 1]
        and b4.dense_mask_by_layer[1].sum() == 2
    )
    assert b4.dense_states.shape == (8, 8)
    node_1 = children_of_batch(b4, 0)[1]
    assert b4.dense_states[1, 2] == children_of_batch(b4, node_1)[2]
    node_3 = children_of_batch(b4, 0)[3]
    assert b4.dense_states[3, 1] == children_of_batch(b4, node_3)[1]

    # ------------------------------------------------------------------
    # 5. Token value 0 — verifies 0-indexed tokens are handled correctly
    #    vocab_size=4, tokens in [0, 3]; sequences contain token 0
    # ------------------------------------------------------------------
    print("\n--- sorted_batch vs trie: sequences with token 0 ---")
    seqs5 = [[0, 1, 2], [0, 3, 0], [2, 0, 1]]
    t5 = Trie()
    for s in seqs5:
        t5.insert(s)
    b5 = csr_from_sorted_batch(lex_sort(seqs5), vocab_size=4)
    assert_same_edges(csr_from_trie(t5, vocab_size=4), b5, "zero-token seqs")
    # token 0 must be a valid root child
    assert 0 in children_of_batch(b5, 0), "token 0 missing from root children"
    assert 2 in children_of_batch(b5, 0), "token 2 missing from root children"
    # dense mask: prefixes (0,1), (0,3), (2,0) must be set; nothing else
    assert b5.dense_mask_by_layer[1].shape == (4, 4)
    assert (
        b5.dense_mask_by_layer[1][0, 1]
        and b5.dense_mask_by_layer[1][0, 3]
        and b5.dense_mask_by_layer[1][2, 0]
    )
    assert b5.dense_mask_by_layer[1].sum() == 3
    assert b5.dense_states.shape == (4, 4)
    node_0 = children_of_batch(b5, 0)[0]
    assert b5.dense_states[0, 1] == children_of_batch(b5, node_0)[1]
    assert b5.dense_states[0, 3] == children_of_batch(b5, node_0)[3]
    node_2 = children_of_batch(b5, 0)[2]
    assert b5.dense_states[2, 0] == children_of_batch(b5, node_2)[0]
    # full walk for each sequence
    for target in seqs5:
        bfs_row = 0
        for token in target:
            ch = children_of_batch(b5, bfs_row)
            assert token in ch, f"token {token} missing at bfs_row={bfs_row}: {ch}"
            bfs_row = ch[token]
        print(f"  walk {target} → terminal BFS row {bfs_row}  ✓")

    # ------------------------------------------------------------------
    # 6. Random batch — walks must agree between both methods
    # ------------------------------------------------------------------
    print("\n--- sorted_batch vs trie: random batch ---")
    ids_raw = make_semantic_ids(12, 3, 8, seed=0)
    ids_sorted = torch.tensor(sorted(ids_raw.tolist()), dtype=torch.long)
    trie_rand = build_trie(ids_sorted)
    csr_t = csr_from_trie(trie_rand, vocab_size=8)
    csr_b = csr_from_sorted_batch(ids_sorted, vocab_size=8)
    assert csr_b.dense_mask_by_layer[1].dtype == torch.bool
    assert csr_b.dense_mask_by_layer[1].shape == (8, 8)
    assert csr_b.dense_mask_by_layer[1].any()
    # Every (tok0, tok1) pair in the corpus must be set in the mask
    assert csr_b.dense_mask_by_layer[1][ids_sorted[:, 0], ids_sorted[:, 1]].all()
    # dense_states must be nonzero at every corpus prefix
    assert (csr_b.dense_states[ids_sorted[:, 0], ids_sorted[:, 1]] > 0).all()
    assert_same_edges(csr_t, csr_b, "random batch")
    for row in ids_sorted.tolist():
        bfs_t, bfs_b = 0, 0
        for token in row:
            ch_t = children_of_batch(csr_t, bfs_t)
            ch_b = children_of_batch(csr_b, bfs_b)
            assert token in ch_t, f"trie: token {token} missing at {bfs_t}"
            assert token in ch_b, f"batch: token {token} missing at {bfs_b}"
            assert ch_t[token] == ch_b[token], (
                f"next-state mismatch for token {token}: trie={ch_t[token]} batch={ch_b[token]}"
            )
            bfs_t = ch_t[token]
            bfs_b = ch_b[token]
    print("  All walks agree between trie and sorted_batch CSR  ✓")

    print("\nAll csr_from_sorted_batch checks passed.")


def test_vtnk() -> None:
    from rectokens.decoding.csr import csr_from_sorted_batch
    from rectokens.decoding.vntk import vtnk_pytorch

    def lex_sort(rows: list[list[int]]) -> torch.Tensor:
        return torch.tensor(sorted(rows), dtype=torch.long)

    def uniform_logits(B: int, V: int) -> torch.Tensor:
        return torch.zeros(B, V)

    seqs = [[1, 2, 1], [3, 1, 2], [3, 1, 3]]
    csr = csr_from_sorted_batch(lex_sort(seqs), vocab_size=8)
    # row_ptrs = [0, 2, 3, 4, 5, 7, 7, 7]; nodes 0-6 safe, node 7 is last
    # BFS:  root(0) → {1:1, 3:2}
    #       node1(1) → {2:3}
    #       node2(2) → {1:4}
    #       node3(3) → {1:5}
    #       node4(4) → {2:6, 3:7}

    # ------------------------------------------------------------------
    # 1. Batch size 1 — root at step 0
    # ------------------------------------------------------------------
    print("\n--- vtnk B=1: root at step 0 ---")
    logits = uniform_logits(1, 8)
    nn, vi, cl = vtnk_pytorch(logits, torch.tensor([0]), csr, step=0)
    assert nn.shape == vi.shape == (1, csr.layer_max_branches[0])
    assert cl.shape == (1, 8)
    assert sorted(nn[0][nn[0] >= 0].tolist()) == [1, 2]  # child BFS IDs
    assert sorted(vi[0][vi[0] >= 0].tolist()) == [1, 3]  # valid tokens (1 and 3)
    assert cl[0, 1] == 0.0 and cl[0, 3] == 0.0  # valid tokens keep logit
    assert cl[0, 0].item() == float("-inf")  # invalid → -inf
    assert (cl[0] > float("-inf")).sum() == 2
    print("  ✓")

    # ------------------------------------------------------------------
    # 2. Two beams at root (same node) — all outputs identical across beams
    # ------------------------------------------------------------------
    print("\n--- vtnk B=2: both at root ---")
    logits = uniform_logits(2, 8)
    nn, vi, cl = vtnk_pytorch(logits, torch.tensor([0, 0]), csr, step=0)
    assert nn.shape == vi.shape == (2, csr.layer_max_branches[0])
    assert cl.shape == (2, 8)
    assert (nn[0] == nn[1]).all() and (vi[0] == vi[1]).all() and (cl[0] == cl[1]).all()
    assert sorted(vi[0][vi[0] >= 0].tolist()) == [1, 3]
    assert (cl[0] > float("-inf")).sum() == 2
    print("  ✓")

    # ------------------------------------------------------------------
    # 3. Two beams at different step-1 nodes — per-beam outputs differ
    #    node 1 → token 2, child BFS 3; node 2 → token 1, child BFS 4
    # ------------------------------------------------------------------
    print("\n--- vtnk B=2: two different nodes at step 1 ---")
    logits = uniform_logits(2, 8)
    nn, vi, cl = vtnk_pytorch(logits, torch.tensor([1, 2]), csr, step=1)
    assert nn.shape == vi.shape == (2, csr.layer_max_branches[1])
    assert cl.shape == (2, 8)
    assert nn[0, 0] == 3 and nn[1, 0] == 4  # child BFS IDs
    assert vi[0, 0] == 2 and vi[1, 0] == 1  # valid tokens
    assert cl[0, 2] == 0.0 and cl[0, 1].item() == float("-inf")
    assert cl[1, 1] == 0.0 and cl[1, 2].item() == float("-inf")
    print("  ✓")

    # ------------------------------------------------------------------
    # 4. Three beams at step 2: nodes 3, 3, 4
    #    node 3 → token 1, child BFS 5; node 4 → tokens 2,3, children BFS 6,7
    # ------------------------------------------------------------------
    print("\n--- vtnk B=3: mixed branching at step 2 ---")
    logits = uniform_logits(3, 8)
    nn, vi, cl = vtnk_pytorch(logits, torch.tensor([3, 3, 4]), csr, step=2)
    assert nn.shape == vi.shape == (3, csr.layer_max_branches[2])
    assert cl.shape == (3, 8)
    assert (nn[0] == nn[1]).all() and (vi[0] == vi[1]).all()  # same node → same row
    assert sorted(nn[2][nn[2] >= 0].tolist()) == [6, 7]
    assert sorted(vi[2][vi[2] >= 0].tolist()) == [2, 3]
    assert (cl[0] > float("-inf")).sum() == 1
    assert (cl[2] > float("-inf")).sum() == 2
    assert cl[2, 2] == 0.0 and cl[2, 3] == 0.0
    assert cl[2, 0].item() == float("-inf")
    print("  ✓")

    print("\nAll vtnk_pytorch checks passed.")


if __name__ == "__main__":
    main()
    test_csr()
    test_csr_sorted_batch()
    test_vtnk()
