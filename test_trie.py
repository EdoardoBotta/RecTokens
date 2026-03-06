# ---------------------------------------------------------------------------
# Trie smoke-test
#
# Builds a Trie from a batch of semantic ID tuples (e.g. from RQVAETokenizer),
# then exercises prefix lookup and constrained-decoding queries.
# ---------------------------------------------------------------------------

import torch
from rectokens.decoding.trie import Trie


def make_semantic_ids(num_items: int, num_levels: int, codebook_size: int, seed: int = 0) -> torch.Tensor:
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
    CODEBOOK_SIZE = 8   # small so collisions are visible

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
    bad_prefix = [CODEBOOK_SIZE + 99]   # guaranteed out-of-vocab
    node = trie.find_prefix(bad_prefix)
    print(f"\n--- OOV prefix {bad_prefix}: node={node} (expected None) ---")

    # ------------------------------------------------------------------
    # 4. Duplicate insertion is idempotent
    # ------------------------------------------------------------------
    trie2 = build_trie(ids)
    build_trie(ids)   # insert same IDs again
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
    print(f"\nShared prefix [1,2]: valid_next={sorted(nexts_after_shared)}  (expected [3, 7])")
    assert sorted(nexts_after_shared) == [3, 7], f"Got {nexts_after_shared}"

    print("\nAll checks passed.")


def test_csr() -> None:
    from rectokens.decoding.csr import csr_from_trie as trie_to_csr

    # ------------------------------------------------------------------
    # Helper: look up valid (token, child_bfs_row) pairs for a node
    # ------------------------------------------------------------------
    def children_of(csr, bfs_row: int) -> dict[int, int]:
        """Return {token_id: child_bfs_row} for the given BFS row index.

        Leaf nodes that are last in BFS order are not assigned a CSR row
        (their row_ptrs entry is appended but n_rows = len(crow)-1 falls
        one short), so bfs_row >= n_rows means the node is a leaf.
        """
        crow = csr.crow_indices().tolist()
        n_rows = len(crow) - 1
        if bfs_row >= n_rows:
            return {}   # leaf node beyond the CSR row range
        cols = csr.col_indices().tolist()
        vals = csr.values().tolist()
        start, end = crow[bfs_row], crow[bfs_row + 1]
        return {cols[j]: vals[j] for j in range(start, end)}

    # ------------------------------------------------------------------
    # 1. Single token [42] — root has one child, child has none
    # ------------------------------------------------------------------
    print("\n--- CSR: single token [42] ---")
    t1 = Trie()
    t1.insert([42])
    csr1 = trie_to_csr(t1)
    crow1 = csr1.crow_indices().tolist()
    cols1 = csr1.col_indices().tolist()
    vals1 = csr1.values().tolist()
    print(f"  crow_indices : {crow1}")
    print(f"  col_indices  : {cols1}")
    print(f"  values       : {vals1}")
    # BFS: root(0) → node_42(1)
    # crow=[0,1] → n_rows=1; node_42 (BFS 1) has no row — leaf beyond CSR range
    # The -1 sentinel sits at col_indices[1] but crow[-1]=1 excludes it from row 0.
    assert crow1 == [0, 1],              f"crow: {crow1}"
    assert cols1[0] == 42,               f"root child token: {cols1[0]}"
    assert vals1[0] == 1,                f"root child BFS index: {vals1[0]}"
    assert children_of(csr1, 0) == {42: 1}, "root → node_42 via token 42"
    assert children_of(csr1, 1) == {},      "node_42 is a leaf (beyond n_rows)"
    print("  ✓")

    # ------------------------------------------------------------------
    # 2. Linear chain [1, 2, 3] — 4 BFS nodes, 3 edges, n_rows=3
    # ------------------------------------------------------------------
    print("\n--- CSR: linear chain [1,2,3] ---")
    t2 = Trie()
    t2.insert([1, 2, 3])
    csr2 = trie_to_csr(t2)
    crow2 = csr2.crow_indices().tolist()
    cols2 = csr2.col_indices().tolist()
    vals2 = csr2.values().tolist()
    print(f"  crow_indices : {crow2}")
    print(f"  col_indices  : {cols2}")
    print(f"  values       : {vals2}")
    # BFS: root(0), node@1(1), node@2(2), node@3(3)
    # crow=[0,1,2,3] → n_rows=3; node@3 (BFS 3) has no row — leaf beyond CSR
    assert crow2 == [0, 1, 2, 3], f"crow: {crow2}"
    assert children_of(csr2, 0) == {1: 1}, "root → node@1 via token 1"
    assert children_of(csr2, 1) == {2: 2}, "node@1 → node@2 via token 2"
    assert children_of(csr2, 2) == {3: 3}, "node@2 → node@3 via token 3"
    assert children_of(csr2, 3) == {},     "node@3 leaf (beyond n_rows)"
    print("  ✓")

    # ------------------------------------------------------------------
    # 3. Branching trie [1,2,3] + [1,2,7] — 5 BFS nodes, 4 edges, n_rows=4
    # ------------------------------------------------------------------
    print("\n--- CSR: branching [1,2,3] + [1,2,7] ---")
    t3 = Trie()
    t3.insert([1, 2, 3])
    t3.insert([1, 2, 7])
    csr3 = trie_to_csr(t3)
    crow3 = csr3.crow_indices().tolist()
    cols3 = csr3.col_indices().tolist()
    vals3 = csr3.values().tolist()
    print(f"  crow_indices : {crow3}")
    print(f"  col_indices  : {cols3}")
    print(f"  values       : {vals3}")
    # BFS: root(0), node@1(1), node@12(2), node@123(3), node@127(4)
    # crow=[0,1,2,4,4] → n_rows=4
    # node@123 (BFS 3) is row 3 — empty row (leaf within CSR)
    # node@127 (BFS 4) has no row — last BFS leaf beyond CSR range
    assert crow3 == [0, 1, 2, 4, 4],          f"crow: {crow3}"
    assert children_of(csr3, 0) == {1: 1},       "root → node@1"
    assert children_of(csr3, 1) == {2: 2},       "node@1 → node@12"
    assert children_of(csr3, 2) == {3: 3, 7: 4}, "node@12 → node@123 and node@127"
    assert children_of(csr3, 3) == {},            "node@123 leaf (empty row in CSR)"
    assert children_of(csr3, 4) == {},            "node@127 leaf (beyond n_rows)"
    print("  ✓")

    # ------------------------------------------------------------------
    # 4. Constrained beam walk through the CSR mirrors trie prefix lookup
    # ------------------------------------------------------------------
    print("\n--- CSR constrained-decoding walk: [1,2,3] + [1,2,7] ---")
    for target in ([1, 2, 3], [1, 2, 7]):
        bfs_row = 0
        for token in target:
            ch = children_of(csr3, bfs_row)
            assert token in ch, f"token {token} not valid at bfs_row={bfs_row}; children={ch}"
            bfs_row = ch[token]
        print(f"  walk {target} → terminal BFS row {bfs_row}  ✓")

    # ------------------------------------------------------------------
    # 5. CSR from the random batch trie — num rows = num trie nodes
    # ------------------------------------------------------------------
    print("\n--- CSR: random-batch trie ---")
    ids = make_semantic_ids(12, 3, 8, seed=0)
    trie_batch = build_trie(ids)
    csr_batch = trie_to_csr(trie_batch)
    n_rows = csr_batch.crow_indices().shape[0] - 1
    print(f"  CSR rows (= trie nodes): {n_rows}")
    # Every item in the batch must be reachable via CSR walk
    for row in ids.tolist():
        bfs_row = 0
        for token in row:
            ch = children_of(csr_batch, bfs_row)
            assert token in ch, f"token {token} missing at bfs_row={bfs_row}"
            bfs_row = ch[token]
    print("  All batch items reachable via CSR walk  ✓")

    # ------------------------------------------------------------------
    # 6. Multi-branch: [1,2,1], [3,1,2], [3,1,3]
    #    Two root children, mixed depths, two leaves in CSR + one beyond range
    # ------------------------------------------------------------------
    print("\n--- CSR: [1,2,1] + [3,1,2] + [3,1,3] ---")
    t6 = Trie()
    t6.insert([1, 2, 1])
    t6.insert([3, 1, 2])
    t6.insert([3, 1, 3])
    csr6 = trie_to_csr(t6)
    crow6 = csr6.crow_indices().tolist()
    cols6 = csr6.col_indices().tolist()
    vals6 = csr6.values().tolist()
    print(f"  crow_indices : {crow6}")
    print(f"  col_indices  : {cols6}")
    print(f"  values       : {vals6}")

    # BFS assignment:
    #   0=root, 1=node[1], 2=node[3], 3=node[1,2], 4=node[3,1],
    #   5=node[1,2,1], 6=node[3,1,2], 7=node[3,1,3]
    # crow=[0,2,3,4,5,7,7,7] → n_rows=7; BFS 7 (node[3,1,3]) has no row
    assert crow6 == [0, 2, 3, 4, 5, 7, 7, 7],       f"crow: {crow6}"
    assert cols6  == [1, 3, 2, 1, 1, 2, 3, -1],      f"cols: {cols6}"
    assert vals6  == [1, 2, 3, 4, 5, 6, 7, -1],      f"vals: {vals6}"

    assert children_of(csr6, 0) == {1: 1, 3: 2}, "root has two children"
    assert children_of(csr6, 1) == {2: 3},        "node[1] → node[1,2] via token 2"
    assert children_of(csr6, 2) == {1: 4},        "node[3] → node[3,1] via token 1"
    assert children_of(csr6, 3) == {1: 5},        "node[1,2] → node[1,2,1] via token 1"
    assert children_of(csr6, 4) == {2: 6, 3: 7},  "node[3,1] → leaves via tokens 2 and 3"
    assert children_of(csr6, 5) == {},             "node[1,2,1] leaf (empty row)"
    assert children_of(csr6, 6) == {},             "node[3,1,2] leaf (empty row)"
    assert children_of(csr6, 7) == {},             "node[3,1,3] leaf (beyond n_rows)"

    # Full CSR walk for all three sequences
    for target in ([1, 2, 1], [3, 1, 2], [3, 1, 3]):
        bfs_row = 0
        for token in target:
            ch = children_of(csr6, bfs_row)
            assert token in ch, f"token {token} not in children of bfs_row={bfs_row}: {ch}"
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

    def children_of_batch(csr, bfs_row: int) -> dict[int, int]:
        """Same semantics as children_of() but works with -1 leaf crow entries."""
        crow = csr.crow_indices().tolist()
        n_rows = len(crow) - 1
        if bfs_row >= n_rows:
            return {}
        cols = csr.col_indices().tolist()
        vals = csr.values().tolist()
        start, end = crow[bfs_row], crow[bfs_row + 1]
        # -1 crow entries mean an empty leaf row (range(-1,-1) is empty)
        if start == end:
            return {}
        return {cols[j]: vals[j] for j in range(start, end)}

    def assert_same_edges(csr_t, csr_b, label: str) -> None:
        """cols and vals must match; crow may differ only for leaf rows."""
        cols_t = csr_t.col_indices().tolist()
        cols_b = csr_b.col_indices().tolist()
        vals_t = csr_t.values().tolist()
        vals_b = csr_b.values().tolist()
        crow_t = csr_t.crow_indices().tolist()
        crow_b = csr_b.crow_indices().tolist()
        print(f"  trie crow : {crow_t}")
        print(f"  batch crow: {crow_b}")
        assert cols_t == cols_b, f"{label} cols mismatch\n  trie : {cols_t}\n  batch: {cols_b}"
        assert vals_t == vals_b, f"{label} vals mismatch\n  trie : {vals_t}\n  batch: {vals_b}"
        # crow must agree on all non-leaf rows (positive values in both)
        for i, (ct, cb) in enumerate(zip(crow_t, crow_b)):
            if ct >= 0 and cb >= 0:
                assert ct == cb, f"{label} crow[{i}] mismatch: trie={ct} batch={cb}"

    # ------------------------------------------------------------------
    # 1. Single token [42]
    # ------------------------------------------------------------------
    print("\n--- sorted_batch vs trie: [42] ---")
    t1 = Trie(); t1.insert([42])
    b1, _ = csr_from_sorted_batch(lex_sort([[42]]))
    assert_same_edges(csr_from_trie(t1), b1, "[42]")
    assert children_of_batch(b1, 0) == {42: 1}
    assert children_of_batch(b1, 1) == {}
    print("  ✓")

    # ------------------------------------------------------------------
    # 2. Linear chain [1, 2, 3]
    # ------------------------------------------------------------------
    print("\n--- sorted_batch vs trie: [1,2,3] ---")
    t2 = Trie(); t2.insert([1, 2, 3])
    b2, _ = csr_from_sorted_batch(lex_sort([[1, 2, 3]]))
    assert_same_edges(csr_from_trie(t2), b2, "[1,2,3]")
    assert children_of_batch(b2, 0) == {1: 1}
    assert children_of_batch(b2, 1) == {2: 2}
    assert children_of_batch(b2, 2) == {3: 3}
    assert children_of_batch(b2, 3) == {}
    print("  ✓")

    # ------------------------------------------------------------------
    # 3. Branching [1,2,3] + [1,2,7]
    # ------------------------------------------------------------------
    print("\n--- sorted_batch vs trie: [1,2,3]+[1,2,7] ---")
    t3 = Trie(); t3.insert([1, 2, 3]); t3.insert([1, 2, 7])
    b3, _ = csr_from_sorted_batch(lex_sort([[1, 2, 3], [1, 2, 7]]))
    assert_same_edges(csr_from_trie(t3), b3, "[1,2,3]+[1,2,7]")
    assert children_of_batch(b3, 0) == {1: 1}
    assert children_of_batch(b3, 1) == {2: 2}
    assert children_of_batch(b3, 2) == {3: 3, 7: 4}
    assert children_of_batch(b3, 3) == {}
    assert children_of_batch(b3, 4) == {}
    print("  ✓")

    # ------------------------------------------------------------------
    # 4. Multi-branch [1,2,1] + [3,1,2] + [3,1,3]
    # ------------------------------------------------------------------
    print("\n--- sorted_batch vs trie: [1,2,1]+[3,1,2]+[3,1,3] ---")
    seqs4 = [[1, 2, 1], [3, 1, 2], [3, 1, 3]]
    t4 = Trie()
    for s in seqs4:
        t4.insert(s)
    b4, _ = csr_from_sorted_batch(lex_sort(seqs4))
    assert_same_edges(csr_from_trie(t4), b4, "[1,2,1]+[3,1,2]+[3,1,3]")
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

    # ------------------------------------------------------------------
    # 5. Random batch — walks must agree between both methods
    # ------------------------------------------------------------------
    print("\n--- sorted_batch vs trie: random batch ---")
    ids_raw = make_semantic_ids(12, 3, 8, seed=0)
    ids_sorted = torch.tensor(sorted(ids_raw.tolist()), dtype=torch.long)
    trie_rand = build_trie(ids_sorted)
    csr_t = csr_from_trie(trie_rand)
    csr_b, _ = csr_from_sorted_batch(ids_sorted)
    assert_same_edges(csr_t, csr_b, "random batch")
    for row in ids_sorted.tolist():
        bfs_t, bfs_b = 0, 0
        for token in row:
            ch_t = children_of_batch(csr_t, bfs_t)
            ch_b = children_of_batch(csr_b, bfs_b)
            assert token in ch_t, f"trie: token {token} missing at {bfs_t}"
            assert token in ch_b, f"batch: token {token} missing at {bfs_b}"
            assert ch_t[token] == ch_b[token], \
                f"next-state mismatch for token {token}: trie={ch_t[token]} batch={ch_b[token]}"
            bfs_t = ch_t[token]
            bfs_b = ch_b[token]
    print("  All walks agree between trie and sorted_batch CSR  ✓")

    print("\nAll csr_from_sorted_batch checks passed.")


if __name__ == "__main__":
    main()
    test_csr()
    test_csr_sorted_batch()
