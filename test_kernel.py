"""
Verify that the Triton constrained_node_transition kernel produces outputs
identical to the vtnk_pytorch CPU reference, when both run on GPU.
"""

import torch
from rectokens.decoding.csr import csr_from_sorted_batch
from rectokens.decoding.vntk import vtnk_pytorch
from rectokens.decoding.kernels.vtnk import constrained_node_transition, fused_linear_constrained_node_transition


def lex_sort(rows: list[list[int]]) -> torch.Tensor:
    return torch.tensor(sorted(rows), dtype=torch.long)


def check(
    name: str,
    logits: torch.Tensor,
    cur_node: torch.Tensor,
    csr,
    step: int,
    vocab_size: int,
    device: torch.device,
) -> None:
    logits_gpu = logits.to(device)
    cur_node_gpu = cur_node.to(device)

    ref_nn, ref_vi, ref_cl = vtnk_pytorch(logits_gpu, cur_node_gpu, csr, step)
    ker_nn, ker_vi, ker_cl = constrained_node_transition(logits_gpu, cur_node_gpu, csr, step, vocab_size)

    assert ker_nn.shape == ref_nn.shape, (
        f"[{name}] next_node shape mismatch: kernel={ker_nn.shape}, ref={ref_nn.shape}"
    )
    assert ker_vi.shape == ref_vi.shape, (
        f"[{name}] valid_idxs shape mismatch: kernel={ker_vi.shape}, ref={ref_vi.shape}"
    )
    assert ker_cl.shape == ref_cl.shape, (
        f"[{name}] corrected_logits shape mismatch: kernel={ker_cl.shape}, ref={ref_cl.shape}"
    )
    assert torch.equal(ker_nn, ref_nn), (
        f"[{name}] next_node mismatch:\n  kernel={ker_nn}\n  ref={ref_nn}"
    )
    assert torch.equal(ker_vi, ref_vi), (
        f"[{name}] valid_idxs mismatch:\n  kernel={ker_vi}\n  ref={ref_vi}"
    )
    assert torch.allclose(ker_cl, ref_cl, equal_nan=True), (
        f"[{name}] corrected_logits mismatch:\n  kernel={ker_cl}\n  ref={ref_cl}"
    )
    print(f"  PASS  {name}")


def main() -> None:
    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")

    torch.manual_seed(0)

    # ------------------------------------------------------------------ #
    # Trie: seqs = [[1,2,1],[3,1,2],[3,1,3]]
    # BFS row_ptrs = [0,2,3,4,5,7,7,7]
    # node 0 (root): children {1->1, 3->2}
    # node 1: children {2->3}
    # node 2: children {1->4}
    # node 3: children {1->5}       (leaf in 3-step trie, but safe: row 3+1=4 exists)
    # node 4: children {2->6, 3->7}
    # nodes 5,6,7: leaves (6 and 7 are the last two BFS nodes — unsafe to call)
    # ------------------------------------------------------------------ #
    seqs = [[1, 2, 1], [3, 1, 2], [3, 1, 3]]
    vocab_size = 8
    csr = csr_from_sorted_batch(lex_sort(seqs), vocab_size=vocab_size)
    csr = csr._replace(
        row_ptrs=csr.row_ptrs.to(device),
        stacked_cols_vals=csr.stacked_cols_vals.to(device),
        dense_lookup_mask=csr.dense_lookup_mask.to(device),
        dense_states=csr.dense_states.to(device),
    )

    def logits(B: int) -> torch.Tensor:
        return torch.randn(B, vocab_size)

    # 1. B=1, step 0, root node
    check("B=1 step=0 root", logits(1), torch.tensor([0]), csr, step=0, vocab_size=vocab_size, device=device)

    # 2. B=2, step=0, same root node
    check("B=2 step=0 root", logits(2), torch.tensor([0, 0]), csr, step=0, vocab_size=vocab_size, device=device)

    # 3. B=2, step=1, different nodes (node 1 and node 2)
    check("B=2 step=1 nodes[1,2]", logits(2), torch.tensor([1, 2]), csr, step=1, vocab_size=vocab_size, device=device)

    # 4. B=3, step=2, mixed nodes (nodes 3, 4, 3)
    check("B=3 step=2 nodes[3,4,3]", logits(3), torch.tensor([3, 4, 3]), csr, step=2, vocab_size=vocab_size, device=device)

    # 5. B=4, step=1, node 4 has 2 children (branching)
    check("B=4 step=1 nodes[1,2,1,2]", logits(4), torch.tensor([1, 2, 1, 2]), csr, step=1, vocab_size=vocab_size, device=device)

    # 6. Large random batch at step 0 (all at root)
    B_large = 256
    check(
        f"B={B_large} step=0 root (large batch)",
        torch.randn(B_large, vocab_size),
        torch.zeros(B_large, dtype=torch.long),
        csr, step=0, vocab_size=vocab_size, device=device,
    )

    # 7. Longer sequence trie — verify step-by-step walk
    seqs2 = [[i, j, k] for i in range(4) for j in range(4) for k in range(4)]
    vocab_size2 = 16
    csr2 = csr_from_sorted_batch(lex_sort(seqs2), vocab_size=vocab_size2)
    csr2 = csr2._replace(
        row_ptrs=csr2.row_ptrs.to(device),
        stacked_cols_vals=csr2.stacked_cols_vals.to(device),
        dense_lookup_mask=csr2.dense_lookup_mask.to(device),
        dense_states=csr2.dense_states.to(device),
    )
    B7 = 32
    check(
        "dense trie B=32 step=0",
        torch.randn(B7, vocab_size2),
        torch.zeros(B7, dtype=torch.long),
        csr2, step=0, vocab_size=vocab_size2, device=device,
    )
    # advance to step 1: pick the first valid child for each batch element
    ref_nn0, _vi0, _cl0 = vtnk_pytorch(
        torch.randn(B7, vocab_size2, device=device),
        torch.zeros(B7, dtype=torch.long, device=device),
        csr2, step=0,
    )
    next_nodes = ref_nn0[:, 0]  # first child of root for each element (all the same here)
    check(
        "dense trie B=32 step=1",
        torch.randn(B7, vocab_size2),
        next_nodes.cpu(),
        csr2, step=1, vocab_size=vocab_size2, device=device,
    )

    # ------------------------------------------------------------------ #
    # valid_idxs correctness: verify semantic content, not just ref match
    # ------------------------------------------------------------------ #
    print("\nvalid_idxs correctness tests:")

    def check_valid_idxs(name: str, cur_node: torch.Tensor, csr, step: int, expected_tokens_per_batch: list[list[int]], device: torch.device) -> None:
        """
        Verify that valid_idxs for each batch element contains exactly the
        expected token indices (in any order), with -1 padding for the rest.
        """
        dummy_logits = torch.zeros(cur_node.shape[0], vocab_size, device=device)
        _, vi, _ = constrained_node_transition(dummy_logits, cur_node.to(device), csr, step, vocab_size)
        for b, expected in enumerate(expected_tokens_per_batch):
            actual = sorted(vi[b][vi[b] >= 0].tolist())
            assert actual == sorted(expected), (
                f"[{name}] batch {b}: valid_idxs={actual}, expected={expected}"
            )
            assert (vi[b][vi[b] < 0] == -1).all(), (
                f"[{name}] batch {b}: padding entries should be -1, got {vi[b][vi[b] < 0].tolist()}"
            )
            assert vi[b].shape[0] == csr.layer_max_branches[step], (
                f"[{name}] batch {b}: static length should be layer_max_branches[{step}]={csr.layer_max_branches[step]}, got {vi[b].shape[0]}"
            )
        print(f"  PASS  {name}")

    # node 0 (root): children tokens {1, 3}
    check_valid_idxs("root has tokens {1,3}", torch.tensor([0]), csr, step=0, expected_tokens_per_batch=[[1, 3]], device=device)

    # node 1: child token {2}; node 2: child token {1} — different children per batch element
    check_valid_idxs("nodes[1,2] have tokens {2},{1}", torch.tensor([1, 2]), csr, step=1, expected_tokens_per_batch=[[2], [1]], device=device)

    # node 4: children tokens {2, 3}
    check_valid_idxs("node 4 has tokens {2,3}", torch.tensor([4]), csr, step=2, expected_tokens_per_batch=[[2, 3]], device=device)

    # node 3: child token {1}; node 4: children tokens {2, 3} — mixed branch counts
    check_valid_idxs("nodes[3,4] have tokens {1},{2,3}", torch.tensor([3, 4]), csr, step=2, expected_tokens_per_batch=[[1], [2, 3]], device=device)

    # ------------------------------------------------------------------ #
    # Fused linear kernel
    # ------------------------------------------------------------------ #
    print("\nFused linear kernel tests:")

    def check_fused(
        name: str,
        a: torch.Tensor,
        b: torch.Tensor,
        cur_node: torch.Tensor,
        csr,
        step: int,
        device: torch.device,
    ) -> None:
        a_gpu = a.to(device)
        b_gpu = b.to(device)
        cur_node_gpu = cur_node.to(device)

        # Reference: unfused matmul then constrained transition
        ref_logits = (a_gpu @ b_gpu).float()
        ref_nn, ref_vi, ref_cl = vtnk_pytorch(ref_logits, cur_node_gpu, csr, step)

        ker_nn, ker_vi, ker_cl = fused_linear_constrained_node_transition(a_gpu, b_gpu, cur_node_gpu, csr, step)

        assert ker_nn.shape == ref_nn.shape, f"[{name}] next_node shape: {ker_nn.shape} vs {ref_nn.shape}"
        assert ker_vi.shape == ref_vi.shape, f"[{name}] valid_idxs shape: {ker_vi.shape} vs {ref_vi.shape}"
        assert ker_cl.shape == ref_cl.shape, f"[{name}] corrected_logits shape: {ker_cl.shape} vs {ref_cl.shape}"
        assert torch.equal(ker_nn, ref_nn), f"[{name}] next_node mismatch:\n  kernel={ker_nn}\n  ref={ref_nn}"
        assert torch.equal(ker_vi, ref_vi), f"[{name}] valid_idxs mismatch:\n  kernel={ker_vi}\n  ref={ref_vi}"
        # atol for tf32 accumulation differences
        assert torch.allclose(ker_cl, ref_cl, atol=1e-2, equal_nan=True), (
            f"[{name}] corrected_logits mismatch (max diff={( (ker_cl - ref_cl)[torch.isfinite(ker_cl - ref_cl)]).abs().max():.4f})"
        )
        print(f"  PASS  {name}")

    K = 16  # small hidden dim for fast tests
    torch.manual_seed(42)

    # 1. B=1, step=0, root
    check_fused(
        "fused B=1 step=0 root",
        torch.randn(1, K), torch.randn(K, vocab_size),
        torch.tensor([0]), csr, step=0, device=device,
    )

    # 2. B=2, step=1, different nodes
    check_fused(
        "fused B=2 step=1 nodes[1,2]",
        torch.randn(2, K), torch.randn(K, vocab_size),
        torch.tensor([1, 2]), csr, step=1, device=device,
    )

    # 3. B=3, step=2, mixed branching (node 4 has 2 children)
    check_fused(
        "fused B=3 step=2 nodes[3,4,3]",
        torch.randn(3, K), torch.randn(K, vocab_size),
        torch.tensor([3, 4, 3]), csr, step=2, device=device,
    )

    # 4. Large batch with K > BLOCK_K to exercise multi-tile K-loop
    K_large = 128
    check_fused(
        "fused B=8 K=128 step=0 root",
        torch.randn(8, K_large), torch.randn(K_large, vocab_size),
        torch.zeros(8, dtype=torch.long), csr, step=0, device=device,
    )

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
