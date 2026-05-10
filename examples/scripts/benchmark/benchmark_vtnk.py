"""
Benchmark: fused_linear_constrained_node_transition
    vs torch.compile(nn.Linear) + constrained_node_transition (Triton kernel)
    vs torch.compile(nn.Linear) + vtnk_pytorch
    vs torch.compile(sparse_linear_pytorch)

Grid: B (batch size) × N (vocab / logits size). K (hidden dim) fixed.
"""

import argparse
import os
import time
import torch
import torch.nn as nn
import triton.testing as testing
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from rectokens.schemas.compact_csr_trie import CompactCSRTrie
from rectokens.schemas.state import ConstraintState
from rectokens.decoding.vntk import vtnk_pytorch, sparse_linear_pytorch
from rectokens.decoding.trie import Trie, TrieNode
from rectokens.ops.constrained_node_transition import (
    constrained_node_transition,
    fused_linear_constrained_node_transition,
)

DEVICE = torch.device("cuda")
K = 512
WARMUP = 25
REP = 100

ALL_ALGORITHMS = ["fused", "kernel", "pytorch", "sparse_pytorch", "trie_cpu"]
DEFAULT_ALGORITHMS = ["fused", "kernel", "pytorch", "sparse_pytorch"]
DEFAULT_SPARSITY = 0.01


def lex_sort(rows: list[list[int]]) -> torch.Tensor:
    return torch.tensor(sorted(rows), dtype=torch.long)


def make_csr(vocab_size: int, max_branches: int) -> object:
    seqs = [[i] for i in range(max_branches)]
    csr = CompactCSRTrie.from_sorted_batch(lex_sort(seqs), vocab_size=vocab_size)
    return csr._replace(
        row_ptrs=csr.row_ptrs.to(DEVICE),
        stacked_cols_vals=csr.stacked_cols_vals.to(DEVICE),
        dense_mask_by_layer=[v.to(DEVICE) for v in csr.dense_mask_by_layer],
        dense_states=csr.dense_states.to(DEVICE),
    )


def make_trie(max_branches: int) -> tuple[Trie, list[TrieNode]]:
    """Build a Trie and return it along with a BFS-ordered node list for integer indexing."""
    trie = Trie()
    for i in range(max_branches):
        trie.insert([i])
    # BFS node list so that node index 0 == root (matching cur_node convention)
    nodes: list[TrieNode] = []
    queue = [trie.root]
    while queue:
        node = queue.pop(0)
        nodes.append(node)
        for child in node.children.values():
            queue.append(child)
    return trie, nodes


def trie_cpu_traversal(
    cur_node_cpu: torch.Tensor, nodes: list[TrieNode], vocab_size: int
) -> torch.Tensor:
    """For each batch item, collect allowed next tokens by walking the CPU trie."""
    B = cur_node_cpu.shape[0]
    mask = torch.zeros(B, vocab_size, dtype=torch.bool)
    for i in range(B):
        node = nodes[cur_node_cpu[i].item()]
        for tok in node.children:
            mask[i, tok] = True
    return mask


def run_bench(fn):
    return testing.do_bench(fn, warmup=WARMUP, rep=REP)


def run_bench_cpu(fn, warmup: int = WARMUP, rep: int = REP) -> float:
    """Wall-clock benchmark for CPU-only functions (ms)."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(rep):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    return float(sum(times[: max(1, rep // 2)]) / max(1, rep // 2))


def benchmark_grid(B_vals, N_vals, algorithms, sparsity):
    alg_set = set(algorithms)
    gpu_algos = alg_set & {"fused", "kernel", "pytorch", "sparse_pytorch"}
    records = []

    for B in B_vals:
        for N in N_vals:
            max_branches = max(1, int(N * sparsity))
            print(f"  B={B:6d}  N={N:6d}  max_branches={max_branches}")

            if gpu_algos:
                csr = make_csr(vocab_size=N, max_branches=max_branches)
            if "trie_cpu" in alg_set:
                _, trie_nodes = make_trie(max_branches=max_branches)

            a = torch.randn(B, K, device=DEVICE)
            weight = torch.randn(N, K, device=DEVICE)
            cur_node = torch.zeros(B, dtype=torch.long, device=DEVICE)
            if "trie_cpu" in alg_set:
                cur_node_cpu = cur_node.cpu()

            if "pytorch" in alg_set:
                linear = torch.compile(nn.Linear(K, N, bias=False).to(DEVICE))
                with torch.no_grad():
                    linear.weight.data.copy_(weight)

            if "sparse_pytorch" in alg_set:
                sparse_linear_pytorch_compiled = torch.compile(sparse_linear_pytorch)

            if gpu_algos:
                cs = ConstraintState(step=0, trie=csr, cur_node=cur_node)

            # --- warmup / force compilation ---
            with torch.no_grad():
                if "fused" in alg_set:
                    fused_linear_constrained_node_transition(a, weight.T, cs)
                if "kernel" in alg_set:
                    constrained_node_transition(a @ weight.T, cs)
                if "pytorch" in alg_set:
                    vtnk_pytorch(linear(a), cur_node, csr, step=0)
                if "sparse_pytorch" in alg_set:
                    sparse_linear_pytorch_compiled(a, weight, cur_node, csr, step=0)

            record = {"B": B, "N": N}

            # --- benchmark ---
            with torch.no_grad():
                if "fused" in alg_set:
                    record["ms_fused"] = run_bench(
                        lambda: fused_linear_constrained_node_transition(a, weight.T, cs)
                    )
                if "kernel" in alg_set:
                    record["ms_kernel"] = run_bench(
                        lambda: constrained_node_transition(a @ weight.T, cs)
                    )
                if "pytorch" in alg_set:
                    record["ms_pytorch"] = run_bench(
                        lambda: vtnk_pytorch(linear(a), cur_node, csr, step=0)
                    )
                if "sparse_pytorch" in alg_set:
                    record["ms_sparse_pytorch"] = run_bench(
                        lambda: sparse_linear_pytorch_compiled(a, weight, cur_node, csr, step=0)
                    )
            if "trie_cpu" in alg_set:
                record["ms_trie_cpu"] = run_bench_cpu(
                    lambda: trie_cpu_traversal(cur_node_cpu, trie_nodes, N)
                )

            if "fused" in alg_set and "kernel" in alg_set:
                record["speedup_fused_vs_kernel"] = record["ms_kernel"] / record["ms_fused"]
            if "fused" in alg_set and "pytorch" in alg_set:
                record["speedup_fused_vs_pytorch"] = record["ms_pytorch"] / record["ms_fused"]
            if "fused" in alg_set and "sparse_pytorch" in alg_set:
                record["speedup_fused_vs_sparse_pytorch"] = record["ms_sparse_pytorch"] / record["ms_fused"]
            if "fused" in alg_set and "trie_cpu" in alg_set:
                record["speedup_fused_vs_trie_cpu"] = record["ms_trie_cpu"] / record["ms_fused"]

            records.append(record)

    return pd.DataFrame(records)


def plot_heatmap(
    df, value_col, title, filename, fmt=".2f", cbar_label="Speedup vs fused"
):
    pivot = df.pivot(index="B", columns="N", values=value_col)
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        pivot.sort_index(),
        annot=True,
        fmt=fmt,
        cmap="viridis",
        cbar_kws={"label": cbar_label},
    )
    plt.title(title)
    plt.ylabel("Batch size (B)")
    plt.xlabel("Vocab size (N)")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, format="jpg")
    plt.close()
    print(f"  Saved {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark constrained node transition algorithms."
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=ALL_ALGORITHMS,
        default=DEFAULT_ALGORITHMS,
        metavar="ALGO",
        help=f"Algorithms to benchmark. Choices: {ALL_ALGORITHMS} (default: {DEFAULT_ALGORITHMS})",
    )
    parser.add_argument(
        "--sparsity",
        type=float,
        default=DEFAULT_SPARSITY,
        help="Fraction of vocab used as max branches (default: %(default)s)",
    )
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    os.makedirs("out", exist_ok=True)

    B_vals = [32, 256, 1024]
    N_vals = [512, 1024, 8192, 150000]

    print(f"Benchmarking K={K}, sparsity={args.sparsity}")
    print(f"Algorithms: {args.algorithms}")
    print(f"B_vals={B_vals}")
    print(f"N_vals={N_vals}\n")

    df = benchmark_grid(B_vals, N_vals, algorithms=args.algorithms, sparsity=args.sparsity)
    csv_path = "out/bench_vtnk.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}\n")

    print(df.to_string(index=False))

    if "speedup_fused_vs_kernel" in df.columns:
        plot_heatmap(
            df,
            value_col="speedup_fused_vs_kernel",
            title=f"Fused speedup vs compiled_linear+constrained_kernel  (K={K})",
            filename="out/heatmap_fused_vs_kernel.jpg",
            cbar_label="Speedup (>1 = fused faster)",
        )
    if "speedup_fused_vs_pytorch" in df.columns:
        plot_heatmap(
            df,
            value_col="speedup_fused_vs_pytorch",
            title=f"Fused speedup vs compiled_linear+vtnk_pytorch  (K={K})",
            filename="out/heatmap_fused_vs_pytorch.jpg",
            cbar_label="Speedup (>1 = fused faster)",
        )
    if "speedup_fused_vs_sparse_pytorch" in df.columns:
        plot_heatmap(
            df,
            value_col="speedup_fused_vs_sparse_pytorch",
            title=f"Fused speedup vs sparse_linear_pytorch  (K={K})",
            filename="out/heatmap_fused_vs_sparse_pytorch.jpg",
            cbar_label="Speedup (>1 = fused faster)",
        )
    if "speedup_fused_vs_trie_cpu" in df.columns:
        plot_heatmap(
            df,
            value_col="speedup_fused_vs_trie_cpu",
            title=f"Fused speedup vs CPU trie traversal  (K={K})",
            filename="out/heatmap_fused_vs_trie_cpu.jpg",
            cbar_label="Speedup (>1 = fused faster)",
        )
