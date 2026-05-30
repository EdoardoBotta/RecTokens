"""
Benchmark: fused_linear_constrained_node_transition
    vs torch.compile(nn.Linear) + constrained_node_transition (Triton kernel)
    vs torch.compile(nn.Linear) + vtnk_pytorch
    vs torch.compile(sparse_linear_pytorch)

Grid: B (batch size) × N (vocab / logits size). K (hidden dim) fixed per run.
"""

import argparse
import os
import time
from typing import Any, cast

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
from benchmark_config import VTNK_SUITE, BenchmarkSuite, RunConfig, trie_runs

DEVICE = torch.device("cuda")

ALL_ALGORITHMS = ["fused", "kernel", "pytorch", "sparse_pytorch", "dense_lookup", "dense_topk", "trie_cpu"]
DEFAULT_ALGORITHMS = ["fused", "kernel", "pytorch", "sparse_pytorch", "dense_lookup", "dense_topk"]

torch.set_float32_matmul_precision("high")


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


def make_csr_diverse(
    vocab_size: int, max_branches: int, B: int
) -> tuple[object, torch.Tensor]:
    """2-level trie: root → num_nodes children, each → max_branches children.

    Returns (csr, cur_node) where cur_node spreads batch items across level-1 nodes
    so different batch elements traverse different parts of the trie.
    Use with step=1 in ConstraintState so layer_max_branches[1] == max_branches.
    """
    num_nodes = min(B, 512)
    seqs = lex_sort([[i, j] for i in range(num_nodes) for j in range(max_branches)])
    csr = CompactCSRTrie.from_sorted_batch(seqs, vocab_size=vocab_size)
    csr = csr._replace(
        row_ptrs=csr.row_ptrs.to(DEVICE),
        stacked_cols_vals=csr.stacked_cols_vals.to(DEVICE),
        dense_mask_by_layer=[v.to(DEVICE) for v in csr.dense_mask_by_layer],
        dense_states=csr.dense_states.to(DEVICE),
    )
    cur_node = (torch.arange(B, dtype=torch.long) % num_nodes + 1).to(DEVICE)
    return csr, cur_node


def make_trie(max_branches: int) -> tuple[Trie, list[TrieNode]]:
    """Build a Trie and return a BFS-ordered node list for integer indexing."""
    trie = Trie()
    for i in range(max_branches):
        trie.insert([i])
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


def _dense_lookup_inner(a: torch.Tensor, weight: torch.Tensor, mask1d: torch.Tensor) -> torch.Tensor:
    """Full linear + dense boolean mask — the hot path wrapped by torch.compile."""
    logits = (a @ weight.T).float()
    return logits.masked_fill(~mask1d.unsqueeze(0), float("-inf"))


_dense_lookup_compiled = torch.compile(_dense_lookup_inner)


def _dense_matmul_inner(a: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (a @ weight.T).float()


_dense_matmul_compiled = torch.compile(_dense_matmul_inner)


def dense_lookup_pytorch(
    a: torch.Tensor, weight: torch.Tensor, cur_node: torch.Tensor, csr, step: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full dense linear + dense mask lookup.

    Only supports step=0 (all batch items at the root node); incompatible with
    --diverse-nodes because step=1 requires the previous-token path to index the
    2-D dense_mask_by_layer, which is not stored in cur_node.
    """
    device = a.device
    B = a.shape[0]

    mask1d: torch.Tensor = csr.dense_mask_by_layer[step]
    corrected_logits = _dense_lookup_compiled(a, weight, mask1d)

    max_branches = csr.layer_max_branches[step]
    valid_toks = mask1d.nonzero(as_tuple=True)[0]
    n_valid = valid_toks.shape[0]

    padded_valid = valid_toks.new_full((max_branches,), -1)
    padded_valid[:n_valid] = valid_toks

    padded_next = valid_toks.new_full((max_branches,), -1)
    padded_next[:n_valid] = csr.dense_states[valid_toks]

    valid_idxs = padded_valid.unsqueeze(0).expand(B, -1)
    next_node = padded_next.unsqueeze(0).expand(B, -1)

    return next_node, valid_idxs, corrected_logits


def run_bench(fn, suite: BenchmarkSuite) -> float:
    return cast(float, testing.do_bench(fn, warmup=suite.warmup, rep=suite.rep))


def run_bench_cpu(fn, suite: BenchmarkSuite) -> float:
    """Wall-clock benchmark for CPU-only functions (ms)."""
    for _ in range(suite.warmup):
        fn()
    times = []
    for _ in range(suite.rep):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    return float(sum(times[: max(1, suite.rep // 2)]) / max(1, suite.rep // 2))


def benchmark_run(run: RunConfig, suite: BenchmarkSuite, algorithms: list[str], max_N: int, max_B: int) -> dict[str, Any]:
    alg_set = set(algorithms)
    gpu_algos = alg_set & {"fused", "kernel", "pytorch", "sparse_pytorch", "dense_lookup", "dense_topk"}
    max_branches = max(1, int(run.N * run.sparsity))
    print(f"  B={run.B:6d}  N={run.N:6d}  max_branches={max_branches}")

    # trie_cpu is too slow at max B/N and incompatible with diverse_nodes.
    # dense_lookup requires step=0; diverse_nodes uses step=1.
    skip_cpu = run.B == max_B or run.N == max_N or suite.diverse_nodes
    skip_algos = {"trie_cpu"} if skip_cpu else set()
    if suite.diverse_nodes:
        skip_algos.add("dense_lookup")
    active_alg_set = alg_set - skip_algos

    if gpu_algos:
        if suite.diverse_nodes:
            csr, cur_node = make_csr_diverse(
                vocab_size=run.N, max_branches=max_branches, B=run.B
            )
            step = 1
        else:
            csr = make_csr(vocab_size=run.N, max_branches=max_branches)
            cur_node = torch.zeros(run.B, dtype=torch.long, device=DEVICE)
            step = 0
    if "trie_cpu" in active_alg_set:
        _, trie_nodes = make_trie(max_branches=max_branches)

    a = torch.randn(run.B, run.k, device=DEVICE)
    weight = torch.randn(run.N, run.k, device=DEVICE)
    if not gpu_algos:
        cur_node = torch.zeros(run.B, dtype=torch.long, device=DEVICE)
        step = 0
    if "trie_cpu" in active_alg_set:
        cur_node_cpu = cur_node.cpu()

    if "pytorch" in active_alg_set:
        linear = torch.compile(nn.Linear(run.k, run.N, bias=False).to(DEVICE))
        with torch.no_grad():
            linear.weight.data.copy_(weight)

    if "sparse_pytorch" in active_alg_set:
        sparse_linear_pytorch_compiled = torch.compile(sparse_linear_pytorch)

    if gpu_algos:
        cs = ConstraintState(step=step, trie=csr, cur_node=cur_node)

    # --- warmup / force compilation ---
    with torch.no_grad():
        if "fused" in active_alg_set:
            fused_linear_constrained_node_transition(a, weight.T, cs)
        if "kernel" in active_alg_set:
            constrained_node_transition(a @ weight.T, cs)
        if "pytorch" in active_alg_set:
            vtnk_pytorch(linear(a), cur_node, csr, step=step)
        if "sparse_pytorch" in active_alg_set:
            sparse_linear_pytorch_compiled(a, weight, cur_node, csr, step=step)
        if "dense_lookup" in active_alg_set:
            dense_lookup_pytorch(a, weight, cur_node, csr, step=step)
        if "dense_topk" in active_alg_set:
            _, _, bl = constrained_node_transition(_dense_matmul_compiled(a, weight), cs)
            torch.topk(bl, run.k_top, dim=-1)

    record: dict[str, Any] = {"B": run.B, "N": run.N}

    # --- benchmark ---
    with torch.no_grad():
        if "fused" in active_alg_set:
            record["ms_fused"] = run_bench(
                lambda: fused_linear_constrained_node_transition(a, weight.T, cs),
                suite,
            )
        if "kernel" in active_alg_set:
            record["ms_kernel"] = run_bench(
                lambda: constrained_node_transition(a @ weight.T, cs),
                suite,
            )
        if "pytorch" in active_alg_set:
            record["ms_pytorch"] = run_bench(
                lambda: vtnk_pytorch(linear(a), cur_node, csr, step=step),
                suite,
            )
        if "sparse_pytorch" in active_alg_set:
            record["ms_sparse_pytorch"] = run_bench(
                lambda: sparse_linear_pytorch_compiled(a, weight, cur_node, csr, step=step),
                suite,
            )
        if "dense_lookup" in active_alg_set:
            record["ms_dense_lookup"] = run_bench(
                lambda: dense_lookup_pytorch(a, weight, cur_node, csr, step=step),
                suite,
            )
        if "dense_topk" in active_alg_set:
            record["ms_dense_topk"] = run_bench(
                lambda: torch.topk(
                    constrained_node_transition(_dense_matmul_compiled(a, weight), cs)[2], run.k_top, dim=-1
                ),
                suite,
            )
    if "trie_cpu" in active_alg_set:
        record["ms_trie_cpu"] = run_bench_cpu(
            lambda: trie_cpu_traversal(cur_node_cpu, trie_nodes, run.N),
            suite,
        )

    if "fused" in active_alg_set and "kernel" in active_alg_set:
        record["speedup_fused_vs_kernel"] = record["ms_kernel"] / record["ms_fused"]
    if "fused" in active_alg_set and "pytorch" in active_alg_set:
        record["speedup_fused_vs_pytorch"] = record["ms_pytorch"] / record["ms_fused"]
    if "fused" in active_alg_set and "sparse_pytorch" in active_alg_set:
        record["speedup_fused_vs_sparse_pytorch"] = record["ms_sparse_pytorch"] / record["ms_fused"]
    if "fused" in active_alg_set and "trie_cpu" in active_alg_set:
        record["speedup_fused_vs_trie_cpu"] = record["ms_trie_cpu"] / record["ms_fused"]
    if "fused" in active_alg_set and "dense_lookup" in active_alg_set:
        record["speedup_fused_vs_dense_lookup"] = record["ms_dense_lookup"] / record["ms_fused"]
    if "fused" in active_alg_set and "dense_topk" in active_alg_set:
        record["speedup_fused_vs_dense_topk"] = record["ms_dense_topk"] / record["ms_fused"]

    return record


def benchmark_grid(suite: BenchmarkSuite, algorithms: list[str]) -> pd.DataFrame:
    max_B = max(r.B for r in suite.runs)
    max_N = max(r.N for r in suite.runs)
    return pd.DataFrame([
        benchmark_run(run, suite, algorithms, max_N=max_N, max_B=max_B)
        for run in suite.runs
    ])


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
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    os.makedirs("out", exist_ok=True)

    suite = VTNK_SUITE

    print(f"Algorithms: {args.algorithms}")
    print(f"diverse_nodes={suite.diverse_nodes}")
    for run in suite.runs:
        print(f"  B={run.B}, N={run.N}, k={run.k}, sparsity={run.sparsity}")
    print()

    df = benchmark_grid(suite, algorithms=args.algorithms)
    csv_path = "out/bench_vtnk.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}\n")

    print(df.to_string(index=False))

    if "speedup_fused_vs_kernel" in df.columns:
        plot_heatmap(
            df,
            value_col="speedup_fused_vs_kernel",
            title=f"Fused speedup vs compiled_linear+constrained_kernel  (K={suite.runs[0].k})",
            filename="out/heatmap_fused_vs_kernel.jpg",
            cbar_label="Speedup (>1 = fused faster)",
        )
    if "speedup_fused_vs_pytorch" in df.columns:
        plot_heatmap(
            df,
            value_col="speedup_fused_vs_pytorch",
            title=f"Fused speedup vs compiled_linear+vtnk_pytorch  (K={suite.runs[0].k})",
            filename="out/heatmap_fused_vs_pytorch.jpg",
            cbar_label="Speedup (>1 = fused faster)",
        )
    if "speedup_fused_vs_sparse_pytorch" in df.columns:
        plot_heatmap(
            df,
            value_col="speedup_fused_vs_sparse_pytorch",
            title=f"Fused speedup vs sparse_linear_pytorch  (K={suite.runs[0].k})",
            filename="out/heatmap_fused_vs_sparse_pytorch.jpg",
            cbar_label="Speedup (>1 = fused faster)",
        )
    if "speedup_fused_vs_trie_cpu" in df.columns:
        plot_heatmap(
            df,
            value_col="speedup_fused_vs_trie_cpu",
            title=f"Fused speedup vs CPU trie traversal  (K={suite.runs[0].k})",
            filename="out/heatmap_fused_vs_trie_cpu.jpg",
            cbar_label="Speedup (>1 = fused faster)",
        )
    if "speedup_fused_vs_dense_lookup" in df.columns:
        plot_heatmap(
            df,
            value_col="speedup_fused_vs_dense_lookup",
            title=f"Fused speedup vs dense_lookup  (K={suite.runs[0].k})",
            filename="out/heatmap_fused_vs_dense_lookup.jpg",
            cbar_label="Speedup (>1 = fused faster)",
        )
    if "speedup_fused_vs_dense_topk" in df.columns:
        plot_heatmap(
            df,
            value_col="speedup_fused_vs_dense_topk",
            title=f"Fused speedup vs compile(dense_matmul)+topk  (K={suite.runs[0].k}, k_top={suite.runs[0].k_top})",
            filename="out/heatmap_fused_vs_dense_topk.jpg",
            cbar_label="Speedup (>1 = fused faster)",
        )
