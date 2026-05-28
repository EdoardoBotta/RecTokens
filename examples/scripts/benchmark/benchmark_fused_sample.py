"""
Benchmark: fused_linear_constrained_node_transition_sampling (Triton, single kernel)
    vs torch.compile(sparse_linear_pytorch) + separate torch.multinomial sampling

    fused_linear_constrained_node_transition_topk (Triton, single kernel)
    vs torch.compile(sparse_linear_pytorch) + separate torch.topk

Grid: B (batch size) × N (vocab / logits size). K (hidden dim) fixed per run.
"""

import argparse
import os
from typing import Any, cast

os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

import torch
import torch.nn.functional as F
import triton.testing as testing
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from rectokens.schemas.compact_csr_trie import CompactCSRTrie
from rectokens.schemas.compact_ell_trie import CompactELLTrie
from rectokens.schemas.state import ConstraintState
from rectokens.decoding.vntk import (
    sparse_linear_pytorch,
    sparse_linear_compact_pytorch,
    sparse_linear_ell_pytorch,
    sparse_linear_compact_ell_pytorch,
)
from rectokens.ops.constrained_node_transition import (
    constrained_node_transition,
    fused_linear_constrained_node_transition_sampling,
    fused_linear_constrained_node_transition_topk,
)
from rectokens.kernels.constrained_node_transition_ell import (
    _ell_fused_linear_constrained_node_transition_sampling_op as ell_sampling_op,
    _ell_fused_linear_constrained_node_transition_topk_op as ell_topk_op,
)
from benchmark_config import FUSED_SAMPLE_SUITE_N256, FUSED_SAMPLE_SUITE_N150K, BenchmarkSuite, RunConfig

DEVICE = torch.device("cuda")

ALL_ALGORITHMS = [
    "fused_sample",
    "ell_sample",
    "sparse_pytorch_sample",
    "sparse_pytorch_sample_compact",
    "sparse_pytorch_ell_sample",
    "fused_topk",
    "ell_topk",
    "sparse_pytorch_topk",
    "sparse_pytorch_topk_compact",
    "sparse_pytorch_ell_topk_compact",
    "dense_topk",
]
DEFAULT_ALGORITHMS = [
    "fused_topk",
    #"ell_topk",
    "sparse_pytorch_topk_compact",
    #"sparse_pytorch_ell_topk_compact",
    "fused_sample",
    "sparse_pytorch_sample_compact",
    "dense_topk",
]


def lex_sort(rows: list[list[int]]) -> torch.Tensor:
    return torch.tensor(sorted(rows), dtype=torch.long)


def make_csr(vocab_size: int, max_branches: int) -> CompactCSRTrie:
    seqs = [[i] for i in range(max_branches)]
    csr = CompactCSRTrie.from_sorted_batch(
        lex_sort(seqs), vocab_size=vocab_size, dense_lookup_layers=0
    )
    return csr._replace(
        row_ptrs=csr.row_ptrs.to(DEVICE),
        stacked_cols_vals=csr.stacked_cols_vals.to(DEVICE),
        dense_mask_by_layer=[v.to(DEVICE) for v in csr.dense_mask_by_layer],
        dense_states=csr.dense_states.to(DEVICE),
    )


def make_csr_diverse(
    vocab_size: int, max_branches: int, B: int
) -> tuple[CompactCSRTrie, torch.Tensor]:
    """2-level trie: root → num_nodes children, each → max_branches children.

    Returns (csr, cur_node) where cur_node spreads batch items across level-1 nodes
    so different batch elements traverse different parts of the trie.
    Use with step=1 in ConstraintState so layer_max_branches[1] == max_branches.
    """
    num_nodes = min(B, 512)
    seqs = lex_sort([[i, j] for i in range(num_nodes) for j in range(max_branches)])
    csr = CompactCSRTrie.from_sorted_batch(
        seqs, vocab_size=vocab_size, dense_lookup_layers=0
    )
    csr = csr._replace(
        row_ptrs=csr.row_ptrs.to(DEVICE),
        stacked_cols_vals=csr.stacked_cols_vals.to(DEVICE),
        dense_mask_by_layer=[v.to(DEVICE) for v in csr.dense_mask_by_layer],
        dense_states=csr.dense_states.to(DEVICE),
    )
    cur_node = (torch.arange(B, dtype=torch.long) % num_nodes + 1).to(DEVICE)
    return csr, cur_node


def make_ell(csr: CompactCSRTrie) -> CompactELLTrie:
    return CompactELLTrie.from_csr(csr)


def _dense_matmul_inner(a: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (a @ weight.T).float()


_dense_matmul_compiled = torch.compile(_dense_matmul_inner)


def run_bench(fn, suite: BenchmarkSuite) -> float:
    return cast(float, testing.do_bench(fn, warmup=suite.warmup, rep=suite.rep))


def benchmark_run(run: RunConfig, suite: BenchmarkSuite, algorithms: list[str]) -> dict:
    alg_set = set(algorithms)
    max_branches = max(1, int(run.N * run.sparsity))
    k = min(run.k_top, max_branches)
    print(f"  B={run.B:6d}  N={run.N:6d}  max_branches={max_branches}  k={k}")

    if suite.diverse_nodes:
        csr, cur_node = make_csr_diverse(
            vocab_size=run.N, max_branches=max_branches, B=run.B
        )
        step = 1
    else:
        csr = make_csr(vocab_size=run.N, max_branches=max_branches)
        cur_node = torch.zeros(run.B, dtype=torch.long, device=DEVICE)
        step = 0

    a = torch.randn(run.B, run.k, device=DEVICE, dtype=torch.bfloat16)
    weight = torch.randn(run.N, run.k, device=DEVICE, dtype=torch.bfloat16)
    cs = ConstraintState(step=step, trie=csr, cur_node=cur_node)

    needs_ell = alg_set & {"ell_sample", "ell_topk"}
    if needs_ell:
        ell = make_ell(csr)
        _bias = a.new_empty(0)

    needs_sparse_full = alg_set & {"sparse_pytorch_sample", "sparse_pytorch_topk"}
    needs_sparse_compact = "sparse_pytorch_topk_compact" in alg_set
    needs_sparse_compact_sample = "sparse_pytorch_sample_compact" in alg_set
    needs_sparse_ell_full = "sparse_pytorch_ell_sample" in alg_set
    needs_sparse_ell_compact = "sparse_pytorch_ell_topk_compact" in alg_set
    if needs_sparse_full:
        sparse_linear_pytorch_compiled = torch.compile(sparse_linear_pytorch)
    if needs_sparse_compact:

        def _sparse_compact_topk(a, weight, cur_node, csr, step, k):
            nn, vi, branch_logits = sparse_linear_compact_pytorch(
                a, weight, cur_node, csr, step
            )
            topk_logits, topk_branch_idxs = torch.topk(branch_logits, k, dim=-1)
            topk_idxs = vi.gather(1, topk_branch_idxs)
            return nn, vi, topk_logits, topk_idxs

        sparse_compact_topk_compiled = torch.compile(_sparse_compact_topk)

    if needs_sparse_ell_full or needs_sparse_ell_compact:
        if not needs_ell:
            ell = make_ell(csr)
    if needs_sparse_ell_full:
        sparse_linear_ell_pytorch_compiled = torch.compile(sparse_linear_ell_pytorch)
    if needs_sparse_ell_compact:

        def _sparse_ell_compact_topk(a, weight, cur_node, ell_trie, step, k):
            nn, vi, branch_logits = sparse_linear_compact_ell_pytorch(
                a, weight, cur_node, ell_trie, step
            )
            topk_logits, topk_branch_idxs = torch.topk(branch_logits, k, dim=-1)
            topk_idxs = vi.gather(1, topk_branch_idxs)
            return nn, vi, topk_logits, topk_idxs

        sparse_ell_compact_topk_compiled = torch.compile(_sparse_ell_compact_topk)

    if "sparse_pytorch_sample" in alg_set:

        def sparse_pytorch_with_sample():
            _, _, corrected_logits = sparse_linear_pytorch_compiled(
                a, weight, cur_node, csr, step=step
            )
            probs = F.softmax(corrected_logits, dim=-1)
            return torch.multinomial(probs, num_samples=1).squeeze(-1)

    if needs_sparse_compact_sample:

        def _sparse_compact_sample(a, weight, cur_node, csr, step):
            nn, vi, branch_logits = sparse_linear_compact_pytorch(
                a, weight, cur_node, csr, step
            )
            probs = F.softmax(branch_logits, dim=-1)
            branch_sample = torch.multinomial(probs, num_samples=1)
            return nn, vi.gather(1, branch_sample).squeeze(-1)

        sparse_compact_sample_compiled = torch.compile(_sparse_compact_sample)

    if "sparse_pytorch_ell_sample" in alg_set:

        def sparse_pytorch_ell_with_sample():
            _, _, corrected_logits = sparse_linear_ell_pytorch_compiled(
                a, weight, cur_node, ell, step=step
            )
            probs = F.softmax(corrected_logits, dim=-1)
            return torch.multinomial(probs, num_samples=1).squeeze(-1)

    if "sparse_pytorch_topk" in alg_set:

        def sparse_pytorch_with_topk():
            _, _, corrected_logits = sparse_linear_pytorch_compiled(
                a, weight, cur_node, csr, step=step
            )
            return torch.topk(corrected_logits, k, dim=-1)

    if needs_sparse_compact:

        def sparse_pytorch_compact_with_topk():
            return sparse_compact_topk_compiled(a, weight, cur_node, csr, step, k)

    if needs_sparse_ell_compact:

        def sparse_pytorch_ell_compact_with_topk():
            return sparse_ell_compact_topk_compiled(a, weight, cur_node, ell, step, k)

    # --- warmup / force compilation ---
    with torch.no_grad():
        if "fused_sample" in alg_set:
            fused_linear_constrained_node_transition_sampling(a, weight.T, cs)
        if "ell_sample" in alg_set:
            ell_sampling_op(a, weight.T, _bias, cur_node, ell.ell_cols_vals, ell.n_children, max_branches, False)
        if "sparse_pytorch_sample" in alg_set:
            sparse_pytorch_with_sample()
        if needs_sparse_compact_sample:
            sparse_compact_sample_compiled(a, weight, cur_node, csr, step)
        if "sparse_pytorch_ell_sample" in alg_set:
            sparse_pytorch_ell_with_sample()
        if "fused_topk" in alg_set:
            fused_linear_constrained_node_transition_topk(a, weight.T, cs, k=k)
        if "ell_topk" in alg_set:
            ell_topk_op(a, weight.T, _bias, cur_node, ell.ell_cols_vals, ell.n_children, max_branches, False, k)
        if "sparse_pytorch_topk" in alg_set:
            sparse_pytorch_with_topk()
        if "sparse_pytorch_topk_compact" in alg_set:
            sparse_pytorch_compact_with_topk()
        if "sparse_pytorch_ell_topk_compact" in alg_set:
            sparse_pytorch_ell_compact_with_topk()
        if "dense_topk" in alg_set:
            _, _, bl = constrained_node_transition(_dense_matmul_compiled(a, weight), cs)
            torch.topk(bl, k, dim=-1)

    record: dict[str, Any] = {"B": run.B, "N": run.N}

    # --- benchmark ---
    with torch.no_grad():
        if "fused_sample" in alg_set:
            record["ms_fused_sample"] = run_bench(
                lambda: fused_linear_constrained_node_transition_sampling(a, weight.T, cs),
                suite,
            )
        if "ell_sample" in alg_set:
            record["ms_ell_sample"] = run_bench(
                lambda: ell_sampling_op(
                    a, weight.T, _bias, cur_node,
                    ell.ell_cols_vals, ell.n_children, max_branches, False,
                ),
                suite,
            )
        if "sparse_pytorch_sample" in alg_set:
            record["ms_sparse_pytorch_sample"] = run_bench(sparse_pytorch_with_sample, suite)
        if needs_sparse_compact_sample:
            record["ms_sparse_pytorch_sample_compact"] = run_bench(
                lambda: sparse_compact_sample_compiled(a, weight, cur_node, csr, step),
                suite,
            )
        if "sparse_pytorch_ell_sample" in alg_set:
            record["ms_sparse_pytorch_ell_sample"] = run_bench(sparse_pytorch_ell_with_sample, suite)
        if "fused_topk" in alg_set:
            record["ms_fused_topk"] = run_bench(
                lambda: fused_linear_constrained_node_transition_topk(a, weight.T, cs, k=k),
                suite,
            )
        if "ell_topk" in alg_set:
            record["ms_ell_topk"] = run_bench(
                lambda: ell_topk_op(
                    a, weight.T, _bias, cur_node,
                    ell.ell_cols_vals, ell.n_children, max_branches, False, k,
                ),
                suite,
            )
        if "sparse_pytorch_topk" in alg_set:
            record["ms_sparse_pytorch_topk"] = run_bench(sparse_pytorch_with_topk, suite)
        if "sparse_pytorch_topk_compact" in alg_set:
            record["ms_sparse_pytorch_topk_compact"] = run_bench(sparse_pytorch_compact_with_topk, suite)
        if "sparse_pytorch_ell_topk_compact" in alg_set:
            record["ms_sparse_pytorch_ell_topk_compact"] = run_bench(sparse_pytorch_ell_compact_with_topk, suite)
        if "dense_topk" in alg_set:
            record["ms_dense_topk"] = run_bench(
                lambda: torch.topk(
                    constrained_node_transition(_dense_matmul_compiled(a, weight), cs)[2], k, dim=-1
                ),
                suite,
            )

    if "fused_sample" in alg_set and "sparse_pytorch_sample" in alg_set:
        record["speedup_fused_vs_sparse_pytorch_sample"] = (
            record["ms_sparse_pytorch_sample"] / record["ms_fused_sample"]
        )
    if "fused_sample" in alg_set and "sparse_pytorch_sample_compact" in alg_set:
        record["speedup_fused_vs_sparse_pytorch_sample_compact"] = (
            record["ms_sparse_pytorch_sample_compact"] / record["ms_fused_sample"]
        )
    if "ell_sample" in alg_set and "fused_sample" in alg_set:
        record["speedup_ell_vs_csr_sample"] = (
            record["ms_fused_sample"] / record["ms_ell_sample"]
        )
    if "ell_sample" in alg_set and "sparse_pytorch_sample" in alg_set:
        record["speedup_ell_vs_sparse_pytorch_sample"] = (
            record["ms_sparse_pytorch_sample"] / record["ms_ell_sample"]
        )
    if "fused_topk" in alg_set and "sparse_pytorch_topk" in alg_set:
        record["speedup_fused_topk_vs_sparse_pytorch_topk"] = (
            record["ms_sparse_pytorch_topk"] / record["ms_fused_topk"]
        )
    if "ell_topk" in alg_set and "fused_topk" in alg_set:
        record["speedup_ell_vs_csr_topk"] = (
            record["ms_fused_topk"] / record["ms_ell_topk"]
        )
    if "ell_topk" in alg_set and "sparse_pytorch_topk" in alg_set:
        record["speedup_ell_vs_sparse_pytorch_topk"] = (
            record["ms_sparse_pytorch_topk"] / record["ms_ell_topk"]
        )
    if "fused_topk" in alg_set and "sparse_pytorch_topk_compact" in alg_set:
        record["speedup_fused_topk_vs_sparse_pytorch_topk_compact"] = (
            record["ms_sparse_pytorch_topk_compact"] / record["ms_fused_topk"]
        )
    if "sparse_pytorch_topk" in alg_set and "sparse_pytorch_topk_compact" in alg_set:
        record["speedup_compact_vs_full_pytorch_topk"] = (
            record["ms_sparse_pytorch_topk"] / record["ms_sparse_pytorch_topk_compact"]
        )
    if "sparse_pytorch_ell_sample" in alg_set and "sparse_pytorch_sample" in alg_set:
        record["speedup_ell_pytorch_vs_csr_pytorch_sample"] = (
            record["ms_sparse_pytorch_sample"] / record["ms_sparse_pytorch_ell_sample"]
        )
    if "sparse_pytorch_ell_sample" in alg_set and "ell_sample" in alg_set:
        record["speedup_ell_triton_vs_ell_pytorch_sample"] = (
            record["ms_sparse_pytorch_ell_sample"] / record["ms_ell_sample"]
        )
    if "sparse_pytorch_ell_topk_compact" in alg_set and "sparse_pytorch_topk_compact" in alg_set:
        record["speedup_ell_pytorch_vs_csr_pytorch_topk_compact"] = (
            record["ms_sparse_pytorch_topk_compact"] / record["ms_sparse_pytorch_ell_topk_compact"]
        )
    if "sparse_pytorch_ell_topk_compact" in alg_set and "ell_topk" in alg_set:
        record["speedup_ell_triton_vs_ell_pytorch_topk_compact"] = (
            record["ms_sparse_pytorch_ell_topk_compact"] / record["ms_ell_topk"]
        )
    if "fused_topk" in alg_set and "dense_topk" in alg_set:
        record["speedup_fused_topk_vs_dense_topk"] = (
            record["ms_dense_topk"] / record["ms_fused_topk"]
        )
    return record


def benchmark_grid(suite: BenchmarkSuite, algorithms: list[str]) -> pd.DataFrame:
    return pd.DataFrame([benchmark_run(run, suite, algorithms) for run in suite.runs])


def plot_heatmap(df, value_col, title, filename, fmt=".2f", cbar_label="Speedup"):
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
        description="Benchmark fused sampling and top-k vs sparse pytorch baselines."
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

    suites = [FUSED_SAMPLE_SUITE_N256, FUSED_SAMPLE_SUITE_N150K]

    print(f"Algorithms: {args.algorithms}")
    for suite in suites:
        print(f"  diverse_nodes={suite.diverse_nodes}")
        for run in suite.runs:
            print(f"    B={run.B}, N={run.N}, k={run.k}, k_top={run.k_top}, sparsity={run.sparsity}")
    print()

    df = pd.concat([benchmark_grid(suite, algorithms=args.algorithms) for suite in suites], ignore_index=True)
    csv_path = "out/bench_fused_sample.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}\n")

    print(df.to_string(index=False))

    if "speedup_ell_vs_csr_sample" in df.columns:
        plot_heatmap(
            df,
            value_col="speedup_ell_vs_csr_sample",
            title="ELL sample speedup vs CSR sample",
            filename="out/heatmap_ell_vs_csr_sample.jpg",
            cbar_label="Speedup (>1 = ELL faster)",
        )
    if "speedup_ell_vs_sparse_pytorch_sample" in df.columns:
        plot_heatmap(
            df,
            value_col="speedup_ell_vs_sparse_pytorch_sample",
            title="ELL sample speedup vs compile(sparse_linear_pytorch)+multinomial",
            filename="out/heatmap_ell_sample_vs_sparse_pytorch.jpg",
            cbar_label="Speedup (>1 = ELL faster)",
        )
    if "speedup_ell_vs_csr_topk" in df.columns:
        plot_heatmap(
            df,
            value_col="speedup_ell_vs_csr_topk",
            title="ELL top-k speedup vs CSR top-k",
            filename="out/heatmap_ell_vs_csr_topk.jpg",
            cbar_label="Speedup (>1 = ELL faster)",
        )
    if "speedup_ell_vs_sparse_pytorch_topk" in df.columns:
        plot_heatmap(
            df,
            value_col="speedup_ell_vs_sparse_pytorch_topk",
            title="ELL top-k speedup vs compile(sparse_linear_pytorch)+topk",
            filename="out/heatmap_ell_topk_vs_sparse_pytorch.jpg",
            cbar_label="Speedup (>1 = ELL faster)",
        )
    if "speedup_fused_vs_sparse_pytorch_sample" in df.columns:
        plot_heatmap(
            df,
            value_col="speedup_fused_vs_sparse_pytorch_sample",
            title="Fused sample speedup vs compile(sparse_linear_pytorch)+multinomial",
            filename="out/heatmap_fused_sample_vs_sparse_pytorch.jpg",
            cbar_label="Speedup (>1 = fused faster)",
        )
    if "speedup_fused_topk_vs_sparse_pytorch_topk" in df.columns:
        plot_heatmap(
            df,
            value_col="speedup_fused_topk_vs_sparse_pytorch_topk",
            title="Fused top-k speedup vs compile(sparse_linear_pytorch)+topk",
            filename="out/heatmap_fused_topk_vs_sparse_pytorch_topk.jpg",
            cbar_label="Speedup (>1 = fused faster)",
        )
    if "speedup_fused_topk_vs_sparse_pytorch_topk_compact" in df.columns:
        plot_heatmap(
            df,
            value_col="speedup_fused_topk_vs_sparse_pytorch_topk_compact",
            title="Fused top-k speedup vs compile(sparse_linear_compact_pytorch)+topk",
            filename="out/heatmap_fused_topk_vs_sparse_pytorch_topk_compact.jpg",
            cbar_label="Speedup (>1 = fused faster)",
        )
    if "speedup_compact_vs_full_pytorch_topk" in df.columns:
        plot_heatmap(
            df,
            value_col="speedup_compact_vs_full_pytorch_topk",
            title="Compact pytorch top-k speedup vs full (B,N) pytorch top-k",
            filename="out/heatmap_compact_vs_full_pytorch_topk.jpg",
            cbar_label="Speedup (>1 = compact faster)",
        )
    if "speedup_fused_topk_vs_dense_topk" in df.columns:
        plot_heatmap(
            df,
            value_col="speedup_fused_topk_vs_dense_topk",
            title="Fused top-k speedup vs compile(dense_matmul)+topk",
            filename="out/heatmap_fused_topk_vs_dense_topk.jpg",
            cbar_label="Speedup (>1 = fused faster)",
        )
