"""
Benchmark: fused_linear_constrained_node_transition_sampling (Triton, single kernel)
    vs torch.compile(sparse_linear_pytorch) + separate torch.multinomial sampling

    fused_linear_constrained_node_transition_topk (Triton, single kernel)
    vs torch.compile(sparse_linear_pytorch) + separate torch.topk

Grid: B (batch size) × N (vocab / logits size). K (hidden dim) fixed.
"""

import argparse
import os

os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

import torch
import torch.nn.functional as F
import triton.testing as testing
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from rectokens.schemas.compact_csr_trie import CompactCSRTrie
from rectokens.schemas.state import ConstraintState
from rectokens.decoding.vntk import sparse_linear_pytorch
from rectokens.ops.constrained_node_transition import (
    CUTE_DSL_AVAILABLE,
    fused_linear_constrained_node_transition_sampling,
    fused_linear_constrained_node_transition_topk,
    fused_linear_constrained_node_transition_topk_cute,
)

DEVICE = torch.device("cuda")
K = 512
K_TOP = 50
WARMUP = 25
REP = 100

ALL_ALGORITHMS = [
    "fused_sample",
    "sparse_pytorch_sample",
    "fused_topk",
    "sparse_pytorch_topk",
    "cute_topk",
]
DEFAULT_ALGORITHMS = [
    "fused_topk",
    "cute_topk",
]
DEFAULT_SPARSITY = 0.01


def lex_sort(rows: list[list[int]]) -> torch.Tensor:
    return torch.tensor(sorted(rows), dtype=torch.long)


def make_csr(vocab_size: int, max_branches: int) -> CompactCSRTrie:
    seqs = [[i] for i in range(max_branches)]
    csr = CompactCSRTrie.from_sorted_batch(lex_sort(seqs), vocab_size=vocab_size)
    return csr._replace(
        row_ptrs=csr.row_ptrs.to(DEVICE),
        stacked_cols_vals=csr.stacked_cols_vals.to(DEVICE),
        dense_mask_by_layer=[v.to(DEVICE) for v in csr.dense_mask_by_layer],
        dense_states=csr.dense_states.to(DEVICE),
    )


def run_bench(fn):
    return testing.do_bench(fn, warmup=WARMUP, rep=REP)


def benchmark_grid(B_vals, N_vals, algorithms, sparsity, k_top):
    alg_set = set(algorithms)
    records = []

    for B in B_vals:
        for N in N_vals:
            max_branches = max(1, int(N * sparsity))
            k = min(k_top, max_branches)
            print(f"  B={B:6d}  N={N:6d}  max_branches={max_branches}  k={k}")

            csr = make_csr(vocab_size=N, max_branches=max_branches)

            a = torch.randn(B, K, device=DEVICE)
            weight = torch.randn(N, K, device=DEVICE)
            cur_node = torch.zeros(B, dtype=torch.long, device=DEVICE)

            cs = ConstraintState(step=0, trie=csr, cur_node=cur_node)

            if "cute_topk" in alg_set and not CUTE_DSL_AVAILABLE:
                print("  [WARNING] cute_topk requested but nvidia-cutlass-dsl not installed — skipping")
                alg_set = alg_set - {"cute_topk"}

            needs_sparse = alg_set & {"sparse_pytorch_sample", "sparse_pytorch_topk"}
            if needs_sparse:
                sparse_linear_pytorch_compiled = torch.compile(sparse_linear_pytorch)

            if "sparse_pytorch_sample" in alg_set:

                def sparse_pytorch_with_sample():
                    _, _, corrected_logits = sparse_linear_pytorch_compiled(
                        a, weight, cur_node, csr, step=0
                    )
                    probs = F.softmax(corrected_logits, dim=-1)
                    return torch.multinomial(probs, num_samples=1).squeeze(-1)

            if "sparse_pytorch_topk" in alg_set:

                def sparse_pytorch_with_topk():
                    _, _, corrected_logits = sparse_linear_pytorch_compiled(
                        a, weight, cur_node, csr, step=0
                    )
                    return torch.topk(corrected_logits, k, dim=-1)

            # --- warmup / force compilation ---
            with torch.no_grad():
                if "fused_sample" in alg_set:
                    fused_linear_constrained_node_transition_sampling(a, weight.T, cs)
                if "sparse_pytorch_sample" in alg_set:
                    sparse_pytorch_with_sample()
                if "fused_topk" in alg_set:
                    fused_linear_constrained_node_transition_topk(a, weight.T, cs, k=k)
                if "sparse_pytorch_topk" in alg_set:
                    sparse_pytorch_with_topk()
                if "cute_topk" in alg_set:
                    fused_linear_constrained_node_transition_topk_cute(a, weight.T, cs, k=k)

            record = {"B": B, "N": N}

            # --- benchmark ---
            with torch.no_grad():
                if "fused_sample" in alg_set:
                    record["ms_fused_sample"] = run_bench(
                        lambda: fused_linear_constrained_node_transition_sampling(
                            a, weight.T, cs
                        )
                    )
                if "sparse_pytorch_sample" in alg_set:
                    record["ms_sparse_pytorch_sample"] = run_bench(
                        sparse_pytorch_with_sample
                    )
                if "fused_topk" in alg_set:
                    record["ms_fused_topk"] = run_bench(
                        lambda: fused_linear_constrained_node_transition_topk(
                            a, weight.T, cs, k=k
                        )
                    )
                if "sparse_pytorch_topk" in alg_set:
                    record["ms_sparse_pytorch_topk"] = run_bench(
                        sparse_pytorch_with_topk
                    )
                if "cute_topk" in alg_set:
                    record["ms_cute_topk"] = run_bench(
                        lambda: fused_linear_constrained_node_transition_topk_cute(
                            a, weight.T, cs, k=k
                        )
                    )

            if "fused_sample" in alg_set and "sparse_pytorch_sample" in alg_set:
                record["speedup_fused_vs_sparse_pytorch_sample"] = (
                    record["ms_sparse_pytorch_sample"] / record["ms_fused_sample"]
                )
            if "fused_topk" in alg_set and "sparse_pytorch_topk" in alg_set:
                record["speedup_fused_topk_vs_sparse_pytorch_topk"] = (
                    record["ms_sparse_pytorch_topk"] / record["ms_fused_topk"]
                )
            if "cute_topk" in alg_set and "fused_topk" in alg_set:
                record["speedup_cute_topk_vs_triton_topk"] = (
                    record["ms_fused_topk"] / record["ms_cute_topk"]
                )

            records.append(record)

    return pd.DataFrame(records)


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
    parser.add_argument(
        "--sparsity",
        type=float,
        default=DEFAULT_SPARSITY,
        help="Fraction of vocab used as max branches (default: %(default)s)",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=K_TOP,
        help="k for top-k benchmarks (default: %(default)s)",
    )
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    os.makedirs("out", exist_ok=True)

    B_vals = [256, 1024, 4096]
    N_vals = [150000]

    print(f"Benchmarking K={K}, sparsity={args.sparsity}, topk={args.topk}")
    print(f"Algorithms: {args.algorithms}")
    print(f"B_vals={B_vals}")
    print(f"N_vals={N_vals}\n")

    df = benchmark_grid(
        B_vals,
        N_vals,
        algorithms=args.algorithms,
        sparsity=args.sparsity,
        k_top=args.topk,
    )
    csv_path = "out/bench_fused_sample.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}\n")

    print(df.to_string(index=False))

    if "speedup_fused_vs_sparse_pytorch_sample" in df.columns:
        plot_heatmap(
            df,
            value_col="speedup_fused_vs_sparse_pytorch_sample",
            title=f"Fused sample speedup vs compile(sparse_linear_pytorch)+multinomial  (K={K})",
            filename="out/heatmap_fused_sample_vs_sparse_pytorch.jpg",
            cbar_label="Speedup (>1 = fused faster)",
        )
    if "speedup_fused_topk_vs_sparse_pytorch_topk" in df.columns:
        plot_heatmap(
            df,
            value_col="speedup_fused_topk_vs_sparse_pytorch_topk",
            title=f"Fused top-k speedup vs compile(sparse_linear_pytorch)+topk  (K={K}, k={args.topk})",
            filename="out/heatmap_fused_topk_vs_sparse_pytorch_topk.jpg",
            cbar_label="Speedup (>1 = fused faster)",
        )
    if "speedup_cute_topk_vs_triton_topk" in df.columns:
        plot_heatmap(
            df,
            value_col="speedup_cute_topk_vs_triton_topk",
            title=f"CuTe top-k speedup vs Triton top-k  (K={K}, k={args.topk})",
            filename="out/heatmap_cute_vs_triton_cst_topk.jpg",
            cbar_label="Speedup (>1 = CuTe faster)",
        )
