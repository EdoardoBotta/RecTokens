"""
Benchmark: fused_linear_constrained_node_transition
    vs torch.compile(nn.Linear) + constrained_node_transition (Triton kernel)
    vs torch.compile(nn.Linear) + vtnk_pytorch
    vs torch.compile(sparse_linear_pytorch)

Grid: B (batch size) × N (vocab / logits size). K (hidden dim) fixed.
"""

import os
import torch
import torch.nn as nn
import triton.testing as testing
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from rectokens.decoding.csr import csr_from_sorted_batch
from rectokens.decoding.vntk import vtnk_pytorch, sparse_linear_pytorch
from rectokens.decoding.kernels.vtnk import (
    constrained_node_transition,
    fused_linear_constrained_node_transition,
)

DEVICE = torch.device("cuda")
K = 512
MAX_BRANCHES = 16
WARMUP = 25
REP = 100


def lex_sort(rows: list[list[int]]) -> torch.Tensor:
    return torch.tensor(sorted(rows), dtype=torch.long)


def make_csr(vocab_size: int, max_branches: int) -> object:
    seqs = [[i] for i in range(max_branches)]
    csr = csr_from_sorted_batch(lex_sort(seqs), vocab_size=vocab_size)
    return csr._replace(
        row_ptrs=csr.row_ptrs.to(DEVICE),
        stacked_cols_vals=csr.stacked_cols_vals.to(DEVICE),
        dense_mask_by_layer=[v.to(DEVICE) for v in csr.dense_mask_by_layer],
        dense_states=csr.dense_states.to(DEVICE),
    )


def run_bench(fn):
    return testing.do_bench(fn, warmup=WARMUP, rep=REP)


def benchmark_grid(B_vals, N_vals):
    records = []

    for B in B_vals:
        for N in N_vals:
            print(f"  B={B:6d}  N={N:6d}")

            csr = make_csr(vocab_size=N, max_branches=MAX_BRANCHES)

            # Inputs
            a = torch.randn(B, K, device=DEVICE)
            weight = torch.randn(
                N, K, device=DEVICE
            )  # nn.Linear weight shape (out, in)
            cur_node = torch.zeros(B, dtype=torch.long, device=DEVICE)

            # compiled linear (shared across fn2 and fn3)
            linear = torch.compile(nn.Linear(K, N, bias=False).to(DEVICE))
            with torch.no_grad():
                linear.weight.data.copy_(weight)

            sparse_linear_pytorch_compiled = torch.compile(sparse_linear_pytorch)

            # --- warmup / force compilation ---
            with torch.no_grad():
                fused_linear_constrained_node_transition(
                    a, weight.T, cur_node, csr, step=0
                )
                logits_w = a @ weight.T
                constrained_node_transition(
                    logits_w, cur_node, csr, step=0, vocab_size=N
                )
                vtnk_pytorch(logits_w, cur_node, csr, step=0)
                linear(a)
                sparse_linear_pytorch_compiled(a, weight, cur_node, csr, step=0)

            # --- benchmark ---
            with torch.no_grad():
                ms_fused = run_bench(
                    lambda: fused_linear_constrained_node_transition(
                        a, weight.T, cur_node, csr, step=0
                    )
                )
                ms_kernel = run_bench(
                    lambda: constrained_node_transition(
                        a @ weight.T, cur_node, csr, step=0, vocab_size=N
                    )
                )
                ms_pytorch = run_bench(
                    lambda: vtnk_pytorch(linear(a), cur_node, csr, step=0)
                )
                ms_sparse_pytorch = run_bench(
                    lambda: sparse_linear_pytorch_compiled(
                        a, weight, cur_node, csr, step=0
                    )
                )

            records.append(
                {
                    "B": B,
                    "N": N,
                    "ms_fused": ms_fused,
                    "ms_kernel": ms_kernel,
                    "ms_pytorch": ms_pytorch,
                    "ms_sparse_pytorch": ms_sparse_pytorch,
                    "speedup_fused_vs_kernel": ms_kernel / ms_fused,
                    "speedup_fused_vs_pytorch": ms_pytorch / ms_fused,
                    "speedup_fused_vs_sparse_pytorch": ms_sparse_pytorch / ms_fused,
                }
            )

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
    assert torch.cuda.is_available(), "CUDA required"
    os.makedirs("out", exist_ok=True)

    B_vals = [32, 256, 1024]
    N_vals = [512, 1024, 8192, 150000]

    print(f"Benchmarking K={K}, max_branches={MAX_BRANCHES}")
    print(f"B_vals={B_vals}")
    print(f"N_vals={N_vals}\n")

    df = benchmark_grid(B_vals, N_vals)
    csv_path = "out/bench_vtnk.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}\n")

    print(
        df[
            [
                "B",
                "N",
                "ms_fused",
                "ms_kernel",
                "ms_pytorch",
                "ms_sparse_pytorch",
                "speedup_fused_vs_kernel",
                "speedup_fused_vs_pytorch",
                "speedup_fused_vs_sparse_pytorch",
            ]
        ].to_string(index=False)
    )

    plot_heatmap(
        df,
        value_col="speedup_fused_vs_kernel",
        title=f"Fused speedup vs compiled_linear+constrained_kernel  (K={K})",
        filename="out/heatmap_fused_vs_kernel.jpg",
        cbar_label="Speedup (>1 = fused faster)",
    )
    plot_heatmap(
        df,
        value_col="speedup_fused_vs_pytorch",
        title=f"Fused speedup vs compiled_linear+vtnk_pytorch  (K={K})",
        filename="out/heatmap_fused_vs_pytorch.jpg",
        cbar_label="Speedup (>1 = fused faster)",
    )
    plot_heatmap(
        df,
        value_col="speedup_fused_vs_sparse_pytorch",
        title=f"Fused speedup vs sparse_linear_pytorch  (K={K})",
        filename="out/heatmap_fused_vs_sparse_pytorch.jpg",
        cbar_label="Speedup (>1 = fused faster)",
    )
