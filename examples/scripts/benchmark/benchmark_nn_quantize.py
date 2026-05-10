"""
Benchmark: nearest-neighbor quantization kernels
    quantize_fwd   – Triton kernel (sequential scan over N per B-block)
    quantize_fwd_mm – Triton kernel (MM-style, parallel over B×N tiles)
    cdist_compiled – torch.compile(torch.cdist + argmin)
    faiss_search   – FAISS-GPU flat L2, index pre-built (static codebook)

Grid: B (batch size) × D (embedding dim). N (codebook size) fixed per run.
Heatmap axes: B (batch size) vs D (embedding dim).
"""

import argparse
import os
import torch
import triton.testing as testing
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from rectokens.kernels.nn_quantize import quantize_fwd, quantize_fwd_mm
from rectokens.ops.faiss_quantize import make_gpu_index

DEVICE = torch.device("cuda")
N = 256
WARMUP = 25
FAISS_WARMUP = 100
REP = 100

ALL_ALGORITHMS = ["quantize_fwd", "quantize_fwd_mm", "cdist_compiled", "faiss_search"]


def cdist_nn(x: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    return torch.cdist(x, codebook).argmin(-1)


cdist_nn_compiled = torch.compile(cdist_nn)


def run_bench(fn, warmup=WARMUP):
    return testing.do_bench(fn, warmup=warmup, rep=REP)


def benchmark_grid(B_vals, D_vals, algorithms):
    alg_set = set(algorithms)
    records = []

    for B in B_vals:
        for D in D_vals:
            print(f"  B={B:6d}  D={D:6d}")

            x = torch.randn(B, D, device=DEVICE)
            codebook = torch.randn(N, D, device=DEVICE)

            if "faiss_search" in alg_set:
                gpu_index = make_gpu_index(codebook)

            # warmup / force compilation / autotuning
            with torch.no_grad():
                if "quantize_fwd" in alg_set:
                    quantize_fwd(x, codebook)
                if "quantize_fwd_mm" in alg_set:
                    quantize_fwd_mm(x, codebook)
                if "cdist_compiled" in alg_set:
                    cdist_nn_compiled(x, codebook)
                if "faiss_search" in alg_set:
                    gpu_index.search(x.contiguous(), 1)

            record = {"B": B, "D": D, "BN": B * N}

            with torch.no_grad():
                if "quantize_fwd" in alg_set:
                    record["ms_quantize_fwd"] = run_bench(
                        lambda: quantize_fwd(x, codebook)
                    )
                if "quantize_fwd_mm" in alg_set:
                    record["ms_quantize_fwd_mm"] = run_bench(
                        lambda: quantize_fwd_mm(x, codebook)
                    )
                if "cdist_compiled" in alg_set:
                    record["ms_cdist_compiled"] = run_bench(
                        lambda: cdist_nn_compiled(x, codebook)
                    )
                if "faiss_search" in alg_set:
                    record["ms_faiss_search"] = run_bench(
                        lambda: gpu_index.search(x.contiguous(), 1),
                        warmup=FAISS_WARMUP,
                    )

            if "quantize_fwd" in alg_set and "quantize_fwd_mm" in alg_set:
                record["speedup_fwd_vs_mm"] = (
                    record["ms_quantize_fwd_mm"] / record["ms_quantize_fwd"]
                )
            if "quantize_fwd" in alg_set and "cdist_compiled" in alg_set:
                record["speedup_fwd_vs_cdist"] = (
                    record["ms_cdist_compiled"] / record["ms_quantize_fwd"]
                )
            if "quantize_fwd_mm" in alg_set and "cdist_compiled" in alg_set:
                record["speedup_mm_vs_cdist"] = (
                    record["ms_cdist_compiled"] / record["ms_quantize_fwd_mm"]
                )
            if "quantize_fwd" in alg_set and "faiss_search" in alg_set:
                record["speedup_fwd_vs_faiss"] = (
                    record["ms_faiss_search"] / record["ms_quantize_fwd"]
                )
            if "quantize_fwd_mm" in alg_set and "faiss_search" in alg_set:
                record["speedup_mm_vs_faiss"] = (
                    record["ms_faiss_search"] / record["ms_quantize_fwd_mm"]
                )

            records.append(record)

    return pd.DataFrame(records)


def plot_heatmap(df, value_col, title, filename, fmt=".2f", cbar_label="Speedup"):
    pivot = df.pivot(index="B", columns="D", values=value_col)
    pivot = pivot.sort_index()
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=fmt,
        cmap="viridis",
        cbar_kws={"label": cbar_label},
    )
    plt.title(title)
    plt.ylabel("Batch size (B)")
    plt.xlabel("Embedding dim (D)")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, format="jpg")
    plt.close()
    print(f"  Saved {filename}")


def run_for_N(n_val, B_vals, D_vals, algorithms):
    global N
    N = n_val

    print(f"\n{'=' * 50}")
    print(f"Benchmarking N={N} (fixed)")
    print(f"Algorithms: {algorithms}")
    print(f"B_vals={B_vals}")
    print(f"D_vals={D_vals}\n")

    df = benchmark_grid(B_vals, D_vals, algorithms=algorithms)
    csv_path = f"out/bench_nn_quantize_N{N}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}\n")

    print(df.to_string(index=False))

    if "speedup_fwd_vs_mm" in df.columns:
        plot_heatmap(
            df,
            value_col="speedup_fwd_vs_mm",
            title=f"quantize_fwd speedup vs quantize_fwd_mm  (N={N})",
            filename=f"out/heatmap_fwd_vs_mm_N{N}.jpg",
            cbar_label="Speedup (>1 = fwd faster)",
        )
    if "speedup_fwd_vs_cdist" in df.columns:
        plot_heatmap(
            df,
            value_col="speedup_fwd_vs_cdist",
            title=f"quantize_fwd speedup vs cdist_compiled  (N={N})",
            filename=f"out/heatmap_fwd_vs_cdist_N{N}.jpg",
            cbar_label="Speedup (>1 = fwd faster)",
        )
    if "speedup_mm_vs_cdist" in df.columns:
        plot_heatmap(
            df,
            value_col="speedup_mm_vs_cdist",
            title=f"quantize_fwd_mm speedup vs cdist_compiled  (N={N})",
            filename=f"out/heatmap_mm_vs_cdist_N{N}.jpg",
            cbar_label="Speedup (>1 = mm faster)",
        )
    if "speedup_fwd_vs_faiss" in df.columns:
        plot_heatmap(
            df,
            value_col="speedup_fwd_vs_faiss",
            title=f"quantize_fwd speedup vs faiss_search  (N={N})",
            filename=f"out/heatmap_fwd_vs_faiss_N{N}.jpg",
            cbar_label="Speedup (>1 = fwd faster)",
        )
    if "speedup_mm_vs_faiss" in df.columns:
        plot_heatmap(
            df,
            value_col="speedup_mm_vs_faiss",
            title=f"quantize_fwd_mm speedup vs faiss_search  (N={N})",
            filename=f"out/heatmap_mm_vs_faiss_N{N}.jpg",
            cbar_label="Speedup (>1 = mm faster)",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark nearest-neighbor quantization algorithms."
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=ALL_ALGORITHMS,
        default=ALL_ALGORITHMS,
        metavar="ALGO",
        help=f"Algorithms to benchmark. Choices: {ALL_ALGORITHMS} (default: all)",
    )
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    os.makedirs("out", exist_ok=True)

    B_vals = [32, 256, 1024, 4096, 16384, 32768, 65536]
    D_vals = [64, 128, 256]

    for n_val in [64, 128, 256, 512]:
        run_for_N(n_val, B_vals, D_vals, algorithms=args.algorithms)
