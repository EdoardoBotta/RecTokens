"""
Benchmark: trie update speed — CSR full rebuild vs ELL incremental update.

CSR must sort ALL sequences and call from_sorted_batch() from scratch on every
update.  ELL uses MutableELLTrie which pre-allocates a buffer and inserts only
the new nodes/edges, touching zero existing data.

Two benchmark modes
-------------------
single  – Fixed initial catalog of N seqs.  Time one update of M new seqs.
          CSR: sort(initial + new) + from_sorted_batch.
          ELL: MutableELLTrie.update(new_seqs) — pure traversal, no copy.
          State is reset between timing reps by copy_() outside the hot path.

growing – Simulate a catalog that grows over time.  Each of K rounds adds M
          new sequences.  CSR rebuilds from all sequences every round (cost
          grows).  ELL calls update() every round (cost stays constant).
          No state reset needed — rounds chain naturally.
"""

import argparse
import os
import random
import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

from rectokens.schemas.compact_csr_trie import CompactCSRTrie
from rectokens.schemas.compact_ell_trie import CompactELLTrie, MutableELLTrie
from benchmark_config import UPDATE_SUITE, UpdateBenchmarkSuite, UpdateRunConfig


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def lex_sort(seqs: list[list[int]]) -> torch.Tensor:
    return torch.tensor(sorted(seqs), dtype=torch.long)


def generate_unique_seqs(
    n: int,
    L: int,
    vocab_size: int,
    exclude: set[tuple[int, ...]] | None = None,
    seed: int = 0,
) -> list[list[int]]:
    rng = random.Random(seed)
    exclude = exclude or set()
    seqs: set[tuple[int, ...]] = set()
    while len(seqs) < n:
        seq = tuple(rng.randint(0, vocab_size - 1) for _ in range(L))
        if seq not in exclude:
            seqs.add(seq)
    return [list(s) for s in seqs]


# ---------------------------------------------------------------------------
# "single" benchmark
# ---------------------------------------------------------------------------

def bench_single(run: UpdateRunConfig, suite: UpdateBenchmarkSuite) -> dict:
    """One update of run.update_n sequences into a trie pre-loaded with run.initial_n seqs."""
    initial_seqs = generate_unique_seqs(run.initial_n, suite.seq_len, suite.vocab_size, seed=0)
    initial_set = {tuple(s) for s in initial_seqs}
    new_seqs = generate_unique_seqs(
        run.update_n, suite.seq_len, suite.vocab_size, exclude=initial_set, seed=1
    )

    initial_tensor = lex_sort(initial_seqs)
    csr_init = CompactCSRTrie.from_sorted_batch(initial_tensor, suite.vocab_size)
    ell_init = CompactELLTrie.from_csr(csr_init)

    all_seqs = initial_seqs + new_seqs

    def csr_rebuild() -> None:
        CompactCSRTrie.from_sorted_batch(lex_sort(all_seqs), suite.vocab_size)

    capacity = run.update_n * suite.seq_len
    mutable = MutableELLTrie(ell_init, extra_capacity=capacity)

    snap_cv = mutable.ell_cv.clone()
    snap_nch = mutable.n_ch.clone()
    snap_nn = mutable.num_nodes
    snap_mb = mutable.max_branches

    def ell_update() -> None:
        mutable.update(new_seqs)

    def ell_reset() -> None:
        if mutable.ell_cv.shape == snap_cv.shape:
            mutable.ell_cv.copy_(snap_cv)
            mutable.n_ch.copy_(snap_nch)
        else:
            mutable.ell_cv = snap_cv.clone()
            mutable.n_ch = snap_nch.clone()
        mutable.num_nodes = snap_nn
        mutable.max_branches = snap_mb

    for _ in range(suite.warmup):
        ell_reset()
        ell_update()
    ell_times: list[float] = []
    for _ in range(suite.rep):
        ell_reset()
        t0 = time.perf_counter()
        ell_update()
        ell_times.append((time.perf_counter() - t0) * 1e3)

    for _ in range(suite.warmup):
        csr_rebuild()
    csr_times: list[float] = []
    for _ in range(suite.rep):
        t0 = time.perf_counter()
        csr_rebuild()
        csr_times.append((time.perf_counter() - t0) * 1e3)

    def median_lower(ts: list[float]) -> float:
        ts = sorted(ts)
        half = max(1, len(ts) // 2)
        return sum(ts[:half]) / half

    ms_csr = median_lower(csr_times)
    ms_ell = median_lower(ell_times)
    return {
        "initial_n": run.initial_n,
        "update_n": run.update_n,
        "ms_csr": ms_csr,
        "ms_ell": ms_ell,
        "speedup_ell_vs_csr": ms_csr / ms_ell,
    }


# ---------------------------------------------------------------------------
# "growing" benchmark
# ---------------------------------------------------------------------------

def bench_growing(run: UpdateRunConfig, suite: UpdateBenchmarkSuite) -> dict:
    """Simulate a growing catalog.  Each round adds run.update_n new sequences."""
    total_needed = run.initial_n + run.update_n * suite.growing_rounds
    all_seqs = generate_unique_seqs(total_needed, suite.seq_len, suite.vocab_size, seed=42)
    initial_seqs = all_seqs[:run.initial_n]
    update_batches = [
        all_seqs[run.initial_n + i * run.update_n: run.initial_n + (i + 1) * run.update_n]
        for i in range(suite.growing_rounds)
    ]

    csr_times: list[float] = []
    current_seqs = list(initial_seqs)
    for batch in update_batches:
        current_seqs.extend(batch)
        t0 = time.perf_counter()
        CompactCSRTrie.from_sorted_batch(lex_sort(current_seqs), suite.vocab_size)
        csr_times.append((time.perf_counter() - t0) * 1e3)

    csr_init = CompactCSRTrie.from_sorted_batch(lex_sort(initial_seqs), suite.vocab_size)
    mutable = MutableELLTrie(
        CompactELLTrie.from_csr(csr_init),
        extra_capacity=run.update_n * suite.seq_len * suite.growing_rounds,
    )
    ell_times: list[float] = []
    for batch in update_batches:
        t0 = time.perf_counter()
        mutable.update(batch)
        ell_times.append((time.perf_counter() - t0) * 1e3)

    catalog_sizes = [run.initial_n + (i + 1) * run.update_n for i in range(suite.growing_rounds)]
    return {
        "catalog_sizes": catalog_sizes,
        "csr_times": csr_times,
        "ell_times": ell_times,
        "speedups": [c / e for c, e in zip(csr_times, ell_times)],
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_speedup_heatmap(df: pd.DataFrame, filename: str) -> None:
    pivot = df.pivot(index="initial_n", columns="update_n", values="speedup_ell_vs_csr")
    plt.figure(figsize=(9, 5))
    sns.heatmap(
        pivot.sort_index(),
        annot=True,
        fmt=".1f",
        cmap="viridis",
        cbar_kws={"label": "Speedup (CSR / ELL, >1 = ELL faster)"},
    )
    plt.title("ELL incremental update speedup over CSR full rebuild\n(single-update mode)")
    plt.ylabel("Initial catalogue size")
    plt.xlabel("Update batch size")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, format="jpg")
    plt.close()
    print(f"  Saved {filename}")


def plot_abs_times(df: pd.DataFrame, initial_n: int, filename: str) -> None:
    sub = df[df["initial_n"] == initial_n].copy()
    x = list(range(len(sub)))
    labels = [str(v) for v in sub["update_n"]]
    w = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars_csr = ax.bar([i - w / 2 for i in x], sub["ms_csr"], w, label="CSR (full rebuild)")
    bars_ell = ax.bar([i + w / 2 for i in x], sub["ms_ell"], w, label="ELL (incremental)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Update batch size")
    ax.set_ylabel("Time (ms, log scale)")
    ax.set_title(
        f"Update latency: CSR vs ELL  (initial catalogue = {initial_n:,} seqs,\n"
        f"CSR time includes sort of all sequences; ELL times only new inserts)"
    )
    ax.set_yscale("log")
    ax.legend()
    for bar in list(bars_csr) + list(bars_ell):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h * 1.1,
            f"{h:.2f}", ha="center", va="bottom", fontsize=8,
        )
    plt.tight_layout()
    plt.savefig(filename, dpi=150, format="jpg")
    plt.close()
    print(f"  Saved {filename}")


def plot_growing(result: dict, initial_n: int, update_n: int, filename: str) -> None:
    sizes = result["catalog_sizes"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(sizes, result["csr_times"], marker="o", label="CSR (full rebuild)")
    ax1.plot(sizes, result["ell_times"], marker="s", label="ELL (incremental)")
    ax1.set_xlabel("Total catalogue size")
    ax1.set_ylabel("Update latency (ms)")
    ax1.set_title(
        f"Update latency vs catalogue size\n"
        f"(initial={initial_n:,}, +{update_n} seqs/round)"
    )
    ax1.legend()

    ax2.plot(sizes, result["speedups"], marker="^", color="C2")
    ax2.axhline(1, linestyle="--", color="gray", linewidth=0.8)
    ax2.set_xlabel("Total catalogue size")
    ax2.set_ylabel("Speedup (CSR / ELL)")
    ax2.set_title("Speedup over CSR\n(>1 = ELL faster)")

    plt.tight_layout()
    plt.savefig(filename, dpi=150, format="jpg")
    plt.close()
    print(f"  Saved {filename}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark CSR full rebuild vs ELL incremental update."
    )
    parser.add_argument("--mode", choices=["single", "growing", "both"], default="both")
    args = parser.parse_args()

    os.makedirs("out", exist_ok=True)

    suite = UPDATE_SUITE
    print(
        f"vocab_size={suite.vocab_size}  seq_len={suite.seq_len}  "
        f"warmup={suite.warmup}  rep={suite.rep}"
    )

    # ---- single mode -------------------------------------------------------
    if args.mode in ("single", "both"):
        print("\n=== single-update mode ===")
        print(f"{'initial_n':>10}  {'update_n':>9}  {'ms_csr':>9}  {'ms_ell':>9}  {'speedup':>9}")
        print("-" * 55)

        records = []
        for run in suite.runs:
            rec = bench_single(run, suite)
            records.append(rec)
            print(
                f"{rec['initial_n']:>10}  {rec['update_n']:>9}  "
                f"{rec['ms_csr']:>9.3f}  {rec['ms_ell']:>9.3f}  "
                f"{rec['speedup_ell_vs_csr']:>8.1f}x"
            )

        df = pd.DataFrame(records)
        df.to_csv("out/bench_update_single.csv", index=False)
        print(f"\nSaved out/bench_update_single.csv")
        plot_speedup_heatmap(df, "out/bench_update_speedup_heatmap.jpg")
        largest = max(r.initial_n for r in suite.runs)
        plot_abs_times(df, largest, f"out/bench_update_abs_times_n{largest}.jpg")

    # ---- growing mode -------------------------------------------------------
    if args.mode in ("growing", "both"):
        print("\n=== growing-catalog mode ===")
        initial_n = suite.runs[0].initial_n
        growing_records = []

        for run in suite.runs:
            print(f"  initial_n={run.initial_n}  update_n={run.update_n}  rounds={suite.growing_rounds}")
            result = bench_growing(run, suite)
            plot_growing(
                result,
                initial_n=run.initial_n,
                update_n=run.update_n,
                filename=f"out/bench_update_growing_m{run.update_n}.jpg",
            )
            for size, tc, te, sp in zip(
                result["catalog_sizes"],
                result["csr_times"],
                result["ell_times"],
                result["speedups"],
            ):
                growing_records.append(
                    {
                        "update_n": run.update_n,
                        "catalog_size": size,
                        "ms_csr": tc,
                        "ms_ell": te,
                        "speedup": sp,
                    }
                )

        gdf = pd.DataFrame(growing_records)
        gdf.to_csv("out/bench_update_growing.csv", index=False)
        print(f"  Saved out/bench_update_growing.csv")

        print("\n  Final-round speedup (after all catalog growth):")
        print(f"  {'update_n':>9}  {'catalog_size':>13}  {'ms_csr':>9}  {'ms_ell':>9}  {'speedup':>9}")
        unique_update_ns = sorted(gdf["update_n"].unique())
        for update_n in unique_update_ns:
            row = gdf[gdf["update_n"] == update_n].iloc[-1]
            print(
                f"  {int(row.update_n):>9}  {int(row.catalog_size):>13}  "
                f"{row.ms_csr:>9.3f}  {row.ms_ell:>9.3f}  {row.speedup:>8.1f}x"
            )
