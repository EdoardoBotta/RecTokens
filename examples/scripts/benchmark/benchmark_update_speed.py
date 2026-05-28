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

WARMUP = 5
REP = 20
VOCAB_SIZE = 256    # typical RQ codebook size; bounds max_branches
SEQ_LEN = 4
GROWING_ROUNDS = 30  # rounds in growing mode


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

def bench_single(
    initial_n: int,
    update_n: int,
    vocab_size: int,
    seq_len: int,
    warmup: int,
    rep: int,
) -> dict:
    """One update of update_n sequences into a trie pre-loaded with initial_n seqs."""
    initial_seqs = generate_unique_seqs(initial_n, seq_len, vocab_size, seed=0)
    initial_set = {tuple(s) for s in initial_seqs}
    new_seqs = generate_unique_seqs(
        update_n, seq_len, vocab_size, exclude=initial_set, seed=1
    )

    # Build initial structures (not timed)
    initial_tensor = lex_sort(initial_seqs)
    csr_init = CompactCSRTrie.from_sorted_batch(initial_tensor, vocab_size)
    ell_init = CompactELLTrie.from_csr(csr_init)

    all_seqs = initial_seqs + new_seqs

    # --- CSR: sort ALL sequences then rebuild from scratch ---
    # Sorting is included because CSR must process every sequence on every update.
    def csr_rebuild() -> None:
        CompactCSRTrie.from_sorted_batch(lex_sort(all_seqs), vocab_size)

    # --- ELL: in-place insert of only new_seqs ---
    # Pre-allocate once.  Between timing reps we reset via copy_() outside hot path.
    capacity = update_n * seq_len
    mutable = MutableELLTrie(ell_init, extra_capacity=capacity)

    # Snapshots for resetting mutable state (done outside the hot path, not timed).
    snap_cv = mutable.ell_cv.clone()
    snap_nch = mutable.n_ch.clone()
    snap_nn = mutable.num_nodes
    snap_mb = mutable.max_branches

    def ell_update() -> None:
        mutable.update(new_seqs)

    def ell_reset() -> None:
        # Shape may change if max_branches was unexpectedly expanded; handle both.
        if mutable.ell_cv.shape == snap_cv.shape:
            mutable.ell_cv.copy_(snap_cv)
            mutable.n_ch.copy_(snap_nch)
        else:
            mutable.ell_cv = snap_cv.clone()
            mutable.n_ch = snap_nch.clone()
        mutable.num_nodes = snap_nn
        mutable.max_branches = snap_mb

    # Timed loop — reset happens *outside* the hot path
    for _ in range(warmup):
        ell_reset()
        ell_update()
    ell_times: list[float] = []
    for _ in range(rep):
        ell_reset()
        t0 = time.perf_counter()
        ell_update()
        ell_times.append((time.perf_counter() - t0) * 1e3)

    for _ in range(warmup):
        csr_rebuild()
    csr_times: list[float] = []
    for _ in range(rep):
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
        "initial_n": initial_n,
        "update_n": update_n,
        "ms_csr": ms_csr,
        "ms_ell": ms_ell,
        "speedup_ell_vs_csr": ms_csr / ms_ell,
    }


# ---------------------------------------------------------------------------
# "growing" benchmark
# ---------------------------------------------------------------------------

def bench_growing(
    initial_n: int,
    update_n: int,
    n_rounds: int,
    vocab_size: int,
    seq_len: int,
) -> dict:
    """Simulate a growing catalog.  Each round adds update_n new sequences.

    CSR rebuilds everything each round; ELL only inserts the new sequences.
    No reset between rounds — each builds on the previous state.
    """
    total_needed = initial_n + update_n * n_rounds
    all_seqs = generate_unique_seqs(total_needed, seq_len, vocab_size, seed=42)
    initial_seqs = all_seqs[:initial_n]
    update_batches = [
        all_seqs[initial_n + i * update_n: initial_n + (i + 1) * update_n]
        for i in range(n_rounds)
    ]

    # --- CSR path ---
    csr_times: list[float] = []
    current_seqs = list(initial_seqs)
    for batch in update_batches:
        current_seqs.extend(batch)
        t0 = time.perf_counter()
        CompactCSRTrie.from_sorted_batch(lex_sort(current_seqs), vocab_size)
        csr_times.append((time.perf_counter() - t0) * 1e3)

    # --- ELL path ---
    csr_init = CompactCSRTrie.from_sorted_batch(lex_sort(initial_seqs), vocab_size)
    mutable = MutableELLTrie(
        CompactELLTrie.from_csr(csr_init),
        extra_capacity=update_n * seq_len * n_rounds,
    )
    ell_times: list[float] = []
    for batch in update_batches:
        t0 = time.perf_counter()
        mutable.update(batch)
        ell_times.append((time.perf_counter() - t0) * 1e3)

    catalog_sizes = [initial_n + (i + 1) * update_n for i in range(n_rounds)]
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
    parser.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--rep", type=int, default=REP)
    parser.add_argument(
        "--initial-ns", nargs="+", type=int, default=[10000, 100_000, 100_000], metavar="N"
    )
    parser.add_argument(
        "--update-ns", nargs="+", type=int, default=[100, 1000, 10000], metavar="M"
    )
    parser.add_argument(
        "--growing-rounds", type=int, default=GROWING_ROUNDS,
        help="Number of update rounds for growing mode",
    )
    args = parser.parse_args()

    os.makedirs("out", exist_ok=True)

    print(
        f"vocab_size={args.vocab_size}  seq_len={args.seq_len}  "
        f"warmup={args.warmup}  rep={args.rep}"
    )

    # ---- single mode -------------------------------------------------------
    if args.mode in ("single", "both"):
        print("\n=== single-update mode ===")
        print(f"{'initial_n':>10}  {'update_n':>9}  {'ms_csr':>9}  {'ms_ell':>9}  {'speedup':>9}")
        print("-" * 55)

        records = []
        for initial_n in args.initial_ns:
            for update_n in args.update_ns:
                rec = bench_single(
                    initial_n=initial_n,
                    update_n=update_n,
                    vocab_size=args.vocab_size,
                    seq_len=args.seq_len,
                    warmup=args.warmup,
                    rep=args.rep,
                )
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
        largest = max(args.initial_ns)
        plot_abs_times(df, largest, f"out/bench_update_abs_times_n{largest}.jpg")

    # ---- growing mode -------------------------------------------------------
    if args.mode in ("growing", "both"):
        print("\n=== growing-catalog mode ===")
        # Use one representative (initial_n, update_n) pair per CLI update_n
        initial_n = args.initial_ns[0]
        growing_records = []

        for update_n in args.update_ns:
            print(f"  initial_n={initial_n}  update_n={update_n}  rounds={args.growing_rounds}")
            result = bench_growing(
                initial_n=initial_n,
                update_n=update_n,
                n_rounds=args.growing_rounds,
                vocab_size=args.vocab_size,
                seq_len=args.seq_len,
            )
            plot_growing(
                result,
                initial_n=initial_n,
                update_n=update_n,
                filename=f"out/bench_update_growing_m{update_n}.jpg",
            )
            for size, tc, te, sp in zip(
                result["catalog_sizes"],
                result["csr_times"],
                result["ell_times"],
                result["speedups"],
            ):
                growing_records.append(
                    {
                        "update_n": update_n,
                        "catalog_size": size,
                        "ms_csr": tc,
                        "ms_ell": te,
                        "speedup": sp,
                    }
                )

        gdf = pd.DataFrame(growing_records)
        gdf.to_csv("out/bench_update_growing.csv", index=False)
        print(f"  Saved out/bench_update_growing.csv")

        # Summary: final-round speedup per update_n
        print("\n  Final-round speedup (after all catalog growth):")
        print(f"  {'update_n':>9}  {'catalog_size':>13}  {'ms_csr':>9}  {'ms_ell':>9}  {'speedup':>9}")
        for update_n in args.update_ns:
            row = gdf[gdf["update_n"] == update_n].iloc[-1]
            print(
                f"  {int(row.update_n):>9}  {int(row.catalog_size):>13}  "
                f"{row.ms_csr:>9.3f}  {row.ms_ell:>9.3f}  {row.speedup:>8.1f}x"
            )
