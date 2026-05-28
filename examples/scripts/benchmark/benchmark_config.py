from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Trie benchmarks (benchmark_fused_sample, benchmark_vtnk)
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    """Parameters for a single trie benchmark run."""
    B: int
    N: int
    k: int
    k_top: int
    sparsity: float


@dataclass
class BenchmarkSuite:
    """Collection of trie benchmark runs with shared timing / mode settings."""
    runs: list[RunConfig]
    warmup: int = 25
    rep: int = 100
    diverse_nodes: bool = False


def trie_runs(
    B_vals: list[int],
    N_vals: list[int],
    k: int,
    k_top: int,
    sparsity: float,
) -> list[RunConfig]:
    """Cross-product of B_vals × N_vals, all sharing the same sparsity."""
    return [
        RunConfig(B=B, N=N, k=k, k_top=k_top, sparsity=sparsity)
        for B in B_vals
        for N in N_vals
    ]


# ---------------------------------------------------------------------------
# Quantize benchmarks (benchmark_nn_quantize)
# ---------------------------------------------------------------------------

@dataclass
class QuantizeRunConfig:
    """Parameters for a single nearest-neighbour quantize benchmark run."""
    B: int
    D: int
    N: int  # codebook size


@dataclass
class QuantizeBenchmarkSuite:
    """Collection of quantize benchmark runs with shared timing settings."""
    runs: list[QuantizeRunConfig]
    warmup: int = 25
    rep: int = 100
    faiss_warmup: int = 100


# ---------------------------------------------------------------------------
# Update-speed benchmarks (benchmark_update_speed)
# ---------------------------------------------------------------------------

@dataclass
class UpdateRunConfig:
    """Parameters for a single trie-update benchmark run."""
    initial_n: int
    update_n: int


@dataclass
class UpdateBenchmarkSuite:
    """Collection of update benchmark runs with shared timing / vocab settings."""
    runs: list[UpdateRunConfig]
    warmup: int = 5
    rep: int = 20
    vocab_size: int = 256
    seq_len: int = 4
    growing_rounds: int = 30


# ---------------------------------------------------------------------------
# Default suites
# ---------------------------------------------------------------------------

FUSED_SAMPLE_SUITE_N256 = BenchmarkSuite(
    runs=trie_runs(B_vals=[256, 1024], N_vals=[256], k=1024, k_top=50, sparsity=0.9),
)

FUSED_SAMPLE_SUITE_N150K = BenchmarkSuite(
    runs=trie_runs(B_vals=[256, 1024], N_vals=[150_000], k=1024, k_top=50, sparsity=0.01),
)

VTNK_SUITE = BenchmarkSuite(
    runs=trie_runs(
        B_vals=[256, 1024],
        N_vals=[150_000],
        k=512,
        k_top=50,
        sparsity=0.01,
    ),
)

QUANTIZE_SUITE = QuantizeBenchmarkSuite(
    runs=[
        QuantizeRunConfig(B=B, D=D, N=N)
        for N in [64, 128, 256, 512]
        for B in [32, 256, 1024, 16384, 32768, 65536]
        for D in [64, 128, 256]
    ],
)

UPDATE_SUITE = UpdateBenchmarkSuite(
    runs=[
        UpdateRunConfig(initial_n=initial_n, update_n=update_n)
        for initial_n in [10_000, 100_000, 100_000]
        for update_n in [100, 1_000, 10_000]
    ],
)
