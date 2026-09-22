"""Tests for the fused Triton EMA-update kernel.

Each test runs the same scenario through both the pure-PyTorch reference
(``_cpu_ema_update``) and the CUDA Triton kernel (``_cuda_ema_update``) and
asserts numerical equivalence.

Test matrix:
    - All codes active (every codebook entry used at least once)
    - Some codes inactive (zero assignments → counter incremented, no EMA change)
    - Dead-code restart triggered (counter reaches threshold → replacement)
    - Single batch sample (B=1)
    - Large codebook relative to batch (forces many inactive / dead codes)
    - Non-power-of-two batch size (verifies boundary masking in BLOCK_B loop)
"""

from __future__ import annotations

import copy
import unittest

import torch

if not torch.cuda.is_available():
    raise unittest.SkipTest("CUDA required for EMA kernel tests")

from rectokens.ops.ema_update import _cpu_ema_update, _cuda_ema_update

DEVICE = torch.device("cuda")
CPU = torch.device("cpu")

# Tolerances: the Triton kernel accumulates in fp32; minor rounding differences
# vs. the PyTorch reference are expected.
ATOL = 1e-5
RTOL = 1e-4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(
    K: int,
    D: int,
    *,
    ema_decay: float = 0.99,
    seed: int = 0,
    device: torch.device = DEVICE,
) -> dict[str, torch.Tensor]:
    """Allocate fresh EMA buffers and a codebook for K entries of dimension D."""
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return {
        "ema_cluster_size": torch.rand(K, device=device, generator=g),
        "ema_embed_sum": torch.randn(K, D, device=device, generator=g),
        "codebook": torch.randn(K, D, device=device, generator=g),
        "steps_since_active": torch.randint(0, 5, (K,), device=device, dtype=torch.long),
    }


def _clone_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in state.items()}


def _run_both(
    x: torch.Tensor,
    codes: torch.Tensor,
    state: dict[str, torch.Tensor],
    decay: float,
    restart_after_steps: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Run CPU reference and CUDA kernel on identical state copies.

    Returns ``(cpu_state, gpu_state)`` after the update.
    """
    # CPU reference operates on CPU tensors.
    cpu_state = {k: v.cpu().clone() for k, v in state.items()}
    _cpu_ema_update(
        x.cpu(), codes.cpu(),
        cpu_state["ema_cluster_size"],
        cpu_state["ema_embed_sum"],
        cpu_state["codebook"],
        cpu_state["steps_since_active"],
        decay,
        restart_after_steps,
    )

    # CUDA kernel operates on CUDA tensors.
    gpu_state = _clone_state(state)
    _cuda_ema_update(
        x, codes,
        gpu_state["ema_cluster_size"],
        gpu_state["ema_embed_sum"],
        gpu_state["codebook"],
        gpu_state["steps_since_active"],
        decay,
        restart_after_steps,
    )
    torch.cuda.synchronize()

    return cpu_state, gpu_state


def _assert_states_close(
    cpu: dict[str, torch.Tensor],
    gpu: dict[str, torch.Tensor],
    msg: str = "",
) -> None:
    """Assert that the CPU and GPU states are numerically close."""
    for key in ("ema_cluster_size", "ema_embed_sum"):
        assert torch.allclose(cpu[key], gpu[key].cpu(), atol=ATOL, rtol=RTOL), (
            f"{msg}: mismatch in {key}\n"
            f"  cpu={cpu[key].flatten()[:8]}\n"
            f"  gpu={gpu[key].cpu().flatten()[:8]}"
        )
    # steps_since_active must be bit-exact (integer).
    assert torch.equal(cpu["steps_since_active"], gpu["steps_since_active"].cpu()), (
        f"{msg}: mismatch in steps_since_active\n"
        f"  cpu={cpu['steps_since_active']}\n"
        f"  gpu={gpu['steps_since_active'].cpu()}"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEMAUpdateKernel(unittest.TestCase):

    # ── basic EMA update — all codes active ─────────────────────────────────

    def test_all_codes_active(self) -> None:
        """Every codebook entry receives at least one assignment."""
        torch.manual_seed(1)
        K, D, B = 16, 64, 128
        x = torch.randn(B, D, device=DEVICE)
        # Assign at least one sample per code, then fill the rest randomly.
        codes = torch.cat([
            torch.arange(K, device=DEVICE),
            torch.randint(K, (B - K,), device=DEVICE),
        ])
        state = _make_state(K, D)
        cpu, gpu = _run_both(x, codes, state, decay=0.99, restart_after_steps=20)
        _assert_states_close(cpu, gpu, "all_codes_active")

    # ── some codes inactive ──────────────────────────────────────────────────

    def test_some_codes_inactive(self) -> None:
        """Only the first half of codes receive assignments."""
        torch.manual_seed(2)
        K, D, B = 16, 64, 64
        x = torch.randn(B, D, device=DEVICE)
        # Only codes [0, K//2) are assigned; codes [K//2, K) stay inactive.
        codes = torch.randint(K // 2, (B,), device=DEVICE)
        state = _make_state(K, D)
        cpu, gpu = _run_both(x, codes, state, decay=0.99, restart_after_steps=20)
        _assert_states_close(cpu, gpu, "some_codes_inactive")

        # Inactive codes should have their step counter incremented, not reset.
        inactive_steps_cpu = cpu["steps_since_active"][K // 2 :]
        inactive_steps_gpu = gpu["steps_since_active"].cpu()[K // 2 :]
        assert torch.equal(inactive_steps_cpu, inactive_steps_gpu)

    # ── dead-code restart ────────────────────────────────────────────────────

    def test_dead_code_restart(self) -> None:
        """Codes whose counter hits the threshold are replaced.

        We seed ``steps_since_active`` so that inactive codes are exactly one
        step away from the threshold, then run one update step without
        assigning them.  After the update their counters must be 0 and the
        EMA accumulators must be zeroed.
        """
        torch.manual_seed(3)
        K, D, B = 8, 32, 16
        restart_thresh = 5

        x = torch.randn(B, D, device=DEVICE)
        # Only code 0 is active; codes 1..K-1 are inactive and at threshold-1.
        codes = torch.zeros(B, device=DEVICE, dtype=torch.long)
        state = _make_state(K, D)
        # Set inactive codes to one step below the threshold so this update
        # pushes them to exactly restart_thresh and triggers a restart.
        state["steps_since_active"][1:] = restart_thresh - 1

        cpu, gpu = _run_both(x, codes, state, decay=0.99, restart_after_steps=restart_thresh)

        # Dead codes (1..K-1) must have their counters reset to 0.
        assert (cpu["steps_since_active"][1:] == 0).all(), (
            "CPU: dead-code steps not reset"
        )
        assert (gpu["steps_since_active"].cpu()[1:] == 0).all(), (
            "GPU: dead-code steps not reset"
        )
        # Dead codes' EMA cluster sizes must be zeroed.
        assert (cpu["ema_cluster_size"][1:] == 0).all()
        assert torch.allclose(gpu["ema_cluster_size"].cpu()[1:],
                              torch.zeros(K - 1), atol=ATOL)
        # Dead codes' EMA embed sums must be zeroed.
        assert (cpu["ema_embed_sum"][1:] == 0).all()
        assert torch.allclose(gpu["ema_embed_sum"].cpu()[1:],
                              torch.zeros(K - 1, D), atol=ATOL)

    # ── single sample ────────────────────────────────────────────────────────

    def test_single_sample(self) -> None:
        """B=1 — exercises the BLOCK_B boundary masking."""
        torch.manual_seed(4)
        K, D, B = 8, 64, 1
        x = torch.randn(B, D, device=DEVICE)
        codes = torch.zeros(B, device=DEVICE, dtype=torch.long)
        state = _make_state(K, D)
        cpu, gpu = _run_both(x, codes, state, decay=0.9, restart_after_steps=10)
        _assert_states_close(cpu, gpu, "single_sample")

    # ── non-power-of-two batch ───────────────────────────────────────────────

    def test_non_pow2_batch(self) -> None:
        """B=100 — batch size not a multiple of any BLOCK_B config."""
        torch.manual_seed(5)
        K, D, B = 32, 64, 100
        x = torch.randn(B, D, device=DEVICE)
        codes = torch.randint(K, (B,), device=DEVICE)
        state = _make_state(K, D)
        cpu, gpu = _run_both(x, codes, state, decay=0.99, restart_after_steps=20)
        _assert_states_close(cpu, gpu, "non_pow2_batch")

    # ── large codebook ───────────────────────────────────────────────────────

    def test_large_codebook(self) -> None:
        """K=256, B=64 — most codes are inactive each step."""
        torch.manual_seed(6)
        K, D, B = 256, 64, 64
        x = torch.randn(B, D, device=DEVICE)
        codes = torch.randint(K, (B,), device=DEVICE)
        state = _make_state(K, D)
        cpu, gpu = _run_both(x, codes, state, decay=0.99, restart_after_steps=20)
        _assert_states_close(cpu, gpu, "large_codebook")

    # ── high-dimensional embeddings ──────────────────────────────────────────

    def test_large_dim(self) -> None:
        """D=256 — verifies register pressure is manageable."""
        torch.manual_seed(7)
        K, D, B = 64, 256, 128
        x = torch.randn(B, D, device=DEVICE)
        codes = torch.randint(K, (B,), device=DEVICE)
        state = _make_state(K, D)
        cpu, gpu = _run_both(x, codes, state, decay=0.99, restart_after_steps=20)
        _assert_states_close(cpu, gpu, "large_dim")

    # ── EMA decay edge cases ─────────────────────────────────────────────────

    def test_decay_zero(self) -> None:
        """decay=0 → EMA collapses to per-batch statistics."""
        torch.manual_seed(8)
        K, D, B = 16, 64, 64
        x = torch.randn(B, D, device=DEVICE)
        codes = torch.arange(K, device=DEVICE).repeat(B // K)
        state = _make_state(K, D)
        cpu, gpu = _run_both(x, codes, state, decay=0.0, restart_after_steps=20)
        _assert_states_close(cpu, gpu, "decay_zero")

    def test_decay_one(self) -> None:
        """decay=1 → EMA never changes (new statistics have zero weight)."""
        torch.manual_seed(9)
        K, D, B = 16, 64, 64
        x = torch.randn(B, D, device=DEVICE)
        codes = torch.arange(K, device=DEVICE).repeat(B // K)
        state = _make_state(K, D)
        # Stash original EMA values.
        orig_ema_cs = state["ema_cluster_size"].clone()
        orig_ema_es = state["ema_embed_sum"].clone()

        cpu, gpu = _run_both(x, codes, state, decay=1.0, restart_after_steps=20)
        _assert_states_close(cpu, gpu, "decay_one")

        # With decay=1 the EMA values must not change at all.
        assert torch.allclose(gpu["ema_cluster_size"].cpu(), orig_ema_cs.cpu(), atol=ATOL), (
            "ema_cluster_size changed with decay=1"
        )
        assert torch.allclose(gpu["ema_embed_sum"].cpu(), orig_ema_es.cpu(), atol=ATOL), (
            "ema_embed_sum changed with decay=1"
        )

    # ── steps_since_active semantics ─────────────────────────────────────────

    def test_active_code_resets_counter(self) -> None:
        """An active code's step counter must be exactly 0 after the update."""
        torch.manual_seed(10)
        K, D, B = 4, 32, 8
        x = torch.randn(B, D, device=DEVICE)
        codes = torch.zeros(B, device=DEVICE, dtype=torch.long)  # only code 0 used
        state = _make_state(K, D)
        state["steps_since_active"][0] = 99  # should be reset to 0

        cpu, gpu = _run_both(x, codes, state, decay=0.99, restart_after_steps=20)
        _assert_states_close(cpu, gpu, "active_code_resets_counter")

        assert cpu["steps_since_active"][0].item() == 0, "CPU: active code step not reset"
        assert gpu["steps_since_active"].cpu()[0].item() == 0, "GPU: active code step not reset"
