"""Compute and compare sequence-length statistics for the training datasets used in
finetune_qwen_beauty.gin and sid_align_qwen_beauty.gin.

Both configs draw from the same four .pt files but assign them to different splits
(finetune swaps the order of beauty_train.pt vs beauty_train_item.pt in the list).
This script reports per-file stats and per-config combined train/eval stats side-by-side.

Usage:
    python examples/scripts/training/sequence_length_stats.py
    python examples/scripts/training/sequence_length_stats.py --data_root /path/to/project/root
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Dataset file sets per config (paths relative to project root)
# ---------------------------------------------------------------------------

FINETUNE_TRAIN = [
    "data/precomputed/beauty/beauty_train.pt",
    "data/precomputed/beauty/beauty_train_item.pt",
]
FINETUNE_EVAL = [
    "data/precomputed/beauty/beauty_eval.pt",
    "data/precomputed/beauty/beauty_eval_item.pt",
]

SID_ALIGN_TRAIN = [
    "data/precomputed/beauty/beauty_train_item.pt",
    "data/precomputed/beauty/beauty_train.pt",
]
SID_ALIGN_EVAL = [
    "data/precomputed/beauty/beauty_eval_item.pt",
    "data/precomputed/beauty/beauty_eval.pt",
]

# Unique files across both configs
ALL_FILES = sorted(
    {
        *FINETUNE_TRAIN,
        *FINETUNE_EVAL,
        *SID_ALIGN_TRAIN,
        *SID_ALIGN_EVAL,
    }
)

PERCENTILES = [25, 50, 75, 90, 95, 99]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_lengths(path: Path) -> np.ndarray:
    """Return an array of input_ids lengths for every sample in a .pt file."""
    data = torch.load(path, weights_only=False)
    return np.array([len(s["input_ids"]) for s in data["samples"]], dtype=np.int64)


def compute_stats(lengths: np.ndarray) -> dict:
    pct = np.percentile(lengths, PERCENTILES)
    return {
        "count": len(lengths),
        "min": int(lengths.min()),
        "max": int(lengths.max()),
        "mean": float(lengths.mean()),
        "std": float(lengths.std()),
        **{f"p{p}": float(v) for p, v in zip(PERCENTILES, pct)},
    }


def combined_lengths(paths: list[str], root: Path) -> np.ndarray:
    arrays = [load_lengths(root / p) for p in paths]
    return np.concatenate(arrays)


def print_stats_table(label: str, stats: dict) -> None:
    print(f"\n  {label}")
    print(f"    count : {stats['count']:>10,}")
    print(f"    min   : {stats['min']:>10}")
    print(f"    max   : {stats['max']:>10}")
    print(f"    mean  : {stats['mean']:>10.1f}  (std={stats['std']:.1f})")
    for p in PERCENTILES:
        print(f"    p{p:<3}  : {stats[f'p{p}']:>10.1f}")


def print_comparison_row(
    metric: str,
    val_a: float | int,
    val_b: float | int,
    fmt: str = ".1f",
) -> None:
    diff = val_b - val_a  # type: ignore[operator]
    pct = (diff / val_a * 100) if val_a != 0 else float("nan")
    flag = ""
    if isinstance(val_a, float):
        print(
            f"    {metric:<8} {val_a:{fmt}}  {val_b:{fmt}}  "
            f"Δ={diff:+.1f} ({pct:+.1f}%){flag}"
        )
    else:
        print(
            f"    {metric:<8} {val_a:>10}  {val_b:>10}  Δ={diff:+d} ({pct:+.1f}%){flag}"
        )


def side_by_side(label_a: str, stats_a: dict, label_b: str, stats_b: dict) -> None:
    col_w = max(len(label_a), len(label_b), 30)
    print(f"\n  {'metric':<8} {label_a:>{col_w}}  {label_b:>{col_w}}  delta")
    print("  " + "-" * (col_w * 2 + 30))
    for key in ["count", "min", "max", "mean", "std"] + [f"p{p}" for p in PERCENTILES]:
        va, vb = stats_a[key], stats_b[key]
        metric = key
        if isinstance(va, int):
            diff = vb - va
            pct = (diff / va * 100) if va != 0 else float("nan")
            print(
                f"  {metric:<8} {va:>{col_w},}  {vb:>{col_w},}  Δ={diff:+,} ({pct:+.1f}%)"
            )
        else:
            diff = vb - va
            pct = (diff / va * 100) if va != 0 else float("nan")
            print(
                f"  {metric:<8} {va:>{col_w}.1f}  {vb:>{col_w}.1f}  "
                f"Δ={diff:+.1f} ({pct:+.1f}%)"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_root",
        default=".",
        help="Project root directory (default: current working directory)",
    )
    args = parser.parse_args()
    root = Path(args.data_root)

    # ------------------------------------------------------------------
    # 1. Per-file statistics
    # ------------------------------------------------------------------
    print("=" * 70)
    print("PER-FILE SEQUENCE LENGTH STATISTICS")
    print("=" * 70)

    file_stats: dict[str, dict] = {}
    for rel_path in ALL_FILES:
        abs_path = root / rel_path
        if not abs_path.exists():
            print(f"\n  [SKIP] {rel_path}  (file not found)")
            continue
        lengths = load_lengths(abs_path)
        stats = compute_stats(lengths)
        file_stats[rel_path] = stats
        print_stats_table(rel_path, stats)

    # ------------------------------------------------------------------
    # 2. Per-config combined stats (train + eval treated separately)
    # ------------------------------------------------------------------
    configs = {
        "finetune_qwen_beauty": {
            "train": FINETUNE_TRAIN,
            "eval": FINETUNE_EVAL,
        },
        "sid_align_qwen_beauty": {
            "train": SID_ALIGN_TRAIN,
            "eval": SID_ALIGN_EVAL,
        },
    }

    config_stats: dict[str, dict[str, dict]] = {}
    for cfg_name, splits in configs.items():
        config_stats[cfg_name] = {}
        print(f"\n{'=' * 70}")
        print(f"CONFIG: {cfg_name}")
        print("=" * 70)
        for split_name, paths in splits.items():
            available = [p for p in paths if (root / p).exists()]
            if not available:
                print(f"\n  [{split_name}] No files found, skipping.")
                continue
            if len(available) < len(paths):
                missing = set(paths) - set(available)
                print(f"\n  [{split_name}] WARNING: missing files: {missing}")
            lengths = combined_lengths(available, root)
            stats = compute_stats(lengths)
            config_stats[cfg_name][split_name] = stats
            print_stats_table(f"{split_name} ({len(available)} file(s))", stats)

    # ------------------------------------------------------------------
    # 3. Side-by-side comparison between configs
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("SIDE-BY-SIDE COMPARISON: finetune vs sid_align")
    print("=" * 70)

    ft = config_stats.get("finetune_qwen_beauty", {})
    sa = config_stats.get("sid_align_qwen_beauty", {})

    for split in ("train", "eval"):
        if split not in ft or split not in sa:
            print(
                f"\n  [{split}] Cannot compare — data missing for one or both configs."
            )
            continue
        print(f"\n  SPLIT: {split}")
        side_by_side(
            "finetune",
            ft[split],
            "sid_align",
            sa[split],
        )

    # ------------------------------------------------------------------
    # 4. Cross-file comparison: _item vs non-_item files
    # ------------------------------------------------------------------
    item_keys = [k for k in file_stats if k.endswith("_item.pt")]
    base_keys = [k for k in file_stats if not k.endswith("_item.pt")]

    if item_keys and base_keys:
        print(f"\n{'=' * 70}")
        print("FILE-LEVEL COMPARISON: *_item.pt  vs  non-item .pt files")
        print("=" * 70)

        item_lengths = np.concatenate(
            [load_lengths(root / k) for k in item_keys if (root / k).exists()]
        )
        base_lengths = np.concatenate(
            [load_lengths(root / k) for k in base_keys if (root / k).exists()]
        )
        side_by_side(
            "_item files",
            compute_stats(item_lengths),
            "non-item files",
            compute_stats(base_lengths),
        )


if __name__ == "__main__":
    main()
