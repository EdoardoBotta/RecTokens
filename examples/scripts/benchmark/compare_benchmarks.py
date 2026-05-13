"""
Compare two benchmark CSV files (e.g. before vs after an optimisation) and
report per-algorithm timing improvements.

Usage:
    python -m examples.scripts.benchmark.compare_benchmarks \
        --before out/bench_vtnk_before.csv \
        --after  out/bench_vtnk_after.csv

Produces a summary table with absolute timings and percentage speedups.
"""

import argparse
import pandas as pd


def compare(before_path: str, after_path: str) -> pd.DataFrame:
    before = pd.read_csv(before_path)
    after = pd.read_csv(after_path)

    key_cols = [c for c in ("B", "N", "D") if c in before.columns and c in after.columns]
    ms_cols = [c for c in before.columns if c.startswith("ms_") and c in after.columns]

    if not ms_cols:
        raise ValueError("No common ms_* timing columns found in both CSVs.")

    merged = before[key_cols + ms_cols].merge(
        after[key_cols + ms_cols],
        on=key_cols,
        suffixes=("_before", "_after"),
    )

    rows = []
    for _, r in merged.iterrows():
        row = {k: r[k] for k in key_cols}
        for col in ms_cols:
            bval = r[f"{col}_before"]
            aval = r[f"{col}_after"]
            row[f"{col}_before"] = round(bval, 4)
            row[f"{col}_after"] = round(aval, 4)
            if bval > 0:
                row[f"{col}_speedup"] = round(bval / aval, 3)
        rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two benchmark CSV files.")
    parser.add_argument("--before", required=True, help="Path to baseline CSV")
    parser.add_argument("--after", required=True, help="Path to optimised CSV")
    args = parser.parse_args()

    df = compare(args.before, args.after)
    print(df.to_string(index=False))
