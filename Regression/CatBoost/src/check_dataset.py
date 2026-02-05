# src/check_dataset.py
import argparse
import pandas as pd

DEFAULT_TARGETS = [
    "avg_upload_ms",
    "p95_upload_ms",
    "avg_download_ms",
    "p95_download_ms",
    "reconstruction_time_ms",
]

LEAKY_OR_POSTRUN_HINTS = [
    "_after",
    "_during",
    "_throughput",
    "_bytes_delta",
    "_pct_delta",
    "total_upload_bytes",
    "total_download_bytes",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw/dataset.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    print(f"[INFO] Shape: {df.shape[0]} rows × {df.shape[1]} cols\n")

    print("=== Columns ===")
    print(df.columns.tolist(), "\n")

    print("=== Missing values per column (top) ===")
    miss = df.isna().sum().sort_values(ascending=False)
    miss = miss[miss > 0]
    print(miss.head(25).to_string() if len(miss) else "None", "\n")

    if "failure_state" in df.columns:
        print("=== failure_state counts ===")
        print(df["failure_state"].value_counts(dropna=False).to_string(), "\n")

    if "policy_name" in df.columns:
        print("=== policy_name counts ===")
        print(df["policy_name"].value_counts(dropna=False).to_string(), "\n")

    # Constant columns
    nunq = df.nunique(dropna=False)
    const_cols = nunq[nunq <= 1].index.tolist()
    print("=== Constant columns (nunique <= 1) ===")
    print(const_cols if const_cols else "None", "\n")

    # Quick detect potentially leaky columns
    leaky = []
    for c in df.columns:
        cl = c.lower()
        if any(h in cl for h in LEAKY_OR_POSTRUN_HINTS):
            leaky.append(c)
    print("=== Potentially post-run/leaky columns (name heuristic) ===")
    print(leaky if leaky else "None", "\n")

    # Target quick stats
    print("=== Target quick stats ===")
    for t in DEFAULT_TARGETS:
        if t in df.columns:
            s = df[t]
            print(
                f"{t}: non-null={int(s.notna().sum())} "
                f"min={s.min(skipna=True)} median={s.median(skipna=True)} max={s.max(skipna=True)}"
            )


if __name__ == "__main__":
    main()
