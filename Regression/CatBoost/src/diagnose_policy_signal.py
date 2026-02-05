# src/diagnose_policy_signal.py
import argparse
import pandas as pd
import numpy as np

DEFAULT_WEIGHTS = {
    "avg_upload_ms": 1.0,
    "p95_upload_ms": 2.0,
    "avg_download_ms": 1.0,
    "p95_download_ms": 2.0,
    "reconstruction_time_ms": 1.5,
}

TARGETS = list(DEFAULT_WEIGHTS.keys())

def compute_score(df: pd.DataFrame, weights: dict) -> pd.Series:
    s = 0.0
    for k, w in weights.items():
        if k not in df.columns:
            continue
        x = df[k].astype(float)
        s = s + w * x.fillna(0.0)
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/raw/dataset.csv")
    ap.add_argument("--group_cols", default="workload_id,file_size_mb,num_objects,concurrency,read_ratio,failure_state")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    group_cols = [c.strip() for c in args.group_cols.split(",") if c.strip()]
    missing = [c for c in group_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing group cols: {missing}")

    # recon irrelevant when no failure
    if "failure_state" in df.columns:
        df.loc[df["failure_state"] != 1, "reconstruction_time_ms"] = np.nan

    df["score"] = compute_score(df, DEFAULT_WEIGHTS)

    # require that within each context we have multiple policies
    grp = df.groupby(group_cols, dropna=False)

    margins = []
    n_ok = 0
    for _, g in grp:
        if "policy_name" not in g.columns:
            continue
        if g["policy_name"].nunique() < 2:
            continue
        # best and second best actual score
        scores = g.groupby("policy_name")["score"].mean().sort_values()
        if len(scores) < 2:
            continue
        best = float(scores.iloc[0])
        second = float(scores.iloc[1])
        margins.append(second - best)
        n_ok += 1

    margins = np.array(margins, dtype=float) if margins else np.array([np.nan])
    print(f"[INFO] contexts_with_2+policies = {n_ok}")
    print(f"[INFO] margin_mean = {np.nanmean(margins):.4f}")
    print(f"[INFO] margin_median = {np.nanmedian(margins):.4f}")
    print(f"[INFO] margin_p10/p50/p90 = {np.nanpercentile(margins,10):.4f} / {np.nanpercentile(margins,50):.4f} / {np.nanpercentile(margins,90):.4f}")
    print("[NOTE] If margins are often near 0, Top-1 accuracy will be low no matter what.")

if __name__ == "__main__":
    main()
