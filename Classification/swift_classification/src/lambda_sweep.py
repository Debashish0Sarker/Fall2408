from __future__ import annotations
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data import load_dataset
from .config import ScoreConfig
from .oracle import compute_scores

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--lambda_min", type=float, default=0.0)
    ap.add_argument("--lambda_max", type=float, default=0.6)
    ap.add_argument("--lambda_step", type=float, default=0.05)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    df = load_dataset(args.csv)
    cfg = ScoreConfig()

    lambdas = np.arange(args.lambda_min, args.lambda_max + 1e-9, args.lambda_step).round(4)

    rows = []
    for lam in lambdas:
        scores = compute_scores(df, cfg, lambda_storage=float(lam))
        # Oracle best per context
        best = scores.loc[scores.groupby("workload_id")["S_total"].idxmin(), ["workload_id","policy_name","S_total","storage_overhead"]]
        # Summaries
        rows.append({
            "lambda_storage": float(lam),
            "avg_overhead_selected": float(best["storage_overhead"].mean()),
            "median_overhead_selected": float(best["storage_overhead"].median()),
            "policy_entropy": float(pd.Series(best["policy_name"]).value_counts(normalize=True).pipe(lambda p: -(p*np.log2(p)).sum())),
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(os.path.join(args.out, "lambda_sweep.csv"), index=False)

    # Plot overhead vs lambda
    plt.figure(figsize=(8,4))
    plt.plot(out_df["lambda_storage"], out_df["avg_overhead_selected"], marker="o")
    plt.xlabel("lambda_storage")
    plt.ylabel("Avg selected storage_overhead")
    plt.title("Storage-overhead vs lambda (oracle)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "lambda_vs_overhead.png"), dpi=200)
    plt.close()

    # Plot entropy vs lambda (label diversity)
    plt.figure(figsize=(8,4))
    plt.plot(out_df["lambda_storage"], out_df["policy_entropy"], marker="o")
    plt.xlabel("lambda_storage")
    plt.ylabel("Entropy of selected policies (bits)")
    plt.title("Oracle label diversity vs lambda")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "lambda_vs_entropy.png"), dpi=200)
    plt.close()

    print("Saved:", os.path.join(args.out, "lambda_sweep.csv"))
    print("Saved plots:", os.path.join(args.out, "lambda_vs_overhead.png"), "and", os.path.join(args.out, "lambda_vs_entropy.png"))

if __name__ == "__main__":
    main()
