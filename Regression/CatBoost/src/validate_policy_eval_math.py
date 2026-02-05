# src/validate_policy_eval_math.py
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

DEFAULT_WEIGHTS = {
    "avg_upload_ms": 1.0,
    "p95_upload_ms": 2.0,
    "avg_download_ms": 1.0,
    "p95_download_ms": 2.0,
    "reconstruction_time_ms": 1.5,
}

def compute_score_row(d: dict, weights: dict) -> float:
    s = 0.0
    for k, w in weights.items():
        v = d.get(k, None)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        s += float(w) * float(v)
    return float(s)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--details_csv", default="results/policy_selection_details.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.details_csv)

    # regret should be >= 0 if actual_best_score is truly minimum and pred_best_score is among candidates
    neg = df[df["regret"] < -1e-9]
    print(f"[INFO] rows={len(df)} | negative_regret_rows={len(neg)}")

    if len(neg) > 0:
        print("[WARN] Negative regret detected. This almost always means evaluation inconsistency.")
        print(neg.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
