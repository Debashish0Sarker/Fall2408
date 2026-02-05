# src/feature_signal_report.py
import argparse
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression

TARGETS = ["avg_upload_ms","p95_upload_ms","avg_download_ms","p95_download_ms","reconstruction_time_ms"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/raw/dataset.csv")
    ap.add_argument("--max_features", type=int, default=20)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # Basic numeric-only correlation with targets
    num = df.select_dtypes(include=[np.number]).copy()
    print("=== Top abs(corr) with targets (numeric only) ===")
    for t in TARGETS:
        if t not in num.columns: 
            continue
        corr = num.corr(numeric_only=True)[t].dropna().drop(t, errors="ignore").abs().sort_values(ascending=False)
        print(f"\nTarget: {t}")
        print(corr.head(args.max_features).to_string())

    # Mutual information for one target (example: avg_upload)
    # For MI we need to impute missing and remove non-numerics
    if "avg_upload_ms" in df.columns:
        X = num.drop(columns=[c for c in TARGETS if c in num.columns], errors="ignore").fillna(0.0)
        y = df["avg_upload_ms"].astype(float).fillna(df["avg_upload_ms"].median())
        if X.shape[1] > 0:
            mi = mutual_info_regression(X, y, random_state=42)
            mi_s = pd.Series(mi, index=X.columns).sort_values(ascending=False)
            print("\n=== Mutual Information vs avg_upload_ms (numeric only) ===")
            print(mi_s.head(args.max_features).to_string())

if __name__ == "__main__":
    main()
