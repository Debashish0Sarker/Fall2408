# src/plot_feature_importance.py
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out_dir", default="results/report_artifacts")
    ap.add_argument("--topn", type=int, default=15)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for csv_path in results_dir.glob("feature_importance_*.csv"):
        df = pd.read_csv(csv_path).sort_values("importance", ascending=False).head(args.topn)
        plt.figure()
        plt.barh(df["feature"][::-1], df["importance"][::-1])
        plt.xlabel("importance")
        plt.title(csv_path.stem.replace("feature_importance_", "Feature importance: "))
        plt.tight_layout()
        plt.savefig(out_dir / f"{csv_path.stem}_top{args.topn}.png", dpi=150)
        plt.close()

    print(f"[OK] Wrote plots to {out_dir}")


if __name__ == "__main__":
    main()
