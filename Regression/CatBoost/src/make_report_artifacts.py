# src/make_report_artifacts.py
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out_dir", default="results/report_artifacts")
    ap.add_argument("--details_csv", default="policy_selection_details.csv")
    ap.add_argument("--summary_json", default="policy_selection_summary.json")
    ap.add_argument("--topk_max", type=int, default=10)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    details_path = results_dir / args.details_csv
    summary_path = results_dir / args.summary_json

    df = pd.read_csv(details_path)

    # --- Regret plots (requires actual_regret column OR regret column)
    regret_col = None
    for c in ["actual_regret", "regret"]:
        if c in df.columns:
            regret_col = c
            break
    if regret_col is None:
        raise ValueError("No regret column found in details CSV.")

    regrets = df[regret_col].astype(float).to_numpy()

    # histogram
    plt.figure()
    plt.hist(regrets, bins=60)
    plt.xlabel(regret_col)
    plt.ylabel("count")
    plt.title("Regret distribution")
    plt.savefig(out_dir / "regret_hist.png", dpi=150)
    plt.close()

    # CDF
    r = np.sort(regrets)
    y = np.arange(1, len(r) + 1) / len(r)
    plt.figure()
    plt.plot(r, y)
    plt.xlabel(regret_col)
    plt.ylabel("CDF")
    plt.title("Regret CDF")
    plt.savefig(out_dir / "regret_cdf.png", dpi=150)
    plt.close()

    # --- Selection frequency
    if "pred_best_policy" in df.columns:
        freq = df["pred_best_policy"].value_counts().reset_index()
        freq.columns = ["policy", "count"]
        freq["pct"] = freq["count"] / freq["count"].sum()
        freq.to_csv(out_dir / "pred_best_policy_frequency.csv", index=False)

    # --- Top-K table (if details has ranks or topk flags)
    # If your evaluate_policy_selection writes topk flags, compute them.
    # Otherwise we can only write Top-1 from 'correct' column.
    topk_table = []
    if "correct" in df.columns:
        top1 = float(df["correct"].mean())
        topk_table.append({"k": 1, "accuracy": top1})

    # Write summary copy (for convenience)
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        (out_dir / "policy_selection_summary_copy.json").write_text(json.dumps(summary, indent=2))

    pd.DataFrame(topk_table).to_csv(out_dir / "topk_accuracy_table.csv", index=False)

    print(f"[OK] Wrote report artifacts to: {out_dir}")


if __name__ == "__main__":
    main()
