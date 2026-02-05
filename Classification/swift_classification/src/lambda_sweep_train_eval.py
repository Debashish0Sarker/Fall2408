from __future__ import annotations
import argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data import load_dataset
from .config import ScoreConfig
from .oracle import compute_scores, build_oracle_contexts
from .train import split_by_workload, make_preprocessor

from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

EPS = 1e-9

def stats(arr: np.ndarray) -> dict:
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"mean": np.nan, "median": np.nan, "p95": np.nan}
    return {"mean": float(np.mean(a)), "median": float(np.median(a)), "p95": float(np.percentile(a, 95))}

def minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mn, mx = np.nanmin(x), np.nanmax(x)
    return (x - mn) / (mx - mn + EPS)

def compute_perf_regret(scores: pd.DataFrame, workload_ids: np.ndarray, pred_policies: np.ndarray) -> pd.DataFrame:
    """
    perf_regret = chosen S_perf_norm - best S_perf_norm (best perf among policies)
    """
    chosen_map = dict(zip(workload_ids, pred_policies))
    tmp = scores[["workload_id","policy_name","S_perf_norm","storage_overhead"]].copy()
    tmp["chosen_policy"] = tmp["workload_id"].map(chosen_map)

    best_perf = tmp.groupby("workload_id")["S_perf_norm"].min().reset_index().rename(columns={"S_perf_norm":"best_perf"})
    chosen = tmp[tmp["policy_name"] == tmp["chosen_policy"]][["workload_id","S_perf_norm","storage_overhead"]].rename(
        columns={"S_perf_norm":"chosen_perf", "storage_overhead":"chosen_overhead"}
    )

    out = best_perf.merge(chosen, on="workload_id", how="left").dropna(subset=["chosen_perf"])
    out["perf_regret"] = out["chosen_perf"] - out["best_perf"]
    return out

def lineplot(df, x, y, title, ylabel, outpath):
    plt.figure(figsize=(8,4))
    plt.plot(df[x], df[y], marker="o")
    plt.grid(True)
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lambda_min", type=float, default=0.0)
    ap.add_argument("--lambda_max", type=float, default=1.0)
    ap.add_argument("--lambda_step", type=float, default=0.05)
    ap.add_argument("--overhead_max", type=float, default=None,
                    help="Optional constraint: only consider lambdas where avg chosen overhead <= this")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    df = load_dataset(args.csv)
    cfg = ScoreConfig()
    lambdas = np.arange(args.lambda_min, args.lambda_max + 1e-9, args.lambda_step).round(4)

    rows = []
    for lam in lambdas:
        # 1) Compute scores + oracle labels for THIS lambda
        scores = compute_scores(df, cfg, lambda_storage=float(lam))
        ctx, _ = build_oracle_contexts(df, scores)

        # 2) Split by workload_id (no leakage)
        train_ids, val_ids, _test_ids = split_by_workload(ctx, seed=args.seed)
        train_df = ctx[ctx["workload_id"].isin(train_ids)].copy()
        val_df   = ctx[ctx["workload_id"].isin(val_ids)].copy()

        target = "best_policy_true"
        drop_cols = ["workload_id", target, "best_score_true"]
        feature_cols = [c for c in ctx.columns if c not in drop_cols]

        # 3) Preprocess for XGB
        cat_cols = [c for c in ["num_objects_bin", "failure_disk"] if c in feature_cols]
        pre, _, _ = make_preprocessor(feature_cols, cat_cols)

        Xtr = pre.fit_transform(train_df[feature_cols])
        Xva = pre.transform(val_df[feature_cols])

        le = LabelEncoder()
        ytr = le.fit_transform(train_df[target].astype(str))
        yva = le.transform(val_df[target].astype(str))

        # 4) Train XGB (proxy learner)
        xgb = XGBClassifier(
            n_estimators=900,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="multi:softprob",
            num_class=len(le.classes_),
            random_state=args.seed,
            tree_method="hist",
            eval_metric="mlogloss",
        )
        xgb.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)

        # 5) Predict policies on validation
        proba = xgb.predict_proba(Xva)
        pred_ids = np.argmax(proba, axis=1)
        pred_policies = le.inverse_transform(pred_ids)

        y_true = val_df[target].astype(str).values
        top1 = float(np.mean(pred_policies == y_true))

        # 6) Compute PERFORMANCE regret (independent of lambda)
        perf_reg_df = compute_perf_regret(scores, val_df["workload_id"].values, pred_policies)
        perf_stats = stats(perf_reg_df["perf_regret"].values)

        avg_overhead = float(perf_reg_df["chosen_overhead"].mean()) if perf_reg_df.shape[0] else np.nan

        rows.append({
            "lambda_storage": float(lam),
            "val_top1": top1,
            "val_perf_regret_mean": perf_stats["mean"],
            "val_perf_regret_median": perf_stats["median"],
            "val_perf_regret_p95": perf_stats["p95"],
            "val_avg_chosen_overhead": avg_overhead,
            "n_val_contexts": int(val_df.shape[0]),
        })

    out_df = pd.DataFrame(rows)
    out_csv = os.path.join(args.out, "lambda_sweep_balanced.csv")
    out_df.to_csv(out_csv, index=False)

    # ---- Choose lambda: closest-to-ideal in (perf_regret_p95, overhead) space
    filt = out_df.copy()
    if args.overhead_max is not None:
        filt = filt[filt["val_avg_chosen_overhead"] <= float(args.overhead_max)]

    if filt.shape[0] == 0:
        print("No lambda satisfied overhead_max. Try increasing it.")
        return

    perf_norm = minmax(filt["val_perf_regret_p95"].values)
    oh_norm   = minmax(filt["val_avg_chosen_overhead"].values)
    dist = np.sqrt(perf_norm**2 + oh_norm**2)
    best_i = int(np.argmin(dist))
    best_row = filt.iloc[best_i].to_dict()

    print("\n[Suggested lambda] (closest-to-ideal: low tail perf regret AND low overhead)")
    print(best_row)

    # Plots
    lineplot(out_df, "lambda_storage", "val_perf_regret_p95",
             "Validation tail PERFORMANCE regret vs lambda",
             "val_perf_regret_p95",
             os.path.join(args.out, "lambda_vs_val_perf_regret_p95.png"))

    lineplot(out_df, "lambda_storage", "val_avg_chosen_overhead",
             "Avg chosen overhead vs lambda",
             "val_avg_chosen_overhead",
             os.path.join(args.out, "lambda_vs_overhead.png"))

    lineplot(out_df, "lambda_storage", "val_top1",
             "Validation Top-1 accuracy vs lambda",
             "val_top1",
             os.path.join(args.out, "lambda_vs_val_top1.png"))

    # Pareto-style scatter
    plt.figure(figsize=(6,5))
    plt.scatter(out_df["val_avg_chosen_overhead"], out_df["val_perf_regret_p95"])
    plt.xlabel("Avg chosen overhead (lower better)")
    plt.ylabel("p95 perf regret (lower better)")
    plt.title("Trade-off: overhead vs tail perf regret")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "pareto_overhead_vs_perf_regret_p95.png"), dpi=200)
    plt.close()

    print("\nSaved:", out_csv)
    print("Saved plots in:", args.out)

if __name__ == "__main__":
    main()
