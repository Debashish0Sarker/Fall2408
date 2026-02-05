# src/evaluate_policy_selection.py
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

LAT_TARGETS = ["avg_upload_ms", "p95_upload_ms", "avg_download_ms", "p95_download_ms"]
RECON_TARGET = "reconstruction_time_ms"

DEFAULT_WEIGHTS = {
    "avg_upload_ms": 1.0,
    "p95_upload_ms": 2.0,
    "avg_download_ms": 1.0,
    "p95_download_ms": 2.0,
    "reconstruction_time_ms": 1.5,
}


def load_model(path: Path):
    m = CatBoostRegressor()
    m.load_model(str(path))
    return m


def weighted_score_df(df: pd.DataFrame, weights: dict) -> pd.Series:
    s = 0.0
    for k, w in weights.items():
        if k not in df.columns:
            continue
        x = df[k].astype(float)
        s = s + float(w) * x.where(~x.isna(), 0.0)
    return s.astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", default="data/processed")
    ap.add_argument("--splits_dir", default="data/splits")
    ap.add_argument("--models_dir", default="models")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--weights_json", default="", help="Optional path OR JSON string")
    ap.add_argument("--group_cols", default="", help="Comma-separated context columns (default: workload_id if present).")
    ap.add_argument("--topk", type=int, default=3, help="K for Top-K accuracy (default 3).")
    args = ap.parse_args()

    processed_dir = Path(args.processed_dir)
    splits_dir = Path(args.splits_dir)
    models_dir = Path(args.models_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    meta = json.loads((processed_dir / "metadata.json").read_text())
    feature_cols = meta["feature_cols"]
    cat_cols = meta["categorical_cols"]

    df_base = pd.read_csv(processed_dir / "train_base.csv")

    test_idx = np.array(
        [int(x) for x in (splits_dir / "test_idx.txt").read_text().splitlines() if x.strip()],
        dtype=int
    )
    df_test = df_base.iloc[test_idx].copy()

    # weights
    weights = DEFAULT_WEIGHTS
    if args.weights_json:
        p = Path(args.weights_json)
        weights = json.loads(p.read_text()) if p.exists() else json.loads(args.weights_json)

    # detect transform
    transform = "none"
    model_info_path = models_dir / "model_info.json"
    if model_info_path.exists():
        try:
            model_info = json.loads(model_info_path.read_text())
            transform = model_info.get("target_transform", "none")
        except Exception:
            transform = "none"
    use_log = (transform == "log1p")

    # group cols
    if args.group_cols.strip():
        group_cols = [c.strip() for c in args.group_cols.split(",") if c.strip()]
    else:
        group_cols = ["workload_id"] if "workload_id" in df_test.columns else [
            c for c in ["file_size_mb", "num_objects", "concurrency", "read_ratio", "failure_state"]
            if c in df_test.columns
        ]

    for c in group_cols:
        if c not in df_test.columns:
            raise ValueError(f"group col '{c}' not present in df_test")

    if "policy_name" not in df_test.columns:
        raise ValueError("policy_name missing in processed test data")

    policies = sorted(df_test["policy_name"].astype(str).unique().tolist())
    topk = max(1, int(args.topk))

    # -------------------------
    # ACTUAL: per-row score
    # -------------------------
    df_actual = df_test.copy()
    if "failure_state" in df_actual.columns:
        df_actual.loc[df_actual["failure_state"] != 1, RECON_TARGET] = np.nan
    df_actual["actual_score_row"] = weighted_score_df(df_actual, weights)

    # Aggregate actual mean score per (context, policy)
    actual_gp = (
        df_actual.groupby(group_cols + ["policy_name"], dropna=False)["actual_score_row"]
        .mean()
        .reset_index()
        .rename(columns={"actual_score_row": "actual_score_mean"})
    )

    # -------------------------
    # PRED: per-row score
    # -------------------------
    models = {t: load_model(models_dir / f"{t}.cbm") for t in LAT_TARGETS if (models_dir / f"{t}.cbm").exists()}
    recon_path = models_dir / f"{RECON_TARGET}.cbm"
    recon_model = load_model(recon_path) if recon_path.exists() else None

    X = df_test[feature_cols].copy()
    cat_idx = [feature_cols.index(c) for c in cat_cols if c in feature_cols]
    for c in cat_cols:
        if c in X.columns:
            X[c] = X[c].astype(str)

    pool = Pool(X, cat_features=cat_idx)

    df_pred = df_test[group_cols + ["policy_name", "failure_state"]].copy()

    for t, m in models.items():
        yhat = m.predict(pool)
        df_pred[t] = np.expm1(yhat) if use_log else yhat

    if recon_model is not None:
        yhat = recon_model.predict(pool)
        df_pred[RECON_TARGET] = np.expm1(yhat) if use_log else yhat
    else:
        df_pred[RECON_TARGET] = np.nan

    if "failure_state" in df_pred.columns:
        df_pred.loc[df_pred["failure_state"] != 1, RECON_TARGET] = np.nan

    df_pred["pred_score_row"] = weighted_score_df(df_pred, weights)

    pred_gp = (
        df_pred.groupby(group_cols + ["policy_name"], dropna=False)["pred_score_row"]
        .mean()
        .reset_index()
        .rename(columns={"pred_score_row": "pred_score_mean"})
    )

    # Merge: every (context, policy) gets both actual and pred mean scores
    merged = pd.merge(actual_gp, pred_gp, on=group_cols + ["policy_name"], how="inner")

    contexts = merged[group_cols].drop_duplicates()
    details = []

    top1_correct = 0
    topk_correct = 0
    total = 0
    actual_regrets = []

    for _, ctx_row in contexts.iterrows():
        ctx = ctx_row.to_dict()

        mctx = merged
        for c in group_cols:
            mctx = mctx[mctx[c] == ctx[c]]
        if mctx.empty:
            continue

        # true best by actual
        true_row = mctx.loc[mctx["actual_score_mean"].idxmin()]
        true_best_policy = str(true_row["policy_name"])
        true_best_actual = float(true_row["actual_score_mean"])

        # predicted ranking by pred_score_mean
        mctx_sorted = mctx.sort_values("pred_score_mean", ascending=True).reset_index(drop=True)
        pred_best_policy = str(mctx_sorted.loc[0, "policy_name"])

        # Top-K list
        topk_policies = mctx_sorted["policy_name"].astype(str).head(topk).tolist()

        is_top1 = (pred_best_policy == true_best_policy)
        is_topk = (true_best_policy in topk_policies)

        # actual regret of the chosen predicted policy
        actual_of_pred_choice = float(mctx[mctx["policy_name"] == pred_best_policy]["actual_score_mean"].iloc[0])
        actual_regret = actual_of_pred_choice - true_best_actual
        if actual_regret < 0:
            # numeric jitter protection; should be non-negative in theory
            actual_regret = 0.0

        top1_correct += int(is_top1)
        topk_correct += int(is_topk)
        total += 1
        actual_regrets.append(float(actual_regret))

        # rank of true best within predicted order (1-based)
        true_rank = int(np.where(mctx_sorted["policy_name"].astype(str).to_numpy() == true_best_policy)[0][0]) + 1

        details.append({
            **ctx,
            "actual_best_policy": true_best_policy,
            "pred_best_policy": pred_best_policy,
            "true_best_rank_in_pred": true_rank,
            "topk": topk,
            "topk_hit": int(is_topk),
            "top1_hit": int(is_top1),
            "actual_best_score_mean": true_best_actual,
            "actual_score_mean_of_pred_choice": actual_of_pred_choice,
            "actual_regret": float(actual_regret),
            "pred_topk_policies": ",".join(topk_policies),
        })

    acc1 = (top1_correct / total) if total else 0.0
    acck = (topk_correct / total) if total else 0.0
    arr = np.array(actual_regrets, dtype=float) if actual_regrets else np.array([0.0], dtype=float)

    summary = {
        "n_contexts": int(total),
        "top1_accuracy": float(acc1),
        f"top{topk}_accuracy": float(acck),
        "actual_regret_mean": float(np.mean(arr)),
        "actual_regret_median": float(np.median(arr)),
        "actual_regret_p95": float(np.percentile(arr, 95)),
        "transform": transform,
        "group_cols": group_cols,
        "policies_considered": policies,
        "weights": weights,
        "notes": [
            "REAL-ROW evaluation: no synthetic context vectors are created.",
            "Scores are averaged per (context, policy) over real measured rows (actual) and model predictions (pred).",
            "actual_regret = actual_score_mean(pred_best_policy) - actual_score_mean(actual_best_policy) and is >= 0.",
        ],
    }

    out_json = results_dir / "policy_selection_summary.json"
    out_csv = results_dir / "policy_selection_details.csv"
    out_json.write_text(json.dumps(summary, indent=2))
    pd.DataFrame(details).to_csv(out_csv, index=False)

    print(f"[OK] Wrote -> {out_json}")
    print(f"[OK] Wrote -> {out_csv}")
    print(f"[OK] Top-1 accuracy: {acc1:.4f} | Top-{topk} accuracy: {acck:.4f} | actual_regret_mean={summary['actual_regret_mean']:.4f} | transform={transform} | group_cols={group_cols}")


if __name__ == "__main__":
    main()
