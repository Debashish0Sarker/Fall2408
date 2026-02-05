# src/evaluate.py
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TARGETS = ["avg_upload_ms", "p95_upload_ms", "avg_download_ms", "p95_download_ms", "reconstruction_time_ms"]


def load_model(path: Path) -> CatBoostRegressor:
    m = CatBoostRegressor()
    m.load_model(str(path))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", default="data/processed")
    ap.add_argument("--splits_dir", default="data/splits")
    ap.add_argument("--models_dir", default="models")
    ap.add_argument("--results_dir", default="results")
    args = ap.parse_args()

    processed_dir = Path(args.processed_dir)
    splits_dir = Path(args.splits_dir)
    models_dir = Path(args.models_dir)
    results_dir = Path(args.results_dir)

    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "plots").mkdir(parents=True, exist_ok=True)

    meta = json.loads((processed_dir / "metadata.json").read_text())
    feature_cols = meta["feature_cols"]
    cat_cols = meta["categorical_cols"]
    cat_idx = [feature_cols.index(c) for c in cat_cols if c in feature_cols]

    df_base = pd.read_csv(processed_dir / "train_base.csv")
    test_idx = np.array([int(x) for x in (splits_dir / "test_idx.txt").read_text().splitlines() if x.strip()])

    X_test = df_base.iloc[test_idx][feature_cols]
    table_rows = []

    # Detect log transform from model_info.json
    transform = "none"
    model_info_path = models_dir / "model_info.json"
    if model_info_path.exists():
        try:
            model_info = json.loads(model_info_path.read_text())
            transform = model_info.get("target_transform", "none")
        except Exception:
            transform = "none"
    use_log = (transform == "log1p")

    for tgt in TARGETS:
        model_path = models_dir / f"{tgt}.cbm"
        if not model_path.exists():
            continue
        if tgt not in df_base.columns:
            continue

        y_test = df_base.iloc[test_idx][tgt].astype(float)

        # recon: evaluate only on failure rows where recon exists
        if tgt == "reconstruction_time_ms":
            mask = (df_base.iloc[test_idx]["failure_state"] == 1) & (~y_test.isna())
            if mask.sum() < 5:
                table_rows.append([tgt, None, None, None, None, None, None, int(mask.sum()), "not enough recon rows in test split"])
                continue
            X_eval = X_test[mask]
            y_eval = y_test[mask]
        else:
            mask = ~y_test.isna()
            X_eval = X_test[mask]
            y_eval = y_test[mask]

        model = load_model(model_path)
        pool = Pool(X_eval, cat_features=cat_idx)
        pred = model.predict(pool)

        pred_ms = np.expm1(pred) if use_log else pred
        y_eval_ms = y_eval.to_numpy(dtype=float)

        mae = float(mean_absolute_error(y_eval_ms, pred_ms))
        rmse = float(mean_squared_error(y_eval_ms, pred_ms, squared=False))
        r2 = float(r2_score(y_eval_ms, pred_ms))

        median_y = float(np.median(y_eval_ms)) if len(y_eval_ms) > 0 else np.nan
        nmae = float(mae / median_y) if (median_y and median_y > 0) else np.nan
        nrmse = float(rmse / median_y) if (median_y and median_y > 0) else np.nan

        table_rows.append([tgt, mae, rmse, r2, nmae, nrmse, median_y, int(len(y_eval_ms)), "ok"])

        # Feature importance
        pool_with_label = Pool(X_eval, y_eval_ms, cat_features=cat_idx)
        imp = model.get_feature_importance(pool_with_label)
        imp_df = pd.DataFrame({"feature": feature_cols, "importance": imp}).sort_values("importance", ascending=False)
        imp_df.to_csv(results_dir / f"feature_importance_{tgt}.csv", index=False)

        # Plot predicted vs actual
        plt.figure()
        plt.scatter(y_eval_ms, pred_ms)
        plt.xlabel("Actual (ms)")
        plt.ylabel("Predicted (ms)")
        plt.title(f"Predicted vs Actual: {tgt}" + (" (log1p-trained)" if use_log else ""))
        plt.savefig(results_dir / "plots" / f"pred_vs_actual_{tgt}.png", dpi=150)
        plt.close()

    metrics_df = pd.DataFrame(
        table_rows,
        columns=["target", "MAE_ms", "RMSE_ms", "R2", "NMAE_vs_median", "NRMSE_vs_median", "median_y_ms", "n_eval", "notes"],
    )
    metrics_df.to_csv(results_dir / "metrics_table.csv", index=False)

    print(f"[OK] Transform: {transform}")
    print(f"[OK] Wrote -> {results_dir / 'metrics_table.csv'}")
    print(f"[OK] Wrote plots -> {results_dir / 'plots'}")


if __name__ == "__main__":
    main()
