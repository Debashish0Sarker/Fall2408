# src/train_models.py
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

LATENCY_TARGETS = ["avg_upload_ms", "p95_upload_ms", "avg_download_ms", "p95_download_ms"]
RECON_TARGET = "reconstruction_time_ms"


def load_indices(path: Path) -> np.ndarray:
    return np.array([int(x.strip()) for x in path.read_text().splitlines() if x.strip()], dtype=int)


def _safe_log1p(y: np.ndarray) -> np.ndarray:
    if np.nanmin(y) < 0:
        raise ValueError("Negative target encountered; cannot apply log1p safely.")
    return np.log1p(y)


def fit_one_model(df, feature_cols, cat_cols, y_col, train_idx, test_idx, params, log_targets: bool):
    X = df[feature_cols]
    y_raw = df[y_col].astype(float).to_numpy()

    y = _safe_log1p(y_raw) if log_targets else y_raw

    cat_idx = [feature_cols.index(c) for c in cat_cols if c in feature_cols]
    train_pool = Pool(X.iloc[train_idx], y[train_idx], cat_features=cat_idx)
    test_pool = Pool(X.iloc[test_idx], y[test_idx], cat_features=cat_idx)

    model = CatBoostRegressor(**params)
    model.fit(train_pool, eval_set=test_pool, verbose=False)

    pred = model.predict(test_pool)

    if log_targets:
        pred_ms = np.expm1(pred)
        y_test_ms = np.expm1(y[test_idx])
    else:
        pred_ms = pred
        y_test_ms = y[test_idx]

    mae = float(mean_absolute_error(y_test_ms, pred_ms))
    rmse = float(mean_squared_error(y_test_ms, pred_ms, squared=False))
    r2 = float(r2_score(y_test_ms, pred_ms))
    return model, {"mae": mae, "rmse": rmse, "r2": r2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", default="data/processed")
    ap.add_argument("--splits_dir", default="data/splits")
    ap.add_argument("--models_dir", default="models")
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--log_targets", action="store_true")
    args = ap.parse_args()

    processed_dir = Path(args.processed_dir)
    splits_dir = Path(args.splits_dir)
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    meta = json.loads((processed_dir / "metadata.json").read_text())
    feature_cols = meta["feature_cols"]
    cat_cols = meta["categorical_cols"]

    df_base = pd.read_csv(processed_dir / "train_base.csv")
    train_idx = load_indices(splits_dir / "train_idx.txt")
    test_idx = load_indices(splits_dir / "test_idx.txt")

    params = dict(
        loss_function="RMSE",
        random_seed=args.random_state,
        iterations=2000,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3.0,
        eval_metric="RMSE",
        od_type="Iter",
        od_wait=100,
        allow_writing_files=False,
    )

    results = {
        "latency_models": {},
        "recon_model": None,
        "params": params,
        "features": feature_cols,
        "categoricals": cat_cols,
        "target_transform": "log1p" if args.log_targets else "none",
    }

    # Latency models
    for target in LATENCY_TARGETS:
        model, metrics = fit_one_model(
            df_base, feature_cols, cat_cols, target, train_idx, test_idx, params, log_targets=args.log_targets
        )
        out_path = models_dir / f"{target}.cbm"
        model.save_model(out_path)
        results["latency_models"][target] = {"model_file": str(out_path), "metrics": metrics}
        print(f"[OK] Trained {target} -> {out_path} | {metrics}")

    # Recon model: same split as base, but filtered to failure_state==1 and recon not null
    if "failure_state" not in df_base.columns:
        raise ValueError("failure_state missing from train_base.csv")

    mask_all = (df_base["failure_state"] == 1) & (~df_base[RECON_TARGET].isna())
    eligible_idx = np.where(mask_all.to_numpy())[0]

    tr = np.intersect1d(train_idx, eligible_idx)
    te = np.intersect1d(test_idx, eligible_idx)

    if len(tr) >= 50 and len(te) >= 20:
        model, metrics = fit_one_model(
            df_base, feature_cols, cat_cols, RECON_TARGET, tr, te, params, log_targets=args.log_targets
        )
        out_path = models_dir / f"{RECON_TARGET}.cbm"
        model.save_model(out_path)
        results["recon_model"] = {"model_file": str(out_path), "metrics": metrics}
        print(f"[OK] Trained {RECON_TARGET} -> {out_path} | {metrics}")
    else:
        print(f"[WARN] Not enough recon rows in split (train={len(tr)}, test={len(te)}).")
        results["recon_model"] = {"model_file": None, "metrics": None, "warning": "Not enough recon rows in split."}

    (models_dir / "model_info.json").write_text(json.dumps(results, indent=2))
    print(f"[OK] Wrote -> {models_dir / 'model_info.json'}")


if __name__ == "__main__":
    main()
