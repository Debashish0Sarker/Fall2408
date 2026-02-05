# src/recommend.py
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

LAT_TARGETS = ["avg_upload_ms", "p95_upload_ms", "avg_download_ms", "p95_download_ms"]
RECON_TARGET = "reconstruction_time_ms"


def load_model(path: Path):
    m = CatBoostRegressor()
    m.load_model(str(path))
    return m


def compute_score(pred: dict, weights: dict) -> float:
    score = 0.0
    for k, w in weights.items():
        v = pred.get(k, None)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        score += float(w) * float(v)
    return float(score)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_dir", default="models")
    ap.add_argument("--metadata", default="data/processed/metadata.json")
    ap.add_argument("--defaults", default="data/processed/defaults.json", help="Defaults written by preprocessing.py")
    ap.add_argument("--workload_json", required=True)
    ap.add_argument("--policies", required=True, help="Comma-separated e.g. EC-2-2,EC-4-2,EC-6-3")
    ap.add_argument("--weights_json", default="", help="Optional path OR JSON string")
    args = ap.parse_args()

    meta = json.loads(Path(args.metadata).read_text())
    feature_cols = meta["feature_cols"]
    cat_cols = meta["categorical_cols"]
    cat_idx = [feature_cols.index(c) for c in cat_cols if c in feature_cols]

    defaults_path = Path(args.defaults)
    defaults = json.loads(defaults_path.read_text()) if defaults_path.exists() else {"numeric": {}, "categorical": {}}

    models_dir = Path(args.models_dir)

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

    # Load models
    models = {t: load_model(models_dir / f"{t}.cbm") for t in LAT_TARGETS if (models_dir / f"{t}.cbm").exists()}
    recon_model_path = models_dir / f"{RECON_TARGET}.cbm"
    recon_model = load_model(recon_model_path) if recon_model_path.exists() else None

    wl = json.loads(Path(args.workload_json).read_text(encoding="utf-8-sig"))
    policies = [p.strip() for p in args.policies.split(",") if p.strip()]

    default_weights = {
        "avg_upload_ms": 1.0,
        "p95_upload_ms": 2.0,
        "avg_download_ms": 1.0,
        "p95_download_ms": 2.0,
        "reconstruction_time_ms": 1.5,
    }
    weights = default_weights
    if args.weights_json:
        p = Path(args.weights_json)
        weights = json.loads(p.read_text()) if p.exists() else json.loads(args.weights_json)

    rows = []
    for pol in policies:
        row = dict(wl)
        row["policy_name"] = pol
        rows.append(row)

    df = pd.DataFrame(rows)

    # Ensure all features exist using defaults (NOT 0)
    for c in feature_cols:
        if c not in df.columns:
            if c in cat_cols:
                df[c] = defaults["categorical"].get(c, "NONE")
            else:
                df[c] = defaults["numeric"].get(c, 0.0)

    X = df[feature_cols].copy()

    # Fill missing values realistically
    for c in feature_cols:
        if c in cat_cols:
            X[c] = X[c].astype(str).replace({"nan": np.nan}).fillna(defaults["categorical"].get(c, "NONE")).astype(str)
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(defaults["numeric"].get(c, 0.0))

    pool = Pool(X, cat_features=cat_idx)

    # predict all at once
    pred_lat = {t: m.predict(pool) for t, m in models.items()}
    pred_recon = recon_model.predict(pool) if recon_model is not None else None

    pred_table = []
    fs = int(wl.get("failure_state", 0) or 0)

    for i, pol in enumerate(policies):
        pred = {"policy_name": pol}

        for t in LAT_TARGETS:
            if t in pred_lat:
                yhat = float(pred_lat[t][i])
                pred[t] = float(np.expm1(yhat)) if use_log else yhat

        if fs == 1 and pred_recon is not None:
            yhat = float(pred_recon[i])
            pred[RECON_TARGET] = float(np.expm1(yhat)) if use_log else yhat
        else:
            pred[RECON_TARGET] = None

        pred["score"] = compute_score(pred, weights)
        pred_table.append(pred)

    pred_df = pd.DataFrame(pred_table).sort_values("score")
    best = pred_df.iloc[0].to_dict()

    print(f"=== Candidate predictions (sorted by score) | transform={transform} ===")
    print(pred_df.to_string(index=False))
    print("\n=== Recommended policy ===")
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
