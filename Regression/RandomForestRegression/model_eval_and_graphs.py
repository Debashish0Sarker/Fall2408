# make_plots_eval.py
# ---------------------------------------------------------
# ✅ Uses saved artifacts (NO retraining)
# ✅ Re-evaluates on a fresh 20% holdout split (same SEED)
# ✅ Generates graphs:
#    - Predicted vs Actual (per target)
#    - Residuals vs Predicted + Residual histogram (per target)
#    - Metrics summary bars
#    - Feature importance for ALL metrics (aggregated back to original columns)
#    - Demo scatter plot (illustrative)
#
# ✅ FIXED: Handles NaN/inf in targets safely (prevents your error)
# ---------------------------------------------------------

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error


# =========================
# CONFIG
# =========================
CSV_PATH = "dataset_30k.csv"
ARTIFACT_DIR = "rf_artifacts"

PREPROCESSOR_PATH = os.path.join(ARTIFACT_DIR, "preprocessor.joblib")
MODELS_PATH       = os.path.join(ARTIFACT_DIR, "rf_models.joblib")

# Where plots will be saved
PLOTS_DIR = os.path.join(ARTIFACT_DIR, "plots_eval")
os.makedirs(PLOTS_DIR, exist_ok=True)

SEED = 42
TEST_SIZE = 0.2

TARGETS = [
    "avg_upload_ms",
    "p95_upload_ms",
    "avg_download_ms",
    "p95_download_ms",
    "reconstruction_time_ms",
]

# -------------------------------------------------
# If your models were trained using log1p(target),
# set LOG1P_TRAINED=True so predictions are inverted
# back to ms for plots/metrics.
# -------------------------------------------------
LOG1P_TRAINED = False          # <-- set True only if you trained on log1p(y)
LOG1P_TARGETS = set(TARGETS)   # if only some targets are log1p, put them here


# =========================
# UTILS
# =========================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def inv_target_transform(y_pred, target_name):
    if LOG1P_TRAINED and (target_name in LOG1P_TARGETS):
        return np.expm1(y_pred)  # exp(y') - 1
    return y_pred


# =========================
# PLOTTING
# =========================
def plot_pred_vs_actual(y_true_ms, y_pred_ms, target, outdir, title_suffix=""):
    ensure_dir(outdir)

    y_true_ms = np.asarray(y_true_ms, dtype=float)
    y_pred_ms = np.asarray(y_pred_ms, dtype=float)

    # ✅ keep only finite values (avoid matplotlib warnings + empty plots)
    mask = np.isfinite(y_true_ms) & np.isfinite(y_pred_ms)
    y_true_ms = y_true_ms[mask]
    y_pred_ms = y_pred_ms[mask]
    if y_true_ms.size == 0:
        return

    plt.figure(figsize=(8, 5))
    plt.scatter(y_true_ms, y_pred_ms, s=10, alpha=0.6)
    lo = float(np.nanmin([y_true_ms.min(), y_pred_ms.min()]))
    hi = float(np.nanmax([y_true_ms.max(), y_pred_ms.max()]))
    plt.plot([lo, hi], [lo, hi])  # y=x
    plt.xlabel("Actual (ms)")
    plt.ylabel("Predicted (ms)")
    plt.title(f"Predicted vs Actual: {target}{title_suffix}")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{target}_pred_vs_actual.png"), dpi=220)
    plt.close()

def plot_residuals(y_true_ms, y_pred_ms, target, outdir):
    ensure_dir(outdir)

    y_true_ms = np.asarray(y_true_ms, dtype=float)
    y_pred_ms = np.asarray(y_pred_ms, dtype=float)

    mask = np.isfinite(y_true_ms) & np.isfinite(y_pred_ms)
    y_true_ms = y_true_ms[mask]
    y_pred_ms = y_pred_ms[mask]
    if y_true_ms.size == 0:
        return

    resid = y_true_ms - y_pred_ms

    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred_ms, resid, s=10, alpha=0.6)
    plt.axhline(0.0)
    plt.xlabel("Predicted (ms)")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title(f"Residuals vs Predicted: {target}")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{target}_residuals_vs_pred.png"), dpi=220)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(resid, bins=50)
    plt.xlabel("Residual (ms)")
    plt.ylabel("Count")
    plt.title(f"Residual Distribution: {target}")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{target}_residual_hist.png"), dpi=220)
    plt.close()

def plot_metrics_summary(metrics_by_target, outdir):
    ensure_dir(outdir)

    keys = ["MAE", "RMSE", "R2", "MAPE_pct", "Accuracy@10pct", "Accuracy@5ms", "n_eval"]
    targets = list(metrics_by_target.keys())

    for k in keys:
        vals = [metrics_by_target[t].get(k, np.nan) for t in targets]
        plt.figure(figsize=(10, 4))
        plt.bar(targets, vals)
        plt.xticks(rotation=30, ha="right")
        plt.ylabel(k)
        plt.title(f"{k} by Target (Holdout Re-eval)")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"metrics_{k}.png"), dpi=220)
        plt.close()

def demo_scatter_plot(outdir):
    """
    Illustrative demo scatter showing increased dispersion at tail.
    """
    ensure_dir(outdir)
    rng = np.random.RandomState(42)
    n = 1200
    actual = rng.gamma(shape=2.0, scale=60.0, size=n)  # right-skewed
    noise = rng.normal(loc=0.0, scale=0.12 * actual + 5.0, size=n)
    pred = np.clip(actual + noise, 0, None)

    plt.figure(figsize=(8, 5))
    plt.scatter(actual, pred, s=10, alpha=0.6)
    lo = float(np.min([actual.min(), pred.min()]))
    hi = float(np.max([actual.max(), pred.max()]))
    plt.plot([lo, hi], [lo, hi])
    plt.xlabel("Actual (ms)")
    plt.ylabel("Predicted (ms)")
    plt.title("Demo: Predicted vs Actual (Increasing Dispersion at Tail)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "DEMO_pred_vs_actual.png"), dpi=220)
    plt.close()


# =========================
# METRICS (✅ NaN-safe)
# =========================
def compute_metrics_ms(y_true_ms, y_pred_ms):
    y_true_ms = np.asarray(y_true_ms, dtype=float)
    y_pred_ms = np.asarray(y_pred_ms, dtype=float)

    # ✅ Drop NaN/inf rows
    mask = np.isfinite(y_true_ms) & np.isfinite(y_pred_ms)
    y_true_ms = y_true_ms[mask]
    y_pred_ms = y_pred_ms[mask]

    if y_true_ms.size == 0:
        return {
            "MAE": float("nan"),
            "RMSE": float("nan"),
            "R2": float("nan"),
            "MAPE_pct": float("nan"),
            "Accuracy@10pct": float("nan"),
            "Accuracy@5ms": float("nan"),
            "n_eval": 0,
        }

    mae = float(mean_absolute_error(y_true_ms, y_pred_ms))
    _rmse = rmse(y_true_ms, y_pred_ms)
    r2 = float(r2_score(y_true_ms, y_pred_ms))

    eps = 1e-9
    mask2 = np.abs(y_true_ms) > 1e-6
    if mask2.any():
        mape = float(np.mean(np.abs((y_true_ms[mask2] - y_pred_ms[mask2]) / (y_true_ms[mask2] + eps))) * 100.0)
    else:
        mape = float("nan")

    abs_err = np.abs(y_true_ms - y_pred_ms)
    acc_10pct = float(np.mean(abs_err <= 0.10 * (np.abs(y_true_ms) + eps)) * 100.0)
    acc_5ms = float(np.mean(abs_err <= 5.0) * 100.0)

    return {
        "MAE": mae,
        "RMSE": _rmse,
        "R2": r2,
        "MAPE_pct": mape,
        "Accuracy@10pct": acc_10pct,
        "Accuracy@5ms": acc_5ms,
        "n_eval": int(y_true_ms.size),
    }


# =========================
# FEATURE IMPORTANCE (aggregated)
# =========================
def build_transformed_feature_to_original_map(preprocessor):
    """
    Map each transformed feature index back to original feature name.
    Works for ColumnTransformer with:
      - num: imputer+scaler -> one output per numeric column
      - cat: imputer+onehot -> many outputs per categorical column
    """
    orig_names = []

    for name, trans, cols in preprocessor.transformers_:
        if name == "remainder" and trans == "drop":
            continue
        if cols is None:
            continue

        if name == "num":
            for c in cols:
                orig_names.append(str(c))

        elif name == "cat":
            ohe = None
            if hasattr(trans, "named_steps"):
                ohe = trans.named_steps.get("onehot", None)

            if ohe is None:
                # best-effort fallback
                for c in cols:
                    orig_names.append(str(c))
                continue

            ohe_names = ohe.get_feature_names_out(cols)
            for feat in ohe_names:
                feat = str(feat)
                matched = None
                for c in cols:
                    prefix = str(c) + "_"
                    if feat.startswith(prefix):
                        matched = str(c)
                        break
                orig_names.append(matched if matched is not None else str(cols[0]))

        else:
            # unknown transformer: best-effort
            if isinstance(cols, (list, tuple, np.ndarray)):
                for c in cols:
                    orig_names.append(str(c))
            else:
                orig_names.append(str(cols))

    return orig_names

def plot_aggregated_feature_importance(models, preprocessor, outdir, top_n=25):
    ensure_dir(outdir)

    idx_to_orig = np.array(build_transformed_feature_to_original_map(preprocessor), dtype=object)

    combined_rows = []

    for target, model in models.items():
        if not hasattr(model, "feature_importances_"):
            continue

        imp = np.asarray(model.feature_importances_, dtype=float)

        # If mismatch, try fallback using feature_names_out()
        if imp.shape[0] != idx_to_orig.shape[0]:
            try:
                tnames = preprocessor.get_feature_names_out()
                idx_to_orig2 = []
                for n in tnames:
                    n = str(n).split("__", 1)[-1]  # remove "num__" / "cat__"
                    idx_to_orig2.append(n.split("_", 1)[0])  # best-effort
                idx_to_orig = np.array(idx_to_orig2, dtype=object)
            except Exception:
                idx_to_orig = np.array([f"f{i}" for i in range(len(imp))], dtype=object)

        df_imp = pd.DataFrame({"orig_feature": idx_to_orig, "importance": imp})
        agg = df_imp.groupby("orig_feature", dropna=False)["importance"].sum().sort_values(ascending=False)

        # save CSV per target
        agg.reset_index().to_csv(
            os.path.join(outdir, f"{target}_feature_importance_aggregated.csv"),
            index=False
        )

        # plot top N
        top = agg.head(top_n)
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(top))[::-1], top.values)
        plt.yticks(range(len(top))[::-1], top.index.astype(str))
        plt.xlabel("Aggregated Importance")
        plt.title(f"Top {top_n} Feature Importances (Aggregated): {target}")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{target}_feature_importance_aggregated_top{top_n}.png"), dpi=220)
        plt.close()

        # combined table
        for rank, (fname, val) in enumerate(agg.items(), start=1):
            combined_rows.append({
                "target": target,
                "rank": rank,
                "feature": str(fname),
                "importance": float(val),
            })

    if combined_rows:
        pd.DataFrame(combined_rows).to_csv(
            os.path.join(outdir, "ALL_targets_feature_importance_aggregated.csv"),
            index=False
        )


# =========================
# MAIN
# =========================
def main():
    # 1) Load artifacts (NO training)
    if not os.path.exists(PREPROCESSOR_PATH) or not os.path.exists(MODELS_PATH):
        raise FileNotFoundError(
            "Missing artifacts. Expected:\n"
            "  rf_artifacts/preprocessor.joblib\n"
            "  rf_artifacts/rf_models.joblib\n"
            "Run your training script once to create them."
        )

    preprocessor = joblib.load(PREPROCESSOR_PATH)
    models = joblib.load(MODELS_PATH)

    # 2) Load dataset
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Missing CSV: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    missing = [t for t in TARGETS if t not in df.columns]
    if missing:
        raise ValueError(f"CSV missing targets: {missing}")

    # Optional: match your training logic (only recon got fillna in your training code)
    # But for eval we keep NaNs and filter them per target (safer + honest).
    # If you prefer fillna, uncomment:
    # for t in TARGETS:
    #     df[t] = df[t].fillna(0)

    # 3) Prepare X/y like your training script
    drop_cols = [c for c in ["timestamp", "run_id"] if c in df.columns]
    y = df[TARGETS].copy()
    X = df.drop(columns=TARGETS + drop_cols, errors="ignore").copy()

    for c in ["ec_k", "ec_m"]:
        if c in X.columns and "policy_name" in X.columns:
            X = X.drop(columns=[c])

    # 4) Holdout split (same SEED)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )

    # 5) Transform test set with saved preprocessor
    Xt_test = preprocessor.transform(X_test)

    # 6) Predict + plot + metrics
    metrics_by_target = {}
    title_suffix = " (log1p-trained)" if LOG1P_TRAINED else ""

    for t in TARGETS:
        model = models[t]
        y_true_ms = y_test[t].values.astype(float)

        y_pred_model_space = model.predict(Xt_test)
        y_pred_ms = inv_target_transform(y_pred_model_space, t)

        # plots
        plot_pred_vs_actual(y_true_ms, y_pred_ms, t, PLOTS_DIR, title_suffix=title_suffix)
        plot_residuals(y_true_ms, y_pred_ms, t, PLOTS_DIR)

        # metrics (NaN-safe)
        metrics_by_target[t] = compute_metrics_ms(y_true_ms, y_pred_ms)

    # 7) Metrics summary plots + json
    plot_metrics_summary(metrics_by_target, PLOTS_DIR)
    with open(os.path.join(PLOTS_DIR, "metrics_recalc.json"), "w") as f:
        json.dump(metrics_by_target, f, indent=2)

    # 8) Feature importance (aggregated) for ALL targets
    plot_aggregated_feature_importance(models, preprocessor, PLOTS_DIR, top_n=25)

    # 9) Demo scatter plot (for report explanation)
    demo_scatter_plot(PLOTS_DIR)

    print("\n✅ Done. Saved all graphs to:")
    print("   ", PLOTS_DIR)
    print("\nFiles you will see:")
    print(" - *_pred_vs_actual.png")
    print(" - *_residuals_vs_pred.png")
    print(" - *_residual_hist.png")
    print(" - metrics_*.png")
    print(" - *_feature_importance_aggregated_top25.png")
    print(" - ALL_targets_feature_importance_aggregated.csv")
    print(" - DEMO_pred_vs_actual.png\n")


if __name__ == "__main__":
    main()
