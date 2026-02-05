#!/usr/bin/env python3
"""
rf_recommender.py

What this script does:
1) Trains 5 RandomForestRegressor models (one per target) on dataset_30k.csv
2) Uses the trained models to RECOMMEND an EC policy (EC-2-2 / EC-4-2 / EC-6-3)
3) Evaluates policy recommendation quality on the 20% holdout set:
   - Top-1 accuracy
   - Top-3 accuracy
   - actual_regret_mean / median / p95
4) Saves:
   - regret distribution plot (hist)
   - regret CDF plot

Run:
  python rf_recommender.py

Artifacts saved in: rf_artifacts/
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

import matplotlib.pyplot as plt


# =========================
# CONFIG
# =========================
CSV_PATH = "dataset_30k.csv"
ARTIFACT_DIR = "rf_artifacts"

PREPROCESSOR_PATH = os.path.join(ARTIFACT_DIR, "preprocessor.joblib")
MODELS_PATH       = os.path.join(ARTIFACT_DIR, "rf_models.joblib")
RANGES_PATH       = os.path.join(ARTIFACT_DIR, "target_ranges.joblib")
METRICS_PATH      = os.path.join(ARTIFACT_DIR, "metrics.json")
WEIGHTS_PATH      = os.path.join(ARTIFACT_DIR, "weights.json")

RECOMMENDER_REPORT_PATH = os.path.join(ARTIFACT_DIR, "recommender_report.json")
REGRET_HIST_PATH        = os.path.join(ARTIFACT_DIR, "regret_distribution.png")
REGRET_CDF_PATH         = os.path.join(ARTIFACT_DIR, "regret_cdf.png")

SEED = 42
TEST_SIZE = 0.2

N_TREES = 500
TREE_STEP = 25
N_JOBS = -1

# Force retrain if you want
FORCE_RETRAIN = False

# =========================
# TARGETS + POLICIES
# =========================
TARGETS = [
    "avg_upload_ms",
    "p95_upload_ms",
    "avg_download_ms",
    "p95_download_ms",
    "reconstruction_time_ms",
]

POLICY_SPECS = {
    "EC-2-2": (2, 2),
    "EC-4-2": (4, 2),
    "EC-6-3": (6, 3),
}

DEFAULT_WEIGHTS = {
    "avg_upload_ms": 0.15,
    "p95_upload_ms": 0.20,
    "avg_download_ms": 0.15,
    "p95_download_ms": 0.20,
    "reconstruction_time_ms": 0.20,
    # storage terms
    "storage_overhead": 0.07,
    "storage_efficiency": 0.03,
}


# =========================
# Helpers
# =========================
def regression_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))  # version-safe
    r2 = r2_score(y_true, y_pred)

    eps = 1e-9
    mask = np.abs(y_true) > 1e-6
    if mask.any():
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / (y_true[mask] + eps))) * 100.0)
    else:
        mape = float("nan")

    abs_err = np.abs(y_true - y_pred)
    acc_10pct = float(np.mean(abs_err <= 0.10 * (np.abs(y_true) + eps)) * 100.0)
    acc_5ms = float(np.mean(abs_err <= 5.0) * 100.0)

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "MAPE_pct": float(mape),
        "Accuracy@10pct": acc_10pct,
        "Accuracy@5ms": acc_5ms,
    }


def storage_overhead(k, m):
    return (k + m) / k


def storage_efficiency(k, m):
    return k / (k + m)


def normalize(x, xmin, xmax):
    if np.isclose(xmax, xmin):
        return 0.0
    return (x - xmin) / (xmax - xmin)


def compute_target_ranges(y_train: pd.DataFrame):
    ranges = {}
    for t in TARGETS:
        v = y_train[t].astype(float).values
        vmin, vmax = float(np.nanmin(v)), float(np.nanmax(v))
        if np.isclose(vmin, vmax):
            vmax = vmin + 1.0
        ranges[t] = {"min": vmin, "max": vmax}
    return ranges


def build_preprocessor(X: pd.DataFrame):
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )


def print_model_metrics(metrics: dict):
    print("\nEvaluation (20% holdout) — per-target regression:")
    for t in TARGETS:
        m = metrics.get(t, {})
        print(
            f"  {t:24s} "
            f"MAE={m.get('MAE', float('nan')):.3f}  "
            f"RMSE={m.get('RMSE', float('nan')):.3f}  "
            f"R2={m.get('R2', float('nan')):.3f}  "
            f"MAPE={m.get('MAPE_pct', float('nan')):.2f}%  "
            f"Acc@10%={m.get('Accuracy@10pct', float('nan')):.2f}%  "
            f"Acc@5ms={m.get('Accuracy@5ms', float('nan')):.2f}%"
        )


def score_policy_from_targets(
    policy_name: str,
    target_values: dict,
    target_ranges: dict,
    weights: dict,
    candidate_policies: list,
) -> float:
    """
    Lower score is better (same convention as your original code).
    """
    score = 0.0

    # normalized performance targets
    for t in TARGETS:
        n = normalize(float(target_values[t]), target_ranges[t]["min"], target_ranges[t]["max"])
        score += weights.get(t, 0.0) * n

    # storage terms
    k, m = POLICY_SPECS[policy_name]
    ov = storage_overhead(k, m)
    eff = storage_efficiency(k, m)

    ovs = [storage_overhead(*POLICY_SPECS[p]) for p in candidate_policies]
    efs = [storage_efficiency(*POLICY_SPECS[p]) for p in candidate_policies]
    ov_n = normalize(ov, min(ovs), max(ovs))
    eff_n = normalize(eff, min(efs), max(efs))

    score += weights.get("storage_overhead", 0.0) * ov_n
    score -= weights.get("storage_efficiency", 0.0) * eff_n

    return float(score)


def align_user_input_to_training_schema(preprocessor: ColumnTransformer, X_one: pd.DataFrame) -> pd.DataFrame:
    expected = list(preprocessor.feature_names_in_)
    for col in expected:
        if col not in X_one.columns:
            X_one[col] = 0
    X_one = X_one[expected]
    for col in expected:
        if X_one[col].dtype == object:
            X_one[col] = X_one[col].fillna("missing").astype(str)
    return X_one


# =========================
# Train / Load
# =========================
def load_or_train():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    # load cached if allowed
    if (not FORCE_RETRAIN) and os.path.exists(PREPROCESSOR_PATH) and os.path.exists(MODELS_PATH) and os.path.exists(RANGES_PATH):
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        models = joblib.load(MODELS_PATH)
        target_ranges = joblib.load(RANGES_PATH)

        weights = dict(DEFAULT_WEIGHTS)
        if os.path.exists(WEIGHTS_PATH):
            with open(WEIGHTS_PATH, "r") as f:
                weights = json.load(f)

        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, "r") as f:
                metrics = json.load(f)
            print("✅ Loaded trained artifacts from:", ARTIFACT_DIR)
            print_model_metrics(metrics)
        else:
            print("✅ Loaded trained artifacts from:", ARTIFACT_DIR)

        return preprocessor, models, target_ranges, weights

    # train
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Couldn't find {CSV_PATH}. Put it in this folder or edit CSV_PATH.")

    df = pd.read_csv(CSV_PATH)

    missing = [t for t in TARGETS if t not in df.columns]
    if missing:
        raise ValueError(f"Missing target columns in CSV: {missing}")

    if "reconstruction_time_ms" in df.columns:
        df["reconstruction_time_ms"] = df["reconstruction_time_ms"].fillna(0)

    drop_cols = [c for c in ["timestamp", "run_id"] if c in df.columns]
    y = df[TARGETS].copy()
    X = df.drop(columns=TARGETS + drop_cols, errors="ignore").copy()

    # if you have ec_k/ec_m but also policy_name, drop ec_k/ec_m
    for c in ["ec_k", "ec_m"]:
        if c in X.columns and "policy_name" in X.columns:
            X = X.drop(columns=[c])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )

    preprocessor = build_preprocessor(X_train)
    Xt_train = preprocessor.fit_transform(X_train)
    Xt_test = preprocessor.transform(X_test)

    models = {}
    metrics = {}

    print("\n🚀 Training Random Forest models (warm_start) ...\n")

    for t in tqdm(TARGETS, desc="Targets", unit="target"):
        rf = RandomForestRegressor(
            n_estimators=TREE_STEP,
            random_state=SEED,
            n_jobs=N_JOBS,
            min_samples_leaf=2,
            warm_start=True,
        )

        for n in tqdm(range(TREE_STEP, N_TREES + 1, TREE_STEP), desc=f"  Trees ({t})", unit="trees", leave=False):
            rf.set_params(n_estimators=n)
            rf.fit(Xt_train, y_train[t].values)

        pred = rf.predict(Xt_test)
        metrics[t] = regression_report(y_test[t].values, pred)
        models[t] = rf

    target_ranges = compute_target_ranges(y_train)

    # save artifacts
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    joblib.dump(models, MODELS_PATH)
    joblib.dump(target_ranges, RANGES_PATH)

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    with open(WEIGHTS_PATH, "w") as f:
        json.dump(DEFAULT_WEIGHTS, f, indent=2)

    print("\n✅ Trained + tested once, saved artifacts to:", ARTIFACT_DIR)
    print_model_metrics(metrics)

    return preprocessor, models, target_ranges, dict(DEFAULT_WEIGHTS)


# =========================
# Recommendation (single context)
# =========================
def predict_all_policies(preprocessor, models, context_features: dict) -> dict:
    """
    Returns {policy_name: {target: pred, ...}, ...}
    """
    preds = {}
    for pol in POLICY_SPECS.keys():
        row = dict(context_features)
        row["policy_name"] = pol
        X_one = pd.DataFrame([row])
        X_one = align_user_input_to_training_schema(preprocessor, X_one)
        Xt = preprocessor.transform(X_one)
        preds[pol] = {t: float(models[t].predict(Xt)[0]) for t in TARGETS}
    return preds


def rank_policies_by_score(preds_by_policy: dict, target_ranges: dict, weights: dict) -> list:
    """
    Returns list of dicts sorted by score asc: [{policy_name, score, predicted}, ...]
    """
    candidate_policies = list(POLICY_SPECS.keys())
    ranked = []
    for pol, tvals in preds_by_policy.items():
        s = score_policy_from_targets(pol, tvals, target_ranges, weights, candidate_policies)
        ranked.append({"policy_name": pol, "score": float(s), "predicted": tvals})
    ranked.sort(key=lambda x: x["score"])
    return ranked


# =========================
# Recommendation quality eval on holdout
# =========================
def evaluate_recommender(df_all: pd.DataFrame, preprocessor, models, target_ranges, weights) -> dict:
    """
    We evaluate on the same holdout split style, BUT recommendation quality requires
    comparing policies under identical "workload context".

    So we:
    - split rows into train/test with same random seed
    - on TEST rows, group by "context columns" = all feature columns EXCEPT policy_name
    - require that each context group contains ALL policies (EC-2-2, EC-4-2, EC-6-3)
    - for each context:
        * predicted ranking (by predicted scores)
        * actual ranking (by actual scores computed from true targets)
        * Top-1 accuracy: predicted_best == actual_best
        * Top-3 accuracy: actual_best is inside predicted_top3
        * regret: actual_score(predicted_best) - actual_score(actual_best)
    """
    candidate_policies = list(POLICY_SPECS.keys())

    # prepare split
    drop_cols = [c for c in ["timestamp", "run_id"] if c in df_all.columns]
    df = df_all.drop(columns=drop_cols, errors="ignore").copy()

    # optional: fill recon time
    if "reconstruction_time_ms" in df.columns:
        df["reconstruction_time_ms"] = df["reconstruction_time_ms"].fillna(0)

    # create "features only" list
    feature_cols = [c for c in df.columns if c not in TARGETS]
    if "policy_name" not in feature_cols:
        raise ValueError("policy_name must exist as a feature column for policy recommendation evaluation.")

    # split by rows (same as training split)
    idx = np.arange(len(df))
    idx_train, idx_test = train_test_split(idx, test_size=TEST_SIZE, random_state=SEED)

    df_test = df.iloc[idx_test].copy()

    # context columns = all features except policy_name
    context_cols = [c for c in feature_cols if c != "policy_name"]

    # group test rows by context
    regrets = []
    top1_hits = 0
    top3_hits = 0
    used_contexts = 0
    skipped_contexts = 0

    # Make grouping stable with NaNs handled
    df_test_groupable = df_test.copy()
    for c in context_cols:
        if df_test_groupable[c].dtype == object:
            df_test_groupable[c] = df_test_groupable[c].fillna("missing").astype(str)
        else:
            # keep numeric NaNs; groupby drops NaNs by default in some cases
            # so we fill with a sentinel for grouping only
            if df_test_groupable[c].isna().any():
                df_test_groupable[c] = df_test_groupable[c].fillna(-999999.0)

    grouped = df_test_groupable.groupby(context_cols, dropna=False)

    for _, g in tqdm(grouped, desc="Evaluating contexts", unit="ctx"):
        # ensure all policies present
        present = set(g["policy_name"].astype(str).tolist())
        if any(p not in present for p in candidate_policies):
            skipped_contexts += 1
            continue

        used_contexts += 1

        # reconstruct a single context dict from the group (take first row values)
        context_features = {c: g.iloc[0][c] for c in context_cols}

        # if we used numeric sentinel for grouping, restore NaN for model input
        for c in context_cols:
            if isinstance(context_features[c], (float, np.floating)) and context_features[c] == -999999.0:
                context_features[c] = np.nan

        # predicted ranking
        preds_by_policy = predict_all_policies(preprocessor, models, context_features)
        pred_ranked = rank_policies_by_score(preds_by_policy, target_ranges, weights)
        pred_best = pred_ranked[0]["policy_name"]
        pred_top3 = [x["policy_name"] for x in pred_ranked[:3]]

        # actual ranking using true targets from g
        actual_scores = {}
        for pol in candidate_policies:
            row_pol = g[g["policy_name"].astype(str) == pol].iloc[0]
            actual_targets = {t: float(row_pol[t]) for t in TARGETS}
            actual_scores[pol] = score_policy_from_targets(pol, actual_targets, target_ranges, weights, candidate_policies)

        actual_ranked = sorted(actual_scores.items(), key=lambda kv: kv[1])  # (policy, score)
        actual_best = actual_ranked[0][0]
        actual_best_score = float(actual_ranked[0][1])
        actual_predbest_score = float(actual_scores[pred_best])

        # metrics
        if pred_best == actual_best:
            top1_hits += 1
        if actual_best in pred_top3:
            top3_hits += 1

        regret = actual_predbest_score - actual_best_score
        regrets.append(float(regret))

    regrets = np.array(regrets, dtype=float)
    if used_contexts == 0:
        raise RuntimeError("No valid contexts found in the test split that contain all policies.")

    report = {
        "n_contexts_used": int(used_contexts),
        "n_contexts_skipped_missing_policies": int(skipped_contexts),
        "top1_accuracy": float(top1_hits / used_contexts),
        "top3_accuracy": float(top3_hits / used_contexts),
        "actual_regret_mean": float(np.mean(regrets)),
        "actual_regret_median": float(np.median(regrets)),
        "actual_regret_p95": float(np.percentile(regrets, 95)),
    }

    # save plots
    save_regret_plots(regrets)

    # persist report
    with open(RECOMMENDER_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    return report


def save_regret_plots(regrets: np.ndarray):
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    # --- Regret distribution (hist)
    plt.figure()
    plt.hist(regrets, bins=40)
    plt.xlabel("Actual Regret (score(pred_best) - score(actual_best))")
    plt.ylabel("Count")
    plt.title("Regret Distribution (Holdout Contexts)")
    plt.tight_layout()
    plt.savefig(REGRET_HIST_PATH, dpi=200)
    plt.close()

    # --- Regret CDF
    r = np.sort(regrets)
    y = np.arange(1, len(r) + 1) / len(r)

    plt.figure()
    plt.plot(r, y)
    plt.xlabel("Actual Regret")
    plt.ylabel("CDF")
    plt.title("Regret CDF (Holdout Contexts)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(REGRET_CDF_PATH, dpi=200)
    plt.close()


# =========================
# Main
# =========================
def main():
    preprocessor, models, target_ranges, weights = load_or_train()

    df = pd.read_csv(CSV_PATH)

    # Evaluate recommendation quality and print the exact items you asked for
    rec_report = evaluate_recommender(df, preprocessor, models, target_ranges, weights)

    print("\n========================================")
    print("Policy recommendation quality (RF)")
    print("========================================")
    print(f"Top-1 accuracy = {rec_report['top1_accuracy']:.4f}")
    print(f"Top-3 accuracy = {rec_report['top3_accuracy']:.4f}")
    print(f"actual_regret_mean   ≈ {rec_report['actual_regret_mean']:.6f}")
    print(f"actual_regret_median = {rec_report['actual_regret_median']:.6f}")
    print(f"actual_regret_p95    ≈ {rec_report['actual_regret_p95']:.6f}")

    print("\nSaved plots:")
    print(f"  - Regret distribution: {REGRET_HIST_PATH}")
    print(f"  - Regret CDF:          {REGRET_CDF_PATH}")
    print(f"\nSaved report json: {RECOMMENDER_REPORT_PATH}\n")


if __name__ == "__main__":
    main()
