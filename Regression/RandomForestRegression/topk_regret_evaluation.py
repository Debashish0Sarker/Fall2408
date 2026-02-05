# make_regret_eval.py
# ---------------------------------------------------------
# ✅ Uses saved artifacts (NO retraining)
# ✅ Computes Top-1 / Top-3 accuracy + regret mean/median/p95
# ✅ Creates:
#    - regret_distribution.png
#    - regret_cdf.png
# ✅ Saves to: rf_artifacts/regret/
# ---------------------------------------------------------

import os
import json
import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
CSV_PATH = "dataset_30k.csv"
ARTIFACT_DIR = "rf_artifacts"

PREPROCESSOR_PATH = os.path.join(ARTIFACT_DIR, "preprocessor.joblib")
MODELS_PATH       = os.path.join(ARTIFACT_DIR, "rf_models.joblib")
RANGES_PATH       = os.path.join(ARTIFACT_DIR, "target_ranges.joblib")
WEIGHTS_PATH      = os.path.join(ARTIFACT_DIR, "weights.json")

# ✅ NEW output folder
OUT_DIR = os.path.join(ARTIFACT_DIR, "regret")
os.makedirs(OUT_DIR, exist_ok=True)

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
    "storage_overhead": 0.07,
    "storage_efficiency": 0.03,
}

DROP_COLS_ALWAYS = {"timestamp", "run_id"}  # ignore if present


# =========================
# Helpers
# =========================
def storage_overhead(k, m):
    return (k + m) / k

def storage_efficiency(k, m):
    return k / (k + m)

def normalize(x, xmin, xmax):
    if np.isclose(xmax, xmin):
        return 0.0
    return (x - xmin) / (xmax - xmin)

def align_user_input_to_training_schema(preprocessor, X_any: pd.DataFrame) -> pd.DataFrame:
    expected = list(preprocessor.feature_names_in_)
    for col in expected:
        if col not in X_any.columns:
            X_any[col] = 0
    X_any = X_any[expected]
    for col in expected:
        if X_any[col].dtype == object:
            X_any[col] = X_any[col].fillna("missing").astype(str)
    return X_any

def load_artifacts():
    for p in [PREPROCESSOR_PATH, MODELS_PATH, RANGES_PATH]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing artifact: {p}\nTrain first using your rf.py.")
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    models = joblib.load(MODELS_PATH)
    target_ranges = joblib.load(RANGES_PATH)

    weights = dict(DEFAULT_WEIGHTS)
    if os.path.exists(WEIGHTS_PATH):
        with open(WEIGHTS_PATH, "r") as f:
            weights = json.load(f)

    return preprocessor, models, target_ranges, weights

def compute_true_score_for_row(row_targets: pd.Series, policy_name: str, target_ranges: dict, weights: dict) -> float:
    score = 0.0

    for t in TARGETS:
        v = float(row_targets[t])
        n = normalize(v, target_ranges[t]["min"], target_ranges[t]["max"])
        score += float(weights.get(t, 0.0)) * n

    policies = list(POLICY_SPECS.keys())
    ovs = [storage_overhead(*POLICY_SPECS[p]) for p in policies]
    efs = [storage_efficiency(*POLICY_SPECS[p]) for p in policies]
    ov_min, ov_max = min(ovs), max(ovs)
    ef_min, ef_max = min(efs), max(efs)

    k, m = POLICY_SPECS[policy_name]
    ov = storage_overhead(k, m)
    eff = storage_efficiency(k, m)

    ov_n = normalize(ov, ov_min, ov_max)
    eff_n = normalize(eff, ef_min, ef_max)

    score += float(weights.get("storage_overhead", 0.0)) * ov_n
    score -= float(weights.get("storage_efficiency", 0.0)) * eff_n

    return float(score)

def predict_scores_for_contexts(preprocessor, models, contexts_base: pd.DataFrame, target_ranges: dict, weights: dict) -> pd.DataFrame:
    policies = list(POLICY_SPECS.keys())
    n_ctx = len(contexts_base)

    ovs = [storage_overhead(*POLICY_SPECS[p]) for p in policies]
    efs = [storage_efficiency(*POLICY_SPECS[p]) for p in policies]
    ov_min, ov_max = min(ovs), max(ovs)
    ef_min, ef_max = min(efs), max(efs)

    out = pd.DataFrame(index=contexts_base.index)

    for pol in policies:
        Xp = contexts_base.copy()
        Xp["policy_name"] = pol
        Xp = align_user_input_to_training_schema(preprocessor, Xp)
        Xt = preprocessor.transform(Xp)

        score = np.zeros(n_ctx, dtype=float)

        for t in TARGETS:
            yhat = models[t].predict(Xt).astype(float)
            denom = (target_ranges[t]["max"] - target_ranges[t]["min"]) + 1e-12
            n = (yhat - target_ranges[t]["min"]) / denom
            n = np.clip(n, 0.0, 1.0)
            score += float(weights.get(t, 0.0)) * n

        k, m = POLICY_SPECS[pol]
        ov = storage_overhead(k, m)
        eff = storage_efficiency(k, m)

        ov_n = normalize(ov, ov_min, ov_max)
        eff_n = normalize(eff, ef_min, ef_max)

        score += float(weights.get("storage_overhead", 0.0)) * ov_n
        score -= float(weights.get("storage_efficiency", 0.0)) * eff_n

        out[pol] = score

    return out

def make_regret_plots(regrets: np.ndarray, out_dir: str):
    regrets = np.asarray(regrets, dtype=float)
    regrets = regrets[np.isfinite(regrets)]
    if regrets.size == 0:
        raise RuntimeError("No finite regret values to plot.")

    # Distribution
    plt.figure()
    plt.hist(regrets, bins=40)
    plt.xlabel("Regret (chosen score − oracle best score)")
    plt.ylabel("Count")
    plt.title("Regret Distribution")
    dist_path = os.path.join(out_dir, "regret_distribution.png")
    plt.tight_layout()
    plt.savefig(dist_path, dpi=220)
    plt.close()

    # CDF
    xs = np.sort(regrets)
    ys = np.arange(1, xs.size + 1) / xs.size

    plt.figure()
    plt.plot(xs, ys)
    plt.xlabel("Regret")
    plt.ylabel("CDF")
    plt.title("Regret CDF")
    cdf_path = os.path.join(out_dir, "regret_cdf.png")
    plt.tight_layout()
    plt.savefig(cdf_path, dpi=220)
    plt.close()

    return dist_path, cdf_path

def get_context_id(df: pd.DataFrame) -> pd.Series:
    # Prefer existing id if present
    for cand in ["context_id", "workload_id", "ctx_id", "context"]:
        if cand in df.columns:
            return df[cand].astype(str)

    drop = set(TARGETS) | {"policy_name"} | DROP_COLS_ALWAYS
    feature_cols = [c for c in df.columns if c not in drop]

    tmp = df[feature_cols].copy()
    for c in tmp.columns:
        if tmp[c].dtype == object:
            tmp[c] = tmp[c].fillna("missing").astype(str)

    h = pd.util.hash_pandas_object(tmp, index=False)
    return h.astype(str)


# =========================
# MAIN
# =========================
def main():
    preprocessor, models, target_ranges, weights = load_artifacts()

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Couldn't find {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    # Checks
    for t in TARGETS:
        if t not in df.columns:
            raise ValueError(f"Missing target column: {t}")
    if "policy_name" not in df.columns:
        raise ValueError("dataset must contain a 'policy_name' column for regret/top-k evaluation.")

    df["reconstruction_time_ms"] = df["reconstruction_time_ms"].fillna(0.0)

    # Build contexts
    df["_context_id"] = get_context_id(df)

    policies = list(POLICY_SPECS.keys())
    df = df[df["policy_name"].isin(policies)].copy()

    # Require all policies per context (real regret/top-k evaluation)
    ctx_counts = df.groupby("_context_id")["policy_name"].nunique()
    good_ctx = ctx_counts[ctx_counts == len(policies)].index
    df = df[df["_context_id"].isin(good_ctx)].copy()

    if df.empty:
        raise RuntimeError(
            "No contexts contain all candidate policies. "
            "Your dataset must have the SAME workload context executed under each policy."
        )

    # Feature matrix like training
    drop_cols = [c for c in ["timestamp", "run_id"] if c in df.columns]
    X_all = df.drop(columns=TARGETS + drop_cols, errors="ignore").copy()

    for c in ["ec_k", "ec_m"]:
        if c in X_all.columns and "policy_name" in X_all.columns:
            X_all = X_all.drop(columns=[c])

    contexts_base = (
        X_all.drop(columns=["policy_name"], errors="ignore")
        .assign(_context_id=df["_context_id"].values)
        .drop_duplicates(subset=["_context_id"])
        .set_index("_context_id")
    )

    # Predict composite scores
    pred_scores = predict_scores_for_contexts(preprocessor, models, contexts_base, target_ranges, weights)

    # True scores from ground-truth targets
    true_scores = []
    for (ctx, pol), g in df.groupby(["_context_id", "policy_name"], sort=False):
        y = g[TARGETS].astype(float).mean()
        s = compute_true_score_for_row(y, pol, target_ranges, weights)
        true_scores.append((ctx, pol, s))

    true_scores_df = pd.DataFrame(true_scores, columns=["context_id", "policy_name", "true_score"])
    true_scores_pivot = true_scores_df.pivot(index="context_id", columns="policy_name", values="true_score")

    common_ctx = pred_scores.index.intersection(true_scores_pivot.index)
    pred_scores = pred_scores.loc[common_ctx]
    true_scores_pivot = true_scores_pivot.loc[common_ctx]

    # Oracle and chosen
    oracle_best = true_scores_pivot.idxmin(axis=1)
    oracle_best_score = true_scores_pivot.min(axis=1)
    chosen = pred_scores.idxmin(axis=1)

    # Top-k accuracy
    top3_pred = pred_scores.apply(lambda r: list(r.nsmallest(3).index), axis=1)
    top1_acc = float((chosen == oracle_best).mean())
    top3_acc = float(np.mean([oracle_best.loc[i] in top3_pred.loc[i] for i in common_ctx]))

    # chosen_true_score (✅ pandas>=2 safe)
    chosen_true_score = []
    for ctx in common_ctx:
        pol = chosen.loc[ctx]
        chosen_true_score.append(true_scores_pivot.loc[ctx, pol])
    chosen_true_score = pd.Series(chosen_true_score, index=common_ctx).astype(float)

    # Regret
    regrets = (chosen_true_score - oracle_best_score).astype(float).values
    regrets = np.clip(regrets, 0.0, None)

    regret_mean = float(np.mean(regrets))
    regret_median = float(np.median(regrets))
    regret_p95 = float(np.percentile(regrets, 95))

    # Save per-context regret table
    out_csv = os.path.join(OUT_DIR, "per_context_regret.csv")
    out_df = pd.DataFrame({
        "context_id": common_ctx,
        "oracle_best_policy": oracle_best.loc[common_ctx].values,
        "chosen_policy": chosen.loc[common_ctx].values,
        "oracle_best_score": oracle_best_score.loc[common_ctx].values,
        "chosen_true_score": chosen_true_score.loc[common_ctx].values,
        "regret": regrets,
    })
    out_df.to_csv(out_csv, index=False)

    # Plots
    dist_path, cdf_path = make_regret_plots(regrets, OUT_DIR)

    # Print required 5 answers
    print("\nFor policy recommendation, Random Forest Regression gives -")
    print(f"Top-1 accuracy = {top1_acc:.4f}")
    print(f"Top-3 accuracy = {top3_acc:.4f}")
    print(f"actual_regret_mean ≈ {regret_mean:.6f}")
    print(f"actual_regret_median = {regret_median:.6f}")
    print(f"actual_regret_p95 ≈ {regret_p95:.6f}")

    print("\n✅ Saved outputs to:", OUT_DIR)
    print("  -", os.path.basename(dist_path))
    print("  -", os.path.basename(cdf_path))
    print("  - per_context_regret.csv")


if __name__ == "__main__":
    main()
