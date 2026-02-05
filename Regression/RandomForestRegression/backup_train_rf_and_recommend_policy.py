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


# =========================
# CONFIG (edit if needed)
# =========================
CSV_PATH = "dataset_30k.csv"
ARTIFACT_DIR = "rf_artifacts"

PREPROCESSOR_PATH = os.path.join(ARTIFACT_DIR, "preprocessor.joblib")
MODELS_PATH       = os.path.join(ARTIFACT_DIR, "rf_models.joblib")
RANGES_PATH       = os.path.join(ARTIFACT_DIR, "target_ranges.joblib")
METRICS_PATH      = os.path.join(ARTIFACT_DIR, "metrics.json")
WEIGHTS_PATH      = os.path.join(ARTIFACT_DIR, "weights.json")
POLICIES_PATH     = os.path.join(ARTIFACT_DIR, "policy_specs.json")  # ✅ NEW

USER_INPUT_PATH = "user_input.json"

SEED = 42
TEST_SIZE = 0.2

N_TREES = 500
TREE_STEP = 25
N_JOBS = -1

# =========================
# ✅ FORCE RETRAIN FLAG
# =========================
FORCE_RETRAIN = True   # change to False to stop forcing retrain


# =========================
# TARGETS
# =========================
TARGETS = [
    "avg_upload_ms",
    "p95_upload_ms",
    "avg_download_ms",
    "p95_download_ms",
    "reconstruction_time_ms",
]

DEFAULT_WEIGHTS = {
    "avg_upload_ms": 0.15,
    "p95_upload_ms": 0.20,
    "avg_download_ms": 0.15,
    "p95_download_ms": 0.20,
    "reconstruction_time_ms": 0.20,
    "storage_overhead": 0.07,
    "storage_efficiency": 0.03,
}


def regression_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
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

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )
    return preprocessor


# =========================
# ✅ NEW: infer ALL policies from dataset
# =========================
def infer_policy_specs(df: pd.DataFrame) -> dict:
    """
    Builds POLICY_SPECS from the dataset.
    Expects columns: policy_name, ec_k, ec_m
    Returns: { "EC-4-2": (4,2), ... } for all unique policies.
    """
    if "policy_name" not in df.columns:
        raise ValueError("dataset missing 'policy_name' column.")
    if "ec_k" not in df.columns or "ec_m" not in df.columns:
        # fallback: parse from name "EC-k-m"
        specs = {}
        for p in df["policy_name"].dropna().astype(str).unique():
            if p.startswith("EC-"):
                parts = p.split("-")
                if len(parts) == 3:
                    k = int(parts[1])
                    m = int(parts[2])
                    specs[p] = (k, m)
        if not specs:
            raise ValueError("Couldn't infer EC policies. Need 'ec_k'/'ec_m' or EC-* policy_name format.")
        return specs

    # primary: group by policy_name and read k/m
    sub = df[["policy_name", "ec_k", "ec_m"]].dropna()
    specs = {}
    for pol, g in sub.groupby("policy_name"):
        k = int(g["ec_k"].iloc[0])
        m = int(g["ec_m"].iloc[0])
        specs[str(pol)] = (k, m)

    if len(specs) < 2:
        raise ValueError(f"Inferred too few policies: {len(specs)}")
    return specs


def ensure_user_input_template(df: pd.DataFrame):
    if os.path.exists(USER_INPUT_PATH):
        return

    template_candidates = [
        ("file_size_mb", 10),
        ("num_objects", 50),
        ("concurrency", 8),
        ("read_ratio", 0.7),
        ("failure_state", 1),
        ("failure_disk", "disk1_down"),
    ]

    template = {}
    feature_cols = [c for c in df.columns if c not in TARGETS]
    feature_cols = [c for c in feature_cols if c not in ["timestamp", "run_id"]]

    for k, v in template_candidates:
        if k in feature_cols:
            template[k] = v

    if len(template) < 3:
        numeric_cols = df.drop(columns=TARGETS, errors="ignore").select_dtypes(include=[np.number]).columns.tolist()
        for c in numeric_cols[:6]:
            if c not in template and c not in ["timestamp", "run_id"]:
                med = df[c].median()
                template[c] = float(med) if np.isfinite(med) else 0.0

    if "failure_disk" in df.columns and df["failure_disk"].dtype == object and "failure_disk" in template:
        vals = df["failure_disk"].dropna().astype(str).unique().tolist()
        if vals:
            template["failure_disk"] = vals[0]

    with open(USER_INPUT_PATH, "w") as f:
        json.dump(template, f, indent=2)

    print(f"\n⚠️  Created {USER_INPUT_PATH}. Edit it with your workload values, then run again.\n")


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


def print_metrics(metrics: dict):
    print("\nEvaluation (20% holdout):")
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


def load_or_train():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    # ✅ If artifacts exist and not forcing retrain, load them + policy specs
    if (not FORCE_RETRAIN) and os.path.exists(PREPROCESSOR_PATH) and os.path.exists(MODELS_PATH) and os.path.exists(RANGES_PATH):
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        models = joblib.load(MODELS_PATH)
        target_ranges = joblib.load(RANGES_PATH)

        weights = DEFAULT_WEIGHTS
        if os.path.exists(WEIGHTS_PATH):
            with open(WEIGHTS_PATH, "r") as f:
                weights = json.load(f)

        policy_specs = None
        if os.path.exists(POLICIES_PATH):
            with open(POLICIES_PATH, "r") as f:
                tmp = json.load(f)
            policy_specs = {k: (int(v[0]), int(v[1])) for k, v in tmp.items()}

        # If policy specs missing but dataset present, infer again
        if policy_specs is None:
            if os.path.exists(CSV_PATH):
                df = pd.read_csv(CSV_PATH)
                policy_specs = infer_policy_specs(df)
            else:
                raise FileNotFoundError("Missing policy_specs.json and dataset CSV not found to infer policies.")

        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, "r") as f:
                metrics = json.load(f)
            print("✅ Loaded trained artifacts from:", ARTIFACT_DIR)
            print("\nCached evaluation (from training time):")
            print_metrics(metrics)
        else:
            print("✅ Loaded trained artifacts from:", ARTIFACT_DIR)

        return preprocessor, models, target_ranges, weights, policy_specs

    # =========================
    # Train forced OR missing
    # =========================
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Couldn't find {CSV_PATH}. Put it in this folder or edit CSV_PATH.")

    df = pd.read_csv(CSV_PATH)

    # ✅ infer ALL dataset policies (10 in your CSV)
    policy_specs = infer_policy_specs(df)
    with open(POLICIES_PATH, "w") as f:
        json.dump({k: [v[0], v[1]] for k, v in policy_specs.items()}, f, indent=2)

    missing = [t for t in TARGETS if t not in df.columns]
    if missing:
        raise ValueError(f"Missing target columns in CSV: {missing}")

    if "reconstruction_time_ms" in df.columns:
        df["reconstruction_time_ms"] = df["reconstruction_time_ms"].fillna(0)

    drop_cols = [c for c in ["timestamp", "run_id"] if c in df.columns]
    y = df[TARGETS].copy()
    X = df.drop(columns=TARGETS + drop_cols, errors="ignore").copy()

    # ✅ Drop ec_k/ec_m if policy_name exists to avoid duplicate info / leakage
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

    print("\n🚀 Training Random Forest models with progress bars...\n")
    print(f"✅ Policies found in dataset: {len(policy_specs)}")
    print("   " + ", ".join(sorted(policy_specs.keys())) + "\n")

    for t in tqdm(TARGETS, desc="Targets", unit="target"):
        rf = RandomForestRegressor(
            n_estimators=TREE_STEP,
            random_state=SEED,
            n_jobs=N_JOBS,
            min_samples_leaf=2,
            warm_start=True,
        )

        for n in tqdm(range(TREE_STEP, N_TREES + 1, TREE_STEP),
                      desc=f"  Trees ({t})",
                      unit="trees",
                      leave=False):
            rf.set_params(n_estimators=n)
            rf.fit(Xt_train, y_train[t].values)

        pred = rf.predict(Xt_test)
        metrics[t] = regression_report(y_test[t].values, pred)
        models[t] = rf

    target_ranges = compute_target_ranges(y_train)

    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    joblib.dump(models, MODELS_PATH)
    joblib.dump(target_ranges, RANGES_PATH)

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    with open(WEIGHTS_PATH, "w") as f:
        json.dump(DEFAULT_WEIGHTS, f, indent=2)

    print("\n✅ Trained + tested once, saved artifacts to:", ARTIFACT_DIR)
    print_metrics(metrics)

    return preprocessor, models, target_ranges, DEFAULT_WEIGHTS, policy_specs


def recommend(preprocessor, models, user_context, target_ranges, weights, policy_specs):
    candidate_policies = list(policy_specs.keys())  # ✅ ALL dataset policies
    results = []

    for pol in candidate_policies:
        row = dict(user_context)
        row["policy_name"] = pol

        X_one = pd.DataFrame([row])
        X_one = align_user_input_to_training_schema(preprocessor, X_one)
        Xt = preprocessor.transform(X_one)

        pred_map = {t: float(models[t].predict(Xt)[0]) for t in TARGETS}

        score = 0.0
        for t in TARGETS:
            n = normalize(pred_map[t], target_ranges[t]["min"], target_ranges[t]["max"])
            score += weights.get(t, 0.0) * n

        k, m = policy_specs[pol]
        ov = storage_overhead(k, m)
        eff = storage_efficiency(k, m)

        ovs = [storage_overhead(*policy_specs[p]) for p in candidate_policies]
        efs = [storage_efficiency(*policy_specs[p]) for p in candidate_policies]
        ov_n = normalize(ov, min(ovs), max(ovs))
        eff_n = normalize(eff, min(efs), max(efs))

        score += weights.get("storage_overhead", 0.0) * ov_n
        score -= weights.get("storage_efficiency", 0.0) * eff_n

        results.append({
            "policy_name": pol,
            "score": float(score),
            "predicted": pred_map,
            "storage_overhead_factor": float(ov),
            "storage_efficiency": float(eff),
        })

    results.sort(key=lambda x: x["score"])
    return results[0], results


def main():
    preprocessor, models, target_ranges, weights, policy_specs = load_or_train()

    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        ensure_user_input_template(df)

    if not os.path.exists(USER_INPUT_PATH):
        return

    with open(USER_INPUT_PATH, "r") as f:
        user_context = json.load(f)

    best, ranked = recommend(preprocessor, models, user_context, target_ranges, weights, policy_specs)

    print("\n========================")
    print("RECOMMENDATION RESULT")
    print("========================")
    print("Best policy:", best["policy_name"])
    print("Final score:", best["score"])

    print("\nPredicted metrics (ms):")
    for k, v in best["predicted"].items():
        print(f"  {k}: {v:.3f}")

    print("\nStorage:")
    print("  overhead factor:", best["storage_overhead_factor"])
    print("  efficiency:", best["storage_efficiency"])

    print("\nTop-3 ranking:")
    for r in ranked[:3]:
        print(f"  {r['policy_name']:8s} score={r['score']:.4f}")


if __name__ == "__main__":
    main()
