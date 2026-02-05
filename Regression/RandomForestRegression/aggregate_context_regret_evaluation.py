import os, json, joblib, re
import numpy as np
import pandas as pd
import itertools

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CSV_PATH = "dataset_30k.csv"
ARTIFACT_DIR = "rf_artifacts"
PREPROCESSOR_PATH = os.path.join(ARTIFACT_DIR, "preprocessor.joblib")
MODELS_PATH       = os.path.join(ARTIFACT_DIR, "rf_models.joblib")
WEIGHTS_PATH      = os.path.join(ARTIFACT_DIR, "weights.json")

OUT_DIR = os.path.join(ARTIFACT_DIR, "eval_agg")
os.makedirs(OUT_DIR, exist_ok=True)

REPORT_PATH       = os.path.join(OUT_DIR, "policy_reco_report.json")
REGRET_HIST_PATH  = os.path.join(OUT_DIR, "regret_hist.png")
REGRET_CDF_PATH   = os.path.join(OUT_DIR, "regret_cdf.png")
SPREAD_HIST_PATH  = os.path.join(OUT_DIR, "actual_score_spread_hist.png")
REGRETS_CSV_PATH  = os.path.join(OUT_DIR, "regrets.csv")

SEED = 42
np.random.seed(SEED)

TARGETS = ["avg_upload_ms","p95_upload_ms","avg_download_ms","p95_download_ms","reconstruction_time_ms"]

DEFAULT_WEIGHTS = {
    "avg_upload_ms": 0.15,
    "p95_upload_ms": 0.20,
    "avg_download_ms": 0.15,
    "p95_download_ms": 0.20,
    "reconstruction_time_ms": 0.20,
    "storage_overhead": 0.07,
    "efficiency": 0.03,
}

POLICY_DERIVED_COLS = {
    "storage_overhead", "efficiency",
    "k", "m", "ec_k", "ec_m", "data_fragments", "parity_fragments",
    "n_data", "n_parity", "fragment_size", "chunk_size",
    "stripe_size", "policy_index", "ec_policy",
}

def looks_like_id_or_time(colname: str) -> bool:
    c = colname.lower()
    patterns = [
        r"(^|_)id($|_)", r"uuid", r"guid", r"hash",
        r"timestamp", r"(^|_)ts($|_)", r"datetime", r"(^|_)date($|_)", r"(^|_)time($|_)",
        r"run", r"trial", r"seed", r"iteration", r"iter", r"epoch",
        r"container", r"object", r"filename", r"file", r"path",
        r"workload", r"scenario"
    ]
    return any(re.search(p, c) for p in patterns)

def normalize(x, xmin, xmax):
    if np.isclose(xmax, xmin):
        return 0.0
    return (x - xmin) / (xmax - xmin)

def compute_ranges(df, cols):
    out = {}
    for c in cols:
        v = df[c].astype(float).values
        vmin = float(np.nanmin(v)); vmax = float(np.nanmax(v))
        if np.isclose(vmin, vmax):
            vmax = vmin + 1.0
        out[c] = {"min": vmin, "max": vmax}
    return out

def score_row_vals(target_vals: dict, overhead_vals: dict, ranges, weights):
    # accepts raw values dicts instead of pandas row
    s = 0.0
    for t in TARGETS:
        s += weights.get(t, 0.0) * normalize(float(target_vals[t]), ranges[t]["min"], ranges[t]["max"])

    if "storage_overhead" in overhead_vals and "storage_overhead" in ranges:
        s += weights.get("storage_overhead", 0.0) * normalize(
            float(overhead_vals["storage_overhead"]), ranges["storage_overhead"]["min"], ranges["storage_overhead"]["max"]
        )
    if "efficiency" in overhead_vals and "efficiency" in ranges:
        s -= weights.get("efficiency", 0.0) * normalize(
            float(overhead_vals["efficiency"]), ranges["efficiency"]["min"], ranges["efficiency"]["max"]
        )
    return float(s)

def align_user_input_to_training_schema(preprocessor, X_one: pd.DataFrame) -> pd.DataFrame:
    expected = list(preprocessor.feature_names_in_)
    for col in expected:
        if col not in X_one.columns:
            X_one[col] = 0
    X_one = X_one[expected]
    for col in expected:
        if X_one[col].dtype == object:
            X_one[col] = X_one[col].fillna("missing").astype(str)
    return X_one

def predict_targets(preprocessor, models, feature_row_dict):
    X_one = pd.DataFrame([feature_row_dict])
    X_one = align_user_input_to_training_schema(preprocessor, X_one)
    Xt = preprocessor.transform(X_one)
    return {t: float(models[t].predict(Xt)[0]) for t in TARGETS}

def pick_context_cols(df: pd.DataFrame,
                      max_unique_ratio=0.05,
                      max_unique_abs=500,
                      min_unique_abs=2):
    cols = [c for c in df.columns if c not in TARGETS and c != "policy_name"]
    n = len(df)
    context_cols = []
    dropped = {}

    for c in cols:
        if c in POLICY_DERIVED_COLS:
            dropped[c] = "policy_derived"
            continue
        if looks_like_id_or_time(c):
            dropped[c] = "id_or_time"
            continue
        nunique = df[c].nunique(dropna=False)
        if nunique < min_unique_abs:
            dropped[c] = f"too_few_unique({nunique})"
            continue
        if nunique > max_unique_abs or (nunique / max(1, n)) > max_unique_ratio:
            dropped[c] = f"high_cardinality(nunique={nunique})"
            continue
        context_cols.append(c)

    return context_cols, dropped

def build_context_key(df: pd.DataFrame, context_cols: list) -> pd.Series:
    tmp = df[context_cols].copy()
    for c in context_cols:
        if tmp[c].dtype == object:
            tmp[c] = tmp[c].fillna("missing").astype(str)
        else:
            tmp[c] = tmp[c].fillna(-999999.0)
    return tmp.astype(str).agg("|".join, axis=1)

def plot_hist(arr, title, xlabel, path):
    plt.figure()
    plt.hist(arr, bins=40)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()

def plot_cdf(arr, title, xlabel, path):
    r = np.sort(arr)
    y = np.arange(1, len(r) + 1) / len(r)
    plt.figure()
    plt.plot(r, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("CDF")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()

def main():
    print("=== RF Eval (aggregate per ctx-policy) ===")

    preprocessor = joblib.load(PREPROCESSOR_PATH)
    models = joblib.load(MODELS_PATH)

    expected = set(preprocessor.feature_names_in_)
    leak = expected.intersection(set(TARGETS))
    if leak:
        raise RuntimeError(f"LEAKAGE: targets in model inputs: {sorted(leak)}")

    weights = dict(DEFAULT_WEIGHTS)
    if os.path.exists(WEIGHTS_PATH):
        with open(WEIGHTS_PATH, "r") as f:
            weights.update(json.load(f))

    df = pd.read_csv(CSV_PATH)
    if "reconstruction_time_ms" in df.columns:
        df["reconstruction_time_ms"] = df["reconstruction_time_ms"].fillna(0)

    context_cols, dropped = pick_context_cols(df)
    if not context_cols:
        raise RuntimeError(f"No context cols left. Dropped sample: {list(dropped.items())[:30]}")

    df["_ctx"] = build_context_key(df, context_cols)

    policies = sorted(df["policy_name"].astype(str).unique().tolist())
    ctx_policy_sets = df.groupby("_ctx")["policy_name"].apply(lambda s: set(s.astype(str).tolist()))

    # best triple (max shared contexts)
    best_triple = None
    best_shared = -1
    for triple in itertools.combinations(policies, 3):
        shared = sum(1 for s in ctx_policy_sets.values if all(p in s for p in triple))
        if shared > best_shared:
            best_shared = shared
            best_triple = triple

    if not best_triple or best_shared <= 0:
        raise RuntimeError("No triple shares contexts. Cannot evaluate Top-k/regret.")

    print("Best triple:", best_triple, "shared contexts:", best_shared)

    valid_ctx = [k for k, s in ctx_policy_sets.items() if all(p in s for p in best_triple)]
    df_eval = df[df["_ctx"].isin(valid_ctx)].copy()
    print("Contexts evaluated:", df_eval["_ctx"].nunique())

    # scoring ranges computed globally
    range_cols = list(TARGETS)
    if "storage_overhead" in df.columns: range_cols.append("storage_overhead")
    if "efficiency" in df.columns: range_cols.append("efficiency")
    ranges = compute_ranges(df, range_cols)

    # aggregate actual metrics per (ctx, policy): mean
    agg_cols = TARGETS[:]
    extra_cols = []
    if "storage_overhead" in df_eval.columns: extra_cols.append("storage_overhead")
    if "efficiency" in df_eval.columns: extra_cols.append("efficiency")

    actual_agg = (
        df_eval.groupby(["_ctx", "policy_name"])[agg_cols + extra_cols]
        .mean(numeric_only=True)
        .reset_index()
    )

    # For prediction, we need ONE representative feature row per (ctx, policy)
    # We'll take the first row but only to build model features (targets not used)
    rep = (
        df_eval.groupby(["_ctx", "policy_name"])
        .first()
        .reset_index()
    )

    # index helpers
    rep_idx = {(r["_ctx"], str(r["policy_name"])): r for _, r in rep.iterrows()}
    act_idx = {(r["_ctx"], str(r["policy_name"])): r for _, r in actual_agg.iterrows()}

    # Evaluate
    contexts = sorted(df_eval["_ctx"].unique().tolist())

    top1 = 0
    top3 = 0
    regrets = []
    spreads = []

    for ctx in contexts:
        # ensure all policies exist
        if not all((ctx, p) in act_idx for p in best_triple):
            continue

        # actual score per policy using aggregated actual metrics
        actual_scores = {}
        for p in best_triple:
            ar = act_idx[(ctx, p)]
            target_vals = {t: float(ar[t]) for t in TARGETS}
            overhead_vals = {}
            if "storage_overhead" in ar.index: overhead_vals["storage_overhead"] = float(ar["storage_overhead"])
            if "efficiency" in ar.index: overhead_vals["efficiency"] = float(ar["efficiency"])
            actual_scores[p] = score_row_vals(target_vals, overhead_vals, ranges, weights)

        vals = np.array([actual_scores[p] for p in best_triple], dtype=float)
        spreads.append(float(vals.max() - vals.min()))

        actual_best = min(actual_scores.items(), key=lambda kv: kv[1])[0]

        # predicted score per policy
        pred_scores = {}
        for p in best_triple:
            rr = rep_idx[(ctx, p)]
            feat = {}

            for col in expected:
                if col == "policy_name":
                    feat[col] = p
                elif col in rr.index:
                    v = rr[col]
                    if col in df.columns and df[col].dtype == object:
                        feat[col] = "missing" if pd.isna(v) else str(v)
                    else:
                        feat[col] = 0 if pd.isna(v) else v
                else:
                    feat[col] = 0

            # neutralize id/time-like inputs
            for col in expected:
                if looks_like_id_or_time(col):
                    feat[col] = "missing" if (col in df.columns and df[col].dtype == object) else 0

            pred_targets = predict_targets(preprocessor, models, feat)

            # score predicted using predicted targets + aggregated overhead/eff (actual)
            ar = act_idx[(ctx, p)]
            overhead_vals = {}
            if "storage_overhead" in ar.index: overhead_vals["storage_overhead"] = float(ar["storage_overhead"])
            if "efficiency" in ar.index: overhead_vals["efficiency"] = float(ar["efficiency"])

            pred_scores[p] = score_row_vals(pred_targets, overhead_vals, ranges, weights)

        ranked = sorted(pred_scores.items(), key=lambda kv: kv[1])
        pred_best = ranked[0][0]
        pred_top3 = [x[0] for x in ranked[:3]]

        top1 += int(pred_best == actual_best)
        top3 += int(actual_best in pred_top3)

        regret = float(actual_scores[pred_best] - actual_scores[actual_best])
        regrets.append(regret)

    regrets = np.asarray(regrets, dtype=np.float64)
    spreads = np.asarray(spreads, dtype=np.float64)

    if len(regrets) == 0:
        raise RuntimeError("No contexts were evaluated after aggregation.")

    report = {
        "best_triple": list(best_triple),
        "shared_contexts": int(best_shared),
        "n_contexts_used": int(len(regrets)),
        "Top-1": float(top1 / len(regrets)),
        "Top-3": float(top3 / len(regrets)),
        "regret_mean": float(np.mean(regrets)),
        "regret_median": float(np.median(regrets)),
        "regret_p95": float(np.percentile(regrets, 95)),
        "regret_min": float(np.min(regrets)),
        "regret_max": float(np.max(regrets)),
        "n_nonzero_regret": int(np.sum(regrets > 1e-12)),
        "actual_score_spread_median": float(np.median(spreads)),
        "actual_score_spread_p95": float(np.percentile(spreads, 95)),
        "context_cols_used": context_cols,
    }

    print("\n=== RESULTS ===")
    print("Top-1:", report["Top-1"])
    print("Top-3:", report["Top-3"])
    print("Regret mean/median/p95:", report["regret_mean"], report["regret_median"], report["regret_p95"])
    print("Regret min/max:", report["regret_min"], report["regret_max"])
    print("Non-zero regrets:", report["n_nonzero_regret"], "/", report["n_contexts_used"])
    print("Actual spread median:", report["actual_score_spread_median"])

    # save plots + csv
    pd.DataFrame({"regret": regrets}).to_csv(REGRETS_CSV_PATH, index=False)
    plot_hist(regrets, "Regret Histogram", "regret", REGRET_HIST_PATH)
    plot_cdf(regrets, "Regret CDF", "regret", REGRET_CDF_PATH)
    plot_hist(spreads, "Actual Score Spread Histogram", "max-min actual score", SPREAD_HIST_PATH)

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print("\nSaved to:", OUT_DIR)
    print(" ", REGRETS_CSV_PATH)
    print(" ", REGRET_HIST_PATH)
    print(" ", REGRET_CDF_PATH)
    print(" ", SPREAD_HIST_PATH)
    print(" ", REPORT_PATH)

if __name__ == "__main__":
    main()
