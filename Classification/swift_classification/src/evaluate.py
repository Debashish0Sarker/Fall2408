from __future__ import annotations
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from catboost import CatBoostClassifier, Pool

from .utils import regret_stats

def load_artifacts(artifact_dir: str):
    with open(os.path.join(artifact_dir, "metadata.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    with open(os.path.join(artifact_dir, "splits.json"), "r", encoding="utf-8") as f:
        splits = json.load(f)
    models_dir = os.path.join(artifact_dir, "models")
    return meta, splits, models_dir

def predict_proba_all(models_dir: str, meta: dict, X: pd.DataFrame):
    feature_cols = meta["feature_cols"]
    cat_cols = meta["cat_cols"]
    cat_idx = meta["cat_feature_indices_catboost"]

    # CatBoost
    cb = CatBoostClassifier()
    cb.load_model(os.path.join(models_dir, "catboost.cbm"))
    cb_pool = Pool(X[feature_cols], cat_features=cat_idx)
    cb_proba = cb.predict_proba(cb_pool)
    cb_classes = cb.classes_

    # LogReg
    lr_pipe = joblib.load(os.path.join(models_dir, "logreg.joblib"))
    lr_proba = lr_pipe.predict_proba(X[feature_cols])
    lr_classes = lr_pipe.classes_

    # XGB
    pack = joblib.load(os.path.join(models_dir, "xgb.joblib"))
    pre = pack["pre"]
    xgb = pack["model"]
    le = pack["label_encoder"]
    Xtx = pre.transform(X[feature_cols])
    xgb_proba = xgb.predict_proba(Xtx)
    xgb_classes = le.inverse_transform(np.arange(len(le.classes_)))

    return {
        "CatBoostCls": (cb_proba, cb_classes),
        "LogReg": (lr_proba, lr_classes),
        "XGBCls": (xgb_proba, xgb_classes),
    }

def topk_accuracy(y_true: np.ndarray, proba: np.ndarray, classes: np.ndarray, k: int) -> float:
    idx = np.argsort(-proba, axis=1)[:, :k]
    topk = np.array(classes)[idx]
    return float(np.mean([y_true[i] in topk[i] for i in range(len(y_true))]))

def compute_regret(scores: pd.DataFrame, workload_ids: np.ndarray, pred_policies: np.ndarray):
    # scores: workload_id, policy_name, S_total, S_perf_norm, S_storage
    chosen_map = dict(zip(workload_ids, pred_policies))
    tmp = scores.copy()
    tmp["chosen_policy"] = tmp["workload_id"].map(chosen_map)

    # chosen score per workload
    chosen = tmp[tmp["policy_name"] == tmp["chosen_policy"]][["workload_id","S_total"]].rename(columns={"S_total":"chosen_score"})
    best = tmp.groupby("workload_id")["S_total"].min().reset_index().rename(columns={"S_total":"best_score"})
    out = best.merge(chosen, on="workload_id", how="left").dropna(subset=["chosen_score"])
    out["regret"] = out["chosen_score"] - out["best_score"]
    return out

def plot_bar(df: pd.DataFrame, x: str, y: str, title: str, path: str, ylim=None):
    plt.figure(figsize=(8,4))
    plt.bar(df[x], df[y])
    if ylim is not None:
        plt.ylim(*ylim)
    plt.ylabel(y)
    plt.title(title)
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def plot_regret_cdf_overlay(regrets: dict[str, np.ndarray], path: str):
    plt.figure(figsize=(8,5))
    for name, r in regrets.items():
        r = np.asarray(r, dtype=float)
        r = r[np.isfinite(r)]
        if r.size < 2:
            continue
        rs = np.sort(r)
        cdf = np.arange(1, len(rs)+1) / len(rs)
        plt.plot(rs, cdf, label=name)
    plt.xlabel("Regret (total score units)")
    plt.ylabel("CDF")
    plt.title("Regret CDF Overlay")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray, path: str, title: str):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    plt.figure(figsize=(10,8))
    disp.plot(xticks_rotation=45, values_format="d")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def evaluate_and_plot(ctx: pd.DataFrame, scores: pd.DataFrame, artifact_dir: str, results_dir: str):
    os.makedirs(results_dir, exist_ok=True)
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    meta, splits, models_dir = load_artifacts(artifact_dir)
    feature_cols = meta["feature_cols"]

    # test split
    test_ids = set(splits["test_ids"])
    test_ctx = ctx[ctx["workload_id"].isin(test_ids)].copy()

    X_test = test_ctx[feature_cols]
    y_true = test_ctx["best_policy_true"].astype(str).values
    wids = test_ctx["workload_id"].values

    pred_pack = predict_proba_all(models_dir, meta, test_ctx)

    rows = []
    regret_map = {}

    for model_name, (proba, classes) in pred_pack.items():
        pred = classes[np.argmax(proba, axis=1)]
        top1 = float(np.mean(pred == y_true))
        top3 = topk_accuracy(y_true, proba, classes, k=3)

        reg_df = compute_regret(scores, wids, pred)
        stats = regret_stats(reg_df["regret"].values)
        regret_map[model_name] = reg_df["regret"].values

        rows.append({
            "model": model_name,
            "top1_accuracy": top1,
            "top3_accuracy": top3,
            "regret_mean": stats["mean"],
            "regret_median": stats["median"],
            "regret_p95": stats["p95"],
            "n_contexts": int(reg_df.shape[0]),
        })

        # confusion matrix
        plot_confusion(
            y_true=y_true, y_pred=pred, classes=classes,
            path=os.path.join(plots_dir, f"confusion_{model_name}.png"),
            title=f"Confusion Matrix (Strategy A) - {model_name}"
        )

    results = pd.DataFrame(rows).sort_values("top1_accuracy", ascending=False)
    results.to_csv(os.path.join(results_dir, "model_comparison.csv"), index=False)

    # Bar plots
    plot_bar(results, "model", "top1_accuracy", "Top-1 Accuracy Comparison", os.path.join(plots_dir, "compare_top1.png"), ylim=(0,1))
    plot_bar(results, "model", "top3_accuracy", "Top-3 Coverage Comparison", os.path.join(plots_dir, "compare_top3.png"), ylim=(0,1))
    plot_bar(results, "model", "regret_mean", "Mean Regret Comparison", os.path.join(plots_dir, "compare_regret_mean.png"))
    plot_bar(results, "model", "regret_median", "Median Regret Comparison", os.path.join(plots_dir, "compare_regret_median.png"))
    plot_bar(results, "model", "regret_p95", "p95 Regret Comparison (Tail Risk)", os.path.join(plots_dir, "compare_regret_p95.png"))

    # Regret CDF overlay
    plot_regret_cdf_overlay(regret_map, os.path.join(plots_dir, "regret_cdf_overlay.png"))

    # True label distribution
    vc = pd.Series(y_true).value_counts()
    plt.figure(figsize=(10,5))
    plt.bar(vc.index, vc.values)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("True Best Policy")
    plt.ylabel("Count (test contexts)")
    plt.title("Label Distribution (Test)")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "label_distribution_test.png"), dpi=200)
    plt.close()
    vc.to_csv(os.path.join(results_dir, "label_distribution_test.csv"))

    return results
