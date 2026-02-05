from __future__ import annotations
import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from catboost import CatBoostClassifier, Pool
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

from .config import CATEGORICAL_FEATURES

def split_by_workload(ctx: pd.DataFrame, seed: int = 42, test_size: float = 0.15, val_size: float = 0.15):
    # ctx has one row per workload_id
    ids = ctx["workload_id"].unique()
    train_ids, tmp_ids = train_test_split(ids, test_size=test_size+val_size, random_state=seed, shuffle=True)
    # split tmp into val/test
    rel_test = test_size / (test_size + val_size)
    val_ids, test_ids = train_test_split(tmp_ids, test_size=rel_test, random_state=seed, shuffle=True)
    return train_ids, val_ids, test_ids

def make_preprocessor(feature_cols: list[str], cat_cols: list[str]):
    num_cols = [c for c in feature_cols if c not in cat_cols]
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
    )
    return pre, num_cols, cat_cols

def train_all_models(ctx: pd.DataFrame, outdir: str, seed: int = 42):
    os.makedirs(outdir, exist_ok=True)
    models_dir = os.path.join(outdir, "models")
    os.makedirs(models_dir, exist_ok=True)

    # Features/target
    target = "best_policy_true"
    drop_cols = ["workload_id", target, "best_score_true"]
    feature_cols = [c for c in ctx.columns if c not in drop_cols]

    # cat cols intersection
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in feature_cols]

    train_ids, val_ids, test_ids = split_by_workload(ctx, seed=seed)
    split = {"train_ids": train_ids.tolist(), "val_ids": val_ids.tolist(), "test_ids": test_ids.tolist()}
    with open(os.path.join(outdir, "splits.json"), "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)

    train_df = ctx[ctx["workload_id"].isin(train_ids)].copy()
    val_df   = ctx[ctx["workload_id"].isin(val_ids)].copy()
    test_df  = ctx[ctx["workload_id"].isin(test_ids)].copy()

    X_train = train_df[feature_cols]
    y_train = train_df[target].astype(str)
    X_val   = val_df[feature_cols]
    y_val   = val_df[target].astype(str)
    X_test  = test_df[feature_cols]
    y_test  = test_df[target].astype(str)

    # =======================
    # 1) CatBoostClassifier
    # =======================
    # CatBoost can use categorical columns without OHE
    cat_feature_indices = [feature_cols.index(c) for c in cat_cols]

    cb = CatBoostClassifier(
        loss_function="MultiClass",
        iterations=3000,
        depth=8,
        learning_rate=0.1,
        random_seed=seed,
        eval_metric="Accuracy",
        auto_class_weights="Balanced",
        verbose=200,
    )
    train_pool = Pool(X_train, y_train, cat_features=cat_feature_indices)
    val_pool   = Pool(X_val, y_val, cat_features=cat_feature_indices)

    cb.fit(train_pool, eval_set=val_pool, use_best_model=True)

    cb.save_model(os.path.join(models_dir, "catboost.cbm"))

    # =======================
    # 2) Logistic Regression (OHE + scaler)
    # =======================
    pre, num_cols, cat_cols_used = make_preprocessor(feature_cols, cat_cols)
    lr = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )

    lr_pipe = Pipeline([("pre", pre), ("clf", lr)])
    lr_pipe.fit(X_train, y_train)
    joblib.dump(lr_pipe, os.path.join(models_dir, "logreg.joblib"))

    # =======================
    # 3) XGBoostClassifier (OHE) + LabelEncoder for y
    # =======================
    # We'll reuse the same preprocessor but train XGB on transformed matrices.
    Xtr = pre.fit_transform(X_train)  # fit here; store pre separately for XGB
    Xva = pre.transform(X_val)
    Xte = pre.transform(X_test)

    le = LabelEncoder()
    ytr_enc = le.fit_transform(y_train)
    yva_enc = le.transform(y_val)

    xgb = XGBClassifier(
        n_estimators=1200,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="multi:softprob",
        num_class=len(le.classes_),
        random_state=seed,
        tree_method="hist",
        eval_metric="mlogloss",
    )
    xgb.fit(Xtr, ytr_enc, eval_set=[(Xva, yva_enc)], verbose=200)
    joblib.dump({"pre": pre, "model": xgb, "label_encoder": le}, os.path.join(models_dir, "xgb.joblib"))

    # Save metadata
    meta = {
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
        "cat_feature_indices_catboost": cat_feature_indices,
        "seed": seed,
        "n_contexts_train": int(train_df.shape[0]),
        "n_contexts_val": int(val_df.shape[0]),
        "n_contexts_test": int(test_df.shape[0]),
    }
    with open(os.path.join(outdir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return {
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
        "cat_feature_indices": cat_feature_indices,
        "splits": split,
        "ctx_splits": {"train": train_df, "val": val_df, "test": test_df},
    }
