from __future__ import annotations
import pandas as pd
import numpy as np
from .config import ScoreConfig, CONTEXT_FEATURES
from .utils import ensure_columns, p95_reliability, group_minmax_norm, minmax_norm, num_objects_bin

def compute_scores(df: pd.DataFrame, cfg: ScoreConfig) -> pd.DataFrame:
    # Ensure required columns exist
    req = [
        "workload_id","policy_name","read_ratio","num_objects","failure_state",
        "avg_upload_ms","p95_upload_ms","avg_download_ms","p95_download_ms","reconstruction_time_ms",
        "storage_overhead",
        cfg.VOLUME_COL,
    ]
    ensure_columns(df, req)

    # p95 reliability
    rho = p95_reliability(df["num_objects"].values, cfg.T_RELIABILITY)

    # read/write mix weights per row (context)
    r_read = df["read_ratio"].fillna(0.0).clip(0.0, 1.0).astype(float).values
    r_write = 1.0 - r_read

    # performance component in ms-like units
    S_perf = (
        r_write * (cfg.W_U_AVG * df["avg_upload_ms"].values + cfg.W_U_P95 * rho * df["p95_upload_ms"].values)
        + r_read * (cfg.W_D_AVG * df["avg_download_ms"].values + cfg.W_D_P95 * rho * df["p95_download_ms"].values)
        + (cfg.W_RECON * df["reconstruction_time_ms"].values)
    )

    out = df[["workload_id","policy_name","storage_overhead", cfg.VOLUME_COL]].copy()
    out["rho_p95"] = rho
    out["S_perf"] = S_perf

    # Normalize performance *within each context* (workload_id)
    out["S_perf_norm"] = group_minmax_norm(out["S_perf"], out["workload_id"], eps=cfg.EPS)

    # Normalize overhead globally (deterministic across policies)
    out["OH_norm"] = minmax_norm(out["storage_overhead"].values, eps=cfg.EPS)

    # Volume factor (context-level), use log1p then normalize globally
    V = out[cfg.VOLUME_COL].fillna(0.0).astype(float).values
    V_log = np.log1p(np.clip(V, 0, None))
    out["V_norm"] = minmax_norm(V_log, eps=cfg.EPS)

    # Storage penalty
    out["S_storage"] = out["OH_norm"] * out["V_norm"]

   # Fixed manual weights (no lambda)
    W_PERF = 0.65
    W_STORAGE = 0.35

    out["S_total"] = W_PERF * out["S_perf_norm"] + W_STORAGE * out["S_storage"]
    out["W_PERF"] = W_PERF
    out["W_STORAGE"] = W_STORAGE

    return out

def build_oracle_contexts(df: pd.DataFrame, scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Pick best policy per workload_id
    idx = scores.groupby("workload_id")["S_total"].idxmin()
    best = scores.loc[idx, ["workload_id","policy_name","S_total"]].rename(
        columns={"policy_name":"best_policy_true","S_total":"best_score_true"}
    )

    # Build one row per context using context-only columns (take first row per workload_id)
    ctx = df.sort_values(["workload_id","policy_name"]).drop_duplicates("workload_id").copy()
    # Remove any policy-specific context features that vary across policies (segment_size is suspicious)
    # If segment_size differs across policies, drop it from context features.
    ctx_features = [c for c in CONTEXT_FEATURES if c in ctx.columns]
    if "segment_size" in ctx_features:
        # check variability within contexts
        seg_var = df.groupby("workload_id")["segment_size"].nunique()
        if (seg_var > 1).any():
            ctx_features.remove("segment_size")

    ctx = ctx[["workload_id"] + ctx_features].copy()

    # engineered features
    ctx["num_objects_bin"] = ctx["num_objects"].apply(num_objects_bin)

    # merge oracle labels
    ctx = ctx.merge(best, on="workload_id", how="inner")

    # also merge failure_state for easy slicing (already in ctx_features usually)
    return ctx, best
