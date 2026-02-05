# src/preprocessing.py
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

WORKLOAD_ID_COL = "workload_id"

TARGETS = [
    "avg_upload_ms",
    "p95_upload_ms",
    "avg_download_ms",
    "p95_download_ms",
    "reconstruction_time_ms",
]

DROP_ALWAYS = ["run_id", "timestamp"]
DROP_REDUNDANT_EC = ["ec_k", "ec_m", "storage_overhead", "efficiency"]

# Decision-time feature allowlist (safe, pre-run/exogenous only)
ALLOWED_FEATURES = [
    "policy_name",
    "file_size_mb",
    "num_objects",
    "concurrency",
    "read_ratio",
    "failure_state",

    # Derived workload features (still decision-time; computed from workload definition)
    "total_mb",
    "read_mb",
    "write_mb",
    "per_thread_mb",
    "io_intensity",
    "payload_bytes",

    # Pre-run state / cheap probes
    "data_disks_used_pct_before",
    "mem_available_kb_before",
    "load1_before",
    "cpu_steal_pct_before",
    "runnable_procs",
    "dirty_kb",
    "writeback_kb",
    "swap_activity_delta",

    # Disk pressure deltas (pre-run window deltas)
    "disk_reads_delta_max",
    "disk_writes_delta_max",
    "disk_io_time_ms_delta_max",
    "disk_weighted_io_time_delta_max",
    "disk_sectors_rw_delta_max",

    # Failure config
    "failed_disks",
    "failed_nodes",
    "failure_disk",
    "time_since_last_failure_s",
]

# Always drop leaky/post-run
DROP_LEAKY = [
    "data_disks_used_pct_after",
    "data_disks_used_pct_delta",
    "cpu_util_pct_during_run",
    "net_rx_bytes_delta",
    "net_tx_bytes_delta",
    "total_upload_bytes",
    "total_download_bytes",
    "upload_throughput_mbps",
    "download_throughput_mbps",
    "upload_success_rate",
    "download_success_rate",
]

BASE_CATEGORICALS = ["policy_name", "failure_disk"]


def constant_columns(df: pd.DataFrame, ignore_cols=None):
    ignore_cols = set(ignore_cols or [])
    out = []
    for c in df.columns:
        if c in ignore_cols:
            continue
        if df[c].nunique(dropna=False) <= 1:
            out.append(c)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw/dataset.csv")
    ap.add_argument("--outdir", default="data/processed")
    ap.add_argument("--splits_dir", default="data/splits")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=42)

    # IMPORTANT:
    # - we DO keep workload_id in the processed CSV for evaluation grouping
    # - but we do NOT include it in feature_cols unless you explicitly ask
    ap.add_argument("--include_workload_id_as_feature", action="store_true")
    args = ap.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    splits_dir = Path(args.splits_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    missing_targets = [t for t in TARGETS if t not in df.columns]
    if missing_targets:
        raise ValueError(f"Missing target columns: {missing_targets}")

    drop_cols = set()
    for c in DROP_ALWAYS + DROP_REDUNDANT_EC + DROP_LEAKY:
        if c in df.columns:
            drop_cols.add(c)

    # Feature list = allowlist ∩ columns, minus drop list
    allowed = [c for c in ALLOWED_FEATURES if c in df.columns]
    feature_cols = [c for c in allowed if c not in drop_cols]

    if args.include_workload_id_as_feature and WORKLOAD_ID_COL in df.columns:
        feature_cols.append(WORKLOAD_ID_COL)

    if "policy_name" not in feature_cols:
        raise ValueError("policy_name must be present in feature_cols.")

    # Categoricals used by CatBoost
    categorical_cols = [c for c in BASE_CATEGORICALS if c in feature_cols]
    if args.include_workload_id_as_feature and WORKLOAD_ID_COL in feature_cols:
        categorical_cols.append(WORKLOAD_ID_COL)

    # Drop constant features (except categoricals)
    const_cols = constant_columns(df[feature_cols], ignore_cols=set(categorical_cols))
    feature_cols = [c for c in feature_cols if c not in const_cols]

    # Build df_base:
    # keep workload_id as a META column (if present) for evaluation grouping
    meta_cols = []
    if WORKLOAD_ID_COL in df.columns:
        meta_cols.append(WORKLOAD_ID_COL)

    keep_cols = meta_cols + feature_cols + TARGETS
    keep_cols = [c for c in keep_cols if c in df.columns]
    df_base = df[keep_cols].copy()

    # Make categoricals strings
    for c in categorical_cols:
        if c in df_base.columns:
            df_base[c] = df_base[c].astype(str)

    base_out = outdir / "train_base.csv"
    df_base.to_csv(base_out, index=False)

    # Recon dataset
    if "failure_state" not in df_base.columns:
        raise ValueError("failure_state is required for recon gating.")
    df_recon = df_base[df_base["failure_state"] == 1].copy()
    df_recon = df_recon[~df_recon["reconstruction_time_ms"].isna()].copy()

    recon_out = outdir / "train_recon.csv"
    df_recon.to_csv(recon_out, index=False)

    # Split groups: best leakage control is workload_id if available
    if WORKLOAD_ID_COL in df.columns:
        groups = df[WORKLOAD_ID_COL].astype(str)
        grouping_used = "workload_id"
    else:
        # fallback context grouping (exogenous knobs only)
        base_group_cols = [c for c in ["file_size_mb", "num_objects", "concurrency", "read_ratio", "failure_state"] if c in df.columns]
        groups = df[base_group_cols].astype(str).agg("|".join, axis=1) if base_group_cols else pd.Series(df.index.astype(str), index=df.index)
        grouping_used = "context"

    gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.random_state)
    idx = np.arange(len(df_base))
    train_idx, test_idx = next(gss.split(idx, groups=groups.iloc[: len(df_base)]))

    (splits_dir / "train_idx.txt").write_text("\n".join(map(str, train_idx)))
    (splits_dir / "test_idx.txt").write_text("\n".join(map(str, test_idx)))

    metadata = {
        "input_file": str(input_path),
        "rows_raw": int(len(df)),
        "rows_base": int(len(df_base)),
        "rows_recon": int(len(df_recon)),
        "feature_cols": feature_cols,
        "categorical_cols": [c for c in categorical_cols if c in feature_cols],
        "meta_cols": meta_cols,
        "targets": TARGETS,
        "dropped_cols": sorted(list(drop_cols)),
        "dropped_constant_cols": const_cols,
        "split": {
            "test_size": args.test_size,
            "random_state": args.random_state,
            "grouping": grouping_used,
            "include_workload_id_as_feature": bool(args.include_workload_id_as_feature),
        },
        "notes": [
            "workload_id kept in train_base.csv as META for evaluation grouping (not used as feature unless include_workload_id_as_feature).",
            "Explicit allowlist used to avoid leakage.",
            "Leaky/post-run columns force-dropped.",
            "reconstruction_time_ms model trains only on failure_state==1 rows.",
        ],
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"[OK] Wrote {base_out}")
    print(f"[OK] Wrote {recon_out}")
    print(f"[OK] Wrote {outdir / 'metadata.json'}")
    print(f"[OK] Wrote splits -> {splits_dir}")


if __name__ == "__main__":
    main()
