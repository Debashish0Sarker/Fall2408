from __future__ import annotations
import pandas as pd
from .utils import ensure_columns, safe_numeric

REQUIRED_COLS = [
    "policy_name", "ec_k", "ec_m", "segment_size", "workload_id",
    "file_size_mb", "num_objects", "concurrency", "read_ratio", "failure_state",
    "storage_overhead", "efficiency",
    "total_mb", "read_mb", "write_mb", "per_thread_mb", "io_intensity",
    "failed_disks", "failed_nodes", "failure_disk", "time_since_last_failure_s",
    "avg_upload_ms", "p95_upload_ms", "avg_download_ms", "p95_download_ms",
    "reconstruction_time_ms",
]

NUMERIC_COLS = [
    "ec_k","ec_m","segment_size","file_size_mb","num_objects","concurrency","read_ratio","failure_state",
    "storage_overhead","efficiency","total_mb","read_mb","write_mb","per_thread_mb","io_intensity",
    "failed_disks","failed_nodes","time_since_last_failure_s",
    "avg_upload_ms","p95_upload_ms","avg_download_ms","p95_download_ms","reconstruction_time_ms",
]

def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    ensure_columns(df, REQUIRED_COLS)

    # types
    df["policy_name"] = df["policy_name"].astype(str)
    df["workload_id"] = df["workload_id"].astype(str)
    df["failure_disk"] = df["failure_disk"].astype(str)

    df = safe_numeric(df, NUMERIC_COLS)

    # normalize key flags
    df["failure_state"] = df["failure_state"].fillna(0).astype(int)

    # reconstruction meaningful only in failure rows
    df["reconstruction_time_ms"] = df["reconstruction_time_ms"].fillna(0.0)
    df.loc[df["failure_state"] == 0, "reconstruction_time_ms"] = 0.0

    # fill missing failure_disk
    df["failure_disk"] = df["failure_disk"].replace({"nan": "none", "None":"none"})
    df.loc[df["failure_state"] == 0, "failure_disk"] = "none"

    return df
