import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_PATH = "dataset_30k.csv"
OUT_PATH = "correlation_matrix_selected_features.png"

# =========================
# Load dataset
# =========================
df = pd.read_csv(CSV_PATH)

# =========================
# Keep only the selected features
# (policy_name, failure_disk are categorical and excluded from Pearson corr)
# =========================
selected_numeric_features = [
    # basic workload parameters
    "file_size_mb", "num_objects", "concurrency", "read_ratio", "failure_state",

    # derived workload parameters
    "total_mb", "read_mb", "write_mb", "per_thread_mb", "io_intensity", "payload_bytes",

    # pre-run system state
    "data_disks_used_pct_before", "mem_available_kb_before", "load1_before",
    "cpu_steal_pct_before", "runnable_procs", "dirty_kb", "writeback_kb",
    "swap_activity_delta",

    # disk pressure indicators
    "disk_reads_delta_max", "disk_writes_delta_max", "disk_io_time_ms_delta_max",
    "disk_weighted_io_time_ms_delta_max", "disk_sectors_rw_delta_max",

    # failure situation features (numeric ones)
    "failed_disks", "failed_nodes", "time_since_last_failure_s",

    # targets
    "avg_upload_ms", "p95_upload_ms",
    "avg_download_ms", "p95_download_ms",
    "reconstruction_time_ms",
]

# keep only columns that actually exist in the CSV (avoids KeyError)
existing = [c for c in selected_numeric_features if c in df.columns]
missing = [c for c in selected_numeric_features if c not in df.columns]

if missing:
    print("WARNING: These selected columns were NOT found in CSV and will be skipped:")
    for c in missing:
        print(" -", c)

selected_df = df[existing].copy()

# ensure numeric + drop zero-variance columns
selected_df = selected_df.apply(pd.to_numeric, errors="coerce")
selected_df = selected_df.dropna(axis=1, how="all")
selected_df = selected_df.loc[:, selected_df.nunique(dropna=True) > 1]

# =========================
# Correlation
# =========================
corr = selected_df.corr(method="pearson")

# =========================
# Plot
# =========================
plt.figure(figsize=(14, 12))
plt.imshow(corr, aspect="auto")
plt.colorbar()

plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=7)
plt.yticks(range(len(corr.columns)), corr.columns, fontsize=7)

plt.title("Correlation Matrix (Selected Features)", fontsize=14)
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=300)
plt.close()

print("Correlation matrix saved at:", OUT_PATH)
