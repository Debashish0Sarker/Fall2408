from dataclasses import dataclass

@dataclass(frozen=True)
class ScoreConfig:
    # Percentile reliability: rho = min(1, num_objects / T)
    T_RELIABILITY: int = 20
    EPS: float = 1e-9

    # Base weights inside the *performance* component
    # (These are interpretable priorities, not learned parameters.)
    W_U_AVG: float = 1.0
    W_U_P95: float = 2.0
    W_D_AVG: float = 1.0
    W_D_P95: float = 2.0
    W_RECON: float = 1.5

    # Which volume column to use for storage penalty scaling
    # (write_mb is ideal: overhead costs money when storing/writing data)
    VOLUME_COL: str = "write_mb"

# Context features used for classification (must be known at decision time)
# We intentionally exclude post-run counters/deltas (net/disk/cpu deltas), success rates, throughputs, and latency targets.
CONTEXT_FEATURES = [
    "file_size_mb",
    "num_objects",
    "concurrency",
    "read_ratio",
    "failure_state",
    "total_mb",
    "read_mb",
    "write_mb",
    "per_thread_mb",
    "io_intensity",
    "failed_disks",
    "failed_nodes",
    "time_since_last_failure_s",
    "segment_size",  # optional: if this is fixed globally, it won't harm; if it's policy-specific, it will be removed.
    "failure_disk",  # categorical (filled with 'none' when no failure)
]

CATEGORICAL_FEATURES = [
    "num_objects_bin",
    "failure_disk",
]
