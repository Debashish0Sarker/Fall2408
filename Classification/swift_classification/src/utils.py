from __future__ import annotations
import numpy as np
import pandas as pd

def ensure_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def safe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def num_objects_bin(num_objects: float) -> str:
    try:
        n = int(num_objects)
    except Exception:
        return "unknown"
    if n <= 4: return "1-4"
    if n <= 10: return "5-10"
    if n <= 20: return "11-20"
    if n <= 50: return "21-50"
    return "51+"

def p95_reliability(num_objects: np.ndarray, T: int) -> np.ndarray:
    # rho = min(1, num_objects / T)
    x = np.asarray(num_objects, dtype=float)
    x = np.clip(x, 0, None)
    return np.clip(x / float(T), 0.0, 1.0)

def minmax_norm(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mn = np.nanmin(x)
    mx = np.nanmax(x)
    return (x - mn) / (mx - mn + eps)

def group_minmax_norm(values: pd.Series, group: pd.Series, eps: float = 1e-9) -> pd.Series:
    # min-max normalize within each group
    gmin = values.groupby(group).transform("min")
    gmax = values.groupby(group).transform("max")
    return (values - gmin) / (gmax - gmin + eps)

def regret_stats(regret: np.ndarray) -> dict:
    r = np.asarray(regret, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "p95": float("nan")}
    return {
        "mean": float(np.mean(r)),
        "median": float(np.median(r)),
        "p95": float(np.percentile(r, 95)),
    }
