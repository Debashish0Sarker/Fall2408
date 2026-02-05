# dataset_analysis/run_all_dataset_graphs.py
# Generates ALL dataset graphs into ./dataset_analysis/outputs/
# Uses only matplotlib (no seaborn), and tries to be robust to missing/renamed columns.

import os
import re
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------------------------
# Config
# -------------------------
CSV_PATH = "dataset_30k.csv"   # change if needed
OUT_DIR = os.path.join("dataset_analysis", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)

# Targets you care about
LAT_COLS = ["avg_upload_ms", "p95_upload_ms", "avg_download_ms", "p95_download_ms"]
RECON_COL_CANDIDATES = ["reconstruction_time_ms", "reconstruction_time_mean_ms", "reconstruction_ms", "recon_time_mean_ms"]
POLICY_COL_CANDIDATES = ["policy_name", "ec_policy", "policy", "policy_id"]
FAIL_COL_CANDIDATES = ["failure_state", "is_failure", "failed", "disk_failure", "failure"]

# Workload intensity candidates (we’ll use the best available)
WORKLOAD_MB_CANDIDATES = ["total_mb", "totalMB", "workload_total_mb", "bytes_total", "total_bytes", "workload_bytes"]

# Optional workload/context identifier candidates
CTX_ID_CANDIDATES = ["workload_id", "context_id", "ctx_id", "scenario_id", "run_context_id"]

# -------------------------
# Helpers
# -------------------------
def pick_first_existing(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def safe_numeric(s):
    return pd.to_numeric(s, errors="coerce")

def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def save_fig(path, dpi=300):
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()

def plot_bar_from_series(series, title, xlabel, ylabel, outpath, rotation=45):
    plt.figure(figsize=(10, 5))
    series.plot(kind="bar")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation)
    save_fig(outpath)

def plot_hist(arr, title, xlabel, outpath, bins=40):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return
    plt.figure(figsize=(8, 5))
    plt.hist(arr, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    save_fig(outpath)

def plot_cdf(arr, title, xlabel, outpath):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return
    x = np.sort(arr)
    y = np.arange(1, len(x) + 1) / len(x)
    plt.figure(figsize=(8, 5))
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("CDF")
    plt.grid(True)
    save_fig(outpath)

def parse_km_from_policy(policy_str):
    """
    Tries to extract k,m from strings like:
      'EC-4-2', 'ec_6_3', 'k=4,m=2', '4+2', etc.
    Returns (k,m) or (None,None).
    """
    if policy_str is None or (isinstance(policy_str, float) and np.isnan(policy_str)):
        return None, None
    s = str(policy_str)

    # EC-4-2 like patterns
    m = re.search(r"EC[-_ ]?(\d+)[-_ ]?(\d+)", s, flags=re.IGNORECASE)
    if m:
        return int(m.group(1)), int(m.group(2))

    # k=4,m=2 patterns
    m = re.search(r"k\s*=\s*(\d+)\s*,?\s*m\s*=\s*(\d+)", s, flags=re.IGNORECASE)
    if m:
        return int(m.group(1)), int(m.group(2))

    # 4-2 / 4_2 / 4+2 patterns (fallback)
    m = re.search(r"(\d+)\s*[-_+]\s*(\d+)", s)
    if m:
        return int(m.group(1)), int(m.group(2))

    return None, None

def compute_storage_overhead(df, policy_col):
    """
    If storage_overhead exists, use it; otherwise try compute (k+m)/k from policy name.
    """
    if "storage_overhead" in df.columns:
        return safe_numeric(df["storage_overhead"])

    k_list, m_list = [], []
    for p in df[policy_col].astype(str).tolist():
        k, m = parse_km_from_policy(p)
        k_list.append(k)
        m_list.append(m)

    k_arr = pd.Series(k_list, index=df.index, dtype="float")
    m_arr = pd.Series(m_list, index=df.index, dtype="float")
    overhead = (k_arr + m_arr) / k_arr
    return overhead

def compute_efficiency(df, policy_col):
    """
    If efficiency exists, use it; otherwise compute k/(k+m).
    """
    if "efficiency" in df.columns:
        return safe_numeric(df["efficiency"])

    k_list, m_list = [], []
    for p in df[policy_col].astype(str).tolist():
        k, m = parse_km_from_policy(p)
        k_list.append(k)
        m_list.append(m)

    k_arr = pd.Series(k_list, index=df.index, dtype="float")
    m_arr = pd.Series(m_list, index=df.index, dtype="float")
    eff = k_arr / (k_arr + m_arr)
    return eff

def infer_failure_state(df, fail_col):
    s = df[fail_col]
    # normalize to 0/1 if it’s boolean-ish or strings
    if s.dtype == bool:
        return s.astype(int)
    if s.dtype == object:
        ss = s.fillna("0").astype(str).str.lower().str.strip()
        return ss.isin(["1", "true", "yes", "y", "fail", "failure"]).astype(int)
    return safe_numeric(s).fillna(0).astype(int)

def infer_context_id(df, policy_col):
    """
    Prefer a real context/workload id if present.
    Otherwise build a stable key from “workload-ish” columns (excluding policy + targets).
    """
    ctx_col = pick_first_existing(df, CTX_ID_CANDIDATES)
    if ctx_col:
        return df[ctx_col].astype(str)

    # Build from all non-target numeric + categorical columns excluding policy column
    exclude = set(LAT_COLS)
    recon = pick_first_existing(df, RECON_COL_CANDIDATES)
    if recon:
        exclude.add(recon)
    exclude.add(policy_col)
    exclude.add("storage_overhead")
    exclude.add("efficiency")

    cols = [c for c in df.columns if c not in exclude]

    # Drop high-cardinality columns (filenames, timestamps, ids)
    def looks_like_id_time(name):
        n = name.lower()
        pats = ["id", "uuid", "guid", "hash", "time", "timestamp", "date", "path", "file", "name", "run", "trial"]
        return any(p in n for p in pats)

    cols = [c for c in cols if not looks_like_id_time(c)]

    # keep a small set: choose up to 10 columns with moderate uniqueness
    chosen = []
    n = len(df)
    for c in cols:
        nunique = df[c].nunique(dropna=False)
        if 2 <= nunique <= min(500, max(10, int(0.05 * n))):
            chosen.append(c)
        if len(chosen) >= 10:
            break

    if not chosen:
        # last resort: use row index buckets (not ideal, but avoids crash)
        return (df.index // 3).astype(str)

    tmp = df[chosen].copy()
    for c in chosen:
        if tmp[c].dtype == object:
            tmp[c] = tmp[c].fillna("missing").astype(str)
        else:
            tmp[c] = safe_numeric(tmp[c]).fillna(-999999).astype(str)

    return tmp.astype(str).agg("|".join, axis=1)

def plot_scatter_with_trend(x, y, title, xlabel, ylabel, outpath):
    x = safe_numeric(x)
    y = safe_numeric(y)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask].values.astype(float)
    y = y[mask].values.astype(float)
    if len(x) < 5:
        return

    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, s=10)

    # Trend line (simple linear fit)
    coeff = np.polyfit(x, y, deg=1)
    xx = np.linspace(x.min(), x.max(), 200)
    yy = coeff[0] * xx + coeff[1]
    plt.plot(xx, yy)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    save_fig(outpath)

def plot_box_by_group(df, group_col, value_col, title, ylabel, outpath, max_groups=20):
    # drop NaNs and cap number of groups for readability
    data = df[[group_col, value_col]].copy()
    data[value_col] = safe_numeric(data[value_col])
    data = data.dropna(subset=[value_col])

    groups = data[group_col].astype(str).unique().tolist()
    # stable order: sort groups and cap
    groups = sorted(groups)[:max_groups]

    series_list = [data.loc[data[group_col].astype(str) == g, value_col].values for g in groups]
    if len(series_list) == 0:
        return

    plt.figure(figsize=(10, 5))
    plt.boxplot(series_list, labels=groups, showfliers=False)
    plt.title(title)
    plt.xlabel(group_col)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    save_fig(outpath)

def plot_corr_heatmap(df, outpath, title="Correlation Matrix"):
    num = df.select_dtypes(include=["int64", "float64"]).copy()
    # drop constant columns
    num = num.loc[:, num.nunique() > 1]
    if num.shape[1] < 2:
        return
    corr = num.corr(method="pearson")

    plt.figure(figsize=(14, 12))
    plt.imshow(corr, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=7)
    plt.yticks(range(len(corr.columns)), corr.columns, fontsize=7)
    plt.title(title)
    save_fig(outpath)

# -------------------------
# Main
# -------------------------
def main():
    print("Reading:", CSV_PATH)
    df = pd.read_csv(CSV_PATH)

    policy_col = pick_first_existing(df, POLICY_COL_CANDIDATES)
    if not policy_col:
        raise RuntimeError(f"No policy column found. Tried: {POLICY_COL_CANDIDATES}")

    fail_col = pick_first_existing(df, FAIL_COL_CANDIDATES)
    if not fail_col:
        print("Warning: no failure column found. Some failure/non-failure graphs will be skipped.")

    recon_col = pick_first_existing(df, RECON_COL_CANDIDATES)
    if not recon_col:
        print("Warning: no reconstruction column found. Recon graphs will be skipped.")

    workload_mb_col = pick_first_existing(df, WORKLOAD_MB_CANDIDATES)
    if not workload_mb_col:
        print("Warning: no workload-intensity MB column found. Latency-vs-workload scatter graphs will be skipped.")

    # Compute/ensure derived fields
    df["storage_overhead"] = compute_storage_overhead(df, policy_col)
    df["efficiency"] = compute_efficiency(df, policy_col)

    if fail_col:
        df["failure_state_norm"] = infer_failure_state(df, fail_col)
    else:
        df["failure_state_norm"] = 0

    # =========================
    # Graph 1 — Storage overhead vs policy (bar plot)
    # =========================
    g1 = df.groupby(policy_col)["storage_overhead"].mean().sort_values()
    plot_bar_from_series(
        g1,
        title="Storage overhead vs EC policy (mean)",
        xlabel="EC policy",
        ylabel="Storage overhead ( (k+m)/k )",
        outpath=os.path.join(OUT_DIR, "graph1_overhead_by_policy.png"),
    )

    # =========================
    # Graph 2 — Latency vs workload intensity (scatter + trend line)
    # =========================
    if workload_mb_col and "avg_upload_ms" in df.columns:
        plot_scatter_with_trend(
            df[workload_mb_col], df["avg_upload_ms"],
            title="Workload intensity vs avg upload latency",
            xlabel=workload_mb_col,
            ylabel="avg_upload_ms",
            outpath=os.path.join(OUT_DIR, "graph2_totalmb_vs_avg_upload_ms.png"),
        )
    if workload_mb_col and "avg_download_ms" in df.columns:
        plot_scatter_with_trend(
            df[workload_mb_col], df["avg_download_ms"],
            title="Workload intensity vs avg download latency",
            xlabel=workload_mb_col,
            ylabel="avg_download_ms",
            outpath=os.path.join(OUT_DIR, "graph2_totalmb_vs_avg_download_ms.png"),
        )

    # =========================
    # Graph 3 — Failure vs non-failure latency distributions (boxplots)
    # =========================
    if fail_col:
        for c in ["avg_upload_ms", "p95_upload_ms", "avg_download_ms", "p95_download_ms"]:
            if c in df.columns:
                plot_box_by_group(
                    df, "failure_state_norm", c,
                    title=f"{c} distribution: failure(1) vs non-failure(0)",
                    ylabel=c,
                    outpath=os.path.join(OUT_DIR, f"graph3_box_{c}.png"),
                    max_groups=2
                )

    # =========================
    # Graph 4 — Reconstruction time distribution (failure-only)
    # =========================
    if recon_col:
        recon_vals = df[recon_col]
        if fail_col:
            recon_vals = df.loc[df["failure_state_norm"] == 1, recon_col]
        recon_vals = safe_numeric(recon_vals).dropna().values
        plot_hist(
            recon_vals,
            title="Reconstruction time distribution (failure-only)",
            xlabel=recon_col,
            outpath=os.path.join(OUT_DIR, "graph4_recon_hist.png"),
            bins=45
        )
        plot_cdf(
            recon_vals,
            title="Reconstruction time CDF (failure-only)",
            xlabel=recon_col,
            outpath=os.path.join(OUT_DIR, "graph4_recon_cdf.png")
        )

    # =========================
    # Graph 5 — Policy-wise latency distribution (boxplots by policy)
    # =========================
    for c in ["avg_upload_ms", "p95_upload_ms", "avg_download_ms", "p95_download_ms"]:
        if c in df.columns:
            plot_box_by_group(
                df, policy_col, c,
                title=f"{c} distribution across EC policies",
                ylabel=c,
                outpath=os.path.join(OUT_DIR, f"graph5_{c}_by_policy.png"),
                max_groups=20
            )

    # =========================
    # Graph 6 — Tail latency amplification (p95/avg ratio per policy)
    # =========================
    if all(col in df.columns for col in ["avg_upload_ms", "p95_upload_ms", "avg_download_ms", "p95_download_ms"]):
        up_ratio = safe_numeric(df["p95_upload_ms"]) / safe_numeric(df["avg_upload_ms"])
        dn_ratio = safe_numeric(df["p95_download_ms"]) / safe_numeric(df["avg_download_ms"])
        df["upload_tail_ratio"] = up_ratio.replace([np.inf, -np.inf], np.nan)
        df["download_tail_ratio"] = dn_ratio.replace([np.inf, -np.inf], np.nan)

        tail = df.groupby(policy_col)[["upload_tail_ratio", "download_tail_ratio"]].mean(numeric_only=True)
        tail = tail.dropna(how="all").sort_index()

        plt.figure(figsize=(10, 5))
        tail.plot(kind="bar")
        plt.title("Tail latency amplification across EC policies (mean p95/avg)")
        plt.xlabel("EC policy")
        plt.ylabel("p95 / average ratio")
        plt.xticks(rotation=45)
        save_fig(os.path.join(OUT_DIR, "graph6_tail_latency_amplification.png"))

    # =========================
    # Graph 7 — Policy ranking instability across workload contexts
    #   We compute a simple "score" (lower is better):
    #     mean of available latency cols + recon (if present; recon only counted on failure if you want)
    # =========================
    # Build a context id
    ctx = infer_context_id(df, policy_col)
    df["_ctx"] = ctx

    score_parts = []
    for c in ["avg_upload_ms", "p95_upload_ms", "avg_download_ms", "p95_download_ms"]:
        if c in df.columns:
            score_parts.append(safe_numeric(df[c]))
    if recon_col:
        # include recon always; you can change to failure-only logic if needed
        score_parts.append(safe_numeric(df[recon_col]).fillna(0))

    if len(score_parts) >= 2:
        df["_score"] = pd.concat(score_parts, axis=1).mean(axis=1, skipna=True)

        # For each context, rank policies by actual score
        rank_counts = {}  # policy -> [best, second, third]
        for _, g in df.groupby("_ctx"):
            gg = g.dropna(subset=["_score"]).copy()
            if gg.empty:
                continue
            ranked = gg.sort_values("_score")
            top_policies = ranked[policy_col].astype(str).tolist()

            for idx, pol in enumerate(top_policies[:3]):
                rank_counts.setdefault(pol, [0, 0, 0])
                rank_counts[pol][idx] += 1

        if rank_counts:
            rank_df = pd.DataFrame(rank_counts, index=["Best", "Second", "Third"]).T
            rank_df = rank_df.sort_values(by="Best", ascending=False).head(20)  # keep readable

            plt.figure(figsize=(10, 5))
            rank_df.plot(kind="bar", stacked=True)
            plt.title("EC policy ranking variability across contexts (Top-3 counts)")
            plt.xlabel("EC policy")
            plt.ylabel("Number of contexts")
            plt.xticks(rotation=45)
            save_fig(os.path.join(OUT_DIR, "graph7_policy_ranking_instability.png"))

    # =========================
    # Graph 8 — Workload context diversity (histograms)
    # =========================
    # Use the best available workload columns (if present)
    diversity_cols = []
    for c in WORKLOAD_MB_CANDIDATES + ["object_size", "obj_size", "concurrency", "clients", "req_rate", "request_rate"]:
        if c in df.columns and c not in diversity_cols:
            diversity_cols.append(c)
    diversity_cols = diversity_cols[:6]  # cap

    for c in diversity_cols:
        vals = safe_numeric(df[c]).dropna().values
        if len(vals) == 0:
            continue
        plot_hist(
            vals,
            title=f"Distribution of {c}",
            xlabel=c,
            outpath=os.path.join(OUT_DIR, f"graph8_{c}_distribution.png"),
            bins=45
        )

    # =========================
    # Graph 9 — Correlation matrix
    # =========================
    plot_corr_heatmap(df, os.path.join(OUT_DIR, "graph9_correlation_matrix.png"))

    # =========================
    # Graph 10 — Failure impact bar (mean latency by failure state)
    # =========================
    if fail_col:
        cols = [c for c in ["avg_upload_ms", "avg_download_ms", "p95_download_ms", "p95_upload_ms"] if c in df.columns]
        if cols:
            agg = df.groupby("failure_state_norm")[cols].mean(numeric_only=True)
            plt.figure(figsize=(8, 5))
            agg.plot(kind="bar")
            plt.title("Impact of failure on latency (mean)")
            plt.xlabel("failure_state (0=non-failure, 1=failure)")
            plt.ylabel("Latency (ms)")
            plt.xticks(rotation=0)
            save_fig(os.path.join(OUT_DIR, "graph10_failure_impact.png"))

    print("\nDone. Saved figures to:", OUT_DIR)
    for f in sorted(os.listdir(OUT_DIR)):
        if f.lower().endswith(".png"):
            print(" -", f)

if __name__ == "__main__":
    main()
