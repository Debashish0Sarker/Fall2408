# Swift EC Policy Recommender (Strategy A, classifier-only, storage-aware score)

This project trains and evaluates **three** models (CatBoost, XGBoost, Logistic Regression) to recommend the best
OpenStack Swift erasure-coding policy for a workload **context**.

Key improvements vs earlier notebook:
- Uses **read_ratio-aware** performance scoring (reads vs writes)
- Adds **storage overhead penalty** (deterministic from EC k,m) using **Option A**:
  normalized overhead * normalized workload volume
- Uses **percentile reliability** factor for p95 terms: `rho = min(1, num_objects / T)`
- Creates oracle labels (`best_policy_true`) by **minimizing the total score** per workload_id.
- Splits by **workload_id** to avoid leakage.

## 0) Put your dataset here
Copy `dataset_30k.csv` into: `data/dataset_30k.csv`

## 1) Create environment (Windows / macOS / Linux)
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## 2) Run the full pipeline
```bash
python -m src.run_all --csv data/dataset_30k.csv --out artifacts --results results --lambda_storage 0.35
```

This will generate:
- `artifacts/contexts.csv` : one row per workload_id with features + oracle label
- `artifacts/scores.csv`   : per (workload_id, policy) score components
- trained models in `artifacts/models/`
- evaluation tables and plots in `results/`

## 3) Optional: sweep lambda to justify the storage weight
```bash
python -m src.lambda_sweep --csv data/dataset_30k.csv --out results --lambda_min 0.0 --lambda_max 0.6 --lambda_step 0.05
```

## Notes on configuration
See `src/config.py` for:
- p95 reliability threshold `T`
- base weights for avg/p95/reconstruction
- volume feature used for storage penalty
- list of input features (context-only)

