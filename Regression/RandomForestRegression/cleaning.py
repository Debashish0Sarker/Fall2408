import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# 1) Load
df = pd.read_csv("dataset_30k.csv")

# 2) Targets (keep these unchanged, append after preprocessing)
target_cols = [
    "avg_upload_ms", "p95_upload_ms",
    "avg_download_ms", "p95_download_ms",
    "reconstruction_time_ms"
]
target_cols = [c for c in target_cols if c in df.columns]

# (Swift-specific cleanup that doesn't change the pipeline style)
# If reconstruction_time_ms is NaN for non-failure runs, set it to 0
if "reconstruction_time_ms" in df.columns:
    df["reconstruction_time_ms"] = df["reconstruction_time_ms"].fillna(0)

# 3) Drop ID-like columns (optional, but recommended to avoid useless leakage)
drop_cols = [c for c in ["timestamp", "run_id", "workload_id"] if c in df.columns]

y = df[target_cols].copy()
X = df.drop(columns=target_cols + drop_cols)

# 4) Detect numeric vs categorical
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = [c for c in X.columns if c not in categorical_cols]

# 5) Preprocessing pipeline (LIKE YOUR NOTEBOOK)
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ],
    remainder="drop"
)

# 6) Fit+transform and save
X_proc = preprocessor.fit_transform(X)
feature_names = preprocessor.get_feature_names_out()

X_proc_df = pd.DataFrame(X_proc, columns=feature_names)
cleaned = pd.concat([X_proc_df, y.reset_index(drop=True)], axis=1)

cleaned.to_csv("cleaned.csv", index=False)
print("Saved cleaned.csv with shape:", cleaned.shape)
