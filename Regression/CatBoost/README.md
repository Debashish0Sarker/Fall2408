0. Project setup
Create project directories
--------------------------
mkdir swift-ec-ml
cd swift-ec-ml
mkdir data
mkdir data\raw
mkdir data\processed
mkdir data\splits
mkdir models
mkdir results
mkdir results\plots
mkdir src
----------------------------

Place the dataset:

# copy your dataset (120-sample or full dataset) or paste it manually
copy dataset_120.csv data\raw\dataset.csv
---------------------
python -m venv .venv
.\.venv\Scripts\Activate.ps1
-----------------------

2. Install dependencies

Create requirements.txt:

pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.1
catboost==1.2.7
matplotlib==3.9.2
joblib==1.4.2


Install:
-----------------------------
pip install -r requirements.txt
-----------------------------
3. (Optional but recommended) Dataset sanity check
python src/check_dataset.py --input data/raw/dataset.csv


This prints:

dataset shape

missing values

constant columns

failure counts

basic target statistics

4. Preprocessing (feature cleaning + splits
--------------------------
python src/preprocessing.py --input data/raw/dataset.csv --outdir data/processed --splits_dir data/splits
--------------------------
Outputs created:
data/processed/train_base.csv
data/processed/train_recon.csv
data/processed/metadata.json
data/splits/train_idx.txt
data/splits/test_idx.txt

5. Train CatBoost regression models
----------------------------------
python src/train_models.py --processed_dir data/processed --splits_dir data/splits --models_dir models
----------------------------------

Models trained:

avg_upload_ms.cbm

p95_upload_ms.cbm

avg_download_ms.cbm

p95_download_ms.cbm

reconstruction_time_ms.cbm (only if enough failure rows)

Metadata:
models/model_info.json


⚠️ FutureWarning messages from scikit-learn are expected and safe to ignore.

6. Evaluate models (metrics + plots)
------------------------------------
python src/evaluate.py --processed_dir data/processed --splits_dir data/splits --models_dir models --results_dir results
------------------------------------
Outputs created:
results/metrics_table.csv
results/feature_importance_*.csv
results/plots/pred_vs_actual_*.png

7. Create demo workload JSON (Windows-safe, no BOM)

---------------------------------------------------
Set-Content -Path workload_demo.json -Encoding UTF8 -Value '{
  "file_size_mb": 5,
  "num_objects": 20,
  "concurrency": 2,
  "read_ratio": 0.5,
  "failure_state": 1,
  "segment_size": 1048576
}'
----------------------------------------------------

Verify:

Get-Content workload_demo.json

8. Run policy recommendation (inference)

----------------------------------------------------
python src/recommend.py --workload_json workload_demo.json --policies EC-2-2,EC-4-2,EC-6-3
----------------------------------------------------

Output:

Predicted metrics per EC policy

Composite score

Recommended EC policy