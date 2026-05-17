# 🏦 Loan Default Risk Predictor

End-to-end **MLOps** project predicting loan default probability from borrower
demographics + employment profile. Production-style scaffolding: modular
pipeline, MLflow tracking, DVC + DagsHub for data/model versioning, and a
FastAPI web UI for interactive scoring.

> Part of a Medium series on transitioning from **DevOps → MLOps**.

---

## ✨ Features

| Capability | Tooling |
|---|---|
| Dataset ingestion | `kagglehub` (auto-download) |
| Preprocessing (imputation, outliers, skewness, encoding) | `scikit-learn` (OneHotEncoder, TargetEncoder, PowerTransformer) |
| Class imbalance | `imblearn.SMOTE` |
| Scaling | `StandardScaler` (bundled with model) |
| Model | `CatBoostClassifier` |
| Experiment tracking | `MLflow` (params, metrics, datasets, model + signature) |
| Data / artifact versioning | `DVC` → `DagsHub` remote |
| Serving UI | `FastAPI` + Jinja2 + vanilla CSS/JS (dark theme, collapsible sidebar) |
| Single-artifact deploy | `joblib` bundle (`{model, preprocessor, scaler, feature_columns, metadata}`) |

---

## 📁 Project structure

```
finance_modular_project/
├── config/
│   ├── params.yaml          # all hyperparameters, paths, dataset config
│   └── encoding.yaml        # categorical encoding plan
├── data/
│   ├── raw/                 # original CSVs (tracked by DVC)
│   └── processed/           # cleaned features (tracked by DVC)
├── artifacts/               # model.joblib, metrics.json, plots (tracked by DVC)
├── src/
│   ├── config.py            # loads params.yaml + .env into typed constants
│   ├── utils.py             # YAML loader, dataset loader
│   ├── logger.py            # structured logging + section banners
│   ├── data_ingestion.py    # kagglehub download → data/raw/
│   ├── preprocessing.py     # DataPreprocessing class (.run() + .transform())
│   ├── feature_engineering.py  # split, SMOTE, StandardScaler
│   ├── model.py             # ModelTrainer — fit + bundled save
│   ├── evaluation.py        # metrics, confusion matrix, ROC curves
│   ├── main.py              # pipeline orchestrator with inline MLflow logging
│   ├── app.py               # FastAPI inference UI
│   ├── templates/index.html
│   └── static/{style.css, app.js}
├── playground/eda.ipynb
├── requirements.txt
├── .gitignore               # excludes data/artifacts (DVC handles them)
└── README.md
```

---

## 🚀 Quick start

### 1. Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` (optional, only for remote tracking):

```bash
MLFLOW_TRACKING_URI=https://dagshub.com/<user>/<repo>.mlflow
MLFLOW_TRACKING_USERNAME=<user>
MLFLOW_TRACKING_PASSWORD=<dagshub-token>
```

### 2. Train the model

```bash
python -m src.main                          # uses cached data if present
python -m src.main --force-download         # re-pull from Kaggle
python -m src.main --run-name "baseline"    # custom MLflow run name
```

Produces:
- `artifacts/model.joblib` — single-file bundle (model + preprocessor + scaler + metadata)
- `artifacts/metrics.json` — train/test accuracy, precision, recall, F1, ROC-AUC
- `artifacts/confusion_matrix.png` — train + test side-by-side
- `artifacts/roc_curve.png` — train + test overlay
- MLflow run with params, metrics, datasets, and signed model

### 3. Launch the UI

```bash
uvicorn src.app:app --reload --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000**.

The UI offers:
- **Collapsible left sidebar** — model name, hyperparameters, train/test metrics (tabbed)
- **Main form** — 11 raw features with sensible defaults
- **Animated result card** — probability meter + confidence %

### 4. View experiments (MLflow UI)

```bash
mlflow ui --port 5000
```
Open **http://127.0.0.1:5000**. (Skip if you've set a remote `MLFLOW_TRACKING_URI` → use DagsHub UI instead.)

---

## 🔬 Sample inputs

The model is most confident on these (verified):

**🔴 High Risk (prediction = 1, ~96% confidence)**
> Income `9046786`, Age `43`, Experience `8`, Single, Rented, No car,
> `Civil_engineer`, `Bhalswa_Jahangir_Pur`, Delhi, Job yrs `4`, House yrs `11`

**🟢 Low Risk (prediction = 0, ~93% confidence)**
> Income `1753350`, Age `77`, Experience `10`, Married, Rented, No car,
> `Technician`, `Jalandhar`, Punjab, Job yrs `10`, House yrs `14`

---

## ⚙️ Configuration

All knobs live in **`config/params.yaml`** — no need to edit Python.

```yaml
dataset:
  name: subhamjain/loan-prediction-based-on-customer-behavior
  raw_path: data/raw
  train_file: Training Data.csv
  test_file: Test Data.csv

split:
  test_size: 0.2
  random_state: 42
  target_column: Risk_Flag
  stratify: true

balancing:
  method: smote
  sampling_strategy: not majority

scaling:
  method: standard

model:
  name: catboost
  params:
    iterations: 500
    learning_rate: 0.05
    depth: 6
    random_seed: 42
    verbose: 0

artifacts:
  path: artifacts
  model_file: model.joblib
  metrics_file: metrics.json

mlflow:
  experiment_name: loan_default_prediction
  tracking_uri: ""   # falls back to MLFLOW_TRACKING_URI env var, then ./mlruns
```

Categorical encoding rules → **`config/encoding.yaml`** (one-hot vs target encoder per column).

---

## 🧪 API reference

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | HTML UI |
| `/api/model-info` | GET | Bundle metadata + metrics JSON |
| `/api/predict` | POST | Score a single application (returns `prediction`, `probability_default`, `confidence`) |

**Example:**
```bash
curl -X POST http://127.0.0.1:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Income": 9046786, "Age": 43, "Experience": 8,
    "Married/Single": "single", "House_Ownership": "rented", "Car_Ownership": "no",
    "Profession": "Civil_engineer", "CITY": "Bhalswa_Jahangir_Pur", "STATE": "Delhi",
    "CURRENT_JOB_YRS": 4, "CURRENT_HOUSE_YRS": 11
  }'
```

---

## 📚 Tech stack

`Python 3.10+` · `scikit-learn` · `CatBoost` · `imbalanced-learn` · `pandas` · `MLflow` · `DVC` · `FastAPI` · `Uvicorn` · `Jinja2` · `joblib` · `matplotlib` · `kagglehub`

