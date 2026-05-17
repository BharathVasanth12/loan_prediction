import os
from dotenv import load_dotenv

from src.utils import load_yaml_config

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))

# Load .env (optional, for secrets only — API keys, DB URLs, etc.)
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

# Project parameters (versioned in git)
_params = load_yaml_config("params.yaml")
_dataset = _params["dataset"]
_processed = _params["processed"]
_split = _params["split"]

DATASET_NAME: str = _dataset["name"]
RAW_PATH: str = os.path.join(PROJECT_ROOT, _dataset["raw_path"])
TRAIN_DATA_PATH: str = os.path.join(RAW_PATH, _dataset["train_file"])
TEST_DATA_PATH: str = os.path.join(RAW_PATH, _dataset["test_file"])

PROCESSED_PATH: str = os.path.join(PROJECT_ROOT, _processed["path"])
TRAIN_PROCESSED_PATH: str = os.path.join(PROCESSED_PATH, _processed["train_file"])
TEST_PROCESSED_PATH: str = os.path.join(PROCESSED_PATH, _processed["test_file"])

TEST_SIZE: float = float(_split["test_size"])
RANDOM_STATE: int = int(_split["random_state"])
TARGET_COLUMN: str = _split["target_column"]
STRATIFY: bool = bool(_split.get("stratify", True))

# Feature engineering & model configs
BALANCING_CONFIG = _params["balancing"]
SCALING_CONFIG = _params["scaling"]
MODEL_CONFIG = _params["model"]

# Artifact paths
_artifacts = _params["artifacts"]
ARTIFACTS_PATH: str = os.path.join(PROJECT_ROOT, _artifacts["path"])
SCALER_PATH: str = os.path.join(ARTIFACTS_PATH, _artifacts["scaler_file"])
MODEL_PATH: str = os.path.join(ARTIFACTS_PATH, _artifacts["model_file"])
METRICS_PATH: str = os.path.join(ARTIFACTS_PATH, _artifacts["metrics_file"])
CM_PLOT_PATH: str = os.path.join(ARTIFACTS_PATH, "confusion_matrix.png")
ROC_PLOT_PATH: str = os.path.join(ARTIFACTS_PATH, "roc_curve.png")

# MLflow
_mlflow = _params.get("mlflow", {}) or {}
MLFLOW_EXPERIMENT: str = _mlflow.get("experiment_name", "loan_default_prediction")
# Tracking URI: prefer env var (e.g. DagsHub) over yaml; fall back to local ./mlruns
MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI") or _mlflow.get("tracking_uri", "")

# Preprocessing config
ENCODING_CONFIG = load_yaml_config("encoding.yaml")