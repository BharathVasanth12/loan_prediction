"""End-to-end pipeline runner with inline MLflow tracking.

Usage:
    python -m src.main
    python -m src.main --force-download
    python -m src.main --run-name "gradient-boosting-baseline"

Tracking URI resolution:
    1. MLFLOW_TRACKING_URI env var (recommended for DagsHub / remote)
    2. `mlflow.tracking_uri` in params.yaml
    3. Default: local ./mlruns directory
"""
import argparse
import os
import warnings

# Silence benign MLflow noise:
#  - UCVolumeDatasetSource probe failure on local paths
#  - LocalArtifactDatasetSource ambiguity for local files
#  - Integer-column schema hint (our training data has no missing values)
warnings.filterwarnings("ignore", message=".*UCVolumeDatasetSource.*")
warnings.filterwarnings("ignore", message=".*LocalArtifactDatasetSource.*")
warnings.filterwarnings("ignore", message=".*Inferred schema contains integer column.*")

# IMPORTANT: import src.config BEFORE mlflow so that load_dotenv() runs and
# MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD are present in os.environ
# when mlflow first reads them.
from src.config import (
    DATASET_NAME,
    PROCESSED_PATH,
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    TRAIN_PROCESSED_PATH,
    MODEL_PATH,
    METRICS_PATH,
    CM_PLOT_PATH,
    ROC_PLOT_PATH,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    MODEL_CONFIG,
    BALANCING_CONFIG,
    SCALING_CONFIG,
    TEST_SIZE,
    RANDOM_STATE,
    STRATIFY,
    TARGET_COLUMN,
)

import mlflow
from mlflow.models import infer_signature
from src.data_ingestion import DataIngestion
from src.preprocessing import DataPreprocessing
from src.feature_engineering import FeatureEngineer
from src.model import ModelTrainer
from src.evaluation import ModelEvaluator
from src.logger import logging, log_section


def run_pipeline(force_download: bool = False, run_name: str | None = None) -> dict:
    log_section("PIPELINE START", char="=")

    # ---- MLflow setup ----
    if MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=run_name) as run:
        logging.info(f"MLflow run started: {run.info.run_id} (exp='{MLFLOW_EXPERIMENT}')")

        # ---- Log params (config snapshot) ----
        params = {
            "dataset_name": DATASET_NAME,
            "target_column": TARGET_COLUMN,
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "stratify": STRATIFY,
            "balancing_method": BALANCING_CONFIG.get("method"),
            "scaling_method": SCALING_CONFIG.get("method"),
            "model_name": MODEL_CONFIG.get("name"),
            **{f"model__{k}": v for k, v in (MODEL_CONFIG.get("params") or {}).items()},
        }
        mlflow.log_params(params)

        # 0. Data ingestion
        log_section("[0/4] DataIngestion.run()")
        raw_present = os.path.exists(TRAIN_DATA_PATH) and os.path.exists(TEST_DATA_PATH)
        if force_download or not raw_present:
            DataIngestion(DATASET_NAME).run()
        else:
            logging.info("Raw data present; skipping download")

        # 1. Preprocessing (train only)
        log_section("[1/4] DataPreprocessing.run()")
        os.makedirs(PROCESSED_PATH, exist_ok=True)
        preprocessor = DataPreprocessing(dataset_type="train")
        processed_df = preprocessor.run()
        processed_df.to_csv(TRAIN_PROCESSED_PATH, index=False)

        # ---- Log dataset (raw + processed) ----
        try:
            raw_ds = mlflow.data.from_pandas(
                preprocessor.df if hasattr(preprocessor, "df") else processed_df,
                source=TRAIN_DATA_PATH,
                name="train_raw",
                targets=TARGET_COLUMN,
            )
            processed_ds = mlflow.data.from_pandas(
                processed_df,
                source=TRAIN_PROCESSED_PATH,
                name="train_processed",
                targets=TARGET_COLUMN,
            )
            mlflow.log_input(raw_ds, context="raw")
            mlflow.log_input(processed_ds, context="processed")
            logging.info("MLflow: logged train_raw + train_processed datasets")
        except Exception as e:
            logging.warning(f"MLflow dataset logging failed: {e}")

        # 2. Feature engineering
        log_section("[2/4] FeatureEngineer.run()")
        fe = FeatureEngineer()
        X_train, X_test, y_train, y_test = fe.run()

        # 3. Model training + bundled save
        log_section("[3/4] ModelTrainer.run()")
        trainer = ModelTrainer()
        model = trainer.run(
            X_train=X_train,
            y_train=y_train,
            preprocessor=preprocessor,
            feature_columns=list(X_train.columns),
            scaler=fe.scaler,
        )

        # 4. Evaluation
        log_section("[4/4] ModelEvaluator.run()")
        evaluator = ModelEvaluator(model=model)
        report = evaluator.run(X_train, y_train, X_test, y_test)

        # ---- Log metrics (flattened train_/test_ prefixed) ----
        flat_metrics = {
            f"{split}_{k}": float(v)
            for split in ("train", "test")
            for k, v in report[split].items()
            if isinstance(v, (int, float))
        }
        mlflow.log_metrics(flat_metrics)

        # ---- Log artifacts (bundle, metrics.json, CM plot, ROC plot) ----
        for path in (MODEL_PATH, METRICS_PATH, CM_PLOT_PATH, ROC_PLOT_PATH):
            if path and os.path.exists(path):
                mlflow.log_artifact(path)

        # ---- Log model with signature + input example ----
        X_sample = X_train.head(5)
        signature = infer_signature(X_sample, model.predict(X_sample))
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=X_sample,
        )
        logging.info(f"MLflow: logged {len(params)} params, {len(flat_metrics)} metrics, model with signature")

    log_section("PIPELINE COMPLETE", char="=")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full ML pipeline end-to-end.")
    parser.add_argument("--force-download", action="store_true",
                        help="Re-download raw data from Kaggle even if local files exist.")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Optional MLflow run name.")
    args = parser.parse_args()
    run_pipeline(force_download=args.force_download, run_name=args.run_name)
