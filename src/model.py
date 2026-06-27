import os

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from src.config import (
    ARTIFACTS_PATH,
    MODEL_CONFIG,
    MODEL_PATH,
    PREPROCESSOR_PATH,
    SCALER_PATH,
    X_TRAIN_PATH,
    Y_TRAIN_PATH,
)
from src.logger import log_section, logging
from src.preprocessing import DataPreprocessing

_SUPPORTED_MODELS = {"gradient_boosting", "gradientboosting", "gb"}


class ModelTrainer:
    """Train and persist the configured classifier as a single bundle."""

    def __init__(self):
        self.model_name: str = MODEL_CONFIG.get("name", "gradient_boosting").lower()
        self.params: dict = dict(MODEL_CONFIG.get("params", {}) or {})
        self.model: GradientBoostingClassifier | None = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> GradientBoostingClassifier:
        log_section("Step: fit (GradientBoosting)")
        if self.model_name not in _SUPPORTED_MODELS:
            raise ValueError(
                f"Model '{self.model_name}' not supported. Only 'gradient_boosting' is wired up."
            )
        self.model = GradientBoostingClassifier(**self.params)
        logging.info(f"Built GradientBoostingClassifier with params: {self.params}")
        self.model.fit(X_train, y_train)
        logging.info(f"Model trained on {X_train.shape[0]} rows")
        return self.model

    def save(
        self, preprocessor: DataPreprocessing, feature_columns: list[str], scaler=None
    ) -> None:
        """Persist a single-bundle artifact: model + preprocessor + scaler + feature schema."""
        log_section("Step: save (bundle artifact)")
        if self.model is None:
            raise RuntimeError("No model to save; call fit() first.")
        os.makedirs(ARTIFACTS_PATH, exist_ok=True)
        artifact = {
            "model": self.model,
            "preprocessor": preprocessor,
            "scaler": scaler,
            "feature_columns": feature_columns,
            "model_name": self.model_name,
            "model_params": self.params,
        }
        joblib.dump(artifact, MODEL_PATH)
        logging.info(f"Bundle saved to {MODEL_PATH} (keys: {list(artifact.keys())})")

    def run(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        preprocessor: DataPreprocessing,
        feature_columns: list[str],
        scaler=None,
    ) -> GradientBoostingClassifier:
        self.fit(X_train, y_train)
        self.save(preprocessor=preprocessor, feature_columns=feature_columns, scaler=scaler)
        assert self.model is not None
        return self.model


if __name__ == "__main__":
    # DVC stage entry point: loads splits + fitted preprocessor + scaler from disk,
    # trains the model, and writes the single-bundle artifact to MODEL_PATH.
    X_train = pd.read_csv(X_TRAIN_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH).squeeze("columns")
    logging.info(f"Loaded X_train={X_train.shape}, y_train={y_train.shape}")

    preprocessor = joblib.load(PREPROCESSOR_PATH)
    scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None

    trainer = ModelTrainer()
    trainer.run(
        X_train=X_train,
        y_train=y_train,
        preprocessor=preprocessor,
        feature_columns=list(X_train.columns),
        scaler=scaler,
    )
    logging.info(f"Bundle written to '{MODEL_PATH}'")
