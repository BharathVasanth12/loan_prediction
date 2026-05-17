import os
import joblib
from catboost import CatBoostClassifier
import pandas as pd

from src.config import ARTIFACTS_PATH, MODEL_CONFIG, MODEL_PATH
from src.logger import logging, log_section
from src.preprocessing import DataPreprocessing
from src.feature_engineering import FeatureEngineer

class ModelTrainer:
    """Train and persist the configured classifier as a single bundle."""

    def __init__(self):
        self.model_name: str = MODEL_CONFIG.get("name", "catboost").lower()
        self.params: dict = dict(MODEL_CONFIG.get("params", {}) or {})
        self.model: CatBoostClassifier | None = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> CatBoostClassifier:
        log_section("Step: fit (CatBoost)")
        if self.model_name != "catboost":
            raise ValueError(
                f"Model '{self.model_name}' not supported. Only 'catboost' is wired up."
            )
        self.model = CatBoostClassifier(**self.params)
        logging.info(f"Built CatBoostClassifier with params: {self.params}")
        self.model.fit(X_train, y_train)
        logging.info(f"Model trained on {X_train.shape[0]} rows")
        return self.model

    def save(self, preprocessor: DataPreprocessing, feature_columns: list[str], scaler=None) -> None:
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

    def run(self, X_train: pd.DataFrame, y_train: pd.Series, preprocessor: DataPreprocessing, feature_columns: list[str], scaler=None) -> CatBoostClassifier:
        self.fit(X_train, y_train)
        self.save(preprocessor=preprocessor, feature_columns=feature_columns, scaler=scaler)
        assert self.model is not None
        return self.model


if __name__ == "__main__":
    # 1. Preprocess (fits encoders, outlier bounds, skew transformers)
    preprocessor = DataPreprocessing(dataset_type="train")
    preprocessor.run()

    # 2. Feature engineering (split + balance)
    fe = FeatureEngineer()
    X_train, X_test, y_train, y_test = fe.run()

    # 3. Train + bundle save
    trainer = ModelTrainer()
    trainer.run(
        X_train=X_train,
        y_train=y_train,
        preprocessor=preprocessor,
        feature_columns=list(X_train.columns),
        scaler=fe.scaler,
    )
