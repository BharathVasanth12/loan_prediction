import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from src.config import (
    TARGET_COLUMN,
    TEST_SIZE,
    RANDOM_STATE,
    STRATIFY,
    BALANCING_CONFIG,
    SCALING_CONFIG,
    ARTIFACTS_PATH,
    SCALER_PATH,
    SPLITS_DIR,
    X_TRAIN_PATH,
    X_TEST_PATH,
    Y_TRAIN_PATH,
    Y_TEST_PATH,
)
from src.utils import load_dataset
from src.logger import logging, log_section


class FeatureEngineer:
    """Prepare modelling-ready feature matrices from the processed dataset."""

    def __init__(self, dataset_type: str = "train_processed"):
        self.dataset_type = dataset_type
        self.scaler: StandardScaler | None = None

    def split_x_y(self, df: pd.DataFrame):
        log_section("Step: split_x_y")
        if TARGET_COLUMN not in df.columns:
            raise KeyError(f"Target column '{TARGET_COLUMN}' not in dataframe")
        X = df.drop(columns=[TARGET_COLUMN])
        y = df[TARGET_COLUMN]
        logging.info(f"Split features/target: X={X.shape}, y={y.shape}")
        return X, y

    def train_test_split(self, X: pd.DataFrame, y: pd.Series):
        log_section("Step: train_test_split")
        stratify = y if STRATIFY else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=stratify,
        )
        logging.info(
            f"Train/test split: train={X_train.shape}, test={X_test.shape}"
        )
        return X_train, X_test, y_train, y_test

    def balance_target_column(self, X_train: pd.DataFrame, y_train: pd.Series):
        log_section("Step: balance_target_column (SMOTE)")
        method = BALANCING_CONFIG.get("method", "smote").lower()
        if method != "smote":
            logging.info(f"Balancing method '{method}' not supported; skipping")
            return X_train, y_train

        sampler = SMOTE(
            sampling_strategy=BALANCING_CONFIG.get("sampling_strategy", "not majority"),
            random_state=BALANCING_CONFIG.get("random_state", RANDOM_STATE),
        )
        X_res, y_res = sampler.fit_resample(X_train, y_train)
        logging.info(
            f"SMOTE: before={y_train.value_counts().to_dict()} "
            f"after={pd.Series(y_res).value_counts().to_dict()}"
        )
        return X_res, y_res

    def scaling(self, X_train: pd.DataFrame, X_test: pd.DataFrame):
        log_section("Step: scaling")
        method = SCALING_CONFIG.get("method", "standard").lower()
        if method != "standard":
            logging.info(f"Scaling method '{method}' not supported; skipping")
            return X_train, X_test

        self.scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index,
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index,
        )
        logging.info("Applied StandardScaler to train/test")
        return X_train_scaled, X_test_scaled

    def run(self):
        df = load_dataset(self.dataset_type)
        X, y = self.split_x_y(df)
        X_train, X_test, y_train, y_test = self.train_test_split(X, y)
        X_train_bal, y_train_bal = self.balance_target_column(X_train, y_train)
        X_train_scaled, X_test_scaled = self.scaling(X_train_bal, X_test)
        # Scaler is NOT saved here — it ships inside the model bundle (see model.py)
        return X_train_scaled, X_test_scaled, y_train_bal, y_test

if __name__ == "__main__":
    # DVC stage entry point: writes train/test split CSVs + fitted scaler.
    os.makedirs(SPLITS_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_PATH, exist_ok=True)

    fe = FeatureEngineer()
    X_train, X_test, y_train, y_test = fe.run()

    X_train.to_csv(X_TRAIN_PATH, index=False)
    X_test.to_csv(X_TEST_PATH, index=False)
    y_train.to_csv(Y_TRAIN_PATH, index=False)
    y_test.to_csv(Y_TEST_PATH, index=False)
    logging.info(
        f"Splits saved to '{SPLITS_DIR}' (X_train={X_train.shape}, X_test={X_test.shape})"
    )

    if fe.scaler is not None:
        joblib.dump(fe.scaler, SCALER_PATH)
        logging.info(f"Fitted scaler saved to '{SCALER_PATH}'")
