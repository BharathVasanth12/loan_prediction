import os

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    OneHotEncoder,
    PowerTransformer,
    TargetEncoder,
)

from src.config import (
    ARTIFACTS_PATH,
    ENCODING_CONFIG,
    PREPROCESSOR_PATH,
    PROCESSED_PATH,
    TRAIN_PROCESSED_PATH,
)
from src.logger import log_section, logging
from src.utils import load_dataset


class DataPreprocessing:
    def __init__(self, dataset_type: str):
        self.dataset_type = dataset_type
        self.df = load_dataset(self.dataset_type)

        # Fitted artifacts kept here so the same objects can transform test/prod data
        self.fitted_encoders: dict = {}
        self.outlier_bounds: dict[str, tuple[float, float]] = {}
        self.skew_transformers: dict[str, PowerTransformer] = {}
        self.feature_columns: list[str] = []

    def drop_unused_columns(self):
        """Drop columns listed under `drop_columns` in encoding.yaml (e.g., Id)."""
        log_section("Step: drop_unused_columns")
        df_copy = self.df.copy()
        cols = ENCODING_CONFIG.get("drop_columns", []) or []
        cols_present = [c for c in cols if c in df_copy.columns]
        if cols_present:
            df_copy = df_copy.drop(columns=cols_present)
            logging.info(f"Dropped unused columns: {cols_present}")
        missing = set(cols) - set(cols_present)
        if missing:
            logging.warning(f"drop_columns entries not in dataframe: {missing}")
        return df_copy

    def drop_duplicates(self):
        """
        Remove duplicate rows from the DataFrame.

        Returns:
            pd.DataFrame: DataFrame without duplicate rows
        """
        log_section("Step: drop_duplicates")
        initial_rows = len(self.df) # type: ignore
        df_clean = self.df.drop_duplicates().reset_index(drop=True) # type: ignore
        duplicates_removed = initial_rows - len(df_clean) # type: ignore

        if duplicates_removed > 0:
            logging.info(
                f"Removed {duplicates_removed} duplicate rows ({duplicates_removed / initial_rows * 100:.2f}%)"
            )
            logging.info(f"Dataset size: {initial_rows} → {len(df_clean)} rows")
        else:
            logging.info("No duplicate rows found")

        return df_clean

    def handle_missing_values(self, strategy: str = "auto") -> pd.DataFrame:
        """
        Handle missing values in the DataFrame using intelligent strategies based on data types.

        For numeric columns:
            - If outliers exist: use median (robust to outliers)
            - If no outliers: use mean
        For categorical columns:
            - Always use mode (most frequent value)

        For columns with >25% missing values:
            - Drop the column (too sparse to impute reliably)

        Args:
            strategy (str): Strategy to handle missing values
                        'auto' - intelligent choice based on data type and outliers
                        'drop' - drop rows with missing values

        Returns:
            pd.DataFrame: DataFrame with missing values handled
        """
        log_section("Step: handle_missing_values")
        logging.info(f"Starting missing value imputation with strategy: '{strategy}'")
        total_missing = self.df.isnull().sum().sum()
        logging.info(f"Total missing values: {total_missing}")

        if strategy == "drop":
            initial_rows = len(self.df)
            df_clean = self.df.dropna().reset_index(drop=True)
            rows_dropped = initial_rows - len(df_clean)
            logging.info(f"Dropped {rows_dropped} rows with missing values")
            return df_clean

        df_copy = self.df.copy()

        # Check for columns with >25% missing values and drop them
        missing_threshold = 0.25
        total_rows = len(df_copy)
        cols_to_drop: list[str] = []

        for col in df_copy.columns:
            missing_pct = df_copy[col].isnull().sum() / total_rows
            if missing_pct > missing_threshold:
                cols_to_drop.append(col)
                logging.warning(
                    f"Column '{col}' has {missing_pct * 100:.2f}% missing values (>{missing_threshold * 100}%) → DROPPING column"
                )

        if cols_to_drop:
            df_copy = df_copy.drop(columns=cols_to_drop)
            logging.info(f"Dropped {len(cols_to_drop)} columns with >25% missing values")
        else:
            logging.info("No columns with >25% missing values to drop")

        # Separate numeric and categorical columns
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        categorical_cols = df_copy.select_dtypes(include=["object", "category"]).columns

        logging.info(
            f"Found {len(numeric_cols)} numeric columns and {len(categorical_cols)} categorical columns"
        )

        # Handle numeric columns
        for col in numeric_cols:
            if df_copy[col].isnull().any():
                missing_count = df_copy[col].isnull().sum()
                missing_pct = missing_count / total_rows * 100

                # Check for outliers using IQR method
                Q1 = df_copy[col].quantile(0.25)
                Q3 = df_copy[col].quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Check if outliers exist
                has_outliers = ((df_copy[col] < lower_bound) | (df_copy[col] > upper_bound)).any()
                outlier_count = ((df_copy[col] < lower_bound) | (df_copy[col] > upper_bound)).sum()

                if has_outliers:
                    # Use median for columns with outliers
                    median_val = df_copy[col].median()
                    df_copy[col].fillna(median_val, inplace=True)
                    logging.info(
                        f"Column '{col}': {missing_count} missing values ({missing_pct:.2f}%) → imputed with MEDIAN ({median_val:.2f})"
                    )
                    logging.info(
                        f"  Reason: {outlier_count} outliers detected (robust to extreme values)"
                    )
                else:
                    # Use mean for columns without outliers
                    mean_val = df_copy[col].mean()
                    df_copy[col].fillna(mean_val, inplace=True)
                    logging.info(
                        f"Column '{col}': {missing_count} missing values ({missing_pct:.2f}%) → imputed with MEAN ({mean_val:.2f})"
                    )
                    logging.info(f"  Reason: No outliers detected (normal distribution)")

        # Handle categorical columns - always use mode
        for col in categorical_cols:
            if df_copy[col].isnull().any():
                missing_count = df_copy[col].isnull().sum()
                missing_pct = missing_count / total_rows * 100
                mode_value = df_copy[col].mode()
                if len(mode_value) > 0:
                    df_copy[col].fillna(mode_value[0], inplace=True)
                    logging.info(
                        f"Column '{col}': {missing_count} missing values ({missing_pct:.2f}%) → imputed with MODE ('{mode_value[0]}')"
                    )
                    logging.info(f"  Reason: Categorical data - using most frequent value")

        logging.info("Missing value imputation completed successfully")
        return df_copy

    def handle_outliers(self, exclude_cols: list[str] | None = None):
        """Handle outliers in numeric columns using the IQR capping method.

        Values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR] are clipped to those bounds.

        Args:
            exclude_cols: columns to skip (e.g., the target or ID columns).

        Returns:
            pd.DataFrame: DataFrame with outliers capped.
        """
        log_section("Step: handle_outliers")
        exclude_cols = exclude_cols or []
        df_copy = self.df.copy()
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col in exclude_cols:
                logging.info(f"Skipping column '{col}' (excluded from outlier capping)")
                continue

            Q1 = df_copy[col].quantile(0.25)
            Q3 = df_copy[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = (df_copy[col] < lower_bound) | (df_copy[col] > upper_bound)
            outlier_count = int(outliers.sum())

            if outlier_count > 0:
                df_copy[col] = df_copy[col].clip(lower=lower_bound, upper=upper_bound)
                outlier_pct = outlier_count / len(df_copy) * 100
                logging.info(
                    f"Column '{col}': capped {outlier_count} outliers ({outlier_pct:.2f}%) "
                    f"to [{lower_bound:.2f}, {upper_bound:.2f}]"
                )
            else:
                logging.info(f"Column '{col}': no outliers detected")

            # Persist bounds so inference can clip with the same limits
            self.outlier_bounds[col] = (float(lower_bound), float(upper_bound))
        logging.info(f"Outlier bounds saved for {len(self.outlier_bounds)} columns")
        logging.info("Outlier handling completed successfully")
        return df_copy

    def handle_skewness(self, exclude_cols: list[str] | None = None):
        """Handle skewness in numeric columns using log transformation.

        Applies log1p (log(1+x)) to reduce skewness. Only applied to columns with skewness > 0.5.

        Args:
            exclude_cols: columns to skip (e.g., the target or ID columns).

        Returns:
            pd.DataFrame: DataFrame with skewed columns transformed.
        """
        log_section("Step: handle_skewness")
        exclude_cols = exclude_cols or []
        df_copy = self.df.copy()
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col in exclude_cols:
                logging.info(f"Skipping column '{col}' (excluded from skewness handling)")
                continue

            skewness = df_copy[col].skew()
            if abs(skewness) > 0.5:
                pt = PowerTransformer(method="yeo-johnson", standardize=False)
                df_copy[col] = pt.fit_transform(df_copy[[col]])
                self.skew_transformers[col] = pt
                logging.info(
                    f"Column '{col}': applied Yeo-Johnson transformation to reduce skewness ({skewness:.2f} → {df_copy[col].skew():.2f})"
                )
            else:
                logging.info(
                    f"Column '{col}': no significant skewness detected (skewness={skewness:.2f})"
                )
        logging.info(f"Skew transformers applied: {self.skew_transformers}")
        logging.info("Skewness handling completed successfully")
        return df_copy

    def encoding(self, target: pd.Series | None = None):
        """Encode categorical variables based on the plan defined in encoding.yaml.

        Runs all configured encoders sequentially: onehot -> target.

        Args:
            target: target Series (required only if 'target' encoder is configured).

        Returns:
            pd.DataFrame: DataFrame with categorical variables encoded.
        """
        log_section("Step: encoding")
        df_copy = self.df.copy()
        encoders_cfg = ENCODING_CONFIG.get("encoders", {})

        # Fixed execution order regardless of YAML key order
        for encoder_name in ("onehot", "target"):
            spec = encoders_cfg.get(encoder_name)
            if not spec:
                continue

            columns = spec.get("columns", [])
            params = dict(spec.get("params", {}) or {})

            cols_present = [c for c in columns if c in df_copy.columns]
            missing = set(columns) - set(cols_present)
            if missing:
                logging.warning(f"Encoder '{encoder_name}': columns not in dataframe: {missing}")
            if not cols_present:
                continue

            if encoder_name == "onehot":
                ohe = OneHotEncoder(**params)
                encoded = ohe.fit_transform(df_copy[cols_present])
                encoded_df = pd.DataFrame(
                    encoded,
                    columns=ohe.get_feature_names_out(cols_present),
                    index=df_copy.index,
                )
                df_copy = pd.concat([df_copy.drop(columns=cols_present).reset_index(drop=True), encoded_df], axis=1)
                self.fitted_encoders["onehot"] = ohe
                logging.info(f"One-Hot encoded: {cols_present}")

            elif encoder_name == "target":
                if target is None:
                    raise ValueError(
                        "Target encoder configured but no `target` argument provided to encoding()"
                    )
                te = TargetEncoder(**params)
                df_copy[cols_present] = te.fit_transform(df_copy[cols_present], target)
                self.fitted_encoders["target"] = te
                logging.info(f"Target encoded: {cols_present}")

        logging.info("Encoding completed successfully")
        return df_copy

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted preprocessing pipeline to new data (inference).

        Uses persisted `outlier_bounds`, `skew_transformers`, `fitted_encoders`, and
        `feature_columns` — no re-fitting. Output is reindexed to `feature_columns`
        so the column order/shape exactly matches what the model was trained on.
        """
        df_out = df.copy()

        # 1. Drop unused columns (e.g., Id / ID)
        for col in ENCODING_CONFIG.get("drop_columns", []) or []:
            for variant in (col, col.lower(), col.upper(), col.capitalize()):
                if variant in df_out.columns:
                    df_out = df_out.drop(columns=[variant])
                    break

        # 2. Clip numeric columns with the training bounds
        for col, (lo, hi) in self.outlier_bounds.items():
            if col in df_out.columns:
                df_out[col] = df_out[col].clip(lower=lo, upper=hi)

        # 3. Apply skewness transformers
        for col, pt in self.skew_transformers.items():
            if col in df_out.columns:
                df_out[col] = pt.transform(df_out[[col]])

        # 4. Apply encoders (onehot then target)
        encoders_cfg = ENCODING_CONFIG.get("encoders", {})

        if "onehot" in self.fitted_encoders:
            spec = encoders_cfg.get("onehot", {})
            cols = [c for c in spec.get("columns", []) if c in df_out.columns]
            if cols:
                ohe = self.fitted_encoders["onehot"]
                encoded = ohe.transform(df_out[cols])
                encoded_df = pd.DataFrame(
                    encoded,
                    columns=ohe.get_feature_names_out(cols),
                    index=df_out.index,
                )
                df_out = pd.concat([df_out.drop(columns=cols).reset_index(drop=True), encoded_df], axis=1)

        if "target" in self.fitted_encoders:
            spec = encoders_cfg.get("target", {})
            cols = [c for c in spec.get("columns", []) if c in df_out.columns]
            if cols:
                te = self.fitted_encoders["target"]
                df_out[cols] = te.transform(df_out[cols])

        # 5. Align to the training feature contract
        if self.feature_columns:
            for c in self.feature_columns:
                if c not in df_out.columns:
                    df_out[c] = 0
            df_out = df_out[self.feature_columns]

        return df_out

    def run(self):
        logging.info(f"Starting preprocessing for {self.dataset_type} dataset...")
        target_col = ENCODING_CONFIG.get("target_column")
        # Target must be excluded from numeric transforms; otherwise its 0/1 values
        # get treated as outliers and clipped, destroying the minority class.
        exclude = [target_col] if target_col else []
        self.df = self.drop_unused_columns()
        self.df = self.drop_duplicates()
        self.df = self.handle_missing_values()
        self.df = self.handle_outliers(exclude_cols=exclude)
        self.df = self.handle_skewness(exclude_cols=exclude)
        target_series = (
            self.df[target_col] if target_col and target_col in self.df.columns else None
        )
        self.df = self.encoding(target=target_series)
        # Final column order excluding target — used as the inference contract
        self.feature_columns = [c for c in self.df.columns if c != target_col]
        logging.info(f"Preprocessing for {self.dataset_type} dataset completed successfully.")
        return self.df


if __name__ == "__main__":
    # DVC stage entry point: writes processed CSV + pickled fitted preprocessor.
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    os.makedirs(ARTIFACTS_PATH, exist_ok=True)

    preprocessor = DataPreprocessing(dataset_type="train")
    processed_df = preprocessor.run()

    processed_df.to_csv(TRAIN_PROCESSED_PATH, index=False)
    logging.info(
        f"Processed train data saved to '{TRAIN_PROCESSED_PATH}' (shape={processed_df.shape})"
    )

    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    logging.info(f"Fitted preprocessor saved to '{PREPROCESSOR_PATH}'")
