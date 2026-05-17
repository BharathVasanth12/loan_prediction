import os
from typing import Any
import pandas as pd
import yaml
from src.logger import logging

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_yaml_config(filename: str) -> dict[str, Any]:
    """Load a YAML config file from the project's config/ directory."""
    path = os.path.join(BASE_DIR, '..', 'config', filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_dataset(dataset_type: str) -> pd.DataFrame:
    """Load a dataset by logical name.

    Supported values:
        - 'train'           : raw training CSV
        - 'test'            : raw test CSV
        - 'train_processed' : preprocessed training CSV
        - 'test_processed'  : preprocessed test CSV
    """
    # Imported lazily to avoid circular import with src.config
    from src.config import (
        TRAIN_DATA_PATH,
        TEST_DATA_PATH,
        TRAIN_PROCESSED_PATH,
        TEST_PROCESSED_PATH,
    )

    paths = {
        "train": TRAIN_DATA_PATH,
        "test": TEST_DATA_PATH,
        "train_processed": TRAIN_PROCESSED_PATH,
        "test_processed": TEST_PROCESSED_PATH,
    }
    if dataset_type not in paths:
        raise ValueError(
            f"Invalid dataset type: '{dataset_type}'. Expected one of {list(paths)}"
        )

    path = paths[dataset_type]
    if not os.path.exists(path):
        hint = (
            " Run preprocessing first (python -m src.preprocessing)."
            if dataset_type.endswith("_processed")
            else ""
        )
        raise FileNotFoundError(f"{dataset_type} dataset not found at {path}.{hint}")

    df = pd.read_csv(path)
    logging.info(f"Loaded '{dataset_type}' dataset: {df.shape} from {path}")
    return df