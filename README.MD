# Finance Modular Project

A modular end-to-end data science project demonstrating data loading, preprocessing, model training, evaluation and prediction export.


## Module summaries

- src/data_loader.py
  - Responsibilities: read raw CSVs, validate required columns, apply basic type conversions, and return pandas DataFrames for training and test phases. Raises clear errors when files or expected columns are missing.

- src/preprocessing.py
  - Responsibilities: define and build the preprocessing pipeline (imputation, scaling, categorical encoding), split features and target, perform train/validation split, and prepare test features with the same fitted transformers used for training.

- src/model.py
  - Responsibilities: construct the ML model (or estimator pipeline), encapsulate training logic, persist model artifacts (optional), and expose a predict interface compatible with the prediction module.

- src/evaluation.py
  - Responsibilities: compute evaluation metrics (e.g., RMSE, MAE, R^2), produce summary reports or dictionaries of results, and provide helper functions to compare models or log metrics.

- src/prediction.py
  - Responsibilities: load the trained model and preprocessor, run predictions on prepared test features, attach predictions to identifiers, and write the final CSV to PREDICTIONS_PATH.

- src/main.py
  - Responsibilities: pipeline orchestration — load data, build preprocessor and model, train, evaluate, and call prediction export. Implements CLI or script entrypoint (python -m src.main).

- src/config.py / .env
  - Responsibilities: centralize dataset paths, output paths, random seed, test split fraction and other configurable parameters so modules remain decoupled from environment specifics.


## Setup
1. Create a Python 3.8+ virtual environment and activate it.
2. Install dependencies:
```sh
pip install -r requirements.txt
```

## Configuration
Edit `.env` or `src/config.py` to adjust dataset paths, test split and random seed. Key variables:
- TRAIN_DATA_PATH, TEST_DATA_PATH, PREDICTIONS_PATH
- TEST_SIZE, RANDOM_STATE

## Running the pipeline
Run the full pipeline which loads data, builds the preprocessor and model, trains, evaluates and writes test predictions:
```sh
python -m src.main
```

The pipeline entrypoint is `main.run_pipeline`. Intermediate helpers live in the modules linked above.

## Output
Predictions are saved to the path defined by PREDICTIONS_PATH (default: `data/predictions/test_predictions.csv`).


Home work:

1. tryout to improve the model
2. complete all the previous tasks
3. practice modular coding