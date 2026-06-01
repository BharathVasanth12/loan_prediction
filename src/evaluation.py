import json
import os
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for pipeline runs
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)

from src.config import (
    ARTIFACTS_PATH,
    METRICS_PATH,
    CM_PLOT_PATH,
    ROC_PLOT_PATH,
    MODEL_PATH,
    X_TRAIN_PATH,
    X_TEST_PATH,
    Y_TRAIN_PATH,
    Y_TEST_PATH,
)
from src.logger import logging, log_section


class ModelEvaluator:
    """Compute classification metrics and persist them to disk."""

    def __init__(self, model, average: str = "weighted"):
        self.model = model
        self.average = average

    def _metrics(self, y_true: pd.Series, y_pred, y_proba=None) -> dict:
        out = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average=self.average, zero_division=0),
            "recall": recall_score(y_true, y_pred, average=self.average, zero_division=0),
            "f1": f1_score(y_true, y_pred, average=self.average, zero_division=0),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }
        if y_proba is not None:
            try:
                out["roc_auc"] = roc_auc_score(y_true, y_proba)
            except ValueError as e:
                logging.warning(f"ROC-AUC could not be computed: {e}")
        return out

    def evaluate(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series,) -> dict:
        log_section("Step: evaluate (compute metrics)")
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)

        y_proba_train = None
        y_proba_test = None
        if hasattr(self.model, "predict_proba"):
            y_proba_train = self.model.predict_proba(X_train)[:, 1]
            y_proba_test = self.model.predict_proba(X_test)[:, 1]

        report = {
            "train": self._metrics(y_train, y_pred_train, y_proba_train),
            "test": self._metrics(y_test, y_pred_test, y_proba_test),
        }
        logging.info(
            f"Accuracy: Train={report['train']['accuracy']:.4f} | Test={report['test']['accuracy']:.4f} | "
            f"F1: Train={report['train']['f1']:.4f} | Test={report['test']['f1']:.4f} | "
            f"Precision: Train={report['train']['precision']:.4f} | Test={report['test']['precision']:.4f} | "
            f"Recall: Train={report['train']['recall']:.4f} | Test={report['test']['recall']:.4f} | "
            f"ROC-AUC: Train={report['train'].get('roc_auc', 'N/A')} | Test={report['test'].get('roc_auc', 'N/A')}"
        )
        return report

    def save(self, report: dict) -> None:
        log_section("Step: save metrics")
        os.makedirs(ARTIFACTS_PATH, exist_ok=True)
        with open(METRICS_PATH, "w") as f:
            json.dump(report, f, indent=2)
        logging.info(f"Metrics saved to {METRICS_PATH}")

    def save_plots(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        cmap: str = "Purples",
    ) -> dict:
        """Save confusion matrix (train+test side-by-side) and ROC curve (train+test overlay) PNGs."""
        log_section("Step: save plots (CM + ROC)")
        os.makedirs(ARTIFACTS_PATH, exist_ok=True)

        # Confusion matrix — train and test side-by-side
        fig_cm, (ax_tr, ax_te) = plt.subplots(1, 2, figsize=(10, 4))
        ConfusionMatrixDisplay.from_estimator(
            self.model, X_train, y_train, ax=ax_tr, cmap=cmap, colorbar=False
        )
        ax_tr.set_title("Confusion Matrix (Train)")
        ConfusionMatrixDisplay.from_estimator(
            self.model, X_test, y_test, ax=ax_te, cmap=cmap, colorbar=False
        )
        ax_te.set_title("Confusion Matrix (Test)")
        fig_cm.tight_layout()
        fig_cm.savefig(CM_PLOT_PATH, dpi=120)
        plt.close(fig_cm)
        logging.info(f"Confusion matrix (train+test) saved to {CM_PLOT_PATH}")

        # ROC curve: train + test on the same axes
        roc_path = None
        if hasattr(self.model, "predict_proba"):
            fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
            RocCurveDisplay.from_estimator(
                self.model, X_train, y_train, ax=ax_roc, name="Train",
                curve_kwargs={"color": "steelblue"},
            )
            RocCurveDisplay.from_estimator(
                self.model, X_test, y_test, ax=ax_roc, name="Test",
                curve_kwargs={"color": "darkorange"},
            )
            ax_roc.plot([0, 1], [0, 1], "--", color="grey", linewidth=1, label="Chance")
            ax_roc.set_title("ROC Curve (Train vs Test)")
            ax_roc.legend(loc="lower right")
            fig_roc.tight_layout()
            fig_roc.savefig(ROC_PLOT_PATH, dpi=120)
            plt.close(fig_roc)
            roc_path = ROC_PLOT_PATH
            logging.info(f"ROC curve (train+test) saved to {ROC_PLOT_PATH}")
        else:
            logging.warning("Model has no predict_proba; skipping ROC curve")

        return {"confusion_matrix": CM_PLOT_PATH, "roc_curve": roc_path}

    def run(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series,) -> dict:
        report = self.evaluate(X_train, y_train, X_test, y_test)
        self.save(report)
        self.save_plots(X_train, y_train, X_test, y_test)
        return report

if __name__ == "__main__":
    # DVC stage entry point: loads trained bundle + splits from disk,
    # writes metrics.json + confusion matrix + ROC plots.
    import joblib

    X_train = pd.read_csv(X_TRAIN_PATH)
    X_test = pd.read_csv(X_TEST_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH).squeeze("columns")
    y_test = pd.read_csv(Y_TEST_PATH).squeeze("columns")

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"] if isinstance(bundle, dict) and "model" in bundle else bundle

    evaluator = ModelEvaluator(model)
    evaluator.run(X_train, y_train, X_test, y_test)
    logging.info(f"Metrics written to '{METRICS_PATH}'")
