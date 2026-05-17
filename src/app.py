"""FastAPI app for serving loan-default predictions with a polished UI.

Run:
    uvicorn src.app:app --reload --port 8000

Open http://localhost:8000
"""
from __future__ import annotations

import json
import os
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from src.config import MODEL_PATH, METRICS_PATH

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

app = FastAPI(title="Loan Default Risk Predictor")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


# ---- Load bundle + metrics once at startup ----
def _load_bundle() -> dict[str, Any]:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model bundle not found at {MODEL_PATH}. Run `python -m src.main` first."
        )
    return joblib.load(MODEL_PATH)


def _load_metrics() -> dict[str, Any]:
    if not os.path.exists(METRICS_PATH):
        return {}
    with open(METRICS_PATH) as f:
        return json.load(f)


BUNDLE = _load_bundle()
METRICS = _load_metrics()
MODEL = BUNDLE["model"]
PREPROCESSOR = BUNDLE["preprocessor"]
SCALER = BUNDLE.get("scaler")
FEATURE_COLUMNS: list[str] = BUNDLE.get("feature_columns") or []
MODEL_NAME: str = BUNDLE.get("model_name", "model")
MODEL_PARAMS: dict = BUNDLE.get("model_params", {}) or {}


# ---- Input schema (raw columns the user fills in the form) ----
class LoanApplication(BaseModel):
    Income: float = Field(..., ge=0)
    Age: int = Field(..., ge=18, le=100)
    Experience: int = Field(..., ge=0, le=80)
    Married_Single: str = Field(..., alias="Married/Single")
    House_Ownership: str
    Car_Ownership: str
    Profession: str
    CITY: str
    STATE: str
    CURRENT_JOB_YRS: int = Field(..., ge=0, le=80)
    CURRENT_HOUSE_YRS: int = Field(..., ge=0, le=80)

    model_config = {"populate_by_name": True}


def _predict(payload: dict[str, Any]) -> dict[str, Any]:
    """Run the full inference pipeline: raw dict -> preprocessor -> scaler -> model."""
    # Normalise the alias key
    if "Married_Single" in payload and "Married/Single" not in payload:
        payload["Married/Single"] = payload.pop("Married_Single")

    df = pd.DataFrame([payload])
    X = PREPROCESSOR.transform(df)
    if SCALER is not None:
        X = pd.DataFrame(SCALER.transform(X), columns=X.columns, index=X.index)

    pred = int(MODEL.predict(X)[0])
    proba = None
    if hasattr(MODEL, "predict_proba"):
        proba = float(MODEL.predict_proba(X)[0, 1])

    label = "High Risk (Default Likely)" if pred == 1 else "Low Risk (Likely to Repay)"
    return {
        "prediction": pred,
        "label": label,
        "probability_default": proba,
        "confidence": round((proba if pred == 1 else 1 - proba) * 100, 2) if proba is not None else None,
    }


# ---- Routes ----
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "model_name": MODEL_NAME,
            "model_params": MODEL_PARAMS,
            "metrics": METRICS,
            "feature_count": len(FEATURE_COLUMNS),
        },
    )


@app.get("/api/model-info")
def model_info():
    return {
        "model_name": MODEL_NAME,
        "model_params": MODEL_PARAMS,
        "feature_count": len(FEATURE_COLUMNS),
        "metrics": METRICS,
    } # type: ignore


@app.post("/api/predict")
def predict(app_in: LoanApplication):
    try:
        result = _predict(app_in.model_dump(by_alias=True))
        return result
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
