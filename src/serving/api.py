"""
FastAPI REST API for churn prediction serving.
Run: uvicorn src.serving.api:app --reload --port 8000
"""
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── App Setup ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Production ML API for predicting customer churn using XGBoost/LightGBM",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global State ───────────────────────────────────────────────────────────
model = None
preprocessor = None
config = None
prediction_log = []  # In-memory log (use DB in production)


def load_artifacts():
    """Load model and preprocessor at startup."""
    global model, preprocessor, config

    try:
        with open("config.yaml") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        config = {
            "serving": {"model_path": "models/best_model.pkl", "preprocessor_path": "models/preprocessor.pkl"},
            "training": {"threshold": 0.5},
        }

    model_path = config["serving"]["model_path"]
    preprocessor_path = config["serving"]["preprocessor_path"]

    if Path(model_path).exists() and Path(preprocessor_path).exists():
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        logger.info(f"Loaded model from {model_path}")
        logger.info(f"Loaded preprocessor from {preprocessor_path}")
    else:
        logger.warning(
            "Model artifacts not found. Run the training pipeline first:\n"
            "  python scripts/run_pipeline.py"
        )


@app.on_event("startup")
async def startup_event():
    load_artifacts()


# ── Request / Response Schemas ─────────────────────────────────────────────
class CustomerFeatures(BaseModel):
    customer_id: str = Field(..., example="CUST001")
    tenure: int = Field(..., ge=0, le=100, example=12)
    monthly_charges: float = Field(..., ge=0, le=500, example=65.5)
    total_charges: float = Field(..., ge=0, example=786.0)
    contract_type: str = Field(..., example="Month-to-month")
    payment_method: str = Field(..., example="Electronic check")
    internet_service: str = Field(..., example="Fiber optic")
    gender: str = Field(default="Male", example="Male")
    senior_citizen: str = Field(default="No", example="No")
    num_products: int = Field(default=2, ge=1, le=10, example=2)
    has_tech_support: bool = Field(default=False, example=False)
    has_streaming: bool = Field(default=True, example=True)
    has_online_security: bool = Field(default=False, example=False)
    has_online_backup: bool = Field(default=True, example=True)
    paperless_billing: bool = Field(default=True, example=True)
    num_support_tickets: int = Field(default=1, ge=0, example=3)
    avg_call_duration: float = Field(default=8.0, ge=0, example=8.5)
    days_since_last_contact: int = Field(default=30, ge=0, example=45)

    class Config:
        json_schema_extra = {
            "example": {
                "customer_id": "CUST001",
                "tenure": 12,
                "monthly_charges": 65.5,
                "total_charges": 786.0,
                "contract_type": "Month-to-month",
                "payment_method": "Electronic check",
                "internet_service": "Fiber optic",
                "gender": "Male",
                "senior_citizen": "No",
                "num_products": 2,
                "has_tech_support": False,
                "has_streaming": True,
                "has_online_security": False,
                "has_online_backup": True,
                "paperless_billing": True,
                "num_support_tickets": 3,
                "avg_call_duration": 8.5,
                "days_since_last_contact": 45,
            }
        }


class PredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    churn_prediction: bool
    risk_level: str
    confidence: float
    timestamp: str


class BatchRequest(BaseModel):
    customers: List[CustomerFeatures]


class BatchResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_customers: int
    high_risk_count: int
    processing_time_ms: float


# ── Helper Functions ────────────────────────────────────────────────────────
def customer_to_dataframe(customer: CustomerFeatures) -> pd.DataFrame:
    data = customer.dict()
    # Convert booleans to int (matches training pipeline)
    for col in ["has_tech_support", "has_streaming", "has_online_security",
                "has_online_backup", "paperless_billing"]:
        if col in data:
            data[col] = int(data[col])
    return pd.DataFrame([data])


def classify_risk(probability: float) -> str:
    if probability >= 0.75:
        return "HIGH"
    elif probability >= 0.50:
        return "MEDIUM"
    elif probability >= 0.25:
        return "LOW"
    else:
        return "VERY_LOW"


def predict_single(customer: CustomerFeatures) -> PredictionResponse:
    from src.features.engineering import engineer_features

    df = customer_to_dataframe(customer)
    df_eng = engineer_features(df)

    threshold = config.get("training", {}).get("threshold", 0.5)

    X = preprocessor.transform(df_eng)
    prob = float(model.predict_proba(X)[0, 1])
    prediction = prob >= threshold
    risk = classify_risk(prob)
    confidence = max(prob, 1 - prob)

    return PredictionResponse(
        customer_id=customer.customer_id,
        churn_probability=round(prob, 4),
        churn_prediction=prediction,
        risk_level=risk,
        confidence=round(confidence, 4),
        timestamp=datetime.utcnow().isoformat(),
    )


# ── Endpoints ──────────────────────────────────────────────────────────────
@app.get("/health", tags=["Health"])
def health_check():
    """API health check endpoint."""
    model_loaded = model is not None and preprocessor is not None
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
    }


@app.get("/model/info", tags=["Model"])
def model_info():
    """Get information about the loaded model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training pipeline first.")

    model_type = type(model).__name__
    metrics_path = "models/metrics_summary.json"
    metrics = {}
    if Path(metrics_path).exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

    return {
        "model_type": model_type,
        "model_path": config["serving"]["model_path"],
        "performance_metrics": metrics,
        "threshold": config.get("training", {}).get("threshold", 0.5),
    }


@app.get("/metrics", tags=["Model"])
def get_metrics():
    """Get latest model performance metrics."""
    metrics_path = "models/metrics_summary.json"
    if not Path(metrics_path).exists():
        raise HTTPException(status_code=404, detail="Metrics not found. Run training pipeline.")
    with open(metrics_path) as f:
        return json.load(f)


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(customer: CustomerFeatures):
    """Predict churn probability for a single customer."""
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run: python scripts/run_pipeline.py first.",
        )
    try:
        result = predict_single(customer)
        prediction_log.append(result.dict())
        return result
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchResponse, tags=["Prediction"])
def predict_batch(batch: BatchRequest):
    """Predict churn for a batch of customers."""
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run: python scripts/run_pipeline.py first.",
        )
    if len(batch.customers) == 0:
        raise HTTPException(status_code=400, detail="Empty batch")
    if len(batch.customers) > 1000:
        raise HTTPException(status_code=400, detail="Batch size limit: 1000")

    start = time.time()
    predictions = []
    for customer in batch.customers:
        try:
            pred = predict_single(customer)
            predictions.append(pred)
        except Exception as e:
            logger.error(f"Error predicting for {customer.customer_id}: {e}")

    elapsed_ms = (time.time() - start) * 1000
    high_risk = sum(1 for p in predictions if p.risk_level == "HIGH")

    return BatchResponse(
        predictions=predictions,
        total_customers=len(predictions),
        high_risk_count=high_risk,
        processing_time_ms=round(elapsed_ms, 2),
    )


@app.get("/predictions/recent", tags=["Monitoring"])
def recent_predictions(limit: int = 50):
    """Get recent prediction log (in-memory)."""
    return {
        "total_predictions": len(prediction_log),
        "recent": prediction_log[-limit:],
    }


@app.post("/model/reload", tags=["Model"])
def reload_model():
    """Reload model artifacts from disk."""
    try:
        load_artifacts()
        return {"status": "success", "message": "Model reloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
