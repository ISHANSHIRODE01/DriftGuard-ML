"""
Student Performance API
=======================
FastAPI application for serving student grade predictions.

Endpoints:
- POST /predict: Predict G3 grade based on student features
- GET /metrics: View latest model performance metrics
- GET /drift-status: Check for data drift in the system
"""

import joblib
import json
import pandas as pd
import sys
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Dict, Any, List

# --- Deployment Patch: Add root to path ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from preprocessing.pipeline import clean_raw_data # Centralized Logic
from database.repository import AsyncMLRepository
from database.connection import db_manager

# --- API Initialization ---
app = FastAPI(
    title="Student Performance Prediction API",
    description="API for predicting final student grades (G3) with drift monitoring.",
    version="1.0.0"
)

# Initialize Repository
repo = AsyncMLRepository()

# --- Model Paths ---
MODEL_PATH = Path("model/model.pkl")
PREPROCESSOR_PATH = Path("model/preprocessor.pkl")
METRICS_PATH = Path("model/metrics.json")
DRIFT_PATH = Path("Data/reports/drift_report.json")
METADATA_PATH = Path("model/metadata.json")

# Global variables for persistent loading
model = None
preprocessor = None
current_version = "unknown"

# --- Pydantic Data Model ---

# --- Pydantic Data Model ---
class StudentData(BaseModel):
    school: str = Field(..., description="Student school (GP or MS)")
    sex: str = Field(..., description="Student sex (F or M)")
    age: int = Field(..., ge=15, le=22)
    address: str = Field(..., description="Urban (U) or Rural (R)")
    famsize: str = Field(..., description="Family size (LE3 or GT3)")
    Pstatus: str = Field(..., description="Parent status (T or A)")
    Medu: int = Field(..., ge=0, le=4)
    Fedu: int = Field(..., ge=0, le=4)
    Mjob: str
    Fjob: str
    reason: str
    guardian: str
    traveltime: int = Field(..., ge=1, le=4)
    studytime: int = Field(..., ge=1, le=4)
    failures: int = Field(..., ge=0, le=4)
    schoolsup: str = Field(..., description="yes or no")
    famsup: str = Field(..., description="yes or no")
    paid: str = Field(..., description="yes or no")
    activities: str = Field(..., description="yes or no")
    nursery: str = Field(..., description="yes or no")
    higher: str = Field(..., description="yes or no")
    internet: str = Field(..., description="yes or no")
    romantic: str = Field(..., description="yes or no")
    famrel: int = Field(..., ge=1, le=5)
    freetime: int = Field(..., ge=1, le=5)
    goout: int = Field(..., ge=1, le=5)
    Dalc: int = Field(..., ge=1, le=5)
    Walc: int = Field(..., ge=1, le=5)
    health: int = Field(..., ge=1, le=5)
    absences: int = Field(..., ge=0, le=93)
    G1: int = Field(..., ge=0, le=20)
    G2: int = Field(..., ge=0, le=20)

    class Config:
        json_schema_extra = {
            "example": {
                "school": "GP", "sex": "F", "age": 18, "address": "U", "famsize": "GT3",
                "Pstatus": "A", "Medu": 4, "Fedu": 4, "Mjob": "at_home", "Fjob": "teacher",
                "reason": "course", "guardian": "mother", "traveltime": 2, "studytime": 2,
                "failures": 0, "schoolsup": "yes", "famsup": "no", "paid": "no",
                "activities": "no", "nursery": "yes", "higher": "yes", "internet": "no",
                "romantic": "no", "famrel": 4, "freetime": 3, "goout": 4, "Dalc": 1,
                "Walc": 1, "health": 3, "absences": 6, "G1": 5, "G2": 6
            }
        }

# --- Event Handlers ---
@app.on_event("startup")
async def startup_event():
    """Load model and connect to DB on startup."""
    global model, preprocessor, current_version
    
    # 1. Connect to MongoDB
    db_manager.connect()
    
    # 2. Load model and preprocessor
    try:
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        
        # Load version from metadata if exists
        if METADATA_PATH.exists():
            with open(METADATA_PATH, 'r') as f:
                meta = json.load(f)
                current_version = f"v{meta.get('current_production_version', '1')}"
        
        print(f"[OK] Model ({current_version}) and Preprocessor loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load model artifacts: {e}")

@app.on_event("shutdown")
def shutdown_event():
    """Close DB connection on shutdown."""
    db_manager.close()

# --- Endpoints ---

@app.post("/predict")
async def predict(data: StudentData):
    """
    Predict the final grade (G3) for a student.
    """
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded on server.")

    try:
        # 1. Convert Pydantic model to DataFrame
        df = pd.DataFrame([data.dict()])

        # 2. Preprocess Raw Data (Centralized)
        df_cleaned = clean_raw_data(df)

        # 3. Transform data using saved preprocessor
        X_processed = preprocessor.transform(df_cleaned)

        # 4. Inference
        prediction = model.predict(X_processed)
        result_val = float(prediction[0])

        # 5. Log to MongoDB (Fire and Forget)
        try:
            await repo.log_prediction(
                features=data.dict(),
                prediction=result_val,
                model_version=current_version
            )
        except Exception as db_err:
            print(f"[DB Warning] Failed to log prediction: {db_err}")

        return {
            "status": "success",
            "prediction": {
                "G3_score": result_val,
                "rounding": int(round(result_val))
            },
            "model_version": current_version
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Return the latest model performance metrics."""
    if not METRICS_PATH.exists():
        raise HTTPException(status_code=404, detail="Metrics file not found.")
    
    with open(METRICS_PATH, "r") as f:
        return json.load(f)

@app.get("/drift-status")
async def get_drift_status():
    """Return the status of the last drift detection run."""
    if not DRIFT_PATH.exists():
        return {
            "status": "unknown",
            "message": "No drift detection report found. Run drift detection first."
        }
    
    with open(DRIFT_PATH, "r") as f:
        drift_report = json.load(f)
        
    return {
        "drift_detected": drift_report.get("summary", {}).get("drift_detected_overall", False),
        "drifted_features_count": drift_report.get("summary", {}).get("drifted_features_count", 0),
        "timestamp": drift_report.get("summary", {}).get("timestamp", "unknown"),
        "details": drift_report.get("details", {})
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":
    import uvicorn
    # In a real environment, run with: uvicorn app:app --reload
    uvicorn.run(app, host="0.0.0.0", port=8000)
