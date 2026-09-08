"""
FastAPI service: serve the champion model and expose monitoring state.

Endpoints
    GET  /health              liveness + which model version is serving
    POST /predict             score records with the current champion
    GET  /monitoring/timeline drift + performance history
    GET  /monitoring/features feature drift frequency ranking
    GET  /models              the model registry
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.data.loader import FEATURES
from src.db.repository import DriftRepository

app = FastAPI(
    title="DriftGuard",
    description="Drift-monitored model serving with a champion-challenger registry",
    version="1.0.0",
)

_repo: Optional[DriftRepository] = None
_model = None
_model_version: Optional[int] = None


def repo() -> DriftRepository:
    global _repo
    if _repo is None:
        _repo = DriftRepository(os.getenv("DATABASE_URL"))
        _repo.create_schema()
    return _repo


def load_champion():
    """Lazy-load the champion artifact named by the registry."""
    global _model, _model_version
    champ = repo().get_champion()
    if champ is None:
        return None, None
    if _model_version != champ.version:
        path = Path(champ.artifact_path)
        if not path.exists():
            raise HTTPException(503, f"artifact missing on disk: {path}")
        _model = joblib.load(path)
        _model_version = champ.version
    return _model, _model_version


class Record(BaseModel):
    nswprice: float = Field(..., ge=0, le=1)
    nswdemand: float = Field(..., ge=0, le=1)
    vicprice: float = Field(..., ge=0, le=1)
    vicdemand: float = Field(..., ge=0, le=1)
    transfer: float = Field(..., ge=0, le=1)
    period: float = Field(..., ge=0, le=1)
    day: int = Field(..., ge=1, le=7)


class PredictRequest(BaseModel):
    records: list[Record]


@app.get("/health")
def health():
    champ = repo().get_champion()
    return {
        "status": "ok",
        "champion_version": champ.version if champ else None,
        "champion_val_accuracy": champ.val_accuracy if champ else None,
    }


@app.post("/predict")
def predict(req: PredictRequest):
    model, version = load_champion()
    if model is None:
        raise HTTPException(503, "no champion model registered; run the pipeline first")

    df = pd.DataFrame([r.model_dump() for r in req.records])[FEATURES]
    t0 = time.perf_counter()
    preds = model.predict(df)
    probs = model.predict_proba(df)[:, 1]
    latency_ms = (time.perf_counter() - t0) * 1000

    return {
        "model_version": version,
        "n_records": len(df),
        "latency_ms": round(latency_ms, 2),
        "predictions": [
            {"prediction": int(p), "probability_up": round(float(q), 4)}
            for p, q in zip(preds, probs)
        ],
    }


@app.get("/monitoring/timeline")
def timeline():
    return {"timeline": repo().timeline()}


@app.get("/monitoring/features")
def features(limit: int = 10):
    return {"features": repo().most_drifted_features(limit)}


@app.get("/models")
def models():
    return {"models": repo().model_history()}
