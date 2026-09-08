"""
Model training with MLflow experiment tracking.

The validation split is the *last* 20% of the training window, never a random
sample. Random validation on time-series data leaks future rows into the
training set and inflates the score -- exactly the failure mode that makes a
monitoring system report healthy while production degrades.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

try:  # MLflow is optional so the pipeline still runs in a bare container.
    import mlflow

    _MLFLOW = True
except ImportError:  # pragma: no cover
    _MLFLOW = False


@dataclass
class TrainResult:
    model: RandomForestClassifier
    version: int
    algorithm: str
    n_train_rows: int
    val_accuracy: float
    val_f1: float
    val_roc_auc: float
    artifact_path: str
    mlflow_run_id: Optional[str] = None


def _temporal_split(X: pd.DataFrame, y: pd.Series, val_fraction: float = 0.2):
    """Chronological hold-out: train on the past, validate on the recent tail."""
    split = int(len(X) * (1 - val_fraction))
    if split < 10 or split >= len(X):
        raise ValueError(f"training window too small to split: n={len(X)}")
    return (
        X.iloc[:split],
        X.iloc[split:],
        y.iloc[:split],
        y.iloc[split:],
    )


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    version: int,
    artifact_dir: str | Path = "artifacts",
    n_estimators: int = 200,
    max_depth: Optional[int] = 12,
    random_state: int = 42,
    experiment: str = "driftguard-elec2",
    log_to_mlflow: bool = True,
) -> TrainResult:
    """Fit a classifier, score it on a temporal hold-out, persist the artifact."""
    X_tr, X_val, y_tr, y_val = _temporal_split(X, y)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=random_state,
        class_weight="balanced_subsample",
    )
    model.fit(X_tr, y_tr)

    pred = model.predict(X_val)
    proba = model.predict_proba(X_val)[:, 1]

    acc = float(accuracy_score(y_val, pred))
    f1 = float(f1_score(y_val, pred, zero_division=0))
    try:
        auc = float(roc_auc_score(y_val, proba))
    except ValueError:  # single-class validation slice
        auc = float("nan")

    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / f"model_v{version}.joblib"
    joblib.dump(model, artifact_path)

    run_id = None
    if log_to_mlflow and _MLFLOW:
        run_id = _log_mlflow(
            experiment, version, model, X_tr, acc, f1, auc,
            n_estimators, max_depth, artifact_path,
        )

    return TrainResult(
        model=model,
        version=version,
        algorithm="RandomForestClassifier",
        n_train_rows=len(X_tr),
        val_accuracy=acc,
        val_f1=f1,
        val_roc_auc=auc,
        artifact_path=str(artifact_path),
        mlflow_run_id=run_id,
    )


def _log_mlflow(
    experiment, version, model, X_tr, acc, f1, auc,
    n_estimators, max_depth, artifact_path,
) -> Optional[str]:
    """Record params, metrics and feature importances for one training run."""
    try:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=f"model_v{version}") as run:
            mlflow.log_params(
                {
                    "version": version,
                    "algorithm": "RandomForestClassifier",
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "n_train_rows": len(X_tr),
                    "split": "temporal_80_20",
                }
            )
            mlflow.log_metrics(
                {"val_accuracy": acc, "val_f1": f1, "val_roc_auc": auc}
            )
            for name, imp in sorted(
                zip(X_tr.columns, model.feature_importances_),
                key=lambda t: -t[1],
            ):
                mlflow.log_metric(f"importance__{name}", float(imp))
            mlflow.log_artifact(str(artifact_path), artifact_path="model")
            return run.info.run_id
    except Exception as exc:  # tracking must never break the pipeline
        print(f"[mlflow] logging skipped: {exc}")
        return None


def evaluate(model, X: pd.DataFrame, y: pd.Series) -> dict:
    """Score a fitted model on a batch."""
    pred = model.predict(X)
    out = {
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": None,
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y, model.predict_proba(X)[:, 1]))
    except (ValueError, AttributeError):
        pass
    return out
