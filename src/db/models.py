"""
SQLAlchemy schema for the DriftGuard monitoring store.

Four tables model the full lifecycle of a monitored model:

    model_versions  - the model registry. One row per trained artifact.
    drift_runs      - one row per monitoring cycle over an incoming batch.
    feature_drift   - per-feature statistical test results for a drift run.
    predictions     - per-batch scored performance, joined to the model used.

The schema is deliberately normalised so that "which model was serving when
feature X drifted?" is a single SQL join rather than a file-system hunt.
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class ModelVersion(Base):
    """A single trained model artifact -- the model registry."""

    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True)
    version = Column(Integer, nullable=False, unique=True)
    algorithm = Column(String(64), nullable=False)

    trained_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    trained_on_batches = Column(Integer, nullable=False, default=0)
    n_train_rows = Column(Integer, nullable=False, default=0)

    # Hold-out metrics captured at training time.
    val_accuracy = Column(Float)
    val_f1 = Column(Float)
    val_roc_auc = Column(Float)

    # Registry state: only one row may be champion at a time.
    is_champion = Column(Boolean, default=False, nullable=False, index=True)
    promoted_at = Column(DateTime(timezone=True))
    promotion_reason = Column(String(256))

    artifact_path = Column(String(512))
    mlflow_run_id = Column(String(64))

    drift_runs = relationship("DriftRun", back_populates="model_version")
    predictions = relationship("PredictionBatch", back_populates="model_version")

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        flag = " CHAMPION" if self.is_champion else ""
        return f"<ModelVersion v{self.version} {self.algorithm}{flag}>"


class DriftRun(Base):
    """One monitoring cycle: compare an incoming batch to the reference window."""

    __tablename__ = "drift_runs"

    id = Column(Integer, primary_key=True)
    batch_index = Column(Integer, nullable=False, index=True)
    run_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)

    model_version_id = Column(Integer, ForeignKey("model_versions.id"), nullable=True)

    n_features_tested = Column(Integer, nullable=False, default=0)
    n_features_drifted = Column(Integer, nullable=False, default=0)
    drift_share = Column(Float, nullable=False, default=0.0)

    # Dataset-level verdict. True when drift_share crosses the configured gate.
    dataset_drift_detected = Column(Boolean, nullable=False, default=False)

    # Observed performance on this batch, used for the retrain decision.
    batch_accuracy = Column(Float)
    batch_f1 = Column(Float)

    retrain_triggered = Column(Boolean, nullable=False, default=False)
    trigger_reason = Column(String(256))

    model_version = relationship("ModelVersion", back_populates="drift_runs")
    feature_results = relationship(
        "FeatureDrift", back_populates="drift_run", cascade="all, delete-orphan"
    )

    __table_args__ = (Index("ix_drift_runs_batch_run", "batch_index", "run_at"),)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"<DriftRun batch={self.batch_index} "
            f"drifted={self.n_features_drifted}/{self.n_features_tested}>"
        )


class FeatureDrift(Base):
    """Per-feature statistical test result belonging to one drift run."""

    __tablename__ = "feature_drift"

    id = Column(Integer, primary_key=True)
    drift_run_id = Column(Integer, ForeignKey("drift_runs.id"), nullable=False)

    feature_name = Column(String(128), nullable=False)
    test_name = Column(String(32), nullable=False)  # ks_2samp | chi2 | psi
    statistic = Column(Float)
    p_value = Column(Float)
    threshold = Column(Float, nullable=False)
    drift_detected = Column(Boolean, nullable=False, default=False)

    # Effect size, so we can rank *how much* a feature moved, not just whether.
    psi = Column(Float)

    drift_run = relationship("DriftRun", back_populates="feature_results")

    __table_args__ = (
        UniqueConstraint("drift_run_id", "feature_name", name="uq_run_feature"),
        Index("ix_feature_drift_name", "feature_name"),
    )


class PredictionBatch(Base):
    """Aggregated scoring record for one batch served by one model version."""

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True)
    batch_index = Column(Integer, nullable=False, index=True)
    scored_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)

    model_version_id = Column(Integer, ForeignKey("model_versions.id"), nullable=False)

    n_rows = Column(Integer, nullable=False)
    accuracy = Column(Float)
    f1 = Column(Float)
    roc_auc = Column(Float)
    mean_latency_ms = Column(Float)

    model_version = relationship("ModelVersion", back_populates="predictions")
