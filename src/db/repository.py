"""
Repository layer -- all database access goes through here.

Keeping SQL out of the pipeline code means the retraining logic can be unit
tested against SQLite while production runs on Postgres. `DATABASE_URL`
selects the backend; nothing else in the codebase needs to know which is live.
"""

from __future__ import annotations

import datetime as dt
import os
from contextlib import contextmanager
from typing import Iterable, Iterator, Optional

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from src.db.models import Base, DriftRun, FeatureDrift, ModelVersion, PredictionBatch

DEFAULT_URL = "sqlite:///driftguard.db"


def get_engine(url: Optional[str] = None):
    """Create an engine. Postgres in prod, SQLite for tests and local runs."""
    url = url or os.getenv("DATABASE_URL", DEFAULT_URL)
    # SQLite needs check_same_thread off for the FastAPI/Streamlit threads.
    kwargs = {"future": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    return create_engine(url, **kwargs)


class DriftRepository:
    """Transaction-scoped data access for the monitoring store."""

    def __init__(self, url: Optional[str] = None):
        self.engine = get_engine(url)
        self._Session = sessionmaker(bind=self.engine, expire_on_commit=False)

    def create_schema(self) -> None:
        Base.metadata.create_all(self.engine)

    def drop_schema(self) -> None:
        Base.metadata.drop_all(self.engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        s = self._Session()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    # ----------------------------------------------------------------- models

    def register_model(
        self,
        version: int,
        algorithm: str,
        n_train_rows: int,
        trained_on_batches: int,
        val_accuracy: float,
        val_f1: float,
        val_roc_auc: float,
        artifact_path: str,
        mlflow_run_id: Optional[str] = None,
    ) -> int:
        """Insert a new model version. Returns its surrogate id."""
        with self.session() as s:
            mv = ModelVersion(
                version=version,
                algorithm=algorithm,
                n_train_rows=n_train_rows,
                trained_on_batches=trained_on_batches,
                val_accuracy=val_accuracy,
                val_f1=val_f1,
                val_roc_auc=val_roc_auc,
                artifact_path=artifact_path,
                mlflow_run_id=mlflow_run_id,
            )
            s.add(mv)
            s.flush()
            return mv.id

    def promote(self, model_id: int, reason: str) -> None:
        """Make `model_id` the sole champion, demoting any incumbent."""
        with self.session() as s:
            incumbent = s.execute(
                select(ModelVersion).where(ModelVersion.is_champion.is_(True))
            ).scalars().all()
            for old in incumbent:
                old.is_champion = False

            new = s.get(ModelVersion, model_id)
            if new is None:
                raise ValueError(f"model_version id={model_id} does not exist")
            new.is_champion = True
            new.promoted_at = dt.datetime.now(dt.timezone.utc)
            new.promotion_reason = reason[:256]

    def get_champion(self) -> Optional[ModelVersion]:
        with self.session() as s:
            return s.execute(
                select(ModelVersion).where(ModelVersion.is_champion.is_(True))
            ).scalars().first()

    def next_version(self) -> int:
        with self.session() as s:
            latest = s.execute(
                select(ModelVersion.version).order_by(ModelVersion.version.desc())
            ).scalars().first()
            return (latest or 0) + 1

    # ------------------------------------------------------------ drift runs

    def log_drift_run(
        self,
        batch_index: int,
        model_version_id: Optional[int],
        n_features_tested: int,
        n_features_drifted: int,
        dataset_drift_detected: bool,
        batch_accuracy: Optional[float],
        batch_f1: Optional[float],
        retrain_triggered: bool,
        trigger_reason: str,
        feature_results: Iterable[dict],
    ) -> int:
        """Persist one monitoring cycle plus its per-feature detail rows."""
        share = (
            n_features_drifted / n_features_tested if n_features_tested else 0.0
        )
        with self.session() as s:
            run = DriftRun(
                batch_index=batch_index,
                model_version_id=model_version_id,
                n_features_tested=n_features_tested,
                n_features_drifted=n_features_drifted,
                drift_share=share,
                dataset_drift_detected=dataset_drift_detected,
                batch_accuracy=batch_accuracy,
                batch_f1=batch_f1,
                retrain_triggered=retrain_triggered,
                trigger_reason=trigger_reason[:256],
            )
            s.add(run)
            s.flush()

            for fr in feature_results:
                s.add(
                    FeatureDrift(
                        drift_run_id=run.id,
                        feature_name=fr["feature_name"],
                        test_name=fr["test_name"],
                        statistic=fr.get("statistic"),
                        p_value=fr.get("p_value"),
                        threshold=fr["threshold"],
                        drift_detected=fr["drift_detected"],
                        psi=fr.get("psi"),
                    )
                )
            return run.id

    def log_predictions(
        self,
        batch_index: int,
        model_version_id: int,
        n_rows: int,
        accuracy: float,
        f1: float,
        roc_auc: Optional[float],
        mean_latency_ms: Optional[float] = None,
    ) -> None:
        with self.session() as s:
            s.add(
                PredictionBatch(
                    batch_index=batch_index,
                    model_version_id=model_version_id,
                    n_rows=n_rows,
                    accuracy=accuracy,
                    f1=f1,
                    roc_auc=roc_auc,
                    mean_latency_ms=mean_latency_ms,
                )
            )

    # --------------------------------------------------------------- queries

    def timeline(self) -> list[dict]:
        """Batch-ordered monitoring history -- powers the dashboard chart."""
        with self.session() as s:
            rows = s.execute(
                select(DriftRun).order_by(DriftRun.batch_index)
            ).scalars().all()
            return [
                {
                    "batch_index": r.batch_index,
                    "n_features_drifted": r.n_features_drifted,
                    "drift_share": r.drift_share,
                    "dataset_drift_detected": r.dataset_drift_detected,
                    "batch_accuracy": r.batch_accuracy,
                    "batch_f1": r.batch_f1,
                    "retrain_triggered": r.retrain_triggered,
                    "trigger_reason": r.trigger_reason,
                    "model_version_id": r.model_version_id,
                }
                for r in rows
            ]

    def most_drifted_features(self, limit: int = 10) -> list[dict]:
        """Rank features by how often they tripped the drift test."""
        with self.session() as s:
            rows = s.execute(select(FeatureDrift)).scalars().all()
        counts: dict[str, dict] = {}
        for r in rows:
            e = counts.setdefault(
                r.feature_name, {"feature": r.feature_name, "times_drifted": 0, "n": 0}
            )
            e["n"] += 1
            if r.drift_detected:
                e["times_drifted"] += 1
        out = sorted(counts.values(), key=lambda d: -d["times_drifted"])
        for e in out:
            e["drift_rate"] = round(e["times_drifted"] / e["n"], 3) if e["n"] else 0.0
        return out[:limit]

    def model_history(self) -> list[dict]:
        with self.session() as s:
            rows = s.execute(
                select(ModelVersion).order_by(ModelVersion.version)
            ).scalars().all()
            return [
                {
                    "version": m.version,
                    "algorithm": m.algorithm,
                    "n_train_rows": m.n_train_rows,
                    "val_accuracy": m.val_accuracy,
                    "val_f1": m.val_f1,
                    "val_roc_auc": m.val_roc_auc,
                    "is_champion": m.is_champion,
                    "promotion_reason": m.promotion_reason,
                }
                for m in rows
            ]
