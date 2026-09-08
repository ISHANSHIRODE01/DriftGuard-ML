"""
The monitoring loop: detect -> decide -> retrain -> gate -> promote.

Design decisions worth defending in an interview:

* Retraining is triggered by *either* distribution drift or a performance
  drop. Drift alone can be harmless (a shifted feature the model barely
  uses); a performance drop alone can happen without any measurable feature
  drift (label/concept drift). Monitoring only one of the two misses cases.

* A challenger never auto-promotes. It must beat the champion on the SAME
  held-out window by a minimum margin. Without that gate the system swaps
  models on noise, which is worse than not retraining -- you lose the ability
  to attribute a regression to anything.

* Retraining uses a sliding window, not all history. Under concept drift,
  stale data actively hurts: the old regime's relationship no longer holds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from src.data.loader import MONITORED_FEATURES, Batch
from src.db.repository import DriftRepository
from src.drift.detector import DriftDetector
from src.models.trainer import evaluate, train_model


@dataclass
class MonitorConfig:
    # Drift gate
    p_value_threshold: float = 0.05
    psi_threshold: float = 0.10
    dataset_drift_share: float = 0.5

    # Performance gate: retrain if batch accuracy falls this far below the
    # champion's validation accuracy.
    performance_drop_tolerance: float = 0.05

    # Promotion gate: challenger must beat champion by at least this margin.
    promotion_margin: float = 0.01

    # Sliding retrain window, in batches.
    retrain_window_batches: int = 6

    # Cool-down: minimum batches between retrains, prevents thrashing.
    min_batches_between_retrains: int = 2

    artifact_dir: str = "artifacts"
    log_to_mlflow: bool = True


@dataclass
class MonitorOutcome:
    batch_index: int
    n_features_drifted: int
    drift_share: float
    dataset_drift_detected: bool
    batch_accuracy: float
    batch_f1: float
    retrain_triggered: bool
    promoted: bool
    trigger_reason: str
    champion_version: int
    drifted_features: list[str] = field(default_factory=list)


class DriftMonitor:
    """Stateful monitor over a chronological batch stream."""

    def __init__(
        self,
        repo: DriftRepository,
        config: Optional[MonitorConfig] = None,
    ):
        self.repo = repo
        self.cfg = config or MonitorConfig()
        self.detector = DriftDetector(
            p_value_threshold=self.cfg.p_value_threshold,
            psi_threshold=self.cfg.psi_threshold,
            dataset_drift_share=self.cfg.dataset_drift_share,
        )

        self.champion = None            # fitted estimator
        self.champion_id: Optional[int] = None
        self.champion_version: int = 0
        self.champion_val_accuracy: float = 0.0

        self.reference: Optional[pd.DataFrame] = None
        self.history: list[Batch] = []  # sliding retrain window
        self._last_retrain_batch: Optional[int] = None

    # ------------------------------------------------------------ bootstrap

    def fit_initial(self, batches: list[Batch]) -> None:
        """Train the first champion and set the drift reference window."""
        X = pd.concat([b.X for b in batches], ignore_index=True)
        y = pd.concat([b.y for b in batches], ignore_index=True)

        res = train_model(
            X, y,
            version=self.repo.next_version(),
            artifact_dir=self.cfg.artifact_dir,
            log_to_mlflow=self.cfg.log_to_mlflow,
        )
        model_id = self.repo.register_model(
            version=res.version,
            algorithm=res.algorithm,
            n_train_rows=res.n_train_rows,
            trained_on_batches=len(batches),
            val_accuracy=res.val_accuracy,
            val_f1=res.val_f1,
            val_roc_auc=res.val_roc_auc,
            artifact_path=res.artifact_path,
            mlflow_run_id=res.mlflow_run_id,
        )
        self.repo.promote(model_id, "initial baseline model")

        self.champion = res.model
        self.champion_id = model_id
        self.champion_version = res.version
        self.champion_val_accuracy = res.val_accuracy

        # Reference window for drift = the data the champion learned from.
        self.reference = pd.concat([b.raw for b in batches], ignore_index=True)
        self.history = list(batches)

    # --------------------------------------------------------------- stepping

    def process(self, batch: Batch) -> MonitorOutcome:
        """Run one monitoring cycle over an incoming batch."""
        if self.champion is None or self.reference is None:
            raise RuntimeError("call fit_initial() before process()")

        # 1. Statistical drift on monitored features only.
        report = self.detector.detect(
            self.reference[MONITORED_FEATURES], batch.raw[MONITORED_FEATURES]
        )

        # 2. Observed performance of the serving model on this batch.
        perf = evaluate(self.champion, batch.X, batch.y)
        self.repo.log_predictions(
            batch_index=batch.index,
            model_version_id=self.champion_id,
            n_rows=len(batch),
            accuracy=perf["accuracy"],
            f1=perf["f1"],
            roc_auc=perf["roc_auc"],
        )

        # 3. Decide.
        perf_gap = self.champion_val_accuracy - perf["accuracy"]
        perf_breach = perf_gap > self.cfg.performance_drop_tolerance

        reasons = []
        if report.dataset_drift_detected:
            reasons.append(
                f"dataset drift ({report.n_features_drifted}/"
                f"{report.n_features_tested} features)"
            )
        if perf_breach:
            reasons.append(f"accuracy dropped {perf_gap:.3f} below baseline")

        cooling = (
            self._last_retrain_batch is not None
            and batch.index - self._last_retrain_batch
            < self.cfg.min_batches_between_retrains
        )
        should_retrain = bool(reasons) and not cooling
        if reasons and cooling:
            reasons.append("suppressed by cool-down")

        reason_text = "; ".join(reasons) if reasons else "healthy"

        promoted = False
        if should_retrain:
            promoted = self._retrain_and_gate(batch)
            self._last_retrain_batch = batch.index
            if promoted:
                reason_text += " -> challenger promoted"
            else:
                reason_text += " -> challenger rejected by gate"

        # 4. Persist the cycle.
        self.repo.log_drift_run(
            batch_index=batch.index,
            model_version_id=self.champion_id,
            n_features_tested=report.n_features_tested,
            n_features_drifted=report.n_features_drifted,
            dataset_drift_detected=report.dataset_drift_detected,
            batch_accuracy=perf["accuracy"],
            batch_f1=perf["f1"],
            retrain_triggered=should_retrain,
            trigger_reason=reason_text,
            feature_results=report.to_records(),
        )

        # 5. Batch joins history after being scored -- never before, or the
        #    model would be evaluated on data it already trained on.
        self.history.append(batch)
        self.history = self.history[-self.cfg.retrain_window_batches :]

        return MonitorOutcome(
            batch_index=batch.index,
            n_features_drifted=report.n_features_drifted,
            drift_share=report.drift_share,
            dataset_drift_detected=report.dataset_drift_detected,
            batch_accuracy=perf["accuracy"],
            batch_f1=perf["f1"],
            retrain_triggered=should_retrain,
            promoted=promoted,
            trigger_reason=reason_text,
            champion_version=self.champion_version,
            drifted_features=report.drifted_features(),
        )

    # -------------------------------------------------------------- retraining

    def _retrain_and_gate(self, batch: Batch) -> bool:
        """
        Train a challenger on the recent window and hold a fair contest.

        Both models are scored on the same window -- the most recent batch,
        which neither has trained on. Comparing a challenger's fresh
        validation score against the champion's months-old score would be
        rigged in the challenger's favour.
        """
        window = self.history[-self.cfg.retrain_window_batches :]
        X = pd.concat([b.X for b in window], ignore_index=True)
        y = pd.concat([b.y for b in window], ignore_index=True)

        if y.nunique() < 2:
            return False  # degenerate window, nothing to learn

        challenger = train_model(
            X, y,
            version=self.repo.next_version(),
            artifact_dir=self.cfg.artifact_dir,
            log_to_mlflow=self.cfg.log_to_mlflow,
        )
        challenger_id = self.repo.register_model(
            version=challenger.version,
            algorithm=challenger.algorithm,
            n_train_rows=challenger.n_train_rows,
            trained_on_batches=len(window),
            val_accuracy=challenger.val_accuracy,
            val_f1=challenger.val_f1,
            val_roc_auc=challenger.val_roc_auc,
            artifact_path=challenger.artifact_path,
            mlflow_run_id=challenger.mlflow_run_id,
        )

        # Head-to-head on the identical unseen window.
        champ_score = evaluate(self.champion, batch.X, batch.y)["accuracy"]
        chall_score = evaluate(challenger.model, batch.X, batch.y)["accuracy"]

        if chall_score >= champ_score + self.cfg.promotion_margin:
            self.repo.promote(
                challenger_id,
                f"beat champion v{self.champion_version} on batch {batch.index}: "
                f"{chall_score:.4f} vs {champ_score:.4f}",
            )
            self.champion = challenger.model
            self.champion_id = challenger_id
            self.champion_version = challenger.version
            self.champion_val_accuracy = challenger.val_accuracy

            # Reference window follows the new champion's training data.
            self.reference = pd.concat([b.raw for b in window], ignore_index=True)
            return True

        return False
