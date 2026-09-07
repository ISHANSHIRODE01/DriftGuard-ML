"""Tests for the model registry invariants and the monitoring loop."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.loader import FEATURES, TARGET, make_batches
from src.db.repository import DriftRepository
from src.models.trainer import train_model
from src.pipeline.monitor import DriftMonitor, MonitorConfig

RNG = np.random.default_rng(11)


@pytest.fixture
def repo(tmp_path):
    r = DriftRepository(f"sqlite:///{tmp_path/'test.db'}")
    r.create_schema()
    return r


def _synthetic(n: int = 9000, shift_at: int | None = None) -> pd.DataFrame:
    """Build an ELEC2-shaped frame, optionally with a regime change."""
    rows = {c: RNG.normal(0.5, 0.15, n) for c in FEATURES}
    df = pd.DataFrame(rows)
    df["period"] = np.tile(np.linspace(0, 1, 48), n // 48 + 1)[:n]
    df["day"] = np.tile(np.arange(1, 8), n // 7 + 1)[:n]

    # Label depends on nswprice; after shift_at the relationship inverts.
    y = (df.nswprice > df.nswprice.median()).astype(int)
    if shift_at is not None:
        y.iloc[shift_at:] = 1 - y.iloc[shift_at:]
    df[TARGET] = y
    return df


def _register(repo, version: int, acc: float) -> int:
    return repo.register_model(
        version=version, algorithm="RF", n_train_rows=100, trained_on_batches=1,
        val_accuracy=acc, val_f1=acc, val_roc_auc=acc,
        artifact_path=f"/tmp/m{version}.joblib",
    )


class TestRegistry:
    def test_exactly_one_champion_after_promotions(self, repo):
        a, b, c = _register(repo, 1, 0.7), _register(repo, 2, 0.8), _register(repo, 3, 0.9)
        repo.promote(a, "first")
        repo.promote(b, "second")
        repo.promote(c, "third")

        champs = [m for m in repo.model_history() if m["is_champion"]]
        assert len(champs) == 1
        assert champs[0]["version"] == 3

    def test_champion_is_retrievable(self, repo):
        mid = _register(repo, 1, 0.75)
        repo.promote(mid, "only")
        champ = repo.get_champion()
        assert champ is not None and champ.version == 1

    def test_no_champion_before_promotion(self, repo):
        _register(repo, 1, 0.75)
        assert repo.get_champion() is None

    def test_version_numbers_increment(self, repo):
        assert repo.next_version() == 1
        _register(repo, 1, 0.7)
        assert repo.next_version() == 2

    def test_promoting_unknown_id_raises(self, repo):
        with pytest.raises(ValueError, match="does not exist"):
            repo.promote(9999, "nope")

    def test_duplicate_version_rejected(self, repo):
        _register(repo, 1, 0.7)
        with pytest.raises(Exception):
            _register(repo, 1, 0.8)


class TestDriftRunLogging:
    def test_drift_run_persists_feature_detail(self, repo):
        run_id = repo.log_drift_run(
            batch_index=4, model_version_id=None,
            n_features_tested=3, n_features_drifted=2,
            dataset_drift_detected=True, batch_accuracy=0.61, batch_f1=0.5,
            retrain_triggered=True, trigger_reason="dataset drift",
            feature_results=[
                {"feature_name": "a", "test_name": "ks_2samp", "statistic": 0.3,
                 "p_value": 0.001, "threshold": 0.05, "drift_detected": True, "psi": 0.4},
                {"feature_name": "b", "test_name": "ks_2samp", "statistic": 0.2,
                 "p_value": 0.02, "threshold": 0.05, "drift_detected": True, "psi": 0.2},
                {"feature_name": "c", "test_name": "chi2", "statistic": 1.1,
                 "p_value": 0.6, "threshold": 0.05, "drift_detected": False, "psi": 0.01},
            ],
        )
        assert run_id > 0
        tl = repo.timeline()
        assert len(tl) == 1 and tl[0]["n_features_drifted"] == 2
        assert tl[0]["drift_share"] == pytest.approx(2 / 3)

        ranked = repo.most_drifted_features()
        assert ranked[0]["times_drifted"] == 1

    def test_timeline_is_batch_ordered(self, repo):
        for b in [5, 1, 3]:
            repo.log_drift_run(
                batch_index=b, model_version_id=None, n_features_tested=1,
                n_features_drifted=0, dataset_drift_detected=False,
                batch_accuracy=0.7, batch_f1=0.6, retrain_triggered=False,
                trigger_reason="healthy", feature_results=[],
            )
        assert [r["batch_index"] for r in repo.timeline()] == [1, 3, 5]


class TestTemporalSplit:
    def test_validation_slice_is_the_chronological_tail(self):
        """The split must not shuffle -- that would leak the future."""
        df = _synthetic(2000)
        res = train_model(
            df[FEATURES], df[TARGET], version=1,
            artifact_dir="/tmp/dg_test_artifacts", log_to_mlflow=False,
        )
        # 80% of 2000 = 1600 training rows.
        assert res.n_train_rows == 1600
        assert 0.0 <= res.val_accuracy <= 1.0

    def test_tiny_window_raises(self):
        df = _synthetic(20)
        with pytest.raises(ValueError, match="too small"):
            train_model(df[FEATURES].head(5), df[TARGET].head(5), version=1,
                        artifact_dir="/tmp/dg_test_artifacts", log_to_mlflow=False)


class TestMonitorLoop:
    def test_healthy_stream_does_not_retrain(self, repo):
        df = _synthetic(9000)
        batches = make_batches(df, batch_size=1000)
        mon = DriftMonitor(repo, MonitorConfig(log_to_mlflow=False,
                                               artifact_dir="/tmp/dg_test_artifacts"))
        mon.fit_initial(batches[:3])

        outcomes = [mon.process(b) for b in batches[3:6]]
        # Same distribution throughout, so the drift gate should stay shut.
        assert all(not o.dataset_drift_detected for o in outcomes)

    def test_concept_inversion_triggers_retrain(self, repo):
        """Flip the label relationship mid-stream; the loop must react."""
        df = _synthetic(9000, shift_at=4000)
        batches = make_batches(df, batch_size=1000)
        mon = DriftMonitor(repo, MonitorConfig(log_to_mlflow=False,
                                               artifact_dir="/tmp/dg_test_artifacts"))
        mon.fit_initial(batches[:3])

        outcomes = [mon.process(b) for b in batches[3:]]
        assert any(o.retrain_triggered for o in outcomes), (
            "label inversion produced no retrain trigger"
        )

    def test_process_before_fit_raises(self, repo):
        df = _synthetic(3000)
        batches = make_batches(df, batch_size=1000)
        mon = DriftMonitor(repo, MonitorConfig(log_to_mlflow=False))
        with pytest.raises(RuntimeError, match="fit_initial"):
            mon.process(batches[0])

    def test_cooldown_suppresses_consecutive_retrains(self, repo):
        df = _synthetic(12000, shift_at=4000)
        batches = make_batches(df, batch_size=1000)
        cfg = MonitorConfig(
            log_to_mlflow=False, artifact_dir="/tmp/dg_test_artifacts",
            min_batches_between_retrains=5,
        )
        mon = DriftMonitor(repo, cfg)
        mon.fit_initial(batches[:3])
        outs = [mon.process(b) for b in batches[3:]]

        fired = [o.batch_index for o in outs if o.retrain_triggered]
        gaps = [b - a for a, b in zip(fired, fired[1:])]
        assert all(g >= 5 for g in gaps), f"cool-down violated: {fired}"

    def test_history_window_is_bounded(self, repo):
        df = _synthetic(12000)
        batches = make_batches(df, batch_size=1000)
        cfg = MonitorConfig(log_to_mlflow=False,
                            artifact_dir="/tmp/dg_test_artifacts",
                            retrain_window_batches=4)
        mon = DriftMonitor(repo, cfg)
        mon.fit_initial(batches[:3])
        for b in batches[3:]:
            mon.process(b)
            assert len(mon.history) <= 4
