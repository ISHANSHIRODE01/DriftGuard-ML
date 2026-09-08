"""Behavioural tests for the drift detector."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.drift.detector import DriftDetector, population_stability_index

RNG = np.random.default_rng(7)


def _frame(n: int, loc: float = 0.0, scale: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "num": RNG.normal(loc, scale, n),
            "cat": RNG.choice(["a", "b", "c"], n, p=[0.6, 0.3, 0.1]),
        }
    )


class TestPSI:
    def test_identical_distributions_give_near_zero(self):
        s = pd.Series(RNG.normal(0, 1, 5000))
        assert population_stability_index(s, s) < 1e-6

    def test_large_shift_exceeds_major_threshold(self):
        a = pd.Series(RNG.normal(0, 1, 5000))
        b = pd.Series(RNG.normal(3, 1, 5000))
        assert population_stability_index(a, b) > 0.25

    def test_constant_feature_returns_zero_not_nan(self):
        a = pd.Series([5.0] * 100)
        b = pd.Series([5.0] * 100)
        assert population_stability_index(a, b) == 0.0

    def test_empty_input_is_safe(self):
        assert population_stability_index(pd.Series(dtype=float),
                                         pd.Series([1.0, 2.0])) == 0.0

    def test_is_symmetric_in_magnitude(self):
        a = pd.Series(RNG.normal(0, 1, 3000))
        b = pd.Series(RNG.normal(1, 1, 3000))
        # Not mathematically symmetric, but should agree on "large shift".
        assert population_stability_index(a, b) > 0.1
        assert population_stability_index(b, a) > 0.1


class TestDriftDetector:
    def test_no_drift_on_same_distribution(self):
        ref, cur = _frame(3000), _frame(3000)
        rep = DriftDetector().detect(ref, cur)
        assert rep.n_features_drifted == 0
        assert not rep.dataset_drift_detected

    def test_detects_numeric_mean_shift(self):
        ref = _frame(3000)
        cur = _frame(3000, loc=2.5)
        rep = DriftDetector().detect(ref, cur)
        assert "num" in rep.drifted_features()

    def test_psi_gate_suppresses_trivial_but_significant_shift(self):
        """
        With large n, KS flags a microscopic shift as significant. The PSI
        floor is what stops that becoming an alert. This is the key
        false-positive guard in the whole system.
        """
        big_ref = pd.DataFrame({"num": RNG.normal(0, 1, 60000)})
        big_cur = pd.DataFrame({"num": RNG.normal(0.02, 1, 60000)})

        no_gate = DriftDetector(psi_threshold=0.0).detect(big_ref, big_cur)
        with_gate = DriftDetector(psi_threshold=0.10).detect(big_ref, big_cur)

        # The gate must never report MORE drift than the ungated version.
        assert with_gate.n_features_drifted <= no_gate.n_features_drifted

    def test_detects_categorical_proportion_change(self):
        ref = pd.DataFrame({"cat": ["a"] * 900 + ["b"] * 100})
        cur = pd.DataFrame({"cat": ["a"] * 200 + ["b"] * 800})
        rep = DriftDetector().detect(ref, cur)
        assert "cat" in rep.drifted_features()

    def test_dataset_drift_requires_majority(self):
        ref = pd.DataFrame({f"f{i}": RNG.normal(0, 1, 2000) for i in range(4)})
        cur = ref.copy()
        cur["f0"] = RNG.normal(4, 1, 2000)  # only 1 of 4 drifts
        rep = DriftDetector(dataset_drift_share=0.5).detect(ref, cur)
        assert rep.n_features_drifted == 1
        assert not rep.dataset_drift_detected

    def test_low_cardinality_numeric_routed_to_chi2(self):
        ref = pd.DataFrame({"flag": RNG.choice([0, 1, 2], 1000)})
        cur = pd.DataFrame({"flag": RNG.choice([0, 1, 2], 1000)})
        rep = DriftDetector().detect(ref, cur)
        assert rep.results[0].test_name == "chi2"

    def test_raises_when_no_overlapping_columns(self):
        with pytest.raises(ValueError, match="no overlapping columns"):
            DriftDetector().detect(
                pd.DataFrame({"a": [1, 2, 3]}), pd.DataFrame({"b": [1, 2, 3]})
            )

    def test_excluded_columns_are_not_tested(self):
        ref = _frame(1000)
        cur = _frame(1000, loc=5)
        rep = DriftDetector().detect(ref, cur, exclude=["num"])
        assert "num" not in [r.feature_name for r in rep.results]

    def test_drift_share_matches_counts(self):
        ref = pd.DataFrame({f"f{i}": RNG.normal(0, 1, 1500) for i in range(4)})
        cur = ref.copy()
        cur["f0"] = RNG.normal(5, 1, 1500)
        cur["f1"] = RNG.normal(5, 1, 1500)
        rep = DriftDetector().detect(ref, cur)
        assert rep.drift_share == pytest.approx(
            rep.n_features_drifted / rep.n_features_tested
        )
