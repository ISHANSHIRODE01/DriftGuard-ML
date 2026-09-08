"""
Statistical drift detection.

Two questions get asked of every feature, and they are not the same question:

  1. "Did the distribution change?"  -> hypothesis test (KS / chi-square).
     Gives a p-value. Sensitive to sample size: with 50k rows almost anything
     is "significant", which is why (2) exists.

  2. "Did it change enough to matter?" -> Population Stability Index (PSI).
     An effect size, independent of n. Industry convention:
        PSI < 0.10  no meaningful shift
        0.10-0.25   moderate shift, monitor
        PSI > 0.25  major shift, act

A feature is flagged only when the test is significant AND PSI clears the
effect-size floor. That conjunction is what stops the detector from screaming
on every batch once the reference window gets large.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class FeatureResult:
    feature_name: str
    test_name: str
    statistic: Optional[float]
    p_value: Optional[float]
    threshold: float
    drift_detected: bool
    psi: Optional[float] = None

    def as_dict(self) -> dict:
        return {
            "feature_name": self.feature_name,
            "test_name": self.test_name,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "threshold": self.threshold,
            "drift_detected": self.drift_detected,
            "psi": self.psi,
        }


@dataclass
class DriftReport:
    n_features_tested: int
    n_features_drifted: int
    dataset_drift_detected: bool
    drift_share: float
    results: list[FeatureResult] = field(default_factory=list)

    def to_records(self) -> list[dict]:
        return [r.as_dict() for r in self.results]

    def drifted_features(self) -> list[str]:
        return [r.feature_name for r in self.results if r.drift_detected]


def population_stability_index(
    reference: pd.Series, current: pd.Series, n_bins: int = 10
) -> float:
    """
    PSI between two numeric distributions using quantile bins from reference.

    sum over bins of (curr_pct - ref_pct) * ln(curr_pct / ref_pct)

    Bins come from the reference quantiles so the metric answers "how did the
    incoming batch redistribute across the buckets the model was trained on".
    Empty bins are floored to avoid a divide-by-zero blowing the sum to inf.
    """
    ref = pd.to_numeric(reference, errors="coerce").dropna()
    cur = pd.to_numeric(current, errors="coerce").dropna()
    if len(ref) == 0 or len(cur) == 0:
        return 0.0

    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(ref, quantiles))
    if len(edges) < 3:  # near-constant feature; PSI is not meaningful
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)

    eps = 1e-6
    ref_pct = np.clip(ref_counts / max(ref_counts.sum(), 1), eps, None)
    cur_pct = np.clip(cur_counts / max(cur_counts.sum(), 1), eps, None)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


class DriftDetector:
    """
    Feature-level drift detection with a dataset-level roll-up.

    Args:
        p_value_threshold: significance level for the hypothesis tests.
        psi_threshold: minimum effect size before a feature counts as drifted.
        dataset_drift_share: fraction of features that must drift before the
            dataset as a whole is declared drifted. 0.5 = majority vote.
        categorical_max_cardinality: numeric columns with fewer unique values
            than this are treated as categorical and routed to chi-square.
    """

    def __init__(
        self,
        p_value_threshold: float = 0.05,
        psi_threshold: float = 0.10,
        dataset_drift_share: float = 0.5,
        categorical_max_cardinality: int = 12,
    ):
        self.p_value_threshold = p_value_threshold
        self.psi_threshold = psi_threshold
        self.dataset_drift_share = dataset_drift_share
        self.categorical_max_cardinality = categorical_max_cardinality

    # ------------------------------------------------------------------ tests

    def _numeric(self, ref: pd.Series, cur: pd.Series, name: str) -> FeatureResult:
        """Two-sample Kolmogorov-Smirnov, gated by PSI effect size."""
        r = pd.to_numeric(ref, errors="coerce").dropna()
        c = pd.to_numeric(cur, errors="coerce").dropna()
        if len(r) < 2 or len(c) < 2:
            return FeatureResult(name, "ks_2samp", None, None,
                                 self.p_value_threshold, False, None)

        ks = stats.ks_2samp(r, c)
        psi = population_stability_index(r, c)
        significant = bool(ks.pvalue < self.p_value_threshold)
        material = bool(psi >= self.psi_threshold)

        return FeatureResult(
            feature_name=name,
            test_name="ks_2samp",
            statistic=float(ks.statistic),
            p_value=float(ks.pvalue),
            threshold=self.p_value_threshold,
            drift_detected=significant and material,
            psi=round(psi, 5),
        )

    def _categorical(self, ref: pd.Series, cur: pd.Series, name: str) -> FeatureResult:
        """Chi-square test of independence over aligned category counts."""
        ref_counts = ref.astype(str).value_counts()
        cur_counts = cur.astype(str).value_counts()
        categories = sorted(set(ref_counts.index) | set(cur_counts.index))

        table = np.array(
            [
                [ref_counts.get(k, 0) for k in categories],
                [cur_counts.get(k, 0) for k in categories],
            ],
            dtype=float,
        )
        # Drop all-zero columns; chi2 is undefined for them.
        table = table[:, table.sum(axis=0) > 0]
        if table.shape[1] < 2:
            return FeatureResult(name, "chi2", None, None,
                                 self.p_value_threshold, False, None)

        chi2, p, _, _ = stats.chi2_contingency(table)

        # Categorical analogue of PSI over the category proportions.
        eps = 1e-6
        ref_pct = np.clip(table[0] / max(table[0].sum(), 1), eps, None)
        cur_pct = np.clip(table[1] / max(table[1].sum(), 1), eps, None)
        psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

        return FeatureResult(
            feature_name=name,
            test_name="chi2",
            statistic=float(chi2),
            p_value=float(p),
            threshold=self.p_value_threshold,
            drift_detected=bool(p < self.p_value_threshold and psi >= self.psi_threshold),
            psi=round(psi, 5),
        )

    # ------------------------------------------------------------------ entry

    def detect(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        exclude: Optional[list[str]] = None,
    ) -> DriftReport:
        """Compare `current` against `reference` column by column."""
        exclude = set(exclude or [])
        columns = [
            c for c in reference.columns if c in current.columns and c not in exclude
        ]
        if not columns:
            raise ValueError("no overlapping columns between reference and current")

        results: list[FeatureResult] = []
        for col in columns:
            ref_col, cur_col = reference[col], current[col]
            is_num = pd.api.types.is_numeric_dtype(ref_col)
            low_card = ref_col.nunique(dropna=True) <= self.categorical_max_cardinality

            if is_num and not low_card:
                results.append(self._numeric(ref_col, cur_col, col))
            else:
                results.append(self._categorical(ref_col, cur_col, col))

        n_drift = sum(r.drift_detected for r in results)
        share = n_drift / len(results) if results else 0.0

        return DriftReport(
            n_features_tested=len(results),
            n_features_drifted=n_drift,
            dataset_drift_detected=share >= self.dataset_drift_share,
            drift_share=round(share, 4),
            results=results,
        )
