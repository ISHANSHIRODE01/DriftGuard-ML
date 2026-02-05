"""
Drift Detection Module
======================
Statistical drift detection for numerical (KS-Test) and categorical (Chi-Square) features.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any, List

class DriftDetector:
    """
    Detects statistical drift using appropriate tests for feature types.
    """

    def __init__(self, p_value_threshold: float = 0.05, mean_diff_threshold: float = 0.10):
        self.p_value_threshold = p_value_threshold
        self.mean_diff_threshold = mean_diff_threshold
        self.drift_results: Dict[str, Any] = {}
        self.summary_stats: Dict[str, Any] = {}

    def detect_drift(self, train_df: pd.DataFrame, new_df: pd.DataFrame) -> None:
        """
        Analyze two DataFrames to detect feature drift.
        """
        self.drift_results = {}
        common_cols = [c for c in train_df.columns if c in new_df.columns]
        
        if not common_cols:
            raise ValueError("No common columns found between train and new dataframes.")

        drifted_count = 0

        for col in common_cols:
            if pd.api.types.is_numeric_dtype(train_df[col]):
                # Heuristic: If numeric but low cardinality (<20 unique), treat as categorical?
                # For safety, we stick to strict type checking unless configured otherwise.
                result = self._check_numerical_drift(train_df[col], new_df[col])
            else:
                result = self._check_categorical_drift(train_df[col], new_df[col])
                
            self.drift_results[col] = result
            if result['drift_detected']:
                drifted_count += 1

        self.summary_stats = {
            "total_features_analyzed": len(self.drift_results),
            "drifted_features_count": drifted_count,
            "drift_detected_overall": bool(drifted_count > 0),
            "timestamp": pd.Timestamp.now().isoformat()
        }

    def get_drift_report(self) -> Dict[str, Any]:
        if not self.drift_results:
            return {"error": "Drift detection not run. Call detect_drift() first."}
        return {"summary": self.summary_stats, "details": self.drift_results}

    def _check_numerical_drift(self, ref: pd.Series, cur: pd.Series) -> Dict[str, Any]:
        """KS-Test and Mean Difference for numerical data."""
        ref = ref.dropna()
        cur = cur.dropna()
        
        # KS Test
        ks_stat, p_value = stats.ks_2samp(ref, cur)
        
        # Mean Check
        mean_ref, mean_cur = ref.mean(), cur.mean()
        mean_diff_pct = abs((mean_cur - mean_ref) / mean_ref) if mean_ref != 0 else 0.0
        
        is_p_drift = p_value < self.p_value_threshold
        is_mean_drift = mean_diff_pct > self.mean_diff_threshold
        
        reasons = []
        if is_p_drift: reasons.append("KS-Test P-value significant")
        if is_mean_drift: reasons.append(f"Mean shift > {self.mean_diff_threshold*100:.0f}%")
        
        return {
            "type": "numerical",
            "drift_detected": bool(is_p_drift or is_mean_drift),
            "statistical_metrics": {"statistic": float(ks_stat), "p_value": float(p_value)},
            "reasons": reasons
        }

    def _check_categorical_drift(self, ref: pd.Series, cur: pd.Series) -> Dict[str, Any]:
        """Chi-Square Test for categorical data."""
        ref = ref.dropna().astype(str)
        cur = cur.dropna().astype(str)
        
        # Align categories
        ref_counts = ref.value_counts(normalize=True).sort_index()
        cur_counts = cur.value_counts(normalize=True).sort_index()
        
        # Combine index to align
        all_cats = sorted(list(set(ref_counts.index) | set(cur_counts.index)))
        
        # Convert to observed frequencies (using current size)
        # We compare distribution shapes explicitly
        # Ideally, we build a contingency table
        
        # Method: 2-sample Chi-Square on counts
        # We need counts, not proportions
        dataset_size = min(len(ref), len(cur)) # Normalize to smaller size to avoid sample size bias
        
        # Re-sample to equal sizes for fair comparison just for the test
        # (This is a simplified approach; rigorously one uses full contingency)
        
        # Contingency Table Construction
        # Rows: [Reference, Current]
        # Cols: [Cat1, Cat2, ...]
        
        contingency_data = []
        for cat in all_cats:
            ref_c = (ref == cat).sum()
            cur_c = (cur == cat).sum()
            contingency_data.append([ref_c, cur_c])
            
        # Transpose to shape (Categories, Groups) -> chi2_contingency needs (Rows=Groups, Cols=Variables)
        # Actually: (Rows=Groups (Ref/Cur), Cols=Categories)
        contingency_table = np.array(contingency_data).T 
        
        # Chi-Square requires counts >= 5. If not, results are invalid.
        # We proceed but handle errors.
        try:
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
            drift_detected = p_value < self.p_value_threshold
        except ValueError:
            # Likely zero counts or invalid table
            p_value = 1.0
            drift_detected = False
            
        reasons = []
        if drift_detected: reasons.append("Chi-Square P-value significant")
        
        return {
            "type": "categorical",
            "drift_detected": drift_detected,
            "statistical_metrics": {"statistic": float(chi2) if 'chi2' in locals() else 0.0, "p_value": float(p_value)},
            "reasons": reasons
        }
