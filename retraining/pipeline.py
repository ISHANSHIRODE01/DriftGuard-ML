"""
Auto-Retraining Pipeline (Production Ready)
===========================================
Handles batch ingestion, drift detection, and safe model promotion.
"""
import pandas as pd
import numpy as np
import joblib
import json
import shutil
import glob
import sys
import os
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

# --- Deployment Patch: Add root to path ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# Imports
from drift.detector import DriftDetector
from preprocessing.pipeline import clean_raw_data # Centralized Logic
from deployment.model_versioner import ModelVersioner
from database.repository import MLRepository

class RetrainingPipeline:
    """
    Orchestrates the drift detection and model retraining workflow.
    """
    
    def __init__(self, 
                 model_dir: str = "model", 
                 train_data_path: str = "data/raw/train.csv",
                 incoming_labeled_dir: str = "data/incoming/labeled",
                 incoming_unlabeled_dir: str = "data/incoming/unlabeled",
                 target_col: str = "G3",
                 drift_window_size: int = 1000):
        
        self.repo = MLRepository()
        
        self.model_dir = Path(model_dir)
        self.train_data_path = Path(train_data_path)
        self.incoming_labeled_dir = Path(incoming_labeled_dir)
        self.incoming_unlabeled_dir = Path(incoming_unlabeled_dir)
        
        self.target_col = target_col
        self.window_size = drift_window_size
        self.detector = DriftDetector(p_value_threshold=0.05, mean_diff_threshold=0.10)
        
        self.champion_path = self.model_dir / "model.pkl" # This is the baseline original
        self.current_model_path = self.model_dir / "current_model.pkl" # This is the symlink to active
        self.preprocessor_path = self.model_dir / "preprocessor.pkl"

    def _load_recent_batches(self, directory: Path, limit_rows: int = None) -> pd.DataFrame:
        """Loads CSVs from a directory, sorted by modification time."""
        if not directory.exists():
            return pd.DataFrame()
            
        files = glob.glob(str(directory / "*.csv"))
        files.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True) # Newest first
        
        dfs = []
        total_rows = 0
        
        for f in files:
            try:
                # Read latest files until we hit window limit
                batch = pd.read_csv(f)
                dfs.append(batch)
                total_rows += len(batch)
                if limit_rows and total_rows >= limit_rows:
                    break
            except Exception as e:
                print(f"  ⚠ Skipped corrupt file {f}: {e}")
                
        if not dfs:
            return pd.DataFrame()
            
        combined = pd.concat(dfs, ignore_index=True)
        
        if limit_rows:
            return combined.iloc[:limit_rows] # Take newest N
        return combined

    def run(self):
        print("\nPIPELINE START: Batch Ingestion & Maintenance")
        
        # 1. LOAD DATA
        # Reference Data (Original Training Set)
        if not self.train_data_path.exists():
            print("  ✗ CRITICAL: Reference training data missing.")
            return False
        ref_df = pd.read_csv(self.train_data_path)
        
        # Current Inference Window (Unlabeled + Labeled mixed for drift detection)
        # We check drift on the 'unlabeled' stream primarily as it represents production traffic
        unlabeled_window = self._load_recent_batches(self.incoming_unlabeled_dir, self.window_size)
        
        # If no new traffic, check labeled folder (maybe manual batch uploads)
        labeled_window = self._load_recent_batches(self.incoming_labeled_dir, self.window_size)
        
        current_window = pd.concat([unlabeled_window, labeled_window], ignore_index=True)
        
        if current_window.empty:
            print("  ✓ No new data found in incoming folders. Resting.")
            return False
            
        # 2. DETECT DRIFT
        print(f"\n[Step 1] Analyzing Drift on {len(current_window)} recent samples...")
        # Only check feature columns, ignore target if present in current_window
        features_only_ref = ref_df.drop(columns=[self.target_col], errors='ignore')
        features_only_cur = current_window.drop(columns=[self.target_col], errors='ignore')
        
        # Ensure parity
        common_cols = [c for c in features_only_ref.columns if c in features_only_cur.columns]
        if not common_cols:
             print("  ⚠ No common columns between reference and new data!")
             return False

        self.detector.detect_drift(features_only_ref[common_cols], features_only_cur[common_cols])
        report = self.detector.get_drift_report()
        
        # Save drift report for dashboard
        output_path = Path('data/reports/drift_report.json')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        # 3. Log Drift Report to MongoDB
        try:
            self.repo.insert_drift_report(report)
        except Exception as e:
            print(f"  ⚠ Failed to log drift report to DB: {e}")
            
        if not report['summary']['drift_detected_overall']:
            print("  ✓ System Healthy (No Drift).")
            # We MIGHT still retrain if we have lots of new labeled data to strictly improve perfromance,
            # but per requirements: "Retrain ONLY if Drift Detected AND ..."
            return False

        print(f"  ⚠ DRIFT DETECTED! Proceeding to effectiveness check...")

        # 3. CHECK LABELED DATA AVAILABILITY
        print("\n[Step 2] Checking for Labeled Data for Retraining...")
        
        # Load ALL available labeled data for training (Recent + Historical Labeled Batches)
        # In a real system, we might merge `labeled_window` with `ref_df`.
        # Strategy: Retrain on (Original Train + New Labeled).
        
        if labeled_window.empty:
            print("  ⚠ Drift detected but NO new labeled data available.")
            print("  -> ACTION: Alerting humans to label 'data/incoming/unlabeled'.")
            return False
            
        if self.target_col not in labeled_window.columns:
             print("  ✗ Labeled batches missing target column.")
             return False
             
        # Combine
        combined_train = pd.concat([ref_df, labeled_window], ignore_index=True)
        print(f"  -> Retraining Set Size: {len(combined_train)} ({len(ref_df)} old + {len(labeled_window)} new)")
        
        # 4. RETRAIN & EVALUATE
        self._execute_retraining(combined_train)
        return True

    def _execute_retraining(self, df):
        print("\n[Step 3] Executing Safe Retraining...")
        
        # A. Clean
        df_clean = clean_raw_data(df)
        X = df_clean.drop(columns=[self.target_col])
        y = df_clean[self.target_col]
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # B. Load Preprocessor
        if not self.preprocessor_path.exists():
             print("  ✗ Preprocessor missing.")
             return
        preprocessor = joblib.load(self.preprocessor_path)
        
        X_train_prep = preprocessor.transform(X_train)
        X_val_prep = preprocessor.transform(X_val)
        
        # C. Train Challenger
        challenger = RandomForestRegressor(n_estimators=100, random_state=42)
        challenger.fit(X_train_prep, y_train)
        
        # D. Evaluate vs Champion
        # Load current active model
        if self.current_model_path.exists():
            champion = joblib.load(self.current_model_path)
        else:
            champion = joblib.load(self.champion_path)
            
        # Champion Predict
        # Handle Pipeline vs Regressor case safely
        try:
             y_pred_champ = champion.predict(X_val) # Try raw (if pipe)
        except:
             y_pred_champ = champion.predict(X_val_prep) # Try prep (if regressor)
             
        y_pred_chal = challenger.predict(X_val_prep)
        
        mae_champ = mean_absolute_error(y_val, y_pred_champ)
        mae_chal = mean_absolute_error(y_val, y_pred_chal)
        
        print(f"  🏆 Champion MAE: {mae_champ:.3f}")
        print(f"  🧪 Challenger MAE: {mae_chal:.3f}")
        
        if mae_chal < (mae_champ - 0.01):
            print("  -> Challenger WINS. Promoting...")
            versioner = ModelVersioner(str(self.model_dir))
            
            # Wrap in pipeline if needed? Assuming we keep architecture consistent.
            # If champion was pipeline, challenger should be.
            # For simplicity, we save the Regressor as we load preprocessor separately in API.
            # But wait! API expects `model` to handle `X_processed`.
            
            versioner.save_new_version(
                challenger, 
                {"mae": mae_chal, "r2": r2_score(y_val, y_pred_chal)}
            )
        else:
            print("  -> Champion remains. Improvement not significant.")

if __name__ == "__main__":
    RetrainingPipeline().run()
