"""
Bootstrap Script
================
Forces a clean training of the initial model to ensure all artifacts 
are consistent and fitted.
"""

import pandas as pd
import joblib
import json
import shutil
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from preprocessing.pipeline import StudentPerformancePreprocessor, clean_raw_data

def bootstrap():
    print("BOOTSTRAPPING MODEL ARTIFACTS...")
    Path("model").mkdir(exist_ok=True)
    
    # 1. Load Data
    df = pd.read_csv("data/raw/train.csv")
    df_clean = clean_raw_data(df)
    
    # 2. Preprocess (Fit)
    preprocessor_obj = StudentPerformancePreprocessor()
    X, y = preprocessor_obj.prepare_data(df_clean)
    
    # Split
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor_obj.split_data(X, y)
    
    # Fit Pipeline
    preprocessor_obj.create_preprocessing_pipeline()
    X_train_prep, X_val_prep, X_test_prep = preprocessor_obj.fit_transform_pipeline(X_train, X_val, X_test)
    
    # Save Preprocessor (The scikit-learn ColumnTransformer)
    joblib.dump(preprocessor_obj.preprocessor, "model/preprocessor.pkl")
    print("  [OK] Preprocessor saved.")
    
    # 3. Train Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_prep, y_train)
    print("  [OK] Model trained.")
    
    # 4. Save Model (Baseline)
    joblib.dump(model, "model/model.pkl")
    joblib.dump(model, "model/model_v1.pkl")
    joblib.dump(model, "model/current_model.pkl")
    print("  [OK] Model artifacts saved (v1 + current).")
    
    # 5. Save Metadata
    # Evaluate
    y_pred = model.predict(X_val_prep)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    metadata = {
        "latest_version": 1,
        "current_production_version": 1,
        "last_updated": datetime.now().isoformat(),
        "history": [{
            "version": 1,
            "timestamp": datetime.now().isoformat(),
            "metrics": {"mae": mae, "r2": r2},
            "file": "model_v1.pkl"
        }]
    }
    
    with open("model/metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
    print("  [OK] Metadata initialized.")
    
    print("\nSYSTEM RESET COMPLETE. Dashboard should now work.")

if __name__ == "__main__":
    bootstrap()
