"""
ML Monitoring Dashboard
======================
Streamlit dashboard for monitoring drift, performance, and versioning.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import joblib
import time
import sys
import os
from pathlib import Path
from datetime import datetime

# --- Deployment Patch: Add root to path ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from preprocessing.pipeline import clean_raw_data # Centralized Logic
from database.repository import MLRepository

# Initialize Repository
# We wrap this in a check for local vs cloud if needed, but for now, simple init.
db_repo = MLRepository()

# --- Configuration ---
st.set_page_config(
    page_title="Student Performance Monitor",
    page_icon="🤖",
    layout="wide"
)

# --- Path definitions ---
DRIFT_REPORT_PATH = Path("data/reports/drift_report.json")
METADATA_PATH = Path("model/metadata.json")
CURRENT_MODEL_PATH = Path("model/current_model.pkl")
PREPROCESSOR_PATH = Path("model/preprocessor.pkl")
INCOMING_UNLABELED = Path("data/incoming/unlabeled")

# --- Helper Functions ---
def auto_bootstrap():
    """Automatically generates baseline model if missing."""
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error, r2_score
        from preprocessing.pipeline import StudentPerformancePreprocessor

        st.info("📦 First-time setup: Training baseline model...")
        df = pd.read_csv("data/raw/train.csv")
        df_clean = clean_raw_data(df)
        
        preprocessor_obj = StudentPerformancePreprocessor()
        X, y = preprocessor_obj.prepare_data(df_clean)
        X_train, X_val, _, y_train, y_val, _ = preprocessor_obj.split_data(X, y)
        
        preprocessor_obj.create_preprocessing_pipeline()
        X_train_prep, X_val_prep, _ = preprocessor_obj.fit_transform_pipeline(X_train, X_val, X_val)
        
        joblib.dump(preprocessor_obj.preprocessor, PREPROCESSOR_PATH)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_prep, y_train)
        
        joblib.dump(model, CURRENT_MODEL_PATH)
        joblib.dump(model, Path("model/model.pkl"))
        
        # Metadata
        y_pred = model.predict(X_val_prep)
        meta = {
            "latest_version": 1, "current_production_version": 1,
            "last_updated": datetime.now().isoformat(),
            "history": [{"version": 1, "timestamp": datetime.now().isoformat(), 
                         "metrics": {"mae": mean_absolute_error(y_val, y_pred), "r2": r2_score(y_val, y_pred)},
                         "file": "model_v1.pkl"}]
        }
        with open(METADATA_PATH, 'w') as f:
            json.dump(meta, f, indent=4)
        
        st.success("✅ Baseline model initialized!")
        return True
    except Exception as e:
        st.error(f"Failed to auto-bootstrap: {e}")
        return False

def load_json(path):
    if not path.exists():
        return None
    with open(path, 'r') as f:
        return json.load(f)

def load_prediction_artifacts():
    # If files don't exist, try auto-bootstrapping
    if not CURRENT_MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        if not auto_bootstrap():
            return None, None
            
    try:
        model = joblib.load(CURRENT_MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        return model, preprocessor
    except Exception as e:
        return None, None

# --- Data Loading ---
drift_data = load_json(DRIFT_REPORT_PATH)
meta_data = load_json(METADATA_PATH)

# --- Sidebar ---
st.sidebar.title("ML System Status")
if meta_data:
    st.sidebar.metric("Current Version", f"v{meta_data.get('current_production_version', 'N/A')}")
    st.sidebar.metric("Total Retrainings", len(meta_data.get('history', [])))
    st.sidebar.write(f"**Last Updated:**  \n{meta_data.get('last_updated', 'N/A')[:19]}")
else:
    st.sidebar.warning("Model metadata not found.")

if drift_data:
    overall_drift = drift_data.get('summary', {}).get('drift_detected_overall', False)
    status_color = "🔴" if overall_drift else "🟢"
    st.sidebar.markdown(f"**Drift Status:** {status_color} {'DRIFTING' if overall_drift else 'STABLE'}")

# --- Main Dashboard ---
st.title("🛡️ Student Performance Monitoring Dashboard")
st.markdown("Monitor model health, track data drift, and audit retraining history.")

# --- Row 1: Key Metrics ---
col1, col2, col3, col4 = st.columns(4)

if meta_data and meta_data.get('history'):
    latest = meta_data['history'][-1]['metrics']
    col1.metric("Latest MAE", f"{latest.get('mae', 0):.3f}", delta_color="inverse")
    col2.metric("Latest R2 Score", f"{latest.get('r2', 0):.3f}")

if drift_data:
    sum_data = drift_data.get('summary', {})
    col3.metric("Features Analyzed", sum_data.get('total_features_analyzed', 0))
    col4.metric("Drifted Features", sum_data.get('drifted_features_count', 0), delta_color="inverse")

st.divider()

# --- Row 2: Charts ---
tab1, tab2, tab3 = st.tabs(["🚀 Model Performance History", "📡 Data Drift Analysis", "🔮 Live Prediction & Ingestion"])

with tab1:
    st.subheader("Model Accuracy and Performance Over Time")
    if meta_data and meta_data.get('history'):
        history = meta_data['history']
        df_hist = pd.DataFrame([
            {
                "version": f"v{h['version']}",
                "timestamp": h['timestamp'],
                "mae": h['metrics'].get('mae', 0),
                "r2": h['metrics'].get('r2', 0)
            } for h in history
        ])
        
        # Dual axis chart or two subplots
        fig_perf = go.Figure()
        fig_perf.add_trace(go.Scatter(x=df_hist['version'], y=df_hist['mae'], name="MAE (Lower is Better)", line=dict(color='firebrick', width=4)))
        fig_perf.add_trace(go.Scatter(x=df_hist['version'], y=df_hist['r2'], name="R2 Score (Higher is Better)", line=dict(color='royalblue', width=4)))
        
        fig_perf.update_layout(title="Metric Evolution across Versions", xaxis_title="Model Version", yaxis_title="Score", height=400)
        st.plotly_chart(fig_perf, use_container_width=True)
        
        st.write("**Full History Log:**")
        st.dataframe(df_hist, use_container_width=True)
    else:
        st.info("No retraining history available yet.")

with tab2:
    st.subheader("Statistical Drift Analysis per Feature")
    if drift_data and drift_data.get('details'):
        details = drift_data['details']
        
        # Prepare data for plotting
        drift_list = []
        for feat, res in details.items():
            drift_list.append({
                "feature": feat,
                "drift_score": res.get("statistical_metrics", {}).get("statistic", 0),
                "p_value": res.get("statistical_metrics", {}).get("p_value", 1.0),
                "is_drifted": res.get("drift_detected", False)
            })
        
        df_drift = pd.DataFrame(drift_list).sort_values("drift_score", ascending=False)
        
        # Plotly Bar Chart
        fig_drift = px.bar(
            df_drift, 
            x="feature", 
            y="drift_score", 
            color="is_drifted",
            color_discrete_map={True: "red", False: "green"},
            title="Drift Statistic (KS / Chi2)",
            labels={"is_drifted": "Drift Detected", "drift_score": "Statistic Value"}
        )
        st.plotly_chart(fig_drift, use_container_width=True)
        
        # P-Value scatter
        fig_p = px.scatter(
            df_drift,
            x="feature",
            y="p_value",
            color="is_drifted",
            title="Distribution P-Values (Significance < 0.05)",
            labels={"p_value": "P-Value"}
        )
        fig_p.add_hline(y=0.05, line_dash="dash", line_color="red", annotation_text="Significance Threshold")
        st.plotly_chart(fig_p, use_container_width=True)
        
    else:
        st.warning("No drift report found. Please run the drift detection module.")

with tab3:
    st.subheader("Interactive Prediction Simulator")
    st.markdown("Use this form to test the model and simulate incoming traffic.")
    
    with st.form("prediction_form"):
        col_A, col_B, col_C = st.columns(3)
        
        with col_A:
            age = st.number_input("Age", 15, 22, 18)
            Medu = st.selectbox("Mother's Education (0-4)", [0, 1, 2, 3, 4], index=3)
            Fedu = st.selectbox("Father's Education (0-4)", [0, 1, 2, 3, 4], index=3)
            sex = st.selectbox("Sex", ["F", "M"])
            school = st.selectbox("School", ["GP", "MS"])
            address = st.selectbox("Address", ["U", "R"])
            
        with col_B:
            studytime = st.selectbox("Study Time (1-4)", [1, 2, 3, 4], index=1)
            failures = st.number_input("Past Failures", 0, 4, 0)
            schoolsup = st.selectbox("School Support", ["yes", "no"])
            famsup = st.selectbox("Family Support", ["yes", "no"])
            paid = st.selectbox("Extra Classes", ["yes", "no"])
            G1 = st.number_input("G1 Grade", 0, 20, 10)
            
        with col_C:
            absences = st.number_input("Absences", 0, 93, 2)
            G2 = st.number_input("G2 Grade", 0, 20, 11)
            freetime = st.slider("Free Time", 1, 5, 3)
            goout = st.slider("Going Out", 1, 5, 3)
            health = st.slider("Health", 1, 5, 5)
            
        # Add Defaults for remaining fields to keep form clean
        # In a real app, strict form matches 30 columns. Here we fill dummies for demo.
        
        submitted = st.form_submit_button("Predict & Save")
        
        if submitted:
            # Construct Input Dict
            raw_input = {
                "school": school, "sex": sex, "age": age, "address": address, "famsize": "GT3", 
                "Pstatus": "T", "Medu": Medu, "Fedu": Fedu, "Mjob": "other", "Fjob": "other",
                "reason": "course", "guardian": "mother", "traveltime": 1, "studytime": studytime,
                "failures": failures, "schoolsup": schoolsup, "famsup": famsup, "paid": paid,
                "activities": "no", "nursery": "yes", "higher": "yes", "internet": "yes",
                "romantic": "no", "famrel": 4, "freetime": freetime, "goout": goout,
                "Dalc": 1, "Walc": 1, "health": health, "absences": absences,
                "G1": G1, "G2": G2
            }
            
            # Create DataFrame
            df_input = pd.DataFrame([raw_input])
            
            # Load Artificats
            model, preprocessor = load_prediction_artifacts()
            
            if model is not None and preprocessor is not None:
                try:
                    # 1. Cleaning Phase
                    df_clean = clean_raw_data(df_input)
                    
                    # 2. Transform Phase
                    X_processed = preprocessor.transform(df_clean)
                    
                    # 3. Prediction Phase
                    # Handle Pipeline vs Regressor
                    if hasattr(model, 'predict'):
                        try:
                           pred = model.predict(X_processed)[0]
                        except:
                           pred = model.predict(df_clean)[0] # Fallback
                           
                    st.success(f"Predicted Final Grade (G3): **{pred:.2f}**")
                    
                    # 4. Save to "Incoming Unlabeled"
                    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                    save_path = INCOMING_UNLABELED / f"batch_{timestamp}.csv"
                    df_input.to_csv(save_path, index=False)
                    
                    # 5. Log to MongoDB
                    try:
                        version_str = f"v{meta_data.get('current_production_version', '1')}" if meta_data else "v1"
                        db_repo.insert_prediction(
                            features=raw_input,
                            prediction=float(pred),
                            model_version=version_str
                        )
                    except Exception as db_err:
                        st.warning(f"Logged locally, but DB failed: {db_err}")

                    st.toast(f"Data saved to {save_path.name}")
                    
                except Exception as e:
                    st.error(f"Prediction Error: {str(e)}")
            else:
                st.error("Model artifacts not found!")

# --- Footer ---
st.divider()
st.caption("Dashboard generated by ML Engineering Monitoring System. Version 1.1")
