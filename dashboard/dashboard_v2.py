"""
Production-Grade ML Monitoring Dashboard
=========================================
Advanced monitoring with drift detection, alerts, and comprehensive logging.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import joblib
import sys
import os
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any

# --- Deployment Patch: Add root to path ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from preprocessing.pipeline import clean_raw_data
from database.repository import MLRepository

# --- Configuration ---
st.set_page_config(
    page_title="DriftGuard-ML Monitor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Path Definitions ---
DRIFT_REPORT_PATH = Path("Data/reports/drift_report.json")
METADATA_PATH = Path("model/metadata.json")
CURRENT_MODEL_PATH = Path("model/current_model.pkl")
PREPROCESSOR_PATH = Path("model/preprocessor.pkl")
INCOMING_UNLABELED = Path("Data/incoming/unlabeled")
SYSTEM_LOGS_PATH = Path("Data/reports/system_logs.json")
DRIFT_HISTORY_PATH = Path("Data/reports/drift_history.json")

# Initialize Repository
db_repo = MLRepository()

# --- Custom CSS for Production Look ---
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .alert-danger {
        background-color: #ff4b4b;
        padding: 15px;
        border-radius: 8px;
        color: white;
        font-weight: bold;
    }
    .alert-success {
        background-color: #00cc88;
        padding: 15px;
        border-radius: 8px;
        color: white;
        font-weight: bold;
    }
    .alert-warning {
        background-color: #ffa500;
        padding: 15px;
        border-radius: 8px;
        color: white;
        font-weight: bold;
    }
    .log-entry {
        padding: 10px;
        margin: 5px 0;
        border-left: 4px solid #667eea;
        background-color: #f0f2f6;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# PART 4: BACKEND LOGIC - Data Persistence & Management
# ============================================================================

class SystemLogger:
    """Manages system logs with persistence"""
    
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
    def load_logs(self) -> List[Dict]:
        """Load existing logs"""
        if not self.log_path.exists():
            return []
        try:
            with open(self.log_path, 'r') as f:
                return json.load(f)
        except:
            return []
    
    def add_log(self, event_type: str, message: str, severity: str = "INFO", metadata: Dict = None):
        """Add a new log entry"""
        logs = self.load_logs()
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "message": message,
            "severity": severity,
            "metadata": metadata or {}
        }
        
        logs.append(log_entry)
        
        # Keep only last 100 logs
        logs = logs[-100:]
        
        with open(self.log_path, 'w') as f:
            json.dump(logs, f, indent=2)
    
    def get_recent_logs(self, n: int = 10) -> List[Dict]:
        """Get n most recent logs"""
        logs = self.load_logs()
        return logs[-n:][::-1]  # Reverse to show newest first


class DriftHistoryManager:
    """Manages drift score history over time"""
    
    def __init__(self, history_path: Path):
        self.history_path = history_path
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
    
    def load_history(self) -> List[Dict]:
        """Load drift history"""
        if not self.history_path.exists():
            return []
        try:
            with open(self.history_path, 'r') as f:
                return json.load(f)
        except:
            return []
    
    def add_drift_record(self, drift_score: float, drifted_features: int, total_features: int):
        """Add a drift detection record"""
        history = self.load_history()
        
        record = {
            "timestamp": datetime.now().isoformat(),
            "drift_score": drift_score,
            "drifted_features": drifted_features,
            "total_features": total_features,
            "drift_percentage": (drifted_features / total_features * 100) if total_features > 0 else 0
        }
        
        history.append(record)
        
        # Keep last 50 records
        history = history[-50:]
        
        with open(self.history_path, 'w') as f:
            json.dump(history, f, indent=2)
    
    def get_drift_trend(self) -> pd.DataFrame:
        """Get drift trend as DataFrame"""
        history = self.load_history()
        if not history:
            return pd.DataFrame()
        
        df = pd.DataFrame(history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df


# ============================================================================
# DRIFT CALCULATION FUNCTIONS
# ============================================================================

def calculate_overall_drift_score(drift_report: Dict) -> float:
    """
    Calculate a single drift score (0-100) from drift report
    Higher score = more drift
    """
    if not drift_report or 'details' not in drift_report:
        return 0.0
    
    details = drift_report['details']
    scores = []
    
    for feature, result in details.items():
        if result.get('drift_detected', False):
            # Use p-value or statistic to compute score
            p_value = result.get('statistical_metrics', {}).get('p_value', 1.0)
            # Convert p-value to score (lower p-value = higher drift)
            score = (1 - p_value) * 100
            scores.append(score)
    
    if not scores:
        return 0.0
    
    # Average drift score across all drifted features
    return np.mean(scores)


def calculate_drift_severity(drift_score: float) -> tuple:
    """
    Classify drift severity
    Returns: (severity_level, color, icon)
    """
    if drift_score < 20:
        return ("LOW", "green", "✅")
    elif drift_score < 50:
        return ("MODERATE", "orange", "⚠️")
    elif drift_score < 75:
        return ("HIGH", "red", "🔴")
    else:
        return ("CRITICAL", "darkred", "🚨")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_json(path: Path) -> Dict:
    """Load JSON file safely"""
    if not path.exists():
        return None
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except:
        return None


def auto_bootstrap():
    """Automatically generates baseline model if missing"""
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error, r2_score
        from preprocessing.pipeline import StudentPerformancePreprocessor

        st.info("📦 First-time setup: Training baseline model...")
        Path("model").mkdir(parents=True, exist_ok=True)
        Path("Data/reports").mkdir(parents=True, exist_ok=True)

        df = pd.read_csv("Data/raw/train.csv")
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
        
        y_pred = model.predict(X_val_prep)
        meta = {
            "latest_version": 1,
            "current_production_version": 1,
            "last_updated": datetime.now().isoformat(),
            "history": [{
                "version": 1,
                "timestamp": datetime.now().isoformat(),
                "metrics": {"mae": mean_absolute_error(y_val, y_pred), "r2": r2_score(y_val, y_pred)},
                "file": "model_v1.pkl"
            }]
        }
        with open(METADATA_PATH, 'w') as f:
            json.dump(meta, f, indent=4)
        
        st.success("✅ Baseline model initialized!")
        return True
    except Exception as e:
        st.error(f"Failed to auto-bootstrap: {e}")
        return False


def load_prediction_artifacts():
    """Load model and preprocessor"""
    if not CURRENT_MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        if not auto_bootstrap():
            return None, None
            
    try:
        model = joblib.load(CURRENT_MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        return model, preprocessor
    except Exception as e:
        return None, None


# ============================================================================
# INITIALIZE MANAGERS
# ============================================================================

logger = SystemLogger(SYSTEM_LOGS_PATH)
drift_history_mgr = DriftHistoryManager(DRIFT_HISTORY_PATH)

# ============================================================================
# LOAD DATA
# ============================================================================

drift_data = load_json(DRIFT_REPORT_PATH)
meta_data = load_json(METADATA_PATH)

# Update drift history if new drift data exists
if drift_data and drift_data.get('summary'):
    summary = drift_data['summary']
    drift_score = calculate_overall_drift_score(drift_data)
    
    # Check if this is a new drift report (not already logged)
    recent_history = drift_history_mgr.load_history()
    if not recent_history or recent_history[-1].get('drift_score') != drift_score:
        drift_history_mgr.add_drift_record(
            drift_score=drift_score,
            drifted_features=summary.get('drifted_features_count', 0),
            total_features=summary.get('total_features_analyzed', 0)
        )
        
        # Log the event
        if summary.get('drift_detected_overall'):
            logger.add_log(
                event_type="DRIFT_DETECTED",
                message=f"Drift detected in {summary.get('drifted_features_count', 0)} features",
                severity="WARNING",
                metadata={"drift_score": drift_score}
            )


# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.title("🛡️ System Status")

# Model Info
if meta_data:
    st.sidebar.metric("Current Version", f"v{meta_data.get('current_production_version', 'N/A')}")
    st.sidebar.metric("Total Retrainings", len(meta_data.get('history', [])))
    
    last_updated = meta_data.get('last_updated', 'N/A')
    if last_updated != 'N/A':
        try:
            dt = datetime.fromisoformat(last_updated)
            st.sidebar.write(f"**Last Updated:**  \n{dt.strftime('%Y-%m-%d %H:%M')}")
        except:
            st.sidebar.write(f"**Last Updated:**  \n{last_updated[:19]}")
else:
    st.sidebar.warning("⚠️ Model metadata not found")

st.sidebar.divider()

# Drift Status with Alert
if drift_data:
    overall_drift = drift_data.get('summary', {}).get('drift_detected_overall', False)
    drift_score = calculate_overall_drift_score(drift_data)
    severity, color, icon = calculate_drift_severity(drift_score)
    
    if overall_drift:
        st.sidebar.markdown(f"### {icon} Drift Status")
        st.sidebar.error(f"**DRIFTING** - {severity}")
        st.sidebar.metric("Drift Score", f"{drift_score:.1f}/100")
    else:
        st.sidebar.markdown(f"### {icon} Drift Status")
        st.sidebar.success("**STABLE**")
        st.sidebar.metric("Drift Score", f"{drift_score:.1f}/100")
else:
    st.sidebar.info("No drift data available")

st.sidebar.divider()

# Quick Actions
st.sidebar.markdown("### ⚡ Quick Actions")
if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
    st.rerun()

if st.sidebar.button("📥 Download Logs", use_container_width=True):
    logs = logger.load_logs()
    st.sidebar.download_button(
        label="💾 Save Logs JSON",
        data=json.dumps(logs, indent=2),
        file_name=f"system_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )


# ============================================================================
# MAIN DASHBOARD
# ============================================================================

st.title("🛡️ DriftGuard-ML Production Monitor")
st.markdown("**Real-time ML Model Monitoring with Advanced Drift Detection**")

# ============================================================================
# PART 2: ALERT SYSTEM
# ============================================================================

st.markdown("---")
st.subheader("🚨 System Alerts")

alert_col1, alert_col2 = st.columns(2)

with alert_col1:
    if drift_data:
        overall_drift = drift_data.get('summary', {}).get('drift_detected_overall', False)
        drift_score = calculate_overall_drift_score(drift_data)
        severity, color, icon = calculate_drift_severity(drift_score)
        
        if overall_drift:
            st.markdown(f"""
            <div class="alert-danger">
                {icon} <strong>DRIFT ALERT</strong><br>
                Severity: {severity} | Score: {drift_score:.1f}/100<br>
                {drift_data.get('summary', {}).get('drifted_features_count', 0)} features drifting
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="alert-success">
                ✅ <strong>SYSTEM STABLE</strong><br>
                All features within normal range<br>
                Drift Score: {drift_score:.1f}/100
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("⏳ Awaiting drift detection run...")

with alert_col2:
    if meta_data and meta_data.get('history'):
        latest_metrics = meta_data['history'][-1]['metrics']
        mae = latest_metrics.get('mae', 0)
        
        if mae < 2.0:
            st.markdown("""
            <div class="alert-success">
                ✅ <strong>MODEL PERFORMANCE EXCELLENT</strong><br>
                MAE below threshold (< 2.0)<br>
                No retraining needed
            </div>
            """, unsafe_allow_html=True)
        elif mae < 3.0:
            st.markdown("""
            <div class="alert-warning">
                ⚠️ <strong>MODEL PERFORMANCE ACCEPTABLE</strong><br>
                MAE within acceptable range<br>
                Monitor closely
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="alert-danger">
                🔴 <strong>MODEL PERFORMANCE DEGRADED</strong><br>
                MAE above threshold (> 3.0)<br>
                Retraining recommended
            </div>
            """, unsafe_allow_html=True)


# ============================================================================
# PART 1: METRICS SECTION
# ============================================================================

st.markdown("---")
st.subheader("📊 Performance Metrics")

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

if meta_data and meta_data.get('history'):
    history = meta_data['history']
    latest = history[-1]['metrics']
    
    # Calculate improvement if multiple versions exist
    if len(history) > 1:
        previous = history[-2]['metrics']
        mae_improvement = previous.get('mae', 0) - latest.get('mae', 0)
        r2_improvement = latest.get('r2', 0) - previous.get('r2', 0)
    else:
        mae_improvement = 0
        r2_improvement = 0
    
    metric_col1.metric(
        "Current MAE",
        f"{latest.get('mae', 0):.3f}",
        delta=f"{-mae_improvement:.3f}" if mae_improvement != 0 else None,
        delta_color="inverse"
    )
    
    metric_col2.metric(
        "Current R² Score",
        f"{latest.get('r2', 0):.3f}",
        delta=f"{r2_improvement:.3f}" if r2_improvement != 0 else None
    )
    
    # Show before/after if retraining happened
    if len(history) > 1:
        metric_col3.metric(
            "Previous MAE",
            f"{previous.get('mae', 0):.3f}",
            help="Performance before last retraining"
        )
        
        improvement_pct = (mae_improvement / previous.get('mae', 1)) * 100
        metric_col4.metric(
            "Improvement",
            f"{improvement_pct:.1f}%",
            help="MAE improvement from retraining"
        )

if drift_data:
    summary = drift_data.get('summary', {})
    drift_score = calculate_overall_drift_score(drift_data)
    
    drift_col1, drift_col2, drift_col3, drift_col4 = st.columns(4)
    
    drift_col1.metric(
        "Drift Score",
        f"{drift_score:.1f}/100",
        help="Overall drift severity (0=stable, 100=critical)"
    )
    
    drift_col2.metric(
        "Features Analyzed",
        summary.get('total_features_analyzed', 0)
    )
    
    drift_col3.metric(
        "Drifted Features",
        summary.get('drifted_features_count', 0),
        delta_color="inverse"
    )
    
    drift_pct = (summary.get('drifted_features_count', 0) / summary.get('total_features_analyzed', 1)) * 100
    drift_col4.metric(
        "Drift Percentage",
        f"{drift_pct:.1f}%"
    )

# Visualizations
st.markdown("### 📈 Performance Over Time")

viz_col1, viz_col2 = st.columns(2)

with viz_col1:
    if meta_data and meta_data.get('history'):
        history = meta_data['history']
        df_perf = pd.DataFrame([
            {
                "version": f"v{h['version']}",
                "timestamp": h['timestamp'],
                "MAE": h['metrics'].get('mae', 0),
                "R²": h['metrics'].get('r2', 0)
            } for h in history
        ])
        
        fig_mae = go.Figure()
        fig_mae.add_trace(go.Scatter(
            x=df_perf['version'],
            y=df_perf['MAE'],
            mode='lines+markers',
            name='MAE',
            line=dict(color='#ff4b4b', width=3),
            marker=dict(size=10)
        ))
        fig_mae.update_layout(
            title="Model MAE Trend (Lower is Better)",
            xaxis_title="Version",
            yaxis_title="Mean Absolute Error",
            height=300
        )
        st.plotly_chart(fig_mae, use_container_width=True)
    else:
        st.info("No performance history available")

with viz_col2:
    # Drift Score Over Time
    drift_trend = drift_history_mgr.get_drift_trend()
    
    if not drift_trend.empty:
        fig_drift = go.Figure()
        fig_drift.add_trace(go.Scatter(
            x=drift_trend['timestamp'],
            y=drift_trend['drift_score'],
            mode='lines+markers',
            name='Drift Score',
            line=dict(color='#ffa500', width=3),
            marker=dict(size=10),
            fill='tozeroy',
            fillcolor='rgba(255, 165, 0, 0.2)'
        ))
        
        # Add threshold lines
        fig_drift.add_hline(y=20, line_dash="dash", line_color="green", annotation_text="Low Threshold")
        fig_drift.add_hline(y=50, line_dash="dash", line_color="orange", annotation_text="Moderate Threshold")
        fig_drift.add_hline(y=75, line_dash="dash", line_color="red", annotation_text="High Threshold")
        
        fig_drift.update_layout(
            title="Drift Score Trend",
            xaxis_title="Time",
            yaxis_title="Drift Score (0-100)",
            height=300
        )
        st.plotly_chart(fig_drift, use_container_width=True)
    else:
        st.info("No drift history available yet")


# ============================================================================
# PART 3: LOGS PANEL
# ============================================================================

st.markdown("---")
st.subheader("📁 System Logs & History")

log_tab1, log_tab2, log_tab3 = st.tabs(["📋 Recent Events", "📊 Retraining History", "🔍 Drift Details"])

with log_tab1:
    st.markdown("### Recent System Events")
    
    recent_logs = logger.get_recent_logs(n=15)
    
    if recent_logs:
        for log in recent_logs:
            severity_color = {
                "INFO": "🔵",
                "WARNING": "🟠",
                "ERROR": "🔴",
                "SUCCESS": "🟢"
            }.get(log['severity'], "⚪")
            
            timestamp = datetime.fromisoformat(log['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
            
            st.markdown(f"""
            <div class="log-entry">
                {severity_color} <strong>{log['event_type']}</strong> | {timestamp}<br>
                {log['message']}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No system logs available yet")
    
    # Add manual log entry (for testing)
    with st.expander("➕ Add Manual Log Entry"):
        log_type = st.selectbox("Event Type", ["SYSTEM_START", "DRIFT_DETECTED", "MODEL_RETRAINED", "CUSTOM"])
        log_msg = st.text_input("Message")
        log_severity = st.selectbox("Severity", ["INFO", "WARNING", "ERROR", "SUCCESS"])
        
        if st.button("Add Log"):
            logger.add_log(log_type, log_msg, log_severity)
            st.success("Log added!")
            st.rerun()

with log_tab2:
    st.markdown("### Model Retraining History")
    
    if meta_data and meta_data.get('history'):
        history = meta_data['history']
        
        # Summary stats
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Retrainings", len(history))
        
        if len(history) > 1:
            first_mae = history[0]['metrics'].get('mae', 0)
            latest_mae = history[-1]['metrics'].get('mae', 0)
            total_improvement = ((first_mae - latest_mae) / first_mae) * 100
            col2.metric("Total MAE Improvement", f"{total_improvement:.1f}%")
        
        last_retrain = datetime.fromisoformat(history[-1]['timestamp'])
        col3.metric("Last Retraining", last_retrain.strftime('%Y-%m-%d'))
        
        # Detailed table
        st.markdown("#### Detailed History")
        df_history = pd.DataFrame([
            {
                "Version": f"v{h['version']}",
                "Timestamp": datetime.fromisoformat(h['timestamp']).strftime('%Y-%m-%d %H:%M'),
                "MAE": f"{h['metrics'].get('mae', 0):.3f}",
                "R² Score": f"{h['metrics'].get('r2', 0):.3f}",
                "File": h.get('file', 'N/A')
            } for h in history
        ])
        
        st.dataframe(df_history, use_container_width=True, hide_index=True)
    else:
        st.info("No retraining history available")

with log_tab3:
    st.markdown("### Drift Detection Details")
    
    if drift_data and drift_data.get('details'):
        details = drift_data['details']
        
        # Create detailed drift table
        drift_details = []
        for feature, result in details.items():
            drift_details.append({
                "Feature": feature,
                "Type": result.get('type', 'N/A'),
                "Drifted": "🔴 Yes" if result.get('drift_detected') else "🟢 No",
                "Statistic": f"{result.get('statistical_metrics', {}).get('statistic', 0):.4f}",
                "P-Value": f"{result.get('statistical_metrics', {}).get('p_value', 1.0):.4f}",
                "Reasons": ", ".join(result.get('reasons', ['None']))
            })
        
        df_drift_details = pd.DataFrame(drift_details)
        
        # Filter options
        show_all = st.checkbox("Show all features", value=False)
        
        if not show_all:
            df_drift_details = df_drift_details[df_drift_details['Drifted'] == "🔴 Yes"]
        
        st.dataframe(df_drift_details, use_container_width=True, hide_index=True)
        
        # Download option
        csv = df_drift_details.to_csv(index=False)
        st.download_button(
            label="📥 Download Drift Report CSV",
            data=csv,
            file_name=f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No drift details available")


# ============================================================================
# LIVE PREDICTION TAB
# ============================================================================

st.markdown("---")
st.subheader("🔮 Live Prediction & Testing")

with st.expander("📝 Make a Prediction", expanded=False):
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
        
        submitted = st.form_submit_button("🎯 Predict Grade", use_container_width=True)
        
        if submitted:
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
            
            df_input = pd.DataFrame([raw_input])
            model, preprocessor = load_prediction_artifacts()
            
            if model is not None and preprocessor is not None:
                try:
                    df_clean = clean_raw_data(df_input)
                    X_processed = preprocessor.transform(df_clean)
                    
                    if hasattr(model, 'predict'):
                        try:
                            pred = model.predict(X_processed)[0]
                        except:
                            pred = model.predict(df_clean)[0]
                    
                    st.success(f"### Predicted Final Grade (G3): **{pred:.2f}** / 20")
                    
                    # Performance interpretation
                    if pred >= 16:
                        st.balloons()
                        st.success("🎉 Excellent performance predicted!")
                    elif pred >= 12:
                        st.info("👍 Good performance predicted")
                    elif pred >= 10:
                        st.warning("⚠️ Passing grade predicted")
                    else:
                        st.error("❌ At-risk student - intervention recommended")
                    
                    # Save prediction
                    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                    INCOMING_UNLABELED.mkdir(parents=True, exist_ok=True)
                    save_path = INCOMING_UNLABELED / f"batch_{timestamp}.csv"
                    df_input.to_csv(save_path, index=False)
                    
                    # Log prediction
                    logger.add_log(
                        event_type="PREDICTION_MADE",
                        message=f"Predicted G3={pred:.2f} for student age {age}",
                        severity="INFO",
                        metadata={"prediction": float(pred), "G1": G1, "G2": G2}
                    )
                    
                    # Try DB logging
                    try:
                        version_str = f"v{meta_data.get('current_production_version', '1')}" if meta_data else "v1"
                        db_repo.insert_prediction(
                            features=raw_input,
                            prediction=float(pred),
                            model_version=version_str
                        )
                    except Exception as db_err:
                        pass  # Silent fail for DB
                    
                except Exception as e:
                    st.error(f"Prediction Error: {str(e)}")
            else:
                st.error("Model artifacts not found!")


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.caption("🛡️ DriftGuard-ML v2.0")
with col2:
    st.caption(f"⏰ Last Refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
with col3:
    st.caption("🔄 Auto-refresh: Manual")
