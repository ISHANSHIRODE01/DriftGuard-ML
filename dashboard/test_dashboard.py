"""
Dashboard Testing Utility
=========================
Generate sample data to test the enhanced dashboard features.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Paths
DRIFT_REPORT_PATH = Path("Data/reports/drift_report.json")
DRIFT_HISTORY_PATH = Path("Data/reports/drift_history.json")
SYSTEM_LOGS_PATH = Path("Data/reports/system_logs.json")
METADATA_PATH = Path("model/metadata.json")

# Ensure directories exist
Path("Data/reports").mkdir(parents=True, exist_ok=True)


def generate_drift_report(severity="moderate"):
    """
    Generate a sample drift report for testing
    
    Args:
        severity: "low", "moderate", "high", or "critical"
    """
    
    severity_config = {
        "low": {"drifted_count": 2, "p_value_range": (0.04, 0.05)},
        "moderate": {"drifted_count": 8, "p_value_range": (0.01, 0.03)},
        "high": {"drifted_count": 15, "p_value_range": (0.001, 0.01)},
        "critical": {"drifted_count": 22, "p_value_range": (0.0001, 0.001)}
    }
    
    config = severity_config.get(severity, severity_config["moderate"])
    
    # Sample features
    all_features = [
        "age", "absences", "failures", "studytime", "G1", "G2",
        "Medu", "Fedu", "traveltime", "freetime", "goout", "Dalc", "Walc",
        "health", "famrel", "school", "sex", "address", "famsize", "Pstatus",
        "Mjob", "Fjob", "reason", "guardian", "schoolsup", "famsup", "paid",
        "activities", "nursery", "higher"
    ]
    
    drifted_features = np.random.choice(all_features, config["drifted_count"], replace=False)
    
    details = {}
    for feature in all_features:
        is_numerical = feature in ["age", "absences", "failures", "studytime", "G1", "G2", 
                                   "Medu", "Fedu", "traveltime", "freetime", "goout", 
                                   "Dalc", "Walc", "health", "famrel"]
        
        is_drifted = feature in drifted_features
        
        if is_drifted:
            p_value = np.random.uniform(*config["p_value_range"])
            statistic = np.random.uniform(0.2, 0.5)
            reasons = ["KS-Test P-value significant"] if is_numerical else ["Chi-Square P-value significant"]
            
            if is_numerical and np.random.random() > 0.5:
                reasons.append("Mean shift > 10%")
        else:
            p_value = np.random.uniform(0.1, 0.9)
            statistic = np.random.uniform(0.01, 0.1)
            reasons = []
        
        details[feature] = {
            "type": "numerical" if is_numerical else "categorical",
            "drift_detected": is_drifted,
            "statistical_metrics": {
                "statistic": float(statistic),
                "p_value": float(p_value)
            },
            "reasons": reasons
        }
    
    report = {
        "summary": {
            "total_features_analyzed": len(all_features),
            "drifted_features_count": config["drifted_count"],
            "drift_detected_overall": config["drifted_count"] > 0,
            "timestamp": datetime.now().isoformat()
        },
        "details": details
    }
    
    with open(DRIFT_REPORT_PATH, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"[OK] Generated {severity.upper()} severity drift report")
    print(f"   - {config['drifted_count']} features drifting")
    print(f"   - Saved to: {DRIFT_REPORT_PATH}")


def generate_drift_history(num_records=20):
    """Generate historical drift records"""
    
    history = []
    base_time = datetime.now() - timedelta(days=num_records)
    
    for i in range(num_records):
        # Simulate drift increasing over time then stabilizing
        if i < num_records // 2:
            drift_score = np.random.uniform(10, 40)  # Low to moderate
            drifted_features = np.random.randint(1, 8)
        else:
            drift_score = np.random.uniform(30, 70)  # Moderate to high
            drifted_features = np.random.randint(5, 15)
        
        record = {
            "timestamp": (base_time + timedelta(days=i)).isoformat(),
            "drift_score": float(drift_score),
            "drifted_features": int(drifted_features),
            "total_features": 30,
            "drift_percentage": float((drifted_features / 30) * 100)
        }
        
        history.append(record)
    
    with open(DRIFT_HISTORY_PATH, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"[OK] Generated drift history with {num_records} records")
    print(f"   - Saved to: {DRIFT_HISTORY_PATH}")


def generate_system_logs(num_logs=25):
    """Generate sample system logs"""
    
    event_types = [
        ("SYSTEM_START", "System initialized successfully", "INFO"),
        ("DRIFT_DETECTED", "Drift detected in multiple features", "WARNING"),
        ("MODEL_RETRAINED", "Model retrained successfully", "SUCCESS"),
        ("PREDICTION_MADE", "Prediction generated for student", "INFO"),
        ("DATA_INGESTED", "New batch data ingested", "INFO"),
        ("ALERT_TRIGGERED", "Performance threshold exceeded", "WARNING"),
        ("BACKUP_CREATED", "Model backup created", "INFO"),
        ("ERROR_OCCURRED", "Failed to connect to database", "ERROR")
    ]
    
    logs = []
    base_time = datetime.now() - timedelta(hours=num_logs)
    
    for i in range(num_logs):
        event_type, message, severity = event_types[np.random.randint(0, len(event_types))]
        
        # Add some variation to messages
        if event_type == "DRIFT_DETECTED":
            num_features = np.random.randint(3, 15)
            message = f"Drift detected in {num_features} features"
        elif event_type == "PREDICTION_MADE":
            age = np.random.randint(15, 22)
            message = f"Prediction generated for student age {age}"
        
        log_entry = {
            "timestamp": (base_time + timedelta(hours=i)).isoformat(),
            "event_type": event_type,
            "message": message,
            "severity": severity,
            "metadata": {
                "index": i,
                "random_value": float(np.random.random())
            }
        }
        
        logs.append(log_entry)
    
    with open(SYSTEM_LOGS_PATH, 'w') as f:
        json.dump(logs, f, indent=2)
    
    print(f"[OK] Generated {num_logs} system logs")
    print(f"   - Saved to: {SYSTEM_LOGS_PATH}")


def enhance_metadata():
    """Add more retraining history to metadata"""
    
    if not METADATA_PATH.exists():
        print("[WARNING] Metadata file not found. Creating new one...")
        metadata = {
            "latest_version": 1,
            "current_production_version": 1,
            "last_updated": datetime.now().isoformat(),
            "history": []
        }
    else:
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)
    
    # Add 5 more versions with improving performance
    current_version = metadata.get("latest_version", 1)
    base_mae = 2.5
    base_r2 = 0.75
    
    for i in range(5):
        version = current_version + i + 1
        
        # Simulate improvement
        mae = base_mae - (i * 0.15) + np.random.uniform(-0.05, 0.05)
        r2 = base_r2 + (i * 0.03) + np.random.uniform(-0.01, 0.01)
        
        timestamp = datetime.now() - timedelta(days=(5 - i) * 7)
        
        history_entry = {
            "version": version,
            "timestamp": timestamp.isoformat(),
            "metrics": {
                "mae": float(max(1.0, mae)),  # Ensure MAE doesn't go below 1.0
                "r2": float(min(0.95, r2))    # Ensure R2 doesn't exceed 0.95
            },
            "file": f"model_v{version}.pkl"
        }
        
        metadata["history"].append(history_entry)
    
    metadata["latest_version"] = current_version + 5
    metadata["current_production_version"] = current_version + 5
    metadata["last_updated"] = datetime.now().isoformat()
    
    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"[OK] Enhanced metadata with 5 new versions")
    print(f"   - Current version: v{metadata['current_production_version']}")
    print(f"   - Total retrainings: {len(metadata['history'])}")


def run_full_test_setup(drift_severity="moderate"):
    """
    Run complete test data generation
    
    Args:
        drift_severity: "low", "moderate", "high", or "critical"
    """
    print("=" * 60)
    print("DASHBOARD TEST DATA GENERATOR")
    print("=" * 60)
    print()
    
    print("Generating test data...")
    print()
    
    generate_drift_report(severity=drift_severity)
    print()
    
    generate_drift_history(num_records=20)
    print()
    
    generate_system_logs(num_logs=30)
    print()
    
    enhance_metadata()
    print()
    
    print("=" * 60)
    print("[SUCCESS] TEST DATA GENERATION COMPLETE!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Run the dashboard:")
    print("   streamlit run dashboard/dashboard_v2.py")
    print()
    print("2. Check these features:")
    print("   - Drift score in sidebar")
    print("   - Alert banners at top")
    print("   - Performance metrics with deltas")
    print("   - Drift trend chart")
    print("   - System logs tab")
    print()
    print("3. Test different drift severities:")
    print("   python dashboard/test_dashboard.py --severity low")
    print("   python dashboard/test_dashboard.py --severity high")
    print("   python dashboard/test_dashboard.py --severity critical")
    print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate test data for dashboard")
    parser.add_argument(
        "--severity",
        choices=["low", "moderate", "high", "critical"],
        default="moderate",
        help="Drift severity level to simulate"
    )
    
    args = parser.parse_args()
    
    run_full_test_setup(drift_severity=args.severity)
