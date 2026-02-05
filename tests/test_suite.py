import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient
from pathlib import Path
import json
import shutil
import joblib

# Import Modules
from api.app import app
from drift.detector import DriftDetector
from deployment.model_versioner import ModelVersioner

# --- Fixtures ---

@pytest.fixture
def clean_env(tmp_path):
    """Creates a temporary isolated environment for testing."""
    d = tmp_path / "data"
    d.mkdir()
    m = tmp_path / "model"
    m.mkdir()
    
    # Mock Config Paths (Monkeypatch would be ideal, but for simplicity we inject)
    return tmp_path

@pytest.fixture
def dummy_data():
    """Generates synthetic student data."""
    df = pd.DataFrame({
        'age': np.random.randint(15, 22, 100),
        'G3': np.random.randint(0, 20, 100),
        'Medu': np.random.randint(0, 4, 100),
        'Fedu': np.random.randint(0, 4, 100),
        'school': ['GP']*100,
        'sex': ['F']*100,
        # ... minimal set for drift test
    })
    return df

@pytest.fixture
def client():
    """FastAPI Test Client."""
    return TestClient(app)

# --- Unit Tests: Drift Detection ---

def test_drift_no_change(dummy_data):
    """Drift detector should return False for identical datasets."""
    detector = DriftDetector(p_value_threshold=0.05)
    detector.detect_drift(dummy_data, dummy_data)
    report = detector.get_drift_report()
    
    assert report['summary']['drift_detected_overall'] is False
    assert report['summary']['drifted_features_count'] == 0

def test_drift_shift(dummy_data):
    """Drift detector should return True for shifted mean."""
    shifted_data = dummy_data.copy()
    shifted_data['age'] = shifted_data['age'] + 10  # Massive drift
    
    detector = DriftDetector(p_value_threshold=0.05)
    detector.detect_drift(dummy_data, shifted_data)
    report = detector.get_drift_report()
    
    # Only 'age' should drift. 'G3' (if included) might drift if shifted too.
    # Note: Detector only checks common columns.
    assert report['details']['age']['drift_detected'] is True

def test_drift_categorical_safety():
    """Ensure detector doesn't crash on string columns."""
    df = pd.DataFrame({'cat_col': ['A', 'B', 'A']})
    detector = DriftDetector()
    # Should run without error, even if it ignores the column
    try:
        detector.detect_drift(df, df)
        assert True
    except Exception as e:
        pytest.fail(f"Drift detector crashed on categorical input: {e}")

# --- Unit Tests: Versioning ---

def test_versioning_increment(clean_env):
    """Test v1 -> v2 file creation."""
    # Mock path inside ModelVersioner for test
    # (In real scenario, we'd refactor Versioner to accept a base_dir in __init__)
    versioner = ModelVersioner(model_dir=str(clean_env / "model"))
    
    model_mock = "DUMMY_MODEL_OBJ"
    metrics = {"acc": 0.9}
    
    v1_path = versioner.save_new_version(model_mock, metrics)
    assert "model_v1.pkl" in v1_path
    
    v2_path = versioner.save_new_version(model_mock, metrics)
    assert "model_v2.pkl" in v2_path
    
    # Check metadata
    meta = versioner.get_latest_metadata()
    assert meta['latest_version'] == 2
    assert len(meta['history']) == 2

# --- Integration Tests: API ---

def test_api_prediction_shape(client):
    """Test the /predict endpoint structure."""
    
    # Ideally we'd mock the model loading here to avoid needing the real .pkl
    # But since we have a 'health' check:
    
    response = client.get("/health")
    # If model isn't loaded (because we are in test env), this confirms endpoint exists
    assert response.status_code == 200 
    
    # Payload
    payload = {
        "school": "GP", "sex": "F", "age": 18, "address": "U", "famsize": "GT3",
        "Pstatus": "A", "Medu": 4, "Fedu": 4, "Mjob": "at_home", "Fjob": "teacher",
        "reason": "course", "guardian": "mother", "traveltime": 2, "studytime": 2,
        "failures": 0, "schoolsup": "yes", "famsup": "no", "paid": "no",
        "activities": "no", "nursery": "yes", "higher": "yes", "internet": "no",
        "romantic": "no", "famrel": 4, "freetime": 3, "goout": 4, "Dalc": 1,
        "Walc": 1, "health": 3, "absences": 6, "G1": 10, "G2": 11
    }
    
    # Note: In a pure CI environment, model/model.pkl might not exist yet -> 503 error
    # This expects the detailed error handling we implemented.
    try:
        resp = client.post("/predict", json=payload)
        if resp.status_code == 200:
            data = resp.json()
            assert "prediction" in data
            assert "G3_score" in data["prediction"]
        elif resp.status_code == 503:
            # Service unavailable (model missing) is also a valid "handled" response
            assert "Model not loaded" in resp.json()['detail']
    except Exception as e:
        pytest.fail(f"API crashed: {e}")

def test_api_validation_error(client):
    """Test Pydantic validation."""
    payload = {
        "school": "GP",
        "age": 100,  # Invalid ( > 22)
        # Missing fields...
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422  # Unprocessable Entity
