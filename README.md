# 🛡️ Production-Grade Student Performance ML System

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-v0.125+-green.svg)](https://fastapi.tiangolo.com/)
[![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-green.svg)](https://www.mongodb.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Monitoring-red.svg)](https://streamlit.io/)

A comprehensive, production-ready Machine Learning system for predicting student performance with built-in data drift detection, automated retraining loops, and cloud-based observability.

---

## 🏗️ System Architecture

```text
                                  +-------------------+
                                  |   MongoDB Atlas   |
                                  | (Cloud Registry)  |
                                  +---------^---------+
                                            |
        +-----------------------------------+-----------------------------------+
        |                                   |                                   |
+-------+-------+           +---------------+---------------+           +-------+-------+
|  FastAPI API  |           |  Streamlit Monitoring Dashboard|           |   Retraining  |
| (Inference)   |           | (Drift & Performance Analytics)|           |   Pipeline    |
+-------+-------+           +---------------+---------------+           +-------+-------+
        |                                   |                                   |
        +--------------------+--------------+-------------+---------------------+
                             |                            |
                     +-------v-------+            +-------v-------+
                     | Preprocessing |            |  Model Registry|
                     | Single Source |            | (Versioning)  |
                     +---------------+            +---------------+
```

---

## 🚀 Key Features

### 1. **Robust Drift Detection**
*   **Numerical Features**: Uses the Kolmogorov-Smirnov (KS) Test to detect distribution shifts in grades and study habits.
*   **Categorical Features**: Implements Chi-Square analysis to identify changes in demographic or school-related features.
*   **Thresholding**: Configurable p-value (default < 0.05) and effect size thresholds.

### 2. **Automated Retraining Loop**
*   **Batch Ingestion**: Safe data flow separating `unlabeled` (inference) and `labeled` (retraining) data batches.
*   **Promotion Logic**: Only promotes a "Challenger" model if it significantly outperforms (MAE improvement > 0.01) the current "Champion."
*   **Data Windowing**: Prevents unbounded memory growth by training on the most recent $N$ records.

### 3. **Cloud-Native Persistence (MongoDB)**
*   **Live Prediction Logging**: Every inference request via API or Dashboard is logged to MongoDB for future labeling and audit.
*   **Artifact History**: Version-controlled model metrics, file paths, and drift reports stored in a centralized cloud registry.

### 4. **FAANG-Level Software Engineering**
*   **Modular Design**: Clean separation of concerns (Inference, Training, Preprocessing, DB Layers).
*   **Thread-Safety**: File locks protect metadata and model versioning during concurrent operations.
*   **Pre-processing Parity**: Guaranteed consistency between training and serving via a shared logic module.

---

## 🛠️ Tech Stack
- **Core**: Python 3.12, Scikit-Learn
- **Inference**: FastAPI, Uvicorn (4 Workers)
- **Monitoring**: Streamlit, Plotly
- **Database**: MongoDB Atlas (Motor, PyMongo)
- **Pipeline**: Joblib, Pandas, SciPy

---

## 📦 Getting Started

### 1. Prerequisites
```bash
pip install -r deployment/requirements.txt
```

### 2. Configuration
Create a `.env` file in the root directory:
```env
MONGO_URI=mongodb+srv://<user>:<password>@cluster.mongodb.net/ml_system_db
MONGO_POOL_SIZE=10
```

### 3. Initialize & Migrate
If you have local data, push it to the cloud:
```bash
$env:PYTHONPATH="."; python deployment/migrate_to_mongo.py
```

### 4. Running the System
*   **Dashboard**: `streamlit run dashboard/dashboard.py`
*   **API**: `uvicorn api.app:app --port 8000`
*   **Retrain Manually**: `python -m retraining.pipeline`

---

## 🧠 Resume Bullet
> "Architected and deployed a production-grade ML System for student performance prediction, featuring a custom Drift Detection module (KS-Test/Chi-Square) and an automated retraining loop. Engineered a cloud-persistent observability layer with MongoDB Atlas to log live inference results and track 10+ model versions, reducing training-serving skew through shared preprocessing pipelines."

---

## 📂 Folder Structure
```text
ml_drift_system/
├── api/                # FastAPI serving logic
├── dashboard/          # Monitoring UI
├── database/           # MongoDB DAL (Repository Pattern)
├── deployment/         # Docker, Requirements, Migrations
├── docs/               # System documentation
├── drift/              # Statistical detection logic
├── model/              # Live active artifacts (.pkl)
├── preprocessing/      # Centralized cleaning & ETL
├── retraining/         # Orchestration & promotion logic
├── training/           # Research & Baseline scripts
└── README.md           # System entry point
```
