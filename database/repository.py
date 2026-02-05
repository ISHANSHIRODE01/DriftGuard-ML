"""
Data Repository
===============
DAL (Data Access Layer) for storing ML artifacts in MongoDB.
"""

from datetime import datetime
from typing import Dict, Any, List
from .connection import db_manager, sync_db_manager

class MLRepository:
    """Synchronous repository for Dashboard and Training Scripts"""
    
    def __init__(self):
        self.db = sync_db_manager.get_db()
        
    def _ensure_connected(self):
        if self.db is None:
            self.db = sync_db_manager.get_db()
        return self.db is not None

    def insert_prediction(self, features: Dict, prediction: float, model_version: str):
        """Log a single prediction request."""
        if not self._ensure_connected(): return
        
        doc = {
            "timestamp": datetime.utcnow(),
            "type": "inference",
            "features": features,
            "prediction_g3": prediction,
            "model_version": model_version,
            "source": "dashboard"
        }
        self.db.predictions.insert_one(doc)

    def insert_model_version(self, version: int, metrics: Dict, filepath: str):
        """Log a new model version."""
        if not self._ensure_connected(): return

        doc = {
            "version": version,
            "created_at": datetime.utcnow(),
            "metrics": metrics,
            "filepath": filepath,
            "is_active": True
        }
        self.db.models.insert_one(doc)
        
    def insert_drift_report(self, report: Dict):
        """Log a drift detection report."""
        if not self._ensure_connected(): return

        doc = {
            "timestamp": datetime.utcnow(),
            "summary": report.get("summary", {}),
            "details": report.get("details", {}),
            "type": "drift_report"
        }
        self.db.drift_reports.insert_one(doc)

    def insert_batch_data(self, df_dict: List[Dict], dataset_type: str = "unlabeled"):
        """Bulk insert labeled/unlabeled data batches."""
        if not self._ensure_connected() or not df_dict: return
        
        # Add metadata to each row
        timestamp = datetime.utcnow()
        for row in df_dict:
            row["ingested_at"] = timestamp
            row["dataset_type"] = dataset_type
            
        self.db.raw_data.insert_many(df_dict)
        print(f"[DB] Inserted {len(df_dict)} rows into 'raw_data'.")


# --- Async Repository (for API) ---
class AsyncMLRepository:
    """Async repository for FastAPI"""
    
    def __init__(self):
        self.db_manager = db_manager

    async def log_prediction(self, features: Dict, prediction: float, model_version: str):
        db = self.db_manager.get_db()
        doc = {
            "timestamp": datetime.utcnow(),
            "type": "inference",
            "features": features,
            "prediction_g3": prediction,
            "model_version": model_version,
            "source": "api"
        }
        await db.predictions.insert_one(doc)
