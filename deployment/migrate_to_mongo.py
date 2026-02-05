"""
Data Migration Script
=====================
Migrates local CSV/JSON artifacts into MongoDB.
Run this ONCE when you have the DB connection.
"""

import pandas as pd
import json
import os
from pathlib import Path
import glob
from database.repository import MLRepository

def migrate_all():
    repo = MLRepository()
    if not repo._ensure_connected():
        print("Cannot connect to MongoDB. Check MONGO_URI env var.")
        return

    print("Starting Data Migration to MongoDB...")
    
    # 1. Historical Training Data
    print("\n1. Migrating Training Data...")
    try:
        train_path = Path("Data/raw/train.csv")
        if train_path.exists():
            df = pd.read_csv(train_path)
            data = df.to_dict(orient="records")
            repo.insert_batch_data(data, dataset_type="training_historical")
    except Exception as e:
        print(f"Error: {e}")

    # 2. Incoming Batches
    print("\n2. Migrating Incoming Batches...")
    for label_type in ["labeled", "unlabeled"]:
        path = Path(f"Data/incoming/{label_type}")
        if path.exists():
            for f in glob.glob(str(path / "*.csv")):
                try:
                    df = pd.read_csv(f)
                    data = df.to_dict(orient="records")
                    repo.insert_batch_data(data, dataset_type=f"incoming_{label_type}")
                except Exception as e:
                    print(f"Skipping {f}: {e}")

    # 3. Model Metadata
    print("\n3. Migrating Model History...")
    meta_path = Path("model/metadata.json")
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            meta = json.load(f)
            for entry in meta.get("history", []):
                # Align schema
                doc = {
                    "version": entry.get("version"),
                    "created_at": entry.get("timestamp"), # String format kept
                    "metrics": entry.get("metrics"),
                    "filepath": entry.get("file"),
                    "is_active": (entry.get("version") == meta.get("current_production_version"))
                }
                # Use raw insert to bypass datetime.utcnow override in repository method
                repo.db.models.insert_one(doc)
    
    print("\nMigration Complete!")

if __name__ == "__main__":
    uri = input("Enter MongoDB URI (press Enter for localhost): ").strip()
    if uri:
        os.environ["MONGO_URI"] = uri
    migrate_all()
