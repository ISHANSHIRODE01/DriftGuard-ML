"""
Model Versioning System
=======================
Handles automatic versioning of model files and metadata tracking with thread safety.
"""

import joblib
import json
import shutil
import time
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
from database.repository import MLRepository

class ModelVersioner:
    """
    Manages model versioning and the production 'current_model.pkl' link.
    Thread-safe implementation using lock files.
    """
    
    def __init__(self, model_dir: str = "model"):
        self.model_dir = Path(model_dir)
        self.metadata_path = self.model_dir / "metadata.json"
        self.current_model_path = self.model_dir / "current_model.pkl"
        self.lock_file = self.model_dir / "metadata.lock"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.repo = MLRepository()

    def _acquire_lock(self, timeout=10):
        """Simple file-based spin lock."""
        start = time.time()
        while time.time() - start < timeout:
            try:
                # Create lock file exclusively
                with open(self.lock_file, 'x'):
                    return True
            except FileExistsError:
                time.sleep(0.1)
        raise TimeoutError("Could not acquire lock on metadata.json")

    def _release_lock(self):
        """Release the lock file."""
        if self.lock_file.exists():
            os.remove(self.lock_file)

    def _get_next_version(self) -> int:
        """Determines the next version number."""
        if not self.metadata_path.exists():
            return 1
        
        try:
            with open(self.metadata_path, 'r') as f:
                meta = json.load(f)
                return meta.get("latest_version", 0) + 1
        except (json.JSONDecodeError, FileNotFoundError):
             # Fallback scan of files
            versions = [0]
            for f in self.model_dir.glob("model_v*.pkl"):
                try:
                    v = int(f.stem.split('_v')[-1])
                    versions.append(v)
                except ValueError:
                    continue
            return max(versions) + 1

    def save_new_version(self, model: Any, metrics: Dict[str, float]) -> str:
        """
        Saves a new version of the model and updates the production pointer.
        Thread-safe.
        """
        self._acquire_lock()
        try:
            version = self._get_next_version()
            filename = f"model_v{version}.pkl"
            filepath = self.model_dir / filename
            
            # 1. Save the versioned pkl
            print(f"Saving new model version: {filename}...")
            joblib.dump(model, filepath)
            
            # 2. Update metadata.json
            metadata = {
                "latest_version": version,
                "current_production_version": version,
                "last_updated": datetime.now().isoformat(),
                "history": []
            }
            
            if self.metadata_path.exists():
                try:
                    with open(self.metadata_path, 'r') as f:
                        old_meta = json.load(f)
                        metadata["history"] = old_meta.get("history", [])
                except: pass

            entry = {
                "version": version,
                "timestamp": datetime.now().isoformat(),
                "metrics": metrics,
                "file": filename
            }
            metadata["history"].append(entry)
            
            # Atomic write via temp file
            temp_meta = self.metadata_path.with_suffix(".tmp")
            with open(temp_meta, 'w') as f:
                json.dump(metadata, f, indent=4)
            shutil.move(temp_meta, self.metadata_path)
            
            # 3. Automatically Update "current_model.pkl"
            shutil.copy(filepath, self.current_model_path)
            print(f"  ✓ Updated 'current_model.pkl' to point to version {version}")
            
            # 4. Log to MongoDB
            try:
                self.repo.insert_model_version(
                    version=version,
                    metrics=metrics,
                    filepath=str(filepath)
                )
            except Exception as e:
                print(f"  ⚠ Failed to log model version to DB: {e}")

            return str(filepath)
            
        finally:
            self._release_lock()

    def get_latest_metadata(self) -> Dict[str, Any]:
        if not self.metadata_path.exists():
            return {}
        try:
            with open(self.metadata_path, 'r') as f:
                return json.load(f)
        except:
            return {}
