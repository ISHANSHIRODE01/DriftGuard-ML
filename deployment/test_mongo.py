"""
MongoDB Verification Script
===========================
Pings the database and verifies document counts in all collections.
"""

import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

def verify_connection():
    uri = os.getenv("MONGO_URI")
    if not uri:
        print("Error: MONGO_URI not found in .env")
        return

    print(f"Connecting to MongoDB...")
    try:
        client = MongoClient(uri)
        # Ping the server
        client.admin.command('ping')
        print("  [OK] Connection Successful! Ping reached Atlas Cluster.")
        
        db = client["ml_system_db"]
        collections = {
            "raw_data": "Training & Batch Records",
            "models": "Model Versions & Metrics",
            "predictions": "Inference Logs",
            "drift_reports": "Audit Logs"
        }

        print("\n--- DATABASE SUMMARY (ml_system_db) ---")
        for coll_name, description in collections.items():
            count = db[coll_name].count_documents({})
            print(f"  - {coll_name:15s}: {count:4d} entries ({description})")

        # 2. Perform a TEST INSERT and DELETE it to verify write permissions
        print("\nVerifying Write Permissions...")
        test_doc = {"test": True, "note": "Connection verification test"}
        result = db.test_connection.insert_one(test_doc)
        if result.inserted_id:
            print("  [OK] Successfully wrote a test document.")
            db.test_connection.delete_one({"_id": result.inserted_id})
            print("  [OK] Successfully cleaned up test document.")
        
        print("\nALL SYSTEMS GO. Your data is safely stored in the cloud.")

    except Exception as e:
        print(f"\nCONNECTION FAILED: {e}")

if __name__ == "__main__":
    verify_connection()
