"""
MongoDB Connection Manager
==========================
Handles connection pooling and client management for MongoDB.
"""

import os
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from dotenv import load_dotenv

# Load env variables
load_dotenv()

class MongoManager:
    client: AsyncIOMotorClient = None
    db_name: str = "ml_system_db"
    
    def __init__(self):
        self.uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
        self.max_pool_size = int(os.getenv("MONGO_POOL_SIZE", 10))

    def connect(self):
        """Create Async Connection (for FastAPI)"""
        if self.client is None:
            print(f"[DB] Connecting to MongoDB at {self.uri.split('@')[-1]}...") # Hide auth details
            self.client = AsyncIOMotorClient(
                self.uri,
                maxPoolSize=self.max_pool_size
            )
            print("[DB] Connection established.")
            
    def close(self):
        if self.client:
            self.client.close()
            print("[DB] Connection closed.")
            
    def get_db(self):
        if self.client is None:
            self.connect()
        return self.client[self.db_name]

# Global Instance
db_manager = MongoManager()

class SyncMongoManager:
    """Synchronous Manager for scripts/dashboard"""
    client: MongoClient = None
    db_name: str = "ml_system_db"

    def __init__(self):
        self.uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")

    def get_db(self):
        if self.client is None:
            try:
                self.client = MongoClient(self.uri)
                # Quick ping
                self.client.admin.command('ping')
            except Exception as e:
                print(f"[DB Error] Could not connect: {e}")
                return None
        return self.client[self.db_name]

# Global Sync Instance
sync_db_manager = SyncMongoManager()
