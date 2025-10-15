# src/utils/background_task.py
from pymongo import MongoClient
from src.utils.config import MONGODB_URI
from datetime import datetime, timezone

mongo_client = MongoClient(MONGODB_URI)
db = mongo_client["healthcare_app"]
chats_collection = db["chats"]

def save_message(chat_id: str, role: str, content: str):
    """Persist messages to MongoDB Chats collection"""
    chats_collection.update_one(
        {"chat_id": chat_id},
        {"$push": {"messages": {"role": role, "content": content, "timestamp": datetime.now(timezone.utc)}}},
        upsert=True
    )

def update_session_background(chat_id: str, update_dict: dict):
    """Background task to update booking_sessions collection"""
    sessions_collection = db["booking_sessions"]
    sessions_collection.update_one(
        {"chat_id": chat_id},
        {"$set": update_dict},
        upsert=True
    )
