from pymongo import MongoClient
from src.utils.config import  MONGODB_URI
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