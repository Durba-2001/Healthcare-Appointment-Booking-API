# src/utils/background_task.py
from pymongo import MongoClient
from src.utils.config import MONGODB_URI
from datetime import datetime, timezone
from loguru import logger
mongo_client = MongoClient(MONGODB_URI)
db = mongo_client["healthcare_app"]
chats_collection = db["chats"]
sessions_collection=db["booking_sessions"]
def save_message(chat_id: str, role: str, content: str):
    """Persist messages to MongoDB Chats collection"""
    chats_collection.update_one(
        {"chat_id": chat_id},
        {"$push": {"messages": {"role": role, "content": content, "timestamp": datetime.now(timezone.utc)}}},
        upsert=True
    )

def update_session_background(session_update: dict):
    """
    Updates MongoDB booking session info asynchronously.
    Only MongoDB — Redis is not touched.
    """
    try:
        chat_id = session_update.get("chat_id")
        if not chat_id:
            logger.warning("No chat_id found in session_update payload")
            return

        sessions_collection.update_one(
            {"chat_id": chat_id},
            {"$set": session_update},
            upsert=True
        )
        logger.info(f" MongoDB session updated in background for chat_id={chat_id}")

    except Exception as e:
        logger.error(f" Background session update failed: {e}")