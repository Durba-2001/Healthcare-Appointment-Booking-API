from fastapi import APIRouter, BackgroundTasks, HTTPException
from src.models.chat_schema import ChatMessage
from uuid import uuid4
import redis
from pymongo import MongoClient
from loguru import logger
import json

from src.utils.background_task import save_message, update_session_background
from src.utils.helper_func import safe_hset, extract_recommendation,extract_session
from src.mcp.mcp_client import MCPClient
from src.utils.config import REDIS_HOST, REDIS_PASSWORD, REDIS_PORT, MONGODB_URI

# ----------------------------
# Initialize Router
# ----------------------------
router = APIRouter()

# ----------------------------
# Database and Redis setup
# ----------------------------
r = redis.Redis(
    host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD, decode_responses=True
)
mongo_client = MongoClient(MONGODB_URI)
db = mongo_client["healthcare_app"]
sessions_collection = db["booking_sessions"]
booking_collection = db["bookings"]

# ----------------------------
# Initialize MCP client globally
# ----------------------------
mcp_client = MCPClient()


# ----------------------------
# Start a New Chat
# ----------------------------
@router.post("/new")
async def new_chat(payload: ChatMessage, background_tasks: BackgroundTasks):
    user_message = payload.message.strip()
    chat_id = str(uuid4())

    if not user_message:
        return {"chat_id": chat_id, "response": "Please enter a valid message."}

    # Initialize Redis session
    safe_hset(r, f"session:{chat_id}", {"stage": "start"})

    logger.info(f" New Chat Started | ID: {chat_id} | Message: {user_message}")

    # First message → recommend_service
    response_data = await mcp_client.process_user_message(chat_id, user_message, first_message=True)
    print(response_data)
    tool_output = response_data.get("response", "")
    clean_response = extract_recommendation(tool_output)

    # Save messages in background
    background_tasks.add_task(save_message, chat_id, "user", user_message)
    background_tasks.add_task(save_message, chat_id, "assistant", clean_response)

    return {"chat_id": chat_id, "response": clean_response}


# ----------------------------
# Continue Existing Chat
# ----------------------------
@router.post("/continue/{chat_id}")
async def continue_chat(chat_id: str, payload: ChatMessage, background_tasks: BackgroundTasks):
    user_message = payload.message.strip()
    if not user_message:
        return {"chat_id": chat_id, "response": "Please enter a valid message."}

    # Log and store user message
    background_tasks.add_task(save_message, chat_id, "user", user_message)
    logger.info(f"Continue Chat ({chat_id}) | Message: {user_message}")

    # Process with MCP — may call tools like select_professional or collect_user_info
    response_data = await mcp_client.process_user_message(chat_id, user_message, first_message=False)
    print(response_data)
    tool_output = response_data.get("response", "")
    clean_response = extract_recommendation(tool_output)
    # Save assistant reply
    background_tasks.add_task(save_message, chat_id, "assistant", clean_response)

    #  Background MongoDB session update (if MCP tool returned session_update)
    tool_used = response_data.get("tool_used", "")
    if tool_used in ["select_professional", "collect_user_info"]:
        session_update = extract_session(tool_output)
        if session_update and "chat_id" in session_update:
            background_tasks.add_task(update_session_background, session_update)
            logger.info(f"Background task scheduled for session update | chat_id={chat_id} | tool: {tool_used}")
        else:
            logger.debug(f"No valid session_update found for tool: {tool_used}")

    return {"chat_id": chat_id, "response": clean_response}

# ----------------------------
# Retrieve Booking Info
# ----------------------------
@router.get("/booking/{chat_id}")
async def get_booking_info(chat_id: str):
    booking_data = booking_collection.find_one({"chat_id": chat_id})
    if not booking_data:
        raise HTTPException(status_code=404, detail="Chat ID not found")

    booking_info = {key: booking_data.get(key) for key in [
        "professional_name",
        "service_type",
        "customer_name",
        "age",
        "contact",
        "email",
        "booking_date",
        "booking_time",
        "booking_id",
        "status"
    ]}

    return {"status": "success", "chat_id": chat_id, "booking_info": booking_info}

# ----------------------------
# Delete Booking Record
# ----------------------------
@router.delete("/booking/{chat_id}")
async def delete_booking_info(chat_id: str):
    result = booking_collection.delete_one({"chat_id": chat_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Chat ID not found.")

    return {"status": "success", "message": f"Booking with chat_id '{chat_id}' deleted successfully."}
