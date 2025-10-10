# endpoints/chat_router.py
from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel
from uuid import uuid4
from datetime import datetime, timezone
import json
from redis import Redis
from pymongo import MongoClient
from loguru import logger
from fastapi.concurrency import run_in_threadpool
from config import ACCESS_TOKEN, REDIS_HOST, REDIS_PORT, REDIS_PASSWORD, MONGODB_URI
from mcp_tools import (
    recommend_service,
    list_professionals,
    select_professional,
    collect_user_info,
    check_availability
)
from fastmcp.tools.tool import TextContent

router = APIRouter()

# --------------------------
# Redis + Mongo
# --------------------------
r = Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD, decode_responses=True)
mongo_client = MongoClient(MONGODB_URI)
db = mongo_client["healthcare_app"]
sessions_collection = db["booking_sessions"]
chats_collection = db["chats"]

# --------------------------
# Pydantic Models
# --------------------------
class ChatMessage(BaseModel):
    message: str

# --------------------------
# Background Task
# --------------------------
def save_message(chat_id: str, role: str, content: str):
    """Persist messages to MongoDB Chats collection"""
    chats_collection.update_one(
        {"chat_id": chat_id},
        {"$push": {"messages": {"role": role, "content": content, "timestamp": datetime.now(timezone.utc)}}},
        upsert=True
    )

# --------------------------
# Safe Redis Write Helper
# --------------------------
def safe_hset(redis_conn, key, mapping):
    """Safely store mapping in Redis without None values."""
    if not mapping:
        return
    clean = {k: v for k, v in mapping.items() if v is not None}
    if clean:
        redis_conn.hset(key, mapping=clean)

# --------------------------
# Helper Functions
# --------------------------
async def run_tool(tool, params):
    """Execute a tool safely, handling async or sync"""
    try:
        if callable(getattr(tool, "run", None)):
            if getattr(tool.run, "__code__", None) and tool.run.__code__.co_flags & 0x80:
                return await tool.run(params)  # async
            else:
                return await run_in_threadpool(tool.run, params)  # sync
        logger.error(f"Tool {tool} has no run() method")
        return None
    except Exception as e:
        logger.exception(f"Tool execution failed: {tool}, params={params}, error={e}")
        return None

def convert_tool_response(tool_response):
    """Convert any tool response to dict safely"""
    from fastmcp.tools.tool import TextContent
    if isinstance(tool_response, dict):
        return tool_response
    # Handle list of TextContent objects
    if isinstance(tool_response, list):
        return {"content": [{"text": t.text if isinstance(t, TextContent) else str(t)} for t in tool_response]}
    # Handle single TextContent
    if isinstance(tool_response, TextContent):
        return {"content": [{"text": tool_response.text}]}
    if hasattr(tool_response, "__dict__"):
        return tool_response.__dict__
    return {"raw": str(tool_response)}

def extract_recommendation(tool_response):
    """Universal extractor for all tools including TextContent"""
    try:
        if not tool_response:
            return "⚠️ No recommendation available."
        tool_response = convert_tool_response(tool_response)

        # 1. structured_content
        structured = tool_response.get("structured_content", {})
        for key in ["recommendation", "prompt"]:
            if key in structured and structured[key]:
                return structured[key]

        # 2. content list
        content_list = tool_response.get("content", [])
        if isinstance(content_list, list):
            for item in content_list:
                text = item.get("text")
                if text:
                    try:
                        parsed = json.loads(text)
                        if "recommendation" in parsed:
                            return parsed["recommendation"]
                        if "prompt" in parsed:
                            return parsed["prompt"]
                    except json.JSONDecodeError:
                        return text

        # 3. message field
        if "message" in tool_response:
            return tool_response["message"]

        # 4. fallback: first string in dict
        for k, v in tool_response.items():
            if isinstance(v, str):
                return v

        return "⚠️ No recommendation available."
    except Exception as e:
        logger.error(f"Failed to extract recommendation: {e}, tool_response={tool_response}")
        return "⚠️ No recommendation available."

# --------------------------
# Stage-aware input processing
# --------------------------
def preprocess_user_input(stage: str, user_message: str):
    """
    Ensure user input is valid for the current stage.
    """
    if stage in ["awaiting_user_info", "awaiting_availability"]:
        try:
            data = json.loads(user_message)
            # Clean None values to prevent Redis DataError
            return {k: v for k, v in data.items() if v is not None}
        except json.JSONDecodeError:
            return {}
    return user_message.strip()  # keep full string for city/professional

# --------------------------
# New Chat Endpoint
# --------------------------
@router.post("/new-chat")
async def new_chat(payload: ChatMessage, background_tasks: BackgroundTasks):
    user_message = payload.message.strip()
    chat_id = str(uuid4())

    # Initialize session
    safe_hset(r, f"session:{chat_id}", {"stage": "recommendation"})
    sessions_collection.insert_one({"chat_id": chat_id, "stage": "recommendation", "messages": []})

    try:
        processed_input = preprocess_user_input("recommendation", user_message)

        # Run recommendation tool
        tool_response = await run_tool(recommend_service, {
            "chat_id": chat_id,
            "user_message": processed_input,
            "token": ACCESS_TOKEN
        })

        # Convert to dict for safe stage handling
        tool_response_dict = convert_tool_response(tool_response)
        response_text = extract_recommendation(tool_response_dict)

        # Determine next stage (always move to awaiting_city after recommendation)
        next_stage = "awaiting_city"

        # Update Redis and MongoDB
        safe_hset(r, f"session:{chat_id}", {"stage": next_stage, "recommendation": response_text})
        sessions_collection.update_one(
            {"chat_id": chat_id},
            {"$set": {"stage": next_stage, "recommendation": response_text}}
        )

        logger.info(f"[New Chat] Tool response: {tool_response_dict}")

    except Exception as e:
        logger.exception(f"[New Chat] Error chat_id={chat_id}: {e}")
        response_text = "⚠️ Something went wrong. Please try again."

    # Save messages asynchronously
    background_tasks.add_task(save_message, chat_id, "user", user_message)
    background_tasks.add_task(save_message, chat_id, "assistant", response_text)

    return {"chat_id": chat_id, "response": response_text}


# --------------------------
# Continue Chat Endpoint
# --------------------------
@router.post("/continue-chat/{chat_id}")
async def continue_chat(chat_id: str, payload: ChatMessage, background_tasks: BackgroundTasks):
    user_message = payload.message.strip()

    # Fetch stage from Redis (fast), fallback to MongoDB
    session_cache = r.hgetall(f"session:{chat_id}") or {}
    session_data = sessions_collection.find_one({"chat_id": chat_id}) if not session_cache else {}
    stage = session_cache.get("stage", session_data.get("stage", "recommendation"))

    response_text = "⚠️ No recommendation available."
    next_stage = stage

    try:
        logger.info(f"[Continue Chat] Chat ID: {chat_id}, Stage: {stage}, User: {user_message}")

        # Parse JSON input if possible
        try:
            processed_input = json.loads(user_message)
        except json.JSONDecodeError:
            processed_input = user_message.strip()

        # --------------------------
        # Stage-based execution
        # --------------------------
        if stage == "recommendation":
            tool_response = await run_tool(recommend_service, {
                "chat_id": chat_id,
                "user_message": processed_input,
                "token": ACCESS_TOKEN
            })
            response_text = extract_recommendation(tool_response)
            next_stage = "awaiting_city"

        elif stage == "awaiting_city":
            tool_response = await run_tool(list_professionals, {
                "chat_id": chat_id,
                "city": processed_input,
                "token": ACCESS_TOKEN
            })
            response_text = extract_recommendation(tool_response)
            next_stage = "awaiting_prof_selection"

        elif stage == "awaiting_prof_selection":
            tool_response = await run_tool(select_professional, {
                "chat_id": chat_id,
                "name": processed_input,
                "token": ACCESS_TOKEN
            })
            response_text = extract_recommendation(tool_response)
            next_stage = "awaiting_user_info"

        elif stage == "awaiting_user_info":
            user_info_keys = ["name", "age", "contact", "email"]
            tool_params = {"chat_id": chat_id, "token": ACCESS_TOKEN}

            if isinstance(processed_input, dict):
                tool_params.update({k: processed_input.get(k) for k in user_info_keys if k in processed_input})

            tool_response = await run_tool(collect_user_info, tool_params)
            response_text = extract_recommendation(tool_response)
            next_stage = "awaiting_availability"

        elif stage == "awaiting_availability":
            booking_keys = ["booking_date", "booking_time"]
            tool_params = {"chat_id": chat_id, "token": ACCESS_TOKEN}

            if isinstance(processed_input, dict):
                tool_params.update({k: processed_input.get(k) for k in booking_keys if k in processed_input})

            tool_response = await run_tool(check_availability, tool_params)
            response_text = extract_recommendation(tool_response)

            status = tool_response.get("status") if isinstance(tool_response, dict) else None
            booking_id = tool_response.get("booking_id") if isinstance(tool_response, dict) else None

            if status == "confirmed":
                next_stage = "complete"
                r.delete(f"session:{chat_id}")

                # Save chat messages asynchronously
                background_tasks.add_task(save_message, chat_id, "user", user_message)
                background_tasks.add_task(save_message, chat_id, "assistant", response_text)

                # Return booking_id explicitly
                return {
                    "chat_id": chat_id,
                    "response": response_text,
                    "status": status,
                    "booking_id": booking_id
                }
            else:
                next_stage = "awaiting_availability"

        else:
            response_text = "Conversation complete. Thank you!"
            next_stage = "complete"
            r.delete(f"session:{chat_id}")

        # Update stage in Redis/Mongo if not complete
        if next_stage != "complete":
            safe_hset(r, f"session:{chat_id}", {"stage": next_stage})
            sessions_collection.update_one({"chat_id": chat_id}, {"$set": {"stage": next_stage}}, upsert=True)

    except Exception as e:
        logger.exception(f"[Continue Chat] Error chat_id={chat_id}, stage={stage}: {e}")
        response_text = "⚠️ Something went wrong. Please try again."

    # Save chat messages asynchronously
    background_tasks.add_task(save_message, chat_id, "user", user_message)
    background_tasks.add_task(save_message, chat_id, "assistant", response_text)

    return {"chat_id": chat_id, "response": response_text, "status": "in_progress"}
