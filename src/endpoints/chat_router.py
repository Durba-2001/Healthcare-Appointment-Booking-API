# endpoints/chat_router.py
from fastapi import APIRouter, BackgroundTasks
from src.models.chat_schema import ChatMessage
from uuid import uuid4
from redis import Redis
from pymongo import MongoClient
from loguru import logger

from src.utils.config import ACCESS_TOKEN, REDIS_HOST, REDIS_PORT, REDIS_PASSWORD, MONGODB_URI
from src.tools.mcp_tools import (
    recommend_service,
    list_professionals,
    select_professional,
    collect_user_info,
    check_availability
)
from src.utils.background_task import save_message
from src.utils.helper_func import (
    run_tool,
    convert_tool_response,
    extract_recommendation,
    extract_user_info_from_text,
    extract_booking_info_from_text
)
router = APIRouter()

# --------------------------
# Redis + Mongo
# --------------------------
r = Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD, decode_responses=True)
mongo_client = MongoClient(MONGODB_URI)
db = mongo_client["healthcare_app"]
sessions_collection = db["booking_sessions"]
booking_collection = db["bookings"]


# --------------------------
# Safe Redis Write Helper
# --------------------------
def safe_hset(redis_conn, key, mapping):
    """Safely store mapping in Redis without None values."""
    if not mapping:
        return

    clean = {}
    for k, v in mapping.items():
        if v is not None:
            clean[k] = v

    if clean:
        redis_conn.hset(key, mapping=clean)

# --------------------------
# Stage-aware input processing
# --------------------------
def preprocess_user_input(stage: str, user_message: str):
    """Preprocess input depending on conversation stage."""
    if stage == "awaiting_user_info":
        return extract_user_info_from_text(user_message)
    elif stage == "awaiting_availability":
        return extract_booking_info_from_text(user_message)
    else:
        return user_message.strip()

# --------------------------
# New Chat Endpoint
# --------------------------
@router.post("/new-chat")
async def new_chat(payload: ChatMessage, background_tasks: BackgroundTasks):
    user_message = payload.message.strip()
    chat_id = str(uuid4())

    safe_hset(r, f"session:{chat_id}", {"stage": "recommendation"})
    sessions_collection.insert_one({"chat_id": chat_id, "stage": "recommendation", "messages": [user_message]})

    try:
        processed_input = preprocess_user_input("recommendation", user_message)

        tool_response = await run_tool(recommend_service, {
            "chat_id": chat_id,
            "user_message": processed_input,
            "token": ACCESS_TOKEN
        })

        tool_response_dict = convert_tool_response(tool_response)
        response_text = extract_recommendation(tool_response_dict)

        next_stage = "awaiting_city"
        safe_hset(r, f"session:{chat_id}", {"stage": next_stage, "recommendation": response_text})
        sessions_collection.update_one(
            {"chat_id": chat_id},
            {"$set": {"stage": next_stage, "recommendation": response_text}}
        )
    except Exception as e:
        logger.exception(f"[New Chat] Error chat_id={chat_id}: {e}")
        response_text = "⚠️ Something went wrong. Please try again."

    background_tasks.add_task(save_message, chat_id, "user", user_message)
    background_tasks.add_task(save_message, chat_id, "assistant", response_text)

    return {"chat_id": chat_id, "response": response_text}

# --------------------------
# Continue Chat Endpoint
# --------------------------
@router.post("/continue-chat/{chat_id}")
async def continue_chat(chat_id: str, payload: ChatMessage, background_tasks: BackgroundTasks):
    user_message = payload.message.strip()
    session_cache = r.hgetall(f"session:{chat_id}") or {}
    session_data = sessions_collection.find_one({"chat_id": chat_id}) or {}
    stage = session_cache.get("stage") or session_data.get("stage") or "recommendation"

    response_text = "⚠️ No recommendation available."
    next_stage = stage

    try:
        logger.info(f"[Continue Chat] Chat ID: {chat_id}, Stage: {stage}, User: {user_message}")
        processed_input = preprocess_user_input(stage, user_message)

        if stage == "recommendation":
            tool_response = await run_tool(recommend_service, {
                "chat_id": chat_id,
                "user_message": str(processed_input),
                "token": ACCESS_TOKEN
            })
            response_text = extract_recommendation(tool_response)
            next_stage = "awaiting_city"

        elif stage == "awaiting_city":
            city_str = processed_input if isinstance(processed_input, str) else ""
            tool_response = await run_tool(list_professionals, {
                "chat_id": chat_id,
                "user_message": city_str,
                "token": ACCESS_TOKEN
            })
            response_text = extract_recommendation(tool_response)
            next_stage = "awaiting_prof_selection"

        elif stage == "awaiting_prof_selection":
            prof_name = processed_input if isinstance(processed_input, str) else ""
            tool_response = await run_tool(select_professional, {
                "chat_id": chat_id,
                "user_message": prof_name,
                "token": ACCESS_TOKEN
            })
            response_text = extract_recommendation(tool_response)
            next_stage = "awaiting_user_info"

        elif stage == "awaiting_user_info":
            user_info = processed_input if isinstance(processed_input, dict) else {}
            tool_args = {"chat_id": chat_id, **user_info, "token": ACCESS_TOKEN}
            tool_response = await run_tool(collect_user_info, tool_args)
            response_text = extract_recommendation(tool_response)
            next_stage = "awaiting_availability"

        elif stage == "awaiting_availability":
            booking_info = processed_input if isinstance(processed_input, dict) else {}
            date_time_str = f"{booking_info.get('booking_date','')} {booking_info.get('booking_time','')}"
            tool_response = await run_tool(check_availability, {
                "chat_id": chat_id,
                "user_message": date_time_str.strip(),
                "token": ACCESS_TOKEN
            })

            structured = getattr(tool_response, "structured_content", None)
            if structured:
                status = structured.get("status")
                booking_id = structured.get("booking_id")
                response_text = structured.get("recommendation")
            else:
                status = None
                booking_id = None
                response_text = str(tool_response)

            logger.info(f"[Continue Chat] Availability tool response: {structured}")
            logger.info(f"Booking status: {status}, ID: {booking_id}")

            if status == "confirmed":
                next_stage = "complete"
                r.delete(f"session:{chat_id}")

                background_tasks.add_task(save_message, chat_id, "user", user_message)
                background_tasks.add_task(save_message, chat_id, "assistant", response_text)

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

        if next_stage != "complete":
            safe_hset(r, f"session:{chat_id}", {"stage": next_stage})
            sessions_collection.update_one(
                {"chat_id": chat_id},
                {"$set": {"stage": next_stage}},
                upsert=True
            )

    except Exception as e:
        logger.exception(f"[Continue Chat] Error chat_id={chat_id}, stage={stage}: {e}")
        response_text = "⚠️ Something went wrong. Please try again."

    background_tasks.add_task(save_message, chat_id, "user", user_message)
    background_tasks.add_task(save_message, chat_id, "assistant", response_text)

    return {"chat_id": chat_id, "response": response_text, "status": "in_progress"}

# --------------------------
# Get Booking Info by Chat ID
# --------------------------
@router.get("/booking-info/{chat_id}")
async def get_booking_info(chat_id: str):
    booking_data = booking_collection.find_one({"chat_id": chat_id})
    
    if not booking_data:
        return {"status": "error", "message": "Chat ID not found."}
    
    booking_info = {
        "professional_name": booking_data.get("professional_name"),
        "service_type": booking_data.get("service_type"),
        "customer_name": booking_data.get("customer_name"),
        "age": booking_data.get("age"),
        "contact": booking_data.get("contact"),
        "email": booking_data.get("email"),
        "booking_date": booking_data.get("booking_date"),
        "booking_time": booking_data.get("booking_time"),
        "booking_id": booking_data.get("booking_id"),
        "status": booking_data.get("status")
    }

    return {
        "status": "success",
        "chat_id": chat_id,
        "booking_info": booking_info
    }

from fastapi import HTTPException

@router.delete("/booking-info/{chat_id}")
async def delete_booking_info(chat_id: str):
    result = booking_collection.delete_one({"chat_id": chat_id})
    
    if result.deleted_count == 0:
        # No document found with this chat_id
        raise HTTPException(status_code=404, detail="Chat ID not found.")
    
    return {
        "status": "success",
        "message": f"Booking with chat_id '{chat_id}' has been deleted."
    }
