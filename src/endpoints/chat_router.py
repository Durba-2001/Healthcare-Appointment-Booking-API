# endpoints/chat_router.py
from fastapi import APIRouter, BackgroundTasks, HTTPException
from src.models.chat_schema import ChatMessage
from uuid import uuid4
from redis import Redis
from pymongo import MongoClient
from loguru import logger
import re
from src.utils.config import ACCESS_TOKEN, REDIS_HOST, REDIS_PORT, REDIS_PASSWORD, MONGODB_URI
from src.tools.mcp_tools import (
    recommend_service,
    list_professionals,
    select_professional,
    collect_user_info,
    check_availability
)
from src.utils.background_task import save_message, update_session_background
from src.utils.helper_func import (
    run_tool,
    convert_tool_response,
    extract_recommendation,
    safe_hset,
    preprocess_user_input
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
# New Chat Endpoint
# --------------------------
@router.post("/new-chat")
async def new_chat(payload: ChatMessage, background_tasks: BackgroundTasks):
    # Endpoint to start a new chat. Receives a ChatMessage payload and supports background tasks
    
    user_message = payload.message.strip()
    # Extract and clean the user's message

    chat_id = str(uuid4())
    # Generate a unique chat ID

    safe_hset(r, f"session:{chat_id}", {"stage": "recommendation"})
    # Store initial stage in Redis

    # Initial Mongo insert as background
    background_tasks.add_task(update_session_background, chat_id, {
        "stage": "recommendation",
        "messages": [{"role": "user", "content": user_message}]
    })
    # Save initial chat session in MongoDB asynchronously

    logger.info(f"💬 User message received: {user_message}")
    # Log the received message

    # Handle empty message case
    if not user_message:
        logger.warning("⚠️ Empty message received, skipping tool execution.")
        return {
            "chat_id": chat_id,
            "response": "⚠️ Please enter a valid message to start the chat."
        }
    # If the message is empty, warn and return early

    logger.info("🆕 New chat created successfully!")
    # Log chat creation success

    try:
        processed_input = preprocess_user_input("recommendation", user_message)
        # Preprocess user input for the recommendation stage

        tool_response = await run_tool(recommend_service, {
            "chat_id": chat_id,
            "user_message": processed_input,
            "token": ACCESS_TOKEN
        })
        # Call the recommendation tool asynchronously

        tool_response_dict = convert_tool_response(tool_response)
        # Convert tool output into a dictionary

        response_text = extract_recommendation(tool_response_dict)
        # Extract readable recommendation text

        next_stage = "awaiting_city"
        # Determine the next stage in the chat flow

        safe_hset(r, f"session:{chat_id}", {
            "stage": next_stage,
            "recommendation": response_text
        })
        # Update Redis with new stage and recommendation

        background_tasks.add_task(update_session_background, chat_id, {
            "stage": next_stage,
            "recommendation": response_text
        })
        # Update MongoDB in the background

        logger.info(f"💾 Session initialized in Redis for chat_id={chat_id}")
        logger.info(f"📨 New chat response returned for chat_id={chat_id}")
        # Log Redis initialization and response sent

    except Exception as e:
        logger.error(f"❌ Tool execution failed: {e}")  
        response_text = "⚠️ Something went wrong. Please try again."
        # Handle errors gracefully

    # Save both messages
    background_tasks.add_task(save_message, chat_id, "user", user_message)
    background_tasks.add_task(save_message, chat_id, "assistant", response_text)
    # Save user and assistant messages asynchronously

    return {"chat_id": chat_id, "response": response_text}
    # Return chat ID and initial recommendation to client

# --------------------------
# Continue Chat Endpoint
# --------------------------
@router.post("/continue-chat/{chat_id}")
async def continue_chat(chat_id: str, payload: ChatMessage, background_tasks: BackgroundTasks):
    # Endpoint to continue an existing chat

    user_message = payload.message.strip()
    # Clean user's message

    session_cache = r.hgetall(f"session:{chat_id}") or {}
    # Retrieve session info from Redis

    session_data = sessions_collection.find_one({"chat_id": chat_id}) or {}
    # Retrieve session info from MongoDB if not in Redis

    stage = session_cache.get("stage") or session_data.get("stage") or "recommendation"
    # Determine current stage of conversation

    response_text = "⚠️ No recommendation available."
    next_stage = stage  # Default to current stage
    # Initialize response and next stage

    try:
        logger.info(f"[Continue Chat] Chat ID: {chat_id}, Stage: {stage}, User: {user_message}")
        processed_input = preprocess_user_input(stage, user_message)
        # Preprocess user input according to current stage

        # ----------------------------
        # Stage: Recommendation
        # ----------------------------
        if stage == "recommendation":
            tool_response = await run_tool(recommend_service, {
                "chat_id": chat_id,
                "user_message": str(processed_input),
                "token": ACCESS_TOKEN
            })
            response_text = extract_recommendation(tool_response)
            next_stage = "awaiting_city"
        # If at recommendation stage, run recommendation tool

        # ----------------------------
        # Stage: Awaiting City
        # ----------------------------
        elif stage == "awaiting_city":
            city_str = processed_input if isinstance(processed_input, str) else ""
            if not city_str.strip():
                response_text = "⚠️ Please enter a valid city."
                next_stage = stage
            else:
                tool_response = await run_tool(list_professionals, {
                    "chat_id": chat_id,
                    "user_message": city_str,
                    "token": ACCESS_TOKEN
                })
                response_text = extract_recommendation(tool_response)
                next_stage = "awaiting_prof_selection"
        # Validate city input and list professionals

        # ----------------------------
        # Stage: Awaiting Professional Selection
        # ----------------------------
        elif stage == "awaiting_prof_selection":
            prof_name = processed_input if isinstance(processed_input, str) else ""
            if not prof_name.strip():
                response_text = "⚠️ Please select a professional from the list."
                next_stage = stage
            else:
                tool_response = await run_tool(select_professional, {
                    "chat_id": chat_id,
                    "user_message": prof_name,
                    "token": ACCESS_TOKEN
                })
                response_text = extract_recommendation(tool_response)
                next_stage = "awaiting_user_info"
        # Validate professional selection

        # ----------------------------
        # Stage: Awaiting User Info
        # ----------------------------
        elif stage == "awaiting_user_info":
            user_info = processed_input if isinstance(processed_input, dict) else {}
            # Ensure input is a dictionary

            # Validate mandatory fields
            email = user_info.get("email")
            name = user_info.get("name")
            if not name:
                response_text = "⚠️ Please enter your name."
                next_stage = stage
            elif not email or not re.match(r"\w+@\w+\.\w+", email):
                response_text = "⚠️ Please enter a valid email address."
                next_stage = stage
            else:
                # Valid input → call tool
                tool_args = {"chat_id": chat_id, **user_info, "token": ACCESS_TOKEN}
                tool_response = await run_tool(collect_user_info, tool_args)
                response_text = extract_recommendation(tool_response)
                next_stage = "awaiting_availability"
        # Validate user info and call collection tool

        # ----------------------------
        # Stage: Awaiting Availability
        # ----------------------------
        elif stage == "awaiting_availability":
            booking_info = processed_input if isinstance(processed_input, dict) else {}
            booking_date = booking_info.get("booking_date")
            booking_time = booking_info.get("booking_time")

            if not booking_date or not booking_time:
                response_text = "⚠️ Please provide both booking date and time."
                next_stage = stage
            else:
                date_time_str = f"{booking_date} {booking_time}".strip()
                tool_response = await run_tool(check_availability, {
                    "chat_id": chat_id,
                    "user_message": date_time_str,
                    "token": ACCESS_TOKEN
                })
                structured = getattr(tool_response, "structured_content", None)

                if structured:
                    status = structured.get("status")
                    response_text = structured.get("recommendation")
                else:
                    status = None
                    response_text = str(tool_response)

                logger.info(f"[Continue Chat] Booking status: {status}")

                if status == "confirmed":
                    next_stage = "complete"
                    r.delete(f"session:{chat_id}")
                    background_tasks.add_task(update_session_background, chat_id, {
                        "stage": next_stage,
                        "status": status
                    })
                    background_tasks.add_task(save_message, chat_id, "user", user_message)
                    background_tasks.add_task(save_message, chat_id, "assistant", response_text)
                    return {"chat_id": chat_id, "response": response_text, "status": status}
                else:
                    next_stage = stage  # Stay on this stage if booking not confirmed
        # Handle booking availability and confirmation

        # ----------------------------
        # Stage: Complete / Fallback
        # ----------------------------
        else:
            response_text = "Conversation complete. Thank you!"
            next_stage = "complete"
            r.delete(f"session:{chat_id}")
        # If conversation is complete, clean up session

        # -----------------------------
        # Stage-wise logging
        # -----------------------------
        logger.info(
            f"[Continue Chat] Chat ID: {chat_id}, Stage: {stage} -> {next_stage}, "
            f"User: {user_message}, Response: {response_text}"
        )
        # Log the transition and response

        # Update Redis + Mongo in background if not complete
        if next_stage != "complete":
            safe_hset(r, f"session:{chat_id}", {"stage": next_stage})
            background_tasks.add_task(update_session_background, chat_id, {"stage": next_stage})
        # Save the updated stage

    except Exception as e:
        logger.exception(f"[Continue Chat] Error chat_id={chat_id}, stage={stage}: {e}")
        response_text = "⚠️ Something went wrong. Please try again."
        # Handle errors gracefully

    # Save messages asynchronously
    background_tasks.add_task(save_message, chat_id, "user", user_message)
    background_tasks.add_task(save_message, chat_id, "assistant", response_text)
    # Save messages in the background

    return {"chat_id": chat_id, "response": response_text, "status": "in_progress"}
    # Return response with in-progress status

# --------------------------
# Get Booking Info by Chat ID
# --------------------------
@router.get("/booking-info/{chat_id}")
async def get_booking_info(chat_id: str):
    # Endpoint to retrieve booking details for a given chat ID

    logger.info(f"[GET Booking Info] Chat ID: {chat_id}")
    
    booking_data = booking_collection.find_one({"chat_id": chat_id})
    # Query MongoDB for booking info

    if not booking_data:
        logger.error(f"[GET Booking Info] Chat ID not found: {chat_id}")
        return {"status": "error", "message": "Chat ID not found."}
    # Return error if not found

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
    # Structure booking info for response

    logger.info(f"[GET Booking Info] Retrieved booking for Chat ID: {chat_id}")
    return {"status": "success", "chat_id": chat_id, "booking_info": booking_info}
    # Return structured booking data

# --------------------------
# Delete Booking Info by Chat ID
# --------------------------
@router.delete("/booking-info/{chat_id}")
async def delete_booking_info(chat_id: str):
    # Endpoint to delete booking for a given chat ID
    
    logger.info(f"[DELETE Booking Info] Chat ID: {chat_id}")
    
    result = booking_collection.delete_one({"chat_id": chat_id})
    # Delete the booking from MongoDB

    if result.deleted_count == 0:
        logger.error(f"[DELETE Booking Info] Chat ID not found: {chat_id}")
        raise HTTPException(status_code=404, detail="Chat ID not found.")
    # Raise 404 if no document was deleted

    logger.info(f"[DELETE Booking Info] Successfully deleted booking for Chat ID: {chat_id}")
    return {"status": "success", "message": f"Booking with chat_id '{chat_id}' has been deleted."}
    # Return success confirmation