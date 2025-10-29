from fastmcp import FastMCP
from uuid import uuid4
from loguru import logger
from langchain_google_genai import ChatGoogleGenerativeAI
import redis
from pymongo import AsyncMongoClient
from datetime import datetime, timedelta,timezone
import re
from qdrant_client import QdrantClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
import json
from src.utils.config import (
    ACCESS_TOKEN, GOOGLE_API_KEY, MONGODB_URI,
    REDIS_HOST, REDIS_PASSWORD, REDIS_PORT,
    QDRANT_URL, QDRANT_API_KEY
)
from dateutil import parser

# --------------------------
# Initialize Qdrant and Embeddings
# --------------------------
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", api_key=GOOGLE_API_KEY)

# --------------------------
# Initialize MCP
# --------------------------
mcp = FastMCP(name="Healthcare Booking MCP")

# --------------------------
# MongoDB (async)
# --------------------------
mongo_client = AsyncMongoClient(MONGODB_URI)
db = mongo_client["healthcare_app"]
professionals_collection = db["professionals"]
sessions_collection = db["booking_sessions"]
bookings_collection = db["bookings"]

# --------------------------
# Redis
# --------------------------
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD, decode_responses=True)
SESSION_TTL = 3600  # 1 hour
# --------------------------
# Gemini LLM
# --------------------------
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=GOOGLE_API_KEY)

# --------------------------
# Auth Helper
# --------------------------
def check_auth(token: str):
    if token != ACCESS_TOKEN:
        raise ValueError("Unauthorized: Invalid token")

# --------------------------
# 1. Recommend Service
# --------------------------
@mcp.tool()
async def recommend_service(chat_id: str, user_message: str, token: str):
    check_auth(token)
    query_embedding = embedding_model.embed_query(user_message)
    result = qdrant_client.search(collection_name="healthcare_docs", query_vector=query_embedding, limit=5)
    docs_text_list = [hit.payload.get("text_preview", "") if hit.payload else "" for hit in result]
    docs_text = "\n".join(docs_text_list)

    prompt = f"""
You are a healthcare assistant.
A user says: "{user_message}" describing the health issue.
Available services context:
{docs_text}

Recommend the appropriate healthcare service (Cardiologist, Dermatologist, Dentist, Neurologist).
Do not use department. Keep it concise.
Ask the user for their city (e.g., Kolkata, Pune, Bangalore, Delhi).
"""
    response = await llm.ainvoke(prompt)
    reply = response.content.strip()

    session_key = f"session:{chat_id}"
    r.hset(session_key, mapping={"recommendation": reply, "messages": user_message})
    logger.info(f" Session key created: {session_key}")
    logger.info(f"Session data: {r.hgetall(session_key)}")
    r.expire(session_key, SESSION_TTL)
    logger.info(f"TTL after set: {r.ttl(session_key)}")

    return {"recommendation": reply, "prompt": "Please mention your city (e.g., Kolkata, Pune, Bangalore, Delhi)."}

# --------------------------
# 2. List Professionals
# --------------------------
@mcp.tool()
async def list_professionals(chat_id: str, user_message: str, token: str):
    check_auth(token)
    context = r.hgetall(f"session:{chat_id}") or {}
    recommendation = context.get("recommendation", "")
    match = re.search(r"(Cardiologist|Dermatologist|Dentist|Neurologist)", recommendation, re.IGNORECASE)
    service_type = match.group(1) if match else "General Practitioner"

    valid_cities = ["Kolkata", "Pune", "Bangalore", "Delhi"]
    city_match = next((c for c in valid_cities if re.search(fr"\b{c}\b", user_message, re.IGNORECASE)), None)
    if not city_match:
        return {"recommendation": f"Could not detect city. Please mention one of {', '.join(valid_cities)}."}
    city = city_match
    cache_key = f"professionals:{city}:{service_type}"
    professionals = None

    if r.exists(cache_key):
        cached_data = r.get(cache_key)
        if cached_data:
            professionals = json.loads(cached_data)

    if not professionals:
        cursor = professionals_collection.find(
            {"city": {"$regex": f"^{city}$", "$options": "i"}, "type": {"$regex": f"{service_type}", "$options": "i"}},
            {"_id": 0}
        )
        professionals = await cursor.to_list(length=None)

    if not professionals:
        return {"recommendation": f"No {service_type}s found in {city}."}

    lines = [f"- {p['name']} ({p.get('certification', 'N/A')}) — Available: {', '.join(p.get('working_days', []))} — Experience: {p.get('years_experience', 'N/A')} years — Rating: {p.get('rating', 'N/A')}" for p in professionals]

  

    return {"recommendation": f"Here are the {service_type}s in {city}:\n{'\n'.join(lines)}\nPlease type the professional's name to continue."}

# --------------------------
# 3. Select Professional
# --------------------------
@mcp.tool()
async def select_professional(chat_id: str, user_message: str, token: str):
    check_auth(token)

    prof = await professionals_collection.find_one(
        {"name": {"$regex": f"^{user_message}$", "$options": "i"}},
        {"_id": 0}
    )
    if not prof:
        return {"status": "not_found", "recommendation": f"No professional named {user_message} found."}

    r.hset(f"session:{chat_id}", mapping={
        "selected_professional": prof["name"],
        "service_type": prof.get("type", "")
    })
    r.expire(f"session:{chat_id}", SESSION_TTL)

    return {
        "status": "success",
        "recommendation": (
            f"{prof['name']} selected. "
            "Please provide your name, age, contact, and email."
        ),
        "session_update": {
            "chat_id": chat_id,
            "selected_professional": prof["name"],
            "service_type": prof.get("type", ""),
            "status": "in_progress",
            "updated_at":datetime.now(timezone.utc)
        }
    }

# --------------------------
# 4. Collect User Info
# --------------------------
@mcp.tool()
async def collect_user_info(chat_id: str, name: str = None, age: int = None, contact: str = None, email: str = None, token: str = None):
    check_auth(token)
    if isinstance(age, str) and age.isdigit():
        age = int(age)
    missing = []
    if not name:
        missing.append("name")
    if age is None or not isinstance(age, int) or age <= 0:
        missing.append("valid age")
    if not contact or len(re.sub(r"\D", "", contact)) != 10:
        missing.append("valid contact")
    if not email or not re.match(r"^\w+@\w+\.\w+$", email):
        missing.append("valid email")
    redis_key = f"session:{chat_id}"
    r.hset(redis_key, mapping={
        "name": name,
        "age": age,
        "contact": contact,
        "email": email,
        "status": "details_collected"
    })
    r.expire(redis_key, SESSION_TTL)
    # Instead of saving here, just return data for FastAPI
    return {
        "status": "complete",
        "recommendation": f"User info recorded for {name} ({age} yrs, {email}, contact: {contact}). Is this detail correct?",
        "session_update": {
            "chat_id": chat_id,
            "customer_details": {
                "name": name,
                "age": age,
                "contact": contact,
                "email": email
            },
            "missing_field":missing,
            "status": "details_collected",
            "updated_at":datetime.now(timezone.utc)
        }
    }

# --------------------------
# 5. Confirm User Info
# --------------------------
@mcp.tool()
async def confirm_user_info(chat_id: str, user_message: str, token: str = None):
    """
    Confirm or update user details before proceeding to scheduling.
    """
    check_auth(token)
    session_key = f"session:{chat_id}"
    session_cache = r.hgetall(session_key)

    if not session_cache:
        return {
            "status": "error",
            "recommendation": "No user details found. Please provide your details again."
        }

    msg = user_message.strip().lower()

    # ---------------------------
    # User confirms details
    # ---------------------------
    if msg in ["yes", "y", "confirm", "ok", "correct"]:
        customer_name = session_cache.get("name", "User")
        r.hset(session_key, mapping={"status": "confirmed"})
        r.expire(session_key, SESSION_TTL)

        details = "\n".join(f"- {k.capitalize()}: {v}" for k, v in session_cache.items() if k != "status")

        return {
            "status": "confirmed",
            "recommendation": (
                f"Thank you, {customer_name}! Your details have been confirmed:\n"
                f"{details}\n\nLet's proceed to schedule your appointment."
            ),
            "customer_details": session_cache,
            "next_tool": "check_availability"
        }

    # ---------------------------
    # User wants to update details
    # ---------------------------
    if "no" in msg or "change" in msg or "update" in msg:
        updates = {}

        words = msg.split()

        # Extract age
        if "age" in msg:
            for word in words:
                if word.isdigit():
                    updates["age"] = int(word)
                    break

        # Extract name
        if "name" in msg:
            if "is" in words:
                idx = words.index("is") + 1
                if idx < len(words):
                    updates["name"] = words[idx].capitalize()

        # Extract email
        for word in words:
            if "@" in word and "." in word:
                updates["email"] = word
                break

        # Extract contact number
        for word in words:
            if word.isdigit() and len(word) == 10:
                updates["contact"] = word
                break

        # ---------------------------
        # Apply and show updated info
        # ---------------------------
        if updates:
            r.hset(session_key, mapping=updates)
            r.expire(session_key, SESSION_TTL)
            full_info = r.hgetall(session_key)

            details = "\n".join(f"- {k.capitalize()}: {v}" for k, v in full_info.items() if k != "status")
            customer_name = full_info.get("name", "User")

            return {
                "status": "updated",
                "recommendation": (
                    f"Your information has been updated successfully:\n"
                    f"{details}\n\nThank you, {customer_name}! Let's proceed to schedule your appointment."
                ),
                "customer_details": full_info,
                "next_tool": "check_availability"
            }

        return {
            "status": "rejected",
            "recommendation": (
                "I couldn’t detect any changes. "
                "Please specify like: 'No, my age is 30' or 'No, my email is example@gmail.com'."
            )
        }

    # ---------------------------
    # Unclear confirmation
    # ---------------------------
    return {
        "status": "unclear",
        "recommendation": "Please confirm if your details are correct (yes/no)."
    }


# --------------------------
# 6. Check Availability
# --------------------------

@mcp.tool()
async def check_availability(chat_id: str, user_message: str, token: str = None):
    check_auth(token)
    session_cache = r.hgetall(f"session:{chat_id}") or {}
    selected_name = session_cache.get("selected_professional")
    service_type = session_cache.get("service_type")
    customer_name = session_cache.get("name")
    age = session_cache.get("age")
    contact = session_cache.get("contact")
    email = session_cache.get("email")

    if not selected_name:
        return {"status": "unavailable", "recommendation": "No professional selected yet."}

    prof = await professionals_collection.find_one({"name": selected_name}, {"_id": 0})
    if not prof:
        return {"status": "unavailable", "recommendation": f"{selected_name} not found."}

    # Try to parse date and time from user message
    try:
        dt = parser.parse(user_message, fuzzy=True)
        booking_date = dt.strftime("%Y-%m-%d")
        booking_time = dt.strftime("%H:%M")
    except Exception:
        return {"status": "unavailable", "recommendation": "Could not understand the date/time. Please provide something like '21st December 2025 at 15:30'."}

    weekday = dt.strftime("%A")

    if weekday not in prof["working_days"]:
        weekdays_map = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        current_index = weekdays_map.index(weekday)
        for i in range(1, 8):
            next_day_index = (current_index + i) % 7
            next_day = weekdays_map[next_day_index]
            if next_day in prof["working_days"]:
                next_date = dt + timedelta(days=i)
                suggested_date = next_date.strftime("%Y-%m-%d")
                suggested_time = prof.get("default_time", "10:00")
                return {"status": "unavailable", "recommendation": f"{selected_name} not available on {weekday}. Next available slot: {suggested_date} at {suggested_time}."}
        return {"status": "unavailable", "recommendation": f"{selected_name} not available any day this week."}

    conflict = await bookings_collection.find_one({"professional_name": selected_name, "booking_date": booking_date, "booking_time": booking_time})
    if conflict:
        return {"status": "unavailable", "recommendation": f"{selected_name} already booked at that time."}

    booking_id = str(uuid4())
    r.hset(f"session:{chat_id}", mapping={"booking_date": booking_date, "booking_time": booking_time, "booking_id": booking_id})
    r.expire(f"session:{chat_id}", SESSION_TTL)

    return {"status": "In-progress", "recommendation": f"Here are your booking details :: professional_name: {selected_name}, service_type: {service_type}, customer_name: {customer_name}, age: {age}, contact: {contact}, email: {email}. Are you sure you want to book on {booking_date} at {booking_time}? (yes/no)."}

# --------------------------
# 7. Confirm Booking
# --------------------------
@mcp.tool()
async def confirm_booking(chat_id: str, user_message: str, token: str = None):
    check_auth(token)
    session_cache = r.hgetall(f"session:{chat_id}")
    if not session_cache:
        return {"status": "error", "recommendation": "No session found. Please start again."}

    msg = user_message.strip().lower()
    if msg in ["yes", "y", "confirm", "yeah", "correct", "ok"]:
        booking_id = session_cache.get("booking_id")
        if not booking_id:
            return {"status": "error", "recommendation": "No booking to confirm. Please provide date and time first."}

        selected_name = session_cache.get("selected_professional")
        service_type = session_cache.get("service_type")
        customer_name = session_cache.get("name")
        age = session_cache.get("age")
        contact = session_cache.get("contact")
        email = session_cache.get("email")
        booking_date = session_cache.get("booking_date")
        booking_time = session_cache.get("booking_time")

        # Save booking to MongoDB
        await bookings_collection.insert_one({
            "booking_id": booking_id,
            "chat_id": chat_id,
            "professional_name": selected_name,
            "service_type": service_type,
            "customer_name": customer_name,
            "age": age,
            "contact": contact,
            "email": email,
            "booking_date": booking_date,
            "booking_time": booking_time,
            "status": "confirmed"
        })

        r.hset(f"session:{chat_id}", mapping={"status": "booked"})
        r.expire(f"session:{chat_id}", SESSION_TTL)

        return {"status": "confirmed", "recommendation": f"Booking confirmed! Your booking ID is {booking_id}."}

    elif msg in ["no", "n", "wrong", "incorrect", "change", "edit"]:
        return {"status": "rejected", "recommendation": "Please provide your preferred booking date and time (YYYY-MM-DD at HH:MM)."}

    else:
        return {"status": "unclear", "recommendation": "Please confirm your booking (yes/no)."}

# --------------------------
# Run MCP Server
# --------------------------
if __name__ == "__main__":
    logger.info("Starting MCP Healthcare Booking Server...")
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8001)
