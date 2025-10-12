# mcp_tools.py
from fastmcp import FastMCP
from uuid import uuid4
from loguru import logger
from langchain_google_genai import ChatGoogleGenerativeAI
import redis
from src.utils.config import ACCESS_TOKEN, GOOGLE_API_KEY, MONGODB_URI, REDIS_HOST, REDIS_PASSWORD, REDIS_PORT
from pymongo import AsyncMongoClient
from datetime import datetime, timedelta, timezone
import re
from qdrant_client import QdrantClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.utils.config import QDRANT_URL, QDRANT_API_KEY

# Initialize Qdrant and Embeddings (singleton at module level)
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
PROF_LIST_TTL = 1800  # 30 mins

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
# 1. Recommendation Tool
# --------------------------


@mcp.tool()
async def recommend_service(chat_id: str, user_message: str, token: str):
    """Suggest healthcare service based on PDF context using RAG + Gemini"""
    check_auth(token)

    # -------------------------
    # 1️⃣ Embed user query
    # -------------------------
    query_embedding = embedding_model.embed_query(user_message)

    # -------------------------
    # 2️⃣ Retrieve top-k relevant PDFs from Qdrant
    # -------------------------
    result = qdrant_client.search(
        collection_name="healthcare_docs",
        query_vector=query_embedding,
        limit=5
    )

    docs_text_list = []

    for hit in result:
        if hit.payload:
            text_preview = hit.payload.get("text_preview", "")
            docs_text_list.append(text_preview)
        else:
            docs_text_list.append("")

    docs_text = "\n".join(docs_text_list)


    # -------------------------
    # 3️⃣ Generate recommendation using Gemini LLM
    # -------------------------
    prompt = f"""
You are a healthcare assistant.
A user says: "{user_message}" describing the health issue they are facing.
You have access to descriptions of available healthcare services (uploaded by admin as PDFs) as context below:
{docs_text}

Based on this context, recommend the most appropriate healthcare service for the user 
(e.g., Cardiologist, Dermatologist, Dentist, Neurologist). Keep your response concise and clear.

After recommending the service type, ask the user for their city 
(e.g., Kolkata, Pune, Bangalore, Delhi) so that suitable professionals can be suggested.
"""

    response = await llm.ainvoke(prompt)
    reply = response.content.strip()

    # -------------------------
    # 4️⃣ Save recommendation to Redis + Mongo
    # -------------------------
    session_key = f"session:{chat_id}"
    r.hset(session_key, mapping={"recommendation": reply, "messages": user_message})
    r.expire(session_key, SESSION_TTL)

    await sessions_collection.update_one(
        {"chat_id": chat_id}, {"$set": {"recommendation": reply}}, upsert=True
    )

    return {
        "recommendation": reply,
        "prompt": "Please mention your city (e.g., Kolkata, Pune, Bangalore, Delhi)."
    }

# --------------------------
# 2. List Professionals Tool
# --------------------------
@mcp.tool()
async def list_professionals(chat_id: str, user_message: str, token: str):
    """List professionals by city and service type from free text"""
    check_auth(token)

    context = r.hgetall(f"session:{chat_id}") or {}
    recommendation = context.get("recommendation", "General Practitioner")

    # Extract service type from recommendation
    match = re.search(r"(Cardiologist|Dermatologist|Dentist|Neurologist)", recommendation, re.IGNORECASE)
    service_type = match.group(1) if match else "General Practitioner"

    # Extract city from user message
    valid_cities = ["Kolkata", "Pune", "Bangalore", "Delhi"]
    city_match = next((c for c in valid_cities if re.search(fr"\b{c}\b", user_message, re.IGNORECASE)), None)
    if not city_match:
        return {"recommendation": f"Could not detect city. Please mention one of {', '.join(valid_cities)}."}
    city = city_match

    # Check Redis cache
    cache_key = f"professionals:{city}:{service_type}"
    if r.exists(cache_key):
        professionals = eval(r.get(cache_key))
        logger.info(f"Using cached list for {city} ({service_type})")
    else:
        cursor = professionals_collection.find(
            {"city": {"$regex": f"^{city}$", "$options": "i"},
             "type": {"$regex": f"{service_type}", "$options": "i"}},
            {"_id": 0}
        )
        professionals = await cursor.to_list(length=None)
        r.set(cache_key, str(professionals), ex=PROF_LIST_TTL)

    if not professionals:
        return {"recommendation": f"No {service_type}s found in {city}."}

    formatted = "\n".join(
        f"- {p['name']} ({p['certification']}) — Available: {', '.join(p['working_days'])} - Year of experience: {p['years_experience']} - Rating: {p['rating']}"
        for p in professionals
    )

    await sessions_collection.update_one(
        {"chat_id": chat_id}, {"$set": {"professionals": professionals}}, upsert=True
    )
    
    return {
        "recommendation": f"Here are the {service_type}s in {city}:\n{formatted}\n\nPlease type the professional's name to continue."
    }

# --------------------------
# 3. Select Professional Tool
# --------------------------
@mcp.tool()
async def select_professional(chat_id: str, user_message: str, token: str):
    """Select a professional using free text input and store service type"""
    check_auth(token)
    
    session = await sessions_collection.find_one({"chat_id": chat_id}, {"professionals": 1})
    if not session or not session.get("professionals"):
        return {"recommendation": "No professionals available. Please list them first."}

    professionals = session["professionals"]

    # Extract professional name from message
    selected = next((p for p in professionals if re.search(fr"\b{p['name']}\b", user_message, re.IGNORECASE)), None)
    if not selected:
        return {"recommendation": "Could not detect professional's name. Please type the exact name from the list."}

    # Extract service type
    service_type = selected.get("type", "General Practitioner")

    # Save selected professional and service type in MongoDB
    await sessions_collection.update_one(
        {"chat_id": chat_id},
        {"$set": {"selected_professional": selected, "service_type": service_type}},
        upsert=True
    )

    # Save in Redis
    r.hset(
        f"session:{chat_id}",
        mapping={"selected_professional": selected["name"], "service_type": service_type}
    )
    r.expire(f"session:{chat_id}", SESSION_TTL)

    return {
        "recommendation": f"✅ {selected['name']} ({service_type}) selected. "
                          f"Please provide your name, age, contact number, and email."
    }

# --------------------------
# 4. Collect User Info Tool
# --------------------------
@mcp.tool()
async def collect_user_info(chat_id: str, name: str = None, age: int = None, contact: str = None, email: str = None, token: str = None):
    """Collect and validate user details"""
    check_auth(token)
    session_cache = r.hgetall(f"session:{chat_id}") or {}

    name = name or session_cache.get("customer_name")
    age = age or session_cache.get("age")
    contact = contact or session_cache.get("contact")
    email = email or session_cache.get("email")

    missing = []
    if not name: missing.append("name")
    if not age: missing.append("age")
    if not email or not re.match(r"^[\w\.-]+@[\w\.-]+\.\w+$", email): missing.append("valid email")
    if not contact or len(re.sub(r"\D", "", contact)) < 8: missing.append("valid contact number")

    if missing:
        return {"status": "incomplete", "recommendation": f"Please provide missing fields: {', '.join(missing)}."}

    await sessions_collection.update_one(
        {"chat_id": chat_id},
        {"$set": {"customer_name": name, "age": age, "contact": contact, "email": email}}, upsert=True
    )
    r.hset(f"session:{chat_id}", mapping={"customer_name": name, "age": age, "contact": contact, "email": email})
    r.expire(f"session:{chat_id}", SESSION_TTL)

    return {"status": "complete", "recommendation": f"User info recorded for {name} ({age} yrs, {email}). Proceed to check availability."}

# --------------------------
# 5. Check Availability Tool
# --------------------------
@mcp.tool()
async def check_availability(chat_id: str, user_message: str, token: str = None):
    """Check professional availability and create booking with service type from session"""
    check_auth(token)

    # Get session info from Redis
    session_cache = r.hgetall(f"session:{chat_id}") or {}
    selected_name = session_cache.get("selected_professional")
    service_type = session_cache.get("service_type", "General Practitioner")
    customer_name = session_cache.get("customer_name")
    age = session_cache.get("age")
    contact = session_cache.get("contact")
    email = session_cache.get("email")

    if not selected_name:
        return {"status": "unavailable", "recommendation": "No professional selected yet."}

    prof = await professionals_collection.find_one({"name": selected_name}, {"_id": 0})
    if not prof:
        return {"status": "unavailable", "recommendation": f"{selected_name} not found."}

    # Extract date and time from user message
    date_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", user_message)
    time_match = re.search(r"\b(\d{2}:\d{2})\b", user_message)
    booking_date = date_match.group(1) if date_match else None
    booking_time = time_match.group(1) if time_match else None

    if not booking_date or not booking_time:
        return {"status": "unavailable", "recommendation": "Please provide date (YYYY-MM-DD) and time (HH:MM)."}

    dt = datetime.strptime(f"{booking_date} {booking_time}", "%Y-%m-%d %H:%M")
    weekday = dt.strftime("%A")

    # Check professional availability
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
                return {
                    "status": "unavailable",
                    "recommendation": f"{selected_name} not available on {weekday}. "
                                      f"Next available slot: {suggested_date} at {suggested_time}."
                }
        return {"status": "unavailable", "recommendation": f"{selected_name} not available any day this week."}

    # Check for booking conflicts
    conflict = await bookings_collection.find_one({
        "professional_name": selected_name,
        "booking_date": booking_date,
        "booking_time": booking_time
    })
    if conflict:
        return {"status": "unavailable", "recommendation": f"{selected_name} already booked at that time."}

    # Save booking with service type
    booking_id = str(uuid4())
    await bookings_collection.insert_one({
        "booking_id": booking_id,
        "chat_id": chat_id,
        "professional_name": selected_name,
        "service_type": service_type,  # <-- use service_type from Redis
        "customer_name": customer_name,
        "age": age,
        "contact": contact,
        "email": email,
        "booking_date": booking_date,
        "booking_time": booking_time,
        "status": "confirmed"
    })

    # Update session
    r.hset(f"session:{chat_id}", mapping={"status": "complete", "booking_id": booking_id})
    r.expire(f"session:{chat_id}", SESSION_TTL)

    return {
        "status": "confirmed",
        "recommendation": f"✅ Booking confirmed with {selected_name} ({service_type}) on {booking_date} at {booking_time}."
    }
