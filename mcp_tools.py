from fastmcp import FastMCP
from uuid import uuid4
from loguru import logger
from langchain_google_genai import ChatGoogleGenerativeAI
import redis
from config import ACCESS_TOKEN, GOOGLE_API_KEY, MONGODB_URI, REDIS_HOST, REDIS_PASSWORD, REDIS_PORT
from pymongo import AsyncMongoClient  # ✅ Use Motor for async Mongo
from datetime import datetime,timezone
import re

# --------------------------
# Initialize MCP
# --------------------------
mcp = FastMCP(name="Healthcare Booking MCP")

# MongoDB (async)
mongo_client = AsyncMongoClient(MONGODB_URI)
db = mongo_client["healthcare_app"]
professionals_collection = db["professionals"]
sessions_collection = db["booking_sessions"]
bookings_collection = db["bookings"]

# Redis
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD, decode_responses=True)
SESSION_TTL = 3600  # 1 hour
PROF_LIST_TTL = 1800  # 30 mins

# Gemini LLM
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
    """Suggest healthcare service and ask city using Gemini"""
    check_auth(token)

    prompt = f"""
    You are a healthcare assistant. 
    User says: "{user_message}".
    Suggest a relevant healthcare professional type 
    (Cardiologist, Dermatologist, Dentist, Neurologist)
    and ask for their city (Kolkata, Pune, Bangalore, Delhi).
    """

    response = await llm.ainvoke(prompt)
    reply = response.content.strip()

    r.hset(f"session:{chat_id}", mapping={"recommendation": reply, "messages": user_message})
    r.expire(f"session:{chat_id}", SESSION_TTL)
    await sessions_collection.update_one({"chat_id": chat_id}, {"$set": {"recommendation": reply}}, upsert=True)

    return {"recommendation": reply, "prompt": "Please select your city (Kolkata, Pune, Bangalore, Delhi)."}

# --------------------------
# 2. Location-Based Professional Listing Tool
# --------------------------
@mcp.tool()
async def list_professionals(chat_id: str, city: str, token: str):
    """List healthcare professionals available in a city"""
    check_auth(token)

    context = r.hgetall(f"session:{chat_id}") or {}
    recommendation = context.get("recommendation", "General Practitioner")

    # Extract service type from recommendation
    match = re.search(r"(Cardiologist|Dermatologist|Dentist|Neurologist)", recommendation, re.IGNORECASE)
    service_type = match.group(1) if match else "General Practitioner"

    valid_cities = ["Kolkata", "Pune", "Bangalore", "Delhi"]
    if city not in valid_cities:
        return {"recommendation": f"Invalid city. Please choose from {', '.join(valid_cities)}."}

    cache_key = f"professionals:{city}:{service_type}"
    if r.exists(cache_key):
        professionals = eval(r.get(cache_key))
        logger.info(f"Using cached list for {city} ({service_type})")
    else:
        cursor = professionals_collection.find(
            {"city": {"$regex": f"^{city}$", "$options": "i"},
             "profession": {"$regex": f"{service_type}", "$options": "i"}},
            {"_id": 0}
        )
        professionals = await cursor.to_list(length=None)
        r.set(cache_key, str(professionals), ex=PROF_LIST_TTL)

    if not professionals:
        return {"recommendation": f"No {service_type}s found in {city}."}

    formatted = "\n".join(
        f"- {p['name']} ({p['certification']}) — Available: {', '.join(p['availability'])}"
        for p in professionals
    )

    await sessions_collection.update_one({"chat_id": chat_id}, {"$set": {"professionals": professionals}}, upsert=True)

    return {
        "recommendation": f"Here are the {service_type}s in {city}:\n{formatted}\n\nPlease type the professional's name to continue."
    }

# --------------------------
# 3. Professional Selection Tool
# --------------------------
@mcp.tool()
async def select_professional(chat_id: str, name: str, token: str):
    """Select a professional and store it in session"""
    check_auth(token)
    session = await sessions_collection.find_one({"chat_id": chat_id}, {"professionals": 1})
    if not session or not session.get("professionals"):
        return {"recommendation": "No professionals available. Please list them first."}

    professionals = session["professionals"]
    selected = next((p for p in professionals if p["name"].lower() == name.lower()), None)
    if not selected:
        return {"recommendation": f"No professional named {name} found."}

    await sessions_collection.update_one({"chat_id": chat_id}, {"$set": {"selected_professional": selected}}, upsert=True)
    r.hset(f"session:{chat_id}", mapping={"selected_professional": selected["name"]})
    r.expire(f"session:{chat_id}", SESSION_TTL)

    return {
        "recommendation": f"✅ {selected['name']} selected ({selected['profession']}). Please provide your name, age, contact number, and email."
    }

# --------------------------
# 4. Collect User Info Tool
# --------------------------
@mcp.tool()
async def collect_user_info(chat_id: str, name: str = None, age: int = None, contact: str = None, email: str = None, token: str = None):
    """Collect and validate user details, automatically use stored data if missing"""
    check_auth(token)

    # Pull stored session data
    session_cache = r.hgetall(f"session:{chat_id}") or {}

    # Use stored values if any field is None
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

    # Save to MongoDB and Redis
    await sessions_collection.update_one({"chat_id": chat_id},
        {"$set": {"customer_name": name, "age": age, "contact": contact, "email": email}}, upsert=True)
    r.hset(f"session:{chat_id}", mapping={"customer_name": name, "age": age, "contact": contact, "email": email})
    r.expire(f"session:{chat_id}", 3600)

    return {"status": "complete", "recommendation": f"User info recorded for {name} ({age} yrs, {email}). Proceed to check availability."}

# --------------------------
# 5. Availability Checker Tool
# --------------------------
@mcp.tool()
async def check_availability(chat_id: str, booking_date: str = None, booking_time: str = None, token: str = None):
    """Check if selected professional is available, suggest next available slot if not"""
    check_auth(token)

    # Pull session data
    session_cache = r.hgetall(f"session:{chat_id}") or {}
    selected_name = session_cache.get("selected_professional")
    customer_name = session_cache.get("customer_name")
    age = session_cache.get("age")
    contact = session_cache.get("contact")
    email = session_cache.get("email")

    if not selected_name:
        return {"status": "unavailable", "recommendation": "No professional selected yet."}

    prof = await professionals_collection.find_one({"name": selected_name}, {"_id": 0})
    if not prof:
        return {"status": "unavailable", "recommendation": f"{selected_name} not found."}

    # Ensure booking info is provided
    if not booking_date or not booking_time:
        return {"status": "unavailable", "recommendation": "Please provide date (YYYY-MM-DD) and time (HH:MM)."}

    dt = datetime.strptime(f"{booking_date} {booking_time}", "%Y-%m-%d %H:%M")
    weekday = dt.strftime("%A")

    # Check availability
    if weekday not in prof["availability"]:
        # Suggest next available day
        from datetime import timedelta

        weekdays_map = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        current_index = weekdays_map.index(weekday)
        for i in range(1, 8):
            next_day_index = (current_index + i) % 7
            next_day = weekdays_map[next_day_index]
            if next_day in prof["availability"]:
                next_date = dt + timedelta(days=i)
                suggested_date = next_date.strftime("%Y-%m-%d")
                suggested_time = prof.get("default_time", "10:00")
                return {
                    "status": "unavailable",
                    "recommendation": f"{selected_name} not available on {weekday}. "
                                      f"Next available slot: {suggested_date} at {suggested_time}."
                }
        return {"status": "unavailable", "recommendation": f"{selected_name} not available any day this week."}

    # Check for conflict
    conflict = await bookings_collection.find_one({
        "professional_name": selected_name,
        "booking_date": booking_date,
        "booking_time": booking_time
    })
    if conflict:
        return {"status": "unavailable", "recommendation": f"{selected_name} already booked at that time."}

    # Save booking
    booking_id = str(uuid4())
    await bookings_collection.insert_one({
        "booking_id": booking_id,
        "chat_id": chat_id,
        "professional_name": selected_name,
        "customer_name": customer_name,
        "age": age,
        "contact": contact,
        "email": email,
        "booking_date": booking_date,
        "booking_time": booking_time,
        "status": "confirmed"
    })

    r.hset(f"session:{chat_id}", mapping={"status": "complete", "booking_id": booking_id})
    r.expire(f"session:{chat_id}", 3600)

    return {"status": "confirmed", "recommendation": f"✅ Booking confirmed with {selected_name} on {booking_date} at {booking_time}."}
