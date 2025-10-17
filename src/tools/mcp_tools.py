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
from qdrant_client import QdrantClient,models
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.utils.config import QDRANT_URL, QDRANT_API_KEY
import json
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
    # Check token validity
    check_auth(token)

    # -------------------------
    #  Embed user query
    # -------------------------
    # Convert user message to vector embedding
    query_embedding = embedding_model.embed_query(user_message)

    # -------------------------
    #  Retrieve top-k relevant PDFs from Qdrant
    # -------------------------
    # Search top 5 similar vectors in Qdrant collection
    result = qdrant_client.search(
        collection_name="healthcare_docs",
        query_vector=query_embedding,
        limit=5
    )

    # Collect text from retrieved documents
    docs_text_list = []

    # Iterate over Qdrant search results
    for hit in result:
        if hit.payload:  # If metadata exists
            text_preview = hit.payload.get("text_preview", "")
            docs_text_list.append(text_preview)
        else:  # Empty placeholder if no payload
            docs_text_list.append("")

    # Join all previews into a single string
    docs_text = "\n".join(docs_text_list)

    # -------------------------
    # 3️⃣ Generate recommendation using Gemini LLM
    # -------------------------
    # Create prompt to send to LLM
    prompt = f"""
You are a healthcare assistant.
A user says: "{user_message}" describing the health issue they are facing.
You have access to descriptions of available healthcare services (uploaded by admin as PDFs) as context below:
{docs_text}

Based on this context, recommend the most appropriate healthcare service for the user like Cardiologist, Dermatologist, Dentist, Neurologist.
dont use department.
Keep your response concise and clear.

After recommending the service type, ask the user for their city like Kolkata, Pune, Bangalore, Delhi so that suitable professionals can be suggested.
"""

    # Call LLM asynchronously to generate response
    response = await llm.ainvoke(prompt)

    # Get the actual text response
    reply = response.content.strip()

    # -------------------------
    # 4️⃣ Save recommendation to Redis + Mongo
    # -------------------------
    session_key = f"session:{chat_id}"  # Redis key for this chat
    r.hset(session_key, mapping={"recommendation": reply, "messages": user_message})  # Save data
    r.expire(session_key, SESSION_TTL)  # Set expiry

    # Save recommendation to MongoDB session collection
    await sessions_collection.update_one(
        {"chat_id": chat_id}, {"$set": {"recommendation": reply}}, upsert=True
    )

    # Return recommendation and next prompt
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
    # Validate auth
    check_auth(token)

    # Get session data from Redis
    context = r.hgetall(f"session:{chat_id}") or {}
    recommendation = context.get("recommendation", "")

    # Extract service type from recommendation using regex
    match = re.search(r"(Cardiologist|Dermatologist|Dentist|Neurologist)", recommendation, re.IGNORECASE)
    service_type = match.group(1) if match else "General Practitioner"
    logger.info(f"Detected service type: {service_type}")

    # Detect city in user message
    valid_cities = ["Kolkata", "Pune", "Bangalore", "Delhi"]
    city_match = None
    for c in valid_cities:
        if re.search(fr"\b{c}\b", user_message, re.IGNORECASE):
            city_match = c
            break


    # Return error if city not found
    if not city_match:
        return {
            "recommendation": f"Could not detect city. Please mention one of {', '.join(valid_cities)}."
        }

    city = city_match
    cache_key = f"professionals:{city}:{service_type}"  # Redis cache key
    professionals = None  # Initialize variable

    # --- Try cache first ---
    if r.exists(cache_key):
        cached_data = r.get(cache_key)
        if cached_data:
            professionals = json.loads(cached_data)  
            logger.info(f"Using cached list for {city} ({service_type})")

    # --- If cache miss, query MongoDB ---
    if not professionals:
        logger.info(f"Cache miss — fetching {service_type}s from MongoDB for {city}")
        cursor = professionals_collection.find(
            {
                "city": {"$regex": f"^{city}$", "$options": "i"},  # Case-insensitive match
                "type": {"$regex": f"{service_type}", "$options": "i"},
            },
            {"_id": 0}  # Exclude MongoDB _id field
        )
        professionals = await cursor.to_list(length=None)  # Convert cursor to list

        # Cache the result if professionals found
        if professionals:
            r.set(cache_key, json.dumps(professionals), ex=PROF_LIST_TTL)
            logger.info(f"Cached professionals for {city} ({service_type})")

    # --- Handle no results ---
    if not professionals:
        return {"recommendation": f"No {service_type}s found in {city}."}

    # --- Format output nicely ---
    lines = []
    for p in professionals:
      line = (
        f"- {p['name']} ({p.get('certification', 'N/A')}) — "
        f"Available: {', '.join(p.get('working_days', []))} — "
        f"Experience: {p.get('years_experience', 'N/A')} years — "
        f"Rating: {p.get('rating', 'N/A')}"
    )
      lines.append(line)

    formatted = "\n".join(lines)


    # Save professionals list in MongoDB session
    await sessions_collection.update_one(
        {"chat_id": chat_id},
        {"$set": {"professionals": professionals}},
        upsert=True,
    )

    # Return formatted recommendation
    return {
        "recommendation": (
            f"Here are the {service_type}s in {city}:\n"
            f"{formatted}\n\n"
            f"Please type the professional's name to continue."
        )
    }

# --------------------------
# 3. Select Professional Tool
# --------------------------
@mcp.tool()
async def select_professional(chat_id: str, user_message: str, token: str):
    """Select a professional using free text input and store service type"""
    check_auth(token)

    # Retrieve professionals from session
    session = await sessions_collection.find_one({"chat_id": chat_id}, {"professionals": 1})
    if not session or not session.get("professionals"):
        return {"recommendation": "No professionals available. Please list them first."}

    professionals = session["professionals"]

    # Initialize selected professional
    selected = None

    # Search for professional's name in user message
    for p in professionals:
        if re.search(fr"\b{p['name']}\b", user_message, re.IGNORECASE):
            selected = p
            break  # Stop after first match

    if not selected:
        return {"recommendation": "Could not detect professional's name. Please type the exact name from the list."}

    # Get service type from selected professional
    service_type = selected.get("type")

    # Save selection in MongoDB
    await sessions_collection.update_one(
        {"chat_id": chat_id},
        {"$set": {"selected_professional": selected, "service_type": service_type}},
        upsert=True
    )

    # Save selection in Redis
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

    # Get cached session
    session_cache = r.hgetall(f"session:{chat_id}") or {}

    # Use provided or cached data
    name = name or session_cache.get("customer_name")
    age = age or session_cache.get("age")
    contact = contact or session_cache.get("contact")
    email = email or session_cache.get("email")

    # Track missing fields
    missing = []
    if not name: missing.append("name")
    if not age: missing.append("age")
    if not email or not re.match(r"^\w+@\w+\.\w+$", email): missing.append("valid email")
    if not contact or len(re.sub(r"\D", "", contact)) != 10: missing.append("valid contact number")

    # Prompt user if any fields missing
    if missing:
        return {"status": "incomplete", "recommendation": f"Please provide missing fields: {', '.join(missing)}."}

    # Save info in MongoDB
    await sessions_collection.update_one(
        {"chat_id": chat_id},
        {"$set": {"customer_name": name, "age": age, "contact": contact, "email": email}},
        upsert=True
    )

    # Save info in Redis
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

    # Get session data from Redis
    session_cache = r.hgetall(f"session:{chat_id}") or {}
    selected_name = session_cache.get("selected_professional")
    service_type = session_cache.get("service_type")
    customer_name = session_cache.get("customer_name")
    age = session_cache.get("age")
    contact = session_cache.get("contact")
    email = session_cache.get("email")

    if not selected_name:
        return {"status": "unavailable", "recommendation": "No professional selected yet."}

    # Fetch professional data from MongoDB
    prof = await professionals_collection.find_one({"name": selected_name}, {"_id": 0})
    if not prof:
        return {"status": "unavailable", "recommendation": f"{selected_name} not found."}

    # Extract booking date and time from user message
    date_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", user_message)
    time_match = re.search(r"\b(\d{2}:\d{2})\b", user_message)
    booking_date = date_match.group(1) if date_match else None
    booking_time = time_match.group(1) if time_match else None

    if not booking_date or not booking_time:
        return {"status": "unavailable", "recommendation": "Please provide date (YYYY-MM-DD) and time (HH:MM)."}

    dt = datetime.strptime(f"{booking_date} {booking_time}", "%Y-%m-%d %H:%M")
    weekday = dt.strftime("%A")  # Get day of week

    # Check professional working days
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

    # Check booking conflicts
    conflict = await bookings_collection.find_one({
        "professional_name": selected_name,
        "booking_date": booking_date,
        "booking_time": booking_time
    })
    if conflict:
        return {"status": "unavailable", "recommendation": f"{selected_name} already booked at that time."}

    # Save booking with UUID
    booking_id = str(uuid4())
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

    # Update session in Redis
    r.hset(f"session:{chat_id}", mapping={"status": "complete", "booking_id": booking_id})
    r.expire(f"session:{chat_id}", SESSION_TTL)

    return {
        "status": "confirmed",
        "recommendation": f"✅ Booking confirmed with {selected_name} ({service_type}) on {booking_date} at {booking_time}."
    }