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
# 1. Recommend Service
# --------------------------
@mcp.tool()  # register recommend_service as an MCP tool
async def recommend_service(chat_id: str, user_message: str, token: str):  # tool to recommend a service based on user message
    check_auth(token)  # validate token before proceeding
    query_embedding = embedding_model.embed_query(user_message)  # create embedding for the user message
    result = qdrant_client.search(collection_name="healthcare_docs", query_vector=query_embedding, limit=5)  # search Qdrant for related docs
    docs_text_list = [hit.payload.get("text_preview", "") if hit.payload else "" for hit in result]  # extract text previews from hits
    docs_text = "\n".join(docs_text_list)  # join previews into a single context string

    prompt = f"""
You are a healthcare assistant.
A user says: "{user_message}" describing the health issue.
Available services context:
{docs_text}

Recommend the appropriate healthcare service (Cardiologist, Dermatologist, Dentist, Neurologist).
Do not use department. Keep it concise.
Ask the user for their city (e.g., Kolkata, Pune, Bangalore, Delhi).
"""  # compose an LLM prompt providing context and instruction for service recommendation
    response = await llm.ainvoke(prompt)  # call the LLM asynchronously with the prompt
    reply = response.content.strip()  # extract and trim the LLM reply

    session_key = f"session:{chat_id}"  # build Redis session key using chat_id
    r.hset(session_key, mapping={"recommendation": reply, "messages": user_message})  # store recommendation and original message in Redis hash
    logger.info(f" ession key created: {session_key}")  # log creation of session key (note: original message had a typo ' ession')
    logger.info(f"Session data: {r.hgetall(session_key)}")  # log the stored session data for debugging
    r.expire(session_key, SESSION_TTL)  # set TTL on the session key to auto-expire
    logger.info(f"TTL after set: {r.ttl(session_key)}")  # log TTL value for verification

    return {"recommendation": reply, "prompt": "Please mention your city (e.g., Kolkata, Pune, Bangalore, Delhi)."}  # return recommendation and next-step instruction

# --------------------------
# 2. List Professionals
# --------------------------
@mcp.tool()  # register list_professionals as an MCP tool
async def list_professionals(chat_id: str, user_message: str, token: str):  # tool to list professionals for a given service and city
    check_auth(token)  # validate token
    context = r.hgetall(f"session:{chat_id}") or {}  # fetch session context from Redis or default to empty dict
    recommendation = context.get("recommendation", "")  # get previous recommendation from session
    match = re.search(r"(Cardiologist|Dermatologist|Dentist|Neurologist)", recommendation, re.IGNORECASE)  # attempt to extract service type
    service_type = match.group(1) if match else "General Practitioner"  # fallback to General Practitioner if not matched

    valid_cities = ["Kolkata", "Pune", "Bangalore", "Delhi"]  # allowed cities for this flow
    city_match = next((c for c in valid_cities if re.search(fr"\b{c}\b", user_message, re.IGNORECASE)), None)  # detect city in user message
    if not city_match:  # if no city found
        return {"recommendation": f"Could not detect city. Please mention one of {', '.join(valid_cities)}."}  # prompt user to provide valid city
    city = city_match  # set city variable from detected match
    cache_key = f"professionals:{city}:{service_type}"  # build cache key for professionals list
    professionals = None  # initialize variable for professionals list

    if r.exists(cache_key):  # check Redis cache existence for this key
        cached_data = r.get(cache_key)  # retrieve cached JSON string if present
        if cached_data:  # if cached data is not empty
            professionals = json.loads(cached_data)  # parse JSON into Python object

    if not professionals:  # if no cached data found
        cursor = professionals_collection.find(  # create MongoDB cursor to find professionals matching city and type
            {"city": {"$regex": f"^{city}$", "$options": "i"}, "type": {"$regex": f"{service_type}", "$options": "i"}},
            {"_id": 0}
        )
        professionals = await cursor.to_list(length=None)  # convert cursor to list asynchronously
        if professionals:  # if results found
            r.set(cache_key, json.dumps(professionals), ex=PROF_LIST_TTL)  # cache results in Redis with TTL

    if not professionals:  # if still empty after DB lookup
        return {"recommendation": f"No {service_type}s found in {city}."}  # inform user none found

    lines = [f"- {p['name']} ({p.get('certification', 'N/A')}) — Available: {', '.join(p.get('working_days', []))} — Experience: {p.get('years_experience', 'N/A')} years — Rating: {p.get('rating', 'N/A')}" for p in professionals]  # format each professional into a readable line

    return {"recommendation": f"Here are the {service_type}s in {city}:\n{'\n'.join(lines)}\nPlease type the professional's name to continue."}  # return formatted list and next-step instruction

# --------------------------
# 3. Select Professional
# --------------------------
@mcp.tool()  # register select_professional as an MCP tool
async def select_professional(chat_id: str, user_message: str, token: str):  # tool for selecting a specific professional by name
    check_auth(token)  # validate token

    prof = await professionals_collection.find_one(  # query MongoDB for a professional whose name matches the user input
        {"name": {"$regex": f"^{user_message}$", "$options": "i"}},
        {"_id": 0}
    )
    if not prof:  # if no professional found
        return {"status": "not_found", "recommendation": f"No professional named {user_message} found."}  # inform user

    r.hset(f"session:{chat_id}", mapping={  # store selected professional and service type in session hash
        "selected_professional": prof["name"],
        "service_type": prof.get("type", "")
    })
    r.expire(f"session:{chat_id}", SESSION_TTL)  # refresh session TTL

    return {  # return selection confirmation and a session_update payload for downstream services
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
            "updated_at": datetime.now(timezone.utc)
        }
    }

# --------------------------
# 4. Collect User Info
# --------------------------
@mcp.tool()  # register collect_user_info as an MCP tool
async def collect_user_info(chat_id: str, name: str = None, age: int = None, contact: str = None, email: str = None, token: str = None):  # tool to collect and validate user details
    check_auth(token)  # validate token

    missing = []  # list to record missing/invalid fields
    if not name:  # check presence of name
        missing.append("name")  # record missing name
    if age is None or not isinstance(age, int) or age <= 0:  # validate age is positive integer
        missing.append("valid age")  # record invalid age
    if not contact or len(re.sub(r"\D", "", contact)) != 10:  # normalize contact and check length (10 digits)
        missing.append("valid contact")  # record invalid contact
    if not email or not re.match(r"^\w+@\w+\.\w+$", email):  # basic regex check for email format
        missing.append("valid email")  # record invalid email

    # if missing:
    #     return {"status": "incomplete", "recommendation": f"Please provide missing fields: {', '.join(missing)}."}  # commented out: would return incomplete status if missing

    redis_key = f"session:{chat_id}"  # build Redis session key
    r.hset(redis_key, mapping={  # store collected details into session hash
        "name": name,
        "age": age,
        "contact": contact,
        "email": email,
        "status": "details_collected"
    })
    r.expire(redis_key, SESSION_TTL)  # set TTL on session after storing details
    # Instead of saving here, just return data for FastAPI
    return {  # return completion status including session_update payload
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
            "missing_field": missing,
            "status": "details_collected",
            "updated_at": datetime.now(timezone.utc)
        }
    }

# --------------------------
# 5. Confirm User Info
# --------------------------
@mcp.tool()  # register confirm_user_info as an MCP tool
async def confirm_user_info(chat_id: str, user_message: str, token: str = None):  # tool to confirm collected user details
    check_auth(token)  # validate token
    session_cache = r.hgetall(f"session:{chat_id}")  # fetch session values from Redis
    if not session_cache:  # if no session data found
        return {"status": "error", "recommendation": "No user details found. Please provide your details again."}  # prompt re-entry

    msg = user_message.strip().lower()  # normalize user reply for matching
    if msg in ["yes", "y", "confirm", "yeah", "correct", "ok"]:  # affirmative responses
        customer_name = session_cache.get("name", "")  # pull customer name from session
        return {"status": "confirmed", "recommendation": f"Thank you {customer_name}! Let's proceed to schedule your appointment. Please provide your preferred date and time (YYYY-MM-DD at HH:MM)."}  # proceed to next step
    elif msg in ["no", "n", "wrong", "incorrect", "change", "edit"]:  # negative/edit responses
        return {"status": "rejected", "recommendation": "Alright. Please re-enter your details (name, age, contact, email)."}  # ask for re-entry
    else:  # unclear or unexpected responses
        return {"status": "unclear", "recommendation": "Please confirm: are your details correct? (yes/no)"}  # request clear confirmation

# --------------------------
# 6. Check Availability
# --------------------------
@mcp.tool()  # register check_availability as an MCP tool
async def check_availability(chat_id: str, user_message: str, token: str = None):  # tool to check professional availability for requested slot
    check_auth(token)  # validate token
    session_cache = r.hgetall(f"session:{chat_id}") or {}  # load session cache or use empty dict
    selected_name = session_cache.get("selected_professional")  # fetch selected professional name
    service_type = session_cache.get("service_type")  # fetch service type from session
    customer_name = session_cache.get("name")  # fetch customer name
    age = session_cache.get("age")  # fetch age from session
    contact = session_cache.get("contact")  # fetch contact from session
    email = session_cache.get("email")  # fetch email from session

    if not selected_name:  # if no professional has been selected
        return {"status": "unavailable", "recommendation": "No professional selected yet."}  # instruct user to select one

    prof = await professionals_collection.find_one({"name": selected_name}, {"_id": 0})  # fetch professional document by exact name
    if not prof:  # if professional not found in DB
        return {"status": "unavailable", "recommendation": f"{selected_name} not found."}  # inform user

    date_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", user_message)  # extract date in YYYY-MM-DD format from user message
    time_match = re.search(r"\b(\d{2}:\d{2})\b", user_message)  # extract time in HH:MM format from user message
    booking_date = date_match.group(1) if date_match else None  # set booking_date if regex found
    booking_time = time_match.group(1) if time_match else None  # set booking_time if regex found

    if not booking_date or not booking_time:  # if either date or time missing
        return {"status": "unavailable", "recommendation": "Please provide date (YYYY-MM-DD) and time (HH:MM)."}  # prompt user for both

    dt = datetime.strptime(f"{booking_date} {booking_time}", "%Y-%m-%d %H:%M")  # parse combined date and time into datetime
    weekday = dt.strftime("%A")  # determine weekday name from parsed datetime

    if weekday not in prof["working_days"]:  # if selected weekday not in professional working days
        weekdays_map = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]  # ordered list of weekdays
        current_index = weekdays_map.index(weekday)  # find index of requested weekday
        for i in range(1, 8):  # iterate up to next 7 days to find next working day
            next_day_index = (current_index + i) % 7  # compute index of candidate next day
            next_day = weekdays_map[next_day_index]  # weekday name for candidate day
            if next_day in prof["working_days"]:  # if candidate day is in professional working days
                next_date = dt + timedelta(days=i)  # compute calendar date for that next available day
                suggested_date = next_date.strftime("%Y-%m-%d")  # format suggested date as string
                suggested_time = prof.get("default_time", "10:00")  # choose default time or fallback to 10:00
                return {"status": "unavailable", "recommendation": f"{selected_name} not available on {weekday}. Next available slot: {suggested_date} at {suggested_time}."}  # suggest next slot
        return {"status": "unavailable", "recommendation": f"{selected_name} not available any day this week."}  # no working days found in the loop

    conflict = await bookings_collection.find_one({"professional_name": selected_name, "booking_date": booking_date, "booking_time": booking_time})  # check DB for existing booking conflict
    if conflict:  # if a booking already exists for that slot
        return {"status": "unavailable", "recommendation": f"{selected_name} already booked at that time."}  # inform user of conflict

    booking_id = str(uuid4())  # generate a unique booking identifier
    r.hset(f"session:{chat_id}", mapping={"booking_date": booking_date, "booking_time": booking_time, "booking_id": booking_id})  # store booking info in session
    r.expire(f"session:{chat_id}", SESSION_TTL)  # refresh session TTL after storing booking info

    return {"status": "In-progress", "recommendation": f"Here are your booking details :: professional_name: {selected_name}, service_type: {service_type}, customer_name: {customer_name}, age: {age}, contact: {contact}, email: {email}. Are you sure you want to book on {booking_date} at {booking_time}? (yes/no)."}  # present details and ask for confirmation

# --------------------------
# 7. Confirm Booking
# --------------------------
@mcp.tool()  # register confirm_booking as an MCP tool
async def confirm_booking(chat_id: str, user_message: str, token: str = None):  # tool to finalize and save booking
    check_auth(token)  # validate token
    session_cache = r.hgetall(f"session:{chat_id}")  # load session cache from Redis
    if not session_cache:  # if no session data present
        return {"status": "error", "recommendation": "No session found. Please start again."}  # instruct user to restart process

    msg = user_message.strip().lower()  # normalize user confirmation response
    if msg in ["yes", "y", "confirm", "yeah", "correct", "ok"]:  # if user confirms
        booking_id = session_cache.get("booking_id")  # retrieve booking_id from session
        if not booking_id:  # if no booking id found in session
            return {"status": "error", "recommendation": "No booking to confirm. Please provide date and time first."}  # instruct user to provide slot first

        selected_name = session_cache.get("selected_professional")  # pull selected professional name
        service_type = session_cache.get("service_type")  # pull service_type
        customer_name = session_cache.get("name")  # pull customer name
        age = session_cache.get("age")  # pull age
        contact = session_cache.get("contact")  # pull contact
        email = session_cache.get("email")  # pull email
        booking_date = session_cache.get("booking_date")  # pull booking date
        booking_time = session_cache.get("booking_time")  # pull booking time

        # Save booking to MongoDB
        await bookings_collection.insert_one({  # insert booking document into bookings collection
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

        r.hset(f"session:{chat_id}", mapping={"status": "booked"})  # update session status to booked
        r.expire(f"session:{chat_id}", SESSION_TTL)  # refresh TTL for the session key

        return {"status": "confirmed", "recommendation": f"Booking confirmed! Your booking ID is {booking_id}."}  # confirm booking to user

    elif msg in ["no", "n", "wrong", "incorrect", "change", "edit"]:  # if user rejects or wants to change
        return {"status": "rejected", "recommendation": "Please provide your preferred booking date and time (YYYY-MM-DD at HH:MM)."}  # prompt for new date/time

    else:  # unclear confirmation response
        return {"status": "unclear", "recommendation": "Please confirm your booking (yes/no)."}  # ask for clear confirmation

# --------------------------
# Run MCP Server
# --------------------------
if __name__ == "__main__": 
    logger.info("Starting MCP Healthcare Booking Server...")  # log server startup
    mcp.run(transport="sse", host="127.0.0.1", port=8001)  # run the MCP server using SSE transport on localhost:8001