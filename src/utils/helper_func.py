from loguru import logger
import re
from fastapi.concurrency import run_in_threadpool
import json
from datetime import datetime
from fastmcp.tools.tool import TextContent  # Import TextContent from fastMCP tools

# --------------------------
# Tool Execution Helpers
# --------------------------

async def run_tool(tool, params):
    """Execute a tool safely, handling async or sync methods."""
    try:
        # Check if tool has a callable 'run' method
        if callable(getattr(tool, "run", None)):
            # If 'run' has a __code__ attribute, it's an async function
            if getattr(tool.run, "__code__", None):
                return await tool.run(params)  # Run async tool
            else:
                # Run synchronous tool in a threadpool to avoid blocking the event loop
                return await run_in_threadpool(tool.run, params)
        # Log error if tool has no 'run' method
        logger.error(f"Tool {tool} has no run() method")
        return None
    except Exception as e:
        # Log any exception that occurs during tool execution
        logger.exception(f"Tool execution failed: {tool}, params={params}, error={e}")
        return None


def convert_tool_response(tool_response):
    """Convert any tool response to a dictionary safely."""
    
    if isinstance(tool_response, dict):
        # Already a dictionary, return as is
        return tool_response
    if isinstance(tool_response, list):
        # If list, convert each element to text
        content_list = []

        for t in tool_response:
            if isinstance(t, TextContent):
                text_value = t.text
            else:
                text_value = str(t)
    
            content_list.append({"text": text_value})

        return {"content": content_list}

    if isinstance(tool_response, TextContent):
        # If single TextContent, wrap in a dictionary
        return {"content": [{"text": tool_response.text}]}
    if hasattr(tool_response, "__dict__"):
        # If object has __dict__, convert to dict
        return tool_response.__dict__
    # Fallback: return raw string
    return {"raw": str(tool_response)}


def extract_recommendation(tool_response):
    """Universal extractor to get a recommendation or message from any tool response."""
    try:
        if not tool_response:
            # If response is None or empty
            return "⚠️ No recommendation available."
        tool_response = convert_tool_response(tool_response)  # Normalize response to dict

        structured = tool_response.get("structured_content", {})  # Get structured content if available
        for key in ["recommendation", "prompt"]:
            if key in structured and structured[key]:
                return structured[key]  # Return recommendation or prompt from structured content

        # Extract from content list if present
        content_list = tool_response.get("content", [])
        if isinstance(content_list, list):
            for item in content_list:
                text = item.get("text")
                if text:
                    try:
                        # Try parsing JSON string
                        parsed = json.loads(text)
                        if "recommendation" in parsed:
                            return parsed["recommendation"]
                        if "prompt" in parsed:
                            return parsed["prompt"]
                    except json.JSONDecodeError:
                        # If not JSON, return text as is
                        return text

        # Fallback: return 'message' key if present
        if "message" in tool_response:
            return tool_response["message"]

        # Fallback: return first string value found in dictionary
        for k, v in tool_response.items():
            if isinstance(v, str):
                return v

        # Final fallback if nothing found
        return "⚠️ No recommendation available."
    except Exception as e:
        # Log any exception during extraction
        logger.error(f"Failed to extract recommendation: {e}, tool_response={tool_response}")
        return "⚠️ No recommendation available."


# --------------------------
# User Info / Booking Parsing
# --------------------------

def extract_user_info_from_text(text: str):
    """Extract user details like name, age, contact number, and email from free-form text."""
    info = {}  # Initialize empty dictionary to hold extracted info

    # Extract name patterns like "my name is X" or "I am X"
    name_match = re.search(r"(?:my name is|i am)\s+([A-Za-z]+)", text, re.IGNORECASE)
    if name_match:
        info["name"] = name_match.group(1).capitalize()  # Capitalize first letter of name

    # Extract age patterns like "26 years", "26yo", "age 26"
    age_match = re.search(r"\b(\d{1,3})\s*(?:years|yo|y/o)?\b", text.lower())
    if age_match:
        info["age"] = int(age_match.group(1))  # Convert matched age to integer

    # Extract 10-digit contact number
    contact_match = re.search(r"\b\d{10}\b", text)
    if contact_match:
        info["contact"] = contact_match.group(0)

    # Extract email addresses with a simple pattern
    email_match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text)
    if email_match:
        info["email"] = email_match.group(0)

    return info  # Return extracted info as dictionary


def extract_booking_info_from_text(text: str):
    """Extract booking date and time from user text."""
    booking = {"booking_date": None, "booking_time": None}  # Initialize result dictionary
    text = text.lower().replace(",", " ").replace(".", " ")  # Normalize text
    words = text.split()  # Split into words for easier parsing

    # Extract date after "on"
    if "on" in words:
        try:
            idx = words.index("on") + 1
            date_words = []
            while idx < len(words) and words[idx] != "at":
                date_words.append(words[idx])  # Collect words until 'at' (time)
                idx += 1
            if len(date_words) >= 2:
                day = int(date_words[0])
                month = date_words[-1].capitalize()  # Month as capitalized string
                year = datetime.now().year
                # Convert to ISO date string
                booking["booking_date"] = datetime.strptime(f"{day} {month} {year}", "%d %b %Y").strftime("%Y-%m-%d")
        except:
            pass  # Ignore parsing errors

    # Extract time after "at"
    if "at" in words:
        try:
            idx = words.index("at") + 1
            if idx < len(words):
                booking["booking_time"] = words[idx]  # Expecting HH:MM format
        except:
            pass  # Ignore parsing errors

    return booking  # Return booking info dictionary


# --------------------------
# Safe Redis Write Helper
# --------------------------

def safe_hset(redis_conn, key, mapping):
    """Safely store dictionary in Redis, ignoring None values."""
    if not mapping:
        return  # Nothing to store

    clean = {}  # Dictionary to store only non-None values

    # Loop through mapping and add only valid values
    for k, v in mapping.items():
        if v is not None:
            clean[k] = v

    # If there is something to store, write to Redis
    if clean:
        redis_conn.hset(key, mapping=clean)


# --------------------------
# Stage-aware User Input Preprocessing
# --------------------------

def preprocess_user_input(stage: str, user_message: str):
    """Process user input depending on conversation stage."""
    if stage == "awaiting_user_info":
        # Extract user personal info
        return extract_user_info_from_text(user_message)
    elif stage == "awaiting_availability":
        # Extract booking date/time info
        return extract_booking_info_from_text(user_message)
    else:
        # Default: return cleaned text
        return user_message.strip()