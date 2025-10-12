from loguru import logger
import re
from fastapi.concurrency import run_in_threadpool
import json
from datetime import datetime


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
    if isinstance(tool_response, list):
        return {"content": [{"text": t.text if isinstance(t, TextContent) else str(t)} for t in tool_response]}
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

        structured = tool_response.get("structured_content", {})
        for key in ["recommendation", "prompt"]:
            if key in structured and structured[key]:
                return structured[key]

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

        if "message" in tool_response:
            return tool_response["message"]

        for k, v in tool_response.items():
            if isinstance(v, str):
                return v

        return "⚠️ No recommendation available."
    except Exception as e:
        logger.error(f"Failed to extract recommendation: {e}, tool_response={tool_response}")
        return "⚠️ No recommendation available."

# --------------------------
# User Info / Booking Parsing 
# --------------------------
import re

def extract_user_info_from_text(text: str):
    """Extract name, age, contact, and email from text."""
    info = {}

    # Name: "my name is X" or "I am X"
    name_match = re.search(r"(?:my name is|i am)\s+([A-Za-z]+)", text, re.IGNORECASE)
    if name_match:
        info["name"] = name_match.group(1).capitalize()

    # Age: "26 years", "26yo", "age 26"
    age_match = re.search(r"\b(\d{1,3})\s*(?:years|yo|y/o)?\b", text.lower())
    if age_match:
        info["age"] = int(age_match.group(1))

    # Contact: 10-digit number
    contact_match = re.search(r"\b\d{10}\b", text)
    if contact_match:
        info["contact"] = contact_match.group(0)

    # Email: simple pattern
    email_match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text)
    if email_match:
        info["email"] = email_match.group(0)

    return info


def extract_booking_info_from_text(text: str):
    """Extract booking date/time from normal text."""
    booking = {"booking_date": None, "booking_time": None}
    text = text.lower().replace(",", " ").replace(".", " ")
    words = text.split()

    # Date
    if "on" in words:
        try:
            idx = words.index("on") + 1
            date_words = []
            while idx < len(words) and words[idx] != "at":
                date_words.append(words[idx])
                idx += 1
            if len(date_words) >= 2:
                day = int(date_words[0])
                month = date_words[-1].capitalize()
                year = datetime.now().year
                booking["booking_date"] = datetime.strptime(f"{day} {month} {year}", "%d %b %Y").strftime("%Y-%m-%d")
        except:
            pass

    # Time
    if "at" in words:
        try:
            idx = words.index("at") + 1
            if idx < len(words):
                booking["booking_time"] = words[idx]  # HH:MM expected
        except:
            pass

    return booking