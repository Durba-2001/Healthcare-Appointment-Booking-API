import json
import redis
from src.utils.config import REDIS_HOST,REDIS_PASSWORD,REDIS_PORT
r = redis.Redis(
    host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD, decode_responses=True
)


# ----------------------------
# Environment Configuration
# ----------------------------
REDIS_TTL = 3600 # default: 1 hour

# ----------------------------
# Helper to extract recommendation from tool output
# ----------------------------
def extract_recommendation(tool_output: str) -> str:
    try:
        parsed = json.loads(tool_output)
        return parsed.get("recommendation", tool_output)
    except Exception:
        return tool_output

# ----------------------------
# Helper to extract session update from tool output
# ----------------------------
def extract_session(tool_output: str) -> str:
    try:
        parsed = json.loads(tool_output)
        return parsed.get("session_update", tool_output)
    except Exception:
        return tool_output

# ----------------------------
# Redis Helpers
# ----------------------------
def store_message_redis(chat_id: str, role: str, message: str, ttl: int = REDIS_TTL):
    """
    Append chat messages to Redis list and set TTL.
    """
    redis_key = f"session:{chat_id}:messages"
    msg_entry = json.dumps({"role": role, "message": message})
    r.rpush(redis_key, msg_entry)

    # Set expiry if not already set
    if r.ttl(redis_key) == -1:
        r.expire(redis_key, ttl)
