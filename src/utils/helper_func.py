import json
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


# ----------------------------
# Helper to extract recommendation from tool output
# ----------------------------
def extract_recommendation(tool_output: str) -> str:
    try:
        parsed = json.loads(tool_output)
        return parsed.get("recommendation", tool_output)
    except Exception:
        return tool_output

def extract_session(tool_output: str) -> str:
    try:
        parsed = json.loads(tool_output)
        return parsed.get("session_update", tool_output)
    except Exception:
        return tool_output