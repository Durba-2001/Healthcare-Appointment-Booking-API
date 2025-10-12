from pydantic import BaseModel
# --------------------------
# Pydantic Models
# --------------------------
class ChatMessage(BaseModel):
    message: str