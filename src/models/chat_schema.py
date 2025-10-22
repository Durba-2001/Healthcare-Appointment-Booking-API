from pydantic import BaseModel,Field
# --------------------------
# Pydantic Models
# --------------------------
class ChatMessage(BaseModel):
    message: str = Field(...,min_length=1)
    