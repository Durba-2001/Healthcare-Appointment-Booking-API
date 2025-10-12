# main.py
from fastapi import FastAPI
from src.endpoints.chat_router import router as chat_router
from src.endpoints.admin_router import router as admin_router



app = FastAPI(
    title="Healthcare Appointment Booking API",
    version="1.0",
    description="AI-driven appointment booking using MCP tools."
)

# Routers

app.include_router(admin_router, prefix="/admin", tags=["Admin"])
app.include_router(chat_router, prefix="/chat", tags=["Chat"])


