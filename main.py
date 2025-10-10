# main.py
from fastapi import FastAPI
from chat_router import router as chat_router
from admin_router import router as admin_router
from mcp_tools import mcp
import asyncio

app = FastAPI(
    title="Healthcare Appointment Booking API",
    version="1.0",
    description="AI-driven appointment booking using MCP tools."
)

# Routers
app.include_router(chat_router, prefix="/chat", tags=["Chat"])
app.include_router(admin_router, prefix="/admin", tags=["Admin"])

# # Run MCP in background
# @app.on_event("startup")
# async def start_mcp():
#     asyncio.create_task(mcp.run(transport="sse", host="127.0.0.1", port=8001))
