# test/test_chat.py
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
from src.endpoints.chat_router import router
from fastapi import FastAPI
from loguru import logger  # Use the real logger
import json
app = FastAPI()
app.include_router(router)
client = TestClient(app)


# --------------------------
# /new-chat endpoint tests
# --------------------------

@patch("src.endpoints.chat_router.run_tool", new_callable=AsyncMock)
@patch("src.endpoints.chat_router.extract_recommendation", return_value="Dentist recommended in your area.")
def test_new_chat_success(mock_extract, mock_tool):
    payload = {"message": "I need a dentist appointment"}

    mock_tool.return_value = {"dummy": "tool_response"}

    response = client.post("/new-chat", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Basic response checks
    assert "chat_id" in data
    assert data["response"] == "Dentist recommended in your area."


# --------------------------
# Test: Tool Error scenario
# --------------------------
@patch("src.endpoints.chat_router.run_tool", side_effect=Exception("Tool Error"))
def test_new_chat_tool_error(mock_tool):
    payload = {"message": "I need a dentist appointment"}
    response = client.post("/new-chat", json=payload)

    assert response.status_code == 200
    assert "⚠️ Something went wrong" in response.json()["response"]
    # ✅ loguru will automatically print the exception
    # No need to mock logger.error

# --------------------------
# Test: Empty message scenario
# --------------------------
@patch("src.endpoints.chat_router.run_tool", new_callable=AsyncMock)
def test_new_chat_empty_message(mock_tool):
    payload = {"message": "   "}
    response = client.post("/new-chat", json=payload)

    assert response.status_code == 200
    assert "chat_id" in response.json()

# --------------------------
# Stage-wise /continue-chat tests (fixed)
# --------------------------

chat_id = "test-stage-chat"

# Mock responses for each stage
stage_tool_responses = {
    "recommendation": {"recommendation": "Cardiologist recommended!", "status": None},
    "awaiting_city": {"recommendation": "Cities listed: Pune, Delhi", "status": None},
    "awaiting_prof_selection": {"recommendation": "Dr. Meera Joshi selected", "status": None},
    "awaiting_user_info": {"recommendation": "User info collected", "status": None},
    "awaiting_availability": {"recommendation": "Booking confirmed!", "status": "confirmed"}
}

@patch("src.endpoints.chat_router.r")
@patch("src.endpoints.chat_router.sessions_collection")
@patch("src.endpoints.chat_router.save_message", new_callable=AsyncMock)
@patch("src.endpoints.chat_router.update_session_background", new_callable=AsyncMock)
@patch("src.endpoints.chat_router.run_tool", new_callable=AsyncMock)
def test_continue_chat_full_flow(mock_run_tool, mock_update, mock_save, mock_sessions, mock_redis):
    """
    Test full stage-wise continue-chat flow with proper stringified messages
    """
    # Mock Redis and Mongo
    mock_redis.hgetall.return_value = {}
    mock_sessions.find_one.return_value = {}

    # Payloads for each stage (dicts are JSON-dumped)
    user_inputs = {
        "recommendation": "I need a cardiologist",
        "awaiting_city": "Pune",
        "awaiting_prof_selection": "Dr. Meera Joshi",
        "awaiting_user_info": json.dumps({
            "name": "John Doe",
            "age": 35,
            "contact": "1234567890",
            "email": "john@example.com"
        }),
        "awaiting_availability": json.dumps({
            "booking_date": "2025-10-20",
            "booking_time": "10:00"
        })
    }

    stage_order = ["recommendation", "awaiting_city", "awaiting_prof_selection", "awaiting_user_info", "awaiting_availability"]

    for stage in stage_order:
        # Mock run_tool to return stage-specific response
        tool_response = AsyncMock()
        tool_response.structured_content = stage_tool_responses[stage]
        mock_run_tool.return_value = tool_response

        # ✅ define payload inside the loop
        payload = {"message": user_inputs[stage]}

        # Call the continue-chat endpoint
        response = client.post(f"/continue-chat/{chat_id}", json=payload)
        assert response.status_code == 200

        data = response.json()

        # Check stage-specific recommendation in response
        assert stage_tool_responses[stage]["recommendation"] in data["response"]



# --------------------------
# GET /booking-info tests
# --------------------------
booking_mock = {
    "chat_id": chat_id,
    "professional_name": "Dr. Meera Joshi",
    "service_type": "Cardiology",
    "customer_name": "John Doe",
    "age": 35,
    "contact": "1234567890",
    "email": "john@example.com",
    "booking_date": "2025-10-20",
    "booking_time": "10:00",
    "booking_id": "booking-001",
    "status": "confirmed"
}

@patch("src.endpoints.chat_router.booking_collection")
def test_get_booking_info_success(mock_booking):
    mock_booking.find_one.return_value = booking_mock

    response = client.get(f"/booking-info/{chat_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["chat_id"] == chat_id
    assert data["booking_info"]["professional_name"] == "Dr. Meera Joshi"

@patch("src.endpoints.chat_router.booking_collection")
def test_get_booking_info_not_found(mock_booking):
    mock_booking.find_one.return_value = None

    response = client.get(f"/booking-info/{chat_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "error"
    assert "not found" in data["message"]

# --------------------------
# DELETE /booking-info tests
# --------------------------
@patch("src.endpoints.chat_router.booking_collection")
def test_delete_booking_info_success(mock_booking):
    mock_booking.delete_one.return_value.deleted_count = 1

    response = client.delete(f"/booking-info/{chat_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert chat_id in data["message"]

@patch("src.endpoints.chat_router.booking_collection")
def test_delete_booking_info_not_found(mock_booking):
    mock_booking.delete_one.return_value.deleted_count = 0

    response = client.delete(f"/booking-info/{chat_id}")
    assert response.status_code == 404
    data = response.json()
    assert "not found" in data["detail"]
