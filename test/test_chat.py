import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
from src.endpoints.chat_router import router
from src.models.chat_schema import ChatMessage
from fastapi import FastAPI

# Create test FastAPI app
app = FastAPI()
app.include_router(router)

client = TestClient(app)


@pytest.fixture
def mock_chat_message():
    return {"message": "I need a dentist appointment"}


# --------------------------
# /new-chat endpoint tests
# --------------------------
@patch("src.endpoints.chat_router.run_tool", new_callable=AsyncMock)
@patch("src.endpoints.chat_router.extract_recommendation", return_value="Dentist recommended in your area.")
def test_new_chat_success(mock_extract, mock_tool, mock_chat_message):
    mock_tool.return_value = {"dummy": "tool_response"}
    response = client.post("/new-chat", json=mock_chat_message)
    assert response.status_code == 200
    data = response.json()
    assert "chat_id" in data
    assert data["response"] == "Dentist recommended in your area."


@patch("src.endpoints.chat_router.run_tool", side_effect=Exception("Tool Error"))
def test_new_chat_tool_error(mock_tool, mock_chat_message):
    response = client.post("/new-chat", json=mock_chat_message)
    assert response.status_code == 200
    assert "⚠️ Something went wrong" in response.json()["response"]


@patch("src.endpoints.chat_router.run_tool", new_callable=AsyncMock)
def test_new_chat_empty_message(mock_tool):
    payload = {"message": "   "}
    response = client.post("/new-chat", json=payload)
    assert response.status_code == 200
    assert "chat_id" in response.json()

