import pytest
from fastapi.testclient import TestClient
from src.main import app, create_prompt, generate_prompt
from unittest.mock import patch, MagicMock

client = TestClient(app)


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"tests": "test3"}


def test_create_prompt():
    conversation = ["Hello", "Hi there"]
    new_question = "How are you?"
    role = "Friendly AI"

    result = create_prompt(conversation, new_question, role)

    assert "system_message" in result
    assert "new_question" in result
    assert "previous_log" in result
    assert role in result["system_message"]
    assert new_question == result["new_question"]
    assert "\n".join(conversation) == result["previous_log"]


def test_generate_prompt():
    prompt_dict = {
        "system_message": "You are an AI assistant",
        "new_question": "What's the weather like?",
        "previous_log": "User: Hello\nAI: Hi there",
    }

    result = generate_prompt(prompt_dict)

    assert prompt_dict["system_message"] in result
    assert prompt_dict["new_question"] in result
    assert prompt_dict["previous_log"] in result


@patch("src.main.ChatOpenAI")
def test_query_api(mock_chat_openai):
    mock_response = MagicMock()
    mock_response.content = "This is a tests response"
    mock_chat_openai.return_value.invoke.return_value = mock_response

    test_query = {
        "userId": "test_user",
        "role": "Test Role",
        "question": "Test question",
        "chatHistory": ["Previous message"],
    }

    response = client.post("/query", json=test_query)

    assert response.status_code == 200
    assert response.json() == {"message": "This is a tests response"}


@patch("src.main.ChatOpenAI")
def test_query_api_exception(mock_chat_openai):
    mock_chat_openai.return_value.invoke.side_effect = Exception("Test exception")

    test_query = {
        "userId": "test_user",
        "role": "Test Role",
        "question": "Test question",
        "chatHistory": ["Previous message"],
    }

    response = client.post("/query", json=test_query)

    assert response.status_code == 500
    assert "detail" in response.json()
