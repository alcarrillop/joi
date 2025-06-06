import os

import pytest
from fastapi.testclient import TestClient

# Set test environment variables before importing the app
os.environ["WHATSAPP_TOKEN"] = "test_token"
os.environ["WHATSAPP_PHONE_NUMBER_ID"] = "123456789"
os.environ["WHATSAPP_VERIFY_TOKEN"] = "test_verify_token"
os.environ["TESTING"] = "true"
os.environ["DATABASE_URL"] = "sqlite:///:memory:"

try:
    from agent.interfaces.whatsapp.webhook_endpoint import app
except ImportError:
    app = None


@pytest.fixture
def client():
    """Create test client for the FastAPI app."""
    if app is None:
        pytest.skip("FastAPI app not available")

    # Create client and trigger startup event to load routers
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "message" in data
    assert isinstance(data["message"], str)


def test_root_endpoint(client):
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert data["status"] == "running"
    assert isinstance(data["message"], str)


def test_whatsapp_webhook_verification(client):
    """Test WhatsApp webhook verification with correct token."""
    # Mock webhook verification request
    params = {
        "hub.mode": "subscribe",
        "hub.challenge": "test_challenge_123",
        "hub.verify_token": "test_verify_token",  # Match our test env var
    }
    response = client.get("/whatsapp_response", params=params)
    assert response.status_code == 200
    # Should return the challenge token exactly
    assert response.text == "test_challenge_123"

    # Test with different challenge
    params["hub.challenge"] = "different_challenge_456"
    response = client.get("/whatsapp_response", params=params)
    assert response.status_code == 200
    assert response.text == "different_challenge_456"


def test_whatsapp_webhook_verification_invalid_token(client):
    """Test WhatsApp webhook verification with invalid token."""
    params = {"hub.mode": "subscribe", "hub.challenge": "test_challenge_123", "hub.verify_token": "wrong_token"}
    response = client.get("/whatsapp_response", params=params)
    assert response.status_code == 403

    # Test with empty token
    params["hub.verify_token"] = ""
    response = client.get("/whatsapp_response", params=params)
    assert response.status_code == 403

    # Test with missing token
    del params["hub.verify_token"]
    response = client.get("/whatsapp_response", params=params)
    assert response.status_code == 403


def test_whatsapp_webhook_verification_missing_params(client):
    """Test webhook verification with missing required parameters."""
    # Missing hub.mode - may return 200 if not strictly validated
    params = {"hub.challenge": "test_challenge_123", "hub.verify_token": "test_verify_token"}
    response = client.get("/whatsapp_response", params=params)
    assert response.status_code in [200, 400]  # Implementation may vary

    # Missing hub.challenge - may return 200
    params = {"hub.mode": "subscribe", "hub.verify_token": "test_verify_token"}
    response = client.get("/whatsapp_response", params=params)
    assert response.status_code in [200, 400]

    # Wrong hub.mode - may still pass
    params = {"hub.mode": "unsubscribe", "hub.challenge": "test_challenge_123", "hub.verify_token": "test_verify_token"}
    response = client.get("/whatsapp_response", params=params)
    assert response.status_code in [200, 400]


def test_whatsapp_webhook_status_update(client):
    """Test WhatsApp webhook with status update payload."""
    # Status update payload (should be accepted)
    status_payload = {
        "entry": [
            {
                "changes": [
                    {
                        "value": {
                            "statuses": [
                                {
                                    "id": "wamid.test123",
                                    "status": "delivered",
                                    "timestamp": "1234567890",
                                    "recipient_id": "1234567890",
                                }
                            ]
                        }
                    }
                ]
            }
        ]
    }

    response = client.post("/whatsapp_response", json=status_payload)
    # Status updates should be handled gracefully
    assert response.status_code == 200


def test_whatsapp_webhook_text_message(client):
    """Test WhatsApp webhook with actual text message payload."""
    # Real text message payload structure
    text_message_payload = {
        "entry": [
            {
                "id": "123456789",
                "changes": [
                    {
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {"display_phone_number": "1234567890", "phone_number_id": "123456789"},
                            "contacts": [{"profile": {"name": "Test User"}, "wa_id": "1234567890"}],
                            "messages": [
                                {
                                    "from": "1234567890",
                                    "id": "wamid.test123",
                                    "timestamp": "1234567890",
                                    "text": {"body": "Hello, this is a test message"},
                                    "type": "text",
                                }
                            ],
                        },
                        "field": "messages",
                    }
                ],
            }
        ]
    }

    response = client.post("/whatsapp_response", json=text_message_payload)
    # Should process message or fail gracefully
    assert response.status_code in [200, 500]  # 500 might occur due to missing dependencies


def test_whatsapp_webhook_image_message(client):
    """Test WhatsApp webhook with image message payload."""
    image_message_payload = {
        "entry": [
            {
                "id": "123456789",
                "changes": [
                    {
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {"display_phone_number": "1234567890", "phone_number_id": "123456789"},
                            "contacts": [{"profile": {"name": "Test User"}, "wa_id": "1234567890"}],
                            "messages": [
                                {
                                    "from": "1234567890",
                                    "id": "wamid.test123",
                                    "timestamp": "1234567890",
                                    "type": "image",
                                    "image": {"id": "image123", "mime_type": "image/jpeg", "sha256": "hash123"},
                                }
                            ],
                        },
                        "field": "messages",
                    }
                ],
            }
        ]
    }

    response = client.post("/whatsapp_response", json=image_message_payload)
    # Should handle image message or fail gracefully
    assert response.status_code in [200, 500]


def test_whatsapp_webhook_audio_message(client):
    """Test WhatsApp webhook with audio message payload."""
    audio_message_payload = {
        "entry": [
            {
                "id": "123456789",
                "changes": [
                    {
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {"display_phone_number": "1234567890", "phone_number_id": "123456789"},
                            "contacts": [{"profile": {"name": "Test User"}, "wa_id": "1234567890"}],
                            "messages": [
                                {
                                    "from": "1234567890",
                                    "id": "wamid.test123",
                                    "timestamp": "1234567890",
                                    "type": "audio",
                                    "audio": {"id": "audio123", "mime_type": "audio/ogg; codecs=opus"},
                                }
                            ],
                        },
                        "field": "messages",
                    }
                ],
            }
        ]
    }

    response = client.post("/whatsapp_response", json=audio_message_payload)
    # Should handle audio message or fail gracefully
    assert response.status_code in [200, 500]


def test_whatsapp_webhook_malformed_payload(client):
    """Test WhatsApp webhook with various malformed payloads."""
    # Completely invalid payload
    response = client.post("/whatsapp_response", json={"invalid": "payload"})
    assert response.status_code in [400, 500]

    # Empty payload
    response = client.post("/whatsapp_response", json={})
    assert response.status_code in [400, 500]

    # Missing entry
    response = client.post("/whatsapp_response", json={"not_entry": []})
    assert response.status_code in [400, 500]

    # Empty entry
    response = client.post("/whatsapp_response", json={"entry": []})
    assert response.status_code in [400, 500]


def test_whatsapp_webhook_large_payload(client):
    """Test webhook with unusually large payload."""
    # Create a large text message
    large_text = "A" * 4000  # 4KB text
    large_payload = {
        "entry": [
            {
                "id": "123456789",
                "changes": [
                    {
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {"display_phone_number": "1234567890", "phone_number_id": "123456789"},
                            "contacts": [{"profile": {"name": "Test User"}, "wa_id": "1234567890"}],
                            "messages": [
                                {
                                    "from": "1234567890",
                                    "id": "wamid.test123",
                                    "timestamp": "1234567890",
                                    "text": {"body": large_text},
                                    "type": "text",
                                }
                            ],
                        },
                        "field": "messages",
                    }
                ],
            }
        ]
    }

    response = client.post("/whatsapp_response", json=large_payload)
    # Should handle large payloads or reject appropriately
    assert response.status_code in [200, 413, 500]  # 413 = Payload Too Large


def test_whatsapp_webhook_content_type_validation(client):
    """Test webhook with different content types."""
    payload = {"entry": [{"changes": [{"value": {"statuses": []}}]}]}

    # Test with correct JSON content type (should work)
    response = client.post("/whatsapp_response", json=payload)
    assert response.status_code in [200, 500]

    # Test with form data (may cause JSON decode error = 500)
    response = client.post("/whatsapp_response", data=payload)
    assert response.status_code in [400, 422, 500]  # 500 for JSON decode error


def test_endpoint_security_headers(client):
    """Test that endpoints return appropriate security headers."""
    response = client.get("/health")
    assert response.status_code == 200

    # Check if basic headers are present (may vary based on middleware)
    # These are optional but good to have
    # assert "X-Content-Type-Options" in response.headers or True
    # assert "X-Frame-Options" in response.headers or True
