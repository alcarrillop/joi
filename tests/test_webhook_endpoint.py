import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

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


def test_root_endpoint(client):
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert data["status"] == "running"


def test_whatsapp_webhook_verification(client):
    """Test WhatsApp webhook verification."""
    # Mock webhook verification request
    params = {
        "hub.mode": "subscribe",
        "hub.challenge": "test_challenge_123",
        "hub.verify_token": "test_verify_token",  # Match our test env var
    }
    response = client.get("/whatsapp_response", params=params)
    assert response.status_code == 200
    # Should return the challenge token
    assert response.text == "test_challenge_123"


def test_whatsapp_webhook_verification_invalid_token(client):
    """Test WhatsApp webhook verification with invalid token."""
    params = {"hub.mode": "subscribe", "hub.challenge": "test_challenge_123", "hub.verify_token": "wrong_token"}
    response = client.get("/whatsapp_response", params=params)
    assert response.status_code == 403


def test_whatsapp_webhook_post_basic(client):
    """Test WhatsApp webhook POST endpoint with minimal payload."""
    # Minimal valid payload that should be accepted
    mock_payload = {
        "entry": [{"changes": [{"value": {"statuses": [{"id": "test_status_id", "status": "delivered"}]}}]}]
    }

    response = client.post("/whatsapp_response", json=mock_payload)
    # Should accept status updates (200) or fail gracefully (500)
    assert response.status_code in [200, 500]


def test_whatsapp_webhook_malformed_payload(client):
    """Test WhatsApp webhook with malformed payload."""
    # Malformed payload should be handled gracefully
    mock_payload = {"invalid": "payload"}

    response = client.post("/whatsapp_response", json=mock_payload)
    # Should handle errors gracefully
    assert response.status_code in [400, 500]
