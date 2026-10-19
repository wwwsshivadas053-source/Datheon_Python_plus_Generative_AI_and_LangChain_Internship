"""Unit and integration tests for Flask web REST API."""

import pytest
from web.app import app


@pytest.fixture
def client():
    """Create Flask test client."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_index_route(client):
    """Verify home page loads successfully."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"STRUCTURED PROMPT PIPELINE" in response.data


def test_get_samples(client):
    """Verify preloaded samples endpoint returns invoice and resume data."""
    response = client.get("/api/samples")
    assert response.status_code == 200
    data = response.get_json()
    assert "invoice" in data
    assert "resume" in data
    assert len(data["invoice"]) >= 3
    assert len(data["resume"]) >= 3


def test_get_schema(client):
    """Verify schema retrieval endpoint."""
    response = client.get("/api/schema/invoice")
    assert response.status_code == 200
    data = response.get_json()
    assert data["domain"] == "invoice"
    assert "properties" in data["schema"]


def test_prompt_preview(client):
    """Verify live prompt preview generation endpoint."""
    payload = {
        "domain": "invoice",
        "text": "Invoice #101 Total $50",
        "enable_few_shot": True
    }
    response = client.post("/api/prompt_preview", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert "system_prompt" in data
    assert "user_prompt" in data
    assert "Invoice #101" in data["user_prompt"]


def test_extract_invoice_mock(client):
    """Verify extraction endpoint for invoice domain."""
    payload = {
        "domain": "invoice",
        "text": "INVOICE #INV-2024-9042 from OmniTech Solutions Inc. Total $6504.",
        "provider": "mock"
    }
    response = client.post("/api/extract", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert data["domain"] == "invoice"
    assert data["validated_data"]["metadata"]["invoice_number"] == "INV-2024-9042"


def test_extract_resume_mock(client):
    """Verify extraction endpoint for resume domain."""
    payload = {
        "domain": "resume",
        "text": "Elena Rostova - Senior AI Engineer with 6 years experience in PyTorch and LangChain.",
        "provider": "mock"
    }
    response = client.post("/api/extract", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert data["domain"] == "resume"
    assert "Elena Rostova" in data["validated_data"]["personal_info"]["full_name"]


def test_extract_self_healing_flow(client):
    """Verify simulated validation failure and self-correction reflection via API."""
    payload = {
        "domain": "invoice",
        "text": "Invoice #INV-2024-9042 Total $6504.",
        "provider": "mock",
        "simulate_fail_once": True,
        "max_retries": 2
    }
    response = client.post("/api/extract", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert data["attempts"] == 2
    assert len(data["audit_trail"]) == 2
