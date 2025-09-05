#!/usr/bin/env python3
"""Test permission override path guarded by PYTEST_CURRENT_TEST."""
from fastapi.testclient import TestClient

from src.unified_ai_api import app


def test_permission_override_header_active_under_pytest(monkeypatch):
    """Test that permission override header works when running under pytest."""
    # Simulate pytest environment for the app
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")

    client = TestClient(app)

    # login to get token
    login_data = {"username": "testuser@example.com", "password": "testpassword123"}
    login_resp = client.post("/auth/login", json=login_data)
    assert login_resp.status_code == 200, f"Login failed: {login_resp.status_code} {login_resp.text}"
    access = login_resp.json().get("access_token")
    assert access and isinstance(access, str), "Missing access_token in login response"

    # Call an endpoint that checks permissions: batch transcription requires 'batch_processing'
    headers = {
        "Authorization": f"Bearer {access}",
        "X-User-Permissions": "batch_processing",
    }
    files = [("audio_files", ("t.wav", b"fake", "audio/wav"))]
    # Endpoint should accept header-based override in pytest
    resp = client.post("/transcribe/batch", files=files, headers=headers)
    assert resp.status_code in (200, 400)


def test_permission_override_header_inactive_without_pytest(monkeypatch):
    """Test that permission override header is ignored when not running under pytest."""
    # Ensure pytest indicator is not set
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("ENABLE_TEST_PERMISSION_INJECTION", "false")

    client = TestClient(app)

    # login to get token
    login_data = {"username": "testuser@example.com", "password": "testpassword123"}
    login_resp = client.post("/auth/login", json=login_data)
    assert login_resp.status_code == 200
    access = login_resp.json().get("access_token")
    assert access and isinstance(access, str)

    # Provide override header but since not under pytest toggle, it should be ignored
    headers = {
        "Authorization": f"Bearer {access}",
        "X-User-Permissions": "batch_processing",
    }
    files = [("audio_files", ("t.wav", b"fake", "audio/wav"))]
    resp = client.post("/transcribe/batch", files=files, headers=headers)
    # Should be forbidden without real permission
    assert resp.status_code == 403
