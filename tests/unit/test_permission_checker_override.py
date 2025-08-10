#!/usr/bin/env python3
"""Test permission override path guarded by PYTEST_CURRENT_TEST."""

import os
from fastapi.testclient import TestClient

from src.unified_ai_api import app


def test_permission_override_header_active_under_pytest(monkeypatch):
    # Simulate pytest environment for the app
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")

    client = TestClient(app)

    # login to get token
    login_data = {"username": "testuser@example.com", "password": "testpassword123"}
    login_resp = client.post("/auth/login", json=login_data)
    access = login_resp.json()["access_token"]

    # Call an endpoint that checks permissions: batch transcription requires 'batch_processing'
    headers = {
        "Authorization": f"Bearer {access}",
        "X-User-Permissions": "batch_processing",
    }
    files = [("audio_files", ("t.wav", b"fake", "audio/wav"))]
    # Endpoint should accept header-based override in pytest
    resp = client.post("/transcribe/batch", files=files, headers=headers)
    assert resp.status_code in (200, 400)

