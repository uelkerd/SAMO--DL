import io
import json
import os
import tempfile
import pytest
import requests

from flask import Flask

# These tests hit the running Flask app if present, otherwise skip.
BASE_URL = os.environ.get("SECURE_API_BASE", "http://127.0.0.1:8081")
API_KEY = os.environ.get("ADMIN_API_KEY", "test-key-123")

pytestmark = pytest.mark.integration


def _headers(api_key: str):
    return {"X-API-Key": api_key, "Content-Type": "application/json"}


def _multipart_headers(api_key: str):
    return {"X-API-Key": api_key}


def test_summarize_requires_api_key(secure_api_server_url):
    url = f"{secure_api_server_url}/api/summarize"
    resp = requests.post(url, headers={"Content-Type": "application/json"}, data=json.dumps({"text": "hello"}))
    assert resp.status_code == 401


@pytest.mark.parametrize("text", [
    "This is a long example paragraph that should be summarized into something shorter and more concise.",
])
def test_summarize_json_endpoint(secure_api_server_url, text):
    url = f"{secure_api_server_url}/api/summarize"
    resp = requests.post(url, headers=_headers(os.environ.get("ADMIN_API_KEY", "test-key-123")), data=json.dumps({"text": text}))
    assert resp.status_code in (200, 503), resp.text
    if resp.status_code == 200:
        data = resp.json()
        assert "summary" in data
        assert isinstance(data["summary"], str)


def test_analyze_journal_bad_input(secure_api_server_url):
    url = f"{secure_api_server_url}/api/analyze/journal"
    resp = requests.post(url, headers=_headers(os.environ.get("ADMIN_API_KEY", "test-key-123")), data=json.dumps({"text": ""}))
    assert resp.status_code == 400


def test_analyze_journal_json_endpoint(secure_api_server_url):
    url = f"{secure_api_server_url}/api/analyze/journal"
    body = {
        "text": "Today was amazing. I finished my project and felt proud.",
        "generate_summary": True,
    }
    resp = requests.post(url, headers=_headers(os.environ.get("ADMIN_API_KEY", "test-key-123")), data=json.dumps(body))
    assert resp.status_code in (200, 503), resp.text
    if resp.status_code == 200:
        data = resp.json()
        assert "emotion_analysis" in data
        assert "processing_time_ms" in data


def test_transcribe_smoke(secure_api_server_url, tmp_path):
    import wave
    file_path = tmp_path / "sample.wav"
    with wave.open(str(file_path), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 16000)

    url = f"{secure_api_server_url}/api/transcribe"
    with open(file_path, "rb") as f:
        files = {"file": ("sample.wav", f, "audio/wav")}
        resp = requests.post(url, headers=_multipart_headers(os.environ.get("ADMIN_API_KEY", "test-key-123")), files=files)
    assert resp.status_code in (200, 400, 503), resp.text


def test_transcribe_batch_smoke(secure_api_server_url, tmp_path):
    import wave
    f1 = tmp_path / "a.wav"
    f2 = tmp_path / "b.wav"
    for fp in (f1, f2):
        with wave.open(str(fp), "w") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(b"\x00\x00" * 8000)

    url = f"{secure_api_server_url}/api/transcribe_batch"
    files = [
        ("files", ("a.wav", open(f1, "rb"), "audio/wav")),
        ("files", ("b.wav", open(f2, "rb"), "audio/wav")),
    ]
    try:
        resp = requests.post(url, headers=_multipart_headers(os.environ.get("ADMIN_API_KEY", "test-key-123")), files=files)
    finally:
        for _, (name, fh, _) in files:
            try:
                fh.close()
            except Exception:
                pass
    assert resp.status_code in (200, 400, 503), resp.text