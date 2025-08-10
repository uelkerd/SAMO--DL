#!/usr/bin/env python3
"""Test FastAPI HTTP exception handler contract to bump coverage."""

from fastapi import HTTPException
from fastapi.testclient import TestClient

from src.unified_ai_api import app


def test_http_exception_handler_400_detail_shape():
    client = TestClient(app)

    @app.get("/__raise_400_test__")
    def __raise_400_test__():  # type: ignore
        raise HTTPException(status_code=400, detail="Bad input")

    resp = client.get("/__raise_400_test__")
    assert resp.status_code == 400
    body = resp.json()
    assert isinstance(body, dict) and "detail" in body
    assert body["detail"] == "Bad input"


def test_http_exception_handler_500_shape():
    client = TestClient(app)

    @app.get("/__raise_500_test__")
    def __raise_500_test__():  # type: ignore
        raise HTTPException(status_code=500, detail="Boom")

    resp = client.get("/__raise_500_test__")
    assert resp.status_code == 500
    body = resp.json()
    assert "error" in body and body["error"] == "Boom"
    assert body.get("status_code") == 500


def test_http_exception_handler_other_4xx_codes():
    client = TestClient(app)

    @app.get("/__raise_401_test__")
    def __raise_401_test__():  # type: ignore
        raise HTTPException(status_code=401, detail="Unauthorized")

    @app.get("/__raise_403_test__")
    def __raise_403_test__():  # type: ignore
        raise HTTPException(status_code=403, detail="Forbidden")

    for path, expected in [
        ("/__raise_401_test__", (401, "Unauthorized")),
        ("/__raise_403_test__", (403, "Forbidden")),
    ]:
        resp = client.get(path)
        assert resp.status_code == expected[0]
        body = resp.json()
        assert isinstance(body, dict) and "detail" in body
        assert body["detail"] == expected[1]

