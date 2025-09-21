#!/usr/bin/env python3
"""Extra unit tests for JWTManager to increase coverage."""

import time
from datetime import datetime, timedelta

from src.security.jwt_manager import JWTManager


def test_create_token_pair_structure():
    mgr = JWTManager()
    token_pair = mgr.create_token_pair(
        {
            "user_id": "u1",
            "username": "user@example.com",
            "email": "user@example.com",
            "permissions": ["read"],
        },
    )
    # Ensure keys and types look correct
    token_pair_dict = (
        token_pair.dict() if hasattr(token_pair, "dict") else dict(token_pair)
    )
    assert set(token_pair_dict.keys()) == {
        "access_token",
        "refresh_token",
        "token_type",
        "expires_in",
    }
    assert isinstance(token_pair.access_token, str)
    assert isinstance(token_pair.refresh_token, str)
    assert token_pair.token_type == "bearer"
    assert isinstance(token_pair.expires_in, int)


def test_verify_invalid_token_returns_none():
    mgr = JWTManager()
    assert mgr.verify_token("not-a-jwt") is None


def test_blacklist_and_cleanup_flow(monkeypatch):
    mgr = JWTManager()
    # Create a token and blacklist it using public API
    token = mgr.create_access_token(
        {
            "user_id": "u2",
            "username": "user2@example.com",
            "email": "user2@example.com",
            "permissions": [],
        },
    )
    assert mgr.blacklist_token(token) is True
    assert mgr.is_token_blacklisted(token) is True

    # Determine the stored expiration timestamp by decoding without verifying
    import jwt

    payload = jwt.decode(
        token, options={"verify_signature": False, "verify_exp": False}
    )
    exp_ts = payload.get("exp")

    # Monkeypatch datetime.utcnow to simulate time past expiration for cleanup logic
    class _FakeDateTime(datetime):
        @classmethod
        def utcnow(cls):
            # jump past the token's expiration
            return datetime.fromtimestamp(exp_ts) + timedelta(seconds=5)

    monkeypatch.setattr("src.security.jwt_manager.datetime", _FakeDateTime)

    removed = mgr.cleanup_expired_tokens()
    assert removed >= 1


def test_refresh_access_token_success_and_failure():
    mgr = JWTManager()
    user = {
        "user_id": "u3",
        "username": "user3@example.com",
        "email": "user3@example.com",
        "permissions": ["read", "write"],
    }
    # Access token should not be usable as refresh
    access = mgr.create_access_token(user)
    assert mgr.refresh_access_token(access) is None

    # Refresh token should yield a new access token
    refresh = mgr.create_refresh_token(user)
    new_access = mgr.refresh_access_token(refresh)
    assert isinstance(new_access, str) and len(new_access) > 10

    # Expired refresh token should fail (re-sign with manager's secret to keep signature valid)
    import jwt

    payload = jwt.decode(
        refresh, options={"verify_signature": False, "verify_exp": False}
    )
    payload["exp"] = int(time.time()) - 10
    expired_refresh = jwt.encode(payload, mgr.secret_key, algorithm=mgr.algorithm)
    assert mgr.refresh_access_token(expired_refresh) is None


def test_permissions_helpers():
    mgr = JWTManager()
    user = {
        "user_id": "u4",
        "username": "user4@example.com",
        "email": "user4@example.com",
        "permissions": ["alpha"],
    }
    token = mgr.create_access_token(user)
    assert mgr.has_permission(token, "alpha") is True
    assert mgr.has_permission(token, "beta") is False
    perms = mgr.get_user_permissions(token)
    assert "alpha" in perms and "beta" not in perms

    # Malformed token handling
    assert mgr.get_user_permissions("not.a.jwt") == []

    # Token missing permissions field should default to empty list
    user_no_permissions = {
        "user_id": "u5",
        "username": "user5@example.com",
        "email": "user5@example.com",
        # intentionally omit "permissions"
    }
    token_no_perms = mgr.create_access_token(user_no_permissions)
    assert mgr.get_user_permissions(token_no_perms) == []
