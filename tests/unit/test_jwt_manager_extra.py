#!/usr/bin/env python3
"""Extra unit tests for JWTManager to increase coverage."""

from datetime import datetime, timedelta
import json

from src.security.jwt_manager import JWTManager


def test_create_token_pair_structure():
    mgr = JWTManager()
    token_pair = mgr.create_token_pair(
        {
            "user_id": "u1",
            "username": "user@example.com",
            "email": "user@example.com",
            "permissions": ["read"],
        }
    )
    # Ensure keys and types look correct
    assert set(token_pair.keys()) == {"access_token", "refresh_token", "token_type", "expires_in"}
    assert isinstance(token_pair["access_token"], str)
    assert isinstance(token_pair["refresh_token"], str)
    assert token_pair["token_type"] == "bearer"
    assert isinstance(token_pair["expires_in"], int)


def test_verify_invalid_token_returns_none():
    mgr = JWTManager()
    assert mgr.verify_token("not-a-jwt") is None


def test_blacklist_and_cleanup_flow():
    mgr = JWTManager()
    # Create a short-lived token and blacklist it
    token = mgr.create_access_token(
        {
            "user_id": "u2",
            "username": "user2@example.com",
            "email": "user2@example.com",
            "permissions": [],
        }
    )
    assert mgr.blacklist_token(token) is True
    assert mgr.is_token_blacklisted(token) is True

    # Simulate expired blacklist entry and cleanup
    mgr.blacklisted_tokens[token] = datetime.utcnow() - timedelta(seconds=1)
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

