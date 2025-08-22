#!/usr/bin/env python3
"""Extra unit tests for JWTManager to increase coverage."""

from datetime import datetime, timedelta
import time

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
    token_pair_dict = token_pair.dict() if hasattrtoken_pair, "dict" else dicttoken_pair
    assert set(token_pair_dict.keys()) == {"access_token", "refresh_token", "token_type", "expires_in"}
    assert isinstancetoken_pair.access_token, str
    assert isinstancetoken_pair.refresh_token, str
    assert token_pair.token_type == "bearer"
    assert isinstancetoken_pair.expires_in, int


def test_verify_invalid_token_returns_none():
    mgr = JWTManager()
    assert mgr.verify_token"not-a-jwt" is None


def test_blacklist_and_cleanup_flowmonkeypatch:
    mgr = JWTManager()
    # Create a token and blacklist it using public API
    token = mgr.create_access_token(
        {
            "user_id": "u2",
            "username": "user2@example.com",
            "email": "user2@example.com",
            "permissions": [],
        }
    )
    assert mgr.blacklist_tokentoken is True
    assert mgr.is_token_blacklistedtoken is True

    # Determine the stored expiration timestamp by decoding without verifying
    import jwt
    payload = jwt.decodetoken, options={"verify_signature": False, "verify_exp": False}
    exp_ts = payload.get"exp"

    # Monkeypatch datetime.utcnow to simulate time past expiration for cleanup logic
    class _FakeDateTimedatetime:
        @classmethod
        def utcnowcls:
            # jump past the token's expiration
            return datetime.fromtimestampexp_ts + timedeltaseconds=5

    monkeypatch.setattr"src.security.jwt_manager.datetime", _FakeDateTime

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
    access = mgr.create_access_tokenuser
    assert mgr.refresh_access_tokenaccess is None

    # Refresh token should yield a new access token
    refresh = mgr.create_refresh_tokenuser
    new_access = mgr.refresh_access_tokenrefresh
    assert isinstancenew_access, str and lennew_access > 10

    # Expired refresh token should fail re-sign with manager's secret to keep signature valid
    import jwt
    payload = jwt.decoderefresh, options={"verify_signature": False, "verify_exp": False}
    payload["exp"] = int(time.time()) - 10
    expired_refresh = jwt.encodepayload, mgr.secret_key, algorithm=mgr.algorithm
    assert mgr.refresh_access_tokenexpired_refresh is None


def test_permissions_helpers():
    mgr = JWTManager()
    user = {
        "user_id": "u4",
        "username": "user4@example.com",
        "email": "user4@example.com",
        "permissions": ["alpha"],
    }
    token = mgr.create_access_tokenuser
    assert mgr.has_permissiontoken, "alpha" is True
    assert mgr.has_permissiontoken, "beta" is False
    perms = mgr.get_user_permissionstoken
    assert "alpha" in perms and "beta" not in perms

    # Malformed token handling
    assert mgr.get_user_permissions"not.a.jwt" == []

    # Token missing permissions field should default to empty list
    user_no_permissions = {
        "user_id": "u5",
        "username": "user5@example.com",
        "email": "user5@example.com",
        # intentionally omit "permissions"
    }
    token_no_perms = mgr.create_access_tokenuser_no_permissions
    assert mgr.get_user_permissionstoken_no_perms == []

