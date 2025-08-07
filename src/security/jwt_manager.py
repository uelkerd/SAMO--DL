"""
JWT-based Authentication Manager for SAMO Deep Learning API

This module provides comprehensive JWT token management including:
- Access and refresh token creation
- Token validation and verification
- Token blacklisting for logout
- Automatic token refresh
- User permission management
"""

import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
from typing_extensions import TypedDict

import jwt
from pydantic import BaseModel, Field

# Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

class TokenPayload(BaseModel):
    """Token payload structure"""
    user_id: str = Field(..., description="User identifier")
    username: str = Field(..., description="Username")
    email: str = Field(..., description="User email")
    permissions: List[str] = Field(default_factory=list, description="User permissions")
    exp: Optional[int] = Field(None, description="Expiration timestamp")
    iat: Optional[int] = Field(None, description="Issued at timestamp")
    type: Optional[str] = Field(None, description="Token type, e.g., 'refresh'")

class TokenResponse(BaseModel):
    """Token response structure"""
    access_token: str = Field(..., description="Access token")
    refresh_token: str = Field(..., description="Refresh token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Access token expiration in seconds")

class JWTManager:
    """Comprehensive JWT token management system"""
    
    def __init__(self, secret_key: str = SECRET_KEY, algorithm: str = ALGORITHM):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.blacklisted_tokens: dict = {}  # Changed to dict: {token: exp_datetime}
    
    def create_access_token(self, user_data: Dict[str, Any]) -> str:
        """Create a new access token"""
        payload = {
            "user_id": user_data["user_id"],
            "username": user_data["username"],
            "email": user_data["email"],
            "permissions": user_data.get("permissions", []),
            "exp": datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
            "iat": datetime.utcnow()
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def create_refresh_token(self, user_data: Dict[str, Any]) -> str:
        """Create a new refresh token"""
        payload = {
            "user_id": user_data["user_id"],
            "username": user_data["username"],
            "email": user_data["email"],
            "permissions": user_data.get("permissions", []),
            "exp": datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS),
            "iat": datetime.utcnow(),
            "type": "refresh"
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def create_token_pair(self, user_data: Dict[str, Any]) -> TokenResponse:
        """Create both access and refresh tokens"""
        access_token = self.create_access_token(user_data)
        refresh_token = self.create_refresh_token(user_data)
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )
    
    def verify_token(self, token: str) -> Optional[TokenPayload]:
        """Verify and decode a token"""
        try:
            if token in self.blacklisted_tokens:
                return None
            
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return TokenPayload(**payload)
        except jwt.ExpiredSignatureError:
            logger.warning(f"Token expired: {token[:10]}...")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Token verification error: {str(e)}")
            return None
    
    def refresh_access_token(self, refresh_token: str) -> Optional[str]:
        """Refresh an access token using a valid refresh token"""
        payload = self.verify_token(refresh_token)
        if not payload or getattr(payload, "type", None) != "refresh":
            return None
        
        user_data = {
            "user_id": payload.user_id,
            "username": payload.username,
            "email": payload.email,
            "permissions": payload.permissions
        }
        return self.create_access_token(user_data)
    
    def blacklist_token(self, token: str) -> bool:
        """Add a token to the blacklist"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            exp_datetime = datetime.fromtimestamp(payload["exp"]) if payload.get("exp") else None
            self.blacklisted_tokens[token] = exp_datetime
            return True
        except jwt.InvalidTokenError:
            return False
    
    def is_token_blacklisted(self, token: str) -> bool:
        """Check if a token is blacklisted"""
        return token in self.blacklisted_tokens
    
    def get_user_permissions(self, token: str) -> List[str]:
        """Extract user permissions from token"""
        payload = self.verify_token(token)
        return payload.permissions if payload else []
    
    def has_permission(self, token: str, required_permission: str) -> bool:
        """Check if user has a specific permission"""
        permissions = self.get_user_permissions(token)
        return required_permission in permissions
    
    def cleanup_expired_tokens(self) -> int:
        """Clean up expired tokens from blacklist"""
        initial_count = len(self.blacklisted_tokens)
        current_time = datetime.utcnow()
        
        tokens_to_remove = set()
        # self.blacklisted_tokens is now a dict: {token: exp_datetime}
        for token, exp_datetime in self.blacklisted_tokens.items():
            if exp_datetime and exp_datetime < current_time:
                tokens_to_remove.add(token)
        
        for token in tokens_to_remove:
            self.blacklisted_tokens.pop(token, None)
        return initial_count - len(self.blacklisted_tokens)

# Global JWT manager instance
jwt_manager = JWTManager() 