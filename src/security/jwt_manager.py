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
        self.blacklisted_tokens: set = set()
    
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
            return None
        except jwt.InvalidTokenError:
            return None
    
    def refresh_access_token(self, refresh_token: str) -> Optional[str]:
        """Refresh an access token using a valid refresh token"""
        payload = self.verify_token(refresh_token)
        if not payload or payload.get("type") != "refresh":
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
            self.blacklisted_tokens.add(token)
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
        for token in self.blacklisted_tokens:
            try:
                payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
                if payload.get("exp") and datetime.fromtimestamp(payload["exp"]) < current_time:
                    tokens_to_remove.add(token)
            except jwt.InvalidTokenError:
                tokens_to_remove.add(token)
        
        self.blacklisted_tokens -= tokens_to_remove
        return initial_count - len(self.blacklisted_tokens)

# Global JWT manager instance
jwt_manager = JWTManager() 