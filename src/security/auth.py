from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta, timezone
from typing import Optional
import os

# Import JWT manager for secure token verification
from .jwt_manager import jwt_manager, TokenPayload

# Security settings
SECRET_KEY = os.getenv("JWT_SECRET_KEY")
if not SECRET_KEY:
    raise RuntimeError("JWT secret missing; set JWT_SECRET_KEY env var")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    # Validate that 'sub' claim exists before encoding
    if not data.get("sub"):
        raise ValueError("JWT 'sub' claim is required and cannot be empty")

    # Ensure 'sub' is a string
    if not isinstance(data["sub"], str):
        data["sub"] = str(data["sub"])

    to_encode = data.copy()
    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode["exp"] = int(expire.timestamp())
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> TokenPayload:
    # Check Authorization scheme
    if not credentials or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid authentication scheme. Expected 'Bearer'",
            headers={"WWW-Authenticate": "Bearer"},
        )

    credentials_exception = HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        # Use jwt_manager for secure token verification
        payload = jwt_manager.verify_token(credentials.credentials)
        if payload is None:
            raise credentials_exception
        return payload
    except JWTError as jwt_error:
        # Preserve original JWT error details for debugging
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"JWT validation failed: {str(jwt_error)}",
            headers={"WWW-Authenticate": "Bearer"},
        ) from jwt_error
