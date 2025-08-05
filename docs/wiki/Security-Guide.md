# 🔒 Security Guide

Welcome to the SAMO Brain Security Guide! This comprehensive guide covers security best practices, authentication, authorization, data protection, and monitoring to ensure our AI system remains secure and compliant.

## 🚀 **Quick Security Checklist (5 minutes)**

### **Essential Security Measures**
- [ ] **API Key Authentication** - All endpoints require valid API keys
- [ ] **Rate Limiting** - Prevent abuse with request rate limits
- [ ] **Input Validation** - Sanitize all user inputs
- [ ] **HTTPS Only** - All communications encrypted
- [ ] **Secret Management** - Use environment variables for secrets
- [ ] **Regular Security Updates** - Keep dependencies updated

### **Security Verification**
```bash
# Check for security vulnerabilities
pip install safety bandit pip-audit

# Run security scans
safety check
bandit -r src/
pip-audit

# Verify HTTPS configuration
curl -I https://your-api-domain.com/health
```

---

## 🔐 **Authentication & Authorization**

### **API Key Management**

```python
# src/security/api_key_manager.py
import secrets
import hashlib
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import redis

class APIKeyManager:
    """Manages API key authentication and authorization."""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.key_prefix = "api_key:"
        self.rotation_interval = 86400  # 24 hours
    
    def generate_api_key(self, user_id: str, permissions: List[str]) -> str:
        """Generate a new API key for a user."""
        # Generate secure random key
        api_key = secrets.token_urlsafe(32)
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        # Store key metadata
        key_data = {
            "user_id": user_id,
            "permissions": permissions,
            "created_at": datetime.now().isoformat(),
            "last_used": None,
            "is_active": True
        }
        
        # Store in Redis with expiration
        self.redis.setex(
            f"{self.key_prefix}{key_hash}",
            self.rotation_interval,
            str(key_data)
        )
        
        return api_key
    
    def validate_api_key(self, api_key: str) -> Optional[Dict]:
        """Validate API key and return user data."""
        if not api_key:
            return None
        
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        key_data = self.redis.get(f"{self.key_prefix}{key_hash}")
        
        if not key_data:
            return None
        
        # Parse key data
        user_data = eval(key_data.decode())
        
        # Check if key is active
        if not user_data.get("is_active", False):
            return None
        
        # Update last used timestamp
        user_data["last_used"] = datetime.now().isoformat()
        self.redis.setex(
            f"{self.key_prefix}{key_hash}",
            self.rotation_interval,
            str(user_data)
        )
        
        return user_data
    
    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke an API key."""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        return bool(self.redis.delete(f"{self.key_prefix}{key_hash}"))
    
    def list_user_keys(self, user_id: str) -> List[Dict]:
        """List all API keys for a user."""
        keys = []
        pattern = f"{self.key_prefix}*"
        
        for key in self.redis.scan_iter(match=pattern):
            key_data = self.redis.get(key)
            if key_data:
                user_data = eval(key_data.decode())
                if user_data["user_id"] == user_id:
                    keys.append(user_data)
        
        return keys
```

### **JWT Token Authentication**

```python
# src/security/jwt_manager.py
import jwt
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

class JWTManager:
    """Manages JWT token authentication."""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.token_expiry = 3600  # 1 hour
    
    def create_token(self, user_id: str, permissions: List[str]) -> str:
        """Create a JWT token for a user."""
        payload = {
            "user_id": user_id,
            "permissions": permissions,
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(seconds=self.token_expiry)
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def validate_token(self, token: str) -> Optional[Dict]:
        """Validate JWT token and return payload."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def refresh_token(self, token: str) -> Optional[str]:
        """Refresh an existing JWT token."""
        payload = self.validate_token(token)
        if payload:
            return self.create_token(payload["user_id"], payload["permissions"])
        return None
```

### **Role-Based Access Control (RBAC)**

```python
# src/security/rbac.py
from enum import Enum
from typing import List, Dict, Set
from functools import wraps

class Permission(Enum):
    """Available permissions."""
    READ_PREDICTIONS = "read_predictions"
    WRITE_PREDICTIONS = "write_predictions"
    READ_METRICS = "read_metrics"
    WRITE_METRICS = "write_metrics"
    ADMIN = "admin"

class Role(Enum):
    """Available roles."""
    USER = "user"
    ANALYST = "analyst"
    ADMIN = "admin"

class RBACManager:
    """Manages role-based access control."""
    
    def __init__(self):
        self.role_permissions = {
            Role.USER: {
                Permission.READ_PREDICTIONS,
                Permission.WRITE_PREDICTIONS
            },
            Role.ANALYST: {
                Permission.READ_PREDICTIONS,
                Permission.WRITE_PREDICTIONS,
                Permission.READ_METRICS
            },
            Role.ADMIN: {
                Permission.READ_PREDICTIONS,
                Permission.WRITE_PREDICTIONS,
                Permission.READ_METRICS,
                Permission.WRITE_METRICS,
                Permission.ADMIN
            }
        }
    
    def has_permission(self, user_permissions: List[str], required_permission: Permission) -> bool:
        """Check if user has required permission."""
        return required_permission.value in user_permissions
    
    def get_role_permissions(self, role: Role) -> Set[Permission]:
        """Get permissions for a role."""
        return self.role_permissions.get(role, set())
    
    def require_permission(self, permission: Permission):
        """Decorator to require specific permission."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Extract user permissions from request context
                user_permissions = kwargs.get('user_permissions', [])
                
                if not self.has_permission(user_permissions, permission):
                    raise PermissionError(f"Permission {permission.value} required")
                
                return func(*args, **kwargs)
            return wrapper
        return decorator
```

---

## 🛡️ **Input Validation & Sanitization**

### **Request Validation**

```python
# src/security/input_validation.py
import re
from typing import Optional, Dict, Any
from pydantic import BaseModel, validator, Field
import html

class SecurePredictionRequest(BaseModel):
    """Secure prediction request with validation."""
    
    text: str = Field(..., min_length=1, max_length=10000)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    
    @validator('text')
    def validate_text(cls, v):
        """Validate and sanitize text input."""
        if not v or not v.strip():
            raise ValueError("Text cannot be empty")
        
        # Remove HTML tags
        v = re.sub(r'<[^>]+>', '', v)
        
        # Escape HTML entities
        v = html.escape(v)
        
        # Remove control characters
        v = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', v)
        
        # Check for suspicious patterns
        suspicious_patterns = [
            r'script\s*:',  # JavaScript injection
            r'javascript\s*:',  # JavaScript injection
            r'data\s*:',  # Data URL injection
            r'vbscript\s*:',  # VBScript injection
            r'on\w+\s*=',  # Event handlers
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError("Suspicious content detected")
        
        return v.strip()
    
    @validator('threshold')
    def validate_threshold(cls, v):
        """Validate threshold value."""
        if not isinstance(v, (int, float)):
            raise ValueError("Threshold must be a number")
        
        if v < 0 or v > 1:
            raise ValueError("Threshold must be between 0 and 1")
        
        return float(v)

class InputSanitizer:
    """Sanitizes various types of input."""
    
    @staticmethod
    def sanitize_text(text: str) -> str:
        """Sanitize text input."""
        if not text:
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Escape HTML entities
        text = html.escape(text)
        
        # Remove control characters
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        return text.strip()
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename."""
        # Remove path traversal attempts
        filename = filename.replace('..', '').replace('/', '').replace('\\', '')
        
        # Remove dangerous characters
        filename = re.sub(r'[<>:"|?*]', '', filename)
        
        # Limit length
        if len(filename) > 255:
            filename = filename[:255]
        
        return filename
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format."""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @staticmethod
    def validate_url(url: str) -> bool:
        """Validate URL format."""
        pattern = r'^https?://[^\s/$.?#].[^\s]*$'
        return bool(re.match(pattern, url))
```

### **SQL Injection Prevention**

```python
# src/security/sql_injection.py
import sqlite3
from typing import Any, List, Tuple
import re

class SecureDatabaseManager:
    """Database manager with SQL injection prevention."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def execute_query(self, query: str, params: Tuple[Any, ...] = ()) -> List[Dict]:
        """Execute a query with parameterized statements."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(query, params)
                
                if query.strip().upper().startswith('SELECT'):
                    return [dict(row) for row in cursor.fetchall()]
                else:
                    conn.commit()
                    return []
        except sqlite3.Error as e:
            raise Exception(f"Database error: {e}")
    
    def insert_prediction(self, text: str, emotion: str, confidence: float) -> int:
        """Insert prediction with parameterized query."""
        query = """
        INSERT INTO predictions (text, emotion, confidence, timestamp)
        VALUES (?, ?, ?, datetime('now'))
        """
        self.execute_query(query, (text, emotion, confidence))
        return self.execute_query("SELECT last_insert_rowid()")[0]['last_insert_rowid()']
    
    def get_predictions_by_emotion(self, emotion: str) -> List[Dict]:
        """Get predictions by emotion with parameterized query."""
        query = "SELECT * FROM predictions WHERE emotion = ? ORDER BY timestamp DESC"
        return self.execute_query(query, (emotion,))
    
    def search_predictions(self, search_term: str) -> List[Dict]:
        """Search predictions with parameterized query."""
        query = "SELECT * FROM predictions WHERE text LIKE ? ORDER BY timestamp DESC"
        return self.execute_query(query, (f"%{search_term}%",))
```

---

## 🔒 **Data Protection & Encryption**

### **Data Encryption**

```python
# src/security/encryption.py
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os
from typing import Optional

class DataEncryption:
    """Handles data encryption and decryption."""
    
    def __init__(self, secret_key: Optional[str] = None):
        if secret_key:
            self.key = self._derive_key(secret_key)
        else:
            self.key = Fernet.generate_key()
        
        self.cipher = Fernet(self.key)
    
    def _derive_key(self, password: str) -> bytes:
        """Derive encryption key from password."""
        salt = b'samo_brain_salt'  # In production, use random salt
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data."""
        encrypted_data = self.cipher.encrypt(data.encode())
        return base64.urlsafe_b64encode(encrypted_data).decode()
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_data = self.cipher.decrypt(encrypted_bytes)
            return decrypted_data.decode()
        except Exception as e:
            raise ValueError(f"Failed to decrypt data: {e}")
    
    def encrypt_file(self, file_path: str, encrypted_path: str):
        """Encrypt a file."""
        with open(file_path, 'rb') as f:
            data = f.read()
        
        encrypted_data = self.cipher.encrypt(data)
        
        with open(encrypted_path, 'wb') as f:
            f.write(encrypted_data)
    
    def decrypt_file(self, encrypted_path: str, decrypted_path: str):
        """Decrypt a file."""
        with open(encrypted_path, 'rb') as f:
            encrypted_data = f.read()
        
        decrypted_data = self.cipher.decrypt(encrypted_data)
        
        with open(decrypted_path, 'wb') as f:
            f.write(decrypted_data)

class SecureStorage:
    """Secure storage for sensitive data."""
    
    def __init__(self, encryption: DataEncryption):
        self.encryption = encryption
    
    def store_sensitive_data(self, key: str, value: str):
        """Store sensitive data encrypted."""
        encrypted_value = self.encryption.encrypt_data(value)
        # Store in secure storage (e.g., encrypted database)
        return encrypted_value
    
    def retrieve_sensitive_data(self, key: str) -> str:
        """Retrieve and decrypt sensitive data."""
        # Retrieve from secure storage
        encrypted_value = self._get_from_storage(key)
        return self.encryption.decrypt_data(encrypted_value)
    
    def _get_from_storage(self, key: str) -> str:
        """Get data from storage (implement based on storage type)."""
        # Implementation depends on storage backend
        pass
```

### **Data Masking & Anonymization**

```python
# src/security/data_masking.py
import re
import hashlib
from typing import Dict, Any, List

class DataMasking:
    """Handles data masking and anonymization."""
    
    @staticmethod
    def mask_email(email: str) -> str:
        """Mask email address."""
        if '@' not in email:
            return email
        
        username, domain = email.split('@')
        masked_username = username[0] + '*' * (len(username) - 2) + username[-1]
        return f"{masked_username}@{domain}"
    
    @staticmethod
    def mask_phone(phone: str) -> str:
        """Mask phone number."""
        if len(phone) < 4:
            return phone
        
        return '*' * (len(phone) - 4) + phone[-4:]
    
    @staticmethod
    def hash_sensitive_data(data: str) -> str:
        """Hash sensitive data for anonymization."""
        return hashlib.sha256(data.encode()).hexdigest()
    
    @staticmethod
    def anonymize_text(text: str, preserve_length: bool = True) -> str:
        """Anonymize text while preserving structure."""
        # Replace words with placeholders
        words = text.split()
        anonymized_words = []
        
        for word in words:
            if len(word) > 3:
                anonymized_word = word[0] + '*' * (len(word) - 2) + word[-1]
            else:
                anonymized_word = '*' * len(word)
            
            anonymized_words.append(anonymized_word)
        
        return ' '.join(anonymized_words)
    
    @staticmethod
    def mask_predictions(predictions: List[Dict]) -> List[Dict]:
        """Mask sensitive data in predictions."""
        masked_predictions = []
        
        for pred in predictions:
            masked_pred = pred.copy()
            
            # Mask text content
            if 'text' in masked_pred:
                masked_pred['text'] = DataMasking.anonymize_text(masked_pred['text'])
            
            # Hash user identifiers
            if 'user_id' in masked_pred:
                masked_pred['user_id'] = DataMasking.hash_sensitive_data(masked_pred['user_id'])
            
            masked_predictions.append(masked_pred)
        
        return masked_predictions
```

---

## 🚨 **Security Monitoring & Alerting**

### **Security Event Logging**

```python
# src/security/security_monitor.py
import logging
import json
from datetime import datetime
from typing import Dict, Any, List
import requests

class SecurityEvent:
    """Represents a security event."""
    
    def __init__(self, event_type: str, severity: str, details: Dict[str, Any]):
        self.event_type = event_type
        self.severity = severity
        self.details = details
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_type": self.event_type,
            "severity": self.severity,
            "details": self.details,
            "timestamp": self.timestamp
        }

class SecurityMonitor:
    """Monitors and logs security events."""
    
    def __init__(self, log_file: str = "logs/security.log"):
        self.log_file = log_file
        self.setup_logging()
        self.suspicious_patterns = [
            r'script\s*:',
            r'javascript\s*:',
            r'data\s*:',
            r'vbscript\s*:',
            r'on\w+\s*=',
            r'<script',
            r'</script>',
            r'<iframe',
            r'<object',
            r'<embed'
        ]
    
    def setup_logging(self):
        """Setup security logging."""
        logging.basicConfig(
            filename=self.log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def log_security_event(self, event: SecurityEvent):
        """Log a security event."""
        log_entry = json.dumps(event.to_dict())
        logging.warning(f"SECURITY_EVENT: {log_entry}")
        
        # Send alert for high severity events
        if event.severity in ['high', 'critical']:
            self.send_alert(event)
    
    def detect_suspicious_activity(self, request_data: Dict[str, Any]) -> List[SecurityEvent]:
        """Detect suspicious activity in request data."""
        events = []
        
        # Check for suspicious patterns in text
        if 'text' in request_data:
            text = request_data['text']
            for pattern in self.suspicious_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    events.append(SecurityEvent(
                        event_type="suspicious_input",
                        severity="medium",
                        details={
                            "pattern": pattern,
                            "input": text[:100] + "..." if len(text) > 100 else text
                        }
                    ))
        
        # Check for unusual request patterns
        if 'threshold' in request_data:
            threshold = request_data['threshold']
            if threshold < 0 or threshold > 1:
                events.append(SecurityEvent(
                    event_type="invalid_parameter",
                    severity="low",
                    details={"parameter": "threshold", "value": threshold}
                ))
        
        return events
    
    def monitor_rate_limiting(self, client_id: str, request_count: int, limit: int):
        """Monitor rate limiting events."""
        if request_count > limit * 0.8:  # 80% of limit
            event = SecurityEvent(
                event_type="rate_limit_warning",
                severity="medium",
                details={
                    "client_id": client_id,
                    "request_count": request_count,
                    "limit": limit
                }
            )
            self.log_security_event(event)
    
    def detect_brute_force(self, client_id: str, failed_attempts: int):
        """Detect brute force attempts."""
        if failed_attempts > 5:
            event = SecurityEvent(
                event_type="brute_force_detected",
                severity="high",
                details={
                    "client_id": client_id,
                    "failed_attempts": failed_attempts
                }
            )
            self.log_security_event(event)
    
    def send_alert(self, event: SecurityEvent):
        """Send security alert."""
        # Implementation depends on alerting system
        # Could send email, Slack message, etc.
        alert_data = {
            "title": f"Security Alert: {event.event_type}",
            "severity": event.severity,
            "details": event.details,
            "timestamp": event.timestamp
        }
        
        # Example: Send to webhook
        try:
            requests.post(
                "https://your-alerting-webhook.com/security",
                json=alert_data,
                timeout=5
            )
        except Exception as e:
            logging.error(f"Failed to send security alert: {e}")
    
    def generate_security_report(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Generate security report for time period."""
        # Implementation to analyze security logs
        # and generate summary report
        pass
```

### **Intrusion Detection**

```python
# src/security/intrusion_detection.py
import time
from collections import defaultdict
from typing import Dict, List, Set
import ipaddress

class IntrusionDetectionSystem:
    """Basic intrusion detection system."""
    
    def __init__(self):
        self.failed_attempts = defaultdict(int)
        self.suspicious_ips = set()
        self.blocked_ips = set()
        self.request_patterns = defaultdict(list)
        self.alert_threshold = 10
    
    def analyze_request(self, client_ip: str, request_data: Dict[str, Any]) -> bool:
        """Analyze request for suspicious activity."""
        # Check if IP is blocked
        if client_ip in self.blocked_ips:
            return False
        
        # Track request patterns
        self.request_patterns[client_ip].append({
            "timestamp": time.time(),
            "data": request_data
        })
        
        # Clean old patterns (keep last 100 requests)
        if len(self.request_patterns[client_ip]) > 100:
            self.request_patterns[client_ip] = self.request_patterns[client_ip][-100:]
        
        # Check for suspicious patterns
        if self._is_suspicious_pattern(client_ip, request_data):
            self.failed_attempts[client_ip] += 1
            
            if self.failed_attempts[client_ip] >= self.alert_threshold:
                self.suspicious_ips.add(client_ip)
                self._block_ip(client_ip)
                return False
        
        return True
    
    def _is_suspicious_pattern(self, client_ip: str, request_data: Dict[str, Any]) -> bool:
        """Check for suspicious request patterns."""
        patterns = self.request_patterns[client_ip]
        
        if len(patterns) < 5:
            return False
        
        # Check for rapid requests
        recent_requests = [p for p in patterns if time.time() - p["timestamp"] < 60]
        if len(recent_requests) > 50:  # More than 50 requests per minute
            return True
        
        # Check for repeated failed requests
        # Implementation depends on your failure detection logic
        
        return False
    
    def _block_ip(self, client_ip: str):
        """Block an IP address."""
        self.blocked_ips.add(client_ip)
        # Log blocking event
        print(f"IP {client_ip} has been blocked due to suspicious activity")
    
    def unblock_ip(self, client_ip: str):
        """Unblock an IP address."""
        self.blocked_ips.discard(client_ip)
        self.suspicious_ips.discard(client_ip)
        self.failed_attempts[client_ip] = 0
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get current security status."""
        return {
            "blocked_ips": list(self.blocked_ips),
            "suspicious_ips": list(self.suspicious_ips),
            "total_failed_attempts": sum(self.failed_attempts.values()),
            "active_patterns": len(self.request_patterns)
        }
```

---

## 🔧 **Security Configuration**

### **Environment Security**

```bash
# .env.security
# Security Configuration
SECURITY_LEVEL=production
ENABLE_RATE_LIMITING=true
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=3600

# API Key Configuration
API_KEY_ROTATION_INTERVAL=86400
API_KEY_BACKUP_COUNT=3
API_KEY_LENGTH=32

# JWT Configuration
JWT_SECRET_KEY=your-super-secret-jwt-key-here
JWT_ALGORITHM=HS256
JWT_EXPIRY_HOURS=24

# Encryption Configuration
ENCRYPTION_KEY=your-encryption-key-here
ENCRYPTION_ALGORITHM=Fernet

# Monitoring Configuration
SECURITY_LOGGING_ENABLED=true
SECURITY_ALERT_WEBHOOK=https://your-alerting-webhook.com/security
SECURITY_REPORT_INTERVAL=3600

# Input Validation
MAX_TEXT_LENGTH=10000
MIN_TEXT_LENGTH=1
ALLOWED_FILE_TYPES=json,csv,txt
MAX_FILE_SIZE=10485760

# Network Security
ALLOWED_ORIGINS=https://your-frontend-domain.com
CORS_ENABLED=true
HTTPS_REQUIRED=true
```

### **Security Headers**

```python
# src/security/security_headers.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=()"
        
        return response

def setup_security_middleware(app: FastAPI):
    """Setup security middleware."""
    
    # Add security headers
    app.add_middleware(SecurityHeadersMiddleware)
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["https://your-frontend-domain.com"],
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
```

---

## 📋 **Security Checklist**

### **Pre-Deployment Security Checklist**

- [ ] **Dependencies**: All dependencies updated and scanned for vulnerabilities
- [ ] **Environment Variables**: All secrets stored in environment variables
- [ ] **HTTPS**: SSL/TLS certificates properly configured
- [ ] **Input Validation**: All user inputs validated and sanitized
- [ ] **Authentication**: API key or JWT authentication implemented
- [ ] **Authorization**: Role-based access control implemented
- [ ] **Rate Limiting**: Request rate limiting configured
- [ ] **Logging**: Security event logging enabled
- [ ] **Monitoring**: Security monitoring and alerting configured
- [ ] **Backup**: Secure backup procedures in place

### **Runtime Security Checklist**

- [ ] **Regular Scans**: Automated security scans running
- [ ] **Log Analysis**: Security logs being analyzed
- [ ] **Access Control**: User access regularly reviewed
- [ ] **Key Rotation**: API keys rotated regularly
- [ ] **Incident Response**: Security incident response plan ready
- [ ] **Compliance**: Security compliance requirements met

---

## 🚨 **Incident Response**

### **Security Incident Response Plan**

```python
# src/security/incident_response.py
from datetime import datetime
from typing import Dict, List, Any
import logging

class SecurityIncident:
    """Represents a security incident."""
    
    def __init__(self, incident_type: str, severity: str, description: str):
        self.incident_type = incident_type
        self.severity = severity
        self.description = description
        self.timestamp = datetime.now()
        self.status = "open"
        self.actions_taken = []
    
    def add_action(self, action: str):
        """Add action taken for this incident."""
        self.actions_taken.append({
            "action": action,
            "timestamp": datetime.now().isoformat()
        })

class IncidentResponseManager:
    """Manages security incident response."""
    
    def __init__(self):
        self.active_incidents = []
        self.incident_history = []
    
    def report_incident(self, incident: SecurityIncident):
        """Report a new security incident."""
        self.active_incidents.append(incident)
        
        # Log incident
        logging.critical(f"SECURITY_INCIDENT: {incident.incident_type} - {incident.description}")
        
        # Take immediate actions based on severity
        if incident.severity == "critical":
            self._handle_critical_incident(incident)
        elif incident.severity == "high":
            self._handle_high_incident(incident)
    
    def _handle_critical_incident(self, incident: SecurityIncident):
        """Handle critical security incident."""
        # Immediate actions for critical incidents
        actions = [
            "Notify security team immediately",
            "Isolate affected systems",
            "Preserve evidence",
            "Activate incident response team"
        ]
        
        for action in actions:
            incident.add_action(action)
            self._execute_action(action)
    
    def _handle_high_incident(self, incident: SecurityIncident):
        """Handle high severity security incident."""
        actions = [
            "Notify security team",
            "Investigate root cause",
            "Implement temporary mitigations"
        ]
        
        for action in actions:
            incident.add_action(action)
            self._execute_action(action)
    
    def _execute_action(self, action: str):
        """Execute security action."""
        # Implementation depends on specific actions
        logging.info(f"Executing security action: {action}")
    
    def resolve_incident(self, incident_id: int, resolution: str):
        """Resolve a security incident."""
        if incident_id < len(self.active_incidents):
            incident = self.active_incidents[incident_id]
            incident.status = "resolved"
            incident.add_action(f"Incident resolved: {resolution}")
            
            # Move to history
            self.incident_history.append(incident)
            self.active_incidents.pop(incident_id)
    
    def get_incident_report(self) -> Dict[str, Any]:
        """Generate incident report."""
        return {
            "active_incidents": len(self.active_incidents),
            "resolved_incidents": len(self.incident_history),
            "critical_incidents": len([i for i in self.active_incidents if i.severity == "critical"]),
            "high_incidents": len([i for i in self.active_incidents if i.severity == "high"])
        }
```

---

## 📞 **Support & Resources**

- **Security Documentation**: [Complete Security Guide](Security-Guide)
- **Incident Response**: [Security Incident Procedures](incident-response.md)
- **Compliance**: [Security Compliance Guide](compliance-guide.md)
- **Security Team**: [Contact Security Team](mailto:security@your-org.com)

---

**Security is everyone's responsibility!** Follow the [Quick Security Checklist](#-quick-security-checklist-5-minutes) and maintain vigilance! 🔒🛡️ 