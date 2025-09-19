from collections import defaultdict, deque
from datetime import datetime, timedelta
from functools import wraps
from flask import abort, request
import threading
from typing import Callable, Optional

# Simple rate limiter using memory (use Redis for production)
_rate_limit_storage = defaultdict(deque)
_rate_limit_lock = threading.Lock()

def _get_client_identifier(request) -> str:
    """Extract client identifier from request, handling proxies."""
    # Check for X-Forwarded-For header (first trusted proxy)
    forwarded_for = request.headers.get('X-Forwarded-For')
    if forwarded_for:
        # Take the first IP (original client)
        return forwarded_for.split(',')[0].strip()
    
    # Fallback to remote_addr (requires ProxyFix middleware for accuracy behind proxies)
    return request.remote_addr

def rate_limit(max_requests=100, window_minutes=1, key_func: Optional[Callable] = None):
    """
    Rate limiting decorator.
    
    Args:
        max_requests: Maximum requests allowed in the time window
        window_minutes: Time window in minutes
        key_func: Optional function to extract client identifier from request.
                 If not provided, uses X-Forwarded-For header or request.remote_addr.
                 Note: If no key_func is provided, the application should use
                 Werkzeug's ProxyFix middleware to ensure request.remote_addr is accurate.
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Get client identifier
            if key_func:
                if not callable(key_func):
                    raise ValueError("key_func must be callable")
                client_id = key_func(request)
            else:
                client_id = _get_client_identifier(request)
            
            now = datetime.utcnow()
            window_start = now - timedelta(minutes=window_minutes)
            
            # Thread-safe operations
            with _rate_limit_lock:
                # Clean old requests (O(1) with deque vs O(n) with list comprehension)
                client_requests = _rate_limit_storage[client_id]
                while client_requests and client_requests[0] < window_start:
                    client_requests.popleft()

                # Check rate limit
                if len(client_requests) >= max_requests:
                    abort(429, description="Rate limit exceeded")

                # Add current request
                client_requests.append(now)
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator
